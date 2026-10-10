# main.py
import os
import sys
import threading
import time

import pygame
from pygame.locals import *

from resource_manager import ResourceManager
from game_state import GameState
from renderer import Renderer
from input_handler import InputHandler, DIFFICULTY_LABELS
from config import Config
from AI import AdvancedAIController, available_difficulties

try:  # 对局方式（每方由谁出招）的定义
    from jieqi import engines as _engines
    from jieqi.gui_adapter import gui_color_to_side
except ImportError:  # pragma: no cover - 打包路径不同时的兜底
    from Game.jieqi import engines as _engines
    from Game.jieqi.gui_adapter import gui_color_to_side

#: 搜索是纯 Python 的 CPU 密集任务，靠 GIL 与渲染线程交替。
#: 默认 5ms 的切换间隔会让界面在 AI 思考期间明显掉帧；调小到 1ms 让渲染更跟手，
#: 代价只是搜索速度损失几个百分点。
sys.setswitchinterval(0.001)


def _short_llm_status(status: str) -> str:
    """把控制器那段很长的"为什么没启用"缩成启动页那一行放得下的说法。

    ``llm_status`` 是给控制台看的（含配置路径、模型名、blend_weight 这些），
    直接画到启动页会被裁得只剩前半句，反而看不出重点。
    """
    text = str(status or '')
    if not text:
        return "大模型模块不可用（jieqi.llm 导入失败）"
    if 'API Key' in text or 'api_key' in text:
        return "缺少 API Key：请填 Game/ai_config.json 的 llm.api_key"
    if 'enabled' in text:
        return "配置里 llm.enabled = false，点开即可本次运行启用"
    if '导入失败' in text or '不可用' in text:
        return "大模型模块不可用（jieqi.llm 导入失败）"
    if '读取' in text and '失败' in text:
        return "读不到 ai_config.json 的 llm 段"
    return text if len(text) <= 42 else text[:41] + "…"


class ChessGame:
    def __init__(self):
        # 初始化Pygame所有子系统
        self.config = Config()
        pygame.init()
        # 先建立窗口（尺寸已含顶部配置区与右侧信息面板）
        pygame.display.set_mode((self.config.WINDOW_W, self.config.WINDOW_H))

        # 初始化资源管理器并加载资源
        self.rm = ResourceManager()
        self.rm.load_resources()

        # 创建游戏状态实例
        self.gs = GameState(self.rm)
        # 应用配置里的"每方对局方式"（``ai.side_engines``；老配置写
        # ``ai.move_source`` 也能折算过来）。放在这里而不是 GameState 里：
        # GameState 不该知道配置文件在哪，而启动页/AI 控制器都要读它。
        try:
            from jieqi.settings import ai_settings as _ai_settings

            self.gs.apply_engine_settings(_ai_settings())
        except Exception as exc:      # 配置坏了绝不能挡住开局
            print("[揭棋AI] 读取对局方式配置失败（按默认人机对战）：%r" % (exc,))

        # 创建正式的游戏窗口与渲染器
        self.renderer = Renderer(self.rm, self.gs)

        # 初始化输入处理器
        self.input_handler = InputHandler(self.gs, self.renderer)
        self.input_handler.game = self  # 让按钮/按键能派发界面命令
        self.clock = pygame.time.Clock()
        self.running = True

        # AI 控制器与后台计算线程
        self.ai_controller = None
        self.ai_difficulty = None      # None = 用 ai_config.json 里的设置
        self.ai_thread = None
        #: AI 线程只写这几个字段，主线程负责把着法应用到棋局
        #: （旧版在子线程里直接 move_piece，会与渲染线程抢动画列表）
        self.ai_pending_move = None
        self.ai_error = None
        #: **真实挂钟**用时（毫秒）：从"开始思考"到"落子落定"。
        #: 为什么不直接用搜索线程自报的耗时：那是 CPU 时间，与玩家的感受
        #: （含 GIL 争抢、渲染线程抢时间片）差得很远——真实反馈就是
        #: "明明过了很久，却显示才 2000ms"。
        self.ai_think_ms = 0
        #: 开始思考的时刻（``time.monotonic()``；0 = 当前没在思考）。
        #: 名字刻意不叫 ``_ai_think_start``——那是下面那个**方法**的名字，
        #: 同名会把方法覆盖成 float，一调用就 TypeError。
        self._ai_think_began = 0.0
        self._ai_think_ms_at_result = 0
        self._ai_side_started = None
        #: 大模型点评的运行时开关（默认跟随配置：能装配上就是开）
        self.llm_enabled = None
        #: 启动页的"设置面板"用得到控制器（问它大模型装配好了没），先建一个
        self._ensure_ai_controller()

    # ================================================================ 主循环
    def run(self):
        while self.running:
            if self.gs.show_side_selection:
                self._handle_side_selection()
            else:
                self._handle_main_game()
            self.clock.tick(30)
            pygame.display.flip()
        pygame.quit()

    def _handle_main_game(self):
        self.running = self.input_handler.handle_events()
        self._consume_ai_move()
        self._sync_ai_state()
        # draw_all 内部已经包含「游戏结束」覆盖层与三个分区
        self.renderer.draw_all()
        if self.gs.game_over:
            self.ai_controller = None   # 保留 gs.ai_comment 供结算界面显示
        elif self.can_ai_move() and not self.renderer.overlay:
            self._process_ai_move()
        return True

    def _handle_side_selection(self):
        # 启动页上的开关（难度/大模型/自动保存）显示的是**控制器与配置的真实状态**，
        # 所以每帧同步一次；控制器只在第一次进来时创建，之后是纯读操作。
        if self.ai_controller is None:
            self._ensure_ai_controller()
        self.running = self.input_handler.handle_events()
        self.renderer.draw_side_selection()

    # ========================================================== 界面命令
    def handle_command(self, cmd: str) -> None:
        """处理来自按钮/按键的命令。"""
        if not cmd:
            return
        if cmd.startswith('difficulty:'):
            self.set_difficulty(cmd.split(':', 1)[1])
        elif cmd == 'llm:toggle':
            self.toggle_llm()
        elif cmd == 'llm:apply':
            self.toggle_apply_llm_move()
        elif cmd == 'setting:autosave':
            self.toggle_auto_save()
        elif cmd.startswith('engine:'):
            # engine:red:llm（直接选某一档）/ engine:cycle:red（热键循环）
            parts = cmd.split(':')
            if len(parts) == 3 and parts[1] == 'cycle':
                self.cycle_side_engine(parts[2])
            elif len(parts) == 3:
                self.set_side_engine(parts[1], parts[2])
        elif cmd == 'game:start':
            self.choose_side(self.gs.player_side)
        elif cmd == 'side:flip':
            # 中局换边（键盘 C）：改"我执哪一方"，并把对手的档位跟着搬过去。
            # 启动页上不再放阵营选项（用户："去掉我执红或者我执黑选项，
            # 太啰嗦了，有需要我会自己调的"），所以入口只留这一个。
            self.flip_player_side()
        elif cmd.startswith('side:select:'):
            self.select_player_side(cmd.rsplit(':', 1)[1])
        elif cmd.startswith('side:'):
            # 老写法的命令（`side:red`）：只改阵营，不开局
            self.select_player_side(cmd.split(':', 1)[1])
        elif cmd == 'game:continue':
            self.continue_game()
        elif cmd == 'nav:first':
            self.goto_ply(0)
        elif cmd == 'nav:prev':
            self.goto_round(-1)
        elif cmd == 'nav:next':
            self.goto_round(+1)
        elif cmd == 'nav:last':
            self.goto_ply(self.gs.total_plies)
        elif cmd == 'game:restart':
            self.restart()
        elif cmd == 'game:menu':
            self.back_to_start()
        elif cmd == 'record:export':
            self.export_record()
        elif cmd == 'record:import':
            self.import_record()
        elif cmd == 'view:board':
            self.toggle_board_view(True)
        elif cmd == 'view:close':
            self.toggle_board_view(False)
        elif cmd.startswith('score:'):
            self.set_score(cmd[len('score:'):])

    def set_score(self, code: str) -> None:
        """给"从当前局面走出的那一手"打人工分（``code`` 为空 = 清除）。"""
        gs = self.gs
        if int(getattr(gs, 'ply', 0)) <= 0:
            gs.set_toast("开局没有可评的着法，先走一手", loud=True)
            return
        if not gs.set_current_score(code):
            return
        label = gs.label_for_ply(gs.ply) or "已清除"
        gs.set_toast("已打分：%s（第 %d 手）" % (label, gs.ply + 1), loud=True)
        print("[揭棋AI] 第 %d 手人工打分 -> %s"
              % (gs.ply + 1, gs.label_for_ply(gs.ply) or "清除"))

    def set_difficulty(self, name: str) -> None:
        """切换 AI 难度（启动页上点 chip，或按 1~4）。"""
        name = str(name).lower()
        if name not in available_difficulties():
            return
        self.ai_difficulty = name
        self.ai_controller = None      # 下一回合按新难度重建
        self.ai_pending_move = None
        self.renderer.difficulty = name
        # 让启动页的 chip 立刻高亮（不用等下一局）
        self.renderer._start_layout_key = None
        self.gs.toast = f'AI 难度已切换为：{DIFFICULTY_LABELS.get(name, name)}'
        print(f"[揭棋AI] 难度切换为 {name}")

    def _ensure_ai_controller(self) -> None:
        """按需创建 AI 控制器（启动页要问它"大模型装配好了吗"）。

        构造一个控制器很便宜（不搜索、不联网），但**读配置**是有意义的：
        启动页上"大模型点评"就靠它显示"未配置 API Key"这类原因，
        而不是让人点半天没反应。
        """
        if self.ai_controller is None:
            self.ai_controller = AdvancedAIController(
                self.gs, difficulty=self.ai_difficulty)
        self._sync_start_state()

    def _sync_start_state(self) -> None:
        """把控制器的状态同步到渲染器（启动页的开关显示的就是这份状态）。"""
        controller = self.ai_controller
        if controller is None:
            return
        self.llm_enabled = bool(getattr(controller, 'llm_enabled', False))
        r = self.renderer
        r.llm_on = bool(self.llm_enabled and controller.advisor is not None)
        r.apply_llm_move = bool(getattr(controller, 'apply_llm_move', False))
        r.llm_label = getattr(controller, 'llm_model_label', '') or ''
        r.difficulty = getattr(controller, 'difficulty', r.difficulty)
        status = str(getattr(controller, 'llm_status', '') or '')
        if r.llm_on:
            r.llm_note = "已装配 %s，点评只针对 AI 自己的招法" % (
                getattr(controller, 'llm_model_label', '') or "大模型")
        else:
            r.llm_note = _short_llm_status(status)
        r.auto_save = bool(self.gs.auto_save_record())
        r._start_layout_key = None

    def toggle_llm(self) -> None:
        """开关大模型点评（启动页上拨；本次运行有效，不改配置文件）。"""
        self._ensure_ai_controller()
        enabled = self.ai_controller.toggle_llm()
        self._sync_start_state()
        if enabled:
            self.gs.toast = '大模型点评：已开启'
        else:
            status = getattr(self.ai_controller, 'llm_status', '') or ''
            if '未启用' in status or 'API Key' in status or '不可用' in status:
                self.gs.toast = f'大模型点评不可用：{status}'
            else:
                self.gs.toast = '大模型点评：已关闭'

    def toggle_apply_llm_move(self) -> None:
        """开关「应用大模型招法」：开启后 AI 直接下大模型推荐的那一手。"""
        self._ensure_ai_controller()
        enabled = self.ai_controller.toggle_apply_llm_move()
        self._sync_start_state()
        if enabled:
            self.gs.toast = '已开启：AI 将采用大模型推荐的招法（每步会同步等待模型）'
        else:
            status = getattr(self.ai_controller, 'llm_status', '') or ''
            if '未启用' in status or 'API Key' in status or '不可用' in status:
                self.gs.toast = f'无法应用大模型招法：{status}'
            else:
                self.gs.toast = '已关闭：AI 改用引擎自己的招法'

    def toggle_auto_save(self) -> None:
        """开关「对局结束自动保存棋谱」（本次运行有效，不改配置文件）。

        为什么要给开关：自动保存默认开启，但有时就是不想要这一局的棋谱
        （试招、随手摆一局），去改 ai_config.json 再重启太重。
        """
        gs = self.gs
        current = bool(gs.auto_save_record())
        gs.auto_save_override = not current
        self.renderer.auto_save = bool(gs.auto_save_record())
        self.renderer._start_layout_key = None
        gs.set_toast('对局结束自动保存：%s' % ('已开启' if gs.auto_save_record()
                                              else '已关闭'),
                     loud=True)

    # ------------------------------------------------------ 启动页上的动作
    def select_player_side(self, side: str) -> None:
        """把"我执哪一方"设成 ``side``（不重开，也不清棋局）。

        启动页上**不再有**阵营选项（用户要求去掉），这个入口留给：
        ``ai_config.json`` 的 ``ai.player_side``、中局热键 `C`、
        以及导入棋谱时按棋谱 meta 还原。
        """
        side = str(side).lower()
        if side not in ('red', 'black'):
            return
        if not self.gs.set_player_side(side):
            return
        self._sync_start_state()
        self.gs.set_toast('我执%s（对手由%s出招）'
                          % ('红' if side == 'red' else '黑',
                             _engines.LABELS.get(
                                 self.gs.side_engine('black' if side == 'red'
                                                     else 'red'), '')))

    def flip_player_side(self) -> None:
        """中局换边：红<->黑（键盘 `C`）。"""
        current = getattr(self.gs, 'player_side', 'red')
        self.select_player_side('black' if current == 'red' else 'red')

    def choose_side(self, side: str) -> None:
        """在启动页上选定阵营并**开局**（「开始游戏」按钮 / ``side:`` 老命令）。"""
        side = str(side).lower()
        if side not in ('red', 'black'):
            return
        self._ensure_ai_controller()
        gs = self.gs
        if gs.move_history and not gs.game_over:
            # 盘面已经下了一半却又点了开始：等于重开一局，必须先说清楚，
            # 不能默默把棋吃掉（这局棋谱也可能已经被自动保存过）。
            print("[揭棋AI] 启动页上重新开始：开始新的一局")
        gs.reset()                      # 保留玩家挑好的对局方式
        gs.show_side_selection = False
        gs.record_saved = ''
        gs.imported_from = ''
        self.ai_pending_move = None
        self.ai_error = None
        self._ai_think_stop()
        # 阵营与"对手由谁走"一起定：``set_player_side`` 会把玩家那方设成人工、
        # 把对手那方留给原来挑的档位。**必须在 ``reset()`` 之后调**——
        # reset 会把 player_side 与 side_engines 都恢复成默认。
        gs.set_player_side(side)
        # 玩家从没挑过对局方式（两边都还是默认）时，按 config 决定是"人机"
        # 还是"双人"（``config.ai.ai_mode=false`` = 双人对下）。
        if not self.config.AI.get('ai_mode', True):
            gs._init_ai()
            gs.player_side = side
        if self.ai_controller is not None:
            self.ai_controller.discard_inflight_llm()
        self.renderer.log_scroll = 0
        self.renderer._log_state_key = None
        self._sync_start_state()
        print(f"[揭棋AI] 开局：玩家执{'红' if side == 'red' else '黑'}，"
              f"难度 {self.renderer.difficulty}，"
              f"大模型点评 {'开' if self.renderer.llm_on else '关'}")

    def continue_game(self) -> None:
        """回到**当前这局**接着下（从启动页退出，不重置棋局）。"""
        gs = self.gs
        if not gs.move_history:
            gs.set_toast('还没有可以继续的对局', loud=True)
            return
        gs.show_side_selection = False
        gs.goto_ply(gs.total_plies)      # 从"最新一手"接着下，而不是停在复盘位置
        self.renderer.log_scroll = 0
        self.renderer._log_state_key = None
        self._sync_start_state()
        gs.set_toast('继续对局：第 %d 手' % gs.ply, loud=True)

    def back_to_start(self) -> None:
        """回到启动页（改设置 / 换阵营 / 继续或载入棋谱）。

        对局**原样保留**在 ``gs`` 里，点「继续上一局」就能回来；
        所以"回启动页"不是"放弃这一局"。
        """
        if self.ai_controller is not None:
            self.ai_controller.discard_inflight_llm()
        self.ai_pending_move = None
        self.ai_error = None
        self._ai_think_stop()
        self.renderer.overlay = ''
        self.gs.show_side_selection = True
        self._sync_start_state()
        self.gs.set_toast('')

    def _ai_think_start(self) -> None:
        """开始计时（真实挂钟）：从这一刻到落子落定，就是玩家等的时长。"""
        self._ai_think_began = time.monotonic()
        self.gs.ai_thinking = True
        self.gs.ai_think_started = self._ai_think_began

    def _ai_think_stop(self) -> None:
        self.gs.ai_thinking = False
        self.gs.ai_think_started = 0.0

    # ------------------------------------------------------ 步数自由切换
    def goto_ply(self, target: int) -> None:
        """跳到指定手数（浏览历史）。

        浏览期间**暂停 AI 走子、也不请求大模型**；走子或点「最新」即回到最新局面。
        """
        if self.ai_controller is not None:
            self.ai_controller.discard_inflight_llm()
        self.ai_pending_move = None
        self.ai_error = None
        if not self.gs.goto_ply(target):
            return
        self.gs.toast = (f'回到最新第 {self.gs.ply} 手'
                         if not self.gs.is_browsing
                         else f'浏览第 {self.gs.ply}/{self.gs.total_plies} 手（未请求大模型）')

    def goto_round(self, delta: int) -> None:
        """按**回合**翻手（-1 = 上一回合，+1 = 下一回合）。

        用户反馈："点击上一步后实际应该红黑各退一步，这样算一回合"——
        因为一回合就是红黑各一手，按手翻会停在"红走完、该黑走"的半路上，
        看起来像"轮到红方"（那个提示问题已在 ``goto_ply`` 里修掉，
        但按回合翻更符合下棋时的直觉）。

        翻到"整回合"的边界：步数 0（开局）和最新一手都是合法落点，
        所以夹在 ``[0, total_plies]`` 里即可；红先，所以偶数手 = 整回合。
        """
        step = 2 if delta > 0 else -2
        target = int(self.gs.ply) + step
        # 落在"整回合"上（偶数手）；最新一手如果停在奇数手，也允许落在那儿
        if target % 2 != 0 and target != int(self.gs.total_plies):
            target += 1 if step < 0 else -1
        target = max(0, min(target, int(self.gs.total_plies)))
        if target == self.gs.ply:
            self.gs.toast = ('已经是开局了' if target == 0
                             else '已经是最新一手')
            return
        self.goto_ply(target)
        if not self.gs.is_browsing:
            self.gs.toast = '已回到最新一手'
        else:
            self.gs.toast = ('第 %d/%d 手（按回合翻，一回合 = 红黑各一手）'
                             % (self.gs.ply, self.gs.total_plies))

    def undo(self) -> None:
        """回退一个回合。

        **回退不访问模型**：丢弃在途的点评请求结果、清掉旧点评，
        并且不会因为回退本身发出任何请求。
        """
        if self.gs.ply == 0:
            self.gs.toast = '已经回到开局'
            return
        if self.ai_controller is not None:
            self.ai_controller.discard_inflight_llm()
        self.ai_pending_move = None
        self.ai_error = None
        self.gs.undo_round()
        self.gs.toast = (f'已回退到第 {self.gs.ply} 手'
                         f'（该手点评已恢复，未请求大模型）')

    def restart(self) -> None:
        """重新开始一局（保留当前阵营、难度与每方的对局方式）。

        ⚠ 刻意**不**调 ``enable_ai``：那会把玩家挑好的对局方式覆盖成
        "某一方人工 + 另一方本机引擎"。重新开始只是清空棋盘，
        不是重置设置（``gs.reset()`` 默认就保留 ``side_engines``）。
        """
        if self.ai_controller is not None:
            self.ai_controller.discard_inflight_llm()
        self.ai_pending_move = None
        self.ai_error = None
        engines = dict(self.gs.side_engines or {})
        player_side = self.gs.player_side
        self.gs.reset(keep_engines=False)
        self.gs.side_engines.update(engines)      # reset 会清空，这里补回来
        self.gs.show_side_selection = False
        self.gs.player_side = player_side
        self.gs.current_player = 'red'
        self.gs.record_saved = ''
        self.gs.imported_from = ''
        self.gs.toast = '已重新开始'
        self._sync_start_state()

    # ---------------------------------------------------------- 棋谱
    def export_record(self) -> None:
        """把当前对局导出成棋谱（``Game/records/xxx.json`` + ``.txt``）。

        一定会给出**看得见的反馈**：成功时在棋盘上弹一条横幅写明存到哪
        （几秒后消失），失败时弹红色横幅说明原因，控制台也各打印一行。
        """
        if not self.gs.move_history:
            self.gs.set_toast('还没有走过棋，无法导出棋谱', loud=True)
            print("[揭棋AI] 棋谱导出跳过：还没有走过棋")
            return
        # 把引擎最后一步的读数一并写进棋谱，方便以后复盘定位问题手
        meta = {'difficulty': getattr(self.renderer, 'difficulty', '')}
        result = getattr(self.ai_controller, 'last_result', None) if self.ai_controller else None
        if result is not None:
            meta['engine'] = {
                'score': round(float(getattr(result, 'score', 0.0)), 1),
                'verdict': str(getattr(result, 'verdict', '')),
                'depth': int(getattr(result, 'depth', 0)),
                'nodes': int(getattr(result, 'nodes', 0)),
                'samples': int(getattr(result, 'samples', 0)),
                'best_move': self._move_text(getattr(result, 'move', None)),
                'llm_model': getattr(self.ai_controller, 'llm_model_label', ''),
                'llm_enabled': bool(getattr(self.ai_controller, 'llm_enabled', False)),
            }
        info = self.gs.export_record(directory=None)
        if info.get('ok'):
            # 用**实际写出的**文件名（同名防撞会自动改成 `-2`）。
            # ``gs.record_saved`` 已由 export_record 同步，这里只负责提示。
            path = str(info.get('path') or '')
            self.gs.set_toast("棋谱已导出到 %s"
                              % os.path.join("records", os.path.basename(path)),
                              loud=True)
            print(f"[揭棋AI] 棋谱已导出 -> {path}")
        else:
            self.gs.set_toast(f"棋谱导出失败：{info.get('error')}", loud=True)
            print(f"[揭棋AI] 棋谱导出失败：{info.get('error')}")

    @staticmethod
    def _move_text(move) -> str:
        """引擎着法整数 -> ``b3c3``（与棋谱 / 走棋记录同一套写法）。"""
        if move is None:
            return ''
        from jieqi.notation import move_name

        return move_name(int(move))

    # ---------------------------------------------------------- 导入棋谱
    def import_record(self) -> None:
        """导入一份棋谱并重放成当前对局（弹系统文件框选 ``.json``）。"""
        path = self._pick_record_file()
        if not path:
            self.gs.set_toast('已取消导入棋谱', loud=True)
            return
        if self.ai_controller is not None:
            self.ai_controller.discard_inflight_llm()
        self.ai_pending_move = None
        self.ai_error = None
        info = self.gs.import_record_file(path)
        if not info.get('ok'):
            self.gs.set_toast(f"导入失败：{info.get('error')}", loud=True)
            print(f"[揭棋AI] 导入棋谱失败：{info.get('error')}")
            return
        # 导入的是别人的对局，AI 不该接着往下走，先切到"双人对下"
        self.gs._init_ai()
        self.renderer.overlay = ''
        # 走棋记录滚回顶部（导入后停在开局，记录从头看起）
        self.renderer.log_scroll = 0
        self.renderer._log_state_key = None
        # 导入的名字由 gs.imported_from 记着，**不写 record_saved**——
        # 否则"接着下完"会自动保存时把导入的这份棋谱覆盖掉。
        name = info.get('name') or os.path.basename(path)
        result = str(info.get('result') or '*')
        outcome = {'1-0': '红胜', '0-1': '黑胜',
                   '1/2-1/2': '和棋'}.get(result, '未终局')
        self.gs.set_toast((f"已导入棋谱：{name}（{info.get('plies')} 手 {outcome}，"
                          f"已回到开局，用「下一步」复盘）"), loud=True)
        print(f"[揭棋AI] 已导入棋谱 {name}：{info.get('plies')} 手，结果 {result}")
        for warning in info.get('warnings') or []:
            print(f"[揭棋AI] 导入提示：{warning}")

    @staticmethod
    def _pick_record_file() -> str:
        """弹系统文件选择框，返回选中的棋谱路径（取消/不可用返回空串）。

        走 ``tkinter``（标准库）：它是**可选**的，某些精简 Python 没带，
        所以先探测再导入，缺了就给一句明确提示，而不是崩在按钮上。
        """
        import importlib.util

        if importlib.util.find_spec('tkinter') is None:
            print("[揭棋AI] 这台机器的 Python 没带 tkinter，无法弹文件框；"
                  "可以改用命令行导入：python record_tool.py show <棋谱名>")
            return ''
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.update()
            from jieqi import record as _rec

            path = filedialog.askopenfilename(
                title='选择要导入的棋谱（.json）',
                initialdir=_rec.record_dir(),
                filetypes=[('揭棋棋谱', '*.json'), ('所有文件', '*.*')])
            root.destroy()
            return str(path or '')
        except Exception as exc:      # 没有显示环境 / 被策略拦下
            print(f"[揭棋AI] 打不开文件选择框：{exc}")
            return ''

    def toggle_board_view(self, show=None) -> None:
        """打开/关闭「回顾棋局」浮层。"""
        if show is None:
            show = not bool(self.renderer.overlay)
        self.renderer.overlay = 'board' if show else ''
        if show:
            self.gs.toast = '回顾棋局：←→ 或按钮翻手，ESC 关闭'
        else:
            self.gs.toast = '已关闭回顾棋局'

    # ============================================================== AI
    def _last_mover(self):
        """当前局面里"刚走完的那一方"（``'red'``/``'black'``/``None``）。

        为什么要这个：``goto_ply`` 会把 ``current_player`` 恢复成**快照里存的
        那个值**——也就是"当时轮到谁"。所以浏览历史再回到最新时，
        ``current_player`` 会等于"刚走完的那一方"，而不是"该走的那一方"。

        真实 bug（用户报的"黑方连走两步"）：AI 走完 → 玩家点「上一步」→
        点「下一步」回到最新 → ``current_player`` 又变回 ``ai_color``、
        且 ``is_browsing`` 变回 False，于是 AI **又走一步**。

        修复依据：一条线上"轮到的一方"绝不等于"刚走完的一方"。
        """
        ply = int(getattr(self.gs, 'ply', 0) or 0)
        history = getattr(self.gs, 'move_history', None) or []
        if ply <= 0 or ply > len(history):
            return None
        rec = history[ply - 1]
        # ``side`` 是显式记的（0=红 1=黑）；老数据可能没有，退回按棋子颜色推
        side = rec.get('side')
        if side in (0, 1):
            return 'red' if int(side) == 0 else 'black'
        if side in ('red', 'black'):
            return str(side)
        piece = rec.get('piece') or {}
        color = str(piece.get('color') or '')
        return color if color in ('red', 'black') else None

    def set_side_engine(self, side: str, source: str) -> None:
        """切换某一方的**对局方式**（人工输入 / 本机引擎 / 大模型）。

        "切换到哪种就完全应用那个引擎的招法"——所以这里只记档位，真正的
        "完全采用"发生在 :meth:`_move_for_current_side` 里：它按当前档位挑
        一条**明确的分支**，不做任何混合。

        切档位时会：
        * 把在途的点评结果作废（那是上一套档位产生的）；
        * 如果被换掉的那方正要走这一步，取消这次思考（否则会把旧引擎的结果
          落到新档位的局面上）；
        * 弹一条提示说清楚"现在轮到谁、用什么"。
        """
        side = str(side or '').lower()
        if side not in ('red', 'black'):
            return
        gs = self.gs
        before = gs.side_engine(side)
        if not gs.set_side_engine(side, source):
            return
        now = gs.side_engine(side)
        self._sync_start_state()
        if self.ai_controller is not None:
            self.ai_controller.discard_inflight_llm()
        # 换了走子方的档位 → 正在跑的那次思考已经没用了
        if gs.current_player == side and int(getattr(gs, 'ply', 0) or 0) == \
                getattr(self, '_ai_ply_started', -1) and not gs.is_browsing:
            self.ai_pending_move = None
            self._ai_think_stop()
        label = _engines.LABELS.get(now, now)
        gs.set_toast('%s方：%s' % ('红' if side == 'red' else '黑', label),
                     loud=True)
        print('[揭棋AI] 对局方式：%s方 = %s' % (
            '红' if side == 'red' else '黑', label))

    def cycle_side_engine(self, side: str, step: int = 1) -> None:
        """中局热键切换某一方的档位（``E`` 切红方、``T`` 切黑方）。"""
        side = str(side or '').lower()
        if side not in ('red', 'black'):
            return
        self.set_side_engine(side, _engines.cycle(self.gs.side_engine(side), step))

    def _move_for_current_side(self):
        """按**当前走子方的档位**取一招。

        这是"对局方式"的唯一分岔点，每条分支都只用一个引擎：

        * ``python``   → :meth:`AdvancedAIController.analyse_python`
        * ``llm``      → :meth:`AdvancedAIController.analyse_llm`
        * ``human``    → 不在这里（人工输入走输入处理器）

        返回 ``(result, source, error)``；``result.move`` 为 ``None`` 表示
        这一步没拿到着法（例如大模型失败），调用方据此放行给人工。
        """
        source = self.gs.side_engine(self.gs.current_player)
        side = gui_color_to_side(self.gs.current_player)
        if source == _engines.LLM:
            result = self.ai_controller.analyse_llm(side)
            if result is None or result.move is None:
                return None, source, '大模型没有给出可用着法'
            # analyse_llm 内部可能退回本机引擎（它会给 notes 与 last_source）
            return result, getattr(self.ai_controller, 'last_source', source), ''
        result = self.ai_controller.analyse_python(side)
        if result is None or result.move is None:
            return None, source, '本机引擎没有给出可用着法'
        return result, source, ''

    def should_auto_move(self) -> bool:
        """现在**该不该**由引擎/大模型自动走子（按当前走子方的档位判定）。

        比 :meth:`_is_engine_turn` 严一点：不仅要求轮到的那一方不是人工、
        不在浏览历史，还要求"轮到的一方不是刚走完的那一方"——否则把
        "回到最新"误当成"轮到引擎了"，引擎就会连走两步。
        """
        if not self._is_engine_turn():
            return False
        return self._last_mover() != self.gs.current_player

    def can_ai_move(self) -> bool:
        """旧名字，等价于 :meth:`should_auto_move`（老测试与脚本仍在用）。"""
        return self.should_auto_move()

    def _is_ai_turn(self) -> bool:
        """旧名字，等价于 :meth:`_is_engine_turn`（老测试仍在用）。"""
        return self._is_engine_turn()

    def _is_engine_turn(self) -> bool:
        side = self.gs.current_player
        return (
                not self.gs.is_human_side(side)
                and not self.gs.game_over
                and not self.gs.is_browsing      # 浏览历史时暂停自动走子
        )

    def _process_ai_move(self):
        """启动后台思考线程（只计算，不改棋局）。"""
        if not self.ai_controller:
            self.ai_controller = AdvancedAIController(self.gs, difficulty=self.ai_difficulty)
        if self.ai_thread and self.ai_thread.is_alive():
            return
        self.ai_pending_move = None
        self.ai_error = None
        self._ai_result_error = ''
        source = self.gs.side_engine(self.gs.current_player)
        self._ai_result_source = source
        self.gs.toast = '%s思考中…' % _engines.LABELS.get(source, '引擎')
        self._ai_side_started = self.gs.current_player
        #: 记下开始思考时的手数：玩家在思考期间回退/切换步数的话，这个结果就作废，
        #: 否则会把针对旧局面的着法落到回退后的局面上。
        self._ai_ply_started = self.gs.ply
        #: **真实挂钟计时从这里开始**。放在搜索线程内部会漏掉"线程还没被调度"
        #: 和"结果还没被主线程取走"这两段，那正是玩家觉得"等了好久"的部分。
        self._ai_think_start()
        self.ai_thread = threading.Thread(target=self._ai_task, daemon=True)
        self.ai_thread.start()

    def _ai_task(self):
        """后台线程：只读棋局 + 计算，绝不直接改棋盘。

        按**当前走子方的档位**挑一条分支（见 :meth:`_move_for_current_side`）：
        本机引擎 / 大模型各走各的，互不混合。结果与来源都放在
        ``self`` 上，由主线程的 ``_consume_ai_move`` 落盘。
        """
        try:
            result, source, error = self._move_for_current_side()
            self._ai_result_source = source
            self._ai_result_error = error
            move = None
            if result is not None and result.move is not None:
                frm, to = result.move >> 7, result.move & 127
                move = ((frm // 9, frm % 9), (to // 9, to % 9))
            self.ai_error = error or None
        except Exception as exc:  # 引擎异常不应该让游戏崩溃
            move = None
            self._ai_result_source = self.gs.side_engine(self.gs.current_player)
            self._ai_result_error = '%r' % (exc,)
            self.ai_error = exc
        #: 搜索线程自己量到的耗时（CPU 侧）。**只作参考**，顶栏显示的是挂钟时间。
        self._ai_think_ms_at_result = int(
            (time.monotonic() - self._ai_think_began) * 1000)
        self.ai_pending_move = move if move is not None else None

    def _consume_ai_move(self):
        """主线程应用 AI 的着法（保证只在主线程改动棋局与动画）。"""
        if self.ai_pending_move is None:
            if self.ai_error is not None or getattr(self, '_ai_result_error', ''):
                # 拿不到招法（大模型失败等）：**放行给人工**，
                # 而不是替这一方乱走一步。"完全采用某引擎的招法"里没有
                # "拿不到就换一个引擎"这一说；玩家看到提示后可以自己走。
                #
                # 同时**把这一方切成"人工输入"**：否则下一帧
                # `should_auto_move()` 又为真，会每帧重试一次、页面变成
                # 一秒钟刷几十条错误提示（真正的"卡住"）。
                reason = self._ai_result_error or self.ai_error
                source = getattr(self, '_ai_result_source', '')
                label = _engines.LABELS.get(source, source)
                side = self._ai_side_started
                self.gs.set_side_engine(side, _engines.HUMAN)
                self._sync_start_state()
                self.gs.toast = ('%s没有给出可用着法，%s方已切回人工输入：%s'
                                 % (label, '红' if side == 'red' else '黑', reason))
                print('[揭棋AI] %s没有给出可用着法（%s），%s方已切回人工输入'
                      % (label, reason, '红' if side == 'red' else '黑'))
                self.ai_error = None
                self._ai_result_error = ''
                self._finish_ai_think()
            return
        if self.gs.current_player != self._ai_side_started:
            # 局面已经变了（例如玩家回退），丢弃这次计算结果
            self.ai_pending_move = None
            self._ai_think_stop()
            return
        if self.gs.ply != getattr(self, '_ai_ply_started', -1):
            # 玩家在 AI 思考期间切换了步数：这份结果已作废
            self.ai_pending_move = None
            self._ai_think_stop()
            self.gs.toast = '已切换步数，丢弃 AI 的旧计算结果'
            return
        move = self.ai_pending_move
        self.ai_pending_move = None
        (sr, sc), (er, ec) = move
        self._finish_ai_think()
        self.gs.move_piece(sr, sc, er, ec)
        self.gs.toast = self._ai_toast()
        self._switch_side()

    def _finish_ai_think(self) -> None:
        """收尾计时：把**真实挂钟**用时写进 ``ai_think_ms``（顶栏与面板显示的都是它）。

        为什么不在搜索线程里完事：``get_best_move`` 返回之后，结果还要等主线程
        在下一帧 ``_consume_ai_move`` 里取走；这中间的等待（含渲染线程抢 GIL）
        也是玩家实实在在等掉的时间。
        """
        if self._ai_think_began > 0:
            self.ai_think_ms = int((time.monotonic() - self._ai_think_began) * 1000)
        self._ai_think_began = 0.0
        self._ai_think_stop()
        #: 顺手把这次用时同步给渲染用的状态（面板文案立刻变，不用等下一帧）
        self.gs.ai_think_ms = self.ai_think_ms

    def _ai_toast(self) -> str:
        result = getattr(self.ai_controller, 'last_result', None)
        if result is None:
            return f'AI 用时 {self.ai_think_ms}ms'
        mate = result.mate_in()
        bits = [f'AI 用时 {self.ai_think_ms}ms', f'深度 {result.depth}',
                f'评估 {result.score:+.0f}（{result.verdict}）']
        if mate:
            bits.append(f'{mate} 步杀')
        return ' | '.join(bits)

    def _switch_side(self):
        self.gs.current_player = 'red' if self.gs.current_player == 'black' else 'black'
        self.gs.check_game_state()

    # ==================================================== 状态同步（主线程）
    def _sync_ai_state(self):
        """把后台线程的点评/评估搬到棋局状态与渲染器，供渲染线程安全读取。

        点评会**写进当前这一手的走棋记录**，所以回退到那一手时能原样取回；
        浏览历史时不覆盖（显示的是历史那一手的点评）。
        """
        controller = self.ai_controller
        if controller is None:
            return
        self.llm_enabled = bool(getattr(controller, 'llm_enabled', False))
        self.renderer.llm_on = bool(self.llm_enabled and controller.advisor is not None)
        self.renderer.apply_llm_move = bool(getattr(controller, 'apply_llm_move', False))
        self.renderer.llm_label = getattr(controller, 'llm_model_label', '') or ''
        self.renderer.difficulty = getattr(controller, 'difficulty',
                                          self.ai_difficulty or self.renderer.difficulty)
        if self.gs.is_browsing:
            # 正在看历史：不要用最新的点评覆盖历史那一步的点评
            self.gs.refresh_display_comment()
            return
        if self.gs.ai_thinking:
            # **AI 正在思考：一个字都不要往 ``ply_evals`` 里写。**
            # 两个理由：
            #   1. 顶栏这时显示的是实时秒表（`renderer._ai_text_parts`），
            #      此时写评估文案等于把秒表顶掉，玩家看到的就变回上一手的
            #      "AI 用时 2000ms"——正是"计时器好像被重置了"的真凶；
            #   2. `ai_think_ms` 此刻还是**上一手**的值，把它连同评估一起写进
            #      当前步，等于往棋谱里存了一条张冠李戴的用时。
            self.gs.refresh_display_comment()
            return
        result = getattr(controller, 'last_result', None)
        eval_text = ''
        if result is not None:
            mate = result.mate_in()
            # 三段各占一行，顺序与显示位置对应：
            #   第一行（顶栏）读"用时/深度"，第二行右侧读"评估"。
            # "用时"读的是 ``gs.ai_think_ms``（`_finish_ai_think` 写的**真实挂钟**
            # 用时）；渲染层与这里读同一个字段，就不会出现"面板写一个数、
            # 顶栏写另一个数"这种自相矛盾。
            used_ms = int(getattr(self.gs, 'ai_think_ms', 0) or 0)
            eval_text = (
                f"AI 用时 {used_ms}ms"
                + f"\n深度 {result.depth}　{result.samples} 采样　{result.nodes} 节点"
                + f"\nAI 评估：{result.score:+.0f}（{result.verdict}）"
                + (f"  {mate} 步杀" if mate else "")
            )
        # 点评挂在"分析时所在的步数"上（控制器会记下来），而不是当前步数——
        # 异步点评往往在 AI 落子之后才回来，用当前步数会把两手的点评搞混。
        comment = getattr(controller, 'last_comment', '') or ''
        ply = getattr(controller, 'comment_ply', None)
        if ply is not None and (comment or eval_text):
            self.gs.set_ply_comment(int(ply), comment, eval_text)
        self.gs.refresh_display_comment()


if __name__ == "__main__":
    game = ChessGame()
    game.run()
