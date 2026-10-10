# game_state.py
import copy
import os
import random
import time
from typing import Optional, Tuple

try:  # 兼容两种运行方式（PYTHONPATH 含项目根，或直接 python main.py）
    from Game.config import Config
except ImportError:  # pragma: no cover
    from config import Config

try:
    from jieqi import gui_adapter as _adapter
except ImportError:  # pragma: no cover
    try:
        from Game.jieqi import gui_adapter as _adapter
    except ImportError:
        _adapter = None

try:  # 棋谱记录是可选项，缺了也不能影响下棋
    from jieqi import record as _record
except ImportError:  # pragma: no cover
    try:
        from Game.jieqi import record as _record
    except ImportError:
        _record = None

try:  # 比赛规则（长将 / 长捉 / 判和）：纯函数模块，不依赖 pygame
    from jieqi import rules as _rules
except ImportError:  # pragma: no cover
    try:
        from Game.jieqi import rules as _rules
    except ImportError:
        _rules = None

try:  # 对局方式（每方由谁出招）的定义：纯常量模块
    from jieqi import engines as _engines
except ImportError:  # pragma: no cover
    try:
        from Game.jieqi import engines as _engines
    except ImportError:
        # 兜底：极端情况下（打包不全）也要能下棋。这里只保留"人工 / 本机引擎"
        # 两种档位，行为与"人机对战"一致。
        class _EnginesFallback:
            HUMAN = "human"
            PYTHON = "python"
            LLM = "llm"
            SIDES = ("red", "black")
            SOURCES = (HUMAN, PYTHON, LLM)
            DEFAULT_FOR_SIDE = {"red": HUMAN, "black": PYTHON}

            @staticmethod
            def normalize(source, fallback=HUMAN):
                text = str(source or "").strip().lower()
                if text in _EnginesFallback.SOURCES:
                    return text
                # 老配置里的外部引擎档位（已移除）折算成本机引擎
                if text in ("abj", "abjchess", "abjchess_then_python"):
                    return _EnginesFallback.PYTHON
                return fallback

            @staticmethod
            def available(source):
                return _EnginesFallback.normalize(source) in (
                    _EnginesFallback.HUMAN, _EnginesFallback.PYTHON)

            @staticmethod
            def reason(source):
                return "" if _EnginesFallback.available(source) else "该档位不可用"

        _engines = _EnginesFallback

import pygame


class GameState:
    def __init__(self, resource_manager, record_dir: Optional[str] = None):
        self.config = Config()
        #: 棋谱写到哪个目录（``None`` = 默认的 ``Game/records``）。
        #: 给**自动化脚本**用的：自对弈/冒烟测试跑出来的棋谱不该混进玩家的
        #: 棋谱目录里（真实教训：跑几次自检就多出好几份"测试对局"，
        #: 玩家在棋谱列表里分不清哪些是自己的）。
        self.record_dir = record_dir
        # 原有代码保持不变...
        self._init_ai()
        self._init_game_state()
        # 默认对局方式 = 原来的"人机对战"：你执红、黑方由**本机引擎**走。
        # （``_init_ai`` 本身只负责"清空/暂停"，默认值在这里显式给。）
        self.enable_ai('black')
    def _init_game_state(self):
        self.board = self._create_board()
        self.player_side = 'red'
        self.current_player = 'red'#红默认先手
        self.selected = None
        self.game_over = False
        self.winner = None
        self.red_king_pos = (0, 4)
        self.black_king_pos = (9, 4)
        self.in_check = False
        #: 走棋记录：``move_history[i]`` 是产生 ``positions[i+1]`` 的那一步。
        #: 每条记录里带 ``comment``（那一步的大模型点评），回退后能原样取回。
        self.move_history = []
        #: 当前显示的是第几手（0 = 开局，total_plies = 最新）
        self.ply = 0
        #: 每一步的大模型点评：``ply_comments[k]`` 是**对第 k 手之后那个局面**的点评。
        #: 按步数索引而不是塞进走棋记录里，是因为点评描述的是**局面**：
        #: 回退/前进时按当前步数取即可，天然不会串台、也不会被覆盖。
        self.ply_comments = ['']
        self.ply_evals = ['']
        #: **人工打分**：``ply_scores[k]`` 是"从第 k 手之后的局面走出的那一手"
        #: 的档位代码（见 ``jieqi.record.RATINGS``，如 ``"blunder"``）。
        #: 与 ``ply_comments`` 同样是**按局面索引**。
        self.ply_scores = ['']
        #: 人工备注（同样按局面索引，可选）
        self.ply_notes = ['']
        self.capture_count = 0
        #: 已经**离场**的棋子（被吃掉的），按顺序累积。
        #: 每条是 ``{'color', 'revealed', 'actual', 'original'}``——算"剩余身份池"要用它：
        #: 明子被吃 → 该身份少一个；暗子被吃 → 身份**仍然未知**，只能记"少了一枚暗子"。
        #: （``capture_piece`` 是界面用的，回退时不会恢复，所以这里单独维护一份。）
        self.captured_pieces = []
        self.capture_piece = []
        self.animations = []
        self.selected_glow_phase = 0
        self.toast = ''
        #: toast 的"过期时间"（``time.monotonic()``）。渲染器用它决定：
        #: 只在刚发生时把提示**放大显示在棋盘上**（导出棋谱这类操作必须给反馈，
        #: 否则玩家点了不知道成功没有），过几秒自动缩回顶栏那一行。
        self.toast_until = 0.0
        #: toast 是否值得"大提示"（错误/成功这类操作反馈 = True，状态类 = False）
        self.toast_loud = False
        #: 大模型顾问的中文点评（当前显示的那一手的；由主线程同步，渲染线程只读）
        self.ai_comment = ''
        #: 右侧面板显示的引擎评估摘要（同样由主线程同步）
        self.ai_eval_text = ''
        #: AI 是否正在思考 + 开始思考的时刻（``time.monotonic()``）。
        #: 渲染器靠这两个字段把顶栏那行换成**实时秒表**——旧版思考期间一直显示
        #: 上一手的用时，玩家看到的是"明明想了半分钟，还写着 2000ms"。
        self.ai_thinking = False
        self.ai_think_started = 0.0
        #: 上一次 AI 的**真实挂钟**用时（毫秒，从开始思考到落子），顶栏显示的就是它
        self.ai_think_ms = 0
        #: 本次运行的「对局结束自动保存棋谱」开关（``None`` = 跟随 ai_config.json；
        #: 启动页拨一下就写入 True/False，不必去改配置文件）
        self.auto_save_override = None
        #: 当前局面的人工打分（中文标签 + 档位代码），由主线程同步
        self.ai_score_label = ''
        self.ai_score_code = ''
        self.show_side_selection = True
        self.position_counter = {}  # 用于记录局面出现次数
        #: 每一步的**客观事实**（``jieqi.rules.MoveFacts``），与 ``move_history`` 一一对应。
        #: 长将 / 长捉要按"连续多少回合"判，必须逐手留档；只在 ``move_piece`` 里
        #: 追加（分叉时一起截断），检查侧只读——这样判和检测天然**幂等**。
        self.rule_facts = []
        #: 上一次"大声提醒"过的规则提示（去重用：``check_game_state`` 每帧都调，
        #: 同一句话不能每帧重新弹一次）
        self._last_rule_toast = ''
        self.draw_reason = None  # 平局原因
        self.move_count = 0
        #: 本局的棋谱信息（棋盘快照时一起存，方便导出/复盘）
        self.record_meta = {}
        #: 已经自动导出过的棋谱名（避免终局后每帧都写一次盘）。
        #: **只记"这一局自己写出来的文件"**——导入棋谱的名字记在
        #: ``imported_from`` 里，否则"导入 A 接着下"会把 A 的文件覆盖掉。
        self.record_saved = ''
        #: 这一局的局面来自哪份棋谱（导入的名字，仅用于显示/参考，**不当输出名**）
        self.imported_from = ''
        #: 导入棋谱时"暂存"的结果（``'1-0'`` / ``'0-1'`` / ``'1/2-1/2'`` / ``'*'``）。
        #: 不立刻生效，是为了让导入之后仍能从开局自由复盘；走到最后一手才结算。
        self._pending_result = None
        self._pending_draw_reason = ''
        #: 局面序列：``positions[k]`` 是第 k 手之后的完整局面（``positions[0]`` 是开局）。
        #: 有了它就支持"自由切换走棋步数"——前进/后退都只是切指针，不用重放。
        #: 必须放在上面这些字段都初始化之后，因为快照要读它们。
        self.positions = [self._snapshot_state()]

    def _init_ai(self):
        """把"每方由谁出招"重置成**双方都人工**（也就是暂停一切自动走子）。

        ⚠ 语义上是"**没有**引擎在替谁下棋"，不是"恢复默认对局方式"：

        * ``__init__`` / :meth:`reset` 会在调完它之后显式
          ``enable_ai('black')``，恢复成默认的"人机对战"；
        * 而"导入别人的棋谱"这类场景需要的是**暂停**（导入的是别人的对局，
          引擎不该接着往下走）——老代码就是靠 ``_init_ai()`` 实现这个暂停的
          （见 ``test_record.test_paused_after_import_does_not_let_ai_move``）。
          所以这里不能悄悄把黑方设成引擎，否则导入完棋谱引擎会自己走起来。

        真相来源是 ``side_engines``（每方一个档位，见 ``jieqi.engines``）；
        ``ai_mode`` / ``ai_color`` 是派生属性（见下面的 property）。
        """
        self.side_engines = {side: _engines.HUMAN for side in _engines.SIDES}
        self.ai_thread = None  # AI计算线程（主循环用，保留字段名）

    # ---- 每方由谁出招（对局方式）----
    def side_engine(self, side: str) -> str:
        """某一方（``'red'`` / ``'black'``）当前由谁出招。"""
        key = str(side or '').lower()
        if key not in _engines.SIDES:
            return _engines.HUMAN
        return _engines.normalize((self.side_engines or {}).get(key),
                                  fallback=_engines.DEFAULT_FOR_SIDE[key])

    def engine_sides(self):
        """当前由**引擎/大模型**（也就是非人工）出招的那几方。"""
        return [s for s in _engines.SIDES
                if self.side_engine(s) != _engines.HUMAN]

    def is_human_side(self, side: str) -> bool:
        return self.side_engine(side) == _engines.HUMAN

    def set_side_engine(self, side: str, source: str) -> bool:
        """设置某一方的档位；返回是否真的变了。

        不可用的档位会降级成本机引擎（用户点了某一档，总得给个能下的东西），
        降级原因由 ``jieqi.engines.reason(source)`` 负责说明。
        """
        key = str(side or '').lower()
        if key not in _engines.SIDES:
            return False
        source = _engines.normalize(source)
        if not _engines.available(source):
            source = _engines.HUMAN
        if self.side_engines is None:
            self.side_engines = dict(_engines.DEFAULT_FOR_SIDE)
        if self.side_engine(key) == source:
            return False
        self.side_engines[key] = source
        return True

    def set_player_side(self, side: str) -> bool:
        """设定"我执哪一方"，并**把对局方式一起摆正**；返回阵营是否变了。

        为什么要一起摆正（真踩过）：玩家改执黑之后，如果只是记下
        ``player_side='black'`` 而 ``side_engines`` 还是默认的
        "红人工 / 黑本机引擎"，那么**两边都是人工**——AI 一步都不会走，
        看起来就像"电脑坏了"。

        规则：**把"对手那一方原来由谁出招"整体搬到新的对手那一方**。
        所以"黑方=大模型"时改执黑 → 变成"红方=大模型"，对手的档位不丢；
        默认的"黑方=本机引擎"也一样能跟过去。

        ⚠ 唯一照搬会很别扭的情况：**双人对下**（两边都是人工）时改阵营，
        对手段位就是"人工"——那是如实的（双人局本来就该两边都人工），
        想恢复人机按 `F2`。
        """
        key = 'black' if str(side or '').lower() == 'black' else 'red'
        other = 'red' if key == 'black' else 'black'
        changed = (getattr(self, 'player_side', 'red') != key)
        # 先记下"原来的对手由谁出招"，再改 player_side（改完 other 就换了）
        opponent_engine = self.side_engine(
            'red' if getattr(self, 'player_side', 'red') == 'black' else 'black')
        if opponent_engine == _engines.HUMAN:
            # 对手段位是人工：默认（红人工/黑本机引擎）说明"对手还没被配过"，
            # 那就在改阵营时给它本机引擎，免得一动就变成两人对下。
            default_opponent = _engines.DEFAULT_FOR_SIDE.get(other, _engines.PYTHON)
            if self.side_engine(other) == _engines.HUMAN and \
                    self.side_engine(key) != _engines.HUMAN:
                opponent_engine = default_opponent
        self.player_side = key
        self.side_engines = self.side_engines or {}
        self.side_engines[key] = _engines.HUMAN
        self.side_engines[other] = opponent_engine
        return changed

    @property
    def ai_mode(self):
        """是否有人机对战（派生）。

        **任一方**由引擎/大模型出招就是 True。这样"黑方大模型、红方人工"、
        "红黑都是引擎"（自动对弈）都算得上"AI 模式"。
        没有任何一方是引擎时返回 False（与老代码里"双人对下"的语义一致）。
        """
        return bool(self.engine_sides())

    @ai_mode.setter
    def ai_mode(self, value):
        """兼容老代码的直接赋值。

        ``True`` 时不动具体档位（保持现状）；``False`` 时把两方都设为人工
        ——老代码里 ``ai_mode = False`` 的语义就是"双人对下"。
        """
        if not value:
            self.side_engines = {s: _engines.HUMAN for s in _engines.SIDES}

    @property
    def ai_color(self):
        """AI 执哪一方（派生）。

        给界面/棋谱显示用的**单值**：两方都是引擎时给 ``'black'``
        （传统"人机对战"里 AI 执黑），只在显示处使用——**判定"该谁走"
        一律用** ``side_engine(current_player)``，不要再读这个属性。
        """
        sides = self.engine_sides()
        if not sides:
            return ''
        if 'black' in sides:
            return 'black'
        return sides[0]

    @ai_color.setter
    def ai_color(self, value):
        """兼容老代码的直接赋值（``gs.ai_color = 'black'``）。"""
        text = str(value or '').lower()
        if text not in _engines.SIDES:
            if not text:
                self.side_engines = {s: _engines.HUMAN for s in _engines.SIDES}
            return
        other = 'red' if text == 'black' else 'black'
        self.side_engines[text] = _engines.PYTHON
        self.side_engines[other] = _engines.HUMAN

    def reset(self, keep_engines: bool = True):
        """重置游戏状态。

        ``keep_engines=True``（默认）**保留**每方的对局方式：玩家在启动页上
        挑好的"红方人工 / 黑方大模型"不该因为点了"重新开始"就悄悄变回默认
        ——那等于把设置吃了（用户真实会踩：换阵营时设置被重置）。

        需要回到默认时传 ``keep_engines=False``，或显式调
        :meth:`set_side_engine`。
        """
        engines = dict(self.side_engines or {})
        self._init_game_state()
        self._init_ai()
        if keep_engines and engines:
            self.side_engines.update(engines)
        print("游戏已重置")

    def apply_engine_settings(self, settings) -> None:
        """按 ``ai_config.json`` 的 ``ai`` 段设置每方的对局方式（启动时调一次）。

        两种写法都认，优先级从高到低：

        1. ``ai.side_engines = {"red": "human", "black": "llm"}``（**首选**）；
        2. 老写法 ``ai.move_source = "python"``：折算成"黑方用它、红方人工"
           ——老配置一字不改就能继续用。

        没配任何一项时保持当前的默认（红方人工 / 黑方本机引擎）。
        不可用的档位会被 :meth:`set_side_engine`
        降级并记下来，不会让某一方卡住。
        """
        if not isinstance(settings, dict):
            return
        configured = settings.get("side_engines")
        # 玩家执哪一方（``ai.player_side``）：启动页上没有阵营选项了，
        # 所以这是"换边"的主要入口；不写就是默认红方先手。
        wanted_side = str(settings.get("player_side") or "").strip().lower()
        if wanted_side in _engines.SIDES:
            self.set_player_side(wanted_side)
        if isinstance(configured, dict) and configured:
            for side in _engines.SIDES:
                value = configured.get(side)
                if value:
                    self.set_side_engine(side, str(value))
            return
        source = str(settings.get("move_source") or "").strip().lower()
        if source and source != "python":
            # 老写法：move_source 管的是"AI 那一方"，另一方是人工。
            self.set_side_engine("black", source)
            self.set_side_engine("red", _engines.HUMAN)

    # ------------------------------------------------------------ 棋谱
    def build_record(self, extra_meta: Optional[dict] = None) -> Optional[dict]:
        """把当前对局组装成棋谱字典（``jieqi.record`` 不可用时返回 ``None``）。"""
        if _record is None:
            return None
        meta = dict(getattr(self, 'record_meta', None) or {})
        meta.update(extra_meta or {})
        meta.setdefault('player_side', self.player_side)
        try:
            return _record.build_record(self, meta=meta)
        except Exception as exc:  # 组装棋谱失败绝不能影响下棋
            print(f"[揭棋AI] 组装棋谱失败：{exc}")
            return None

    def export_record(self, name: Optional[str] = None,
                      directory: Optional[str] = None,
                      unique: bool = False) -> dict:
        """把当前对局导出成棋谱（``.json`` + ``.txt``）。

        返回 ``{ok, path, text_path, requested, error}``；失败只给中文原因，不抛异常。
        导出**不会**写入暗子真身（见 ``jieqi.record`` 的说明）。

        ``unique=True`` 表示"**一定**开新文件"（终局自动保存用），所以导入来的
        棋谱永远不会被续下的一局改写。默认（``unique=False``）时优先复用
        ``record_saved``——同一局连点两次「导出棋谱」只会更新同一份，不产生垃圾。

        ``directory`` 没给时用 ``self.record_dir``（构造时可指定，自动化脚本专用），
        再退回默认的 ``Game/records``。
        """
        if _record is None:
            return {"ok": False, "path": "", "text_path": "", "requested": "",
                    "error": "棋谱模块不可用"}
        if directory is None:
            directory = self.record_dir
        chosen = None if unique else (name or self.record_saved)
        record = self.build_record({"name": chosen or ""})
        if record is None:
            return {"ok": False, "path": "", "text_path": "", "requested": "",
                    "error": "组装棋谱失败"}
        result = _record.save_record(record, directory=directory, name=chosen,
                                     unique=unique, mine=self.record_saved)
        if result.get("ok"):
            # 记**实际写出的**名字（防撞改名后 record_saved 必须跟着变，
            # 否则提示语会骗人、下次导出还会去撞同一个名字）
            written = os.path.splitext(os.path.basename(result.get("path") or ""))[0]
            if written:
                self.record_saved = written
        return result

    # ------------------------------------------------------- 导入棋谱
    def import_record(self, record: dict, player_side: Optional[str] = None) -> dict:
        """把一份棋谱**重放**成当前对局（用于导入历史棋谱继续复盘）。

        重放而不是"直接套用棋盘快照"，理由有两条：

        1. 棋谱里暗子只存摆位身份（不存真身），重放时按"第几手翻出了什么"
           现场补上身份，走法合法性才和当初一致；
        2. 重放顺带产出了 ``positions`` / ``move_history``，于是导入之后
           **上一步/下一步、回顾棋局、点评按步数取**全都能照常用。

        返回 ``{ok, plies, name, warnings, error}``；任何异常都转成中文错误说明，
        不会让界面崩掉。
        """
        if _record is None:
            return {"ok": False, "plies": 0, "name": "", "warnings": [],
                    "error": "棋谱模块不可用"}
        if not isinstance(record, dict):
            return {"ok": False, "plies": 0, "name": "", "warnings": [],
                    "error": "棋谱内容不是对象"}
        moves, reveals, meta = _record.replay_plan(record)
        if not moves:
            return {"ok": False, "plies": 0, "name": meta.get("name", ""),
                    "warnings": [], "error": "这份棋谱里没有任何着法"}

        # 从标准开局重来（保留难度等配置，不动 AI 设置）
        self._init_game_state()
        self.show_side_selection = False
        side_name = player_side or meta.get("player_side") or "red"
        self.player_side = "black" if str(side_name) == "black" else "red"
        self.current_player = "red"        # 揭棋红先

        warnings: List[str] = []
        for index, (sr, sc, er, ec) in enumerate(moves, start=1):
            # 这一步要翻开的那枚暗子，棋谱里记着"翻出的是什么"，重放时补上，
            # 这样后续的走法合法性与真实对局完全一致。
            setup = reveals.get(index)
            if setup is not None:
                cell = None
                try:
                    cell = self.board[setup[0]][setup[1]]
                except IndexError:
                    cell = None
                if cell is not None and not cell.get("revealed"):
                    cell["actual"] = setup[2]
            if _adapter is not None:
                # 走法必须合法（棋谱坏掉/被改过时不要静默走出非法局面）
                legal = False
                try:
                    side_int = 0 if self.current_player == "red" else 1
                    legal = ((sr * 9 + sc) << 7 | (er * 9 + ec)) in \
                        _adapter.legal_moves(self.board, side_int)
                except Exception:
                    legal = True
                if not legal:
                    warnings.append("第 %d 手 %s 在当前局面里不合法，已停止重放"
                                    % (index, self._coord_text(sr, sc, er, ec)))
                    break
            self.move_piece(sr, sc, er, ec)
            self.current_player = "black" if self.current_player == "red" else "red"

        # 结果**不立刻生效**：棋谱通常是已经下完的，如果一导入就进入"结算"
        # 状态，界面会立刻弹出"红方胜利 / 点击任意位置重新开始"，把整盘棋锁死，
        # 连"上一步/下一步"都点不了（这个 bug 真实出现过）。
        # 所以这里只把结果**记下来**，等玩家真的走到终局那一手时再判定结束。
        result = str(meta.get("result") or "*")
        self._pending_result = result
        self._pending_draw_reason = str(meta.get("draw_reason") or "")
        self.game_over = False
        self.winner = None
        self.draw_reason = None

        # 导入之后**默认停在开局**，玩家按「下一步」一步步复盘。
        # 注意：重放到最后时棋盘停在终局，所以必须真的**把局面切回开局**
        # （直接写 self.ply = 0 只改了指针、棋盘还是终局的——那样"第 0 手"
        #  却画着终局棋盘，前后矛盾）。
        self.ply = int(self.total_plies)      # 先当作"在最新一手"
        # 导入的名字记到 ``imported_from``，**不写进 record_saved**：后者是
        # "这一局自己写出来的文件"，拿导入名去覆盖会把别人的棋谱毁掉。
        self.imported_from = str(meta.get("name") or "")
        self.record_saved = ''
        # 棋谱里的分支（``variations``）字段**读过就丢**：分支棋谱功能已整体移除，
        # 但老棋谱里可能还留着这个字段 —— 不读它、也不报错，就当没这回事。
        self._restore_record_comments(record)
        self.goto_ply(0)                      # 真正把棋盘/回合/标记切回开局
        self.refresh_display_comment()
        self.selected = None
        self.animations.clear()
        return {"ok": True, "plies": self.total_plies,
                "name": meta.get("name", ""), "warnings": warnings, "error": "",
                "result": result}

    def _apply_pending_result(self) -> None:
        """走到导入棋谱的终局那一手时，把结果真正落地。

        只有 ``ply == total_plies``（也就是玩家翻到了最后一手）才判定结束，
        中途浏览历史永远不会触发结算画面。
        """
        result = getattr(self, "_pending_result", None)
        if not result or int(getattr(self, "ply", 0)) < int(self.total_plies):
            return
        if result == "1-0":
            self.game_over, self.winner = True, "red"
        elif result == "0-1":
            self.game_over, self.winner = True, "black"
        elif result == "1/2-1/2":
            self.game_over, self.winner = True, "draw"
            if self._pending_draw_reason:
                self.draw_reason = self._pending_draw_reason
        self._pending_result = None

    def _restore_record_comments(self, record: dict) -> None:
        """把棋谱里的点评/评估/**人工打分**按局面格填回。

        棋谱里第 i 手的 ``comment``/``eval``/``human`` 描述的是**走这步之前**的
        局面（也就是第 i-1 手之后的局面），所以填回下标 ``i-1``——与存盘时严格
        互逆，否则导入后再"上一步/下一步"会看到错位的点评和分数。
        """
        total = self.total_plies
        comments = [""] * (total + 1)
        evals = [""] * (total + 1)
        scores = [""] * (total + 1)
        notes = [""] * (total + 1)
        for move in record.get("moves") or []:
            try:
                index = int(move.get("index") or 0)
            except (TypeError, ValueError):
                continue
            slot = index - 1
            if not (0 <= slot <= total):
                continue
            if move.get("comment"):
                comments[slot] = str(move["comment"])
            if move.get("eval"):
                evals[slot] = str(move["eval"])
            human = move.get("human")
            if isinstance(human, dict):
                code = str(human.get("rating") or "").strip()
                if _record is None or _record.valid_rating(code):
                    scores[slot] = code
                if human.get("note"):
                    notes[slot] = str(human["note"])[:500]
        self.ply_comments = comments
        self.ply_evals = evals
        self.ply_scores = scores
        self.ply_notes = notes

    @staticmethod
    def _coord_text(sr: int, sc: int, er: int, ec: int) -> str:
        """一手棋的坐标写法（``b3c3``，列字母 + 行数字，红方底线是第 1 行）。

        与棋谱、走棋记录、提示词的着法写法一致，见 :mod:`jieqi.notation`。
        """
        from jieqi.notation import move_name

        return move_name((sr, sc), (er, ec))

    def import_record_file(self, path: str,
                           player_side: Optional[str] = None) -> dict:
        """从文件导入棋谱（``path`` 可以是棋谱名、相对路径或绝对路径）。"""
        if _record is None:
            return {"ok": False, "plies": 0, "name": "", "warnings": [],
                    "error": "棋谱模块不可用"}
        record = _record.load_record(path)
        if record is None:
            return {"ok": False, "plies": 0, "name": "", "warnings": [],
                    "error": "读不到棋谱：%s" % path}
        return self.import_record(record, player_side=player_side)
    def enable_ai(self, color: str):
        """兼容老接口：让 ``color`` 那方由**本机引擎**出招，另一方人工。

        老代码/测试大量调用 ``gs.enable_ai('black')``，语义就是"AI 执黑"
        ——现在等价于 ``side_engines = {'red': 'human', 'black': 'python'}``。
        想指定别的引擎（例如大模型）用 :meth:`set_side_engine`。
        """
        self.ai_color = color          # 走下面的 setter（会写好两方档位）

    def _create_board(self):
        """创建初始棋盘"""
        board = [[None for _ in range(9)] for _ in range(10)]
        # 初始化将帅
        board[0][4] = {'original': '帅', 'actual': '帅', 'color': 'red', 'revealed': True, 'moved': False}
        board[9][4] = {'original': '将', 'actual': '将', 'color': 'black', 'revealed': True, 'moved': False}

        # 棋子池
        pools = {
            'red': ['仕'] * 2 + ['相'] * 2 + ['车'] * 2 + ['马'] * 2 + ['炮'] * 2 + ['兵'] * 5,
            'black': ['士'] * 2 + ['象'] * 2 + ['車'] * 2 + ['馬'] * 2 + ['砲'] * 2 + ['卒'] * 5
        }
        random.shuffle(pools['red'])
        random.shuffle(pools['black'])

        # 布置棋子
        for r in range(10):
            for c in range(9):
                if self.config.TRADITIONAL_LAYOUT[r][c] and self.config.TRADITIONAL_LAYOUT[r][c] not in ['帅', '将']:
                    color = 'red' if r < 5 else 'black'
                    board[r][c] = {
                        'original': self.config.TRADITIONAL_LAYOUT[r][c],
                        'actual': pools[color].pop(),
                        'color': color,
                        'revealed': False,
                        'moved': False
                    }
        return board

    def _convert_coords(self, r, c):
        """坐标转换（根据玩家视角）"""
        if self.player_side == 'black':
            return r, c
        return 9 - r, 8 - c

    def move_piece(self, sr: int, sc: int, er: int, ec: int):
        """执行棋子移动"""
        piece = copy.deepcopy(self.board[sr][sc])
        target = self.board[er][ec]
        was_hidden = not piece['revealed']

        self.animations.append({
            'type': 'move',
            'start_pos': (sr, sc),
            'end_pos': (er, ec),
            'piece': piece,
            'flip': was_hidden,  # 添加翻面标记
            'start_time': pygame.time.get_ticks(),
            'duration': 300
        })

        # 如果当前正在浏览历史，就在"这一手"之后走出新的主线：截断后面所有记录。
        # （曾经这里还会把被换掉的那一段"存成分支"，分支棋谱已整体移除。）
        if self.ply < len(self.positions) - 1:
            del self.positions[self.ply + 1:]
            del self.move_history[self.ply:]
            # 被吃子列表也要跟着截断，否则"剩余身份池"会算进那条作废分支里丢掉的子
            # （captured_pieces 有 n 条时对应"已经走了 n 手"，所以切在 self.ply）
            del self.captured_pieces[self.ply:]
            # 规则事实同理：作废分支的将军 / 捉子记录必须一起丢掉，
            # 否则"连续 6 回合长将"会把已经不存在的那条线也算进来。
            del self.rule_facts[self.ply:]
            self.position_counter.clear()

        # 保存走棋记录
        # 注意：piece 是**落子前**的快照（此时还是暗子），所以这里额外记下
        # 翻开后的真实身份。旧代码没有这个字段，导致走棋记录和大模型提示词
        # 在翻子后仍然把棋子说成"暗子"，与棋盘自相矛盾。
        move_record = {
            'piece': copy.deepcopy(piece),
            'start': (sr, sc),
            'end': (er, ec),
            'captured': copy.deepcopy(target) if target else None,
            'captured_sq': (er, ec) if target else None,
            'revealed_as': piece['actual'] if was_hidden else None,
            #: **走这一步的是哪一方**。显式记下来，不要靠"棋子颜色"去推——
            #: 主循环要靠它判断"轮到的一方是不是刚走完的那一方"，从而避免
            #: AI 连走两步（用户真实报过这个 bug）。
            'side': 0 if str(self.current_player) == 'red' else 1,
            #: 走这一手时，**这一方用的是哪个档位**（人工/本机引擎/大模型）。
            #: 记下来才能在复盘时如实说"这手是谁下的"——两方可以配不同引擎，
            #: 而且中途还能换档位，光看棋子颜色分不出来。
            'engine': self.side_engine(self.current_player),
        }
        self.move_history.append(move_record)

        # 记录这一步的**客观事实**（长将 / 长捉 / 空步判和都要按手判）。
        # ⚠ 位置很关键：必须是"走棋记录已经写进 move_history、但**棋盘还没动**"
        # 的这一瞬间——"被吃的是哪枚子""走之前是不是暗子"只有此刻才看得到。
        # ``current_player`` 此刻也还是刚走完的这一方（调用方之后才切换）。
        self._record_rule_facts(was_hidden=was_hidden)

        # 更新棋子状态
        if was_hidden:
            piece['revealed'] = True
        piece['moved'] = True

        # 处理吃子
        if target:
            self.capture_piece.append(target)
            self.captured_pieces.append({
                'color': target.get('color'),
                'revealed': bool(target.get('revealed')),
                'actual': target.get('actual'),
                'original': target.get('original'),
            })
            self.capture_count = 0
            if target['actual'] in ('将', '帅'):
                self.winner = piece['color']
                self.game_over = True
        else:
            self.capture_count += 1

        # 更新棋盘
        self.board[er][ec] = piece
        self.board[sr][sc] = None

        # 更新将帅位置
        if piece['actual'] == '帅':
            self.red_king_pos = (er, ec)
        elif piece['actual'] == '将':
            self.black_king_pos = (er, ec)

        # 检查将军状态
        opponent = 'black' if self.current_player == 'red' else 'red'
        self.in_check = self._check_check(self.board, opponent, self.red_king_pos, self.black_king_pos)
        if self.in_check:
            self.set_toast(f'{opponent}处于被将军状态')

        # 走到最新一手：记录**落子之后**的局面快照（支持自由前进/后退）
        self.move_count += 1  # 新增回合计数器
        self.positions.append(self._snapshot_state())
        self.ply = len(self.positions) - 1
        # 为新局面预留一个空的点评位（这一次走子产生了第 ply+1 个局面）。
        # ⚠ 必须**原地修改**（切片赋值），不能 `x = x[:n] + ['']`——那样会新建
        # 一个列表，界面/棋谱那边持有的引用就断了（打分和点评会静默写丢）。
        for seq in (self.ply_comments, self.ply_evals,
                    self.ply_scores, self.ply_notes):
            del seq[self.ply + 1:]
            seq.append('')
        self.refresh_display_comment()

        # 更新重复检测系统
        current_hash = self._board_hash()
        # 仅保留最近5次相同局面的记录（内存优化）
        if current_hash in self.position_counter:
            self.position_counter[current_hash] = self.position_counter[current_hash][-4:] + [self.move_count]
        else:
            self.position_counter[current_hash] = [self.move_count]

    # ---------------------------------------------------- 比赛规则（jieqi.rules）
    def _record_rule_facts(self, was_hidden: bool = False) -> None:
        """算好并记下这一步的事实。

        ⚠ 调用点在 ``move_piece`` **落子之后**，此时起点已经空了——所以
        "这枚子走之前是不是暗子"必须由调用方（落子前就知道）用 ``was_hidden``
        带进来。少了它，"翻子"永远算不出来，规则 4 的"翻子清零"就整个失效
        （这个坑真实踩过）。
        """
        if _rules is None:
            return
        try:
            rec = self.move_history[-1]
            (sr, sc), (er, ec) = rec['start'], rec['end']
            board = _adapter.gui_board_to_engine(
                self.board,
                turn=_adapter.gui_color_to_side(self.current_player))
            # 走法编码 ``(from << 7) | to``（与引擎一致，两个格子都 < 90 < 128）
            move = (sr * 9 + sc) << 7 | (er * 9 + ec)
            facts = _rules.fact_after_move(board, move, self.current_player,
                                           was_hidden=was_hidden)
        except Exception as exc:      # 算不出来绝不能影响下棋（宁可不判长将）
            print(f"[揭棋AI] 规则事实计算失败：{exc}")
            facts = None
        self.rule_facts.append(facts)

    def _rule_facts_upto_ply(self) -> list:
        """取到**当前显示的那一手**为止的事实序列（浏览历史时只看历史）。"""
        if _rules is None:
            return []
        return _rules.facts_from_game_state(self, upto=self.ply)

    def rule_status(self) -> dict:
        """当前是否有人触犯长将 / 长捉禁招。

        返回 ``{'forbidden', 'offender', 'reason'}``：

        * ``forbidden``：是否已经"必须变招"（判据见 ``jieqi.rules``）；
        * ``offender``：``'red'`` / ``'black'`` / ``None``；
        * ``reason``：中文原因，可以直接显示给玩家。

        这是**只读**查询，重复调用结果完全一致（每帧调都安全）。
        """
        if _rules is None:
            return {'forbidden': False, 'offender': None, 'reason': ''}
        try:
            return _rules.rule_violation(self._rule_facts_upto_ply())
        except Exception as exc:      # pragma: no cover - 防御：绝不让界面崩
            print(f"[揭棋AI] 规则状态检查失败：{exc}")
            return {'forbidden': False, 'offender': None, 'reason': ''}

    def _remind_rule_violation(self) -> None:
        """违规时大声提醒"某某方必须变招"（同一句话不反复刷屏）。"""
        status = self.rule_status()
        if not status.get('forbidden'):
            return
        reason = str(status.get('reason') or '')
        if not reason or reason == self._last_rule_toast:
            return
        self._last_rule_toast = reason
        self.set_toast(reason, loud=True)

    # ==================================================== 局面快照 / 步数切换
    def _snapshot_state(self) -> dict:
        """把当前局面存成一份完整快照（供自由前进/后退使用）。"""
        return {
            'board': copy.deepcopy(self.board),
            'current_player': self.current_player,
            'red_king_pos': self.red_king_pos,
            'black_king_pos': self.black_king_pos,
            'in_check': self.in_check,
            'capture_count': self.capture_count,
            'move_count': self.move_count,
            'game_over': self.game_over,
            'winner': self.winner,
            'draw_reason': self.draw_reason,
            #: 已被吃的子（算"剩余身份池"要用；回退时必须跟着回来）
            'captured_pieces': copy.deepcopy(self.captured_pieces),
        }

    @property
    def has_rule_facts(self) -> bool:
        """是否已经按手记下了规则事实（长度必须与走棋记录一致）。"""
        return len(getattr(self, 'rule_facts', None) or ()) == len(self.move_history)

    @property
    def total_plies(self) -> int:
        """已经走过的总手数。"""
        return len(self.positions) - 1

    @property
    def state_history(self):
        """兼容旧代码：已经走过的那些局面（不含当前显示的最新一手）。

        做成属性而不是独立列表，避免和 ``positions`` 不同步。
        """
        return self.positions[:self.ply]

    @property
    def is_browsing(self) -> bool:
        """是否正在浏览历史（不是最新局面）。"""
        return self.ply < self.total_plies

    def _last_mover_at(self, ply: int) -> Optional[str]:
        """第 ``ply`` 手之后的局面里，**刚走完的是哪一方**（0 手时返回 ``None``）。

        走法历史里显式记了 ``side``；老棋谱没有这个字段时退回按棋子颜色推。
        """
        history = self.move_history or []
        index = int(ply) - 1
        if index < 0 or index >= len(history):
            return None
        rec = history[index]
        side = rec.get('side')
        if side in (0, 1):
            return 'red' if int(side) == 0 else 'black'
        if side in ('red', 'black'):
            return str(side)
        color = str((rec.get('piece') or {}).get('color') or '')
        return color if color in ('red', 'black') else None

    def _side_to_move_at(self, ply: int) -> str:
        """第 ``ply`` 手之后的局面里，**该谁走**。

        红先，之后逐手交替——所以"该谁走"永远是"刚走完的那一方的对面"。
        这是导航（上一步/下一步）唯一该用来决定轮次的地方；
        **不要**读快照里的 ``current_player``（那是"走完时"的值，见 ``goto_ply``）。
        """
        last = self._last_mover_at(ply)
        if last is None:
            return 'red'                     # 开局红先
        return 'black' if last == 'red' else 'red'

    def goto_ply(self, target: int) -> bool:
        """跳到第 ``target`` 手之后的局面（0 = 开局）。

        这是"自由切换走棋步数"的核心：局面本来就已经存在 ``positions`` 里，
        切换只是恢复快照 + 移动指针，不需要重放，也不会访问大模型。
        返回是否真的发生了移动。
        """
        target = max(0, min(int(target), self.total_plies))
        if target == self.ply:
            return False
        # 注意：这里**不能**截断点评/打分数组。旧代码靠 `x = x[:n] + ['']`
        # 的重新赋值"顺带"实现，结果就是"回退两步再前进，点评不见了"——
        # 那是 bug，不是特性（``test_goto_restores_board_and_comment`` 钉着它）。
        # 真正需要截断的时机是**分叉**（在历史中间走新着法），见 ``move_piece``。
        state = self.positions[target]
        self.board = copy.deepcopy(state['board'])
        # ⚠ **不要**用 ``state['current_player']``！
        # 快照是在 ``move_piece`` 里"走完这步之后"存的，而那一刻
        # ``current_player`` 还是**刚走完的那一方**（切换是调用方在之后做的）。
        # 所以直接读快照就会得到"轮到刚走完的一方"——退回一步后界面提示
        # 还是"轮到红方"，实际棋盘却已是"红走完、该黑走"，于是红方连走两步
        # （用户真实报的 bug）。
        # 正确做法：从**走法历史**推出下一手该谁——红先，之后逐手交替。
        self.current_player = self._side_to_move_at(target)
        self.red_king_pos = state['red_king_pos']
        self.black_king_pos = state['black_king_pos']
        self.in_check = state['in_check']
        self.capture_count = state['capture_count']
        self.captured_pieces = copy.deepcopy(state.get('captured_pieces') or [])
        self.move_count = state.get('move_count', target)
        self.game_over = state.get('game_over', False)
        self.winner = state.get('winner')
        self.draw_reason = state.get('draw_reason')
        self.ply = target
        self.selected = None
        self.animations.clear()
        self.position_counter.clear()
        # 浏览历史时"连续将军 / 捉子"只能算到**看到的那一手**为止。
        # 事实是逐手存下来的（``rule_facts[i]`` 对应第 i+1 手），
        # ``_rule_facts_upto_ply`` 会按当前 ``ply`` 切开，所以这里**不用**动
        # ``rule_facts`` 本身——否则再前进回来时那些事实就永久丢失了。
        # 导入的棋谱：只有真的走到最后一手才结算（中途翻历史不弹结算画面）
        self._apply_pending_result()
        self.refresh_display_comment()
        return True

    # ------------------------------------------------- 点评（按步数存档）
    def comment_for_ply(self, k: int) -> str:
        """取第 ``k`` 手之后那个局面的点评。

        如果那一步没有点评（例如玩家自己走的），就退回上一步的点评——
        它解释的正是"为什么会走到现在这个局面"。
        """
        for idx in (k, k - 1):
            if 0 <= idx < len(self.ply_comments) and self.ply_comments[idx]:
                return self.ply_comments[idx]
        return ''

    def eval_for_ply(self, k: int) -> str:
        for idx in (k, k - 1):
            if 0 <= idx < len(self.ply_evals) and self.ply_evals[idx]:
                return self.ply_evals[idx]
        return ''

    def set_ply_comment(self, k: int, comment: str, eval_text: str = '') -> None:
        """把点评存到第 ``k`` 手（``k`` 越界时忽略，绝不写坏别的步数）。"""
        if not (0 <= k < len(self.ply_comments)):
            return
        if comment:
            self.ply_comments[k] = comment
        if eval_text:
            self.ply_evals[k] = eval_text
        if k == self.ply:
            self.refresh_display_comment()


    # ------------------------------------------------- 人工打分（五档）
    def score_for_ply(self, k: int) -> str:
        """取"从第 ``k`` 手之后的局面走出的那一手"的人工档位代码。"""
        if 0 <= k < len(self.ply_scores):
            return str(self.ply_scores[k] or '')
        return ''

    def current_score(self) -> str:
        """当前显示局面对应的分数（站在第 k 手 = 给第 k+1 手打分）。"""
        return self.score_for_ply(self.ply)

    def set_ply_score(self, k: int, code: str) -> bool:
        """给"从第 ``k`` 手之后的局面走出的那一手"打分（``code=''`` = 清除）。

        返回是否真的改动。越界或非法档位一律忽略——打分是辅助功能，
        绝不能因为它把棋局数据写坏。
        """
        if not (0 <= k < len(self.ply_scores)):
            return False
        text = str(code or '').strip()
        if text and _record is not None and not _record.valid_rating(text):
            return False
        if self.ply_scores[k] == text:
            return False
        self.ply_scores[k] = text
        if k == self.ply:
            self.refresh_display_comment()
        return True

    def set_current_score(self, code: str) -> bool:
        return self.set_ply_score(self.ply, code)

    def label_for_ply(self, k: int) -> str:
        """档位代码 -> 中文标签（给界面用）。"""
        if _record is None:
            return ''
        return _record.rating_label(self.score_for_ply(k))

    def set_ply_note(self, k: int, text: str) -> bool:
        """存一条人工备注（与打分同一套按局面索引的规则）。"""
        if not (0 <= k < len(self.ply_notes)):
            return False
        value = str(text or '')[:500]
        if self.ply_notes[k] == value:
            return False
        self.ply_notes[k] = value
        return True

    def note_for_ply(self, k: int) -> str:
        if 0 <= k < len(self.ply_notes):
            return str(self.ply_notes[k] or '')
        return ''

    def refresh_display_comment(self) -> None:
        """把"当前这一步的点评"搬到渲染线程读的字段上。"""
        self.ai_comment = self.comment_for_ply(self.ply)
        self.ai_eval_text = self.eval_for_ply(self.ply)
        # 人工打分（渲染线程只读）
        self.ai_score_label = self.label_for_ply(self.ply)
        self.ai_score_code = self.score_for_ply(self.ply)

    def validate_move(self, sr: int, sc: int, er: int, ec: int, board=None) -> bool:
        """验证移动合法性"""
        board = board or self.board
        piece = board[sr][sc]
        target = board[er][ec]
        if target and target['color'] == piece['color']:
            return False
        if piece['actual'] in ['将', '帅']:
            return self.config._validate_king(sr, sc, er, ec, board)
        # 根据棋子类型调用对应的验证器：暗子按摆位身份，明子按真实身份。
        # 用 revealed 判断（与引擎一致）——旧代码用 moved，两个标志一旦不同步就会错判。
        rule_type = piece['original'] if not piece['revealed'] else piece['actual']
        validator_name = self.config.MOVE_VALIDATORS.get(rule_type, '_invalid_move')
        validator = getattr(self.config, validator_name, self.config._invalid_move)
        return validator(sr, sc, er, ec, board)

    def _check_face_to_face(self, board, red_king_pos: Tuple[int, int], black_king_pos: Tuple[int, int]) -> bool:
        """检查将帅照面"""
        if not red_king_pos or not black_king_pos:
            return False
        if red_king_pos[1] != black_king_pos[1]:
            return False
        start = min(red_king_pos[0], black_king_pos[0]) + 1
        end = max(red_king_pos[0], black_king_pos[0])
        return not any(board[r][red_king_pos[1]] for r in range(start, end))

    def _check_check(self, board, color: str, red_king_pos: Tuple[int, int], black_king_pos: Tuple[int, int]) -> bool:
        """检查将军状态"""
        king_pos = red_king_pos if color == 'red' else black_king_pos
        opponent = 'black' if color == 'red' else 'red'

        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece and piece['color'] == opponent:
                    if self.validate_move(r, c, king_pos[0], king_pos[1], board):
                        return True
        return False

    def _get_king_positions(self, board) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """获取将帅位置"""
        red_king = None
        black_king = None
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece:
                    if piece['actual'] == '帅':
                        red_king = (r, c)
                    elif piece['actual'] == '将':
                        black_king = (r, c)
        return red_king, black_king

    def is_checkmate(self, color: str) -> bool:
        """是否被将死 / 困毙。

        旧实现对每个子、每个目标格都做一次 deepcopy 再全盘扫描（复杂度极高，
        每次都让界面卡顿）。现在直接交给引擎的走法生成器判断，
        语义完全一致（象棋里无棋可走即判负），速度快几百倍。
        """
        if _adapter is not None:
            side = 0 if color == 'red' else 1
            return not _adapter.has_legal_moves(self.board, side)
        return self._is_checkmate_slow(color)

    def _is_checkmate_slow(self, color: str) -> bool:
        for r in range(10):
            for c in range(9):
                piece = self.board[r][c]
                if not piece or piece['color'] != color:
                    continue
                for nr in range(10):
                    for nc in range(9):
                        if self.validate_move(r, c, nr, nc):
                            temp_board = copy.deepcopy(self.board)
                            temp_piece = copy.deepcopy(piece)
                            temp_board[r][c] = None
                            temp_board[nr][nc] = temp_piece
                            temp_red, temp_black = self._get_king_positions(temp_board)
                            current_king_pos = temp_red if color == 'red' else temp_black
                            if not current_king_pos:
                                continue
                            in_check = self._check_check(temp_board, color, temp_red, temp_black)
                            face_to_face = self._check_face_to_face(temp_board, temp_red, temp_black)
                            if not (in_check or face_to_face):
                                return False
        return True

    # ------------------------------------------------------- 前进 / 后退
    def undo_move(self, steps: int = 1):
        """后退 ``steps`` 手（不访问大模型，也不丢历史）。"""
        if self.total_plies == 0:
            self.set_toast("无法回退：还没有走过棋")
            return False
        if self.ply == 0:
            self.set_toast("已经回到开局")
            return False
        steps = max(1, int(steps))
        moved = self.goto_ply(self.ply - steps)
        if moved:
            self.game_over = False
            self.selected = None
            self.capture_piece.clear()
        return moved

    def redo_move(self, steps: int = 1):
        """前进 ``steps`` 手（回到之前回退掉的那一步）。"""
        if not self.is_browsing:
            self.set_toast("已经在最新一步")
            return False
        moved = self.goto_ply(self.ply + max(1, int(steps)))
        return moved

    def undo_round(self):
        """回退一个回合：把玩家和 AI 各一手一起退掉（更符合直觉）。"""
        steps = 2 if self.ply >= 2 else self.ply
        return self.undo_move(steps) if steps else False

    #: 兼容旧调用名
    def _restore_state(self, state):
        """（旧接口）直接从一份快照恢复局面。"""
        self.board = copy.deepcopy(state['board'])
        self.current_player = state['current_player']
        self.red_king_pos = state['red_king_pos']
        self.black_king_pos = state['black_king_pos']
        self.in_check = state['in_check']
        self.capture_count = state['capture_count']
        self.captured_pieces = copy.deepcopy(state.get('captured_pieces') or [])
        self.selected = None

    def check_game_state(self):
        """检查游戏结束条件（**幂等**：每帧都能安全调用）。

        判和条件全部收拢到 ``jieqi.rules``，这里只负责"按优先级问一遍 + 落地"：

        1. **无进攻子力**：双方都没有"可以过河"的子力（按用户口径）；
        2. **空步判和**：连续 80 步没有吃子 / 翻子（将军与应将不计不清）；
        3. **局面重复** ``RULE['repetition_draw']`` 次（本项目 = 6）；
        4. **最大步数** 200 步。

        顺带提醒长将 / 长捉违规方"必须变招"（``rule_status``，只读）。
        """
        # 绝杀检测
        if self.is_checkmate(self.current_player):
            self.winner = 'black' if self.current_player == 'red' else 'red'
            self.game_over = True

        # 长将 / 长捉：违规方必须变招（只提醒，不直接判负——规则是"禁着"）
        self._remind_rule_violation()

        # 局面重复：达到提醒线就每步提示一句（第 4 次起提醒，第 6 次判和）
        self._remind_repetition()

        draw = self._evaluate_draw()
        if draw:
            self.winner = 'draw'
            self.draw_reason = draw['reason']
            self.game_over = True
            self._auto_export_record()
            return
        if self.game_over:
            self._auto_export_record()

    def _evaluate_draw(self) -> Optional[dict]:
        """问一遍全部判和条件 -> ``{'rule', 'reason'}`` 或 ``None``。

        优先用 ``jieqi.rules`` 的实现（纯函数、幂等）；万一模块不可用，
        退回下面 ``_fallback_draw`` 的最小实现，保证老逻辑不至于全丢。
        """
        if _rules is not None:
            try:
                return _rules.evaluate_draws(self, self._rule_facts_upto_ply())
            except Exception as exc:      # pragma: no cover - 防御
                print(f"[揭棋AI] 判和条件检查失败：{exc}")
        return self._fallback_draw()

    def _fallback_draw(self) -> Optional[dict]:
        """``jieqi.rules`` 不可用时的兜底（只保留重复局面与最大步数）。"""
        max_plies = int(getattr(self.config, 'DRAW_CONDITIONS', {}).get(
            'max_plies', 200) or 200)
        if self.total_plies >= max_plies:
            return {'rule': 'max_plies', 'reason': f'达到最大步数 {max_plies} 步判和'}
        count = len(self.position_counter.get(self._board_hash(), []))
        if count >= self._repetition_draw_at():
            return {'rule': 'repetition',
                    'reason': f'局面重复 {self._repetition_draw_at()} 次判和'}
        return None

    def _repetition_draw_at(self) -> int:
        """几次重复判和（读 ``jieqi.rules``，缺了退回配置）。"""
        if _rules is not None:
            return int(_rules.RULE['repetition_draw'])
        return int(getattr(self.config, 'DRAW_CONDITIONS', {}).get(
            'repetition_draw', 6) or 6)

    def _remind_repetition(self) -> None:
        """局面重复到提醒线时给一句提示（**只读**计数，不改任何状态）。"""
        count = len(self.position_counter.get(self._board_hash(), []))
        if _rules is not None:
            verdict = _rules.repetition_verdict(count)
        else:
            verdict = {'warn': count >= 4, 'draw': count >= self._repetition_draw_at(),
                       'message': f'局面已重复 {count} 次，请注意变招'}
        if not verdict.get('warn') or not verdict.get('message'):
            return
        if verdict['message'] == self._last_rule_toast:
            return
        self._last_rule_toast = verdict['message']
        self.set_toast(verdict['message'])

    def _auto_export_record(self) -> None:
        """终局时自动把棋谱存进 ``records/``（只存一次，失败只提示不崩）。

        由配置项 ``record.auto_save`` 控制（默认**开启**）。关掉之后终局不写盘，
        玩家仍可以随时按「导出棋谱」手动保存。

        ``unique=True``：终局保存**永远开新文件**，绝不覆盖任何已有棋谱
        （尤其是"导入一份棋谱接着下完"这种情况——那份导入的棋谱必须原样留着）。
        """
        if not self.auto_save_record():
            return
        if getattr(self, 'record_saved', ''):
            return
        if not self.move_history:
            return
        result = self.export_record(unique=True)
        if result.get('ok'):
            self.set_toast(f"棋谱已保存：{self.record_saved}", loud=True)
            print(f"[揭棋AI] 棋谱已保存 -> {result.get('path')}")
        else:
            self.set_toast(f"棋谱保存失败：{result.get('error')}", loud=True)
            print(f"[揭棋AI] 棋谱保存失败：{result.get('error')}")

    def auto_save_record(self) -> bool:
        """是否"对局结束自动保存棋谱"。

        优先级：

        1. **本次运行的开关** ``self.auto_save_override``（启动页上的「自动保存」
           开关，``None`` = 没动过）；
        2. 配置文件 ``ai_config.json`` 的 ``record.auto_save``。

        默认 **True**。配置缺失/读不动/类型不对时都退回默认值，
        绝不因为配置问题把自动保存搞没。
        """
        override = getattr(self, 'auto_save_override', None)
        if isinstance(override, bool):
            return override
        try:
            from jieqi.settings import load_raw_config

            raw = load_raw_config()
            section = raw.get("record")
            if isinstance(section, dict):
                value = section.get("auto_save")
                # 只有**显式**写成 false 才关闭；null / 缺字段 / 类型不对都按默认（开），
                # 宁可多存一份棋谱，也别因为配置写错就把棋谱弄丢了。
                if isinstance(value, bool):
                    return value
                if value is None:
                    return True
                if isinstance(value, str):
                    return value.strip().lower() not in ("false", "0", "no", "off")
                if isinstance(value, (int, float)):
                    return bool(value)
        except Exception:
            pass
        return True

    def set_toast(self, text: str, loud: bool = False,
                  seconds: float = 4.0) -> None:
        """设置提示文字。

        ``loud=True`` 表示这是**操作反馈**（导出/导入/保存失败…），渲染器会把它
        放大显示在棋盘上几秒——"点了按钮必须知道成没成功"。之后自动缩回顶栏
        那一行小字，不长期占地方。
        """
        self.toast = str(text or '')
        self.toast_loud = bool(loud and self.toast)
        self.toast_until = (time.monotonic() + float(seconds)) if self.toast else 0.0

    def _check_fold_repetition(self):
        """当前局面是否已经重复到判和线（**纯查询，绝不改计数**）。

        ⚠ 这里以前有 bug：每次调用都会往 ``position_counter`` 里 **append 一次**
        当前回合，而 ``check_game_state`` 每帧都被调用 —— 于是同一局面被反复
        计数，几帧之内就把"重复 6 次"凑满，正常对局会被误判和。

        现在 ``position_counter`` 的累积**只发生在真的走了一步时**
        （``move_piece`` 末尾），检查侧一律只读，所以本方法是幂等的。
        """
        count = len(self.position_counter.get(self._board_hash(), []))
        return count >= self._repetition_draw_at()

    def ai_check_fold_repetition(self):
        """三次重复局面检测（返回应该避免的那一步，没有则 None）。

        保留给旧调用方：只用公开签名计数，**不再**因为"快到了"就凭空多算一手
        （旧实现里的 ``+ [self.move_count]`` 会让检查本身改变结论）。
        """
        current_hash = self._board_hash()
        history = list(self.position_counter.get(current_hash, []))
        repeated = len(history) >= 3 and (history[-1] - history[-3]) <= 60
        if repeated and len(self.move_history) >= 2:
            (r, c), (sr, sc) = self.move_history[-2]['start'], self.move_history[-2]['end']
            return (r, c), (sr, sc)
        return None

    def _board_hash(self):
        """公开局面签名：暗子按**摆位身份**参与，不泄漏真实身份。

        旧实现把 ``piece['actual']`` 写进哈希，等于让重复局面判定也依赖暗子真值；
        这里改用引擎的公开信息签名（与 AI 判断三次重复用的是同一把尺子）。
        """
        if _adapter is not None:
            side = 0 if self.current_player == 'red' else 1
            try:
                return _adapter.public_signature(self.board, side)
            except Exception:
                pass
        parts = [f"turn={self.current_player}"]
        for r in range(10):
            for c in range(9):
                piece = self.board[r][c]
                if piece:
                    key = piece['actual'] if piece['revealed'] else f"暗{piece['original']}"
                    parts.append(f"{r}{c}{piece['color'][0]}{key}")
        return "|".join(parts)

    def _check_stalemate(self):
        """检查是否逼和"""
        if self.in_check:
            return False
        return not self._has_legal_moves()

    def _has_legal_moves(self):
        """检查当前玩家是否有合法移动"""
        for r in range(10):
            for c in range(9):
                piece = self.board[r][c]
                if piece and piece['color'] == self.current_player:
                    for nr in range(10):
                        for nc in range(9):
                            if self.validate_move(r, c, nr, nc):
                                return True
        return False
