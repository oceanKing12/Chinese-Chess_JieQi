# input_handler.py
import copy

import pygame
from typing import Optional, Tuple
try:
    from Game.game_state import GameState
except ImportError:  # pragma: no cover
    from game_state import GameState
from renderer import Renderer

#: 运行时切换难度：按 1/2/3/4
DIFFICULTY_KEYS = {
    pygame.K_1: 'easy',
    pygame.K_2: 'normal',
    pygame.K_3: 'hard',
    pygame.K_4: 'expert',
}
DIFFICULTY_LABELS = {'easy': '简单', 'normal': '普通', 'hard': '困难', 'expert': '专家'}

#: 人工打分：5~9 = 好棋/正着/疑问手/错着/漏着，0 = 清除。
#: 与上面的难度键（1~4）同族但不冲突；代码字符串必须与 ``jieqi/record.py``
#: 的 ``RATINGS`` 以及 ``renderer.SCORE_ITEMS`` 对齐。
SCORE_KEYS = {
    pygame.K_5: 'best',
    pygame.K_6: 'good',
    pygame.K_7: 'inaccuracy',
    pygame.K_8: 'mistake',
    pygame.K_9: 'blunder',
    pygame.K_0: '',              # 清除
}


class InputHandler:
    def __init__(self, game_state: GameState, renderer: Renderer):
        self.gs = game_state
        self.renderer = renderer
        self.config = self.gs.config
        #转回人工
        self.change_to_myself=False
        #: 由 main.ChessGame 注入，用来派发界面命令（切难度 / 切大模型 / 回退）
        self.game = None

    # ------------------------------------------------------------------
    def _command(self, cmd: str) -> None:
        game = getattr(self, 'game', None)
        if game is not None and hasattr(game, 'handle_command'):
            game.handle_command(cmd)

    def _scroll_move_log(self, pos, button: int) -> bool:
        """滚轮：指针在走棋记录上就翻记录，在点评卡上就翻点评。

        返回是否消费掉了这次滚轮事件（只在指针真的落在可滚区域时消费，
        否则滚轮不该有任何效果）。每格翻 2 行，滚到头不做处理也不报错。
        """
        renderer = self.renderer
        if not hasattr(renderer, 'scroll_move_log'):
            return False
        local = (pos[0] - renderer.panel_rect.x, pos[1] - renderer.panel_rect.y)
        lay = renderer._panel_layout()
        delta = 2 if button == 4 else -2      # 4 = 向上滚，5 = 向下滚
        comment = lay.get('comment')
        if comment is not None and comment.collidepoint(local):
            renderer.scroll_comment(delta)
            return True
        if lay['log'].collidepoint(local):
            renderer.scroll_move_log(delta)
            return True
        return False

    def handle_events(self) -> bool:
        """处理所有输入事件，返回是否继续运行"""
        self.renderer.mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEMOTION:
                self.renderer.mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 滚轮：在右侧面板的走棋记录上滚动翻看历史。
                # （pygame 把滚轮也报成 MOUSEBUTTONDOWN 的 4/5 号键，所以必须
                #   先拦下来，否则会被当成"点击棋盘"或"重开一局"。）
                if event.button in (4, 5):
                    self._scroll_move_log(event.pos, event.button)
                    continue
                if getattr(self.renderer, 'overlay', ''):
                    # 浮层打开时点击只作用于浮层按钮（点空白不落子）
                    cmd = self.renderer.button_at(event.pos)
                    if cmd:
                        self._command(cmd)
                elif self.gs.show_side_selection:
                    self._handle_start_screen_click(event.pos)
                else:
                    # 按钮优先：终局后"导出棋谱/上一步"这些也必须点得动，
                    # 不能让"点击任意位置重新开始"把它们全吃掉。
                    cmd = self.renderer.button_at(event.pos)
                    if cmd:
                        self._command(cmd)
                    elif self.gs.game_over:
                        self.gs.reset()
                    else:
                        self._handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
        return True

    def _handle_click(self, pos: Tuple[int, int]):
        """先看是不是点在按钮上，否则当成落子。"""
        cmd = self.renderer.button_at(pos)
        if cmd:
            self._command(cmd)
            return
        self._handle_game_click(pos)

    def _handle_start_screen_click(self, pos: Tuple[int, int]):
        """启动页点击：一律走命令分发（与棋盘页同一套机制）。

        为什么不再自己判断"点到哪个阵营"：启动页现在有难度 chip、大模型开关、
        自动保存、继续对局、载入棋谱等一大堆控件，几何只有 ``start_screen``
        一处知道；在这里复写一遍命中逻辑，早晚会与画面不一致（"画在这里、
        点在那里"）。
        """
        cmd = self.renderer.button_at(pos)
        if cmd:
            self._command(cmd)

    def _handle_game_click(self, pos: Tuple[int, int]):
        """处理棋盘点击"""
        # 轮到"非人工"那一方时禁止玩家代走（旧版没这个判断，玩家可以替 AI 走子）。
        # 用 ``is_human_side`` 而不是 ``current_player == ai_color``：现在
        # **两方可以各配不同引擎**，没有"唯一的 AI 方"这个说法了。
        if not self.gs.is_human_side(self.gs.current_player):
            self.gs.toast = '这一方由引擎出招，请稍候…'
            return
        logic_pos = self._pos_to_logic(pos)
        if not logic_pos:
            return

        r, c = logic_pos
        if self.gs.selected:
            self._try_move_piece(r, c)
        else:
            self._select_piece(r, c)

    def _pos_to_logic(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """屏幕坐标 -> 棋盘逻辑坐标（要先减去棋盘在窗口里的偏移）。"""
        board = self.renderer.board_rect
        if not board.collidepoint(pos):
            return None
        x = pos[0] - board.x
        y = pos[1] - board.y
        # 计算有效点击区域
        if not (self.config.OFF_SET < x < self.config.WIDTH - self.config.OFF_SET and
                self.config.OFF_SET < y < self.config.HEIGHT - self.config.OFF_SET):
            return None

        display_c = (x - self.config.OFF_SET) // self.config.CELL_SIZE
        display_r = (y - self.config.OFF_SET) // self.config.CELL_SIZE

        if self.gs.player_side == 'red':
            logic_r = 9 - display_r
            logic_c = 8 - display_c
        else:
            logic_r = display_r
            logic_c = display_c

        if 0 <= logic_r < 10 and 0 <= logic_c < 9:
            return (logic_r, logic_c)
        return None

    def _select_piece(self, r: int, c: int):
        """选择棋子"""
        piece = self.gs.board[r][c]
        if piece and piece['color'] == self.gs.current_player:
            self.gs.selected = (r, c)
            self.gs.toast = ''
        else:
            self.gs.toast = '不能选择敌方棋子' if piece else ''

    def _try_move_piece(self, end_r: int, end_c: int):
        """尝试移动棋子"""
        start_r, start_c = self.gs.selected
        piece = self.gs.board[start_r][start_c]
        target = self.gs.board[end_r][end_c]

        if not piece or piece['color'] != self.gs.current_player:
            self.gs.selected = None
            return
        if target and target['color'] == piece['color']:
            self.gs.selected = end_r, end_c
            return

        if not self.gs.validate_move(start_r, start_c, end_r, end_c):
            self.gs.toast = '这一步不合走法'
            self.gs.selected = None
            return

        temp_board = copy.deepcopy(self.gs.board)
        temp_board[end_r][end_c] = copy.deepcopy(piece)
        temp_board[start_r][start_c] = None
        temp_red, temp_black = self.gs._get_king_positions(temp_board)
        if self.gs.in_check and self.gs._check_check(temp_board, self.gs.current_player,
                                                     temp_red, temp_black):
            self.gs.toast = f'{self.gs.current_player}需要应将'
            self.gs.selected = None
            return
        if self.gs._check_check(temp_board, self.gs.current_player, temp_red, temp_black):
            self.gs.toast = f'{self.gs.current_player}不可送将'
            self.gs.selected = None
            return
        if self.gs._check_face_to_face(temp_board, temp_red, temp_black):
            self.gs.toast = '移动后导致对面笑，行动失败'
            self.gs.selected = None
            return

        self.gs.selected = None
        self.gs.move_piece(start_r, start_c, end_r, end_c)
        self.gs.current_player = 'black' if self.gs.current_player == 'red' else 'red'
        self.gs.check_game_state()

    def _handle_keydown(self, event):
        """键盘：

        * ←/→：上一回合 / 下一回合；Home/End：开局 / 最新
        * 1~4：切换 AI 难度
        * L：开关大模型点评；K：开关「应用大模型招法」
        * S：导出棋谱；I：导入棋谱；V：回顾棋局浮层
        * M：回到启动页（改设置 / 换阵营 / 继续或载入棋谱）
        * F2：人机 与 双人对下 之间切换（原来占的是 ESC，见下）

        ESC 现在**回启动页**，不再兼任"人机/双人切换"：一个键干两件事
        （还都是改对局模式）太容易误触，玩家按 ESC 通常想的只是"退出去"。
        开局前在启动页上就能选执红/执黑，"双人对下"因此也不再是必需品。
        """
        # 浮层打开时：ESC 关浮层，←→/Home/End 仍然翻手，其余按键交给浮层本身
        if getattr(self.renderer, 'overlay', ''):
            if event.key == pygame.K_ESCAPE:
                self._command('view:close')
                return
            if event.key == pygame.K_LEFT:
                self._command('nav:prev')
                return
            if event.key == pygame.K_RIGHT:
                self._command('nav:next')
                return
            if event.key == pygame.K_HOME:
                self._command('nav:first')
                return
            if event.key == pygame.K_END:
                self._command('nav:last')
                return
            return
        if event.key == pygame.K_LEFT:
            self._command('nav:prev')
        elif event.key == pygame.K_RIGHT:
            self._command('nav:next')
        elif event.key == pygame.K_HOME:
            self._command('nav:first')
        elif event.key == pygame.K_END:
            self._command('nav:last')
        elif event.key in DIFFICULTY_KEYS:
            self._command('difficulty:' + DIFFICULTY_KEYS[event.key])
        elif event.key == pygame.K_l:      # 大写 L 在 pygame 里同样是 K_l + SHIFT
            self._command('llm:toggle')
        elif event.key == pygame.K_k:
            self._command('llm:apply')
        elif event.key == pygame.K_e:
            # E = engine：循环切换**红方**的对局方式（人工/本机/外部/大模型）
            self._command('engine:cycle:red')
        elif event.key == pygame.K_t:
            # T = 切换**黑方**的对局方式（T 紧挨着 E，两方各一个键好记）
            self._command('engine:cycle:black')
        elif event.key == pygame.K_c:
            # C = 换边（我执红 <-> 我执黑）。启动页上没有阵营选项了，
            # 换边就靠这个键或 ai_config.json 里的 ai.player_side。
            self._command('side:flip')
        elif event.key == pygame.K_s:
            self._command('record:export')
        elif event.key == pygame.K_i:
            self._command('record:import')
        elif event.key == pygame.K_v:
            self._command('view:board')
        elif event.key == pygame.K_m:      # M = menu，回启动页
            self._command('game:menu')
        elif event.key in SCORE_KEYS:
            # 人工打分：5~9 = 五档，0 = 清除（与 1~4 的难度键同族但不冲突）
            self._command('score:' + SCORE_KEYS[event.key])
        elif event.key == pygame.K_F2:
            self.change_to_myself = not self.change_to_myself
            if self.change_to_myself:
                self.gs._init_ai()
                self.gs.toast = '已切换为双人对下（F2 恢复人机）'
            else:
                self.gs.enable_ai('black' if self.gs.player_side == 'red' else 'red')
                self.gs.toast = '已恢复人机对战'
        elif event.key == pygame.K_ESCAPE:
            self._command('game:menu')
