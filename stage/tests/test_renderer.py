"""渲染层与三区布局测试（无显示器）。

覆盖三类问题：
1. **结算崩溃回归**：旧代码在 `draw_game_over()` 里调用
   `pygame.font.SysFont('simhei', 72)`，pygame 2.6.1 在 Windows 上枚举注册表
   字体会抛 `TypeError`，导致"走完决定胜负的一步后立刻崩溃"。
2. **五区布局**：顶部配置区、棋盘上下的「被吃棋子」展示条、棋盘区、右侧信息面板
   （回合·评估、走棋记录、回退、点评）以及它们的点击命中测试。
3. **被吃棋子展示条**：明子照真身显示，被吃的暗子只能用暗子图标。
"""
import os
import random
import sys
import unittest
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(GAME_DIR.parent))
sys.path.insert(0, str(GAME_DIR))

import pygame  # noqa: E402

from resource_manager import ResourceManager, load_font  # noqa: E402

_ORIGINAL_CWD = os.getcwd()


def setUpModule():
    """必须先把 pygame 全模块（含 font）初始化好。

    类是按名字母序执行的，TestBoardClickMapping 会排在最前面，
    那时还没人调用过 pygame.init()，load_font 会抛 "font not initialized"。
    """
    pygame.init()
    pygame.display.set_mode((1, 1))


def build():
    from game_state import GameState
    from renderer import Renderer

    if not pygame.font.get_init():
        pygame.font.init()
    rm = ResourceManager()
    rm.load_resources()
    gs = GameState(rm)
    gs.show_side_selection = False
    gs.enable_ai('black')
    return gs, Renderer(rm, gs)


class TestFontLoading(unittest.TestCase):
    def test_load_font_never_uses_registry(self):
        """多种字号都能拿到字体，且走缓存；不依赖 SysFont/match_font。"""
        pygame.init()
        pygame.display.set_mode((1, 1))
        for size in (15, 16, 18, 20, 26, 36, 72):
            font = load_font(size)
            self.assertIsNotNone(font)
            self.assertIs(load_font(size), font, "字体没有走缓存")
        surface = load_font(26).render("揭棋测试", True, (0, 0, 0))
        self.assertGreater(surface.get_width(), 0)


class TestLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        self.gs, self.renderer = build()

    def test_regions_do_not_overlap(self):
        """顶栏 / 两条被吃子展示条 / 棋盘 / 右侧面板必须互不重叠。"""
        cfg = self.renderer.config
        self.assertEqual((cfg.WINDOW_W, cfg.WINDOW_H),
                         (cfg.WIDTH + cfg.PANEL_W,
                          cfg.BOARD_Y + cfg.HEIGHT + cfg.BOARD_STRIP_H))
        regions = {
            'top': self.renderer.top_rect,
            'strip_top': self.renderer.capture_top_rect,
            'board': self.renderer.board_rect,
            'strip_bottom': self.renderer.capture_bottom_rect,
            'panel': self.renderer.panel_rect,
        }
        for name_a, rect_a in regions.items():
            for name_b, rect_b in regions.items():
                if name_a < name_b:
                    self.assertFalse(rect_a.colliderect(rect_b),
                                     f"{name_a} 与 {name_b} 重叠：{rect_a} {rect_b}")
        self.assertEqual(self.renderer.window.get_size(), (cfg.WINDOW_W, cfg.WINDOW_H))

    def test_window_is_fully_covered(self):
        """五个区域加起来必须铺满窗口，不能留下没画到的缝。"""
        cfg = self.renderer.config
        r = self.renderer
        self.assertEqual(r.top_rect.width, cfg.WINDOW_W)
        self.assertEqual(r.top_rect.bottom, r.capture_top_rect.top)
        self.assertEqual(r.capture_top_rect.bottom, r.board_rect.top)
        self.assertEqual(r.board_rect.bottom, r.capture_bottom_rect.top)
        self.assertEqual(r.capture_bottom_rect.bottom, cfg.WINDOW_H)
        self.assertEqual(r.panel_rect.top, cfg.TOP_BAR_H)
        self.assertEqual(r.panel_rect.bottom, cfg.WINDOW_H)
        self.assertEqual(r.panel_rect.right, cfg.WINDOW_W)
        self.assertEqual(r.capture_top_rect.width, cfg.WIDTH)
        self.assertEqual(r.capture_bottom_rect.width, cfg.WIDTH)

    def test_draw_all_renders_three_regions(self):
        self.gs.move_count = 7
        self.gs.toast = '测试提示'
        self.gs.ai_eval_text = 'AI 评估：+12（均势）  深度 4'
        self.gs.ai_comment = '建议挺兵，控制中路。'
        self.renderer.draw_all()      # 不能抛异常

    def test_button_hit_testing(self):
        """棋盘页的按钮 + 启动页的按钮都要能被命中（两套界面各测一遍）。"""
        r = self.renderer
        for getter in (r.record_button, r.import_button, r.replay_button,
                       r.restart_button):
            cmd, rect, _text, _active = getter()
            self.assertEqual(r.button_at(rect.center), cmd)
        for cmd, rect, _text, _active in r.nav_buttons():
            self.assertEqual(r.button_at(rect.center), cmd)
        # 棋盘上用不着命中按钮
        self.assertIsNone(r.button_at(r.board_rect.center))
        # 启动页：难度 chip / 每方对局方式 chip / 阵营 / 载入棋谱
        self.gs.show_side_selection = True
        try:
            layout = r.start_layout()
            cases = list(r.difficulty_buttons())
            cases.extend(r.engine_buttons())
            cases.extend(r.start_screen_buttons())
            for cmd, rect, _text, _active in cases:
                if layout.disabled(cmd):
                    continue        # 灰掉的按钮不参与命中（点击无效是设计如此）
                self.assertEqual(r.button_at(rect.center), cmd,
                                 f"{cmd} 在启动页中心点命不中")
        finally:
            self.gs.show_side_selection = False

    def test_buttons_inside_their_regions(self):
        """棋盘页按钮在面板/顶栏里；启动页按钮在整个窗口里。"""
        r = self.renderer
        for getter in (r.record_button, r.import_button, r.replay_button,
                       r.restart_button):
            _cmd, rect, _text, _active = getter()
            self.assertTrue(r.panel_rect.contains(rect), rect)
        for _cmd, rect, _text, _active in r.nav_buttons():
            self.assertTrue(r.top_rect.contains(rect), rect)
        self.gs.show_side_selection = True
        try:
            window = pygame.Rect(0, 0, r.config.WINDOW_W, r.config.WINDOW_H)
            for _cmd, rect, _text, _active in r.engine_buttons():
                self.assertTrue(window.contains(rect), rect)
            for _cmd, rect, _text, _active in r.difficulty_buttons():
                self.assertTrue(window.contains(rect), rect)
        finally:
            self.gs.show_side_selection = False

    def test_move_log_shows_reveal(self):
        """走棋记录必须显示翻开的子是什么。"""
        from jieqi.gui_adapter import move_record_text

        rec = {'piece': {'color': 'red', 'original': '炮', 'revealed': False},
               'start': (2, 1), 'end': (4, 1), 'captured': None,
               'revealed_as': '车'}
        text = move_record_text(rec, 3)
        self.assertIn('暗炮', text)
        self.assertIn('→车', text)
        # 坐标是 ``b7b5`` 这套（列字母 + 行号 0~9）
        self.assertIn('b7b5', text)

        rec2 = {'piece': {'color': 'black', 'original': '卒', 'revealed': True,
                          'actual': '馬'},
                'start': (6, 0), 'end': (5, 0),
                'captured': {'color': 'red', 'revealed': True, 'actual': '兵'},
                'revealed_as': None}
        text2 = move_record_text(rec2, 4)
        self.assertIn('黑馬', text2)
        self.assertIn('吃红兵', text2)
        self.assertNotIn('→', text2)


class TestCapturedStrips(unittest.TestCase):
    """棋盘上下的「被吃棋子」展示条。

    规则要点（也是这几条测试要锁住的东西）：

    * 明子被吃 -> 照真身显示；被吃的**暗子身份未知**，只能用"暗子"图标，
      绝不能把真身画出来（否则等于偷看暗子，AI 与人类信息就不对称了）；
    * 展示条只跟着**当前显示的手数**走：浏览历史时不能把后面几步的吃子也算进来。
    """

    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        self.gs, self.renderer = build()

    def _play(self, plies=40, seed=5):
        import random

        from jieqi.gui_adapter import gui_color_to_side, legal_moves

        rng = random.Random(seed)
        for _ in range(plies):
            side = gui_color_to_side(self.gs.current_player)
            moves = legal_moves(self.gs.board, side)
            if not moves:
                break
            capturing = [mv for mv in moves
                         if self.gs.board[(mv & 127) // 9][(mv & 127) % 9]]
            mv = rng.choice(capturing or moves)
            self.gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                               (mv & 127) // 9, (mv & 127) % 9)
            self.gs.current_player = 'black' if side == 0 else 'red'
        self.gs.animations.clear()

    def test_strips_render_with_captures(self):
        self._play()
        captured = self.renderer._captured_lists()
        self.assertTrue(captured['red'] or captured['black'],
                        "走了 40 手却没有一次吃子，测试样本不成立")
        self.renderer.draw_all()          # 不能抛异常
        for item in captured['red'] + captured['black']:
            self.assertIn('unknown', item)
            if item['unknown']:
                self.assertIsNone(item['piece'], "被吃的暗子不许带真身")

    def test_unknown_capture_uses_dark_icon(self):
        """被吃的暗子必须用暗子图标，明子用真身图标。"""
        dark = self.renderer._captured_image({'piece': None, 'unknown': True})
        self.assertIsNotNone(dark)
        real = {'piece': {'color': 'red', 'actual': '车', 'revealed': True},
                'unknown': False}
        image = self.renderer._captured_image(real)
        self.assertIsNotNone(image)
        dark_black = self.renderer._captured_image(
            {'piece': {'color': 'black', 'actual': '卒', 'revealed': True},
             'unknown': True})
        self.assertIsNotNone(dark_black)
        # 暗子图标与真身图标必须是不同的图（否则等于把暗子亮出来了）
        self.assertIsNot(dark, self.renderer._captured_image(
            {'piece': {'color': 'red', 'actual': '车', 'revealed': True},
             'unknown': False}))

    def test_captures_follow_displayed_ply(self):
        """浏览到历史某一手时，展示条里不能出现那之后才被吃的子。"""
        self._play(plies=30)
        from jieqi.gui_adapter import captured_gui_pieces

        total = len(captured_gui_pieces(self.gs, 'red'))
        if total == 0:
            self.skipTest("这段随机对局里没有红方子被吃")
        self.gs.goto_ply(0)
        self.assertEqual(captured_gui_pieces(self.gs, 'red'), [],
                         "回到开局却还有被吃的子")
        self.renderer.draw_all()
        self.gs.goto_ply(self.gs.total_plies)
        self.renderer.draw_all()

    def test_strip_capacity_and_overflow(self):
        """同种棋子归并之后，一行排不下也不能画到条外面去，更不能撑破窗口。"""
        cfg = self.renderer.config
        fake = [{'piece': {'color': 'red', 'actual': '兵', 'revealed': True},
                 'unknown': False}] * 40
        self.renderer._draw_one_capture_strip(self.renderer.capture_top_rect,
                                              fake, "对方被吃",
                                              label_side='left')
        for rect in (self.renderer.capture_top_rect,
                     self.renderer.capture_bottom_rect):
            self.assertGreaterEqual(rect.top, cfg.TOP_BAR_H)
            self.assertLessEqual(rect.bottom, cfg.WINDOW_H)
            self.assertLessEqual(rect.right, cfg.WIDTH)

    def test_capture_groups_merge_same_pieces(self):
        """同种棋子必须合并成一组（卒×3 -> 一个图标带角标 3）。"""
        items = ([{'piece': {'color': 'black', 'actual': '卒', 'revealed': True},
                   'unknown': False}] * 3
                 + [{'piece': {'color': 'black', 'actual': '卒', 'revealed': True},
                     'unknown': False}]
                 + [{'piece': {'color': 'red', 'actual': '车', 'revealed': True},
                     'unknown': False}]
                 + [{'piece': None, 'unknown': True}] * 2)
        groups = self.renderer._capture_groups(items)
        by_key = {(key[0], key[1]): count for key, _sample, count in groups}
        self.assertEqual(by_key[('black', '卒')], 4)
        self.assertEqual(by_key[('red', '车')], 1)
        self.assertEqual(sum(count for _k, _s, count in groups
                             if _k[1] == '暗'), 2)
        # 大子排在暗子前面（暗子身份未知，不该抢眼）
        idents = [key[1] for key, _s, _c in groups]
        self.assertLess(idents.index('车'), len(idents) - 1)
        self.assertEqual(idents[-1], '暗')

    def test_capture_groups_keep_red_and_black_apart(self):
        """红黑同兵种不能合并（一个是车一个是車，颜色也不同）。"""
        items = [{'piece': {'color': 'red', 'actual': '车', 'revealed': True},
                  'unknown': False},
                 {'piece': {'color': 'black', 'actual': '車', 'revealed': True},
                  'unknown': False}]
        groups = self.renderer._capture_groups(items)
        self.assertEqual(len(groups), 2)
        self.assertEqual({(k[0], k[1]) for k, _s, _c in groups},
                         {('red', '车'), ('black', '車')})

    def test_count_badge_only_when_more_than_one(self):
        """角标只在数量 > 1 时画（单枚棋子不画"1"，免得噪音）。"""
        captured = []
        calls = []
        original = self.renderer._draw_count_badge

        def spy(right_x, top_y, count):
            calls.append(count)
            return original(right_x, top_y, count)

        self.renderer._draw_count_badge = spy
        try:
            rect = self.renderer.capture_top_rect
            self.renderer._draw_one_capture_strip(
                rect, [{'piece': {'color': 'red', 'actual': '车',
                                  'revealed': True}, 'unknown': False}],
                "对方被吃", label_side='left')
            self.assertEqual(calls, [], "单枚棋子不该画角标")
            self.renderer._draw_one_capture_strip(
                rect, [{'piece': {'color': 'red', 'actual': '车',
                                  'revealed': True}, 'unknown': False}] * 3,
                "对方被吃", label_side='left')
            self.assertEqual(calls, [3], "三枚同种棋子应当画角标 3")
        finally:
            self.renderer._draw_count_badge = original

    def test_move_log_scrolls_through_history(self):
        """走棋记录放不下时要能滚动翻看，且不会翻过开局。"""
        r = self.renderer
        lay = r._panel_layout()
        per_page = max(1, lay['log'].height // r._log_line_height())
        # ⚠ 随机对局随时可能终局（几十手就结束），所以这里**逐个种子试**，
        # 直到找到一个能走够两屏的；写死手数的话规则一变就会变成
        # "这个测试没意义"（真实踩过）。
        total = 0
        for seed in range(1, 40):
            self.gs.reset(keep_engines=True)
            self._play(plies=per_page * 2 + 6, seed=seed)
            total = self.gs.total_plies
            if total > per_page:
                break
        lay = r._panel_layout()
        per_page = max(1, lay['log'].height // r._log_line_height())
        self.assertGreater(r._max_log_offset(lay['log']), 0,
                           f"{total} 手应当超过一屏（一屏 {per_page} 行），"
                           "否则这个测试没意义")

        # 默认贴住最新一手（记录最后一行就是第 total 手）
        r._clamp_log_scroll(resync=True)
        self.assertEqual(r.log_scroll, 0)
        newest = [t for t, _c, _cur, _ai in r._move_lines(per_page, offset=0)]
        self.assertTrue(newest and newest[-1].strip().startswith(f"{total}."),
                        newest[-1] if newest else "(空)")

        # 往上滚 -> 看到更早的记录
        r.scroll_move_log(+5)
        self.assertEqual(r.log_scroll, 5)
        older = [t for t, _c, _cur, _ai in r._move_lines(per_page, offset=5)]
        self.assertNotEqual(newest, older)

        # 一路滚到头：不许翻过开局（偏移被夹到上限）
        for _ in range(50):
            r.scroll_move_log(+10)
        self.assertEqual(r.log_scroll, r._max_log_offset(lay['log']))
        top = r._move_lines(per_page, offset=r.log_scroll)
        self.assertTrue(top[0][0].strip().startswith("1."), top[0][0])

        # 往回滚到底：回到最新一手
        for _ in range(50):
            r.scroll_move_log(-10)
        self.assertEqual(r.log_scroll, 0)

    def test_move_log_scroll_resyncs_when_ply_changes(self):
        """切换步数/落子之后滚动位置要回到贴住当前手（否则记录会错位）。"""
        self._play(plies=300)
        r = self.renderer
        r.draw_all()
        self.assertGreater(r._max_log_offset(r._panel_layout()['log']), 8,
                           "记录不够长，滚不动就测不出 resync")
        r.scroll_move_log(+8)
        self.assertEqual(r.log_scroll, 8)
        self.gs.goto_ply(max(0, self.gs.total_plies - 1))
        r.draw_all()                   # 状态变了 -> 应当自动 resync
        self.assertEqual(r.log_scroll, 0, "切换步数后没有滚回当前手")

    def test_move_log_scroll_ignores_short_games(self):
        """记录不足一屏时不该出现滚动、也不该报错。"""
        self._play(plies=3)
        r = self.renderer
        lay = r._panel_layout()
        self.assertEqual(r._max_log_offset(lay['log']), 0)
        r.scroll_move_log(+10)
        self.assertEqual(r.log_scroll, 0)
        r.draw_all()

    def test_all_screens_render_with_strips(self):
        self._play(plies=20)
        for winner in ('red', 'black', 'draw'):
            self.gs.game_over = True
            self.gs.winner = winner
            self.gs.draw_reason = '测试' if winner == 'draw' else None
            self.renderer.draw_all()
        self.gs.game_over = False
        self.gs.winner = None
        self.gs.draw_reason = None


class TestBoardClickMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        from input_handler import InputHandler

        self.gs, self.renderer = build()
        self.handler = InputHandler(self.gs, self.renderer)

    def _screen_pos(self, r, c):
        cfg = self.renderer.config
        if self.gs.player_side == 'red':
            dr, dc = 9 - r, 8 - c
        else:
            dr, dc = r, c
        return (cfg.BOARD_X + cfg.OFF_SET + dc * cfg.CELL_SIZE + cfg.CELL_SIZE // 2,
                cfg.BOARD_Y + cfg.OFF_SET + dr * cfg.CELL_SIZE + cfg.CELL_SIZE // 2)

    def test_click_maps_back_to_square_with_offset(self):
        """加上棋盘偏移后，屏幕坐标必须能正确映射回逻辑坐标。"""
        for (r, c) in ((0, 0), (0, 4), (4, 4), (9, 8), (6, 2)):
            self.assertEqual(self.handler._pos_to_logic(self._screen_pos(r, c)),
                             (r, c), f"({r},{c}) 映射错误")

    def test_clicks_outside_board_are_ignored(self):
        cfg = self.renderer.config
        for pos in ((5, 5), (cfg.WINDOW_W - 5, cfg.WINDOW_H - 5),
                    (cfg.WIDTH + 10, cfg.TOP_BAR_H + 10),
                    (10, cfg.TOP_BAR_H + 10)):
            self.assertIsNone(self.handler._pos_to_logic(pos), pos)


class TestGameOverAndPanels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        self.gs, self.renderer = build()

    def test_game_over_all_outcomes(self):
        for winner in ('red', 'black', 'draw'):
            self.gs.game_over = True
            self.gs.winner = winner
            self.renderer.draw_all()
            self.renderer.draw_game_over()

    def test_game_over_with_comment(self):
        self.gs.game_over = True
        self.gs.winner = 'red'
        self.gs.ai_comment = '黑方 1 步杀，红方已无解。'
        self.renderer.draw_all()

    def test_long_comment_wraps_inside_panel(self):
        small = load_font(self.renderer.config.FONT_SMALL)
        for comment in ('', '均势。', '建议走炮二平五，控制中路，为中局争取先手。' * 20):
            self.gs.ai_comment = comment
            self.renderer.draw_all()
            if comment:
                from renderer import _wrap
                lines = _wrap(small, comment, self.renderer.panel_rect.width - 24)
                self.assertTrue(lines)
                for line in lines:
                    self.assertLessEqual(small.size(line)[0],
                                         self.renderer.panel_rect.width)
                self.assertLessEqual(self.renderer._comment_height(),
                                     self.renderer.panel_rect.height)

    def test_start_screen_buttons_within_window(self):
        """启动页的"开始游戏"按钮要在窗口里，而且不能和其他按钮重叠。

        阵营选中按钮已按用户要求去掉（"去掉我执红或者我执黑选项，太啰嗦了"），
        所以这里改成检查**开始按钮**的几何。
        """
        self.gs.show_side_selection = True
        self.renderer.draw_all()
        cfg = self.renderer.config
        window = pygame.Rect(0, 0, cfg.WINDOW_W, cfg.WINDOW_H)
        layout = self.renderer.start_layout()
        start = layout.start_button
        self.assertTrue(window.contains(start), start)
        for cmd, rect, _text, _active in self.renderer.start_screen_buttons():
            if cmd == 'game:start':
                continue
            self.assertFalse(rect.colliderect(start), cmd)

    def test_animations_render(self):
        from jieqi.gui_adapter import legal_moves, gui_color_to_side

        moves = legal_moves(self.gs.board, gui_color_to_side('red'))
        mv = moves[0]
        self.gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                           (mv & 127) // 9, (mv & 127) % 9)
        self.renderer.draw_all()


class _StubGame:
    """最小替身：只实现 ``handle_command``，把导航命令接到 ``gs`` 上。

    （真实的命令分发在 ``main.ChessGame.handle_command``，这里不想为了几个点击
      就把整个 ChessGame 拉起来；替身只覆盖测试用到的这几条命令。）
    """

    def __init__(self, gs):
        self.gs = gs
        self.seen = []

    def handle_command(self, cmd):
        self.seen.append(cmd)
        gs = self.gs
        if cmd == 'nav:first':
            gs.goto_ply(0)
        elif cmd == 'nav:prev':
            gs.goto_ply(gs.ply - 1)
        elif cmd == 'nav:next':
            gs.goto_ply(gs.ply + 1)
        elif cmd == 'nav:last':
            gs.goto_ply(gs.total_plies)
        elif cmd == 'view:close':
            pass


class TestCommentScrolling(unittest.TestCase):
    """长点评：不截断，改成可滚动（真实反馈："ai点评太长被截断了"）。"""

    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        self.gs, self.renderer = build()

    def test_long_comment_scrolls_instead_of_being_cut(self):
        r = self.renderer
        r.draw_all()                     # 没有点评时卡片高度为 0
        self.assertEqual(r._comment_height(), 0)
        self.gs.ai_comment = ("红方稍优。" + "这是一段故意写得很长的点评，"
                              "用来验证点评卡会变成可滚动的。" * 30)
        r.draw_all()
        all_lines = r._wrapped_comment()
        lines, offset, max_offset = r._comment_view_lines()
        self.assertGreater(len(all_lines), len(lines),
                           "点评没有超出卡片，这个测试没意义")
        self.assertGreater(max_offset, 0)
        self.assertEqual(offset, 0)
        # 卡片高度有上限，不能把走棋记录挤没
        cap = int(r.panel_rect.height * 0.45)
        self.assertLessEqual(r._comment_height(), cap + 1)
        # 滚动之后能读到后面的内容，且滚不过尾
        r.scroll_comment(3)
        later, offset2, _ = r._comment_view_lines()
        self.assertEqual(offset2, 3)
        self.assertNotEqual(lines, later)
        for _ in range(50):
            r.scroll_comment(10)
        _l, offset3, max3 = r._comment_view_lines()
        self.assertEqual(offset3, max3)
        for _ in range(50):
            r.scroll_comment(-10)
        self.assertEqual(r._comment_view_lines()[1], 0)
        r.draw_all()

    def test_short_comment_never_scrolls(self):
        r = self.renderer
        self.gs.ai_comment = "一句很短的点评。"
        r.draw_all()
        _l, offset, max_offset = r._comment_view_lines()
        self.assertEqual(max_offset, 0)
        self.assertEqual(offset, 0)
        r.scroll_comment(5)
        self.assertEqual(r._comment_view_lines()[1], 0)


class TestInputEvents(unittest.TestCase):
    """鼠标/键盘事件的分发：滚轮翻记录、终局后按钮仍可点。"""

    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        from input_handler import InputHandler

        self.gs, self.renderer = build()
        self.handler = InputHandler(self.gs, self.renderer)
        self.game = _StubGame(self.gs)
        self.handler.game = self.game
        # 走够多手，让走棋记录超过一屏（否则滚轮测试没意义）
        rng = random.Random(3)
        from jieqi.gui_adapter import gui_color_to_side, legal_moves

        for _ in range(40):
            side = gui_color_to_side(self.gs.current_player)
            moves = legal_moves(self.gs.board, side)
            if not moves:
                break
            mv = rng.choice(moves)
            self.gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                               (mv & 127) // 9, (mv & 127) % 9)
            self.gs.current_player = 'black' if side == 0 else 'red'
        self.renderer.draw_all()

    def _post_wheel(self, pos, button):
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": button}))
        self.handler.handle_events()

    def _post_click(self, pos):
        """左键点一下（走 ``_handle_click``，所以按钮能命中）。"""
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": 1}))
        self.handler.handle_events()

    def _log_center(self):
        lay = self.renderer._panel_layout()
        panel = self.renderer.panel_rect
        return (panel.x + lay['log'].centerx, panel.y + lay['log'].centery)

    def _post_key(self, key):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": key,
                                                              "mod": 0,
                                                              "unicode": ""}))
        self.handler.handle_events()

    def test_score_keys_dispatch_five_levels_and_clear(self):
        """键盘 5~9 打分、0 清除；命令必须与 record.py 的档位代码对齐。"""
        for key, code in ((pygame.K_5, "best"), (pygame.K_6, "good"),
                          (pygame.K_7, "inaccuracy"), (pygame.K_8, "mistake"),
                          (pygame.K_9, "blunder"), (pygame.K_0, "")):
            self.game.seen.clear()
            self._post_key(key)
            self.assertIn("score:" + code, self.game.seen,
                          "按键 %s 没有派发 score:%s（实际 %s）"
                          % (key, code, self.game.seen))

    def test_branch_keys_are_gone(self):
        """分支棋谱已整体移除：``R``/``Q``/``[``/``]`` 都不该再派发任何命令。

        留着"按了有反应但那个功能已经没了"的键位比直接不管更糟 ——
        玩家会以为功能还在，只是坏了。
        """
        for key in (pygame.K_r, pygame.K_q,
                    pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            with self.subTest(key=key):
                self.game.seen.clear()
                self._post_key(key)
                self.assertEqual([c for c in self.game.seen
                                  if c.startswith('branch:')
                                  or c == 'analysis:toggle'], [],
                                 "按键 %s 不该再派发分支命令（实际 %s）"
                                 % (key, self.game.seen))

    def test_score_keys_do_not_clash_with_difficulty(self):
        """1~4 仍然是难度切换——打分键位不能把它们抢走。"""
        for key, name in ((pygame.K_1, "easy"), (pygame.K_2, "normal"),
                          (pygame.K_3, "hard"), (pygame.K_4, "expert")):
            self.game.seen.clear()
            self._post_key(key)
            self.assertIn("difficulty:" + name, self.game.seen,
                          "按键 %s 的难度命令丢了：%s" % (key, self.game.seen))

    def test_wheel_over_log_scrolls_it(self):
        self.assertEqual(self.renderer.log_scroll, 0)
        self._post_wheel(self._log_center(), 4)      # 4 = 向上滚
        self.assertGreater(self.renderer.log_scroll, 0, "滚轮没有滚动记录")
        top = self.renderer.log_scroll
        self._post_wheel(self._log_center(), 5)      # 5 = 向下滚
        self.assertLess(self.renderer.log_scroll, top, "向下滚没有回退")

    def test_wheel_over_board_does_not_scroll_or_move(self):
        before_ply = self.gs.ply
        before_scroll = self.renderer.log_scroll
        board_center = self.renderer.board_rect.center
        self._post_wheel(board_center, 4)
        self.assertEqual(self.renderer.log_scroll, before_scroll,
                         "在棋盘上滚轮不该翻记录")
        self.assertEqual(self.gs.ply, before_ply, "滚轮不该被当成落子")

    def test_wheel_over_board_does_not_restart_finished_game(self):
        """终局后在棋盘上滚轮，不能被当成"点击重新开始"。"""
        self.gs.game_over = True
        self.gs.winner = 'red'
        self.renderer.draw_all()
        total = self.gs.total_plies
        self._post_wheel(self.renderer.board_rect.center, 5)
        self.assertTrue(self.gs.game_over, "滚轮把终局给重开了")
        self.assertEqual(self.gs.total_plies, total)

    def test_buttons_still_work_when_game_is_over(self):
        """终局后右侧按钮必须仍然点得动（不能被"点击任意位置重新开始"吃掉）。"""
        self.gs.game_over = True
        self.gs.winner = 'red'
        self.renderer.draw_all()
        total = self.gs.total_plies

        # 点「上一步」：应当进入浏览状态，而不是重开一局
        _cmd, rect, _t, enabled = self.renderer.nav_buttons()[1]
        self.assertTrue(enabled, "终局后「上一步」应当是可用状态")
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"pos": rect.center, "button": 1}))
        self.handler.handle_events()
        self.assertIn('nav:prev', self.game.seen, "点击没有被派发成命令")
        self.assertEqual(self.gs.total_plies, total, "点按钮却把对局重开了")
        self.assertEqual(self.gs.ply, total - 1, "点「上一步」没有生效")
        self.assertFalse(self.gs.game_over, "翻到历史后应当退出结算状态")

    def test_clicking_empty_board_area_when_over_restarts(self):
        """终局后点棋盘空白处＝重新开始（这条老行为必须保留）。"""
        self.gs.game_over = True
        self.gs.winner = 'red'
        self.renderer.draw_all()
        self.assertGreater(self.gs.total_plies, 0)
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"pos": self.renderer.board_rect.center, "button": 1}))
        self.handler.handle_events()
        self.assertEqual(self.gs.total_plies, 0, "点棋盘空白没有重新开始")


if __name__ == "__main__":
    unittest.main()
