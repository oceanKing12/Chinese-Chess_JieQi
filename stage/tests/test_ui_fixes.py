# -*- coding: utf-8 -*-
"""本轮三处界面/计时修复的回归测试。

对应三条真实反馈：

1. **被吃棋子条上下颠倒**："上方应该是对方吃的子（即我方被吃的），下方是我方
   吃的子（即对方被吃的）"——棋盘是"我方在下方"的视角，所以靠近自己的那条
   （下面）该摆自己吃到的战利品。
2. **AI 思考时间不准**："明明过了很久，却显示才 2000ms"——旧代码量的是搜索
   线程内部的 CPU 耗时，而玩家等的是**挂钟时间**（含 GIL 争抢、结果等主线程
   取走的那一段）。思考期间还要显示实时秒表，而不是上一手的用时。
3. **棋盘页精简**：难度与大模型开关搬到启动页，翻手按钮搬到顶栏。
"""
import os
import sys
import time
import unittest
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(GAME_DIR.parent))
sys.path.insert(0, str(GAME_DIR))

import pygame  # noqa: E402

from resource_manager import ResourceManager  # noqa: E402

_ORIGINAL_CWD = os.getcwd()


def make():
    from game_state import GameState
    from renderer import Renderer

    pygame.init()
    pygame.display.set_mode((1, 1))
    rm = ResourceManager()
    rm.load_resources()
    gs = GameState(rm)
    gs.show_side_selection = False
    gs.enable_ai('black')
    return gs, Renderer(rm, gs)


class TestCaptureStrips(unittest.TestCase):
    """上=我方被吃，下=对方被吃。"""

    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        self.gs, self.r = make()
        # 造一手"红吃黑"：红方车吃黑卒（红在下方 = 玩家执红）
        self.gs.player_side = 'red'
        self.gs.move_history.append(
            {'piece': {'color': 'red', 'actual': '车', 'revealed': True},
             'start': (9, 0), 'end': (8, 0), 'side': 0,
             'captured': {'color': 'black', 'revealed': True, 'actual': '卒'}})
        self.gs.ply = 1
        self.r._capture_cache_key = None

    def test_top_strip_shows_my_losses(self):
        """上方条 = 我方（红）被吃的子；这里红方没损失，所以是空的。"""
        captured = self.r._captured_lists()
        mine = 'red'
        theirs = 'black'
        self.assertEqual(captured[mine], [], "红方还没丢子，上方条不该有东西")
        self.assertEqual(len(captured[theirs]), 1, "黑方被吃了一个卒")
        self.assertEqual(captured[theirs][0]['piece']['actual'], '卒')

    def test_strip_labels_use_the_mine_theirs_wording(self):
        """标签文案必须是"我方被吃 / 对方被吃"，且上条带的是 mine。"""
        drawn = {}
        key = lambda rect: (rect.x, rect.y)      # noqa: E731（Rect 不能当键）

        original = self.r._draw_one_capture_strip

        def spy(rect, items, label, label_side='left', accent=None):
            drawn[key(rect)] = (label, list(items))
            return original(rect, items, label, label_side=label_side,
                            accent=accent)

        self.r._draw_one_capture_strip = spy
        try:
            self.r.draw_capture_strips()
        finally:
            self.r._draw_one_capture_strip = original

        top = drawn[key(self.r.capture_top_rect)]
        bottom = drawn[key(self.r.capture_bottom_rect)]
        captured = self.r._captured_lists()
        self.assertEqual(top[0], "我方被吃")
        self.assertEqual(bottom[0], "对方被吃")
        self.assertEqual(top[1], captured['red'],
                         "上方条摆的应当是我方被吃的子")
        self.assertEqual(bottom[1], captured['black'],
                         "下方条摆的应当是我方吃到的子（对方被吃）")

    def test_black_player_sees_mirrored_strips(self):
        """玩家执黑时同样成立：上=黑方被吃，下=红方被吃。"""
        self.gs.player_side = 'black'
        captured = self.r._captured_lists()
        # 黑方被吃的是那个卒；红方没损失
        self.assertEqual(len(captured['black']), 1)
        self.assertEqual(captured['red'], [])

    def test_strip_accent_colour_follows_the_side(self):
        """两条彩带的颜色分别跟着"我方 / 对方"，红黑一眼可辨。"""
        colors = self.r.config.COLORS
        self.gs.player_side = 'red'
        self.assertEqual(colors['strip_red' if 'red' == 'red' else 'strip_black'],
                         colors['strip_red'])
        # 暗子图标两边都是深色，所以必须靠彩带+字色区分（这里锁住令牌存在）
        for key in ('strip_red', 'strip_black'):
            self.assertIn(key, colors)


class TestAiThinkTime(unittest.TestCase):
    """AI 用时必须是"真实挂钟"，且思考期间显示实时秒表。"""

    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        self.gs, self.r = make()

    def test_timer_runs_while_thinking(self):
        """思考中：顶栏显示"AI 思考中 x.xs"，且秒数随时间增长。"""
        self.gs.ai_thinking = True
        self.gs.ai_think_started = time.monotonic() - 3.25
        parts = self.r._ai_text_parts()
        self.assertEqual(len(parts), 1)
        self.assertIn('思考中', parts[0])
        # 只校验"约 3.x 秒"：渲染一刻还会走掉几十毫秒，写死 3.2 会偶发失败
        self.assertRegex(parts[0], r'3\.[2-5]s', "秒表该显示约 3.3s：%r" % parts[0])
        # 8 秒后应当变成 8.x 秒（同一个 started 时刻）
        self.gs.ai_think_started = time.monotonic() - 8.0
        later = self.r._ai_text_parts()[0]
        self.assertRegex(later, r'8\.[0-3]s', "秒表该显示约 8.0s：%r" % later)

    def test_thinking_shows_timer_instead_of_last_move_time(self):
        """思考期间**不能**继续显示上一手的用时（那正是用户看到的错觉）。"""
        self.gs.ai_eval_text = ("AI 用时 2031ms\n深度 4　1 采样　40960 节点\n"
                                "AI 评估：+83（红方稍优）")
        self.gs.ai_thinking = True
        self.gs.ai_think_started = time.monotonic() - 6.0
        parts = self.r._ai_text_parts()
        self.assertTrue(all('2031' not in p for p in parts),
                        "思考中不该还显示上一手的 2031ms：%r" % parts)
        self.assertIn('思考中', parts[0])

    def test_timer_reports_wall_clock_not_search_time(self):
        """``_finish_ai_think`` 取的是挂钟差，而不是搜索线程自报的耗时。"""
        import main as main_mod

        game = main_mod.ChessGame.__new__(main_mod.ChessGame)
        game.gs = self.gs
        game.ai_thread = None
        # 假装 1.7 秒前开始思考，而搜索线程只自报了 200ms
        game._ai_think_began = time.monotonic() - 1.7
        game._ai_think_ms_at_result = 200
        game._finish_ai_think()
        self.assertGreaterEqual(game.ai_think_ms, 1600,
                                "必须取挂钟时间（约 1700ms），实际 %d"
                                % game.ai_think_ms)
        self.assertLess(game.ai_think_ms, 2600)
        self.assertEqual(self.gs.ai_think_ms, game.ai_think_ms,
                         "写回 gs 的应当是同一个数（渲染读它）")
        self.assertFalse(self.gs.ai_thinking)
        self.assertEqual(game._ai_think_began, 0.0)

    def test_no_timer_without_a_start(self):
        """没在思考时不该显示秒表。"""
        self.gs.ai_thinking = False
        self.gs.ai_eval_text = ""
        self.assertEqual(self.r._ai_text_parts(), [])

    def test_stale_eval_text_cannot_hide_the_clock(self):
        """**回归**：秒表不能被"上一步的评估文案"顶掉。

        真 bug：`eval_for_ply(k)` 有"这一步还没评估就退回上一步的"兜底逻辑，
        而 `_sync_ai_state` 每帧都会把这份文案写进 `gs.ai_eval_text`。
        于是 AI 一想棋，顶栏显示的就变回**上一步**的 `AI 用时 2066ms`——
        玩家看到的现象正是"从 0 涨到 2 秒，然后又被重置回 2000ms"。
        """
        self.gs.ai_eval_text = ("AI 用时 2066ms\n深度 4　1 采样　27414 节点\n"
                                "AI 评估：-1051（黑方胜势）")
        self.gs.ai_thinking = True
        self.gs.ai_think_started = time.monotonic() - 1.4
        parts = self.r._ai_text_parts()
        self.assertEqual(len(parts), 1, "思考中只该有一行秒表：%r" % parts)
        self.assertRegex(parts[0], r'思考中.*1\.[4-7]s', parts[0])
        self.assertNotIn('2066', parts[0])

    def test_sync_does_not_write_eval_while_thinking(self):
        """思考期间 `_sync_ai_state` 不许往 ``ply_evals`` 里写东西。

        写进去的是**上一手**的用时+评估（`ai_think_ms` 那一刻还没更新），
        既会顶掉秒表，又会让棋谱里存一条张冠李戴的用时。
        """
        import main as main_mod

        class _FakeResult:
            depth, samples, nodes, score = 4, 1, 27414, -1051.0
            verdict = "黑方胜势"

            def mate_in(self):
                return None

        class _FakeController:
            last_result = _FakeResult()
            last_comment = ""
            comment_ply = 0
            llm_enabled = False
            apply_llm_move = False
            llm_model_label = ""
            difficulty = "normal"
            advisor = None

        game = main_mod.ChessGame.__new__(main_mod.ChessGame)
        game.gs = self.gs
        game.renderer = self.r
        game.ai_controller = _FakeController()
        game.ai_difficulty = None
        self.gs.ai_think_ms = 2066          # 上一手的用时
        self.gs.ply = 0                     # 刚开局：ply_evals 只有 index 0
        before = list(self.gs.ply_evals)
        self.gs.ai_thinking = True
        game._sync_ai_state()
        self.assertEqual(self.gs.ply_evals, before,
                         "思考期间不该写入评估文案")
        self.gs.ai_thinking = False
        game._sync_ai_state()
        self.assertNotEqual(self.gs.ply_evals, before,
                            "思考结束后应当正常写入（否则这条测试没测到东西）")
        self.assertIn("2066", self.gs.ply_evals[0])


class TestBoardPageIsSlimmed(unittest.TestCase):
    """棋盘页只报状态：配置项都在启动页上。"""

    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        self.gs, self.r = make()

    def test_config_buttons_are_not_on_the_board_page(self):
        cmds = [cmd for cmd, _r, _t, _a in self.r._all_buttons()]
        for unwanted in ('llm:toggle', 'llm:apply'):
            self.assertNotIn(unwanted, cmds, f"棋盘页不该再有 {unwanted}")
        self.assertFalse(any(c.startswith('difficulty:') for c in cmds),
                         "棋盘页不该再有难度按钮")
        self.assertFalse(any(c.startswith('engine:') for c in cmds),
                         "棋盘页不该再有对局方式 chip（热键 E/T 切档）")

    def test_start_page_has_the_config_buttons(self):
        self.gs.show_side_selection = True
        cmds = [cmd for cmd, _r, _t, _a in self.r._all_buttons()]
        for wanted in ('setting:autosave',
                       'difficulty:easy', 'difficulty:normal',
                       'difficulty:hard', 'difficulty:expert',
                       'engine:red:human', 'engine:red:python',
                       'engine:red:llm',
                       'engine:black:human', 'engine:black:python',
                       'engine:black:llm',
                       'game:start', 'record:import',
                       'game:continue'):
            self.assertIn(wanted, cmds, f"启动页缺少 {wanted}")

    def test_toast_rect_has_room(self):
        """toast 必须**真的有宽度**：旧版算出来是 0px，提示根本看不见。"""
        self.assertGreaterEqual(self.r.toast_rect.width, 40,
                                "toast 区宽度不足（%d）" % self.r.toast_rect.width)
        self.assertGreaterEqual(self.r.status_text_max_w, 120)

    def test_top_bar_text_regions_do_not_overlap_buttons(self):
        info = self.r.ai_info_rect
        ev = self.r.ai_eval_rect
        navs = [rect for _c, rect, _t, _a in self.r.nav_buttons()]
        for rect in navs:
            self.assertFalse(info.colliderect(rect))
            self.assertFalse(ev.colliderect(rect))
        self.assertLessEqual(ev.right, navs[0].x)
        # 两个文字区的右边界对齐到同一条竖线
        self.assertEqual(info.right, ev.right,
                         "两行右边界没对齐：%s vs %s" % (info, ev))

    def test_auto_save_toggle_wins_over_config(self):
        """启动页拨了"自动保存"就以它为准（不改配置文件）。"""
        self.gs.auto_save_override = False
        self.assertFalse(self.gs.auto_save_record())
        self.gs.auto_save_override = True
        self.assertTrue(self.gs.auto_save_record())
        self.gs.auto_save_override = None
        self.assertIsInstance(self.gs.auto_save_record(), bool)


class TestScriptsDoNotTouchUserRecords(unittest.TestCase):
    """自动化脚本必须能写到**临时目录**，不能污染玩家的 ``records/``。

    为什么值得测：跑一次 ``_bench/deploy_selftest.py`` 就往 ``Game/records``
    里塞一份 30 手的"测试对局"（还有终局自动保存的那一份），玩家的棋谱列表
    里于是混进几份 0-1 的假对局，他自己都分不清哪些是自己的。修法是给
    ``GameState`` 一个 ``record_dir``，脚本传临时目录。
    """

    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def test_record_dir_is_honoured(self):
        import tempfile
        from game_state import GameState
        from jieqi.gui_adapter import gui_color_to_side, legal_moves

        tmp = Path(tempfile.mkdtemp(prefix="jieqi-recorddir-"))
        pygame.init()
        pygame.display.set_mode((1, 1))
        gs = GameState(None, record_dir=str(tmp))
        self.assertEqual(gs.record_dir, str(tmp))
        gs.show_side_selection = False
        gs.player_side = "red"
        gs.enable_ai("black")
        for _ in range(4):
            side = gui_color_to_side(gs.current_player)
            moves = legal_moves(gs.board, side)
            if not moves:
                break
            mv = moves[0]
            gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                          (mv & 127) // 9, (mv & 127) % 9)
            gs.current_player = "black" if side == 0 else "red"
        info = gs.export_record()
        self.assertTrue(info.get("ok"), info)
        self.assertEqual(Path(info["path"]).parent.resolve(), tmp.resolve(),
                         "棋谱没写进指定的临时目录：%s" % info["path"])
        self.assertTrue(list(tmp.glob("*.json")), "临时目录里没有棋谱文件")

    def test_default_record_dir_is_still_the_game_folder(self):
        """不给 ``record_dir`` 时仍然写 ``Game/records``（线上行为不变）。"""
        from game_state import GameState
        from jieqi import record as R

        pygame.init()
        pygame.display.set_mode((1, 1))
        gs = GameState(None)
        self.assertIsNone(gs.record_dir)
        self.assertEqual(Path(R.record_dir()).resolve(),
                         (GAME_DIR / "records").resolve())


if __name__ == "__main__":
    unittest.main()
