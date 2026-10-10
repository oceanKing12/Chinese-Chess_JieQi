"""LLM 顾问接线测试：默认不影响选着，显式给权重时才产生（有限的）倾斜。

这里不联网：把 ``LLMAdvisor`` 的网络方法替换成桩函数，检查 AI 控制器是否正确
调用、解析、以及在 blend_weight=0 时**完全不影响**引擎的选着。
"""
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

STAGE = Path(__file__).resolve().parents[1]
GAME = STAGE / "Game"
for p in (str(STAGE), str(GAME)):
    if p not in sys.path:
        sys.path.insert(0, p)

import os  # noqa: E402

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from jieqi import llm as llm_mod  # noqa: E402
from jieqi.board import move_str  # noqa: E402
from jieqi.gui_adapter import engine_board_to_gui, gui_color_to_side, legal_moves  # noqa: E402


def make_gamestate():
    from game_state import GameState

    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.ai_mode = True
    gs.ai_color = "black"
    return gs


def fake_config(**over):
    cfg = llm_mod.LLMConfig()
    cfg.enabled = True
    cfg.api_key = "test-key"
    cfg.model = "stub-model"
    cfg.timeout = 2.0
    cfg.mode = "explain"
    cfg.blend_weight = 0.0
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_result(move=None, comment="点评内容", ok=True, error=""):
    """真正的 LLMResult 实例（blend_bonus 会用 isinstance 判断类型）。"""
    return llm_mod.LLMResult(
        ok=ok, text="{}",
        parsed={"move": move, "comment": comment, "confidence": 0.9},
        error=error, latency=0.01, model="stub-model", usage={})


FakeResult = make_result


FIXED_SETTINGS = {
    "difficulty": "easy",
    "time_limit": 0.3,
    "samples": 2,
    "max_depth": None,
    "seed": 42,
    "emit_decision_tree": False,
    "decision_tree_path": "decision_tree.json",
    "verbose": False,
    "use_hidden_captures": False,
    "weights": {},
}


def base_engine():
    from jieqi.engine import Engine

    return Engine("easy", time_limit=0.3, samples=2, seed=42)


class TestLlmWiring(unittest.TestCase):
    def _analyse(self, gs, **cfg_over):
        import AI as ai_module

        cfg = fake_config(**cfg_over)
        with mock.patch.object(ai_module, "ai_settings",
                               return_value=dict(FIXED_SETTINGS)), \
                mock.patch.object(ai_module, "load_llm_config", return_value=cfg), \
                mock.patch.object(llm_mod.LLMAdvisor, "suggest",
                                  return_value=FakeResult()), \
                mock.patch.object(
                    llm_mod.LLMAdvisor, "suggest_async",
                    lambda self, snapshot, callback=None, style=None:
                    callback(FakeResult(comment="这是顾问点评"))):
            controller = ai_module.AdvancedAIController(gs)
            self.assertIsNotNone(controller.advisor, "LLM 顾问没有装配成功")
            result = controller.analyse()
        return controller, result

    def test_advisory_only_does_not_change_move(self):
        from jieqi.gui_adapter import gui_board_to_engine

        gs = make_gamestate()
        base = base_engine().analyse(
            gui_board_to_engine(gs.board, turn=gui_color_to_side("red")))
        controller, result = self._analyse(gs)
        self.assertEqual(result.move, base.move,
                         "blend_weight=0 时顾问不该改变选着")
        self.assertEqual(controller.last_comment, "这是顾问点评")

    def test_blend_weight_can_tilt_choice(self):
        """给足权重时，顾问推荐的候选着法会被选中（加分上限 50 厘兵）。"""
        from jieqi.gui_adapter import gui_board_to_engine

        gs = make_gamestate()
        base = base_engine().analyse(
            gui_board_to_engine(gs.board, turn=gui_color_to_side("red")))
        target = None
        for st in base.stats[1:]:
            if base.stats[0].select_score - st.select_score < 50:
                target = st.move
                break
        if target is None:
            self.skipTest("本局面候选分差都大于 50，无法演示倾斜")

        frm, to = target >> 7, target & 127
        # LLMResult.move 约定是扁平 4 元组 (fr, fc, tr, tc)
        suggested = (frm // 9, frm % 9, to // 9, to % 9)

        import AI as ai_module

        cfg = fake_config(blend_weight=200.0, mode="suggest")
        with mock.patch.object(ai_module, "ai_settings",
                               return_value=dict(FIXED_SETTINGS)), \
                mock.patch.object(ai_module, "load_llm_config", return_value=cfg), \
                mock.patch.object(llm_mod.LLMAdvisor, "suggest",
                                  return_value=FakeResult(move=suggested)):
            controller = ai_module.AdvancedAIController(gs)
            result = controller.analyse()
        self.assertEqual(result.move, target,
                         f"顾问推荐 {move_str(target)} 但引擎选了 {move_str(result.move)}")
        self.assertTrue(any("大模型" in n for n in result.notes))
        self.assertIn(result.move, legal_moves(gs.board, gui_color_to_side("red")))

    def test_llm_absent_by_default(self):
        """默认配置（enabled=false）下**不点评、不联网**，但顾问要装配好。

        为什么顾问仍要装配：``enabled`` 只决定"开局默认开不开点评"。如果这里
        因为 ``enabled=false`` 就不装顾问，``toggle_llm()`` 会永远返回 False，
        启动页上那个开关就成了摆设（用户必须先去改 json 再重启）。
        所以断言分两半：顾问在（可随时打开），而点评是关的、一次请求都不发。
        """
        import AI as ai_module

        gs = make_gamestate()
        with mock.patch.object(ai_module, "ai_settings",
                               return_value=dict(FIXED_SETTINGS)), \
                mock.patch.object(ai_module, "load_llm_config",
                                  return_value=fake_config(enabled=False)), \
                mock.patch.object(llm_mod.urllib.request, "urlopen",
                                  side_effect=AssertionError("不该联网")):
            controller = ai_module.AdvancedAIController(gs)
            result = controller.analyse()
        self.assertFalse(controller.llm_enabled, "enabled=false 时点评不该默认打开")
        self.assertIsNotNone(controller.advisor,
                             "顾问没装配：启动页的开关会永远打不开")
        self.assertIn("关闭", controller.llm_status)
        self.assertEqual(controller.last_comment, "")
        self.assertIn(result.move, legal_moves(gs.board, gui_color_to_side("red")))

    def test_llm_can_be_turned_on_at_runtime(self):
        """enabled=false 也能靠 ``toggle_llm()`` 打开（启动页那一拨的底层保证）。"""
        import AI as ai_module

        gs = make_gamestate()
        with mock.patch.object(ai_module, "ai_settings",
                               return_value=dict(FIXED_SETTINGS)), \
                mock.patch.object(ai_module, "load_llm_config",
                                  return_value=fake_config(enabled=False)):
            controller = ai_module.AdvancedAIController(gs)
            self.assertTrue(controller.toggle_llm(), "toggle_llm 应当能打开")
            self.assertTrue(controller.llm_enabled)
            self.assertFalse(controller.toggle_llm(), "再拨一次应当关掉")
            self.assertFalse(controller.llm_enabled)

    def test_status_explains_missing_key(self):
        """启用了但没填 key，必须给出可执行的提示，而不是静默失效。"""
        import AI as ai_module

        gs = make_gamestate()
        cfg = fake_config(enabled=True)
        cfg.api_key = ""
        cfg.api_key_env = "DEEPSEEK_API_KEY"
        with mock.patch.object(ai_module, "ai_settings",
                               return_value=dict(FIXED_SETTINGS)), \
                mock.patch.object(ai_module, "load_llm_config", return_value=cfg), \
                mock.patch.object(llm_mod.urllib.request, "urlopen",
                                  side_effect=AssertionError("不该联网")):
            controller = ai_module.AdvancedAIController(gs)
            controller.analyse()
        self.assertIsNone(controller.advisor)
        self.assertIn("API Key", controller.llm_status)
        self.assertIn("DEEPSEEK_API_KEY", controller.llm_status)

    def test_status_when_ready(self):
        import AI as ai_module

        gs = make_gamestate()
        cfg = fake_config(enabled=True)
        cfg.base_url = "https://api.deepseek.com"
        cfg.model = "deepseek-chat"
        with mock.patch.object(ai_module, "ai_settings",
                               return_value=dict(FIXED_SETTINGS)), \
                mock.patch.object(ai_module, "load_llm_config", return_value=cfg), \
                mock.patch.object(
                    llm_mod.LLMAdvisor, "suggest_async",
                    lambda self, snapshot, callback=None, style=None: None):
            controller = ai_module.AdvancedAIController(gs)
            result = controller.analyse()
        self.assertIsNotNone(controller.advisor)
        self.assertIn("deepseek-chat", controller.llm_status)
        self.assertIn("api.deepseek.com/chat/completions", controller.llm_status)
        self.assertIn("不影响选着", controller.llm_status)
        # 点评内容要落到 last_comment，供主线程同步到 gs.ai_comment 显示
        self.assertTrue(controller.last_comment)

    def test_failed_call_surfaces_error_text(self):
        """调用失败时把中文错误放进 last_comment，界面能显示出来。"""
        import AI as ai_module

        failure = llm_mod.LLMResult(ok=False, error="HTTP 401 未授权",
                                    model="deepseek-chat")

        def fake_async(self, snapshot, callback=None, style=None):
            callback(failure)

        gs = make_gamestate()
        cfg = fake_config(enabled=True)
        with mock.patch.object(ai_module, "ai_settings",
                               return_value=dict(FIXED_SETTINGS)), \
                mock.patch.object(ai_module, "load_llm_config", return_value=cfg), \
                mock.patch.object(llm_mod.LLMAdvisor, "suggest_async", fake_async):
            controller = ai_module.AdvancedAIController(gs)
            controller.analyse()
        self.assertIn("HTTP 401", controller.last_comment)


class TestSnapshotBuilding(unittest.TestCase):
    """提示词快照必须在真实棋局上也能组装成功。

    回归：棋子走到没有摆位身份的空行（第 1/4/5/8 行）时，
    旧实现用 HOME_IDENT 去索引 IDENT_CHARS，会 KeyError (0, 0)，
    导致游戏里的顾问一直「提示词组装失败」，只有命令行 demo 能用。
    """

    def test_snapshot_after_pieces_leave_home_squares(self):
        from jieqi.gui_adapter import build_llm_snapshot, gui_board_to_engine
        from jieqi.engine import Engine

        gs = make_gamestate()
        rng = random.Random(7)
        # 先随便走十几步，确保有棋子落在第 1/4/5/8 行
        for _ in range(14):
            side = gui_color_to_side(gs.current_player)
            moves = legal_moves(gs.board, side)
            if not moves:
                break
            mv = rng.choice(moves)
            sr, sc = (mv >> 7) // 9, (mv >> 7) % 9
            er, ec = (mv & 127) // 9, (mv & 127) % 9
            gs.move_piece(sr, sc, er, ec)
            gs.current_player = 'black' if side == 0 else 'red'

        result = Engine("easy", time_limit=0.2, samples=1, seed=1).analyse(
            gui_board_to_engine(gs.board, turn=gui_color_to_side(gs.current_player)))
        snapshot = build_llm_snapshot(gs, result, ai_color=1)  # 不能抛异常
        self.assertIn("board_text", snapshot)
        self.assertTrue(snapshot["board_text"])
        # 每行 9 格，空格为 None，棋子必须带上 side/hidden/home
        self.assertEqual(len(snapshot["board"]), 10)
        for row in snapshot["board"]:
            self.assertEqual(len(row), 9)
            for cell in row:
                if cell is not None:
                    self.assertIn(cell["side"], ("red", "black"))
                    self.assertIsInstance(cell["hidden"], bool)
                    self.assertIsInstance(cell["home"], str)
        self.assertTrue(snapshot.get("engine_candidates"))
        # 快照必须能被真正构造成请求体（这一步才是顾问实际走的路径）
        payload = llm_mod.build_payload(snapshot, cfg=fake_config())
        self.assertIn("messages", payload)

    def test_board_text_survives_moved_pieces(self):
        from jieqi.gui_adapter import board_text

        gs = make_gamestate()
        side = gui_color_to_side(gs.current_player)
        mv = legal_moves(gs.board, side)[0]
        sr, sc = (mv >> 7) // 9, (mv >> 7) % 9
        er, ec = (mv & 127) // 9, (mv & 127) % 9
        gs.move_piece(sr, sc, er, ec)
        text = board_text(gs.board, turn=gui_color_to_side(gs.current_player))
        self.assertIn("轮到", text)

    def test_snapshot_uses_side_to_move_perspective(self):
        """评分与结论必须都是「走棋方视角」，否则大模型会把优劣势说反。

        回归：模型曾对着「轮到黑方走 + 红方大优」写出"引擎判定黑方大优"。
        """
        import AI as ai_module
        from jieqi import board as B
        from jieqi.consts import BLACK, K, R, RED
        from jieqi.engine import Engine
        from jieqi.gui_adapter import (build_llm_snapshot, engine_board_to_gui,
                                       gui_board_to_engine)

        # 红方白多一车，且轮到黑方走 -> 黑方明显劣势
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 4, BLACK, K)
        b.put(4 * 9 + 4, RED, R, hidden=False)
        b.put(9 * 9 + 0, BLACK, B.P, hidden=False)
        b.turn = BLACK

        gs = make_gamestate()
        gs.board = engine_board_to_gui(b)
        gs.current_player = 'black'
        gs.move_count = 30
        result = Engine("easy", time_limit=0.3, samples=1, seed=5).analyse(
            gui_board_to_engine(gs.board, turn=BLACK))
        snap = build_llm_snapshot(gs, result, ai_color=BLACK)

        self.assertEqual(snap["side_to_move"], "black")
        # 走棋方（黑）劣势 -> 候选分应为负，结论应说红方好
        self.assertLess(snap["engine_candidates"][0]["score"], 0,
                        "候选分没有用走棋方视角")
        self.assertIn("红方", snap["engine_verdict"])
        self.assertNotIn("黑方大优", snap["engine_verdict"])
        self.assertIn("走棋", snap["score_basis"])

    def test_last_move_reports_revealed_identity(self):
        """翻子之后，提示词必须写明"翻开后是什么"。

        回归：move_history 存的是落子前的快照，所以一步翻子在提示词里一直写作
        「棋子 暗子」，与棋盘图上那一格已经是明子相矛盾，模型于是说
        "上一手为暗子，可能是兵或炮"。
        """
        from jieqi.consts import RED
        from jieqi.gui_adapter import build_llm_snapshot

        gs = make_gamestate()
        # 造一个"翻开后是红车"的暗子
        sq = next(s for s in range(90)
                  if gs.board[s // 9][s % 9]
                  and gs.board[s // 9][s % 9]['color'] == 'red'
                  and not gs.board[s // 9][s % 9]['revealed'])
        sr, sc = sq // 9, sq % 9
        gs.board[sr][sc]['actual'] = '车'
        mv = next(m for m in legal_moves(gs.board, RED) if (m >> 7) == sq)
        er, ec = (mv & 127) // 9, (mv & 127) % 9
        self.assertFalse(gs.board[sr][sc]['revealed'])
        gs.move_piece(sr, sc, er, ec)
        self.assertTrue(gs.board[er][ec]['revealed'])
        self.assertEqual(gs.board[er][ec]['actual'], '车')

        snap = build_llm_snapshot(gs)
        self.assertEqual(snap['last_move']['piece'], '暗子')
        self.assertEqual(snap['last_move']['revealed_as'], '红车')

        text = llm_mod.render_position_text(snap)
        self.assertIn('翻开后是', text)
        self.assertIn('红车', text)
        # 棋盘图上这一格必须是明子（不带问号）
        self.assertIn('车', snap['board_text'])

    def test_revealed_move_not_marked_hidden(self):
        """已经翻开的子再走一步，不该被说成暗子。"""
        from jieqi.consts import RED
        from jieqi.gui_adapter import build_llm_snapshot

        gs = make_gamestate()
        sq = next(s for s in range(90)
                  if gs.board[s // 9][s % 9]
                  and gs.board[s // 9][s % 9]['color'] == 'red'
                  and not gs.board[s // 9][s % 9]['revealed'])
        sr, sc = sq // 9, sq % 9
        gs.board[sr][sc]['actual'] = '马'
        mv = next(m for m in legal_moves(gs.board, RED) if (m >> 7) == sq)
        gs.move_piece(sr, sc, (mv & 127) // 9, (mv & 127) % 9)   # 第一步：翻开

        # 第二步：这枚已经是明马了
        side = gui_color_to_side(gs.current_player)
        mv2 = next(m for m in legal_moves(gs.board, side)
                   if (m >> 7) == (mv & 127))
        gs.move_piece((mv2 >> 7) // 9, (mv2 >> 7) % 9,
                      (mv2 & 127) // 9, (mv2 & 127) % 9)
        snap = build_llm_snapshot(gs)
        self.assertEqual(snap['last_move']['piece'], '红马')
        self.assertNotIn('revealed_as', snap['last_move'])
        from jieqi.consts import BLACK, RED
        from jieqi.gui_adapter import _side_verdict

        # 走棋方视角的分值 -> 中文结论（红黑按谁占优来说）
        self.assertEqual(_side_verdict(500, RED), "红方大优")
        self.assertEqual(_side_verdict(-500, RED), "黑方大优")
        self.assertEqual(_side_verdict(500, BLACK), "黑方大优")
        self.assertEqual(_side_verdict(-500, BLACK), "红方大优",
                         "黑方落后时不许说成黑方占优")
        self.assertEqual(_side_verdict(0, RED), "均势")
        self.assertEqual(_side_verdict(0, BLACK), "均势")


if __name__ == "__main__":
    unittest.main()
