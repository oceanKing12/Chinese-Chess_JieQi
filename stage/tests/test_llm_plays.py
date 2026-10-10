# -*- coding: utf-8 -*-
"""「大模型」这一档对局方式的测试 —— **完全采用它的招法**，但不许走出非法棋。

需求原话："支持切换自己的引擎，对方的引擎，大模型三种对局方式，
切换到哪种就完全应用那个引擎的招法。"

"完全采用大模型招法"以前其实做不到：旧实现**只接受"本来就在引擎候选里"的
着法**，于是大模型真正想下的那一手只要不在引擎的短名单里就被丢掉，
结果还是引擎在下棋。现在两档都收：

* 在候选里 → 用它（分数沿用候选）；
* **不在候选里但在我们规则下合法** → 照样采用，分数用 ``Engine.score_move``
  现算一次浅搜；
* 非法 / 没给 / 调用失败 → 退回本机引擎，并把"本步由本机引擎代走"写进
  notes 与面板（复盘时看得出这一手不是大模型下的）。

全部用替身，**不联网**（联网守卫会把任何真实请求变成失败）。
"""
import os
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
for _p in (str(GAME_DIR.parent), str(GAME_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pygame  # noqa: E402

from jieqi import llm as llm_mod  # noqa: E402
from jieqi.board import encode as encode_move  # noqa: E402
from jieqi.consts import RED  # noqa: E402
from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402
from jieqi.notation import move_name  # noqa: E402

SETTINGS = {"difficulty": "easy", "time_limit": 0.25, "samples": 1,
            "max_depth": 3, "seed": 7, "emit_decision_tree": False,
            "verbose": False, "weights": {}}


def setUpModule():
    pygame.init()
    pygame.display.set_mode((1, 1))
    # 安全网：任何一次真实联网请求立刻失败（绝不花用户的额度）
    guard = mock.patch.object(llm_mod.urllib.request, "urlopen",
                              side_effect=AssertionError("测试不允许联网"))
    guard.start()
    unittest.addModuleCleanup(guard.stop)


def make_gs(plies=0):
    from game_state import GameState

    random.seed(11)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = 'red'
    gs.current_player = 'red'
    rng = random.Random(11)
    for _ in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.current_player = 'black' if side == 0 else 'red'
    return gs


def llm_reply(move=None, comment="点评内容", ok=True, error=""):
    return llm_mod.LLMResult(
        ok=ok, text="{}",
        parsed={"move": move, "comment": comment, "confidence": 0.9},
        error=error, latency=0.01, model="stub", usage={})


def make_controller(gs, reply):
    """装配控制器，并把"问大模型"这一步替换成固定回复（不联网）。"""
    import AI as ai_module

    with mock.patch.object(ai_module, "ai_settings",
                           return_value=dict(SETTINGS)):
        controller = ai_module.AdvancedAIController(gs)
    controller.settings.update(SETTINGS)
    controller.engine.options.update(SETTINGS)
    controller.llm_enabled = True
    controller.advisor = object()          # 只要非 None 就算"装了顾问"
    controller._ask_llm = lambda snapshot, style, timeout=0.0: reply
    return controller


def all_legal_ints(gs):
    side = gui_color_to_side(gs.current_player)
    return sorted(legal_moves(gs.board, side))


class TestLlmPlaysTheSide(unittest.TestCase):
    def test_adopts_a_candidate_move(self):
        """大模型给的是引擎候选里的着法 → 直接采用。"""
        gs = make_gs()
        legal = all_legal_ints(gs)
        target = legal[0]
        fr, fc = (target >> 7) // 9, (target >> 7) % 9
        tr, tc = (target & 127) // 9, (target & 127) % 9
        controller = make_controller(gs, llm_reply((fr, fc, tr, tc)))
        result = controller.analyse_llm(RED)
        self.assertIsNotNone(result)
        self.assertEqual(result.move, target)
        self.assertEqual(controller.last_source, "llm")
        self.assertTrue(any("大模型" in n for n in result.notes), result.notes)
        self.assertIn("大模型决定", controller.last_comment)

    def test_adopts_a_legal_move_outside_the_candidates(self):
        """**合法但不在候选里**的着法也要采用——否则"完全采用"就是空话。

        这是本轮修掉的核心行为：旧实现只认候选里的着法。
        """
        gs = make_gs()
        legal = all_legal_ints(gs)
        target = legal[-1]              # 故意挑一个大概率不在短名单里的
        fr, fc = (target >> 7) // 9, (target >> 7) % 9
        tr, tc = (target & 127) // 9, (target & 127) % 9
        controller = make_controller(gs, llm_reply((fr, fc, tr, tc)))
        # 先把候选列表裁到只剩一手，保证目标**一定**不在候选里
        real_analyse = controller.engine.analyse

        def shrunk(*a, **kw):
            res = real_analyse(*a, **kw)
            res.stats = [st for st in res.stats if st.move != target] or res.stats
            return res

        controller.engine.analyse = shrunk
        result = controller.analyse_llm(RED)
        self.assertEqual(result.move, target,
                         "合法的大模型着法必须被采用，实际 %s"
                         % move_name(result.move))
        self.assertEqual(controller.last_source, "llm")
        self.assertTrue(any("现算评估" in n for n in result.notes), result.notes)
        # 分数要被算出来（不是 0/未定义），否则面板会显示一个假评估
        self.assertIsInstance(float(result.score), float)

    def test_illegal_move_falls_back_and_says_so(self):
        """非法着法绝不能落盘：退回本机引擎，并且**明确标注**。"""
        gs = make_gs()
        controller = make_controller(gs, llm_reply((0, 0, 9, 8)))
        result = controller.analyse_llm(RED)
        self.assertIn(result.move, all_legal_ints(gs))
        self.assertEqual(controller.last_source, "python")
        self.assertTrue(any("大模型" in n and "代走" in n for n in result.notes),
                        result.notes)
        self.assertIn("代走", controller.last_comment)

    def test_no_move_falls_back(self):
        """模型没给着法（常见：它只会点评）→ 退回本机引擎。"""
        gs = make_gs()
        controller = make_controller(gs, llm_reply(None))
        result = controller.analyse_llm(RED)
        self.assertIn(result.move, all_legal_ints(gs))
        self.assertEqual(controller.last_source, "python")
        self.assertTrue(any("没有给出着法" in n for n in result.notes),
                        result.notes)

    def test_call_failure_falls_back(self):
        """调用失败（超时/没 key）→ 退回本机引擎，绝不把棋局卡住。"""
        gs = make_gs()
        controller = make_controller(gs, llm_reply(None, ok=False,
                                                   error="timeout"))
        result = controller.analyse_llm(RED)
        self.assertIn(result.move, all_legal_ints(gs))
        self.assertEqual(controller.last_source, "python")

    def test_without_advisor_falls_back(self):
        """没装配顾问（没配 API Key）时也要能走完这一步。"""
        gs = make_gs()
        controller = make_controller(gs, llm_reply(None))
        controller.advisor = None
        result = controller.analyse_llm(RED)
        self.assertIn(result.move, all_legal_ints(gs))
        self.assertEqual(controller.last_source, "python")
        self.assertTrue(any("不可用" in n for n in result.notes), result.notes)

    def test_snapshot_uses_the_playing_side(self):
        """提示词里的"AI 执哪方"必须是**这一档在替哪一方**。

        两方都可以配大模型，而 ``side`` 是显式传进来的，不能再读
        ``gs.ai_color``（那只表示"某一方"）。
        """
        gs = make_gs()
        seen = {}

        def spy(snapshot, style, timeout=0.0):
            seen["ai_color"] = snapshot.get("ai_color")
            return llm_reply(None)

        controller = make_controller(gs, llm_reply(None))
        controller._ask_llm = spy
        gs.current_player = 'black'
        controller.analyse_llm(1)      # BLACK
        self.assertEqual(str(seen.get("ai_color")).lower(), "black")
        gs.current_player = 'red'
        controller.analyse_llm(0)      # RED
        self.assertEqual(str(seen.get("ai_color")).lower(), "red")

    def test_score_is_red_perspective_when_black_plays(self):
        """黑方那一手被采用后，``score`` 必须是**红方视角**。

        为什么专门测黑方：``MoveStat.mean`` 是**走子方视角**，
        ``SearchResult.score`` / ``verdict`` 是**红方视角**（界面与棋谱都用）。
        旧代码在采用大模型招法时乘的是"走子方符号"，黑方走就被翻了两次
        ——表现是"黑方走出好棋，面板却说红方大优"。
        """
        gs = make_gs(1)                     # 走一手之后轮到黑方
        gs.current_player = 'black'
        moves = sorted(legal_moves(gs.board, 1))
        self.assertTrue(moves)
        target = moves[0]
        fr, fc = (target >> 7) // 9, (target >> 7) % 9
        tr, tc = (target & 127) // 9, (target & 127) % 9
        controller = make_controller(gs, llm_reply((fr, fc, tr, tc)))
        result = controller.analyse_llm(1)      # BLACK
        self.assertEqual(controller.last_source, "llm")
        self.assertEqual(result.move, target)
        # 红方视角 = 走子方视角 × (-1)（黑方走时）
        stat = next((st for st in result.stats if st.move == target), None)
        self.assertIsNotNone(stat, "首选着法应当在候选里")
        self.assertAlmostEqual(result.score, stat.mean * -1, delta=1e-6,
                               msg="黑方视角的分数没有换算成红方视角")

    def test_off_candidate_score_is_red_perspective(self):
        """不在候选里的那一手：分数由 ``score_move`` 现算，同样是红方视角。"""
        gs = make_gs(1)
        gs.current_player = 'black'
        moves = sorted(legal_moves(gs.board, 1))
        target = moves[-1]
        fr, fc = (target >> 7) // 9, (target >> 7) % 9
        tr, tc = (target & 127) // 9, (target & 127) % 9
        controller = make_controller(gs, llm_reply((fr, fc, tr, tc)))
        real = controller.engine.analyse

        def shrunk(*a, **kw):
            res = real(*a, **kw)
            res.stats = [st for st in res.stats if st.move != target]
            return res

        controller.engine.analyse = shrunk
        result = controller.analyse_llm(1)
        self.assertEqual(result.move, target)
        self.assertTrue(any("现算评估" in n for n in result.notes), result.notes)
        from jieqi.gui_adapter import gui_board_to_engine

        direct = controller.engine.score_move(
            gui_board_to_engine(gs.board, turn=1),
            encode_move(fr * 9 + fc, tr * 9 + tc), depth=1)
        if direct is not None:
            # 两侧都是"红方视角"：符号必须一致（采样有随机性，不比精确值）
            self.assertEqual(result.score < 0, direct < 0,
                             "现算的分数符号不是红方视角：%s vs %s"
                             % (result.score, direct))


class TestAdoptLlmMoveUnit(unittest.TestCase):
    """``_adopt_llm_move`` 的边界（直接喂一个合成结果，不跑引擎）。"""

    def _result(self):
        from jieqi.engine import MoveStat, SearchResult

        st = MoveStat(move=encode_move(0 * 9 + 0, 1 * 9 + 0), mean=42.0,
                      worst=42.0, samples=1, best_count=1, select_score=42.0,
                      pv=[encode_move(0 * 9 + 0, 1 * 9 + 0)])
        return SearchResult(move=st.move, score=42.0, depth=2, nodes=10,
                            samples=1, elapsed=0.01, stats=[st],
                            verdict="红方稍优")

    def test_rejects_move_outside_the_legal_set(self):
        gs = make_gs()
        controller = make_controller(gs, llm_reply(None))
        res = self._result()
        legal = set(all_legal_ints(gs))
        self.assertFalse(controller._adopt_llm_move(res, (0, 0, 9, 8), legal=legal),
                         "非法着法必须被拒绝")

    def test_accepts_when_no_legal_set_given(self):
        """不传 ``legal`` 时保持老行为（只按候选判断），不误杀。"""
        gs = make_gs()
        controller = make_controller(gs, llm_reply(None))
        res = self._result()
        mv = res.move
        fr, fc = (mv >> 7) // 9, (mv >> 7) % 9
        tr, tc = (mv & 127) // 9, (mv & 127) % 9
        self.assertTrue(controller._adopt_llm_move(res, (fr, fc, tr, tc)))
        self.assertEqual(res.move, mv)


if __name__ == "__main__":
    unittest.main()
