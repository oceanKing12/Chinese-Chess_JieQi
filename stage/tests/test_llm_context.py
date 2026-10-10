"""大模型上下文（记住整盘棋）+ 「应用大模型招法」+ 假杀棋分数修正。"""
import contextlib
import os
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(GAME_DIR.parent))
sys.path.insert(0, str(GAME_DIR))

import pygame  # noqa: E402

from jieqi import llm as llm_mod  # noqa: E402
from jieqi.board import move_str  # noqa: E402
from jieqi.consts import MATE_BOUND  # noqa: E402
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    build_llm_snapshot, gui_board_to_engine, gui_color_to_side, legal_moves,
)

SETTINGS = {"difficulty": "easy", "time_limit": 0.25, "samples": 1,
            "max_depth": 3, "seed": 7, "emit_decision_tree": False,
            "verbose": False, "weights": {}}


def setUpModule():
    pygame.init()
    pygame.display.set_mode((1, 1))
    # 安全网：本模块任何一次真的联网都会立刻变成失败，绝不会花掉用户的额度。
    # （需要验证网络路径的用例在 test_llm.py 里，自己 patch urlopen。）
    guard = mock.patch.object(llm_mod.urllib.request, "urlopen",
                              side_effect=AssertionError("测试不允许联网"))
    guard.start()
    unittest.addModuleCleanup(guard.stop)


def make_gs(plies=6):
    from game_state import GameState

    random.seed(11)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = 'red'
    gs.current_player = 'red'
    gs.enable_ai('black')
    rng = random.Random(11)
    for k in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.set_ply_comment(gs.ply, f'第{k + 1}手：我在争取中路。', f'AI 评估：+{k * 10}')
        gs.current_player = 'black' if side == 0 else 'red'
    return gs


class TestSnapshotContext(unittest.TestCase):
    def test_snapshot_carries_move_list_and_evals(self):
        gs = make_gs()
        result = Engine("easy", time_limit=0.2, samples=1, seed=1).analyse(
            gui_board_to_engine(gs.board, turn=gui_color_to_side(gs.current_player)))
        snap = build_llm_snapshot(gs, result, ai_color=1)
        self.assertIn('move_list', snap)
        self.assertEqual(len(snap['move_list']), gs.ply)
        self.assertIn('eval_history', snap)
        self.assertTrue(snap['eval_history'])
        self.assertIn('previous_comments', snap)
        # 上一步的点评必须带进来，模型才能前后对照
        self.assertIn('中路', snap['previous_comments'][-1])

    def test_prompt_renders_context(self):
        gs = make_gs()
        result = Engine("easy", time_limit=0.2, samples=1, seed=1).analyse(
            gui_board_to_engine(gs.board, turn=gui_color_to_side(gs.current_player)))
        snap = build_llm_snapshot(gs, result, ai_color=1)
        text = llm_mod.render_position_text(snap)
        self.assertIn('走棋记录', text)
        self.assertIn('评估走势', text)
        self.assertIn('你上一步的点评', text)
        # 上下文不能落进"其它字段"
        self.assertNotIn('其它字段', text)

    def test_snapshot_uses_displayed_ply_only(self):
        """浏览到第 3 手时，提示词里不能出现第 4 手以后的内容。"""
        gs = make_gs()
        gs.goto_ply(3)
        snap = build_llm_snapshot(gs)
        self.assertEqual(len(snap['move_list']), 3)
        self.assertTrue(snap['move_list'][-1].startswith('  3.'))
        for line in snap['move_list']:
            self.assertFalse(line.startswith('  4.'), line)
        # 评估走势与"上一步点评"也都只能到第 3 手为止；
        # 而且第 3 手自己的点评属于"当前局面"，不该出现在"上一步点评"里
        self.assertEqual(len(snap['eval_history']), 2)
        self.assertEqual(len(snap['previous_comments']), 2)
        self.assertTrue(all('第3手' not in c for c in snap['previous_comments']))
        text = llm_mod.render_position_text(snap)
        self.assertIn('走棋记录', text)
        self.assertNotIn('  4.', text)

    def test_system_prompt_requires_dynamic_assessment(self):
        prompt = llm_mod.POSITION_SYSTEM_PROMPT
        for keyword in ('走棋记录', '评估走势', '上一步点评', '改判'):
            self.assertIn(keyword, prompt)

    def test_system_prompt_pins_the_comment_target_to_ai(self):
        """提示词必须明确"只点评 AI 自己的招法"，不能跑题去指导人类。"""
        prompt = llm_mod.POSITION_SYSTEM_PROMPT
        self.assertIn('点评对象', prompt)
        self.assertIn('AI 一方', prompt)
        self.assertIn('不要去指导人类一方', prompt)

    def test_snapshot_declares_comment_target(self):
        gs = make_gs()
        snap = build_llm_snapshot(gs)
        self.assertEqual(snap['comment_target'], 'AI 自己的招法')
        text = llm_mod.render_position_text(snap)
        self.assertIn('点评对象：AI 自己的招法', text)


class TestApplyLlmMove(unittest.TestCase):
    """「应用大模型招法」：合法就采用，不合法就退回引擎着法。"""

    def _cfg(self):
        cfg = llm_mod.LLMConfig()
        cfg.enabled = True
        cfg.api_key = 'test-key'
        cfg.model = 'stub'
        cfg.blend_weight = 0.0
        cfg.mode = 'suggest'
        return cfg

    @staticmethod
    def _result(move=None):
        return llm_mod.LLMResult(
            ok=True, text="{}",
            parsed={"move": move, "comment": "点评", "confidence": 0.9},
            model='stub')

    def test_adopts_legal_suggestion(self):
        import AI as ai_module

        gs = make_gs()
        with mock.patch.object(ai_module, 'ai_settings',
                               return_value=dict(SETTINGS)), \
                mock.patch.object(ai_module, 'load_llm_config',
                                  return_value=self._cfg()), \
                mock.patch.object(llm_mod.LLMAdvisor, 'suggest',
                                  return_value=self._result(None)) as mocked:
            controller = ai_module.AdvancedAIController(gs)
            controller.apply_llm_move = True
            controller.llm_enabled = True
            # 第一次：模型没给着法，只用来拿到引擎的合法候选
            first = controller.analyse()
            candidates = [st.move for st in first.stats]
            self.assertGreaterEqual(len(candidates), 2)

            target = candidates[1]          # 故意挑一个不是首选的
            fr, fc = (target >> 7) // 9, (target >> 7) % 9
            tr, tc = (target & 127) // 9, (target & 127) % 9
            mocked.return_value = self._result((fr, fc, tr, tc))
            second = controller.analyse()

        self.assertEqual(second.move, target,
                         f"应采纳大模型招法 {move_str(target)}，"
                         f"实际 {move_str(second.move)}")
        self.assertTrue(any('大模型' in n for n in second.notes))
        self.assertIn(second.move, legal_moves(
            gs.board, gui_color_to_side(gs.current_player)))

    def test_rejects_illegal_suggestion(self):
        """**非法**着法必须被拒绝，并由引擎给一个合法的着法。

        怎么造一个"像大模型会犯的错"：拿当前局面的合法着法，把**落点**改成
        起点自己（原地不动/目标格上有自己的子）——这在棋盘上一定非法，
        但形式上是合法的坐标对。旧版本这里喂的是 ``(0,0)->(9,8)``
        （空格走到空格），而旧实现**不看是否合法**，所以那条"非法建议"
        其实被原样落盘了；修好校验之后它才暴露出"测试自己也不合法"。
        """
        import AI as ai_module

        gs = make_gs()
        side = gui_color_to_side(gs.current_player)
        legal = sorted(legal_moves(gs.board, side))
        self.assertTrue(legal)
        bad = legal[0]
        fr, fc = (bad >> 7) // 9, (bad >> 7) % 9
        with mock.patch.object(ai_module, 'ai_settings',
                               return_value=dict(SETTINGS)), \
                mock.patch.object(ai_module, 'load_llm_config',
                                  return_value=self._cfg()), \
                mock.patch.object(llm_mod.LLMAdvisor, 'suggest',
                                  return_value=self._result((fr, fc, fr, fc))):
            controller = ai_module.AdvancedAIController(gs)
            controller.apply_llm_move = True
            controller.llm_enabled = True
            res = controller.analyse()
        self.assertIn(res.move, legal_moves(gs.board, side),
                      "非法的大模型着法被落盘了")
        self.assertTrue(any('不合法' in n or '忽略' in n for n in res.notes),
                        res.notes)

    def test_apply_implied_by_settings_and_toggle(self):
        import AI as ai_module

        gs = make_gs()
        settings = dict(SETTINGS, apply_llm_move=True)
        with mock.patch.object(ai_module, 'ai_settings',
                               return_value=settings), \
                mock.patch.object(ai_module, 'load_llm_config',
                                  return_value=self._cfg()), \
                mock.patch.object(llm_mod.LLMAdvisor, 'suggest',
                                  return_value=self._result(None)):
            controller = ai_module.AdvancedAIController(gs)
            self.assertTrue(controller.apply_llm_move, "配置文件里的开关没生效")
            self.assertFalse(controller.toggle_apply_llm_move())
            self.assertFalse(controller.apply_llm_move)

    def test_toggle_requires_advisor(self):
        import AI as ai_module

        gs = make_gs()
        with mock.patch.object(ai_module, 'ai_settings',
                               return_value=dict(SETTINGS)), \
                mock.patch.object(ai_module, 'load_llm_config',
                                  side_effect=ImportError("no llm")):
            controller = ai_module.AdvancedAIController(gs)
        self.assertFalse(controller.toggle_apply_llm_move())


class TestCommentOnlyForAiMoves(unittest.TestCase):
    """点评只针对 AI 自己的招法。

    玩家点「让 AI 替我走这步」时走的是同一个引擎，但**绝不能**去问大模型，
    否则面板上会出现"AI 在点评我刚走的那一步"，谁也看不出点评在说谁。
    """

    def _cfg(self):
        cfg = llm_mod.LLMConfig()
        cfg.enabled = True
        cfg.api_key = 'test-key'
        cfg.model = 'stub'
        cfg.blend_weight = 0.0
        cfg.mode = 'explain'
        return cfg

    @contextlib.contextmanager
    def _controller(self, gs, spy):
        """构造控制器，并**在整个 with 块内**拦住对模型的所有调用。

        注意 patch 必须活到 ``analyse()`` 之后：构造完控制器就退出 with 的话，
        真实的 ``suggest_async`` 会被调用（进而在测试里真的联网）。
        """
        import AI as ai_module

        with mock.patch.object(ai_module, 'ai_settings',
                               return_value=dict(SETTINGS)), \
                mock.patch.object(ai_module, 'load_llm_config',
                                  return_value=self._cfg()), \
                mock.patch.object(llm_mod.LLMAdvisor, 'suggest_async', spy), \
                mock.patch.object(llm_mod.LLMAdvisor, 'suggest', spy):
            yield ai_module.AdvancedAIController(gs)

    def test_ai_move_requests_comment(self):
        gs = make_gs()
        calls = []
        with self._controller(gs, lambda *a, **k: calls.append(a)) as controller:
            self.assertIsNotNone(controller.advisor, controller.llm_status)
            controller.llm_enabled = True
            controller.analyse()                 # 默认 comment=True
        self.assertTrue(controller.comment_for_ai)
        self.assertTrue(calls, "AI 自己走棋时必须请求点评")

    def test_comment_false_skips_the_model_entirely(self):
        """``comment=False`` 时一次请求都不能发出去（含同步路径）。"""
        gs = make_gs()
        calls = []
        with self._controller(gs, lambda *a, **k: calls.append(a)) as controller:
            controller.llm_enabled = True
            controller.apply_llm_move = True     # 连"应用招法"也一起排除
            result = controller.analyse(comment=False)
        self.assertIsNotNone(result)
        self.assertFalse(controller.comment_for_ai)
        self.assertEqual(calls, [], f"替人类分析时不该访问大模型，却调用了 {calls}")
        self.assertEqual(controller.last_comment, "")

    def test_get_best_move_accepts_comment_flag(self):
        gs = make_gs()
        calls = []
        with self._controller(gs, lambda *a, **k: calls.append(a)) as controller:
            controller.llm_enabled = True
            move = controller.get_best_move(comment=False)
        self.assertIsNotNone(move)
        self.assertEqual(calls, [], "替人类要着法时不该请求点评")


class TestMateStabilisation(unittest.TestCase):
    """只有少数采样看到杀棋时，不能把平均分拉成 50 万这种天文数字。"""

    def test_single_mate_out_of_two_is_not_confirmed(self):
        vals = [999_998, 842]
        out = Engine._stabilise_mates(vals)
        self.assertEqual(out, [842, 842])
        self.assertLess(sum(out) / len(out), MATE_BOUND)

    def test_single_sample_passes_through(self):
        self.assertEqual(Engine._stabilise_mates([999_998]), [999_998])

    def test_strict_majority_mate_is_kept(self):
        vals = [999_998, 999_996, 999_994, 400]
        self.assertEqual(Engine._stabilise_mates(vals), vals)

    def test_two_of_four_is_not_confirmed(self):
        out = Engine._stabilise_mates([999_998, 999_996, 500, 400])
        self.assertTrue(all(abs(v) < MATE_BOUND for v in out),
                        f"2/4 不该确认为杀：{out}")
        # 非杀样本的中位数会被用来替换那些"假杀"分
        self.assertEqual(out, [450, 450, 500, 400])

    def test_all_mates_kept(self):
        vals = [999_998, 999_996]
        self.assertEqual(Engine._stabilise_mates(vals), vals)


if __name__ == "__main__":
    unittest.main()
