# -*- coding: utf-8 -*-
"""大模型「归属」相关测试：事实表、提示词、幻觉质检与自动改判。

这些用例锁的是"AI 点评分不清谁的子"这个具体问题：

1. 提示词里必须**逐子写明归属**（我方/对方 + 兵种 + 位置），棋盘图每行自带列号；
2. 点评质检能抓出"我方车"但车是对方的这一类硬伤；
3. 整句把红黑说反时能**自动对调**改回来；
4. 改不回来并且重问也不干净时，宁可不点评也不误导玩家。
"""
from __future__ import annotations

import os
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
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    build_llm_snapshot, captured_gui_pieces, gui_board_to_engine,
    gui_color_to_side, legal_moves, llm_facts, track_captures,
)
from jieqi.llm import render_facts_text  # noqa: E402


def setUpModule():
    pygame.init()
    pygame.display.set_mode((1, 1))
    # 安全网：本模块绝不允许真的联网。
    guard = mock.patch.object(llm_mod.urllib.request, "urlopen",
                              side_effect=AssertionError("测试不允许联网"))
    guard.start()
    unittest.addModuleCleanup(guard.stop)


def make_gs(plies: int = 0, seed: int = 11, stop_on: str = ""):
    """随机走 ``plies`` 手。

    ``stop_on`` 非空时，走到"轮到该方走棋"就停（黑方摆位随机，手数奇偶会
    受全局随机状态影响，所以不靠奇偶猜，直接按目标方走）。
    """
    import random

    from game_state import GameState

    random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.enable_ai("black")
    rng = random.Random(seed)
    for _ in range(plies):
        if stop_on and gs.current_player == stop_on:
            break
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.current_player = "black" if side == 0 else "red"
    gs.animations.clear()
    return gs


def analyse(gs):
    turn = gui_color_to_side(gs.current_player)
    return Engine("easy", time_limit=0.25, samples=1, seed=3).analyse(
        gui_board_to_engine(gs.board, turn=turn))


def fake_config(**over):
    cfg = llm_mod.LLMConfig()
    cfg.enabled = True
    cfg.api_key = "test-key"
    cfg.model = "stub-model"
    cfg.timeout = 2.0
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def fake_result(comment="点评内容", move=None):
    return llm_mod.LLMResult(ok=True, text="{}",
                             parsed={"move": move, "comment": comment,
                                     "confidence": 0.9},
                             model="stub-model")


#: 人工搭的局面：**AI 执黑**。
#: 红方：车(0,0)、兵(0,1)   黑方：卒(9,1)、将(9,4) —— 两套字形不重叠。
BLACK_AI_SNAPSHOT = {
    "ai_color": "black",
    "llm_facts": {"ai_color": "black"},
    "board": [[None] * 9 for _ in range(10)],
}
BLACK_AI_SNAPSHOT["board"][0][0] = {"side": "red", "hidden": False,
                                    "home": "车", "ident": "车"}
BLACK_AI_SNAPSHOT["board"][0][1] = {"side": "red", "hidden": False,
                                    "home": "兵", "ident": "兵"}
BLACK_AI_SNAPSHOT["board"][9][1] = {"side": "black", "hidden": False,
                                    "home": "卒", "ident": "卒"}
BLACK_AI_SNAPSHOT["board"][9][4] = {"side": "black", "hidden": False,
                                    "home": "将", "ident": "将"}


class TestFactsSheet(unittest.TestCase):
    """事实表：归属 / 明暗 / 被吃 / 威胁都必须由本地算准。"""

    @staticmethod
    def _ai_to_move(gs, plies):
        """走 ``plies`` 手后，轮到的那一方就是"AI 执的一方"（真实流程就是这样）。

        奇数手时轮到红方、偶数手时轮到黑方——测试里不用手写这个奇偶性，
        直接按当前轮到谁来决定 ai_color，避免造出"AI 执黑但轮到红走"的
        非真实局面。
        """
        return gui_color_to_side(gs.current_player)

    def test_facts_have_both_sides_and_no_ambiguity(self):
        gs = make_gs(plies=8)
        facts = llm_facts(gs, ai_color=self._ai_to_move(gs, 8))
        ai_color = self._ai_to_move(gs, 8)
        own_cn = "黑" if ai_color == 1 else "红"
        other_cn = "红" if ai_color == 1 else "黑"
        self.assertEqual(facts["ai_color"], "black" if ai_color == 1 else "red")
        self.assertEqual(facts["opponent_color"], "red" if ai_color == 1 else "black")
        for key in ("own", "opponent"):
            block = facts[key]
            self.assertIn("revealed", block)
            self.assertIn("hidden", block)
            self.assertIn("hidden_count", block)
            self.assertIn("captured", block)
            self.assertIn("value", block)
        # 我方的明子只能是我方字形，对方的只能是对方字形
        for label in facts["own"]["revealed"]:
            self.assertTrue(label.startswith(own_cn), label)
        for label in facts["opponent"]["revealed"]:
            self.assertTrue(label.startswith(other_cn), label)
        # 暗子必须带问号，且写明"真实兵种未知"
        for label in facts["own"]["hidden"]:
            self.assertIn("?", label)
        self.assertIn("未知", facts["own"]["hidden_note"])

    def test_facts_text_lists_every_piece_with_owner(self):
        gs = make_gs(plies=6)
        ai_color = self._ai_to_move(gs, 6)
        own_cn = "黑" if ai_color == 1 else "红"
        other_cn = "红" if ai_color == 1 else "黑"
        facts = llm_facts(gs, ai_color=ai_color)
        text = render_facts_text(facts)
        self.assertIn("局面事实表", text)
        self.assertIn("归属铁律", text)
        self.assertIn("我方(%s方)的明子" % own_cn, text)
        self.assertIn("对方(%s方)的明子" % other_cn, text)
        self.assertIn("尚未翻开，真实兵种未知", text)
        self.assertIn("被吃子总账", text)
        self.assertIn("攻击关系", text)

    def test_every_board_piece_appears_exactly_once(self):
        """棋盘上每一枚子都要在事实表里出现，且只归到一方。

        坐标写法是 ``b7`` 这套（列字母 + 行号 0~9）。
        """
        import re

        gs = make_gs(plies=10)
        facts = llm_facts(gs, ai_color=self._ai_to_move(gs, 10))
        squares = set()
        for block in (facts["own"], facts["opponent"]):
            for label in block["revealed"] + block["hidden"]:
                for item in re.findall(r"[a-i][0-9]", label):
                    squares.add(item)
        board_squares = {"%s%d" % ("abcdefghi"[c], 9 - r)
                         for r in range(10) for c in range(9) if gs.board[r][c]}
        self.assertEqual(squares, board_squares)

    def test_prompt_carries_facts_and_move_annotations(self):
        gs = make_gs(plies=9, seed=11)      # 奇数手 -> 轮到黑方（AI 执黑）
        self.assertEqual(gs.current_player, "black")
        snap = build_llm_snapshot(gs, analyse(gs), ai_color=1)
        self.assertEqual(snap["side_to_move"], "black",
                         "快照的走棋方与游戏状态不一致：%s vs %s"
                         % (snap["side_to_move"], gs.current_player))
        text = llm_mod.render_position_text(snap)
        # 事实表进了提示词
        self.assertIn("局面事实表", text)
        # 棋盘图每行自带列号（模型不必回头数字）
        self.assertIn("列号:", text)
        self.assertIn("行0", text)
        # 候选着法写明了"谁走哪枚子到哪里"
        self.assertIn("走到", text)
        self.assertIn("本方(黑方，即 AI 执的一方)", text)
        # 走棋记录标了归属
        self.assertIn("[我方]", text)
        self.assertIn("[对方]", text)
        # 不能落进"其它字段"（说明快照键已登记）
        self.assertNotIn("其它字段", text)

    def test_top_candidate_is_named_explicitly(self):
        gs = make_gs(plies=9, seed=11)
        self.assertEqual(gs.current_player, "black")
        result = analyse(gs)
        snap = build_llm_snapshot(gs, result, ai_color=1)
        text = llm_mod.render_position_text(snap)
        self.assertIn("本次点评的这手棋", text)
        # 用快照里实际的首选着法去对（引擎每次跑的时间点略有不同，
        # 不能拿另一次分析的结果当基准）。
        top = snap["engine_candidates"][0]
        coord = "%s%d%s%d" % ("abcdefghi"[top["from"][1]], 9 - top["from"][0],
                              "abcdefghi"[top["to"][1]], 9 - top["to"][0])
        self.assertIn(coord, text)
        # 候选 1 的自然语言注解必须写明它是本方的子
        self.assertIn("本方", text)


class TestCaptureAccounting(unittest.TestCase):
    """被吃棋子：明子记身份，暗子只记数量（规则上不得翻看）。"""

    def test_known_and_unknown_captures(self):
        gs = make_gs()
        # 造两条吃子记录：一枚明子、一枚暗子
        gs.move_history = [
            {"piece": {"color": "red", "revealed": True, "actual": "车"},
             "start": (0, 0), "end": (1, 0),
             "captured": {"color": "black", "revealed": True, "actual": "卒"},
             "revealed_as": None},
            {"piece": {"color": "black", "revealed": True, "actual": "馬"},
             "start": (9, 0), "end": (8, 0),
             "captured": {"color": "red", "revealed": False, "actual": "炮",
                          "original": "炮"},
             "revealed_as": None},
        ]
        gs.ply = 2
        captured = track_captures(gs.move_history, 2)["lost"]
        self.assertEqual(captured[1]["known"], 1)
        self.assertEqual(captured[1]["hidden"], 0)
        self.assertEqual(captured[0]["hidden"], 1)
        self.assertEqual(captured[0]["known"], 0)
        self.assertIn("黑卒", captured[1]["text"])
        self.assertIn("身份对双方都未知", captured[0]["text"])

        items = captured_gui_pieces(gs, "red", 2)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["unknown"])
        self.assertIsNone(items[0]["piece"], "被吃的暗子不许带真身")

    def test_captures_respect_ply(self):
        gs = make_gs(plies=20)
        self.assertTrue(any(rec.get("captured") for rec in gs.move_history),
                        "样本里没有吃子，测试不成立")
        self.assertEqual(captured_gui_pieces(gs, "red", 0), [])
        self.assertEqual(track_captures(gs.move_history, 0)["lost"][0]["known"], 0)


class TestOwnershipValidator(unittest.TestCase):
    """幻觉质检：只抓可证明的矛盾，并且能安全地自动改判。"""

    def test_piece_belongs_to(self):
        for ch in "帅仕相马车炮兵":
            self.assertEqual(llm_mod.piece_belongs_to(ch), 0, ch)
        for ch in "将士象馬車砲卒":
            self.assertEqual(llm_mod.piece_belongs_to(ch), 1, ch)
        self.assertIsNone(llm_mod.piece_belongs_to("我"))

    def test_scan_side_piece_finds_every_writing_style(self):
        cases = {
            "我方车": ("我方", "车"),
            "我方的车": ("我方", "车"),
            "对方的马能吃": ("对方", "马"),
            "黑方卒顶上来": ("", "卒"),
            "红帅被将": ("", "帅"),
        }
        for text, (word, piece) in cases.items():
            hits = llm_mod.scan_side_piece(text)
            self.assertTrue(hits, text)
            self.assertEqual(hits[0]["piece"], piece, text)
            if word:
                self.assertEqual(hits[0]["word"], word, text)

    def test_catches_wrong_owner(self):
        issues = llm_mod.check_comment("我方车从中路杀出，吃掉对方的兵。",
                                       BLACK_AI_SNAPSHOT)
        self.assertTrue(issues, "把对方的车说成我方的，必须被抓出来")
        self.assertIn("车", "".join(issues))

    def test_accepts_correct_owner(self):
        clean = [
            "我方的卒顶上去，对方的车被牵制。",
            "对方车压过来，我的卒只能退。",
            "这步棋稳住中路，先手在我。",
            "黑方卒顶上去换掉红车。",
        ]
        for text in clean:
            self.assertEqual(llm_mod.check_comment(text, BLACK_AI_SNAPSHOT), [],
                             f"正确的话被误判了：{text}")

    def test_swapped_sides_are_repaired(self):
        text = "我方的车被对方的卒挡住了，对方的卒很强。"
        issues = llm_mod.check_comment(text, BLACK_AI_SNAPSHOT)
        self.assertTrue(issues)
        fixed, notes = llm_mod.repair_comment(text, BLACK_AI_SNAPSHOT)
        self.assertTrue(notes)
        self.assertIn("对方的车", fixed)
        self.assertIn("我方的卒", fixed)
        # 改判之后必须干净
        self.assertEqual(llm_mod.check_comment(fixed, BLACK_AI_SNAPSHOT), [])

    def test_unrepairable_comment_is_dropped(self):
        text = "我方帅已经安全了。"          # 帅是红方的，而 AI 执黑
        check = llm_mod.validate_comment(text, BLACK_AI_SNAPSHOT)
        self.assertFalse(check["ok"])
        self.assertEqual(check["text"], "", "改不回来时应当丢弃而不是照贴")
        self.assertTrue(check["issues"])

    def test_no_basis_means_no_judgement(self):
        """连AI执哪方都不知道时，宁可什么都不说。"""
        self.assertEqual(llm_mod.check_comment("我方车", {}), [])


class TestAdvisorValidation(unittest.TestCase):
    """LLMAdvisor 层：suggest_checked 会自动改判 / 定向重问。"""

    def test_clean_comment_passes_through(self):
        advisor = llm_mod.LLMAdvisor(fake_config())
        with mock.patch.object(llm_mod.LLMAdvisor, "suggest",
                               return_value=fake_result("我方的卒顶上去。")):
            result = advisor.suggest_checked(BLACK_AI_SNAPSHOT)
        self.assertEqual(result.comment, "我方的卒顶上去。")

    def test_swap_is_repaired_without_second_request(self):
        advisor = llm_mod.LLMAdvisor(fake_config())
        with mock.patch.object(llm_mod.LLMAdvisor, "suggest",
                               return_value=fake_result(
                                   "我方的车被对方的卒挡住了。")) as mocked:
            result = advisor.suggest_checked(BLACK_AI_SNAPSHOT)
        self.assertEqual(mocked.call_count, 1, "能自动改判时不该多花一次请求")
        self.assertIn("对方的车", result.comment)
        self.assertEqual(llm_mod.check_comment(result.comment, BLACK_AI_SNAPSHOT), [])

    def test_retry_with_hint_when_not_repairable(self):
        advisor = llm_mod.LLMAdvisor(fake_config())
        calls = []

        def fake_suggest(*args, **kwargs):
            # mock.patch.object 把普通函数挂上去后不会自动绑 self，
            # 所以这里要容忍"第一个参数是 advisor 实例"的调用形态。
            if args and isinstance(args[0], llm_mod.LLMAdvisor):
                args = args[1:]
            extra_system = kwargs.get("extra_system")
            if extra_system is None and len(args) > 2:
                extra_system = args[2]
            calls.append(extra_system or "")
            if len(calls) == 1:
                return fake_result("我方帅已经安全了。")     # 帅是红方的
            return fake_result("我方的卒顶上去，对方的车被牵制。")

        with mock.patch.object(llm_mod.LLMAdvisor, "suggest", fake_suggest):
            result = advisor.suggest_checked(BLACK_AI_SNAPSHOT)
        self.assertEqual(len(calls), 2, "第一次不合格时必须定向重问一次")
        self.assertTrue(calls[1], "重问要带上'上一次哪里错了'的提示")
        self.assertIn("卒", result.comment)

    def test_gives_up_and_clears_comment(self):
        advisor = llm_mod.LLMAdvisor(fake_config())
        with mock.patch.object(llm_mod.LLMAdvisor, "suggest",
                               return_value=fake_result("我方帅已经安全了。")):
            result = advisor.suggest_checked(BLACK_AI_SNAPSHOT)
        self.assertEqual(result.comment, "",
                         "两次都改不回来时应清空点评，绝不能贴出误导性文字")
        self.assertEqual(result.parsed.get("comment_original"), "我方帅已经安全了。")

    def test_validation_can_be_disabled(self):
        advisor = llm_mod.LLMAdvisor(fake_config(comment_validation=False))
        with mock.patch.object(llm_mod.LLMAdvisor, "suggest",
                               return_value=fake_result("我方帅已经安全了。")):
            result = advisor.suggest_checked(BLACK_AI_SNAPSHOT)
        self.assertEqual(result.comment, "我方帅已经安全了。")


class TestControllerWiring(unittest.TestCase):
    """AI 控制器：面板上显示的点评必须已经过质检。"""

    SETTINGS = {
        "difficulty": "easy", "time_limit": 0.25, "samples": 1,
        "max_depth": 3, "seed": 7, "emit_decision_tree": False,
        "verbose": False, "weights": {}, "use_hidden_captures": False,
    }

    def _run(self, comment):
        import AI as ai_module

        gs = make_gs(plies=4)
        cfg = fake_config()
        with mock.patch.object(ai_module, "ai_settings",
                               return_value=dict(self.SETTINGS)), \
                mock.patch.object(ai_module, "load_llm_config", return_value=cfg), \
                mock.patch.object(llm_mod.LLMAdvisor, "suggest",
                                  return_value=fake_result(comment)), \
                mock.patch.object(
                    llm_mod.LLMAdvisor, "suggest_async",
                    lambda self, snapshot, callback=None, style=None, validate=False:
                    callback(fake_result(comment))):
            controller = ai_module.AdvancedAIController(gs)
            controller.llm_enabled = True
            controller.analyse()
        return controller

    def test_warning_appended_for_contradicting_comment(self):
        controller = self._run("我方帅已经安全了。")
        self.assertIn("!!", controller.last_comment)
        self.assertIn("我方帅", controller.last_comment)

    def test_clean_comment_shown_as_is(self):
        controller = self._run("我方的卒顶上去，控制中路。")
        self.assertIn("卒顶上去", controller.last_comment)
        self.assertNotIn("⚠", controller.last_comment)


if __name__ == "__main__":
    unittest.main()
