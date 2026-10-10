# -*- coding: utf-8 -*-
"""导航与轮次测试：**退回历史后，"该谁走"必须正确**。

用户报的 bug（两次描述，第二次才点中根因）：
    开局 → 红走一步 → 黑走一步 → 点「上一步」
    → 界面**提示轮到红方**，于是红方又走一步 = 红连走两步。

根因：``GameState.goto_ply`` 以前直接读快照里的 ``current_player``，而快照是在
``move_piece`` 里"走完这步之后"存的，那一刻 ``current_player`` 还是**刚走完的
那一方**（切换是调用方之后才做的）。所以退回一步后，轮次显示成了"刚走完的人"。

修法：轮次从**走法历史**推——红先、之后逐手交替，即 `next = last_mover 的对面`。

用户还要求：按钮按**回合**翻（红黑各一手算一回合），所以按钮改成
「上一回合 / 下一回合」，键盘 ←→ 同步如此。
"""
from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
for _p in (str(GAME_DIR.parent), str(GAME_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pygame  # noqa: E402

from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402


def setUpModule():
    pygame.init()
    pygame.display.set_mode((1, 1))


def make_gs(rounds: int = 3, seed: int = 5):
    """走 ``rounds`` 个回合（红黑各一手）。"""
    from game_state import GameState

    gs = GameState(None)
    gs.show_side_selection = False
    rng = random.Random(seed)
    for _ in range(rounds * 2):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.current_player = "black" if gs.current_player == "red" else "red"
        gs.check_game_state()
    return gs


class TestSideToMoveAfterNavigation(unittest.TestCase):
    def test_opening_is_red(self):
        gs = make_gs(0)
        self.assertEqual(gs._side_to_move_at(0), "red")
        self.assertEqual(gs._last_mover_at(0), None)

    def test_odd_ply_hands_the_turn_to_black(self):
        """第 1 手是红走的 → 该黑走（这正是用户看到"轮到红方"的那个错误）。"""
        gs = make_gs(3)
        self.assertEqual(gs._last_mover_at(1), "red")
        self.assertEqual(gs._side_to_move_at(1), "black")

    def test_even_ply_hands_the_turn_to_red(self):
        gs = make_gs(3)
        self.assertEqual(gs._last_mover_at(2), "black")
        self.assertEqual(gs._side_to_move_at(2), "red")

    def test_goto_ply_sets_the_correct_side(self):
        """用户场景：红走 → 黑走 → 上一步，轮次必须是**黑**。"""
        gs = make_gs(1)                    # 红一手 + 黑一手
        self.assertEqual(gs.ply, 2)
        self.assertEqual(gs.current_player, "red")
        gs.goto_ply(1)
        self.assertEqual(gs.current_player, "black",
                         "退回一手后提示仍轮到红方 —— 用户报的就是这个")
        gs.goto_ply(0)
        self.assertEqual(gs.current_player, "red")
        gs.goto_ply(2)
        self.assertEqual(gs.current_player, "red")

    def test_every_ply_has_the_right_side(self):
        gs = make_gs(4)
        total = gs.total_plies
        for ply in range(total + 1):
            gs.goto_ply(ply)
            self.assertEqual(gs.current_player, gs._side_to_move_at(ply),
                             "ply=%d 的轮次不对" % ply)


class TestRoundNavigation(unittest.TestCase):
    """按钮按回合翻（红黑各一手 = 一回合）。"""

    def _game(self, rounds: int = 3):
        import main as M

        gs = make_gs(rounds)
        game = M.ChessGame.__new__(M.ChessGame)
        game.gs = gs
        game.ai_controller = None
        game.ai_error = None
        game.ai_pending_move = None
        return game

    def test_prev_goes_back_one_round(self):
        game = self._game(3)
        self.assertEqual(game.gs.ply, 6)
        game.goto_round(-1)
        self.assertEqual(game.gs.ply, 4)
        game.goto_round(-1)
        self.assertEqual(game.gs.ply, 2)
        game.goto_round(-1)
        self.assertEqual(game.gs.ply, 0)

    def test_next_goes_forward_one_round(self):
        game = self._game(3)
        game.goto_ply(0)
        game.goto_round(+1)
        self.assertEqual(game.gs.ply, 2)
        game.goto_round(+1)
        self.assertEqual(game.gs.ply, 4)

    def test_round_navigation_stops_at_the_ends(self):
        game = self._game(3)
        game.goto_ply(0)
        game.goto_round(-1)                # 已经在开局
        self.assertEqual(game.gs.ply, 0)
        game.goto_ply(game.gs.total_plies)
        game.goto_round(+1)                # 已经在最新
        self.assertEqual(game.gs.ply, 6)

    def test_round_steps_land_on_even_plies(self):
        """落在整回合上（偶数手），这样"该谁走"永远是红。"""
        game = self._game(4)
        game.goto_ply(game.gs.total_plies)
        for _ in range(3):
            game.goto_round(-1)
            if game.gs.ply != 0:
                self.assertEqual(game.gs.ply % 2, 0,
                                 "回合翻没有落在整回合上：ply=%d" % game.gs.ply)
                self.assertEqual(game.gs.current_player, "red",
                                 "整回合落点上应该轮到红方")


class TestNoDoubleMoveAfterNavigation(unittest.TestCase):
    def test_ai_would_not_move_after_browsing_back_and_forth(self):
        """回到最新后，若刚走完的是 AI，就**不该**再让 AI 走。"""
        import main as M

        gs = make_gs(2)                     # 红一手 + 黑一手（AI 执黑）
        gs.ai_mode = True
        gs.ai_color = "black"
        game = M.ChessGame.__new__(M.ChessGame)
        game.gs = gs
        game.ai_controller = None
        game.ai_error = None
        game.ai_pending_move = None
        self.assertFalse(game.can_ai_move(), "前提：轮到红，AI 不该走")
        game.goto_round(-1)
        game.goto_round(+1)
        self.assertEqual(gs.current_player, "red")
        self.assertFalse(game.can_ai_move(),
                         "回到最新后 AI 又想走 —— 连走两步")


if __name__ == "__main__":
    unittest.main()
