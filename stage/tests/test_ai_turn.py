# -*- coding: utf-8 -*-
"""AI 走子时机测试：**不能连走两步**（用户真实报的 bug）。

复现场景（用户描述，已实测重现）：
    AI 走完 → 玩家点「上一步」→ 点「下一步」回到最新 → **AI 又走了一步**。

根因：``GameState.goto_ply`` 会把 ``current_player`` 恢复成**快照里存的那个值**
（= "当时轮到谁"）。所以从历史回到最新时，``current_player`` 会等于
"刚走完的那一方"，而 ``_is_ai_turn()`` 只看"轮到 AI 且不在浏览历史"，
于是把"回到最新"误判成"轮到 AI 了"。

修复：加一条判据——**轮到的一方不能就是刚走完的那一方**。
这条不变量在任何正常对局里都成立，所以拿它当守卫是安全的。
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


def make_game(plies: int = 4, ai_color: str = "black", seed: int = 5):
    """建一个"人机对战"的 ChessGame 外壳（不跑 __init__，只借它的方法）。"""
    import main as M
    from game_state import GameState

    gs = GameState(None)
    gs.show_side_selection = False
    gs.ai_mode = True
    gs.ai_color = ai_color
    gs.player_side = "red" if ai_color == "black" else "black"
    game = M.ChessGame.__new__(M.ChessGame)
    game.gs = gs
    game.ai_controller = None
    game.ai_error = None
    game.ai_pending_move = None
    rng = random.Random(seed)
    for _ in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.current_player = "black" if gs.current_player == "red" else "red"
        gs.check_game_state()
    return game


class TestLastMover(unittest.TestCase):
    def test_last_mover_reads_the_move_at_current_ply(self):
        game = make_game(4)
        self.assertEqual(game._last_mover(), "black")   # 第 4 手是黑走的

    def test_last_mover_is_none_at_opening(self):
        game = make_game(0)
        self.assertIsNone(game._last_mover())

    def test_last_mover_after_goto(self):
        """浏览到第 3 手 → 刚走完的是红（第 3 手是红）。"""
        game = make_game(4)
        game.gs.goto_ply(3)
        self.assertEqual(game._last_mover(), "red")


class TestNoDoubleMove(unittest.TestCase):
    def test_normal_alternation_never_lets_ai_move_twice(self):
        """正常交替里，AI 回合结束后**不能**马上又是 AI 回合。"""
        game = make_game(0)
        gs = game.gs
        rng = random.Random(9)
        for _ in range(6):
            if game.can_ai_move():
                # AI 走一步（这里用"随便一个合法着法"代替真实搜索）
                side = gui_color_to_side(gs.current_player)
                mv = legal_moves(gs.board, side)[0]
                gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                              (mv & 127) // 9, (mv & 127) % 9)
                gs.current_player = ("black" if gs.current_player == "red"
                                     else "red")
                gs.check_game_state()
                self.assertFalse(game.can_ai_move(),
                                 "AI 刚走完就又能走（连走两步）")
            else:
                side = gui_color_to_side(gs.current_player)
                moves = legal_moves(gs.board, side)
                if not moves:
                    break
                mv = rng.choice(moves)
                gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                              (mv & 127) // 9, (mv & 127) % 9)
                gs.current_player = ("black" if gs.current_player == "red"
                                     else "red")
                gs.check_game_state()

    def test_back_and_forward_does_not_hand_the_turn_to_ai(self):
        """用户报的 bug：上一步 + 下一步 之后，不能把回合"送"给 AI。

        ⚠ 这里的前提在后续修复里变了：``goto_ply`` 现在从**走法历史**推轮次
        （``_side_to_move_at``），所以回到最新后 ``current_player`` 是**该谁走**
        （红），而不是旧代码里那个"刚走完的那一方"（黑）。测试改成断言
        **修复后的正确行为**，同时保留"AI 不该走"这条不变式。
        """
        game = make_game(4)
        gs = game.gs
        self.assertFalse(game.can_ai_move(), "前提：现在轮到红，AI 不该走")
        gs.goto_ply(gs.ply - 1)
        gs.goto_ply(gs.ply + 1)
        self.assertEqual(gs.ply, 4)
        self.assertEqual(gs.current_player, "red",
                         "回到最新后轮次应当是「该谁走」（红），"
                         "而不是旧代码里那个「刚走完的那一方」")
        self.assertEqual(game._last_mover(), "black", "第 4 手是黑走的")
        self.assertFalse(game.can_ai_move(),
                         "回到最新后 AI 又想走 —— 这正是连走两步的 bug")

    def test_ai_still_moves_when_it_is_really_its_turn(self):
        """别修过头：真轮到 AI 时必须还能走。

        正常交替下"轮到黑"意味着"红刚走完"，所以造一个这样的局面。
        """
        import random as _random

        game = make_game(3)          # 第 3 手是红走的 → 轮到黑
        gs = game.gs
        self.assertEqual(gs.current_player, "black")
        self.assertEqual(game._last_mover(), "red")
        self.assertTrue(game.can_ai_move(),
                        "真轮到 AI 了却不让它走")

    def test_ai_does_not_move_while_browsing(self):
        game = make_game(4)
        game.gs.goto_ply(2)
        self.assertTrue(game.gs.is_browsing)
        self.assertFalse(game.can_ai_move() or game._is_ai_turn(),
                         "浏览历史时 AI 不该走")


if __name__ == "__main__":
    unittest.main()
