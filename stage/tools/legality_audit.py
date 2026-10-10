"""合法性审计：用 **GUI 自己的规则** 独立复核 AI 的每一步。

为什么需要它：AI 落子走的是 `main._consume_ai_move()` → `gs.move_piece()`，
**完全绕过了 input_handler 里的合法性校验**。只要引擎和 GUI 对规则的理解
有任何分歧（或引擎合法性过滤有漏洞），违规着法就会直接落到棋盘上，
表现就是"AI 送将"。这个脚本用两套独立实现交叉验证：

1. `gs.validate_move` + `gs._check_check` + `gs._check_face_to_face`（GUI 规则）
2. 引擎的 `Board.gen_moves`（搜索用的规则）

并且每一步之后都断言：**刚走完的一方，自己的将帅不能处于被攻击状态**
（送将/自将），以及对局不能出现"双方同时被将"这种不可能的局面。
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
for p in (str(GAME_DIR.parent), str(GAME_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pygame  # noqa: E402

from jieqi import board as B  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    gui_board_to_engine, gui_color_to_side, legal_moves,
)

SETTINGS = {"difficulty": "normal", "time_limit": 0.25, "samples": 2,
            "max_depth": 4, "emit_decision_tree": False, "seed": None}


def make_gs(seed=None):
    from game_state import GameState

    if seed is not None:
        random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.ai_mode = True
    gs.ai_color = "black"
    return gs


def gui_reason_rejected(gs, sr, sc, er, ec):
    """按 GUI 的规则复核这一步；合法返回 None，否则返回拒绝原因。"""
    import copy

    piece = gs.board[sr][sc]
    if not piece:
        return "起点没有棋子"
    if piece['color'] != gs.current_player:
        return f"不是{gs.current_player}的棋子"
    target = gs.board[er][ec]
    if target and target['color'] == piece['color']:
        return "吃自己的子"
    if not gs.validate_move(sr, sc, er, ec):
        return "GUI 走法验证不通过"
    temp = copy.deepcopy(gs.board)
    temp[er][ec] = copy.deepcopy(piece)
    temp[sr][sc] = None
    tr, tb = gs._get_king_positions(temp)
    if gs.in_check and gs._check_check(temp, gs.current_player, tr, tb):
        return "被将军却没有应将"
    if gs._check_check(temp, gs.current_player, tr, tb):
        return "送将（走完自己被将）"
    if gs._check_face_to_face(temp, tr, tb):
        return "走成对面笑"
    return None


def audit_game(seed, ai_side="black", max_plies=200, verbose=False):
    gs = make_gs(seed)
    gs.ai_color = ai_side
    import AI as ai_module

    ai_module.ai_settings = lambda path=None: dict(SETTINGS)
    controller = ai_module.AdvancedAIController(gs, difficulty="normal")

    problems = []
    for ply in range(max_plies):
        if gs.game_over:
            break
        side = gs.current_player
        gui_side = gui_color_to_side(side)
        moves = legal_moves(gs.board, gui_side)
        if not moves:
            gs.game_over = True
            gs.winner = "black" if side == "red" else "red"
            break

        if side == ai_side:
            controller.discard_inflight_llm()
            move = controller.get_best_move()
            if move is None:
                problems.append((seed, ply, side, None, "AI 没有给出着法"))
                break
            (sr, sc), (er, ec) = move
            source = "AI"
        else:
            mv = random.Random(seed * 1000 + ply).choice(moves)
            sr, sc = (mv >> 7) // 9, (mv >> 7) % 9
            er, ec = (mv & 127) // 9, (mv & 127) % 9
            source = "随机"

        reason = gui_reason_rejected(gs, sr, sc, er, ec)
        # 引擎自己也复核一遍
        eng = gui_board_to_engine(gs.board, turn=gui_side)
        mv_int = B.encode(sr * 9 + sc, er * 9 + ec)
        eng_ok = eng.is_legal(mv_int)
        if reason or not eng_ok:
            problems.append((seed, ply, side, B.move_str(mv_int),
                             f"{source} 违规: gui={reason} engine_legal={eng_ok}"))
            break

        # 落子后：走完的一方自己的将帅绝不能被攻击
        gs.move_piece(sr, sc, er, ec)
        after = gui_board_to_engine(gs.board, turn=gui_side)
        if after.kings[gui_side] >= 0 and after.is_attacked(after.kings[gui_side],
                                                           1 - gui_side):
            problems.append((seed, ply, side, B.move_str(mv_int),
                             f"{source} 走完后自己的将帅被攻击（送将）"))
            break
        both = (after.kings[0] >= 0 and after.kings[1] >= 0
                and after.is_attacked(after.kings[0], 1)
                and after.is_attacked(after.kings[1], 0))
        if both:
            problems.append((seed, ply, side, B.move_str(mv_int),
                             "双方同时被将（不可能局面）"))
            break

        gs.current_player = "black" if side == "red" else "red"
        gs.check_game_state()
        if verbose and ply % 20 == 0:
            print(f"    seed={seed} ply={ply} 进行中…")
    return problems


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    max_plies = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    pygame.init()
    pygame.display.set_mode((1, 1))
    all_problems = []
    for i in range(games):
        ai_side = "black" if i % 2 == 0 else "red"
        probs = audit_game(1000 + i, ai_side=ai_side, max_plies=max_plies)
        if probs:
            all_problems.extend(probs)
            for p in probs:
                print(f"  !! seed={p[0]} ply={p[1]} {p[2]} {p[3]} -> {p[4]}")
        else:
            print(f"  [{i + 1}/{games}] seed={1000 + i} AI执{ai_side} 通过")
    print()
    if all_problems:
        print(f"发现 {len(all_problems)} 处违规")
        return 1
    print(f"{games} 局全部通过：AI 每一步都同时满足 GUI 规则与引擎规则，且从不送将")
    return 0


if __name__ == "__main__":
    sys.exit(main())
