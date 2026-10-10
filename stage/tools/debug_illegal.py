"""调试：多处种子下寻找「旧 AI 走出新引擎认为非法」的着法并打印全部细节。"""
import os
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
for p in (str(STAGE), str(STAGE / "Game")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from jieqi import board as B  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    board_text, gui_board_to_engine, gui_color_to_side, legal_moves,
)
from tools.selfplay import LegacyPlayer, NewEnginePlayer, new_gamestate  # noqa: E402

SETTINGS = {"difficulty": "normal", "time_limit": 0.8, "samples": 4,
            "emit_decision_tree": False}


def run_game(seed, verbose_illegal=True):
    gs = new_gamestate(seed)
    new = NewEnginePlayer(gs, "red", settings=SETTINGS)
    legacy = LegacyPlayer(gs, "black")
    for ply in range(80):
        side = gs.current_player
        player = new if side == "red" else legacy
        move = player.get_best_move()
        if move is None:
            return ("no_move", ply, side, None)
        (sr, sc), (er, ec) = move
        sb = gui_color_to_side(side)
        eb = gui_board_to_engine(gs.board, turn=sb)
        mv = B.encode(sr * 9 + sc, er * 9 + ec)
        if not eb.is_legal(mv):
            print(f"!!! seed={seed} ply={ply} {side}({player.name}) {B.move_str(mv)}")
            print(f"    pseudo={mv in eb.gen_pseudo(sb)} "
                  f"gen_moves={mv in eb.gen_moves(sb)} "
                  f"piece={gs.board[sr][sc]}")
            print(f"    engine kings={eb.kings} in_check red={eb.in_check(0)} black={eb.in_check(1)}")
            print(f"    engine_moves={len(eb.gen_moves(sb))} gui_moves={len(legal_moves(gs.board, sb))}")
            gui_mv = [(B.move_str(m), gs.board[(m >> 7) // 9][(m >> 7) % 9])
                      for m in legal_moves(gs.board, sb)]
            eng_mv = [B.move_str(m) for m in eb.gen_moves(sb)]
            print(f"    仅 GUI 认为合法: {[x[0] for x in gui_mv if x[0] not in eng_mv][:12]}")
            print(f"    仅引擎认为合法: {[m for m in eng_mv if m not in [x[0] for x in gui_mv]][:12]}")
            print("引擎棋盘：")
            print(eb.to_text())
            print("GUI 棋盘：")
            print(board_text(gs.board, turn=sb))
            return ("illegal", ply, side, B.move_str(mv))
        gs.move_piece(sr, sc, er, ec)
        gs.current_player = "black" if side == "red" else "red"
        gs.check_game_state()
        if gs.game_over:
            return ("over", ply, side, gs.winner)
    return ("timeout", 80, None, None)


def main():
    seeds = [int(x) for x in sys.argv[1:]] or list(range(1, 5))
    for seed in seeds:
        out = run_game(seed)
        print(f"seed={seed} -> {out}")


if __name__ == "__main__":
    main()
