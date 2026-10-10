"""Baseline benchmark for the current (legacy) AI, run headless.

Usage:  python bench_old_ai.py [plies]
"""
import contextlib
import io
import os
import random
import sys
import time

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import pygame  # noqa: F401
    from game_state import GameState  # noqa: E402
    from AI import AdvancedAIController  # noqa: E402

    class FakeRM:
        config = None

    random.seed(12345)
    gs = GameState(FakeRM())
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.enable_ai("black")
    gs.ai_mode = True

    # Play a few random plies so the position is not the raw opening.
    plies = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    played = 0
    guard = 0
    while played < plies and guard < 400:
        guard += 1
        moves = []
        for r in range(10):
            for c in range(9):
                p = gs.board[r][c]
                if p and p["color"] == gs.current_player:
                    for nr in range(10):
                        for nc in range(9):
                            if gs.validate_move(r, c, nr, nc):
                                moves.append(((r, c), (nr, nc)))
        if not moves:
            break
        mv = random.choice(moves)
        gs.move_piece(mv[0][0], mv[0][1], mv[1][0], mv[1][1])
        gs.current_player = "black" if gs.current_player == "red" else "red"
        played += 1

    gs.ai_color = gs.current_player
    ai = AdvancedAIController(gs)
    t0 = time.time()
    best = ai.get_best_move()
    t1 = time.time()

sys.stderr.write("legacy AI depth=1 best_move=%s time=%.2fs stdout_lines=%d\n"
                 % (best, t1 - t0, len(buf.getvalue().splitlines())))
