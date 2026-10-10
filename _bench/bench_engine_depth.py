# -*- coding: utf-8 -*-
"""引擎体检：同样 2 秒能搜多深、节点率多少、为什么这么慢。

用法::

    python _bench/bench_engine_depth.py                 # 中局若干局面
    python _bench/bench_engine_depth.py --time 2 --samples 2
    python _bench/bench_engine_depth.py --profile       # 带 cProfile 热点
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import random
import sys
import time
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.stdout.reconfigure(encoding="utf-8")
import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from game_state import GameState  # noqa: E402
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    gui_board_to_engine, gui_color_to_side, legal_moves,
)


def positions(count: int = 6, plies: int = 26, seed: int = 5):
    """随机走出一盘，取其中几个局面当体检样本。"""
    random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.enable_ai("black")
    rng = random.Random(seed)
    out = []
    for k in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9, (mv & 127) // 9, (mv & 127) % 9)
        gs.current_player = "black" if side == 0 else "red"
        if k % max(1, plies // count) == 0:
            out.append((k, gs.current_player, gui_board_to_engine(
                gs.board, turn=gui_color_to_side(gs.current_player))))
    return out[:count]


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--difficulty", default="normal")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)

    samples = positions(args.count)
    print("难度=%s 时间=%.1fs 采样=%d 局面数=%d"
          % (args.difficulty, args.time, args.samples, len(samples)))
    print("%6s %-6s %6s %8s %8s %9s %8s" %
          ("手数", "轮到", "深度", "节点", "节点/秒", "耗时(s)", "最佳分"))
    total_nodes = total_time = 0
    for ply, side, board in samples:
        engine = Engine(args.difficulty, time_limit=args.time,
                        samples=args.samples, seed=3)
        t0 = time.perf_counter()
        if args.profile:
            pr = cProfile.Profile()
            pr.enable()
            result = engine.analyse(board)
            pr.disable()
        else:
            result = engine.analyse(board)
        elapsed = time.perf_counter() - t0
        total_nodes += result.nodes
        total_time += elapsed
        print("%6d %-6s %6d %8d %8.0f %9.2f %8.0f"
              % (ply, side, result.depth, result.nodes,
                 result.nodes / max(1e-6, elapsed), elapsed, result.score))
        if args.profile:
            stats = pstats.Stats(pr)
            stats.sort_stats("cumulative").print_stats(12)
            break
    print("合计 %d 节点 / %.1fs = %.0f 节点/秒"
          % (total_nodes, total_time, total_nodes / max(1e-6, total_time)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
