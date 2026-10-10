# -*- coding: utf-8 -*-
"""自对弈 A/B：同一套引擎代码，只切换某个特性，看谁赢（棋力回归用）。

用法::

    # 评估缓存：开 vs 关
    python _bench/ab_selfplay.py --games 8 --time 0.25 --a-cache --b-no-cache

    # 将军延伸：开 vs 关
    python _bench/ab_selfplay.py --games 8 --time 0.25 --a-ext --b-no-ext

    # 采样数：1 vs 2（同样的总时间，采样越多深度越浅）
    python _bench/ab_selfplay.py --games 8 --time 0.25 --a-samples 1 --b-samples 2

设计：两边都用**同一份** jieqi 引擎，只是 options 不同，双方轮流执红，
所以任何一方都不占"先手"或"代码版本"的便宜。默认每步时间很短（0.25s），
目的是在可接受的时间里跑出统计上还能看的结果。
"""
from __future__ import annotations

import argparse
import os
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

from jieqi.board import encode  # noqa: E402
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    captured_info, gui_board_to_engine, gui_color_to_side, legal_moves,
    repetition_counts,
)


def build_options(args, side: str) -> dict:
    """按命令行给某一方组装引擎参数。"""
    opts: dict = {"eval_cache": True, "extensions": True}
    samples = args.samples
    if side == "a":
        if args.a_no_cache:
            opts["eval_cache"] = False
        if args.a_no_ext:
            opts["extensions"] = False
        if args.a_samples:
            samples = args.a_samples
    else:
        if args.b_no_cache:
            opts["eval_cache"] = False
        if args.b_no_ext:
            opts["extensions"] = False
        if args.b_samples:
            samples = args.b_samples
    opts["samples"] = samples
    return opts


def play_one(args, seed: int, a_side: str):
    """下一盘：a 方执 ``a_side``。返回 (winner, plies, 每方平均深度)。"""
    from game_state import GameState

    random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.ai_mode = True
    gs.ai_color = "black"

    engines = {}
    for side in ("red", "black"):
        opts = build_options(args, "a" if side == a_side else "b")
        engines[side] = Engine("normal", time_limit=args.time,
                               samples=opts["samples"], seed=seed,
                               eval_cache=opts["eval_cache"],
                               extensions=opts["extensions"])
    depths = {"a": [], "b": []}
    plies = 0
    for _ in range(args.max_plies):
        if gs.game_over:
            break
        side = gs.current_player
        turn = gui_color_to_side(side)
        known, hidden = captured_info(gs.move_history)
        board = gui_board_to_engine(gs.board, turn=turn)
        result = engines[side].analyse(
            board, history_counts=repetition_counts(gs),
            own_captured=known, captured_hidden=hidden)
        if result.move is None:
            gs.game_over = True
            gs.winner = "black" if side == "red" else "red"
            break
        who = "a" if side == a_side else "b"
        depths[who].append(result.depth)
        frm, to = result.move >> 7, result.move & 127
        gs.move_piece(frm // 9, frm % 9, to // 9, to % 9)
        gs.current_player = "black" if side == "red" else "red"
        gs.check_game_state()
        plies += 1
    avg = {k: (sum(v) / len(v) if v else 0.0) for k, v in depths.items()}
    return gs.winner or "draw", plies, avg


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--time", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--a-no-cache", action="store_true")
    parser.add_argument("--b-no-cache", action="store_true")
    parser.add_argument("--a-no-ext", action="store_true")
    parser.add_argument("--b-no-ext", action="store_true")
    parser.add_argument("--a-samples", type=int, default=0)
    parser.add_argument("--b-samples", type=int, default=0)
    args = parser.parse_args(argv)

    wins = {"a": 0, "b": 0, "draw": 0}
    depth_sum = {"a": 0.0, "b": 0.0}
    t0 = time.monotonic()
    for g in range(args.games):
        a_side = "red" if g % 2 == 0 else "black"
        winner, plies, avg = play_one(args, args.seed + g, a_side)
        depth_sum["a"] += avg["a"]
        depth_sum["b"] += avg["b"]
        if winner == "draw":
            wins["draw"] += 1
            label = "和"
        elif (winner == "red") == (a_side == "red"):
            wins["a"] += 1
            label = "A 胜"
        else:
            wins["b"] += 1
            label = "B 胜"
        print("第 %2d 局：A 执%s  %s  %3d 手  平均深度 A=%.1f B=%.1f  用时 %.0fs"
              % (g + 1, "红" if a_side == "red" else "黑", label, plies,
                 avg["a"], avg["b"], time.monotonic() - t0))
    print("")
    print("总计 %d 局：A %d 胜 / B %d 胜 / 和 %d"
          % (args.games, wins["a"], wins["b"], wins["draw"]))
    print("平均深度：A %.2f  B %.2f" % (depth_sum["a"] / args.games,
                                      depth_sum["b"] / args.games))
    print("（A/B 只是同一引擎的两组开关，样本少时波动很大，看趋势不看单局。）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
