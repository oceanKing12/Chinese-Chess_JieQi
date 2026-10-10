# -*- coding: utf-8 -*-
"""体检：**静态评估 vs 深搜打分**差多少（决定"调参"值不值得做）。

思路（Texel 式调参的前提检查）：

1. 自对弈生成一批中局局面；
2. 每个局面取两个数：
   * ``deep``：深搜（固定深度，无采样噪声）给的分——当作"这一手有多好"的参考；
   * ``static``：**当前评估函数**直接给的分（不搜索）；
3. 如果 ``static`` 与 ``deep`` 的相关性很低，说明评估函数本身有系统性偏差，
   调它的权重/位置表收益就大；如果相关性已经很高，收益就有限。

同时统计各评估项的量级，看有没有"某一项把别的项压过去"的情况。

用法::

    python _bench/check_eval_accuracy.py --positions 120 --deep 6 --time 0.3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
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

from jieqi import board as B  # noqa: E402
from jieqi.belief import Belief  # noqa: E402
from jieqi.consts import BLACK, RED  # noqa: E402
from jieqi.engine import Engine  # noqa: E402
from jieqi.evaluate import EvalContext, evaluate  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    captured_info, gui_board_to_engine, gui_color_to_side, repetition_counts,
)


def spearman(pairs):
    """秩相关（不假设线性，能看出"单调性"）。"""
    if len(pairs) < 4:
        return float("nan")

    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    xs = rank([p[0] for p in pairs])
    ys = rank([p[1] for p in pairs])
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=120)
    parser.add_argument("--deep", type=int, default=6, help="参考搜深")
    parser.add_argument("--time", type=float, default=0.3)
    parser.add_argument("--plies", type=int, default=24, help="自对弈到第几手开始取样")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None, help="把样本写进 JSONL（给调参用）")
    args = parser.parse_args(argv)

    from game_state import GameState

    rows = []
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.enable_ai("black")
    gs.record_saved = "(probe)"
    engines = {"red": Engine("normal", time_limit=0.08, samples=1, seed=1),
               "black": Engine("normal", time_limit=0.08, samples=1, seed=2)}
    rng = random.Random(args.seed)

    while len(rows) < args.positions and gs.total_plies < args.plies * 4:
        # 走几步
        for _ in range(3):
            if gs.game_over:
                break
            side = gs.current_player
            known, hidden = captured_info(gs.move_history)
            board = gui_board_to_engine(gs.board, turn=gui_color_to_side(side))
            res = engines[side].analyse(board, history_counts=repetition_counts(gs),
                                        own_captured=known, captured_hidden=hidden)
            if res.move is None:
                break
            frm, to = res.move >> 7, res.move & 127
            gs.move_piece(frm // 9, frm % 9, to // 9, to % 9)
            gs.current_player = "black" if side == "red" else "red"
            gs.check_game_state()
        if gs.game_over:
            break
        side = gui_color_to_side(gs.current_player)
        pub = gui_board_to_engine(gs.board, turn=side).clone_revealed_only()
        if not pub.gen_moves(side):
            continue
        known, hidden = captured_info(gs.move_history)
        belief = Belief.from_board(pub, captured=known, captured_hidden=hidden)
        ctx = EvalContext(belief)
        static_red = evaluate(pub, ctx)
        deep_engine = Engine("normal", time_limit=args.time, samples=args.samples,
                             seed=args.seed + len(rows), max_depth=args.deep)
        deep = deep_engine.analyse(gui_board_to_engine(gs.board, turn=side),
                                   history_counts=repetition_counts(gs),
                                   own_captured=known, captured_hidden=hidden)
        # 统一成"红方视角"，同号可比
        rows.append({"static_red": static_red, "deep_red": float(deep.score),
                     "depth": int(deep.depth), "ply": int(gs.total_plies),
                     "side": side})
        if len(rows) % 20 == 0:
            print("  已取样 %d 个局面…" % len(rows))

    if not rows:
        print("没取到样本")
        return 2
    pairs = [(r["static_red"], r["deep_red"]) for r in rows]
    n = len(pairs)
    mean_s = sum(p[0] for p in pairs) / n
    mean_d = sum(p[1] for p in pairs) / n
    cov = sum((a - mean_s) * (b - mean_d) for a, b in pairs) / n
    sd_s = math.sqrt(sum((a - mean_s) ** 2 for a in (p[0] for p in pairs)) / n)
    sd_d = math.sqrt(sum((b - mean_d) ** 2 for b in (p[1] for p in pairs)) / n)
    pearson = cov / (sd_s * sd_d) if sd_s and sd_d else float("nan")
    rho = spearman(pairs)
    mae = sum(abs(a - b) for a, b in pairs) / n
    mate_like = sum(1 for _a, b in pairs if abs(b) > 400000)

    print("")
    print("样本 %d 个局面（深搜参考深度 %d）" % (n, args.deep))
    print("  静态评估：均值 %+.0f  标准差 %.0f" % (mean_s, sd_s))
    print("  深搜打分：均值 %+.0f  标准差 %.0f（其中 %d 个是杀棋分，已计入）"
          % (mean_d, sd_d, mate_like))
    print("  皮尔逊相关 %.3f   斯皮尔曼秩相关 %.3f" % (pearson, rho))
    print("  平均绝对偏差 %.0f 厘兵（= %.2f 个兵）" % (mae, mae / 100.0))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("  样本已写入 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
