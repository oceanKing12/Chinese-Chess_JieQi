# -*- coding: utf-8 -*-
"""把 ``evaluate`` 拆成几段分别计时，找出真正的成本中心。

上一轮用 profiler 看到 ``_bits`` 占 14.6%，以为自己写着写着把生成器写成了瓶颈；
实际单独测速只有 1.01x 差别（profiler 把生成器的"累积时间"算进去了，是假象）。
所以这里改成**手动分段计时**：材料 / 攻击图 / 受攻 / 将帅安全，各占多少。
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.stdout.reconfigure(encoding="utf-8")
import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from jieqi import board as B  # noqa: E402
from jieqi import evaluate as EV  # noqa: E402
from jieqi.belief import Belief  # noqa: E402
from jieqi.consts import RED, BLACK  # noqa: E402


def make_board(plies=20, seed=4):
    b = B.Board.new_game(random.Random(seed))
    rng = random.Random(seed)
    for _ in range(plies):
        moves = b.gen_moves(b.turn)
        if not moves:
            break
        b.make(rng.choice(moves))
    return b


def timed(fn, n):
    t0 = time.monotonic()
    for _ in range(n):
        fn()
    dt = time.monotonic() - t0
    return dt / n * 1e6, n / dt      # 微秒/次, 次/秒


def main() -> int:
    board = make_board()
    ctx = EV.EvalContext(Belief.from_board(board), use_cache=False)
    n = 4000

    full_us, full_rate = timed(lambda: EV.evaluate(board, ctx), n)
    att_us, _ = timed(lambda: board.attack_map(RED), n)
    att2_us, _ = timed(lambda: (board.attack_map(RED), board.attack_map(BLACK)), n)
    stats_us, _ = timed(lambda: EV._alive_stats(board, ctx), n)

    # 材料+位置那一圈（只遍历有子的格）
    def material_only():
        occ = board.occ_side[RED] | board.occ_side[BLACK]
        side = board.side
        total = 0.0
        for sq in EV._bits(occ):
            s = side[sq]
            if s < 0:
                continue
            total += EV._piece_value(board, ctx, sq, s)
        return total

    mat_us, _ = timed(material_only, n)

    print("单局面 static evaluate 分段计时（%d 次平均，微秒/次）" % n)
    print("  evaluate 总计              %7.1f us  →  %8.0f 次/秒" % (full_us, full_rate))
    print("    材料 + 位置（有子的格）    %7.1f us   (%.0f%%)"
          % (mat_us, 100.0 * mat_us / full_us))
    print("    攻击图 ×2（红+黑）        %7.1f us   (%.0f%%)"
          % (att2_us, 100.0 * att2_us / full_us))
    print("    明子统计 _alive_stats     %7.1f us   (%.0f%%)"
          % (stats_us, 100.0 * stats_us / full_us))
    rest = full_us - mat_us - att2_us - stats_us
    print("    其余（受攻/被保护/将帅安全/机动）%6.1f us   (%.0f%%)"
          % (rest, 100.0 * rest / full_us))
    print("")
    print("参考：一个 2 秒搜索大约评估 10000~20000 次")
    print("  评估提速 2 倍 ≈ 每步多搜 %.1f 层"
          % (2 ** 0.35 - 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
