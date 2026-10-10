# -*- coding: utf-8 -*-
"""把 ``attack_map`` 的耗时按**棋子类型**拆开，找出该优化哪一支。

上一轮实测 ``attack_map`` 占静态评估的 45%（两个颜色共 23.6us），但里面
分了四种走法（车滑行 / 炮隔子 / 将帅邻格 / 其余查表），到底是哪一支贵？
不拆开就优化等于瞎猜——上一轮已经在 ``_bits`` 生成器上栽过一次
（profiler 把累积时间算进去，看着占 14.6%，实际换掉只快 1.01x）。
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
from jieqi.consts import (  # noqa: E402
    BLACK,
    C,
    K,
    MOVES,
    MOVES_HIDDEN,
    R,
    RAYSQ,
    RED,
    SQ_BIT,
    SQUARES,
)


def make_board(plies=20, seed=4):
    b = B.Board.new_game(random.Random(seed))
    rng = random.Random(seed)
    for _ in range(plies):
        moves = b.gen_moves(b.turn)
        if not moves:
            break
        b.make(rng.choice(moves))
    return b


def main() -> int:
    board = make_board()
    n = 3000

    def timeit(fn):
        t0 = time.monotonic()
        for _ in range(n):
            fn()
        return (time.monotonic() - t0) / n * 1e6      # us/次

    full_red = timeit(lambda: board.attack_map(RED))
    full_black = timeit(lambda: board.attack_map(BLACK))
    full_both = timeit(lambda: (board.attack_map(RED), board.attack_map(BLACK)))

    # 按类型分别计时：临时把"其它类型"的棋子换掉不现实，所以改成
    # 统计**每种类型有几枚子**，再用单枚子的平均成本外推。
    occ = board.occ
    side_arr = board.side
    rule = board.rule_ident
    counts = {RED: {}, BLACK: {}}
    for sq in range(SQUARES):
        s = side_arr[sq]
        if s < 0:
            continue
        counts[s][rule[sq]] = counts[s].get(rule[sq], 0) + 1

    print("局面里各类型棋子数（红/黑）：")
    name = {K: "帅", 2: "仕", 3: "相", 4: "马", R: "车", C: "炮", 7: "兵"}
    for ident in (R, C, 4, 3, 2, 7, K):
        print("  %-4s 红 %d  黑 %d"
              % (name.get(ident, "?"), counts[RED].get(ident, 0),
                 counts[BLACK].get(ident, 0)))

    # 单独测"一次完整走法生成"的射线开销（同一种滑行逻辑）
    def ray_scan():
        m = 0
        raylets = RAYSQ
        for sq in range(SQUARES):
            if side_arr[sq] < 0:
                continue
            for ray in raylets[sq]:
                for s in ray:
                    m |= SQ_BIT[s]
                    if occ & SQ_BIT[s]:
                        break
        return m

    def ray_scan_bits():
        """候选写法：只遍历有子的格，射线用预计算的"位掩码 + 格子序列"。"""
        m = 0
        raylets = RAYSQ
        mask = occ
        while mask:
            low = mask & -mask
            sq = _BIT_TO_SQ[low]
            mask ^= low
            for ray in raylets[sq]:
                for s in ray:
                    m |= SQ_BIT[s]
                    if occ & SQ_BIT[s]:
                        break
        return m

    def ray_scan_ids():
        """候选写法：只遍历有子的格，SQ_BIT 提到循环外当局部变量。"""
        m = 0
        raylets = RAYSQ
        bits = SQ_BIT
        mask = occ
        while mask:
            low = mask & -mask
            sq = _BIT_TO_SQ[low]
            mask ^= low
            for ray in raylets[sq]:
                for s in ray:
                    m |= bits[s]
                    if occ & bits[s]:
                        break
        return m

    global _BIT_TO_SQ
    _BIT_TO_SQ = {SQ_BIT[sq]: sq for sq in range(SQUARES)}

    print("\n单次攻击图成本（微秒）：")
    print("  红方              %6.2f" % full_red)
    print("  黑方              %6.2f" % full_black)
    print("  两个颜色合计      %6.2f" % full_both)
    print("\n滑行射线扫描（车/炮共用的那段逻辑）：")
    base = timeit(ray_scan)
    scan_bits = timeit(ray_scan_bits)
    scan_ids = timeit(ray_scan_ids)
    print("  遍历 90 格        %6.2f" % base)
    print("  只遍历有子的格    %6.2f  (%.2fx)" % (scan_bits, base / scan_bits))
    print("  + SQ_BIT 局部化   %6.2f  (%.2fx)" % (scan_ids, base / scan_ids))
    print("\n参考：一个 2 秒搜索大约评估 10000~20000 次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
