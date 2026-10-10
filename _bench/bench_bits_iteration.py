# -*- coding: utf-8 -*-
"""测速：把 ``evaluate._bits`` 的生成器换成"位扫描 + 查表"能快多少。

只做**速度对比**，逻辑与结果必须完全一致（脚本自己会校正一致性）。
不修改 ``evaluate.py``——它用猴子补丁在进程内替换，跑完即弃。
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
from jieqi.consts import SQ_BIT, SQUARES  # noqa: E402
from jieqi.engine import Engine  # noqa: E402

#: 位掩码（单 bit）-> 格子号。位棋盘是稀疏布局，所以必须查表而不是 bit_length。
_BIT_TO_SQ = {SQ_BIT[sq]: sq for sq in range(SQUARES)}


def bits_table(mask: int):
    """新的遍历方式：纯位运算 + 字典查表，没有生成器、没有函数调用开销。"""
    while mask:
        low = mask & -mask
        yield _BIT_TO_SQ[low]
        mask ^= low


def _make_board(plies=20, seed=4):
    b = B.Board.new_game(random.Random(seed))
    rng = random.Random(seed)
    for _ in range(plies):
        moves = b.gen_moves(b.turn)
        if not moves:
            break
        b.make(rng.choice(moves))
    return b


def _bench(board, n=4000):
    ctx = EV.EvalContext(Belief.from_board(board), use_cache=False)
    t0 = time.monotonic()
    for _ in range(n):
        EV.evaluate(board, ctx)
    return n / (time.monotonic() - t0)


def main() -> int:
    board = _make_board()
    ctx = EV.EvalContext(Belief.from_board(board), use_cache=False)
    want = EV.evaluate(board, ctx)

    old_bits = EV._bits
    old_rate = _bench(board)
    EV._bits = bits_table
    try:
        got = EV.evaluate(board, ctx)
        assert got == want, "换了遍历方式结果不一致：%s != %s" % (got, want)
        new_rate = _bench(board)
    finally:
        EV._bits = old_bits
    assert EV.evaluate(board, ctx) == want

    print("单局面静态评估（同机、无缓存）")
    print("  生成器版本 : %9.0f 次/秒" % old_rate)
    print("  位扫描版本 : %9.0f 次/秒  （%.2fx）" % (new_rate, new_rate / old_rate))
    print("  结果一致性 : 通过")

    # 再量一次"整局搜索"的收益：同样的时间能多搜多少节点
    for label, patched in (("生成器版本", False), ("位扫描版本", True)):
        if patched:
            EV._bits = bits_table
        try:
            depths, nodes = [], []
            for seed in (1, 2, 3):
                eng = Engine("normal", time_limit=2.0, samples=1, seed=seed)
                res = eng.analyse(board)
                depths.append(res.depth)
                nodes.append(res.nodes)
            print("  %s：2 秒内深度 %s，节点 %s"
                  % (label, depths, nodes))
        finally:
            if patched:
                EV._bits = old_bits
    return 0


if __name__ == "__main__":
    sys.exit(main())
