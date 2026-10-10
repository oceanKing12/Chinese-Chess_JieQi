# -*- coding: utf-8 -*-
"""评估函数差分：重写后的 evaluate 必须与"逐格朴素版"完全一致。

朴素版就是重写前的实现（在 tests/test_eval.py 的语义下），这里单独再写一遍，
避免以后有人图快又改出"位号当格子号"之类的错（那类错会在棋盘下半部分算错）。
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
sys.stdout.reconfigure(encoding="utf-8")

from jieqi import evaluate as ev  # noqa: E402
from jieqi.board import Board  # noqa: E402
from jieqi.belief import Belief  # noqa: E402
from jieqi.consts import (  # noqa: E402
    BLACK, KING_NEIGH, RED, SQUARES, SQ_BIT,
)


def naive(board: Board, ctx: ev.EvalContext) -> int:
    """逐格版（重写前的算法），只用于差分。"""
    w = ctx.weights
    side = board.side
    occ_side = board.occ_side
    score = 0.0
    values = [0.0] * SQUARES
    for sq in range(SQUARES):
        s = side[sq]
        if s < 0:
            continue
        v = ev._piece_value(board, ctx, sq, s)
        values[sq] = v
        score += v if s == RED else -v
    att_red = board.attack_map(RED)
    att_black = board.attack_map(BLACK)
    att = (att_red, att_black)
    mob = (bin(att_red & ~occ_side[RED]).count("1")
           - bin(att_black & ~occ_side[BLACK]).count("1"))
    score += w["mobility"] * mob
    for sq in range(SQUARES):
        s = side[sq]
        if s < 0:
            continue
        bit = SQ_BIT[sq]
        if not (att[1 - s] & bit):
            continue
        loss = values[sq] * (w["attacked_defended"] if (att[s] & bit) else w["hanging"])
        score += loss if s == BLACK else -loss
    for s in (RED, BLACK):
        ks = board.kings[s]
        if ks < 0:
            score += -ev._INF if s == RED else ev._INF
            continue
        danger = bin(att[1 - s] & KING_NEIGH[ks]).count("1")
        pen = w["king_danger"] * danger
        pen += max(0.0, 2 - ev.expected_alive(board, ctx, 2, s)) * w["missing_advisor"]
        pen += max(0.0, 2 - ev.expected_alive(board, ctx, 3, s)) * w["missing_elephant"]
        score += -pen if s == RED else pen
    turn = board.turn
    if board.in_check(turn):
        score += -w["in_check"] if turn == RED else w["in_check"]
    score += w["tempo"] if turn == RED else -w["tempo"]
    return int(score)


def main() -> int:
    rng = random.Random(11)
    checked = mismatched = 0
    for game in range(6):
        board = Board.new_game(rng)
        hidden_truth = board.copy()
        for _ in range(40):
            side = board.turn
            moves = board.gen_moves(side)
            if not moves:
                break
            mv = rng.choice(moves)
            board.make(mv)
            # 用"只保留明子"的公开局面建 belief（与真实调用一致）
            pub = board.clone_revealed_only()
            ctx = ev.EvalContext(Belief.from_board(pub))
            a = ev.evaluate(pub, ctx)
            # 朴素版必须用**同一个 ctx**，但 evaluate 会写缓存，所以清掉再算
            ctx.cache.clear()
            b = naive(pub, ctx)
            checked += 1
            if a != b:
                mismatched += 1
                if mismatched <= 5:
                    print("不一致: game=%d 快速=%d 朴素=%d" % (game, a, b))
        del hidden_truth
    print("对照 %d 个局面，不一致 %d 个" % (checked, mismatched))
    return 0 if mismatched == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
