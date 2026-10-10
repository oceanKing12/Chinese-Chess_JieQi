"""把评估函数拆成各项，看开局（子力完全对称）时是谁在贡献分数。

期望：开局红黑完全对称，总分应当≈0（只有先手分 8）。
如果某一项大得离谱，它就是"开局乱走"的元凶。
"""
import random
import sys
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()

from jieqi import board as B  # noqa: E402
from jieqi.belief import Belief  # noqa: E402
from jieqi.consts import (  # noqa: E402
    A, BLACK, E, EVAL_WEIGHTS, KING_NEIGH, PIECE_VALUE, P, RED,
    SQUARES, SQ_BIT,
)
from jieqi.evaluate import EvalContext, expected_alive  # noqa: E402


def breakdown(b, label):
    ctx = EvalContext(Belief.from_board(b))
    w = ctx.weights
    side, ident, occ_side = b.side, b.ident, b.occ_side

    terms = {}
    values = [0.0] * SQUARES
    mat = 0.0
    for sq in range(SQUARES):
        s = side[sq]
        if s < 0:
            continue
        if b.hidden[sq]:
            v = ctx.dark_total.get(s, {}).get(sq, 0.0)
        else:
            v = PIECE_VALUE[ident[sq]] + 0.0
            # 位置分单独看
        values[sq] = v
        mat += v if s == RED else -v
    terms["材料(明子+暗子期望)"] = mat

    # 位置分
    from jieqi.consts import HOME_IDENT, PST_FLAT_SIDE
    pos = 0.0
    for sq in range(SQUARES):
        s = side[sq]
        if s < 0:
            continue
        if b.hidden[sq]:
            p = PST_FLAT_SIDE[s][HOME_IDENT[sq]][sq]
        else:
            p = PST_FLAT_SIDE[s][ident[sq]][sq]
        pos += p if s == RED else -p
    terms["位置分"] = pos

    att_red = b.attack_map(RED)
    att_black = b.attack_map(BLACK)
    att = (att_red, att_black)
    red_mob = bin(att_red & ~occ_side[RED]).count("1")
    blk_mob = bin(att_black & ~occ_side[BLACK]).count("1")
    terms["机动性 (%d vs %d) x%d" % (red_mob, blk_mob, w["mobility"])] = \
        w["mobility"] * (red_mob - blk_mob)

    hang = w["hanging"]
    defd = w["attacked_defended"]
    h = 0.0
    for sq in range(SQUARES):
        s = side[sq]
        if s < 0:
            continue
        bit = SQ_BIT[sq]
        if not (att[1 - s] & bit):
            continue
        loss = values[sq] * (defd if (att[s] & bit) else hang)
        h += loss if s == BLACK else -loss
    terms["受攻/被保护"] = h

    kd = 0.0
    for s in (RED, BLACK):
        ks = b.kings[s]
        danger = bin(att[1 - s] & KING_NEIGH[ks]).count("1")
        pen = w["king_danger"] * danger
        pen += max(0.0, 2 - expected_alive(b, ctx, A, s)) * w["missing_advisor"]
        pen += max(0.0, 2 - expected_alive(b, ctx, E, s)) * w["missing_elephant"]
        kd += -pen if s == RED else pen
    terms["将帅安全"] = kd

    chk = 0.0
    if b.in_check(b.turn):
        chk = -w["in_check"] if b.turn == RED else w["in_check"]
    terms["被将军"] = chk
    tempo = w["tempo"] if b.turn == RED else -w["tempo"]
    terms["先手"] = tempo

    total = sum(terms.values())
    print("--- %s | 合计 = %.1f" % (label, total))
    for k, v in terms.items():
        if abs(v) > 1e-9:
            print("      %-28s %+9.1f" % (k, v))
    return total


def main():
    print("EVAL_WEIGHTS =", EVAL_WEIGHTS)
    b0 = B.Board.new_game(random.Random(7))
    breakdown(b0, "开局（红先，完全对称）")

    # 红方翻出 (3,4) 的兵并推进到 (4,4)
    b1 = B.Board.new_game(random.Random(7))
    sq = 3 * 9 + 4
    b1.ident[sq] = P
    b1.make(B.encode(sq, 4 * 9 + 4))
    breakdown(b1, "红兵 (3,4)->(4,4) 之后（轮到黑）")

    # 黑方"将进一"(9,4)->(8,4)
    b2 = b1.copy()
    b2.make(B.encode(9 * 9 + 4, 8 * 9 + 4))
    breakdown(b2, "黑将 (9,4)->(8,4) 之后（轮到红）")

    # 对比：黑方挺中卒 (6,4)->(5,4)
    b3 = b1.copy()
    b3.make(B.encode(6 * 9 + 4, 5 * 9 + 4))
    breakdown(b3, "黑中卒 (6,4)->(5,4) 之后（轮到红）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
