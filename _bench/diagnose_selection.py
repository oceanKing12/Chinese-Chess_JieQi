"""诊断：为什么 hard/expert 会选"将进一"这种废棋。

假设：``select_score = mean - penalty * stdev``。揭棋里**任何动暗子**的着法
都会在不同采样里翻开成完全不同的兵种（卒 vs 车），样本方差天然巨大；
而"将进一"动的是已经明牌的将，样本方差≈0。于是方差惩罚把"动暗子"
全部压下去，只剩下"动将/动仕"这类零风险废棋。

这里把根节点每个着法的 mean / worst / stdev / select_score 全打出来，
并对比"只用 mean 排序"会选什么。
"""
import random
import sys
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()

from jieqi import board as B  # noqa: E402
from jieqi.belief import Belief  # noqa: E402
from jieqi.board import move_str  # noqa: E402
from jieqi.consts import P, RED  # noqa: E402
from jieqi.engine import DIFFICULTIES, Engine  # noqa: E402
from jieqi.evaluate import EvalContext, evaluate  # noqa: E402

KING_MOVE = B.encode(9 * 9 + 4, 8 * 9 + 4)


def build():
    """复现用户那一局：红方首手 (3,4) 的兵翻出并推进到 (4,4)。"""
    b = B.Board.new_game(random.Random(7))
    sq = 3 * 9 + 4
    b.ident[sq] = P
    b.make(B.encode(sq, 4 * 9 + 4))
    return b


def static_order(b):
    pub = b.clone_revealed_only()
    ctx = EvalContext(Belief.from_board(pub))
    side = pub.turn
    out = []
    for mv in pub.gen_moves(side):
        pub.make(mv)
        v = evaluate(pub, ctx)
        pub.unmake()
        out.append((v if side == RED else -v, mv))
    out.sort(reverse=True)
    return out


def main():
    b = build()
    st = static_order(b)
    print("--- 静态评估（黑方视角，越高越好）---")
    for v, mv in st[:5]:
        print("   前5  %-14s %+8.1f" % (move_str(mv), v))
    king_static = [v for v, mv in st if mv == KING_MOVE]
    if king_static:
        rank = [mv for v, mv in st].index(KING_MOVE) + 1
        print("   将(9,4)->(8,4) 静态=%+.1f 排第 %d/%d"
              % (king_static[0], rank, len(st)))

    for name in ("normal", "hard", "expert"):
        opts = DIFFICULTIES[name]
        penalty = float(opts.get("variance_penalty", 0.0) or 0.0)
        res = Engine(name, seed=1234).analyse(b)
        print()
        print("=== %s  选中 %s | score=%+.0f | depth=%s | samples=%s | %.2fs | penalty=%s"
              % (name, move_str(res.move), res.score, res.depth,
                 res.samples, res.elapsed, penalty))
        print("    %-14s %9s %9s %9s %9s %7s"
              % ("move", "mean", "worst", "stdev", "select", "best#"))
        for s in res.stats[:10]:
            stdev = ((s.mean - s.select_score) / penalty) if penalty else 0.0
            mark = ""
            if s.move == res.move:
                mark = "  <- 选中"
            if s.move == KING_MOVE:
                mark = "  <- 将进一"
            print("    %-14s %9.1f %9.1f %9.1f %9.1f %7d%s"
                  % (move_str(s.move), s.mean, s.worst, stdev,
                     s.select_score, s.best_count, mark))
        by_mean = sorted(res.stats, key=lambda x: -x.mean)
        by_worst = sorted(res.stats, key=lambda x: -x.worst)
        print("    只看 mean   -> %s (%+.1f)" % (move_str(by_mean[0].move), by_mean[0].mean))
        print("    只看 worst  -> %s (%+.1f)" % (move_str(by_worst[0].move), by_worst[0].worst))
        for s in res.stats:
            if s.move == KING_MOVE:
                rank = [x.move for x in res.stats].index(s.move) + 1
                print("    将进一: mean=%+.1f worst=%+.1f select=%+.1f  排名 %d/%d"
                      % (s.mean, s.worst, s.select_score, rank, len(res.stats)))
    return 0


sys.exit(main())
