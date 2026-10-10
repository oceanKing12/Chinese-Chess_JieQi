"""定位"根节点所有着法分数完全相同"的元凶。

手段：在同一份定局化局面上，分别开关 null_move / quiescence、并扫不同深度，
统计"根着法有多少个**互不相同**的分数"。如果只有 1 个，说明搜索完全与着法无关。
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
from jieqi.consts import P  # noqa: E402
from jieqi.evaluate import EvalContext, evaluate  # noqa: E402
from jieqi.search import Searcher  # noqa: E402


def build():
    b = B.Board.new_game(random.Random(7))
    sq = 3 * 9 + 4
    b.ident[sq] = P
    b.make(B.encode(sq, 4 * 9 + 4))
    return b


def main():
    pub = build().clone_revealed_only()
    belief = Belief.from_board(pub)
    ctx = EvalContext(belief)
    rng = random.Random(1234)
    sample = pub.copy()
    belief.apply_sample(sample, rng)
    root_moves = sample.gen_moves(sample.turn)

    print("根着法数 = %d" % len(root_moves))

    # 0) make() 是否正确翻转走子方
    t0 = sample.turn
    sample.make(root_moves[0])
    print("make() 后 turn: %d -> %d  (应翻转)" % (t0, sample.turn))
    k_child = sample.key
    sample.unmake()
    print("unmake() 后 turn = %d, key 还原 = %s"
          % (sample.turn, sample.key != k_child))

    # 1) 逐个开关，看分数是否与着法有关
    for nm in (True, False):
        for q in (True, False):
            for d in (1, 2, 3):
                se = Searcher(sample.copy(), ctx,
                              dict(null_move=nm, quiescence=q))
                se.deadline = None
                _, _, _, sc = se.search_root(d, root_moves)
                vals = [sc.get(mv) for mv in root_moves]
                distinct = len(set(vals))
                flag = "  <-- 与着法无关!" if distinct == 1 else ""
                print("  null_move=%-5s quiescence=%-5s depth=%d -> 不同分数 %2d 种"
                      "  样例 %s%s"
                      % (nm, q, d, distinct,
                         sorted(set(vals))[:3], flag))
    return 0


sys.exit(main())
