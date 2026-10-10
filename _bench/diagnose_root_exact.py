"""决定性实验：``search_root`` 报出的根着法分数，是不是**精确值**？

参照实现：对每个根着法单独开一次**全窗口**搜索（alpha=-INF, beta=+INF），
拿到该着法的真值。把两者逐着法比对。

如果大量着法的 search_root 分数**明显低于**真值（而且彼此相等），
就证明根节点把 PVS 的"fail-low 上界"当成了着法分数——
而引擎的跨采样平均/方差惩罚全部建立在这个分数上，
于是"动暗子"（真值好但被 clamp）全被压掉，只剩"动将"这类废棋。
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
from jieqi.evaluate import EvalContext  # noqa: E402
from jieqi.search import INF, Searcher  # noqa: E402


def build():
    b = B.Board.new_game(random.Random(7))
    sq = 3 * 9 + 4
    b.ident[sq] = P
    b.make(B.encode(sq, 4 * 9 + 4))
    return b


def exact_scores(sample, ctx, root_moves, depth):
    """参照：每个根着法各开一次全窗口搜索。"""
    out = {}
    for mv in root_moves:
        se = Searcher(sample.copy(), ctx, dict(quiescence=True, null_move=True))
        se.deadline = None
        sample.make(mv)
        v = -se._negamax(depth - 1, -INF + 1, INF - 1, 1, True)
        sample.unmake()
        out[mv] = v
    return out


def main():
    pub = build().clone_revealed_only()
    belief = Belief.from_board(pub)
    ctx = EvalContext(belief)
    rng = random.Random(1234)
    sample = pub.copy()
    belief.apply_sample(sample, rng)
    root_moves = sample.gen_moves(sample.turn)

    for depth in (2, 3, 4):
        se = Searcher(sample.copy(), ctx, dict(quiescence=True, null_move=True))
        se.deadline = None
        _, best, pv, fast = se.search_root(depth, root_moves)
        ref = exact_scores(sample, ctx, root_moves, depth)

        bad = 0
        worse_by = []
        for mv in root_moves:
            f, r = fast.get(mv), ref.get(mv)
            if f != r:
                bad += 1
                worse_by.append((r - f, mv, f, r))
        print("=== depth=%d  best=%s(%d)" % (depth, move_str(best), best))
        print("    与全窗口参照不一致的着法: %d / %d" % (bad, len(root_moves)))
        worse_by.sort(reverse=True)
        for d, mv, f, r in worse_by[:6]:
            print("      %-14s search_root=%+7d  真值=%+7d  低估 %+d"
                  % (move_str(mv), f, r, d))
        # 真值最优 vs search_root 选中的最优
        true_best = max(ref.items(), key=lambda kv: kv[1])
        print("    全窗口真值最优: %s (%+d)" % (move_str(true_best[0]), true_best[1]))
        # 只看 fast 里的"分布"
        vals = sorted(set(fast.values()))
        print("    search_root 分数只有 %d 种不同取值: %s"
              % (len(vals), vals[:6]))
    return 0


sys.exit(main())
