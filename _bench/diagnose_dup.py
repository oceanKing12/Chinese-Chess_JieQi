"""诊断：根着法的**逐采样原始分**，查"不同着法统计量完全相同"的原因。"""
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
from jieqi.search import Searcher, TimeUp  # noqa: E402


def build():
    b = B.Board.new_game(random.Random(7))
    sq = 3 * 9 + 4
    b.ident[sq] = P
    b.make(B.encode(sq, 4 * 9 + 4))
    return b


def main():
    b0 = build()
    pub = b0.clone_revealed_only()
    belief = Belief.from_board(pub)
    ctx = EvalContext(belief)
    root_moves = pub.gen_moves(pub.turn)

    rng = random.Random(1234)
    n = 4
    boards = []
    for _ in range(n):
        bb = pub.copy()
        belief.apply_sample(bb, rng)
        boards.append(bb)

    per_move = {mv: [] for mv in root_moves}
    for i, bb in enumerate(boards):
        se = Searcher(bb, ctx, dict(quiescence=True, null_move=True))
        se.deadline = None
        # 固定深度，排除时间抖动
        _, _, _, scores = se.search_root(4, root_moves)
        for mv in root_moves:
            per_move[mv].append(scores.get(mv))
        print("  采样 %d 完成" % i)

    print()
    print("    %-14s %s" % ("move", "  ".join("s%d" % k for k in range(n))))
    rows = []
    for mv in root_moves:
        vals = per_move[mv]
        rows.append((sum(vals) / n, mv, vals))
    rows.sort(reverse=True)
    for mean, mv, vals in rows[:14]:
        print("    %-14s %s   mean=%+.1f"
              % (move_str(mv), "  ".join("%+7d" % v for v in vals), mean))

    # 找出"多枚着法共享同一组原始分"的情况
    print()
    seen = {}
    for mean, mv, vals in rows:
        key = tuple(vals)
        seen.setdefault(key, []).append(move_str(mv))
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    if dup:
        print("  !! 共享同一组原始分的着法：")
        for k, v in dup.items():
            print("     %s -> %s" % (k, v))
    else:
        print("  没有共享原始分的着法（说明统计量相同是巧合或聚合有误）")
    return 0


sys.exit(main())
