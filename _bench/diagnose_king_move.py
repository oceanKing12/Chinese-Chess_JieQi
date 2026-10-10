"""诊断：为什么引擎会在开局走"将进一"这种没有意义的着法。

复现用户那一局：红方首手翻出 (3,4) 的兵并推进到 (4,4)，轮到黑方。
把根节点每个候选的均值 / 最差 / 各采样值都打出来，看是谁在撑这一手。
"""
import os
import random
import sys
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()

from jieqi import board as B  # noqa: E402
from jieqi.board import move_str  # noqa: E402
from jieqi.consts import EVAL_WEIGHTS, P  # noqa: E402
from jieqi.engine import Engine  # noqa: E402


def build():
    """复现：红方首手翻出 (3,4) 的兵，并走到 (4,4)。"""
    b = B.Board.new_game(random.Random(7))
    sq = 3 * 9 + 4
    b.ident[sq] = P            # 强制这枚摆位兵的真实身份就是兵
    mv = B.encode(sq, 4 * 9 + 4)
    assert b.is_legal(mv), "构造的着法不合法"
    b.make(mv)
    return b


def dump(b, difficulty):
    eng = Engine(difficulty, seed=1234)
    res = eng.analyse(b)
    print("=== %s -> %s | score=%s depth=%s nodes=%s samples=%s"
          % (difficulty, move_str(res.move), res.score, res.depth,
             res.nodes, res.samples))
    top = res.stats[0] if res.stats else None
    if top is not None:
        fields = [a for a in dir(top) if not a.startswith("_")]
        print("    RootStat 字段:", fields)
    print("    %-14s %10s %10s %10s" % ("move", "mean", "worst", "best_count"))
    for st in res.stats[:12]:
        print("    %-14s %10.1f %10.1f %10s"
              % (move_str(st.move), st.mean,
                 getattr(st, "worst", float("nan")),
                 getattr(st, "best_count", "-")))
    return res


def main():
    print("EVAL_WEIGHTS = %s" % (EVAL_WEIGHTS,))
    b = build()
    print(b.to_text())
    for d in ("normal", "hard"):
        dump(b, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
