# -*- coding: utf-8 -*-
"""校验：把我的重放结果和棋谱里存的**每一步快照**逐手对比。

如果第 k 手之后就对不上，说明重放逻辑（尤其是翻子/暗子身份）有问题，
那么基于重放做的任何复盘结论都不可信。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.stdout.reconfigure(encoding="utf-8")
import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from jieqi import record as R  # noqa: E402


def sig_gui(board):
    out = []
    for r in range(10):
        for c in range(9):
            p = board[r][c]
            if not p:
                out.append("..")
            else:
                out.append("%s%s%s" % (p["color"][0],
                                       p.get("actual") if p.get("revealed") else "?",
                                       "h" if not p.get("revealed") else "v"))
    return "|".join(out)


def sig_snap(board):
    out = []
    for r in range(10):
        for c in range(9):
            cell = board[r][c]
            if not cell:
                out.append("..")
            else:
                out.append("%s%s%s" % (str(cell.get("side"))[0],
                                       (cell.get("ident") or cell.get("actual"))
                                       if not cell.get("hidden") else "?",
                                       "h" if cell.get("hidden") else "v"))
    return "|".join(out)


def main() -> int:
    # 棋谱路径跟着"当前这棵树"走（原来写死部署树，在开发树里跑就找不到文件）
    rec = R.load_record(str(_paths.GAME / "records"
                            / "20261007-174044-red-vs-ai.json"))
    from game_state import GameState

    moves, reveals, _meta = R.replay_plan(rec)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    snapshots = rec["positions"]
    print("棋谱 %d 手，快照 %d 个" % (len(moves), len(snapshots)))
    for index in range(1, len(moves) + 1):
        sr, sc, er, ec = moves[index - 1]
        setup = reveals.get(index)
        if setup is not None:
            cell = gs.board[setup[0]][setup[1]]
            if cell is not None and not cell.get("revealed"):
                cell["actual"] = setup[2]
        gs.move_piece(sr, sc, er, ec)
        gs.current_player = "black" if gs.current_player == "red" else "red"
        mine = sig_gui(gs.board)
        theirs = sig_snap(snapshots[index]["board"])
        if mine != theirs:
            print("第 %d 手之后**不一致**：%s->%s" % (index, (sr, sc), (er, ec)))
            bad = 0
            for k, (a, b) in enumerate(zip(mine.split("|"), theirs.split("|"))):
                if a != b:
                    bad += 1
                    if bad <= 6:
                        print("   格(%d,%d) 我=%s 棋谱=%s"
                              % (k // 9, k % 9, a, b))
            print("   共 %d 格不同" % bad)
            return 1
    print("OK：%d 手逐步快照全部一致" % len(moves))
    return 0


if __name__ == "__main__":
    sys.exit(main())
