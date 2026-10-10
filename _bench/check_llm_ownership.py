# -*- coding: utf-8 -*-
"""快速自检：归属判定 / 质检修复（不联网）。

覆盖三类写法：
* 显式归属词（我方车 / 我方的车 / 对方的马）
* 颜色写法（红方車 / 黑卒）
* 靠动词判断主语（车可以吃士）

以及"整体说反"时能否安全自动改判。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
GAME = _paths.GAME
STAGE = _paths.STAGE
for path in (str(STAGE), str(GAME)):
    if path not in sys.path:
        sys.path.insert(0, path)

from jieqi.llm import (  # noqa: E402
    check_comment, piece_belongs_to, repair_comment, scan_side_piece,
)

SNAP = {"ai_color": "black", "board": [[None] * 9 for _ in range(10)]}
SNAP["board"][0][0] = {"side": "red", "hidden": False, "home": "车", "ident": "车"}
SNAP["board"][0][1] = {"side": "red", "hidden": False, "home": "兵", "ident": "兵"}
SNAP["board"][0][2] = {"side": "red", "hidden": False, "home": "炮", "ident": "炮"}
SNAP["board"][9][0] = {"side": "black", "hidden": True, "home": "卒", "ident": None}
SNAP["board"][9][1] = {"side": "black", "hidden": False, "home": "卒", "ident": "卒"}
SNAP["board"][9][4] = {"side": "black", "hidden": False, "home": "将", "ident": "将"}

#: 让"我方车/对方卒"变成合法说法（互换红黑的另一套事实），用来验证不误报。
SNAP_ALT = {"ai_color": "black", "board": [[None] * 9 for _ in range(10)]}
SNAP_ALT["board"][0][0] = {"side": "black", "hidden": False, "home": "車", "ident": "車"}
SNAP_ALT["board"][9][1] = {"side": "red", "hidden": False, "home": "兵", "ident": "兵"}
SNAP_ALT["board"][9][4] = {"side": "black", "hidden": False, "home": "将", "ident": "将"}

SCAN_CASES = [
    "红方車正吃我的卒",
    "红方的车压过来",
    "我方车",
    "对方的马能吃",
    "黑方卒顶上来",
    "红帅被将",
    "黑卒被吃",
    "我方的车被对方的卒挡住",
]
CASES = [
    "我方车从中路杀出，吃掉对方的兵。",
    "我方帅已经安全了。",
    "黑方車正吃我的卒，需要小心。",
    "我方卒（黑卒）顶上去，对方车被牵制。",
    "车可以吃士。",
    "红方車已经压到我的卒前面。",
    "卒顶上去换掉对方的车。",
    "这步棋稳住中路，先手在我。",
    "我方的车被对方的卒挡住了，对方的卒很强。",
    "对方的车压过来，我方的卒只能退。",
    "我方的卒顶上去，对方的车被牵制。",
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("--- scan_side_piece ---")
    for text in SCAN_CASES:
        hits = ["%s/%s%s" % (h["word"] or h["color"], h["piece"],
                             "@%d" % h["position"]) for h in scan_side_piece(text)]
        print("%-22s -> %s" % (text, hits))
    print("--- piece_belongs_to ---")
    print({ch: piece_belongs_to(ch) for ch in "帅车马炮兵将士象馬車砲卒我"})
    print("--- check / repair ---")
    print("SNAP chars:", sorted(
        {ch: side for _l, ch, side in __import__("jieqi.llm", fromlist=["x"])
         .piece_ownership(SNAP)}.items()))
    for text in CASES:
        issues = check_comment(text, SNAP)
        fixed, notes = repair_comment(text, SNAP)
        print("原文：%s" % text)
        print("  scan：%s" % [(h["word"] or h["color"], h["piece"])
                             for h in scan_side_piece(text)])
        print("  问题：%s" % ("；".join(issues) if issues else "无"))
        print("  修复：%s %s" % (fixed or "(丢弃)", "；".join(notes) if notes else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
