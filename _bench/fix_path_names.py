# -*- coding: utf-8 -*-
"""把 ``_bench`` 里残留的 ``GAME`` / ``STAGE`` 名字接到 ``_paths`` 上。

背景：第一轮改写把"写死路径"那段换成了 ``import _paths; _paths.add_to_path()``，
但脚本后面还有 ``os.chdir(GAME)`` / ``str(GAME / "records")`` 之类的引用——
``GAME`` 这个名字没了，一跑就是 ``NameError: name 'GAME' is not defined``
（``deploy_selftest.py`` 就是这样被抓出来的）。

修法：在 ``_paths.add_to_path()`` 那一行后面补

    GAME = _paths.GAME
    STAGE = _paths.STAGE

只补**真的用到了**那个名字的脚本；脚本幂等，跑第二遍不会有变化。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
SKIP = {"_paths.py", "migrate_paths.py"}
#: 这些脚本里 ``GAME`` 只是文档字符串里的字样，不补
ANCHOR = "_paths.add_to_path()\n"

#: 变量名历史上出现过三种写法：``GAME`` / ``STAGE`` / ``G``（诊断脚本用惯了 ``G``）
ALIASES = ("GAME", "STAGE", "G")


def used_words(text: str, name: str) -> bool:
    """``name`` 是否作为**标识符**被使用（排除文档/注释里的普通字样）。"""
    return re.search(r'(?<![A-Za-z_.])' + name + r'(?![A-Za-z_])', text) is not None


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "import _paths" not in text or ANCHOR not in text:
        return "跳过（没接 _paths）"
    if "GAME = _paths.GAME" in text:
        return "未变"
    body = text.split(ANCHOR, 1)[1]
    want = [n for n in ALIASES if used_words(body, n)]
    if not want:
        return "未变"
    lines = "".join(
        "%s = _paths.%s\n" % (n, "GAME" if n == "G" else n) for n in want)
    text = text.replace(ANCHOR, ANCHOR + lines, 1)
    path.write_text(text, encoding="utf-8")
    return "补上 " + ", ".join(want)


def main() -> int:
    changed, skipped = [], []
    for path in sorted(BENCH.glob("*.py")):
        if path.name in SKIP:
            continue
        result = patch(path)
        (changed if result not in ("未变",) and "跳过" not in result
         else skipped).append(f"{path.name}（{result}）")
    print("已改写：")
    for item in changed:
        print("  " + item)
    print("无需改动：")
    for item in skipped:
        print("  " + item)
    return 0


if __name__ == "__main__":
    sys.exit(main())
