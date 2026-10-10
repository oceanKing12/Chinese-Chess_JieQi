# -*- coding: utf-8 -*-
"""一次性把 ``_bench`` 里写死的"开发树路径"换成共用的 ``_paths`` 探测。

为什么要有这个脚本：22 个脚本各自抄了一遍

    STAGE = Path(__file__).resolve().parents[1] / "stage"
    GAME = STAGE / "Game"
    for p in (str(STAGE), str(GAME)): ...

这段在开发树没问题，被 ``deploy.py`` 复制到发布目录后就找不到 ``stage``，
于是 ``from jieqi import record`` 直接 ModuleNotFoundError（用户实际遇到的）。
换成 ``import _paths`` 之后两种布局都能认出来。

脚本**幂等**：已经换过的文件再跑一次不会有变化。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent

#: 变量名（``STAGE`` / ``GAME`` / ``G``）+ 引号（单/双）+ 可选的 ``r`` 前缀都要认
_VAR = r'(?P<var>[A-Za-z_][A-Za-z0-9_]*)'
_LIT = (r'(?P<q>["\'])'
        r'(?:[A-Za-z]:[\\/][^"\']*?)'
        r'(?:揭棋|Chinese Chess)'
        r'[\\/](?:stage[\\/])?Game'
        r'(?P=q)')
_ASSIGN = (r'^[ \t]*' + _VAR + r'[ \t]*=[ \t]*'
           r'Path\([rRbBuU]?' + _LIT + r'\)[ \t]*\n')
_INSERTS = (r'(?:[ \t]*sys\.path\.insert\(0,[ \t]*str\([A-Za-z_.]+\)\)[ \t]*;?'
            r'[ \t]*(?:\n|$)'
            r'|[ \t]*for[ \t]+p[ \t]+in[ \t]*\([^\n]*\):[ \t]*\n'
            r'(?:[ \t]*if[ \t]+p[ \t]+not[ \t]+in[ \t]+sys\.path:[ \t]*\n)?'
            r'[ \t]*sys\.path\.insert\(0,[ \t]*p\)[ \t]*\n'
            r'(?:[ \t]*sys\.path\.insert\(0,[ \t]*str\([A-Za-z_.]+\)\)[ \t]*\n)?)')

#: 一段完整的"写死路径 + 加 sys.path"（换行/分号两种写法都吃）
PAT_HARDCODED = re.compile(_ASSIGN + _INSERTS + r'+', re.MULTILINE)

#: 只写死了路径、没加 sys.path 的（例如 legacy 脚本）
PAT_ASSIGN_ONLY = re.compile(_ASSIGN, re.MULTILINE)

#: 形如 "STAGE = Path(__file__).resolve().parents[1] / \"stage\"\n
#:        GAME = STAGE / \"Game\"\n
#:        for p in (str(STAGE), str(GAME)):\n
#:            if p not in sys.path:\n
#:                sys.path.insert(0, p)"
PAT_PARENTS = re.compile(
    r'^[ \t]*STAGE[ \t]*=[ \t]*Path\(__file__\)\.resolve\(\)\.parents\[1\][ \t]*/[ \t]*"stage"\n'
    r'(?:[ \t]*GAME[ \t]*=[ \t]*STAGE[ \t]*/[ \t]*"Game"\n)?'
    r'(?:[ \t]*for[ \t]+p[ \t]+in[ \t]*\([^\n]*\):\n'
    r'(?:[ \t]*if[ \t]+p[ \t]+not[ \t]+in[ \t]+sys\.path:\n)?'
    r'[ \t]*sys\.path\.insert\(0,[ \t]*p\)\n)?'
    r'(?:[ \t]*for[ \t]+p[ \t]+in[ \t]*\([^\n]*\):\n'
    r'(?:[ \t]*if[ \t]+p[ \t]+not[ \t]+in[ \t]+sys\.path:\n)?'
    r'[ \t]*sys\.path\.insert\(0,[ \t]*p\)\n)?',
    re.MULTILINE)

REPLACEMENT = ("#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。\n"
               "import _paths\n"
               "_paths.add_to_path()\n")

SKIP = {"_paths.py", "migrate_paths.py", "migrate_bench_paths.py"}

#: 自检时跳过：``apply_reveal_verdict.py`` 刻意留了一条"另一棵树"的兜底路径
SELFTEST_SKIP = SKIP | {"legacy_AI_original.py", "apply_reveal_verdict.py"}


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    original = text
    n = 0
    for pattern in (PAT_PARENTS, PAT_HARDCODED, PAT_ASSIGN_ONLY):
        text, count = pattern.subn(REPLACEMENT, text)
        if count:
            n += count
            break               # 一个脚本只有头部那一处，换完就走
    if text == original:
        return "未变"
    # 收拾可能出现的连续空行（三行以上压成两行）
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    path.write_text(text, encoding="utf-8")
    return f"已替换 {n} 处"


def main() -> int:
    changed, skipped = [], []
    for path in sorted(BENCH.glob("*.py")):
        if path.name in SKIP:
            continue
        result = patch(path)
        if result == "未变":
            skipped.append(path.name)
        else:
            changed.append(f"{path.name}（{result}）")
    print("已改写：")
    for item in changed:
        print("  " + item)
    print("无需改动：")
    for item in skipped:
        print("  " + item)
    # 自检：改完之后不该再有写死的开发树路径
    # （apply_reveal_verdict.py 例外：它刻意留了一条"另一棵树"的兜底路径）
    bad = []
    for path in sorted(BENCH.glob("*.py")):
        if path.name in SELFTEST_SKIP:
            continue
        text = path.read_text(encoding="utf-8")
        if ('parents[1] / "stage"' in text
                or "揭棋\\stage\\Game" in text
                or "揭棋/stage/Game" in text
                or 'Path(r"D:\\py\\Chinese Chess\\Game")' in text):
            bad.append(path.name)
    if bad:
        print("!! 还有写死的路径：" + ", ".join(bad))
        return 1
    print("自检通过：没有脚本再写死开发树路径")
    return 0


if __name__ == "__main__":
    sys.exit(main())
