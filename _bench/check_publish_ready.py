# -*- coding: utf-8 -*-
"""发布前体检：找出"只发布 Game/ 就会坏掉"的地方。

检查项：
1. Game/ 里有没有写死的绝对路径（``D:\\...`` 之类）—— 别人 clone 下去必然打不开；
2. Game/ 有没有依赖 Game/ 之外的东西（tools/、_bench/、tests/）；
3. 依赖哪些第三方包；
4. 有没有明显的凭据残留；
5. 有没有"依赖当前工作目录"的相对资源路径。

用法：``python _bench/check_publish_ready.py``
"""
from __future__ import annotations

import re
import sys

import _paths  # noqa: E402  必须先 import：它会把 Game 加进 sys.path

_paths.add_to_path()
sys.stdout.reconfigure(encoding="utf-8")

GAME = _paths.GAME

ABS_PATH = re.compile(r"[A-Za-z]:[\\/]")
OUTSIDE_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(tools|_bench|tests|wallpaper)\b")
THIRD_PARTY = re.compile(
    r"^\s*(?:import|from)\s+(pygame|numpy|requests|PIL|yaml|openai|chess)\b")
CRED = re.compile(r"sk-[A-Za-z0-9_\-]{16,}|api[_-]?key\s*[=:]\s*[\"'][^\"']{16,}")
#: 字面量里的"带目录的相对路径"（images/x.png 这种才是危险的）
REL_PATH = re.compile(r"""['"]([\w\-]+/[\w\-./]+\.(?:png|jpg|ttf|ttc))['"]""")


def main() -> int:
    print("扫描目录：%s" % GAME)
    files = sorted(GAME.rglob("*.py"))
    print("Python 文件 %d 个\n" % len(files))

    abs_hits, outside, third, creds, rel = [], [], set(), [], []
    for path in files:
        relpath = path.relative_to(GAME)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if ABS_PATH.search(line) and not s.startswith("#"):
                abs_hits.append((relpath, i, s[:95]))
            if OUTSIDE_IMPORT.match(line):
                outside.append((relpath, i, s[:95]))
            m = THIRD_PARTY.match(line)
            if m:
                third.add(m.group(1))
            if CRED.search(line):
                creds.append((relpath, i))
            for m2 in REL_PATH.finditer(line):
                rel.append((relpath, i, m2.group(1)))

    print("=== 1) 写死的绝对路径（非注释行）===")
    for relpath, i, s in abs_hits:
        print("  %s:%d  %s" % (relpath, i, s))
    if not abs_hits:
        print("  （无）")

    print("\n=== 2) 依赖 Game/ 之外的东西 ===")
    for relpath, i, s in outside:
        print("  %s:%d  %s" % (relpath, i, s))
    if not outside:
        print("  （无）")

    print("\n=== 3) 第三方依赖 ===")
    print("  " + (", ".join(sorted(third)) if third else "（无）"))

    print("\n=== 4) 疑似凭据 ===")
    print("  " + ("、".join("%s:%d" % c for c in creds) if creds else "（无）"))

    print("\n=== 5) 依赖工作目录的相对资源路径 ===")
    for relpath, i, value in rel:
        print("  %s:%d  %s" % (relpath, i, value))
    if not rel:
        print("  （无）")

    bad = bool(outside or creds or rel)
    print("\n结论：%s" % ("**有需要处理的问题**" if bad else "可以发布"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
