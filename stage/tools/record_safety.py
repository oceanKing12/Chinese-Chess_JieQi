# -*- coding: utf-8 -*-
"""棋谱目录的清空/备份工具——**给测试和部署用，防止误删用户棋谱**。

为什么要有这个文件
------------------
开发过程中我曾经在"跑完测试/部署完"之后顺手执行
``Remove-Item Game\\records\\*``，结果把用户真实的棋谱删掉了，而且
``Remove-Item -Force`` **不进回收站**，无法恢复。这是纯粹的破坏，必须堵住。

所以约定：

* **任何**清理动作都不得直接删 ``records/`` 里的文件；要清就先 ``archive()``
  挪到带时间戳的备份目录（``records_archive/``），随时能拿回来；
* 想真删必须显式写 ``purge=True``，并且要打印删了什么；
* 开发/测试请用 ``GAME/records_test`` 之类的**临时目录**，不要碰 ``records/``。

用法::

    python tools/record_safety.py --status            # 看看现有棋谱
    python tools/record_safety.py --archive           # 归档（不删）
    python tools/record_safety.py --archive --purge   # 归档后删（危险，需显式）
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

#: 默认棋谱目录（与 jieqi/record.py 的 DEFAULT_RECORD_DIR 一致）
RECORDS = "records"
#: 归档目录名（与 records 并列）
ARCHIVE = "records_archive"
#: 归档时要跳过、不算"棋谱"的文件
NOT_RECORDS = {"index.json"}


def records_dir(game_dir: Path) -> Path:
    return Path(game_dir) / RECORDS


def list_records(folder: Path):
    """列出棋谱文件（排除索引）。"""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.name not in NOT_RECORDS)


def archive(folder: Path, purge: bool = False) -> dict:
    """把棋谱挪进带时间戳的归档目录（``purge=True`` 时挪完再删归档）。

    返回 ``{ok, moved, archived_to, purged, error}``。**绝不静默删除**。
    """
    items = list_records(folder)
    if not items:
        return {"ok": True, "moved": 0, "archived_to": "", "purged": False,
                "error": ""}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = folder.parent / ARCHIVE / stamp
    try:
        target.mkdir(parents=True, exist_ok=True)
        for item in items:
            shutil.move(str(item), str(target / item.name))
    except OSError as exc:
        return {"ok": False, "moved": 0, "archived_to": str(target),
                "purged": False, "error": str(exc)}
    purged = False
    if purge:
        try:
            shutil.rmtree(str(target))
            purged = True
        except OSError as exc:
            return {"ok": False, "moved": len(items), "archived_to": str(target),
                    "purged": False, "error": str(exc)}
    return {"ok": True, "moved": len(items), "archived_to": str(target),
            "purged": purged, "error": ""}


def main(argv) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-dir", default=None,
                        help="游戏目录（默认自动找 Game）")
    parser.add_argument("--status", action="store_true", help="只列出棋谱")
    parser.add_argument("--archive", action="store_true",
                        help="归档到 records_archive/<时间戳>（不删）")
    parser.add_argument("--purge", action="store_true",
                        help="归档后再删除归档（**真要删**才加，默认不删）")
    args = parser.parse_args(argv)

    game_dir = Path(args.game_dir) if args.game_dir else _guess_game_dir()
    folder = records_dir(game_dir)
    items = list_records(folder)
    print("棋谱目录：%s" % folder)
    print("现有棋谱：%d 个" % len(items))
    for item in items[:20]:
        size = item.stat().st_size if item.is_file() else 0
        print("  %-44s %7d B" % (item.name, size))
    if len(items) > 20:
        print("  …（还有 %d 个）" % (len(items) - 20))
    if args.status or not args.archive:
        return 0

    result = archive(folder, purge=args.purge)
    if not result["ok"]:
        print("归档失败：%s" % result["error"], file=sys.stderr)
        return 1
    print("已归档 %d 个到 %s%s"
          % (result["moved"], result["archived_to"],
             "（归档目录也已删除）" if result["purged"] else "（**没有删除**，随时可拿回）"))
    return 0


def _guess_game_dir() -> Path:
    """从本文件位置往上找 ``Game`` 包目录。

    注意返回的是**含 jieqi 子包的那个 ``Game`` 目录**（棋谱就在它下面的
    ``records/``），不是项目根目录——``--game-dir`` 也按这个口径传。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Game" / "jieqi"
        if candidate.is_dir():
            return parent / "Game"
    return here.parent / "Game"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
