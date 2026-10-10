# -*- coding: utf-8 -*-
"""``_bench`` 里所有脚本共用的**路径探测**。

为什么需要这个文件：这些脚本原本都写死

    STAGE = Path(__file__).resolve().parents[1] / "stage"
    GAME  = STAGE / "Game"

在开发树 ``D:\\py\\揭棋`` 里跑没问题，但同一个文件被 ``deploy.py`` 复制到
``D:\\py\\Chinese Chess`` 之后，``parents[1] / "stage"`` 指向
``D:\\py\\Chinese Chess\\stage``——那里根本没有 ``stage`` 目录，
于是 ``from jieqi import record`` 直接 ``ModuleNotFoundError``
（用户实际遇到的就是这个：在游戏目录里跑 ``_bench`` 的脚本全用不了）。

这里的做法是**向上找 ``Game`` 包**，两条树的三种布局都能认出来：

    <root>/Game/…                （部署树：D:\\py\\Chinese Chess）
    <root>/stage/Game/…          （开发树：D:\\py\\揭棋）
    <root>/Game/…                （将来把仓库直接放成部署布局）

用法（脚本头部按这个顺序写，后面的 import 才能拿到正确的包）::

    import _paths  # noqa: F401  （或在包内 import _paths as _paths）
    import pygame
    from jieqi import record as R
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["STAGE", "GAME", "ROOT", "JIEQI_OK",
           "add_to_path", "describe"]


def _find_game_dir(start: Path) -> Path:
    """从 ``start`` 往上找第一个含 ``Game/jieqi`` 的目录，返回那个 ``Game``。

    找不到就退回 ``<root>/stage/Game``（保持老行为，让报错现场与以前一致）。
    """
    for base in (start, *start.parents):
        for candidate in (base / "Game", base / "stage" / "Game"):
            if (candidate / "jieqi").is_dir():
                return candidate
    return start / "stage" / "Game"


_HERE = Path(__file__).resolve()
ROOT = _HERE.parent.parent           # 含 ``_bench`` 的那个目录（两条树都是）
GAME = _find_game_dir(ROOT)
STAGE = GAME.parent                  # ``jieqi`` 的父目录（开发树 = stage，部署树 = 根）


def add_to_path(*extra) -> None:
    """把 ``Stage``（= ``jieqi`` 的父目录）与 ``Game`` 放进 ``sys.path``。

    ``extra`` 可以再补几个目录（有些脚本还会 import 同目录的兄弟脚本）。
    """
    for path in (str(STAGE), str(GAME), *(str(p) for p in extra)):
        if path and path not in sys.path:
            sys.path.insert(0, path)


JIEQI_OK = (GAME / "jieqi" / "__init__.py").is_file()

#: 无显示环境也要能 import 游戏模块（pygame 在 import 时就会试着开窗口）
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

add_to_path()


def describe() -> str:
    """一行说明"这次跑的是哪棵树"（脚本可以打出来，免得看错棋谱目录）。"""
    which = "部署树" if STAGE.name != "stage" else "开发树"
    return (f"{which}：游戏目录 {GAME}（jieqi {'可用' if JIEQI_OK else '缺失'}）")
