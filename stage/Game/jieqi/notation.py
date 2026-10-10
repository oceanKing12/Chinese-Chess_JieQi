# -*- coding: utf-8 -*-
"""棋盘坐标记法 —— **全项目唯一的真相来源**。

为什么要有这个模块
==================

同一格棋子在项目里出现过三种写法：界面记谱、棋谱 JSON、以及引擎内部的格号。
三种写法各写各的，就会出现"记录里是 (6,0)、引擎里是 a3、界面上是'行6列0'"，
对照一次要换算三次，而且**极易搞反上下**。所以统一成下面这一套：

* **列（file）用 ``a``~``i``**：从红方视角**从左到右**。本项目内部
  ``col 0`` 就是最左边，所以 ``file = "abcdefghi"[col]``，不需要翻转；
* **行（rank）用 ``0``~``9``**：从**下往上**数，红方底线是 rank 0、
  黑方底线是 rank 9。写成坐标时直接写数字（``a3``，上限是 ``i9``）。
* 一格记作 ``b3``；一手记作 ``b3c3``（与 UCI 的 ``bestmove b3c3`` 一致）。

**行号是 0~9 而不是 1~10**：rank 0 是合法值（开局常见 ``f9e8``、``b0c2``）。

与本项目**内部**行号的关系（内部 ``row 9`` = 红方底线）::

    rank = 9 - row        row = 9 - rank

内部 ``row 0``（红帅、黑将那条底线之一）对应 rank 9，``row 9`` 对应 rank 0。

屏上看到的上下顺序由渲染层的 ``player_side`` 翻转决定，与记法无关。

模块内的函数一律**不抛异常**（越界/解析失败返回 ``None`` 或占位串），
因为它们的调用方是渲染与棋谱：宁可显示 ``"??"`` 也不能因为一个坏坐标
把整局棋的界面弄崩。
"""
from __future__ import annotations

from typing import Optional, Tuple

__all__ = [
    "ROWS", "COLS", "FILES", "FILE_LETTERS",
    "row_rank", "rank_row", "col_file", "file_col",
    "square_name", "name_to_square", "square_name_cn",
    "move_name", "move_str", "parse_move_name", "parse_move_str",
    "is_move_name", "square_to_xy", "xy_to_square",
    "parse_legacy_move", "parse_any_move",
]

ROWS = 10
COLS = 9

#: 列字母（红方视角左 -> 右）与列号（中文记谱习惯，右边是"一"）
FILE_LETTERS = "abcdefghi"
FILES = FILE_LETTERS

#: 兼容别名：有些地方按"第几列"来数（0 基）
#: 行号是 **0~9**（红方底线 = rank 0），所以 内部 row = 9 - rank。
row_rank = lambda row: 9 - int(row)                      # noqa: E731
rank_row = lambda rank: 9 - int(rank)                    # noqa: E731
col_file = lambda col: FILE_LETTERS[int(col)]            # noqa: E731
file_col = lambda text: FILE_LETTERS.find(str(text).lower())  # noqa: E731


def _valid(row, col) -> bool:
    try:
        return 0 <= int(row) < ROWS and 0 <= int(col) < COLS
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 一格
# ---------------------------------------------------------------------------
def square_name(row, col) -> str:
    """``(row, col)`` -> ``"b3"`` / ``"i9"``；越界返回 ``"??"``。

    ``rank = 9 - row``：内部 row 9（红方底线）是 rank **0**、
    row 0（黑方底线）是 rank **9**。
    """
    if not _valid(row, col):
        return "??"
    return "%s%d" % (FILE_LETTERS[int(col)], 9 - int(row))


def name_to_square(text) -> Optional[Tuple[int, int]]:
    """``"b3"`` -> ``(row, col)``；解析失败返回 ``None``。

    宽容度：大小写、前后空白都收。行号范围 **0~9**（与引擎一致），
    越界一律返回 ``None``（``a10`` 就是坏数据）。
    """
    raw = str(text or "").strip().lower()
    if len(raw) != 2:
        return None
    col = FILE_LETTERS.find(raw[0])
    if col < 0:
        return None
    if not raw[1].isdigit():
        return None
    row = 9 - int(raw[1])
    if not _valid(row, col):
        return None
    return row, col


def square_name_cn(row, col) -> str:
    """中文写法 ``"b3"`` 的等价（行列都用数字，便于和界面上的坐标轴对齐）。"""
    return square_name(row, col)


def square_to_xy(sq: int) -> Tuple[int, int]:
    """引擎格号（``row * 9 + col``）-> ``(row, col)``。"""
    sq = int(sq)
    return sq // COLS, sq % COLS


def xy_to_square(row: int, col: int) -> int:
    """``(row, col)`` -> 引擎格号。"""
    return int(row) * COLS + int(col)


# ---------------------------------------------------------------------------
# 一手
# ---------------------------------------------------------------------------
def move_name(src, dst=None) -> str:
    """两手写法：``move_name((6, 0), (5, 0)) -> "a3a4"``。

    也接受单个"引擎着法整数"（``(from << 7) | to``）。
    """
    if dst is None:
        try:
            mv = int(src)
        except (TypeError, ValueError):
            return "????"
        return move_name(square_to_xy(mv >> 7), square_to_xy(mv & 127))
    return "%s%s" % (square_name(*src), square_name(*dst))


def move_str(src, dst=None) -> str:
    """``move_name`` 的别名（老代码里叫 ``move_str``，保留以免到处改调用点）。"""
    return move_name(src, dst)


def parse_move_name(text) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """``"b3c3"`` -> ``((6, 1), (6, 2))``；解析失败返回 ``None``。

    也接受 ``"b3-c3"`` / ``"b3 c3"`` / ``"b3->c3"`` 这些分隔写法。
    行号可能是 1 位也可能是 2 位（``a10``），所以两种切分都试一遍。
    """
    raw = str(text or "").strip().lower().replace("->", "").replace("-", "") \
        .replace(" ", "").replace("_", "")
    if len(raw) != 4:                    # 一格固定 2 个字符（字母 + 数字）
        return None
    frm, to = name_to_square(raw[:2]), name_to_square(raw[2:])
    if frm is None or to is None:
        return None
    return frm, to


def parse_two(token: str) -> Optional[Tuple[int, int]]:
    """``"a3"`` -> ``(row, col)``（``parse_move_name`` 内部用，也对外可用）。"""
    return name_to_square(token)


def parse_move_str(text) -> Optional[int]:
    """``"b3c3"`` -> 引擎着法整数；解析失败返回 ``None``。"""
    got = parse_move_name(text)
    if not got:
        return None
    frm, to = got
    return (xy_to_square(*frm) << 7) | xy_to_square(*to)


def is_move_name(text) -> bool:
    """``text`` 是否是合法的着法写法（解析得出来而且两格都合法）。"""
    return parse_move_name(text) is not None


def parse_legacy_move(text) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """老写法 ``"(6,0)->(5,0)"`` -> ``((6,0),(5,0))``；解析失败返回 ``None``。

    2026-10 之前棋谱里记的就是这种"行列都从 0 数、上面是行 0"的写法。
    留一个解析入口是为了**老棋谱还能导入**——历史数据不该因为换了记法就作废。
    """
    import re

    nums = re.findall(r"\d+", str(text or ""))
    if len(nums) < 4:
        return None
    fr, fc, tr, tc = (int(x) for x in nums[:4])
    if not (_valid(fr, fc) and _valid(tr, tc)):
        return None
    return (fr, fc), (tr, tc)


def parse_any_move(text) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """先按新记法解析，不行再按老记法解析（给棋谱导入用）。"""
    got = parse_move_name(text)
    if got is not None:
        return got
    return parse_legacy_move(text)
