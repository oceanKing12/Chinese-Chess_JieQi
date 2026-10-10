# -*- coding: utf-8 -*-
"""揭棋特有的**搜索行为调整**：暗子阶段 / 剩余身份池 / 翻子安全性。

来源与动机
----------
这段是从开源的 C++/NNUE 揭棋引擎 **AB-JChess**（GPL-3.0，派生自 Pikafish）
的 ``src/reveal.cpp`` **照其数学口径重新实现**的（不是复制代码，量纲与公式对齐）。
它解决的是一类本项目原来完全没有覆盖的问题：

> 揭棋的核心决策是"**什么时候翻、翻哪个**"，但本项目原来的搜索对所有着法一视同仁
> ——翻子着法和普通着法用同一套排序、同一套削减。AB-JChess 的做法是让搜索行为
> **随暗子局势自适应**：暗子还多的时候给翻子着法加分/少削减，暗子快没了就收紧。

四个特征（全部按 AB-JChess 的公式，归一化到 ±1024）
---------------------------------------------------
============  ====================================================
``phase``      全局暗子数 / 30 —— 开局≈1024，完全翻开后≈0
``pool``       剩余身份池里"高于兵的价值"归一化 —— 池子越强越大
``unknown``    池子里还有几个身份是我方不知道的 / 15
``comeback``   当前评估的**劣势程度**（落后越多越大）——"要不要冒险"
============  ====================================================

两组加成
--------
* ``bonus``：加在**翻子着法**的返回值上（`REVEAL_WEIGHTS`）
* ``quiet_bonus``：加在**不吃的翻子着法**上，且区分"翻完安不安全"
  （``safe`` / ``contested`` / ``loose``，见 :func:`quiet_safety`）

再加上两处搜索调整：翻子着法/安全静着的**削减量**、以及**额外剪枝名额**。

> ⚠ 默认权重**全是 0**（= 与原来行为完全一致）。要启用必须显式给权重，
> 这样"改动有没有用"永远可以用自对弈 A/B 量化，而不是靠感觉。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .consts import (
    A,
    BLACK,
    C,
    E,
    H,
    K,
    P,
    PIECE_VALUE,
    R,
    RED,
    SQ_BIT,
)

#: 位掩码（单 bit）-> 格子号。位棋盘是 ``STRIDE=16`` 的稀疏布局，
#: **位号不等于格子号**，必须查表（同类错误在 evaluate.py 里踩过一次）。
_BIT_TO_SQ = {SQ_BIT[sq]: sq for sq in range(90)}

#: 池子里"兵"的价值基准（归一化用；与 AB-JChess 的 PawnValue 同口径）
_PAWN = float(PIECE_VALUE[P])
#: 归一化基准：车 - 兵（AB-JChess 用同一个分母）
_ROOK_PAWN = float(PIECE_VALUE[R] - PIECE_VALUE[P])
#: 全局暗子上限（双方共 30 个暗子格）
_DARK_MAX = 30
#: 归一化用的评估幅度（AB-JChess 里是 800）
_EVAL_SCALE = 800.0

#: 翻子加成权重（对应 AB-JChess 的 RevealBonus*）。全 0 = 关闭。
REVEAL_WEIGHTS: Dict[str, int] = {
    "base": 0,          # 常数项
    "phase": 0,         # × phase
    "pool": 0,          # × pool
    "unknown": 0,       # × unknown（池子里还有多少身份未知）
    "comeback": 0,      # × comeback（落后程度）
    #: 翻子着法的削减量调整（正数 = 少削减 = 搜得更深）
    "reduction": 0,
    #: 不吃的翻子着法的额外削减调整
    "quiet_reduction": 0,
    #: 额外剪枝名额（正数 = 多留几个翻子着法不剪）
    "pruning_margin": 0,
}

#: 静着（不吃的翻子）加成权重（对应 AB-JChess 的 RevealQuiet*）
QUIET_WEIGHTS: Dict[str, int] = {
    "base": 0,
    "phase": 0,
    "safety": 0,
    "high_value": 0,    # 池子里"大子"占比（车/炮/马）
    "diversity": 0,     # 池子身份的分散程度
    "comeback": 0,
}


def _round_div(numerator: float, denominator: float) -> int:
    """四舍五入的整数除法（与 AB-JChess 的 ``round_divide`` 一致）。"""
    if denominator == 0:
        return 0
    quotient = numerator / denominator
    return int(quotient + (0.5 if quotient >= 0 else -0.5))


def total_dark(board) -> int:
    """棋盘上还剩多少个暗子（双方合计）。"""
    return int(board.hidden_mask.bit_count())


def pool_counts(board, side: int) -> Tuple[Dict[int, int], int, float]:
    """推某一方的"剩余身份池"。

    池子 = 开局存量 − **场上明子** − （被吃掉的明子，由调用方通过
    ``captured`` 传进来的话更准）。这里只能看到棋盘，所以给出的是
    "**至少还剩哪些身份**"——被吃的明子已经不在盘上，自然也不会重复计入。

    返回 ``(每种身份的数量, 总数, 高于兵的价值和)``。
    """
    initial = {A: 2, E: 2, H: 2, R: 2, C: 2, P: 5}
    left = dict(initial)
    occ_side = board.occ_side[side]
    hidden = board.hidden
    ident = board.ident
    side_arr = board.side
    mask = occ_side
    while mask:
        low = mask & -mask
        sq = _BIT_TO_SQ[low]
        mask ^= low
        if side_arr[sq] != side or hidden[sq]:
            continue                      # 暗子的身份不占池子（它已经"现身"在盘上）
        got = ident[sq]
        if got in left:
            left[got] -= 1
    for key in list(left):
        if left[key] <= 0:
            del left[key]
        elif left[key] > initial[key]:
            left[key] = initial[key]
    total = sum(left.values())
    above = sum(count * (PIECE_VALUE[ident_] - PIECE_VALUE[P])
                for ident_, count in left.items())
    return left, total, float(above)


def features(board, side: int, eval_red: float = 0.0) -> Dict[str, int]:
    """算出四个特征（归一化到 ±1024，口径与 AB-JChess 一致）。

    ``eval_red`` 是**红方视角**的当前评估，用来算 ``comeback``。
    """
    dark = total_dark(board)
    phase = _round_div(dark * 1024, _DARK_MAX)
    _counts, pool_total, above = pool_counts(board, side)
    denominator = pool_total * _ROOK_PAWN
    pool = _round_div(above * 1024, denominator) if denominator > 0 else 0

    own_dark = int((board.occ_side[side] & board.hidden_mask).bit_count())
    unknown_count = max(0, min(15, pool_total - own_dark))
    unknown = _round_div(unknown_count * 1024, 15)

    # 落到自己头上的劣势：红方视角的评估换成"走子方视角"再取负
    own_eval = eval_red if side == RED else -eval_red
    clamped = max(-_EVAL_SCALE, min(_EVAL_SCALE, float(own_eval)))
    comeback = _round_div(-clamped * 1024, _EVAL_SCALE)
    return {"phase": phase, "pool": pool, "unknown": unknown,
            "comeback": comeback}


def reveal_bonus(feat: Dict[str, int],
                 weights: Optional[Dict[str, int]] = None) -> int:
    """翻子着法的加成（厘兵），限制在 ±512 内（与 AB-JChess 同）。"""
    w = weights if weights is not None else REVEAL_WEIGHTS
    weighted = (int(w.get("phase", 0)) * feat["phase"]
                + int(w.get("pool", 0)) * feat["pool"]
                + int(w.get("unknown", 0)) * feat["unknown"]
                + int(w.get("comeback", 0)) * feat["comeback"])
    value = int(w.get("base", 0)) + _round_div(weighted, 1024)
    return max(-512, min(512, value))


def quiet_safety(board, move: int) -> str:
    """翻子着法（不吃子）走完之后安全吗？

    * ``"safe"``：对方没有子能打到落点 —— 翻出来不怕被抓
    * ``"contested"``：对方能打到，但我方也保护着
    * ``"loose"``：对方能打到且我方没保护 —— 翻出来就送

    判定方式与 AB-JChess 的 ``quiet_safety`` 一致：**在"这一步已经走完"的
    棋盘上看落点**（起点已经空出来），攻方只看**明子**（暗子的身份未知，
    不该当成确定威胁）。
    """
    to = move & 127
    own = board.side[to]
    if own < 0:
        return "safe"
    attacked = False
    defended = False
    for sq in range(90):
        if board.side[sq] < 0 or board.hidden[sq]:
            continue
        if not board.is_attacked(to, board.side[sq]):
            continue
        if board.side[sq] == own:
            defended = True
        else:
            attacked = True
    if not attacked:
        return "safe"
    return "contested" if defended else "loose"


def quiet_bonus(board, side: int, feat: Dict[str, int], safety: str,
                weights: Optional[Dict[str, int]] = None) -> int:
    """不吃的翻子着法的加成（厘兵），限制在 ±192 内。

    ``high_value`` / ``diversity`` 刻画"池子里还有什么牌"：
    池子里大子越多，翻子越有吸引力（翻出好东西的概率高）；
    池子越分散，翻子的方差越大（保守方不喜欢）。
    """
    w = weights if weights is not None else QUIET_WEIGHTS
    counts, pool_total, above = pool_counts(board, side)
    if pool_total > 0:
        high = counts.get(R, 0) + counts.get(C, 0) + counts.get(H, 0)
        centered = 5 * high - 2 * pool_total
        high_value = _round_div(centered * 1024,
                                (3 if centered >= 0 else 2) * pool_total)
        squared = sum(c * c for c in counts.values())
        diversity = _round_div((pool_total * pool_total - 5 * squared) * 256,
                               pool_total * pool_total)
        diversity = max(-1024, min(1024, diversity))
    else:
        high_value = diversity = 0

    safety_score = {"safe": 1024, "contested": 0, "loose": -1024}[safety]
    gate = {"safe": 1024, "contested": 512, "loose": 0}[safety]
    comeback = _round_div(feat["comeback"] * gate, 1024)

    weighted = (int(w.get("phase", 0)) * feat["phase"]
                + int(w.get("safety", 0)) * safety_score
                + int(w.get("high_value", 0)) * high_value
                + int(w.get("diversity", 0)) * diversity
                + int(w.get("comeback", 0)) * comeback)
    value = int(w.get("base", 0)) + _round_div(weighted, 1024)
    return max(-192, min(192, value))


def is_dark_move(board, move: int) -> bool:
    """这一步是不是"翻开一枚暗子"（暗子一移动就会翻开）。"""
    frm = move >> 7
    return bool(board.hidden[frm])


def describe(feat: Dict[str, int]) -> str:
    """一行摘要，便于日志/调试。"""
    return ("阶段 %+5d　池子 %+5d　未知 %+5d　劣势 %+5d"
            % (feat["phase"], feat["pool"], feat["unknown"], feat["comeback"]))
