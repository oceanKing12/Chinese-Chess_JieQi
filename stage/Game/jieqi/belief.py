"""暗子后验分布（belief）。

揭棋的信息结构
--------------
* 走法、将军、胜负只依赖公开信息（见 ``board.py``）。
* 唯一隐藏的是**尚未翻开的棋子的真实身份**，而且——
  按通行规则，被吃掉的暗子**仍然保持未知**（双方和旁观者都不得翻看，
  "临局双方及旁观者皆不得揭看棋子"）。因此本模块默认**不使用**被吃暗子的
  真实身份，保证 AI 与人类棋手的信息完全对称；确实想启用时可以传
  ``use_hidden_captures=True``。

后验分布怎么算
--------------
一方 15 枚暗子的身份是子力池的一个均匀随机双射。已知信息：
* 已翻开的活子 -> 身份已知，从池中扣除；
* 被吃掉的**明子** -> 身份已知，从池中扣除；
* 被吃掉的**暗子** -> 身份未知，只减少“未知位置”的个数，不减少身份池。

于是令 ``unknown_positions = 暗子格数 + 被吃暗子数``，身份池
``pool = 全部 15 枚 - 已翻开活子 - 被吃明子``。两者数量相等即自洽。
每个暗子格的边缘分布 = 池中各身份的概率 = ``count(ident) / unknown_positions``。

* 线性评估取边缘期望即得到**精确的期望分值**（``EvalContext``）；
* 需要搜索“翻开之后会怎样”时，从池中无放回抽 ``暗子格数`` 个身份做定局化
  （``apply_sample``），这正是后验联合分布的一次采样。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from .consts import (
    BLACK,
    FULL_POOL,
    IDENT_CHARS,
    IDENT_ORDER,
    K,
    PIECE_VALUE,
    PST_FLAT_SIDE,
    RED,
    SQUARES,
)

POOL_SIZE = len(FULL_POOL)


class Belief:
    """双方暗子身份池 + 暗子格集合 + 未知被吃子数量。"""

    __slots__ = ("pools", "dark", "unknown_positions", "mean_value",
                 "dark_total", "note", "captured_hidden")

    def __init__(self, pools: Dict[int, List[int]], dark: Dict[int, List[int]],
                 captured_hidden: Optional[Dict[int, int]] = None,
                 note: str = "") -> None:
        self.pools = pools
        self.dark = dark
        self.captured_hidden = captured_hidden or {RED: 0, BLACK: 0}
        self.note = note
        self.unknown_positions: Dict[int, int] = {}
        self.mean_value: Dict[int, float] = {}
        self.dark_total: Dict[int, Dict[int, float]] = {}
        self._build_tables()

    # ------------------------------------------------------------ 构造
    @classmethod
    def from_board(cls, board,
                   captured: Optional[Dict[int, List[int]]] = None,
                   captured_hidden: Optional[Dict[int, int]] = None,
                   use_hidden_captures: bool = False,
                   trust_revealed: bool = True) -> "Belief":
        """从棋盘（只看公开信息）推断暗子身份池。

        ``captured[side]``：被吃掉棋子的身份列表。默认只包含**明子**；
        若 ``use_hidden_captures=True``，调用方可以把被吃暗子的真值也放进来
        （那样信息比人类多，属于作弊，默认关闭）。

        ``captured_hidden[side]``：被吃掉的**暗子**数量（身份未知）。
        """
        captured = captured or {}
        captured_hidden = captured_hidden or {}
        pools: Dict[int, List[int]] = {}
        dark: Dict[int, List[int]] = {}
        hidden_counts: Dict[int, int] = {}
        notes = []
        for side in (RED, BLACK):
            pool = list(FULL_POOL)
            squares = []
            for sq in range(SQUARES):
                if board.side[sq] != side:
                    continue
                if board.hidden[sq]:
                    squares.append(sq)
                elif trust_revealed:
                    ident = board.ident[sq]
                    if ident == K:
                        continue  # 将帅永远明牌，本来就不在子力池里
                    if ident in pool:
                        pool.remove(ident)
                    else:  # 信息不一致，尽量兜底
                        notes.append(f"明子{ident}不在池中")
            for ident in captured.get(side, ()):  # 被吃掉的**明子**身份已知
                if ident in pool:
                    pool.remove(ident)
            dark[side] = sorted(squares)
            hidden_counts[side] = int(captured_hidden.get(side, 0) or 0)

            unknown = len(dark[side]) + hidden_counts[side]
            n_pool = len(pool)
            if unknown == 0:
                pool = []
            elif n_pool > unknown:
                # 信息不全（例如中途载入局面）：随机裁掉多余身份
                random.Random(0xC0FFEE).shuffle(pool)
                pool = pool[:unknown]
                notes.append(f"{side}:身份池多于未知位置 {n_pool}>{unknown}，已裁剪")
            elif n_pool < unknown:
                refill = list(FULL_POOL)
                i = 0
                while len(pool) < unknown:
                    pool.append(refill[i % len(refill)])
                    i += 1
                notes.append(f"{side}:身份池少于未知位置 {n_pool}<{unknown}，已补齐")
            pools[side] = pool
        return cls(pools, dark, hidden_counts, "; ".join(notes))

    # ------------------------------------------------------------ 预计算
    def _build_tables(self) -> None:
        for side in (RED, BLACK):
            pool = self.pools[side]
            n_pos = len(self.dark[side]) + self.captured_hidden.get(side, 0)
            self.unknown_positions[side] = max(1, n_pos)
            if not pool:
                self.mean_value[side] = 0.0
                self.dark_total[side] = {}
                continue
            denom = float(max(1, len(pool)))
            #: 一枚未知暗子的期望材料价值
            self.mean_value[side] = sum(PIECE_VALUE[i] for i in pool) / denom
            tables = PST_FLAT_SIDE[side]
            row: Dict[int, float] = {}
            for sq in self.dark[side]:
                row[sq] = sum(PIECE_VALUE[i] + tables[i][sq] for i in pool) / denom
            self.dark_total[side] = row

    def frac(self, side: int, ident: int) -> float:
        """一枚未知暗子是 ``ident`` 的概率。"""
        pool = self.pools[side]
        if not pool:
            return 0.0
        return pool.count(ident) / len(pool)

    # ------------------------------------------------------------ 采样
    def apply_sample(self, board, rng: random.Random) -> None:
        """按后验分布给 ``board`` 的暗子格填身份（原地修改 ``ident``）。

        只改 ``ident``，不动 ``hidden`` / ``rule_ident`` / 哈希：
        定局化对公开信息没有影响。
        """
        for side in (RED, BLACK):
            squares = [sq for sq in self.dark[side]
                       if board.hidden[sq] and board.side[sq] == side]
            if not squares:
                continue
            pool = self.pools[side]
            if not pool:
                continue
            if len(pool) >= len(squares):
                chosen = rng.sample(pool, len(squares))
            else:
                chosen = [pool[rng.randrange(len(pool))] for _ in squares]
            for sq, ident in zip(squares, chosen):
                board.ident[sq] = ident

    def sample_assignment(self, rng: random.Random) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for side in (RED, BLACK):
            squares = self.dark[side]
            pool = self.pools[side]
            if not pool:
                continue
            chosen = (rng.sample(pool, len(squares)) if len(pool) >= len(squares)
                      else [pool[rng.randrange(len(pool))] for _ in squares])
            for sq, ident in zip(squares, chosen):
                out[sq] = ident
        return out

    # ------------------------------------------------------------ 查询
    def remaining(self, side: int) -> List[int]:
        return list(self.pools[side])

    def describe(self) -> str:
        def fmt(side: int) -> str:
            pool = sorted(self.pools[side])
            text = "".join(IDENT_CHARS[(side, i)] for i in pool)
            extra = self.captured_hidden.get(side, 0)
            if extra:
                text += f"（另有 {extra} 枚被吃暗子身份未知）"
            return text or "无"

        return (f"红方未明子力：{fmt(RED)} | 黑方未明子力：{fmt(BLACK)}"
                + (f" | 备注：{self.note}" if self.note else ""))
