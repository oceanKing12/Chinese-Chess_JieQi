"""局面评估。

核心思想：**暗子按"它能干什么"估值，明子按真实身份估值**。

揭棋里暗子是按**摆位身份**行棋的（引擎的 ``rule_ident[sq]`` 正是这个），
所以「一枚暗车」在它没被翻开之前，实实在在就是一辆车：能横冲直撞、能吃子。
把它按"剩余暗子池的后验期望"（车/马/炮/兵混在一起，期望只有半辆车）估值，
会得出两个明显错的结论：

1. **翻开是免费的**：翻暗车最差变兵（等价损失 800），期望上却被算成
   "从 529 变成某个身份"——引擎于是特别喜欢开局就翻暗车；
2. **暗子可以随便丢**：一枚暗马被算成 293 而不是 450，弃暗马看着"不亏"。

所以材料项改成"暗子 = 摆位身份的价值"，翻开的代价立刻变成真实的
（翻暗车翻出兵 = 实打实掉 800），"保留暗车、必要时再翻"就成了引擎的自然选择。

> 唯一用到"后验期望"的地方是**将帅安全**里的仕/相数量：暗子对我方是确定的
> （摆位身份），对敌方却是未知的，那里确实只能取期望。

评估项
------
1. 材料 + 位置价值（暗子用摆位身份）
2. 机动性差
3. 受攻 / 被保护
4. 将帅安全：被将军、缺仕相（按期望数量）、敌方对将帅邻格的威胁
5. 先手分（tempo）
"""

from __future__ import annotations

from typing import Dict, Optional

from .belief import Belief
from .board import Board, bit_to_sq
from .consts import (
    A,
    BLACK,
    E,
    EVAL_WEIGHTS,
    HOME_IDENT,
    IDENT_ORDER,
    KING_NEIGH,
    PIECE_VALUE,
    PST_FLAT_SIDE,
    RED,
    SQUARES,
    SQ_BIT,
)

_INF = 10 ** 9

#: ``SQ_BIT[sq] -> sq`` 的查表（位棋盘是稀疏布局，位号 != 格子号）。
_BIT_TO_SQ = {SQ_BIT[sq]: sq for sq in range(SQUARES)}
#: 启动时自检：查表必须与 ``board.bit_to_sq`` 一致，否则遍历掩码会算错格子。
assert all(_BIT_TO_SQ[SQ_BIT[sq]] == bit_to_sq(SQ_BIT[sq]) for sq in range(SQUARES))

#: 明子统计表：``ident -> 下标``（评估里数"某兵种还剩几个"时用，避免字典查找）。
_IDENT_INDEX = {ident: idx for idx, ident in enumerate(IDENT_ORDER)}
_N_IDENTS = len(IDENT_ORDER)


def _bits(mask: int):
    """逐个吐出掩码里的**格子号**。

    ⚠ 位棋盘用的是 ``STRIDE=16`` 的稀疏布局：``SQ_BIT[sq] == 1 << ((sq//9)*16 + sq%9)``，
    所以"位的下标"并不等于"格子号"。凡是遍历位掩码，都必须走 :func:`bit_to_sq`
    （早期版本直接拿 ``bit_length()-1`` 当格子号用，在棋盘下半部分会算出 90 以上
    的下标，直接把 ``side[]`` 索引越界）。
    """
    while mask:
        low = mask & -mask
        yield _BIT_TO_SQ[low]
        mask ^= low


class EvalContext:
    """搜索期间复用的评估上下文（由根节点的 belief 预先算好）。"""

    __slots__ = ("dark_total", "mean_value", "pool_frac", "weights", "belief",
                 "cache", "stats")

    def __init__(self, belief: Belief, weights: Optional[Dict[str, float]] = None,
                 use_cache: bool = True):
        self.belief = belief
        self.weights = dict(EVAL_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.dark_total = belief.dark_total
        self.mean_value = belief.mean_value
        #: ``pool_frac[side][ident]``：一枚未知暗子是 ``ident`` 的概率
        self.pool_frac = {RED: {}, BLACK: {}}
        for side in (RED, BLACK):
            pool = belief.pools.get(side) or []
            n = len(pool) or 1
            for ident in IDENT_ORDER:
                self.pool_frac[side][ident] = pool.count(ident) / n
        #: 局面分缓存（键 = 棋盘 Zobrist）；评估是局面的纯函数，可安全复用。
        #: 这是搜索里最划算的一处优化：迭代加深与转置会把同一局面反复送到
        #: 叶子节点，命中缓存等于白赚一次评估。``use_cache=False`` 时退化成
        #: 每次真算（自对弈 A/B 对照用）。
        self.cache: Dict[int, int] = {} if use_cache else None
        #: ``key -> (分数, {side: (仕数, 相数)})``；比只存分数多带了"明子统计"，
        #: 这样命中缓存时连"数仕相"那两趟扫描也一起省掉。
        self.stats: Dict[int, tuple] = {}


def _piece_value(board: Board, ctx: EvalContext, sq: int, side: int) -> float:
    """某格棋子的评估价值（材料 + 位置）。

    * **暗子**：按 ``rule_ident``（摆位身份）估值——它在翻开之前就是按这个身份
      行棋的，所以这才是它真实的功能价值。用后验期望估会把暗车算成"半辆车"，
      于是引擎会毫无代价地翻开暗车、随手丢暗马（真实反馈里的问题）。
    * **明子**：按真实身份估值。
    """
    if board.hidden[sq]:
        home = board.rule_ident[sq]
        if home:
            return PIECE_VALUE[home] + PST_FLAT_SIDE[side][home][sq]
        # 兜底：没有摆位身份的暗子（只可能来自构造错误的局面）
        v = ctx.dark_total.get(side, {}).get(sq)
        if v is not None:
            return v
        return ctx.mean_value.get(side, 0.0)
    ident = board.ident[sq]
    return PIECE_VALUE[ident] + PST_FLAT_SIDE[side][ident][sq]


def expected_alive(board: Board, ctx: EvalContext, ident: int, side: int) -> float:
    """某方 ``ident`` 的期望存活数量 = 明子实数 + 暗子概率期望。

    这个函数主要给测试/外部调用用；``evaluate`` 内部走的是"一次性统计所有
    兵种"的快路径（见 :func:`_alive_stats`）。
    """
    stats = _alive_stats(board, ctx)
    hidden_n = (board.occ_side[side] & board.hidden_mask).bit_count()
    return (stats[side][_IDENT_INDEX.get(ident, 0)]
            + hidden_n * ctx.pool_frac[side].get(ident, 0.0))


def _alive_stats(board: Board, ctx: EvalContext):
    """一次数完双方各兵种的**明子**数量，并缓存。

    返回 ``{RED: [各兵种明子数...], BLACK: [...]}``，下标与 ``IDENT_ORDER`` 对齐。
    暗子数量不在这里数（用 ``hidden_mask`` 一次 ``bit_count`` 就够，而且暗子
    从不移动，整局都不变）。
    """
    cache = ctx.stats
    key = board.key
    hit = cache.get(key)
    if hit is not None:
        return hit[1]
    counts = {RED: [0] * _N_IDENTS, BLACK: [0] * _N_IDENTS}
    side_arr = board.side
    ident = board.ident
    hidden = board.hidden
    for sq in _bits(board.occ):
        s = side_arr[sq]
        if s < 0 or hidden[sq]:
            continue
        idx = _IDENT_INDEX.get(ident[sq])
        if idx is not None:
            counts[s][idx] += 1
    stats = (None, counts, None)
    cache[key] = stats
    return counts


def evaluate(board: Board, ctx: EvalContext) -> int:
    """返回**红方视角**的局面分（厘兵）。

    性能注意：这个函数是搜索引擎最热的路径（一次 2 秒搜索里会被调用上万次），
    所以刻意避免"每格都过一遍"的写法：

    * 材料/位置分直接用位掩码只遍历**有子的格**；
    * 受攻/被保护也只遍历"确实被打到的格"（``att[敌方]`` 的位）；
    * 结果按 ``board.key`` 缓存在 :class:`EvalContext` 里——同一个局面被反复
      评估（迭代加深、不同着法转置到同一局面）时直接命中缓存。
    """
    cache = ctx.cache
    stats_cache = ctx.stats
    key = board.key
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    counts = stats_cache.get(key)
    if counts is not None:
        counts = counts[1]

    w = ctx.weights
    side = board.side
    occ_side = board.occ_side

    # 1) 材料 + 位置（只遍历有子的格）
    values: Dict[int, float] = {}
    score = 0.0
    for sq in _bits(occ_side[RED] | occ_side[BLACK]):
        s = side[sq]
        if s < 0:
            continue
        v = _piece_value(board, ctx, sq, s)
        values[sq] = v
        score += v if s == RED else -v

    # 2) 攻击图（机动性 / 威胁 / 将帅危险度共用）
    att_red = board.attack_map(RED)
    att_black = board.attack_map(BLACK)
    att = (att_red, att_black)

    mob = ((att_red & ~occ_side[RED]).bit_count()
           - (att_black & ~occ_side[BLACK]).bit_count())
    score += w["mobility"] * mob

    # 3) 受攻 / 被保护（只遍历被打到的格）。
    #    注意 ``_bits`` 吐出来的是**格子号**，所以要用 ``SQ_BIT[sq]`` 做位判断。
    hang = w["hanging"]
    defd = w["attacked_defended"]
    for s in (RED, BLACK):
        for sq in _bits(att[1 - s] & (occ_side[RED] | occ_side[BLACK])):
            if side[sq] != s:
                continue
            v = values.get(sq)
            if v is None:
                v = _piece_value(board, ctx, sq, s)
                values[sq] = v
            loss = v * (defd if (att[s] & SQ_BIT[sq]) else hang)
            score += loss if s == BLACK else -loss

    # 4) 将帅安全
    if counts is None:
        counts = _alive_stats(board, ctx)
    # 暗子数量用一位掩码一次数完（暗子永不移动，数量整局不变）
    mask = board.hidden_mask
    hidden_n = {RED: (occ_side[RED] & mask).bit_count(),
                BLACK: (occ_side[BLACK] & mask).bit_count()}
    idx_a, idx_e = _IDENT_INDEX[A], _IDENT_INDEX[E]
    for s in (RED, BLACK):
        ks = board.kings[s]
        if ks < 0:
            score += -_INF if s == RED else _INF
            continue
        danger = (att[1 - s] & KING_NEIGH[ks]).bit_count()
        pen = w["king_danger"] * danger
        advisors = counts[s][idx_a] + hidden_n[s] * ctx.pool_frac[s].get(A, 0.0)
        elephants = counts[s][idx_e] + hidden_n[s] * ctx.pool_frac[s].get(E, 0.0)
        pen += max(0.0, 2 - advisors) * w["missing_advisor"]
        pen += max(0.0, 2 - elephants) * w["missing_elephant"]
        score += -pen if s == RED else pen

    # 被将军（合法局面里只可能出现在走子方）
    turn = board.turn
    if board.in_check(turn):
        score += -w["in_check"] if turn == RED else w["in_check"]

    # 5) 先手分
    score += w["tempo"] if turn == RED else -w["tempo"]

    result = int(score)
    if cache is not None:
        if len(cache) > 400_000:
            cache.clear()
        cache[key] = result
    return result
