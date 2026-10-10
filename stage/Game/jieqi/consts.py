"""揭棋（暗象棋 / Jieqi）引擎常量、棋盘布局与预计算表。

坐标约定
--------
* 对外（GUI / 测试）使用 ``sq = r * 9 + c``，``r`` 为 0..9 行（0 是红方底线），
  ``c`` 为 0..8 列。
* 引擎内部位棋盘使用 ``bit = r * 16 + c``（160 位整数），这样纵向平移就是
  ``<< 16`` / ``>> 16``，横向平移是 ``<< 1`` / ``>> 1`` 且列 9..15 永远为空，
  横向不会绕行到下一行。

揭棋核心规则（本引擎实现）
--------------------------
1. 除将/帅外，双方各 15 枚棋子全部背面朝上（暗子）。
2. **暗子的走法由它所在格子的“摆位身份”决定**，而不是它真正是什么棋子。
   例如摆在炮位的暗子按炮走，摆在兵位的暗子按兵走。
3. 暗子第一次移动时翻开；翻开之后永远按它**真实身份**的规则走。
4. 因此暗子只可能存在于它的初始格子上——一旦移动必然翻开。
   这条不变量是引擎高性能的关键：``rule_ident[sq] = ident[sq] if revealed else HOME_IDENT[sq]``。
5. 谁先被将死 / 老将被吃谁就输，将帅照面（对面笑）视为非法局面。
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 基本常量
# --------------------------------------------------------------------------
RED = 0
BLACK = 1

#: 棋子身份（类型）。红黑共用同一套类型编码，只有汉字写法不同。
K, A, E, H, R, C, P = 1, 2, 3, 4, 5, 6, 7

IDENT_ORDER = (K, A, E, H, R, C, P)

#: 每方除将帅之外的子力池：仕/士×2、相/象×2、车×2、马×2、炮×2、兵×5
FULL_POOL = (A, A, E, E, H, H, R, R, C, C, P, P, P, P, P)

IDENT_CHARS = {
    (RED, K): "帅", (RED, A): "仕", (RED, E): "相", (RED, H): "马",
    (RED, R): "车", (RED, C): "炮", (RED, P): "兵",
    (BLACK, K): "将", (BLACK, A): "士", (BLACK, E): "象", (BLACK, H): "馬",
    (BLACK, R): "車", (BLACK, C): "砲", (BLACK, P): "卒",
}
CHAR_TO_PIECE = {v: (s, i) for (s, i), v in IDENT_CHARS.items()}
SIDE_NAMES = {RED: "red", BLACK: "black"}
SIDE_NAMES_CN = {RED: "红", BLACK: "黑"}
NAME_TO_SIDE = {"red": RED, "black": BLACK, "红": RED, "黑": BLACK}

#: 棋子基础价值（厘兵 / centipawn）。将帅不参与材料分，单独处理。
PIECE_VALUE = {K: 0, A: 200, E: 200, H: 450, R: 900, C: 450, P: 100}

#: 将帅被吃 / 将死的分值。
MATE_SCORE = 1_000_000
#: 搜索中判定“必败/必胜”的分数窗口。
MATE_BOUND = MATE_SCORE - 10_000

#: 棋盘尺寸
ROWS, COLS = 10, 9
SQUARES = ROWS * COLS
STRIDE = 16


# --------------------------------------------------------------------------
# 摆位身份：每个格子初始放的是什么棋子（暗子按这个身份行棋）
# --------------------------------------------------------------------------
_LAYOUT_ROWS = [
    [R, H, E, A, K, A, E, H, R],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, C, 0, 0, 0, 0, 0, C, 0],
    [P, 0, P, 0, P, 0, P, 0, P],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [P, 0, P, 0, P, 0, P, 0, P],
    [0, C, 0, 0, 0, 0, 0, C, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [R, H, E, A, K, A, E, H, R],
]

#: 每格的摆位身份（0 表示空格）。
HOME_IDENT = [0] * SQUARES
#: 每格的归属方（0 红 / 1 黑）。
HOME_SIDE = [RED] * SQUARES
for _r, _row in enumerate(_LAYOUT_ROWS):
    for _c, _ident in enumerate(_row):
        HOME_IDENT[_r * 9 + _c] = _ident
        HOME_SIDE[_r * 9 + _c] = RED if _r < 5 else BLACK
del _r, _row, _c, _ident

#: 每方 15 个暗子格（摆位身份的格子，去掉将帅）
HOME_DARK_SQUARES = {RED: [], BLACK: []}
for _sq in range(SQUARES):
    if HOME_IDENT[_sq] not in (0, K):
        HOME_DARK_SQUARES[HOME_SIDE[_sq]].append(_sq)
del _sq

#: 九宫范围
PALACE = {
    RED: {(r, c) for r in range(0, 3) for c in range(3, 6)},
    BLACK: {(r, c) for r in range(7, 10) for c in range(3, 6)},
}


# --------------------------------------------------------------------------
# 位置价值表（从红方视角，单位厘兵；黑方按 180° 旋转取值）
# --------------------------------------------------------------------------
PST = {}

PST[P] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [10, 20, 30, 40, 50, 40, 30, 20, 10],
    [20, 35, 50, 65, 80, 65, 50, 35, 20],
    [30, 45, 60, 85, 100, 85, 60, 45, 30],
    [40, 55, 70, 95, 115, 95, 70, 55, 40],
    [45, 60, 75, 100, 120, 100, 75, 60, 45],
    [50, 65, 80, 105, 125, 105, 80, 65, 50],
]

PST[R] = [
    [0, 0, 10, 15, 15, 15, 10, 0, 0],
    [10, 15, 20, 25, 25, 25, 20, 15, 10],
    [10, 20, 25, 30, 35, 30, 25, 20, 10],
    [15, 25, 30, 35, 40, 35, 30, 25, 15],
    [15, 30, 35, 40, 45, 40, 35, 30, 15],
    [15, 30, 35, 40, 45, 40, 35, 30, 15],
    [15, 30, 35, 40, 45, 40, 35, 30, 15],
    [15, 25, 30, 35, 40, 35, 30, 25, 15],
    [10, 20, 25, 30, 35, 30, 25, 20, 10],
    [5, 10, 15, 20, 25, 20, 15, 10, 5],
]

PST[H] = [
    [0, -5, 0, 0, 0, 0, 0, -5, 0],
    [-5, 5, 10, 10, 10, 10, 10, 5, -5],
    [0, 10, 20, 25, 25, 25, 20, 10, 0],
    [5, 15, 25, 35, 40, 35, 25, 15, 5],
    [10, 20, 30, 45, 50, 45, 30, 20, 10],
    [15, 25, 35, 50, 55, 50, 35, 25, 15],
    [15, 30, 40, 55, 60, 55, 40, 30, 15],
    [15, 30, 40, 55, 60, 55, 40, 30, 15],
    [10, 20, 30, 40, 45, 40, 30, 20, 10],
    [5, 10, 20, 25, 30, 25, 20, 10, 5],
]

PST[C] = [
    [5, 5, 5, 10, 10, 10, 5, 5, 5],
    [5, 10, 15, 20, 20, 20, 15, 10, 5],
    [5, 10, 20, 25, 30, 25, 20, 10, 5],
    [5, 15, 25, 30, 35, 30, 25, 15, 5],
    [5, 15, 25, 30, 40, 30, 25, 15, 5],
    [5, 15, 25, 30, 40, 30, 25, 15, 5],
    [5, 15, 25, 30, 35, 30, 25, 15, 5],
    [5, 10, 20, 25, 30, 25, 20, 10, 5],
    [5, 10, 15, 20, 20, 20, 15, 10, 5],
    [0, 0, 5, 10, 10, 10, 5, 0, 0],
]

PST[A] = [
    [0, 0, 0, 10, 0, 10, 0, 0, 0],
    [0, 0, 0, 0, 25, 0, 0, 0, 0],
    [0, 0, 0, 10, 0, 10, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

PST[E] = [
    [0, 0, 10, 0, 0, 0, 10, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [10, 0, 0, 0, 20, 0, 0, 0, 10],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 10, 0, 0, 0, 10, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

PST[K] = [
    [0, 0, 0, 15, 20, 15, 0, 0, 0],
    [0, 0, 0, 5, 10, 5, 0, 0, 0],
    [0, 0, 0, -10, -5, -10, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -10, -5, -10, 0, 0, 0],
    [0, 0, 0, 5, 10, 5, 0, 0, 0],
    [0, 0, 0, 15, 20, 15, 0, 0, 0],
]

#: 摊平成一维的 ``PST_FLAT[ident][sq]``，已按行主序展开，仅红方视角。
PST_FLAT = {i: [0] * SQUARES for i in IDENT_ORDER}
for _ident, _table in PST.items():
    for _r in range(ROWS):
        for _c in range(COLS):
            PST_FLAT[_ident][_r * 9 + _c] = _table[_r][_c]
del _ident, _table, _r, _c

#: ``PST_FLAT_SIDE[side][ident][sq]``：黑方用 180° 旋转后的表。
PST_FLAT_SIDE = {RED: PST_FLAT}
_black_tables = {}
for _ident in IDENT_ORDER:
    _src = PST_FLAT[_ident]
    _dst = [0] * SQUARES
    for _sq in range(SQUARES):
        _r, _c = divmod(_sq, 9)
        _dst[_sq] = _src[(9 - _r) * 9 + (8 - _c)]
    _black_tables[_ident] = _dst
PST_FLAT_SIDE[BLACK] = _black_tables
del _ident, _src, _dst, _sq, _r, _c, _black_tables

#: 机动性权重：每种棋子每多一个可走点值多少厘兵。
MOBILITY_WEIGHT = {K: 0, A: 1, E: 1, H: 4, R: 3, C: 3, P: 2}


# --------------------------------------------------------------------------
# 评估权重（可调；供自对弈调参使用）
# --------------------------------------------------------------------------
EVAL_WEIGHTS = {
    #: 被攻击且无保护：扣自身价值的这个比例
    "hanging": 0.30,
    #: 被攻击但有保护：扣自身价值的这个比例
    "attacked_defended": 0.08,
    #: 机动性：每多控制一个格子（相对对手）加多少厘兵
    "mobility": 2,
    #: 被将军的固定扣分
    "in_check": 45,
    #: 少一个仕/士 的扣分（按期望数量计算）
    "missing_advisor": 25,
    #: 少一个相/象 的扣分（按期望数量计算）
    "missing_elephant": 12,
    #: 敌方攻击到将帅邻格的危险度（每格）
    "king_danger": 4,
    #: 轮到己方走棋的先手分
    "tempo": 8,
}


# --------------------------------------------------------------------------
# 几何 / 预计算
# --------------------------------------------------------------------------
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
DIR_POSITIVE = (False, True, False, True)
DIR_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

#: 格子编号 -> 位掩码
SQ_BIT = [1 << ((sq // 9) * STRIDE + (sq % 9)) for sq in range(SQUARES)]

#: 四条射线上的所有格子（不含起点），只是位掩码。
RAY = [[0] * 4 for _ in range(SQUARES)]
#: 四条射线上格子的有序列表（由近到远），走子时按列表前进遇到阻挡即停。
RAYSQ = [[()] * 4 for _ in range(SQUARES)]
for _sq in range(SQUARES):
    _r, _c = divmod(_sq, 9)
    for _d in DIRECTIONS:
        _dr, _dc = DIR_DELTA[_d]
        _mask = 0
        _seq = []
        _rr, _cc = _r + _dr, _c + _dc
        while 0 <= _rr < ROWS and 0 <= _cc < COLS:
            _mask |= SQ_BIT[_rr * 9 + _cc]
            _seq.append(_rr * 9 + _cc)
            _rr += _dr
            _cc += _dc
        RAY[_sq][_d] = _mask
        RAYSQ[_sq][_d] = tuple(_seq)
del _sq, _r, _c, _d, _dr, _dc, _mask, _rr, _cc, _seq


def _leaper_moves(sq: int, offsets) -> list[tuple[int, int | None]]:
    """把 (dr, dc) / (dr, dc, leg_dr, leg_dc) 形式的偏移展开成合法目标。"""
    r, c = divmod(sq, 9)
    out = []
    for off in offsets:
        if len(off) == 2:
            dr, dc = off
            ldr = ldc = None
        else:
            dr, dc, ldr, ldc = off
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            continue
        eye = None
        if ldr is not None:
            lr, lc = r + ldr, c + ldc
            eye = lr * 9 + lc
        out.append((nr * 9 + nc, eye))
    return out


_KING_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_ADVISOR_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ELEPHANT_OFFSETS = [(-2, -2, -1, -1), (-2, 2, -1, 1), (2, -2, 1, -1), (2, 2, 1, 1)]
_KNIGHT_OFFSETS = [
    (-2, -1, -1, 0), (-2, 1, -1, 0), (2, -1, 1, 0), (2, 1, 1, 0),
    (-1, -2, 0, -1), (1, -2, 0, -1), (-1, 2, 0, 1), (1, 2, 0, 1),
]

#: 每种棋子的走法表 ``MOVES[ident][side][sq] -> list[(to_sq, eye_sq|None)]``
#:
#: 揭棋的关键规则差异（依据世界象棋联合会/象棋程序维基与国内通行规则）：
#: * **明士、明象可以过河、可以出九宫**（"士相可以过河"），只保留本身的形状限制
#:   （士斜走一格、象走田字且塞象眼）。
#: * **暗士** 的第一步仍限在自家九宫内斜走一格（"暗士：第一步限九宫内沿斜线移动"）。
#: * 暗象按田字走，规则文本未加过河限制，因此与明象一致。
#: 所以士需要两张表，象只有一张。
MOVES = {
    K: {RED: [], BLACK: []},
    A: {RED: [], BLACK: []},
    E: {RED: [], BLACK: []},
    H: {RED: [], BLACK: []},
    P: {RED: [], BLACK: []},
}
#: 暗子专用表（目前只有士与明子不同）
MOVES_HIDDEN = {
    K: {RED: [], BLACK: []},
    A: {RED: [], BLACK: []},
    E: {RED: [], BLACK: []},
    H: {RED: [], BLACK: []},
    P: {RED: [], BLACK: []},
}
for _sq in range(SQUARES):
    _r, _c = divmod(_sq, 9)
    for _side in (RED, BLACK):
        _zone = PALACE[_side]
        in_palace = (_r, _c) in _zone
        # 将帅：永远只能在自家九宫内走一步（离开九宫判负）
        king = ([(t, e) for (t, e) in _leaper_moves(_sq, _KING_OFFSETS)
                 if (t // 9, t % 9) in _zone] if in_palace else [])
        MOVES[K][_side].append(king)
        MOVES_HIDDEN[K][_side].append(king)
        # 士：明士自由斜走一格；暗士限九宫
        advisor_all = _leaper_moves(_sq, _ADVISOR_OFFSETS)
        MOVES[A][_side].append(advisor_all)
        MOVES_HIDDEN[A][_side].append(
            [(t, e) for (t, e) in advisor_all if (t // 9, t % 9) in _zone])
        # 象：田字 + 塞象眼，明暗一致，均可过河
        elephant = _leaper_moves(_sq, _ELEPHANT_OFFSETS)
        MOVES[E][_side].append(elephant)
        MOVES_HIDDEN[E][_side].append(elephant)
    # 马：蹩腿用 eye；明暗一致
    knight = _leaper_moves(_sq, _KNIGHT_OFFSETS)
    MOVES[H][RED].append(knight)
    MOVES[H][BLACK].append(knight)
    MOVES_HIDDEN[H][RED].append(knight)
    MOVES_HIDDEN[H][BLACK].append(knight)
    # 兵卒：过河判定看当前所在行；明暗一致
    for _side in (RED, BLACK):
        if _side == RED:
            crossed = _r >= 5
            offs = [(1, 0)]
        else:
            crossed = _r <= 4
            offs = [(-1, 0)]
        if crossed:
            offs += [(0, -1), (0, 1)]
        pawn = _leaper_moves(_sq, offs)
        MOVES[P][_side].append(pawn)
        MOVES_HIDDEN[P][_side].append(pawn)
del _sq, _r, _c, _side, in_palace, king, advisor_all, elephant, knight, pawn
del crossed, offs, _zone

#: 反向表 ``LEAP_ATTACKERS[side][ident][sq] -> list[(from_sq, eye|None)]``
#: 表示“哪些格子上的该方 ident **明子** 能攻击到 sq”。
LEAP_ATTACKERS = {
    _s: {_i: [[] for _ in range(SQUARES)] for _i in (K, A, E, H, P)}
    for _s in (RED, BLACK)
}
#: 同上，但针对**暗子**（目前只有士不同）。
LEAP_ATTACKERS_HIDDEN = {
    _s: {_i: [[] for _ in range(SQUARES)] for _i in (K, A, E, H, P)}
    for _s in (RED, BLACK)
}
for _s in (RED, BLACK):
    for _i in (K, A, E, H, P):
        for _from in range(SQUARES):
            for _to, _eye in MOVES[_i][_s][_from]:
                LEAP_ATTACKERS[_s][_i][_to].append((_from, _eye))
            for _to, _eye in MOVES_HIDDEN[_i][_s][_from]:
                LEAP_ATTACKERS_HIDDEN[_s][_i][_to].append((_from, _eye))
del _s, _i, _from, _to, _eye

#: 将帅所在格（摆位）
KING_HOME = {RED: 4, BLACK: 85}

#: 将帅周围（含自身）的邻格掩码，用于“将帅危险度”
KING_NEIGH = []
for _sq in range(SQUARES):
    _r, _c = divmod(_sq, 9)
    _m = SQ_BIT[_sq]
    for _dr, _dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)):
        _nr, _nc = _r + _dr, _c + _dc
        if 0 <= _nr < ROWS and 0 <= _nc < COLS:
            _m |= SQ_BIT[_nr * 9 + _nc]
    KING_NEIGH.append(_m)
del _sq, _r, _c, _m, _dr, _dc, _nr, _nc


# --------------------------------------------------------------------------
# Zobrist 哈希
# --------------------------------------------------------------------------
import random as _random  # noqa: E402

_zrng = _random.Random(0x5EED1EAF)
#: ``Z_PIECE[side][ident][sq]``（ident 直接用作下标，0 号位不用）
Z_PIECE = [
    [[_zrng.getrandbits(64) for _ in range(SQUARES)] for _ in range(8)]
    for _ in range(2)
]
#: ``Z_HOME[side][sq]``：暗子只按“某方某格有个暗子”参与哈希
Z_HOME = [[_zrng.getrandbits(64) for _ in range(SQUARES)] for _ in range(2)]
Z_SIDE = _zrng.getrandbits(64)
del _zrng
