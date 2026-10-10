"""揭棋位棋盘与走法生成。

设计要点
--------
* 棋盘上的“走法”只依赖**公开信息**：暗子按摆位身份行棋，颜色公开，将帅始终明牌。
  因此走法生成、将军判定、胜负判定都不需要知道暗子的真实身份。真实身份只影响
  “翻开之后的子局面”与“吃子收益”，这两点分别由 belief 与 evaluate 处理。
* ``rule_ident[sq]`` 是核心缓存：明子取真实身份，暗子取摆位身份。
* 位置用位棋盘表示（``bit = r * 16 + c``），轮子用 160 位 Python 整数。
"""

from __future__ import annotations

import random
from typing import List, Optional

from .consts import (
    A,
    BLACK,
    C,
    DIR_POSITIVE,
    DOWN,
    E,
    H,
    HOME_DARK_SQUARES,
    HOME_IDENT,
    IDENT_CHARS,
    K,
    LEAP_ATTACKERS,
    LEAP_ATTACKERS_HIDDEN,
    LEFT,
    MOVES,
    MOVES_HIDDEN,
    P,
    PIECE_VALUE,
    R,
    RAY,
    RAYSQ,
    RED,
    RIGHT,
    ROWS,
    COLS,
    SQ_BIT,
    SQUARES,
    UP,
    Z_HOME,
    Z_PIECE,
    Z_SIDE,
)

# --------------------------------------------------------------------------
# 走法编码：``(from_sq << 7) | to_sq``，两个格子都 < 90 < 128
# --------------------------------------------------------------------------
MOVE_NONE = -1


def encode(frm: int, to: int) -> int:
    return (frm << 7) | to


def move_from(mv: int) -> int:
    return mv >> 7


def move_to(mv: int) -> int:
    return mv & 127


def move_str(mv: int) -> str:
    """引擎着法整数 -> ``"b3c3"``（与 UCI 的 ``bestmove b3c3`` 同一套写法）。

    坐标口径见 :mod:`jieqi.notation`：列 a-i 左到右、行 0-9 下到上
    （红方底线 = 第 0 行）。旧写法是 ``"(6,0)->(5,0)"``，与分析日志对不上，
    对拍一次要换算三次，还容易把上下搞反。
    """
    from .notation import move_name

    return move_name(mv)


def parse_move_str(text) -> Optional[int]:
    """把 ``"b3c3"`` / ``"b3-c3"`` / ``"b3 c3"`` 解析成走法编码。

    同时**兼容老格式** ``"(6,0)->(5,0)"`` / ``"6,0-5,0"``：老棋谱、老日志、
    老测试里都是那种写法，读得回来才不会把历史数据变成废纸。
    """
    from .notation import parse_move_str as _parse_new

    got = _parse_new(text)
    if got is not None:
        return got
    import re

    nums = re.findall(r"\d+", str(text))
    if len(nums) < 4:
        return None
    fr, fc, tr, tc = (int(x) for x in nums[:4])
    if not (0 <= fr < ROWS and 0 <= fc < COLS and 0 <= tr < ROWS and 0 <= tc < COLS):
        return None
    return encode(fr * 9 + fc, tr * 9 + tc)


def bit_to_sq(b: int) -> int:
    """位掩码（单个置位）-> 格子编号。"""
    bi = b.bit_length() - 1
    return (bi >> 4) * 9 + (bi & 15)


def _nearest(blk: int, positive: bool) -> int:
    """取 ``blk`` 里离起点最近的那一位。"""
    if positive:
        return blk & -blk
    return 1 << (blk.bit_length() - 1)


class Board:
    """揭棋棋盘（可变对象；搜索中用 make/unmake 原地修改）。"""

    __slots__ = (
        "ident", "side", "hidden", "rule_ident",
        "occ", "occ_side", "kings", "turn", "key",
        "_undo", "ply", "hidden_mask",
    )

    def __init__(self) -> None:
        self.ident = [0] * SQUARES
        self.side = [-1] * SQUARES
        self.hidden = [False] * SQUARES
        self.rule_ident = [0] * SQUARES
        self.occ = 0
        self.occ_side = [0, 0]
        self.kings = [-1, -1]
        self.turn = RED
        self.key = 0
        self._undo: List[tuple] = []
        self.ply = 0
        #: 所有"暗子初始格"的位掩码（位棋盘里跨行跨方的格子都算）。
        #: 暗子一旦移动就翻开，所以这些格子上的子**永远**是暗的——评估里
        #: 想数"某方还有几枚暗子"，直接和这个掩码取交集即可，不用扫棋盘。
        #: （两方各 15 个暗子格；黑方在上面那半盘，红方在下面那半盘。）
        mask = 0
        for sq in HOME_DARK_SQUARES[RED]:
            mask |= SQ_BIT[sq]
        for sq in HOME_DARK_SQUARES[BLACK]:
            mask |= SQ_BIT[sq]
        self.hidden_mask = mask

    # ------------------------------------------------------------- 构造
    @classmethod
    def empty(cls) -> "Board":
        return cls()

    @classmethod
    def new_game(cls, rng: Optional[random.Random] = None,
                 pool: Optional[List[int]] = None) -> "Board":
        """开新局：把子力池洗牌分配到摆位格（模拟真实的暗子抽签）。"""
        board = cls()
        rng = rng or random.Random()
        for side in (RED, BLACK):
            squares = list(HOME_DARK_SQUARES[side])
            squares.sort()
            bag = list(pool) if pool is not None else [
                A, A, E, E, H, H, R, R, C, C, P, P, P, P, P,
            ]
            rng.shuffle(bag)
            for sq, ident in zip(squares, bag):
                board.put(sq, side, ident, hidden=True)
        board.put(4, RED, K, hidden=False)
        board.put(85, BLACK, K, hidden=False)
        return board

    def put(self, sq: int, side: int, ident: int, hidden: bool = False,
            home_ident: Optional[int] = None) -> None:
        """直接摆放一枚棋子（会同步位棋盘、哈希与将帅位置）。

        ``home_ident`` 用于 GUI 载入时指定暗子的摆位身份；默认取该格子的
        标准摆位身份（正常对局里两者一致——暗子只可能待在初始格子上，
        因为它一移动就会翻开）。

        若既没有 ``home_ident`` 又不在初始格子上（只可能来自构造错误的局面），
        退化为按真实身份行棋，保证程序不会崩，同时不静默产生 ``rule_ident == 0``。
        """
        bit = SQ_BIT[sq]
        self.ident[sq] = ident
        self.side[sq] = side
        self.hidden[sq] = hidden
        if hidden:
            home = home_ident if home_ident is not None else HOME_IDENT[sq]
            self.rule_ident[sq] = home if home else ident
        else:
            self.rule_ident[sq] = ident
        self.occ |= bit
        self.occ_side[side] |= bit
        self.key ^= Z_HOME[side][sq] if hidden else Z_PIECE[side][ident][sq]
        if ident == K:
            self.kings[side] = sq

    def remove(self, sq: int) -> None:
        side = self.side[sq]
        if side < 0:
            return
        ident = self.ident[sq]
        bit = SQ_BIT[sq]
        self.key ^= Z_HOME[side][sq] if self.hidden[sq] else Z_PIECE[side][ident][sq]
        self.occ ^= bit
        self.occ_side[side] ^= bit
        self.ident[sq] = 0
        self.side[sq] = -1
        self.hidden[sq] = False
        self.rule_ident[sq] = 0
        if ident == K:
            self.kings[side] = -1

    def copy(self) -> "Board":
        b = Board()
        b.ident = self.ident[:]
        b.side = self.side[:]
        b.hidden = self.hidden[:]
        b.rule_ident = self.rule_ident[:]
        b.occ = self.occ
        b.occ_side = self.occ_side[:]
        b.kings = self.kings[:]
        b.turn = self.turn
        b.key = self.key
        b.ply = self.ply
        return b

    # --------------------------------------------------------- 哈希 / 标识
    def public_key(self) -> int:
        """只含公开信息的局面键（暗子按摆位身份参与哈希）。"""
        k = Z_SIDE if self.turn == BLACK else 0
        for sq in range(SQUARES):
            ri = self.rule_ident[sq]
            if ri:
                k ^= Z_PIECE[self.side[sq]][ri][sq]
        return k

    def public_signature(self) -> str:
        """公开信息的可读签名（重复局面检测用）。"""
        parts = []
        for sq in range(SQUARES):
            ri = self.rule_ident[sq]
            if ri:
                parts.append(f"{sq}{self.side[sq]}{ri}{'h' if self.hidden[sq] else 'r'}")
        return f"{self.turn}|" + ",".join(parts)

    # --------------------------------------------------------------- 查询
    def king_sq(self, side: int) -> int:
        return self.kings[side]

    def in_check(self, side: int) -> bool:
        ks = self.kings[side]
        if ks < 0:
            return True
        return self.is_attacked(ks, 1 - side)

    def is_attacked(self, sq: int, by: int) -> bool:
        """``sq`` 是否被 ``by`` 方攻击（含将帅同线照面）。"""
        occ = self.occ
        side = self.side
        rule = self.rule_ident

        # --- 直线：车 / 炮 / 将帅同线 ---
        for d in (UP, DOWN, LEFT, RIGHT):
            ray = RAY[sq][d]
            blk = ray & occ
            if not blk:
                continue
            nb = _nearest(blk, DIR_POSITIVE[d])
            bsq = bit_to_sq(nb)
            if side[bsq] == by:
                ri = rule[bsq]
                if ri == R:
                    return True
                if ri == K and (d == UP or d == DOWN):
                    return True
            blk2 = RAY[bsq][d] & occ
            if blk2:
                b2 = bit_to_sq(_nearest(blk2, DIR_POSITIVE[d]))
                if side[b2] == by and rule[b2] == C:
                    return True

        # --- 马 ---
        for frm, eye in LEAP_ATTACKERS[by][H][sq]:
            if side[frm] == by and rule[frm] == H:
                if eye is None or not (occ & SQ_BIT[eye]):
                    return True
        # --- 兵卒 ---
        for frm, _eye in LEAP_ATTACKERS[by][P][sq]:
            if side[frm] == by and rule[frm] == P:
                return True
        # --- 将帅贴身 ---
        for frm, _eye in LEAP_ATTACKERS[by][K][sq]:
            if side[frm] == by and rule[frm] == K:
                return True
        # --- 仕士：明士自由斜走，暗士限九宫 ---
        hidden = self.hidden
        for frm, _eye in LEAP_ATTACKERS[by][A][sq]:
            if side[frm] == by and rule[frm] == A and not hidden[frm]:
                return True
        for frm, _eye in LEAP_ATTACKERS_HIDDEN[by][A][sq]:
            if side[frm] == by and rule[frm] == A and hidden[frm]:
                return True
        # --- 相象：田字 + 塞象眼（明暗一致） ---
        for frm, eye in LEAP_ATTACKERS[by][E][sq]:
            if side[frm] == by and rule[frm] == E:
                if eye is None or not (occ & SQ_BIT[eye]):
                    return True
        return False

    def attack_map(self, side: int) -> int:
        """``side`` 方所有棋子攻击到的格子位掩码。

        性能注意（实测过，别凭直觉改）：这个方法占静态评估的 **45%**（两个颜色
        合计 23.6us），而成本的**大头是滑行子（车/炮）**——一枚车就有 12~17 条
        射线、每条 3~5 格，加起来近 70 次内层迭代；跳跃子（马相仕兵）平均只有
        2 格，全部加起来才 45 次迭代。

        所以这里做的是"**内层循环减负**"：把 ``SQ_BIT`` / ``RAYSQ`` / ``MOVES``
        提成局部变量（避免每次迭代都做一次模块属性查找）。

        试过但**更慢**的写法（0.91x），别重蹈：
        "只遍历有子的格"（``mask & -mask`` + 查表）——位运算加字典查表的开销
        超过了跳过空格省下的时间，26 枚子里省 65 次空循环换不回来。
        """
        m = 0
        occ = self.occ
        side_arr = self.side
        rule = self.rule_ident
        hidden = self.hidden
        bits = SQ_BIT               # 内层循环里要用几万次，必须局部化
        raylets = RAYSQ
        moves = MOVES
        moves_hidden = MOVES_HIDDEN
        for sq in range(SQUARES):
            if side_arr[sq] != side:
                continue
            ri = rule[sq]
            if ri == R:
                for ray in raylets[sq]:
                    for s in ray:
                        b = bits[s]
                        m |= b
                        if occ & b:
                            break
            elif ri == C:
                for ray in raylets[sq]:
                    screen = False
                    for s in ray:
                        b = bits[s]
                        if occ & b:
                            if screen:
                                if side_arr[s] != side:
                                    m |= b
                                break
                            screen = True
                        else:
                            m |= b
            elif ri == K:
                for to, _eye in moves[K][side][sq]:
                    m |= bits[to]
            elif ri:
                table = (moves_hidden[ri][side][sq] if hidden[sq]
                         else moves[ri][side][sq])
                for to, eye in table:
                    if eye is None or not (occ & bits[eye]):
                        m |= bits[to]
        return m

    # --------------------------------------------------------- 走法生成
    def gen_pseudo(self, side: int, captures_only: bool = False) -> List[int]:
        """生成走法（伪合法 / 仅吃子）。不检查是否自将。

        滑行子（车/炮）沿预计算的射线列表前进，碰到第一枚棋子即停，
        比逐位扫描位棋盘更快，代码也更直白。
        """
        moves: List[int] = []
        add = moves.append
        occ = self.occ
        occ_side = self.occ_side
        rule = self.rule_ident
        own = occ_side[side]
        opp = occ_side[1 - side]
        raylets = RAYSQ

        bb = own
        while bb:
            b = bb & -bb
            bb ^= b
            sq = bit_to_sq(b)
            ri = rule[sq]
            frm7 = sq << 7
            if ri == R:
                for ray in raylets[sq]:
                    for s in ray:
                        if occ & SQ_BIT[s]:
                            if opp & SQ_BIT[s]:
                                add(frm7 | s)
                            break
                        if not captures_only:
                            add(frm7 | s)
            elif ri == C:
                for ray in raylets[sq]:
                    screen = False
                    for s in ray:
                        if occ & SQ_BIT[s]:
                            if screen:
                                if opp & SQ_BIT[s]:
                                    add(frm7 | s)
                                break
                            screen = True
                        elif not captures_only and not screen:
                            add(frm7 | s)
            else:
                table = MOVES_HIDDEN[ri][side][sq] if self.hidden[sq] \
                    else MOVES[ri][side][sq]
                for to, eye in table:
                    if eye is not None and (occ & SQ_BIT[eye]):
                        continue
                    if opp & SQ_BIT[to]:
                        add(frm7 | to)
                    elif not captures_only and not (own & SQ_BIT[to]):
                        add(frm7 | to)
        return moves

    def gen_captures(self, side: int) -> List[int]:
        return self.gen_pseudo(side, captures_only=True)

    def _critical_squares(self, side: int) -> set:
        """从将帅出发的 4 条射线上，前两个己方子所在的格子。

        这些格子一旦被移开就**可能**暴露将帅（车/将对线、炮的炮架）。
        """
        ks = self.kings[side]
        if ks < 0:
            return set()
        occ = self.occ
        own = self.occ_side[side]
        out = set()
        for d in (UP, DOWN, LEFT, RIGHT):
            blk = RAY[ks][d] & occ
            if not blk:
                continue
            nb = _nearest(blk, DIR_POSITIVE[d])
            if own & nb:
                out.add(bit_to_sq(nb))
                rest = blk ^ nb
                if rest:
                    nb2 = _nearest(rest, DIR_POSITIVE[d])
                    if own & nb2:
                        out.add(bit_to_sq(nb2))
        return out

    def _risky_squares(self, side: int) -> set:
        """走子前后**必须逐个用 make/unmake 验证**的格子集合。

        只要着法的起点或终点落在这个集合里，就不能走捷径。三类：

        1. 将帅自己所在的格子，以及 ``_critical_squares``：**离开**将帅射线上的
           挡子（车/将对线、炮的炮架）。
        2. **走到**将帅与敌方炮之间、替对方"摆炮架"的空格。这是最关键的一类——
           实测局面：黑将 (8,4)、红炮 (6,4)、(7,4) 空着时并不将军；黑象从 (9,2)
           走到 (7,4) 之后正好成为唯一炮架，红炮立刻打死黑将。旧实现只看"棋子
           离开射线"，完全漏掉了"棋子走进射线"，于是 AI 把这种送将着法当成了
           合法着法。
        3. 己方棋子正好蹩住敌方马腿 / 塞住象眼，而这匹马/象的目标正是我方将帅：
           移开它就等于自己把马腿放开。
        """
        ks = self.kings[side]
        if ks < 0:
            return set()
        occ = self.occ
        enemy = 1 - side
        out = {ks}
        out |= self._critical_squares(side)

        # --- 2) 走到将帅与敌方炮之间当炮架 ---
        # 沿射线，如果"第一个挡子"就是敌方炮，那么将帅与它之间的空格全是风险位：
        # 把任一己方子走进去，就恰好凑成"一个炮架"，敌方炮立刻打将。
        for d in (UP, DOWN, LEFT, RIGHT):
            blk = RAY[ks][d] & occ
            if not blk:
                continue
            nb = _nearest(blk, DIR_POSITIVE[d])
            bsq = bit_to_sq(nb)
            if self.side[bsq] == enemy and self.rule_ident[bsq] == C:
                between = (RAY[ks][d] ^ RAY[bsq][d]) & ~SQ_BIT[bsq]
                bits = between
                while bits:
                    b = bits & -bits
                    bits ^= b
                    out.add(bit_to_sq(b))

        # --- 3) 蹩马腿 / 塞象眼 ---
        side_arr = self.side
        rule = self.rule_ident
        for ident in (H, E):
            for frm, eye in LEAP_ATTACKERS[enemy][ident][ks]:
                if eye is None:
                    continue
                if side_arr[frm] != enemy or rule[frm] != ident:
                    continue
                if side_arr[eye] == side:
                    # 马腿/象眼正好被我方棋子占着 -> 移开它就会放开这个攻击
                    out.add(eye)
        return out

    def legal_filter(self, side: int, moves: List[int]) -> List[int]:
        """从伪合法走法里筛掉自将 / 送将 / 对面笑的走法。"""
        ks = self.kings[side]
        if ks < 0 or not moves:
            return []
        if self.is_attacked(ks, 1 - side):
            # 被将军时必须每一步都验证
            legal = []
            for mv in moves:
                self.make(mv)
                ok = not self.is_attacked(self.kings[side], 1 - side)
                self.unmake()
                if ok:
                    legal.append(mv)
            return legal
        risky = self._risky_squares(side)
        legal = []
        for mv in moves:
            if (mv >> 7) in risky or (mv & 127) in risky:
                self.make(mv)
                ok = not self.is_attacked(self.kings[side], 1 - side)
                self.unmake()
                if not ok:
                    continue
            legal.append(mv)
        return legal

    def gen_moves(self, side: int) -> List[int]:
        """合法走法（已过滤自将 / 送将 / 对面笑）。"""
        if self.kings[side] < 0:
            return []
        return self.legal_filter(side, self.gen_pseudo(side))

    def is_legal(self, mv: int, side: Optional[int] = None) -> bool:
        side = self.turn if side is None else side
        ks = self.kings[side]
        if ks < 0:
            return False
        frm, to = mv >> 7, mv & 127
        if self.side[frm] != side:
            return False
        if self.ident[to] and self.side[to] == side:
            return False
        if mv not in self.gen_pseudo(side):
            return False
        self.make(mv)
        ok = not self.is_attacked(self.kings[side], 1 - side)
        self.unmake()
        return ok

    # --------------------------------------------------------- 落子 / 撤销
    def make(self, mv: int) -> None:
        frm = mv >> 7
        to = mv & 127
        ident = self.ident
        side = self.side
        hidden = self.hidden
        rule = self.rule_ident
        occ_side = self.occ_side

        ident_f = ident[frm]
        side_f = side[frm]
        hid_f = hidden[frm]
        ident_t = ident[to]
        side_t = side[to]
        hid_t = hidden[to]
        ri_t = rule[to]

        king_before = self.kings[side_f] if ident_f == K else -1
        cap_king_side = side_t if ident_t == K else -1

        self._undo.append((mv, ident_t, side_t, hid_t, ri_t, hid_f,
                           self.key, king_before, cap_king_side))
        self.ply += 1

        bit_f = SQ_BIT[frm]
        bit_t = SQ_BIT[to]

        # 清空起点
        ident[frm] = 0
        side[frm] = -1
        hidden[frm] = False
        rule[frm] = 0
        self.occ ^= bit_f
        occ_side[side_f] ^= bit_f
        self.key ^= (Z_HOME[side_f][frm] if hid_f
                     else Z_PIECE[side_f][ident_f][frm])

        # 处理被吃子
        if ident_t:
            self.occ ^= bit_t
            occ_side[side_t] ^= bit_t
            self.key ^= (Z_HOME[side_t][to] if hid_t
                         else Z_PIECE[side_t][ident_t][to])
            ident[to] = 0
            side[to] = -1
            hidden[to] = False
            rule[to] = 0
            if cap_king_side >= 0:
                self.kings[cap_king_side] = -1

        # 落子（暗子翻开）
        ident[to] = ident_f
        side[to] = side_f
        hidden[to] = False
        rule[to] = ident_f
        self.occ |= bit_t
        occ_side[side_f] |= bit_t
        self.key ^= Z_PIECE[side_f][ident_f][to]

        if ident_f == K:
            self.kings[side_f] = to

        self.turn = 1 - self.turn
        self.key ^= Z_SIDE

    def unmake(self) -> None:
        (mv, ident_t, side_t, hid_t, ri_t, hid_f,
         key, king_before, cap_king_side) = self._undo.pop()
        self.ply -= 1
        frm = mv >> 7
        to = mv & 127
        ident = self.ident
        side = self.side
        hidden = self.hidden
        rule = self.rule_ident
        occ_side = self.occ_side

        ident_f = ident[to]
        side_f = side[to]
        bit_f = SQ_BIT[frm]
        bit_t = SQ_BIT[to]

        ident[to] = ident_t
        side[to] = side_t
        hidden[to] = hid_t
        rule[to] = ri_t
        occ_side[side_f] ^= bit_t
        if ident_t:
            occ_side[side_t] |= bit_t
            if cap_king_side >= 0:
                self.kings[cap_king_side] = to
        else:
            self.occ ^= bit_t

        ident[frm] = ident_f
        side[frm] = side_f
        hidden[frm] = hid_f
        rule[frm] = HOME_IDENT[frm] if hid_f else ident_f
        self.occ |= bit_f
        occ_side[side_f] |= bit_f

        if king_before >= 0:
            self.kings[side_f] = king_before

        self.key = key
        self.turn = 1 - self.turn

    def make_null(self) -> None:
        self._undo.append(("null", self.key, -1, -1, -1, False, 0, -1, -1))
        self.turn = 1 - self.turn
        self.key ^= Z_SIDE
        self.ply += 1

    def unmake_null(self) -> None:
        _, key = self._undo.pop()[:2]
        self.turn = 1 - self.turn
        self.key = key
        self.ply -= 1

    # ------------------------------------------------------------------ 输出
    def piece_char(self, sq: int) -> str:
        """棋子的中文描述，例如 ``红车`` / ``暗炮`` / ``黑将``。"""
        ri = self.rule_ident[sq]
        if not ri:
            return ""
        s = self.side[sq]
        ch = IDENT_CHARS[(s, ri)]
        if self.hidden[sq]:
            return "暗" + ch
        return ("红" if s == RED else "黑") + ch

    def to_text(self, flip: bool = False) -> str:
        """中文棋盘文本（日志 / 大模型提示词用）。``?`` 表示暗子。"""
        lines = ["    " + "  ".join(str(c) for c in range(COLS))]
        rows = range(ROWS - 1, -1, -1) if flip else range(ROWS)
        for r in rows:
            cells = []
            for c in range(COLS):
                sq = r * 9 + c
                ri = self.rule_ident[sq]
                if not ri:
                    cells.append("  ·")
                    continue
                s = self.side[sq]
                if self.hidden[sq]:
                    cells.append(IDENT_CHARS[(s, HOME_IDENT[sq])] + "?")
                else:
                    cells.append(IDENT_CHARS[(s, ri)] + " ")
            lines.append(f"{r:2d}  " + "".join(cells))
        lines.append("    " + "  ".join(str(c) for c in range(COLS)))
        lines.append(f"轮到：{'红方' if self.turn == RED else '黑方'}")
        return "\n".join(lines)

    def revealed_material(self, side: int) -> int:
        total = 0
        for sq in range(SQUARES):
            if self.side[sq] == side and not self.hidden[sq]:
                total += PIECE_VALUE[self.ident[sq]]
        return total

    def clone_revealed_only(self) -> "Board":
        """把暗子的真实身份抹掉（用于测试“AI 没有偷看暗子”）。"""
        b = self.copy()
        for sq in range(SQUARES):
            if b.hidden[sq]:
                b.ident[sq] = HOME_IDENT[sq]
        return b

    def __str__(self) -> str:  # pragma: no cover
        return self.to_text()
