"""确定性搜索：迭代加深 Negamax + Alpha-Beta/PVS。

* 置换表（Zobrist 键）：每个“定局化”局面一份，避免跨采样误用边界。
* 走法排序：置换表着法 > 吃子（MVV-LVA）> 杀手着法 > 历史启发。
* 静态搜索（quiescence）：只搜吃子，消除明显的水平线效应。
* 空着裁剪（null move）：非将军、深度足够时启用。
* 重复局面：搜索路径内出现两次同一局面即视为和棋（0 分）。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from .board import Board, move_str
from .consts import (
    BLACK,
    KING_NEIGH,
    MATE_SCORE,
    MATE_BOUND,
    PIECE_VALUE,
    RED,
    SQ_BIT,
)
from .evaluate import EvalContext, evaluate

INF = 10 ** 9
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
MAX_PLY = 128


class TimeUp(Exception):
    """搜索超出时间预算。"""


class Searcher:
    """针对单个“定局化”局面的搜索器。"""

    __slots__ = ("board", "ctx", "tt", "killers", "history", "nodes",
                 "deadline", "check_mask", "path_counts", "seldepth",
                 "null_move", "quiescence", "extensions",
                 "reveal_weights", "reveal_quiet_weights", "_reveal_feat",
                 "_guard_deadline", "_guard_nodes", "_guard_undo_floor")

    def __init__(self, board: Board, ctx: EvalContext,
                 options: Optional[dict] = None) -> None:
        opts = options or {}
        self.board = board
        self.ctx = ctx
        self.tt: Dict[int, Tuple[int, int, int, int]] = {}
        self.killers: List[List[int]] = [[-1, -1] for _ in range(MAX_PLY)]
        self.history: Dict[int, int] = {}
        self.nodes = 0
        self.seldepth = 0
        self.deadline: Optional[float] = None
        self.check_mask = opts.get("check_mask", 1023)
        self.null_move = opts.get("null_move", True)
        self.quiescence = opts.get("quiescence", True)
        #: 将军延伸（默认为开；关掉只用于自对弈 A/B 对照）
        self.extensions = opts.get("extensions", True)
        self.path_counts: Dict[int, int] = {board.key: 1}
        #: 揭棋特有的"搜索行为调整"（见 jieqi/reveal.py）。
        #: ``None`` = 关闭（默认，与改动前逐位一致）；传入 dict 才启用。
        self.reveal_weights: Optional[dict] = opts.get("reveal_weights")
        self.reveal_quiet_weights: Optional[dict] = opts.get("reveal_quiet_weights")
        self._reveal_feat: Optional[dict] = None
        #: "送杀体检"自己的预算（与整局搜索的 deadline 完全分开）
        self._guard_deadline: Optional[float] = None
        self._guard_nodes = 0
        self._guard_undo_floor = 0

    def prepare_reveal(self, eval_red: float = 0.0) -> Optional[dict]:
        """按**根局面**预计算揭棋特征（整棵树共用一个，省得每个节点重算）。

        只算一次是刻意的：这几个特征描述的是"这一手决策处在什么阶段"，
        在浅搜索里几乎不变，逐节点重算只会拖慢速度。
        返回 ``None`` 表示功能关闭。
        """
        if not self.reveal_weights and not self.reveal_quiet_weights:
            self._reveal_feat = None
            return None
        from .reveal import features

        b = self.board
        self._reveal_feat = features(b, b.turn, eval_red)
        return self._reveal_feat

    def _reveal_bonus(self, move: int) -> int:
        """这一步若是翻子，返回它的加成（厘兵）；否则 0。"""
        feat = self._reveal_feat
        if feat is None or not self.reveal_weights:
            return 0
        from .reveal import is_dark_move, quiet_bonus, quiet_safety, reveal_bonus

        b = self.board
        if not is_dark_move(b, move):
            return 0
        total = reveal_bonus(feat, self.reveal_weights)
        if self.reveal_quiet_weights and not b.ident[move & 127]:
            safety = quiet_safety(b, move)
            total += quiet_bonus(b, b.side[move >> 7], feat, safety,
                                 self.reveal_quiet_weights)
        return total

    # ------------------------------------------------------------ 工具
    def set_board(self, board: Board) -> None:
        self.board = board
        self.path_counts = {board.key: 1}
        self.killers = [[-1, -1] for _ in range(MAX_PLY)]

    def _tick(self) -> None:
        self.nodes += 1
        if not (self.nodes & self.check_mask):
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise TimeUp

    def _victim_value(self, sq: int) -> float:
        b = self.board
        if not b.ident[sq]:
            return 0.0
        if b.hidden[sq]:
            return self.ctx.mean_value.get(b.side[sq], 300.0)
        return PIECE_VALUE[b.ident[sq]]

    def _order(self, moves: List[int], tt_move: int, ply: int) -> List[int]:
        b = self.board
        ident = b.ident
        hidden = b.hidden
        movers = self.ctx.mean_value
        killers = self.killers[ply] if ply < MAX_PLY else (-1, -1)
        k1, k2 = killers
        hist = self.history
        scored = []
        for mv in moves:
            if mv == tt_move:
                scored.append((1 << 40, mv))
                continue
            to = mv & 127
            if ident[to]:
                frm = mv >> 7
                vic = (movers.get(b.side[to], 300.0) if hidden[to]
                       else PIECE_VALUE[ident[to]])
                att = (movers.get(b.side[frm], 300.0) if hidden[frm]
                       else PIECE_VALUE[ident[frm]])
                scored.append(((1 << 30) + int(vic) * 16 - int(att), mv))
            elif mv == k1:
                scored.append((1 << 25, mv))
            elif mv == k2:
                scored.append((1 << 24, mv))
            else:
                scored.append((hist.get(mv, 0), mv))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ------------------------------------------------------- 静态搜索
    def _qsearch(self, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        b = self.board
        side = b.turn
        if b.kings[side] < 0:
            return -MATE_SCORE + ply
        if ply > self.seldepth:
            self.seldepth = ply

        stand = evaluate(b, self.ctx)
        if side == BLACK:
            stand = -stand
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        if not self.quiescence:
            return alpha

        caps = b.legal_filter(side, b.gen_captures(side))
        if not caps:
            return alpha
        caps = self._order(caps, -1, ply)
        for mv in caps:
            # delta 裁剪：吃回来的子还不够填坑就跳过
            gain = self._victim_value(mv & 127)
            if stand + gain * 4 + 200 < alpha:
                continue
            b.make(mv)
            score = -self._qsearch(-beta, -alpha, ply + 1)
            b.unmake()
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return alpha

    # ------------------------------------------------------- 主搜索
    def _negamax(self, depth: int, alpha: int, beta: int, ply: int,
                 can_null: bool = True, ext: int = 0) -> int:
        """Negamax + Alpha-Beta(PVS)。

        ``ext`` 是"本条线已经用过将军延伸"的标记：**一条线最多延伸一次**。
        没有这个限制的话，双方连续将军会让搜索深度无上限地长下去（时间炸掉）。
        """
        self._tick()
        b = self.board
        side = b.turn
        ks = b.kings[side]
        if ks < 0:
            return -MATE_SCORE + ply
        if ply >= MAX_PLY - 2:
            return evaluate(b, self.ctx) * (1 if side == RED else -1)

        key = b.key
        if self.path_counts.get(key, 0) >= 2:
            return 0

        if depth <= 0:
            return self._qsearch(alpha, beta, ply)

        tt = self.tt
        entry = tt.get(key)
        tt_move = -1
        if entry is not None:
            e_depth, e_score, e_flag, e_move = entry
            tt_move = e_move
            if e_depth >= depth:
                if e_flag == TT_EXACT:
                    return e_score
                if e_flag == TT_LOWER:
                    if e_score > alpha:
                        alpha = e_score
                elif e_flag == TT_UPPER:
                    if e_score < beta:
                        beta = e_score
                if alpha >= beta:
                    return e_score

        in_check = b.is_attacked(ks, 1 - side)

        # --- 空着裁剪 ---
        if (self.null_move and can_null and not in_check and depth >= 3
                and b.occ_side[side] & (b.occ_side[side] - 1)):
            b.make_null()
            nk = b.key
            self.path_counts[nk] = self.path_counts.get(nk, 0) + 1
            v: Optional[int] = None
            if not b.is_attacked(b.kings[1 - side], side):
                v = -self._negamax(depth - 3, -beta, -beta + 1, ply + 1, False, ext)
            cnt = self.path_counts[nk] - 1
            if cnt <= 0:
                del self.path_counts[nk]
            else:
                self.path_counts[nk] = cnt
            b.unmake_null()
            if v is not None and v >= beta:
                self.nodes += 1
                return v

        moves = b.gen_moves(side)
        if not moves:
            # 无棋可走（含被将死 / 困毙）：象棋规则判负
            return -MATE_SCORE + ply

        moves = self._order(moves, tt_move, ply)
        orig_alpha = alpha
        best = -INF
        best_move = moves[0]
        can_extend = (self.extensions and ext == 0 and not in_check)
        opp_king = b.kings[1 - side]

        for i, mv in enumerate(moves):
            # 揭棋：翻子着法可以拿一份"加成"（见 jieqi/reveal.py）。
            # 关掉时（默认）这里恒为 0，搜索行为与改动前逐位一致。
            bonus = self._reveal_bonus(mv)
            b.make(mv)
            nk = b.key
            self.path_counts[nk] = self.path_counts.get(nk, 0) + 1
            # 将军延伸：落点贴着对方九宫时才判"这一步是否将军"，
            # 免得给每个着法都加一次 is_attacked 的开销。
            #
            # ``ext`` 是"本线已经用过延伸"的标记：**一条线最多延伸一次**。
            # （试过放开成"每层各延伸一次"，结果连"一步杀"都找不到了——
            #   双方连续将军时深度会被反复回补，原本能搜到的杀线反而被挤出
            #   置换表/时限之外。这条限制不要动，强制线漏杀改用根节点的
            #   "送杀体检"来兜，见 :meth:`forced_loss_moves`。）
            next_ext = 0
            child_depth = depth - 1
            if can_extend and opp_king >= 0 and \
                    (KING_NEIGH[opp_king] & SQ_BIT[mv & 127]):
                if b.is_attacked(b.kings[1 - side], side):
                    child_depth = depth      # 强制线（将军）多算一层
                    next_ext = 1
            if i == 0:
                v = -self._negamax(child_depth, -beta, -alpha, ply + 1, True,
                                   next_ext)
            else:
                v = -self._negamax(child_depth, -alpha - 1, -alpha, ply + 1, True,
                                   next_ext)
                if alpha < v < beta:
                    v = -self._negamax(child_depth, -beta, -alpha, ply + 1, True,
                                       next_ext)
            cnt = self.path_counts[nk] - 1
            if cnt <= 0:
                del self.path_counts[nk]
            else:
                self.path_counts[nk] = cnt
            b.unmake()

            # 加成加在"这一步的分数"上。⚠ 必须在 ``unmake`` 之后算——落点
            # 出了棋盘范围会让 ``b.ident[mv & 127]`` 抛 IndexError。
            # 杀棋分（``MATE_BOUND`` 以上）不加成，免得把"确证杀棋"变成
            # "看着像杀"（AB-JChess 的 ``apply_bonus`` 同样跳过 decisive 分）。
            if bonus and -MATE_BOUND < v < MATE_BOUND:
                v -= bonus

            if v > best:
                best = v
                best_move = mv
            if v > alpha:
                alpha = v
            if alpha >= beta:
                if not b.ident[mv & 127]:
                    kl = self.killers[ply] if ply < MAX_PLY else None
                    if kl is not None and kl[0] != mv:
                        kl[1] = kl[0]
                        kl[0] = mv
                    self.history[mv] = self.history.get(mv, 0) + depth * depth
                break

        if best >= beta:
            flag = TT_LOWER
        elif best > orig_alpha:
            flag = TT_EXACT
        else:
            flag = TT_UPPER
        tt[key] = (depth, best, flag, best_move)
        if len(tt) > 1_500_000:
            tt.clear()
        return best

    # ------------------------------------------------------- 根节点
    def search_root(self, depth: int, root_moves: Optional[List[int]] = None,
                    alpha: int = -INF + 1, beta: int = INF - 1,
                    order: Optional[List[int]] = None,
                    forbid: Optional[Dict[int, int]] = None
                    ) -> Tuple[int, int, List[int], Dict[int, int]]:
        """搜索根节点。

        返回 ``(best_move, best_score, pv, {move: score})``。
        ``forbid`` 里的着法会被强制记为和棋分（0）。
        """
        b = self.board
        side = b.turn
        if root_moves is None:
            root_moves = b.gen_moves(side)
        if not root_moves:
            return -1, -MATE_SCORE, [], {}
        if order:
            rank = {mv: i for i, mv in enumerate(order)}
            root_moves = sorted(root_moves, key=lambda m: rank.get(m, 1 << 30))

        scores: Dict[int, int] = {}
        best_move = root_moves[0]
        best_score = -INF
        pv: List[int] = []
        a = alpha
        for i, mv in enumerate(root_moves):
            b.make(mv)
            self.path_counts[b.key] = self.path_counts.get(b.key, 0) + 1
            if i == 0:
                v = -self._negamax(depth - 1, -beta, -a, 1, True)
            else:
                v = -self._negamax(depth - 1, -a - 1, -a, 1, True)
                if a < v < beta:
                    v = -self._negamax(depth - 1, -beta, -a, 1, True)
            child_pv: List[int] = []
            if depth > 1:
                child_pv = self._extract_pv(depth - 1)
            cnt = self.path_counts[b.key] - 1
            if cnt <= 0:
                del self.path_counts[b.key]
            else:
                self.path_counts[b.key] = cnt
            b.unmake()

            if forbid and mv in forbid:
                v = 0
            scores[mv] = v
            if v > best_score:
                best_score = v
                best_move = mv
                pv = [mv] + child_pv
            if v > a:
                a = v
        return best_move, best_score, pv, scores

    # ------------------------------------------- 根节点"送杀"体检
    #: 这个体检最多花多少节点 / 秒（它只是"兜底"，不能把思考时间吃掉）。
    #: 实测：一步杀的局面上要几万节点才能把"对手每个应手都逃不掉"证完，
    #: 所以给到 6 万；时间上限压得很紧，正常局面下一瞬间就用完并放弃。
    guard_node_cap = 60000
    guard_time_cap = 0.35

    def guard_deadline(self, seconds: float = None) -> None:
        """给体检设一个**独立**的时间上限。

        ⚠ 千万不要动 ``self.deadline``——那是整局搜索的预算，覆盖它会让体检
        跑完之后搜索只剩零点几秒（早期版本就这么写，结果"一步杀"都找不到了）。
        """
        cap = self.guard_time_cap if seconds is None else float(seconds)
        self._guard_deadline = time.monotonic() + cap
        self._guard_nodes = 0
        self._guard_undo_floor = len(self.board._undo)

    def _guard_rewind(self) -> None:
        """把棋盘退回到体检开始时的状态。

        ⚠ **必须做这一步**：体检超时是通过抛 :class:`TimeUp` 实现的，异常会把
        每一层递归中途 "挂在" 棋盘上的 ``make`` 全部跳过 ``unmake``。如果不回滚，
        棋盘上就留下一堆"走了一半"的格子（``rule_ident == 0`` 却有子），
        紧接着 ``gen_pseudo`` 就会 ``KeyError: 0``——真实踩过这个坑。
        """
        floor = int(getattr(self, "_guard_undo_floor", 0))
        b = self.board
        while len(b._undo) > floor:
            b.unmake()
        self.path_counts = {b.key: 1}

    def _guard_tick(self) -> None:
        """体检自己的 tick：用体检的计数器与截止时间。"""
        self._guard_nodes += 1
        if self._guard_nodes >= self.guard_node_cap:
            raise TimeUp
        if not (self._guard_nodes & self.check_mask):
            limit = getattr(self, "_guard_deadline", None)
            if limit is not None and time.monotonic() > limit:
                raise TimeUp

    def forced_loss_moves(self, moves: Sequence[int], plies: int = 4) -> set:
        """哪些根着法走完之后，对方在 ``plies`` 层内**强制吃将**？

        这是一个**独立于评估函数**的小型证明搜索（只看"能不能吃到将"，
        不做任何位置打分），专门用来堵住"评估说大优、其实已经被叫杀"这类
        崩盘：

        * ``plies=4`` 覆盖"对方将军 → 我唯一应手 → 对方吃将"这类两步杀；
        * 代价很小：只对候选着法做"对方所有应手 + 我方所有着法"的枚举，
          而且一发现就短路；同时受 :attr:`guard_node_cap` 与
          :attr:`guard_time_cap` 限制，超了就当"没结论"，绝不拖慢正常搜索；
        * 只有在"被挑中的着法会被强制杀、且还有别的着法不会被强制杀"时才改判，
          所以正常局面里它完全不介入选着。

        返回会被强制杀的着法集合（可能不完整——超预算时提前返回）。
        """
        if plies <= 1:
            return set()
        b = self.board
        side = b.turn
        opp = 1 - side
        out = set()
        for mv in moves:
            try:
                b.make(mv)
                # 走完之后轮到对手：对手只要**有一步能吃我们的将**就说明这手送杀；
                # 更远的强制杀交给 _fc_and（对手每个应手之后我们都还能被吃）。
                lost = (b.kings[side] < 0
                        or self._fc_or(b, opp, plies))
                b.unmake()
            except TimeUp:
                self._guard_rewind()
                break
            except KeyError:
                # 采样出来的暗子身份为 0（对局里出现过）：这不是正常局面，
                # 体检放弃即可，绝不能把异常抛给整局搜索。
                self._guard_rewind()
                break
            if lost:
                out.add(mv)
        return out

    @property
    def guard_nodes(self) -> int:
        return int(getattr(self, "_guard_nodes", 0))

    def _forced_capture(self, b: Board, hunter: int, plies: int) -> bool:
        """``hunter`` 是否能在 ``plies`` 层内**强制吃掉**对方的将。

        递归写法统一成"轮到谁"的形式，避免把"我方/对方"混淆：

        * :meth:`_fc_or`：轮到 ``me`` 走，我只要**有一个**着法能强制吃将就成立；
        * :meth:`_fc_and`：轮到对手走，对手**每个**应手之后我都还得能强制吃将。

        **将死在本游戏里就是"吃将"**，所以：

        * 吃到对方的将（``kings[opp] < 0``）＝成功；
        * 轮到对手却**无棋可走**（含被将死/困毙）＝成功。

        ⚠ 早期版本在这里把"谁吃谁"搞反了：它检查的是"我方将还在不在"，
        于是"红方走完一步杀、轮到黑方无棋可走"被误判成"黑方能强制吃红将"，
        送杀体检就会**把最好的一步杀当成送杀给否掉**（真实踩过，单一测试
        能复现：红车 (1,0)->(1,4) 将死黑方，却被改成了别的着法）。
        """
        if plies <= 0:
            return False
        if b.kings[1 - hunter] < 0:
            return True
        return self._fc_or(b, hunter, plies)

    def _fc_or(self, b: Board, me: int, plies: int) -> bool:
        """轮到 ``me`` 走：存在一个着法能强制吃将。"""
        self._guard_tick()
        opp = 1 - me
        if b.kings[opp] < 0:
            return True
        if plies <= 0:
            return False
        for mv in b.gen_moves(me):
            b.make(mv)
            won = (b.kings[opp] < 0) or self._fc_and(b, opp, plies - 1)
            b.unmake()
            if won:
                return True
        return False

    def _fc_and(self, b: Board, me: int, plies: int) -> bool:
        """轮到对手走：对手的**每一个**应手之后 ``me`` 仍能强制吃将。

        ``me`` 是"进攻方"（吃掉对方将的那一方），不是"当前走子方"。
        """
        self._guard_tick()
        opp = 1 - me
        if b.kings[opp] < 0:
            return True
        replies = b.gen_moves(opp)
        if not replies:
            return True                      # 对手无棋可走（将死/困毙）＝我们赢
        for mv in replies:
            b.make(mv)
            if b.kings[me] < 0:              # 进攻方的将被吃：这条线不成立
                b.unmake()
                return False
            ok = self._fc_or(b, me, plies - 1)
            b.unmake()
            if not ok:
                return False
        return True

    def _extract_pv(self, depth: int) -> List[int]:
        """从**当前**局面顺着置换表往下取主变，取完把棋盘恢复原位。"""
        b = self.board
        out: List[int] = []
        seen = set()
        for _ in range(depth):
            entry = self.tt.get(b.key)
            if not entry or entry[3] < 0:
                break
            mv = entry[3]
            if mv in seen:
                break
            seen.add(mv)
            out.append(mv)
            b.make(mv)
        for _ in out:
            b.unmake()
        return out
