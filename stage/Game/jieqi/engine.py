"""揭棋搜索引擎门面：不完全信息下的“定局化 + 期望聚合”搜索。

算法
----
1. 从 GUI 局面构造 belief（暗子身份的后验分布）——引擎**从不**读取暗子的真实身份。
2. 用 ``EvalContext`` 把暗子折成后验期望，得到一个精确的期望评估函数。
3. 从 belief 采样出 K 个“定局化”局面（每个暗子格按后验概率填上身份）。
4. 每个定局化局面跑一次迭代加深 Alpha-Beta，记录**每一步根着法**的分数。
5. 按采样求平均（可选方差惩罚），选出期望最好的着法。

这样：
* 走法生成 / 将军判定只用公开信息，不存在“偷看暗子”；
* “翻开后可能是什么”由采样覆盖；
* 吃暗子的收益按期望计算，不会被某一次幸运采样带偏。

时间控制：总预算 ``time_limit`` 秒，平均分给各个采样；单步默认 2 秒。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .belief import Belief
from .board import Board, move_str
from .consts import (
    BLACK,
    MATE_BOUND,
    MATE_SCORE,
    RED,
)
from .evaluate import EvalContext, evaluate
from .search import Searcher, TimeUp, INF

# --------------------------------------------------------------------------
# 难度预设
#
# ``samples`` 是暗子身份的采样个数：采样越多，"抽奖式"判断越少，但每个采样分到
# 的时间也越少（深度更浅）。实测在 2 秒预算内 1 个和 4 个采样的对局胜负基本持平，
# 所以默认取中间值：优先保证一点深度，同时保留基本的采样平均。
# --------------------------------------------------------------------------
DIFFICULTIES: Dict[str, dict] = {
    "easy": dict(time_limit=0.5, samples=1, max_depth=2, quiescence=False,
                 null_move=False, temperature=90.0, variance_penalty=0.0),
    "normal": dict(time_limit=2.0, samples=2, max_depth=64, quiescence=True,
                   null_move=True, temperature=0.0, variance_penalty=0.0),
    "hard": dict(time_limit=5.0, samples=4, max_depth=64, quiescence=True,
                 null_move=True, temperature=0.0, variance_penalty=0.15),
    "expert": dict(time_limit=10.0, samples=6, max_depth=64, quiescence=True,
                   null_move=True, temperature=0.0, variance_penalty=0.25),
}

VERDICT_TABLE = (
    (600, "红方胜势"), (250, "红方大优"), (90, "红方占优"), (30, "红方稍优"),
    (-30, "均势"), (-90, "黑方稍优"), (-250, "黑方占优"), (-600, "黑方大优"),
)


def verdict_of(score: float) -> str:
    if score >= MATE_BOUND:
        return "红方必胜"
    if score <= -MATE_BOUND:
        return "黑方必胜"
    for threshold, text in VERDICT_TABLE:
        if score >= threshold:
            return text
    return "黑方胜势"


@dataclass
class MoveStat:
    """根着法的跨采样统计。"""

    move: int
    mean: float = 0.0
    worst: float = 0.0
    best_count: int = 0
    samples: int = 0
    pv: List[int] = field(default_factory=list)
    select_score: float = 0.0

    def describe(self, side: int = RED) -> str:
        return (f"{move_str(self.move)} 期望{self.mean:+.0f} 最差{self.worst:+.0f} "
                f"最佳{self.best_count}/{self.samples} 选分{self.select_score:+.0f}")


@dataclass
class SearchResult:
    move: Optional[int]
    score: float                 # 红方视角期望分（厘兵）
    depth: int
    nodes: int
    samples: int
    elapsed: float
    stats: List[MoveStat] = field(default_factory=list)
    pv: List[int] = field(default_factory=list)
    verdict: str = ""
    belief_text: str = ""
    notes: List[str] = field(default_factory=list)
    tree: dict = field(default_factory=dict)
    sample_depths: List[int] = field(default_factory=list)
    #: 走子方（``Board.turn``：0 = 红、1 = 黑）。用于把红方视角的 ``score``
    #: 换算回"走子方视角"（见 :attr:`mover_score`）。默认红方。
    side: int = 0

    @property
    def mover_score(self) -> float:
        """**走子方视角**的分数（与 ``stats[].mean`` 同一量纲与符号）。

        ``score`` / ``verdict`` 是**红方视角**（界面与棋谱都用它）；
        而 ``MoveStat.mean`` 是**走子方视角**（引擎内部算出来的原始分）。
        要让两者对齐就得知道走子方是谁，所以做在这个类里：
        以前这个换算散在好几处，改一处漏一处（漏掉的那处符号会翻，
        表现是"AI 说黑方大优，其实红方大优"）。

        ``side`` 是 ``Board.turn``：``0`` = 红、``1`` = 黑（见 ``consts``）。
        """
        mover = 1 if int(self.side) == 0 else -1
        return float(self.score) * mover

    @property
    def is_mate(self) -> bool:
        return abs(self.score) >= MATE_BOUND

    def mate_in(self) -> Optional[int]:
        if not self.is_mate:
            return None
        plies = MATE_SCORE - abs(self.score)
        return (plies + 1) // 2

    def summary(self) -> str:
        if self.move is None:
            return "无合法着法"
        bits = [f"最佳：{move_str(self.move)}",
                f"评估：{self.score:+.0f}（{self.verdict}）",
                f"深度：{self.depth}",
                f"节点：{self.nodes}",
                f"采样：{self.samples}",
                f"耗时：{self.elapsed:.2f}s"]
        m = self.mate_in()
        if m:
            bits.insert(1, f"{m} 步杀")
        return " | ".join(bits)


class Engine:
    """揭棋 AI 引擎。"""

    def __init__(self, difficulty: str = "normal",
                 time_limit: Optional[float] = None,
                 samples: Optional[int] = None,
                 max_depth: Optional[int] = None,
                 seed: Optional[int] = None,
                 weights: Optional[Dict[str, float]] = None,
                 **overrides) -> None:
        preset = dict(DIFFICULTIES.get(str(difficulty).lower(),
                                       DIFFICULTIES["normal"]))
        preset.update({k: v for k, v in overrides.items() if v is not None})
        if time_limit is not None:
            preset["time_limit"] = float(time_limit)
        if samples is not None:
            preset["samples"] = max(1, int(samples))
        if max_depth is not None:
            preset["max_depth"] = max(1, int(max_depth))
        self.options = preset
        self.difficulty = str(difficulty).lower()
        self.weights = weights
        self.seed = seed
        self.nodes_last = 0

    # ------------------------------------------------------------------
    def score_move(self, board: Board, move: int,
                   depth: int = 1,
                   belief: Optional[Belief] = None,
                   own_captured: Optional[Dict[int, List[int]]] = None,
                   captured_hidden: Optional[Dict[int, int]] = None,
                   use_hidden_captures: bool = False,
                   ) -> Optional[float]:
        """给**指定的那一手**打一个分（红方视角，厘兵）；不合法返回 ``None``。

        用途：大模型给出"合法但不在本机引擎短名单里"的着法时，我们仍然采用
        它（用户要求"完全采用那个引擎的招法"），但得给它一个可显示的评估分。
        有了这个函数就能回答"这一步值多少"，而不是硬塞一个 0。

        实现方式刻意**复用现成的根搜索**：把"除目标外的全部根着法"都放进
        ``forbid``（根节点会把被禁着法记为和棋分并排到最后），于是
        ``scores[move]`` 就是"只走这一手、其余照常搜索"的分数。

        代价 ≈ 一次浅层搜索（默认 ``depth=1``，也就是"走完这手后看一步"），
        比整局主搜索便宜得多；分数与主搜索**同一量纲**（都是厘兵），
        可以放进面板的评估里显示。
        """
        pub = board.clone_revealed_only()
        side = int(pub.turn)
        root_moves = pub.gen_moves(side)
        if int(move) not in root_moves:
            return None
        if belief is None:
            try:
                belief = Belief.from_board(
                    pub, captured=own_captured, captured_hidden=captured_hidden,
                    use_hidden_captures=use_hidden_captures)
            except Exception:
                return None
        ctx = EvalContext(belief, self.weights, use_cache=True)
        opts = dict(
            quiescence=self.options.get("quiescence", True),
            null_move=self.options.get("null_move", True),
            extensions=self.options.get("extensions", True),
            reveal_weights=self.options.get("reveal_weights"),
            reveal_quiet_weights=self.options.get("reveal_quiet_weights"),
        )
        forbid = {mv: 0 for mv in root_moves if mv != int(move)}
        try:
            b = pub.copy()
            if belief is not None:
                belief.apply_sample(b, random.Random(self.seed or 0))
            searcher = Searcher(b, ctx, opts)
            searcher.deadline = time.monotonic() + max(
                0.05, float(self.options.get("time_limit", 1.0)) * 0.25)
            _bm, _bs, _pv, scores = searcher.search_root(
                max(1, int(depth)), [int(move)], forbid=forbid)
            value = scores.get(int(move))
            if value is None:
                return None
            return float(value if side == RED else -value)
        except Exception:
            # 打分失败**不是错误**（只是少一个显示用的数字），返回 None 让调用方
            # 用 0 或者不显示分数。
            return None

    # ------------------------------------------------------------------
    def analyse(self, board: Board, belief: Optional[Belief] = None,
                history_counts: Optional[Dict[str, int]] = None,
                own_captured: Optional[Dict[int, List[int]]] = None,
                captured_hidden: Optional[Dict[int, int]] = None,
                use_hidden_captures: bool = False,
                rule_facts: Optional[Sequence] = None,
                verbose: bool = False) -> SearchResult:
        """分析局面并给出最佳着法。

        ``board`` 可以带着暗子的真实身份，但引擎会立刻把它抹掉
        （``clone_revealed_only``），所以引擎在结构上不可能偷看暗子。
        ``history_counts`` 是公开局面签名 -> 出现次数，用于避开三次重复判和。
        ``captured_hidden`` 是被吃**暗子**的数量（身份未知）。

        ``rule_facts`` 是**本局到当前为止的 ``MoveFacts`` 序列**（见
        ``jieqi.rules``）。给了它，引擎就会在根节点把"会让走子方触犯长将 /
        长捉禁招"的着法从候选里剔掉（``forbid`` 机制）；**不给时行为与改动前
        逐位一致**（不会多算任何东西，也不会改变随机数与选着顺序）。
        """
        t0 = time.monotonic()
        opts = self.options
        notes: List[str] = []

        pub = board.clone_revealed_only()
        side = pub.turn
        if belief is None:
            belief = Belief.from_board(
                pub, captured=own_captured, captured_hidden=captured_hidden,
                use_hidden_captures=use_hidden_captures)

        root_moves = pub.gen_moves(side)
        if not root_moves:
            return SearchResult(
                move=None, score=-MATE_SCORE if side == RED else MATE_SCORE,
                depth=0, nodes=0, samples=0, elapsed=time.monotonic() - t0,
                belief_text=belief.describe(), notes=["无合法着法（被将死或困毙）"],
                verdict="黑方必胜" if side == RED else "红方必胜")

        ctx = EvalContext(belief, self.weights,
                          use_cache=bool(opts.get("eval_cache", True)))
        rng = random.Random(self.seed if self.seed is not None
                            else int(time.time() * 1000) ^ id(board))
        n_samples = max(1, int(opts["samples"]))
        deadline = t0 + float(opts["time_limit"])
        # 时间太少时自动少采样：``time_limit/samples`` 低于 0.6 秒的话，每个采样
        # 只有零点几秒，连 3 层都搜不完（真实对局日志里出现过"2 秒 / 2 采样只到
        # 3 层"）。这种时候"搜得深一点"比"多抽一次签"更有价值，所以把采样压到
        # 每个至少 0.6 秒。
        budget = float(opts["time_limit"])
        if n_samples > 1:
            fit = max(1, int(budget / 0.6))
            if fit < n_samples:
                notes.append("时间较紧：采样从 %d 降到 %d（保证每采样至少 0.6s）"
                             % (n_samples, fit))
                n_samples = fit

        # 采样：从抹掉真实身份的局面出发，按后验分布填身份
        boards: List[Board] = []
        for _ in range(n_samples):
            b = pub.copy()
            belief.apply_sample(b, rng)
            boards.append(b)

        forbid = self._repetition_moves(pub, history_counts)
        if forbid:
            notes.append(f"{len(forbid)} 步会导致三次重复判和")

        # ---- 比赛规则：长将 / 长捉禁招 ----
        # 规则是"连续 N 回合后禁着"，只看**当前局面**看不出违规，必须结合本局的
        # 着法历史（``rule_facts``）才能判。命中就把这些着法并进 ``forbid``，
        # 让根节点选着时一并避开。
        rule_forbidden = self._rule_forbidden_moves(pub, side, rule_facts, notes)
        if rule_forbidden:
            forbid = set(forbid) | set(rule_forbidden)

        search_opts = dict(
            quiescence=opts.get("quiescence", True),
            null_move=opts.get("null_move", True),
            extensions=opts.get("extensions", True),
            # 揭棋特有的搜索行为调整（见 jieqi/reveal.py）。
            # 配置里不给权重就是 None = 关闭，行为与改动前逐位一致。
            reveal_weights=opts.get("reveal_weights"),
            reveal_quiet_weights=opts.get("reveal_quiet_weights"),
        )
        searchers = [Searcher(b, ctx, search_opts) for b in boards]
        # 特征按**公开局面**算一次（整棵树共用）：它描述的是"这一手处在什么
        # 阶段"，逐节点重算既慢又没意义。用采样前局面的静态评估当 comeback 基准。
        if search_opts["reveal_weights"] or search_opts["reveal_quiet_weights"]:
            try:
                root_eval = float(evaluate(boards[0], ctx))
            except Exception:
                root_eval = 0.0
            for se in searchers:
                se.prepare_reveal(root_eval)

        per_move: Dict[int, List[int]] = {mv: [] for mv in root_moves}
        per_move_pv: Dict[int, List[int]] = {}
        sample_depths: List[int] = []
        total_nodes = 0
        order = list(root_moves)

        for idx, se in enumerate(searchers):
            left = deadline - time.monotonic()
            if left <= 0.0:
                break
            # 时间分配：每次都把"剩余时间"重新按剩余采样数平分。早期版本用
            # 固定预算（left / 总采样数），若第一个采样吃满预算，后面的就会
            # 一秒都分不到；现在最后一个采样会自然而然地吃掉剩下全部时间。
            remaining_samples = n_samples - idx
            se.deadline = time.monotonic() + left / max(1, remaining_samples)
            best_here: Dict[int, int] = {}
            depth_done = 0
            pv_here: List[int] = []
            for depth in range(1, int(opts["max_depth"]) + 1):
                before = time.monotonic()
                try:
                    bm, bs, pv, scores = se.search_root(
                        depth, root_moves, order=order, forbid=forbid)
                except TimeUp:
                    break
                best_here = scores
                depth_done = depth
                pv_here = pv
                if abs(bs) >= MATE_BOUND:
                    break
                # 这一层花的钱比剩余时间还多 -> 下一层肯定搜不完，提前收手
                used = time.monotonic() - before
                if used > max(0.0, se.deadline - time.monotonic()):
                    break
            total_nodes += se.nodes
            se.nodes = 0
            if depth_done == 0:
                continue
            sample_depths.append(depth_done)
            for mv, v in best_here.items():
                per_move.setdefault(mv, []).append(v)
            # 用本采样最深一次的主变作为该着法的代表主变
            for mv in best_here:
                if mv not in per_move_pv and pv_here and pv_here[0] == mv:
                    per_move_pv[mv] = pv_here
            order = sorted(root_moves,
                           key=lambda m: -best_here.get(m, -INF))
            if time.monotonic() >= deadline:
                break

        if not any(per_move.values()):
            # 时间被耗尽在一个采样都没跑完：退化为静态评估
            return self._fallback(pub, root_moves, ctx, t0, belief, forbid, notes)

        penalty = float(opts.get("variance_penalty", 0.0))
        stats: List[MoveStat] = []
        for mv in root_moves:
            vals = per_move.get(mv) or []
            if not vals:
                continue
            vals = self._stabilise_mates(vals)
            mean = sum(vals) / len(vals)
            st = MoveStat(move=mv, mean=mean, worst=min(vals),
                          best_count=0, samples=len(vals),
                          pv=per_move_pv.get(mv, []))
            stats.append(st)
        best_mean = max(s.mean for s in stats)
        for st in stats:
            vals = self._stabilise_mates(per_move[st.move])
            if len(vals) > 1:
                var = sum((v - st.mean) ** 2 for v in vals) / len(vals)
                stdev = math.sqrt(var)
            else:
                stdev = 0.0
            st.best_count = sum(1 for v in vals if v >= best_mean - 1)
            st.select_score = st.mean - penalty * stdev

        stats.sort(key=lambda s: s.select_score, reverse=True)
        # 被禁的着法（重复局面 / 长将 / 长捉）先整体拿掉再选：
        # 这样既保证"选出来的不是禁着"，也保证**真的没有别的着法时**仍然能
        # 走一步（全被禁就保留原样，总不能让 AI 停在那儿）。
        allowed = [st for st in stats if st.move not in forbid] if forbid else stats
        chosen = self._pick(allowed or stats, opts, rng)
        if chosen is None:
            chosen = (allowed or stats)[0]

        # ---- 送杀体检（独立于评估的强制线检查）----
        # 真实对局里出现过"评估说黑方大优，其实已经被叫杀"的崩盘：深搜的视野
        # 不够长，而对方那几步将军又是非吃子着法，静态搜索补不上。所以在根节点
        # 额外做一次**只看吃将**的小型证明搜索：把"走完会被对方强制吃将"的着法
        # 挑出来，只要还有别的着法不会被强制杀，就改走那一个。
        #
        # 这个体检有自己的节点/时间上限（``guard_node_cap`` / ``guard_time_cap``），
        # 超了就当"没结论"，绝不占用正常思考时间。
        if opts.get("mate_guard", True) and time.monotonic() < deadline - 0.05:
            guard_plies = max(2, int(opts.get("mate_guard_plies", 4)))
            searchers[0].guard_deadline(
                min(0.25, max(0.02, deadline - time.monotonic())))
            try:
                unsafe = searchers[0].forced_loss_moves(
                    [st.move for st in stats], plies=guard_plies)
            except TimeUp:
                unsafe = set()
            total_nodes += searchers[0].guard_nodes
            if unsafe:
                notes.append("%d 步会被对方在 %d 层内强制吃将"
                             % (len(unsafe), guard_plies))
                if chosen.move in unsafe:
                    safer = sorted((st for st in stats if st.move not in unsafe),
                                   key=lambda s: s.select_score, reverse=True)
                    if safer:
                        notes.append("放弃 %s（会被强制杀），改走 %s"
                                     % (move_str(chosen.move),
                                        move_str(safer[0].move)))
                        chosen = safer[0]

        side_sign = 1 if side == RED else -1
        red_score = chosen.mean * side_sign
        tree = self._build_tree(pub, stats, side_sign,
                               max(sample_depths) if sample_depths else 0,
                               red_score)

        return SearchResult(
            move=chosen.move,
            score=red_score,
            depth=max(sample_depths) if sample_depths else 0,
            nodes=total_nodes,
            samples=len(sample_depths),
            elapsed=time.monotonic() - t0,
            stats=stats,
            pv=chosen.pv or [chosen.move],
            verdict=verdict_of(red_score),
            belief_text=belief.describe(),
            notes=notes,
            tree=tree,
            sample_depths=sample_depths,
            side=int(side),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _stabilise_mates(vals: List[int]) -> List[int]:
        """避免"抽奖式杀棋"把平均分拉爆。

        在揭棋里，一条杀棋路径可能依赖某个暗子恰好是什么（例如对方必须没有
        某个防守子）。如果只有少数采样看到"杀"，那它并不是真正的杀。
        做法：**必须严格过半**的采样看到同一方向的杀，才认可这是杀；
        否则把这些样本的分值换成"非杀样本的中位数"，即按普通优势局面处理。

        注意必须用**严格过半**：采样数只有 2 时，"1 个采样看到杀"占 50%，
        如果按"≥ 一半"就算确认，平均分就会变成 (999998 + 842)/2 ≈ 500420，
        面板上会显示一个毫无意义的 50 万分（真实对局日志里出现过）。
        """
        if len(vals) < 2:
            return list(vals)
        mates = [v for v in vals if abs(v) >= MATE_BOUND]
        if not mates:
            return list(vals)
        if len(mates) * 2 > len(vals):      # 严格过半才算真的杀
            return list(vals)
        others = sorted(v for v in vals if abs(v) < MATE_BOUND)
        if not others:
            return list(vals)
        mid = len(others) // 2
        fallback = (others[mid] if len(others) % 2
                    else (others[mid - 1] + others[mid]) // 2)
        return [fallback if abs(v) >= MATE_BOUND else v for v in vals]

    def _pick(self, stats: List[MoveStat], opts: dict,
              rng: Optional[random.Random] = None) -> Optional[MoveStat]:
        """按难度选着：temperature=0 时直接取最优，否则按 softmax 随机。

        随机源用 ``analyse`` 里那个已播种的 ``rng``，所以只要设置了 ``seed``，
        整步决策（含低难度的随机性）都是完全可复现的。
        """
        if not stats:
            return None
        temp = float(opts.get("temperature", 0.0) or 0.0)
        if temp <= 0:
            return stats[0]
        top = stats[0].select_score
        pool = [s for s in stats if s.select_score >= top - 3 * temp]
        weights = [math.exp((s.select_score - top) / temp) for s in pool]
        total = sum(weights)
        if total <= 0:
            return stats[0]
        rnd = rng.random() if rng is not None else random.random()
        r = rnd * total
        acc = 0.0
        for s, w in zip(pool, weights):
            acc += w
            if r <= acc:
                return s
        return pool[-1]

    def _repetition_moves(self, pub: Board,
                          history_counts: Optional[Dict[str, int]]
                          ) -> set:
        """找出会让同一公开局面重复出现的着法。

        ``history_counts`` 来自 ``gui_adapter.repetition_counts``（公开签名 ->
        次数）。阈值这里仍按"再走一次就到第 3 次"的旧启发式判（保守避开），
        而**判和**的口径是 ``jieqi.rules.RULE['repetition_draw']``（本项目 = 6 次）：
        避开得比判和早，只会让 AI 更主动变招，不会误判和。
        """
        if not history_counts:
            return set()
        out = set()
        side = pub.turn
        for mv in pub.gen_moves(side):
            pub.make(mv)
            sig = pub.public_signature()
            pub.unmake()
            if history_counts.get(sig, 0) >= 2:
                out.add(mv)
        return out

    def _rule_forbidden_moves(self, pub: Board, side: int,
                              rule_facts: Optional[Sequence],
                              notes: List[str]) -> set:
        """会让走子方触犯**长将 / 长捉禁招**的根着法（没有就是空集）。

        用法说明（很重要）：``rule_facts`` 必须是**本局到当前为止**的
        ``MoveFacts`` 序列，而且最后一手应当就是走子方的对手（也就是"轮到
        走子方走"）。判断依赖"同一方接连多少回合在将军 / 捉子"，只看当前局面
        是判不出来的。

        没传 ``rule_facts``（或它为空）时**直接返回空集**，引擎行为与改动前
        完全一致——这是"现有 350+ 测试必须全绿"的前提。
        """
        if not rule_facts:
            return set()
        try:
            from .rules import forbidden_moves

            out = set(forbidden_moves(rule_facts, pub, side))
        except Exception as exc:          # pragma: no cover - 防御：绝不影响出招
            notes.append("规则禁招检查失败（已忽略）：%r" % (exc,))
            return set()
        if out:
            notes.append("%d 步会触犯长将/长捉禁招，已从候选中剔除"
                         % len(out))
        return out

    def _fallback(self, pub: Board, root_moves: Sequence[int],
                  ctx: EvalContext, t0: float, belief: Belief,
                  forbid: set, notes: List[str]) -> SearchResult:
        """极端情况（时间预算过小）下的兜底：只做一层静态评估。"""
        side = pub.turn
        scored = []
        for mv in root_moves:
            pub.make(mv)
            v = evaluate(pub, ctx)
            pub.unmake()
            scored.append((v if side == RED else -v, mv))
        scored.sort(reverse=True)
        best_v, best_mv = scored[0]
        if best_mv in forbid:
            for v, mv in scored:
                if mv not in forbid:
                    best_v, best_mv = v, mv
                    break
        notes = list(notes) + ["时间预算过小，已退化为静态评估"]
        stats = [MoveStat(move=mv, mean=float(v), worst=float(v), samples=1,
                          best_count=1, select_score=float(v))
                 for v, mv in scored]
        red = (best_v if side == RED else -best_v)
        return SearchResult(move=best_mv, score=red, depth=0, nodes=len(root_moves),
                            samples=1, elapsed=time.monotonic() - t0, stats=stats,
                            pv=[best_mv], verdict=verdict_of(red),
                            belief_text=belief.describe(), notes=notes,
                            tree=self._build_tree(pub, stats, 1 if side == RED else -1,
                                                  0, red), side=int(side))

    def _build_tree(self, pub: Board, stats: List[MoveStat], side_sign: int,
                    depth: int, value: float) -> dict:
        """生成给 tree.html / 1.py 用的决策树（保持原有 JSON 结构）。

        为了兼容原来的查看器，每个节点都补上 ``depth`` / ``maximizing`` /
        ``alpha`` / ``beta`` 字段（缺失时树状图只会显示 "Human"）。
        """
        children = []
        for st in stats:
            frm, to = st.move >> 7, st.move & 127
            piece = pub.piece_char(frm)
            node = {
                "type": "move",
                "depth": max(0, depth - 1),
                "maximizing": True,
                "alpha": None,
                "beta": None,
                "move": {"from": [frm // 9, frm % 9], "to": [to // 9, to % 9],
                         "piece": piece},
                "value": round(st.mean * side_sign, 2),
                "mean": round(st.mean, 2),
                "worst": round(st.worst, 2),
                "best_count": st.best_count,
                "samples": st.samples,
                "pruned": False,
                "children": [],
            }
            cur = node
            for step, mv in enumerate(st.pv[1:], start=1):
                cfrm, cto = mv >> 7, mv & 127
                child = {
                    "type": "move",
                    "depth": max(0, depth - 1 - step),
                    "maximizing": step % 2 == 0,
                    "alpha": None,
                    "beta": None,
                    "move": {"from": [cfrm // 9, cfrm % 9],
                             "to": [cto // 9, cto % 9],
                             "piece": pub.piece_char(cfrm)},
                    "value": None, "pruned": False, "children": [],
                }
                cur["children"].append(child)
                cur = child
            children.append(node)
        return {"type": "root", "depth": depth, "alpha": None, "beta": None,
                "maximizing": True, "value": round(value, 2), "children": children}
