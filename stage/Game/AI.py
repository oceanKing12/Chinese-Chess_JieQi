"""AI 控制器：把揭棋搜索引擎接到 GUI 上。

历史版本（1 层贪心 + 大量 bug）已经整体替换为 ``jieqi`` 引擎：

* 规则引擎：位棋盘 + 正确的揭棋走法（暗子按摆位身份行棋、翻开后按真实身份）；
* 搜索：迭代加深 Negamax / Alpha-Beta(PVS) + 置换表 + 杀手着法 + 历史启发
  + 静态搜索 + 空着裁剪；
* 不完全信息：对暗子身份做后验采样（determinization），跨采样聚合出期望最优着，
  评估时暗子取后验期望而不是真值——**AI 看不到任何暗子的真实身份**；
* 时间控制：默认每步 2 秒（``ai_config.json`` 可改），支持 easy/normal/hard/expert；
* 可选大模型顾问：讲解局面、或按权重轻微影响选着（默认只讲解、不影响）。

对外仍然保留旧接口 ``AdvancedAIController(gs).get_best_move() -> ((sr,sc),(er,ec))``，
所以 ``main.py`` 不需要关心内部实现。
"""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, List, Optional, Tuple

from jieqi.board import move_str
from jieqi.consts import RED
from jieqi.engine import DIFFICULTIES, Engine, SearchResult, verdict_of
from jieqi.gui_adapter import (
    build_llm_snapshot,
    captured_info,
    gui_board_to_engine,
    gui_color_to_side,
    repetition_counts,
)
from jieqi.settings import ai_settings, config_path

try:  # 大模型顾问是可选项，缺了也不能影响下棋
    from jieqi.llm import LLMAdvisor, blend_bonus
    from jieqi.llm import load_config as load_llm_config
except Exception:  # pragma: no cover
    LLMAdvisor = None  # type: ignore
    blend_bonus = None  # type: ignore
    load_llm_config = None  # type: ignore


def _safe_print(text: str) -> None:
    """往控制台打印，遇到编码不支持的字符（例如 Windows 的 GBK 控制台遇到 ⚠）
    自动降级成"能打出来的近似字符"。

    为什么必须挡住：点评是**玩家可见的输出**，如果 print 抛
    ``UnicodeEncodeError``，整个点评就会变成"调用失败"，等于把好不容易
    拿到的分析结果弄丢了。
    """
    try:
        print(text)
        return
    except Exception:
        pass
    try:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, "replace").decode(encoding, "replace"))
    except Exception:
        try:
            print(text.encode("ascii", "replace").decode("ascii"))
        except Exception:
            pass


class AdvancedAIController:
    """GUI 使用的 AI 控制器（名字保持兼容，内部是全新引擎）。"""

    def __init__(self, game_state, difficulty: Optional[str] = None) -> None:
        self.gs = game_state
        self.settings = ai_settings()
        if difficulty:
            # 允许运行时切换难度（游戏里按 1/2/3/4 键）
            self.settings["difficulty"] = difficulty
            self.settings["time_limit"] = None
            self.settings["samples"] = None
            self.settings["max_depth"] = None
        self.difficulty = self.settings.get("difficulty", "normal")
        self.engine = Engine(
            difficulty=self.difficulty,
            time_limit=self.settings.get("time_limit"),
            samples=self.settings.get("samples"),
            max_depth=self.settings.get("max_depth"),
            seed=self.settings.get("seed"),
            weights=self.settings.get("weights") or None,
            # 揭棋特有的翻子搜索加成（jieqi/reveal.py）。空 dict = 关闭。
            reveal_weights=self.settings.get("reveal_weights") or None,
            reveal_quiet_weights=self.settings.get("reveal_quiet_weights") or None,
        )
        self.use_hidden_captures = bool(self.settings.get("use_hidden_captures", False))
        #: 谁来出招：``python``（本机引擎，默认）/ ``llm``（大模型）。见 ai_config。
        #:
        #: ⚠ 曾经还有 ``abjchess`` / ``abjchess_then_python``（外部引擎 AB-JChess），
        #: 现在**已整体移除**。老配置里如果还写着它，会由
        #: :func:`jieqi.engines.normalize` 归一到 ``python``，不会让游戏起不来。
        configured_source = str(self.settings.get("move_source") or "python").lower()
        self.last_source = "python"
        self._move_source = "python"
        self.move_source = configured_source           # 走 setter 归一化一次
        #: 是否直接采用大模型的招法（顶部配置区可开关；默认跟随配置文件）
        self.apply_llm_move = bool(self.settings.get("apply_llm_move", False))
        preset = DIFFICULTIES.get(str(self.difficulty).lower(), {})
        self.time_limit = self.settings.get("time_limit") or preset.get("time_limit")
        self.last_result: Optional[SearchResult] = None
        self.last_comment: str = ""
        self.last_snapshot: Optional[dict] = None
        self.last_llm_result = None
        self.history: List[str] = []
        self.advisor = None
        self.llm_blend_weight = 0.0
        self.llm_style = "explain"
        self.llm_status = ""
        self.llm_config = None
        self.llm_model_label = ""
        self.llm_enabled = False
        #: 点评归属质检（发现红黑/明暗说反就自动改判或定向重问）
        self.llm_validate = True
        #: 这次点评是针对第几手的局面（异步回调时 gs.ply 可能已经变了）
        self.comment_ply = 0
        #: 这次点评是不是在讲"AI 自己的招法"（替人类分析时为 False，不请求点评）
        self.comment_for_ai = True
        #: 每次「丢弃在途点评」就 +1，回调发现代号变了就丢弃结果
        self._llm_generation = 0
        self._setup_advisor()

    # ------------------------------------------------------------------
    def _setup_advisor(self) -> None:
        """装配大模型顾问，并把「为什么没启用」明确打到控制台。

        这段诊断很重要：用户填了 key 却没反应时，必须能一眼看出是
        enabled=false、还是缺 key。

        **顾问只要配置齐了就装配，不看 ``enabled``**：``enabled`` 只决定
        "开局默认开不开点评"。如果这里因为 ``enabled=false`` 就跳过装配，
        ``toggle_llm()`` 会永远返回 False（"无法开启大模型点评"），
        启动页上的那个开关就变成了摆设——用户必须先去改 json 再重启。
        """
        if LLMAdvisor is None or load_llm_config is None:
            self.llm_status = "大模型模块不可用（jieqi.llm 导入失败）"
            _safe_print(f"[揭棋AI] {self.llm_status}")
            return
        try:
            cfg = load_llm_config()
        except Exception as exc:
            self.llm_status = f"读取大模型配置失败：{exc}"
            _safe_print(f"[揭棋AI] {self.llm_status}")
            return
        self.llm_config = cfg
        #: 点评是否做"归属质检"（把红黑/明暗说反的点评抓出来）
        self.llm_validate = bool(getattr(cfg, "comment_validation", True))
        for warning in getattr(cfg, "warnings", []) or []:
            _safe_print(f"[揭棋AI] 配置提醒：{warning}")
        if not getattr(cfg, "api_key", ""):
            self.llm_status = ("找不到 API Key：请填 llm.api_key，"
                               f"或设置环境变量 {getattr(cfg, 'api_key_env', 'OPENAI_API_KEY')}")
            _safe_print(f"[揭棋AI] {self.llm_status}")
            return
        # 配好了就装上；开不开由 llm_enabled 决定（启动页/配置文件都能改）
        try:
            self.advisor = LLMAdvisor(cfg)
        except Exception as exc:
            self.llm_status = f"大模型顾问装配失败：{exc}"
            _safe_print(f"[揭棋AI] {self.llm_status}")
            return
        self.llm_blend_weight = float(getattr(cfg, "blend_weight", 0.0) or 0.0)
        self.llm_style = getattr(cfg, "mode", "explain") or "explain"
        self.llm_enabled = bool(getattr(cfg, "enabled", False))
        self.llm_model_label = str(getattr(cfg, "model", "") or "")
        effect = ("会按权重影响选着" if self.llm_blend_weight > 0 else "只点评，不影响选着")
        check = "开" if self.llm_validate else "关"
        self.llm_status = (f"大模型顾问已装配：{cfg.model} @ {self.advisor.endpoint()}"
                           f"（{effect}，blend_weight={self.llm_blend_weight}，"
                           f"归属质检={check}，"
                           f"当前{'开启' if self.llm_enabled else '关闭'}"
                           f"——启动页或按 L 可随时打开）")
        _safe_print(f"[揭棋AI] {self.llm_status}")

    # ------------------------------------------------------------------
    def toggle_llm(self) -> bool:
        """运行时开关大模型点评（不改配置文件）。返回切换后的状态。"""
        if self.advisor is None:
            # 没配好就别假装打开了，把原因说清楚
            print(f"[揭棋AI] 无法开启大模型点评：{self.llm_status}")
            return False
        self.llm_enabled = not self.llm_enabled
        if not self.llm_enabled:
            self.discard_inflight_llm()
            self.last_comment = ""
        print(f"[揭棋AI] 大模型点评：{'已开启' if self.llm_enabled else '已关闭'}")
        return self.llm_enabled

    def toggle_apply_llm_move(self) -> bool:
        """开关"采用大模型招法"。开启后 AI 会直接下大模型推荐的着法。"""
        if self.advisor is None:
            print(f"[揭棋AI] 无法开启「应用大模型招法」：{self.llm_status}")
            return False
        self.apply_llm_move = not self.apply_llm_move
        if self.apply_llm_move and not self.llm_enabled:
            # 要用它的招法，当然得先让它说话
            self.llm_enabled = True
        print(f"[揭棋AI] 应用大模型招法：{'已开启' if self.apply_llm_move else '已关闭'}")
        return self.apply_llm_move

    def discard_inflight_llm(self) -> None:
        """丢弃在途/已过期的点评结果（回退、关闭开关时调用）。

        回退一步会让之前的点评对应到已经不存在的局面，所以直接把代号推进，
        回调回来时发现代号不对就丢掉——**回退不访问模型**。
        """
        self._llm_generation += 1

    # ------------------------------------------------------------------
    @property
    def move_source(self) -> str:
        """当前实际使用的出招方（``python`` / ``llm``）。

        ⚠ 老配置里可能还写着 ``abjchess``（外部引擎，已**整体移除**）。
        setter 里用 :func:`jieqi.engines.normalize` 归一 —— 这样无论谁改
        （构造函数读配置、启动页、脚本直接赋值）都归一得到，
        而且**不会让游戏起不来**（老配置的语义降级成本机引擎）。
        """
        return getattr(self, "_move_source", "python")

    @move_source.setter
    def move_source(self, value) -> None:
        from jieqi import engines as _engines

        text = str(value or "python").lower()
        normalized = _engines.normalize(text, fallback=_engines.PYTHON)
        if normalized != text and text:
            _safe_print(
                "[揭棋AI] move_source=%s 已不再支持，按 %s（本机引擎）走。"
                % (text, normalized))
        self._move_source = normalized

    # ------------------------------------------------------------------
    @property
    def ai_side(self) -> int:
        return gui_color_to_side(getattr(self.gs, "ai_color", "black") or "black")

    def _rule_facts(self):
        """取本局到当前显示的那一手为止的规则事实（``MoveFacts`` 序列）。

        为什么要在这一层取：引擎是"无状态"的（它只看到一份棋盘），而长将 /
        长捉的判据依赖**本局的着法历史**——所以只能由控制器从 ``GameState``
        取好再传进去。

        任何异常都退回 ``None``（= 引擎不做规则剔除），绝不让规则模块的问题
        把 AI 弄得出不了招。
        """
        try:
            from jieqi import rules as rules_mod

            return rules_mod.facts_from_game_state(self.gs, upto=self.comment_ply)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 按方出招：三种"对局方式"各走一条明确的分支
    # ------------------------------------------------------------------
    def _prepare(self, side: int):
        """一次分析的公共准备（board / 被吃信息 / 重复计数 / 规则事实）。"""
        gs = self.gs
        self._analyse_side = int(side)
        board = gui_board_to_engine(gs.board, turn=int(side))
        known, hidden_count = captured_info(
            getattr(gs, "move_history", None),
            use_hidden_captures=self.use_hidden_captures)
        return {
            "board": board,
            "known": known,
            "hidden": hidden_count,
            "counts": repetition_counts(gs),
            "rule_facts": self._rule_facts(),
        }

    def analyse_python(self, side: int) -> Optional[SearchResult]:
        """**只跑本机引擎**（对局方式 ``python``）。

        "完全采用本机引擎的招法"就是这里：不掺外部引擎、不掺大模型影响选着，
        返回的就是引擎自己的选择。
        """
        self.comment_ply = int(getattr(self.gs, 'ply', 0) or 0)
        ctx = self._prepare(side)
        self.last_source = "python"
        result = self.engine.analyse(
            ctx["board"], belief=None,
            history_counts=ctx["counts"],
            own_captured=ctx["known"],
            captured_hidden=ctx["hidden"],
            use_hidden_captures=self.use_hidden_captures,
            rule_facts=ctx["rule_facts"])
        self._after_analyse(result)
        return result

    def analyse_llm(self, side: int) -> Optional[SearchResult]:
        """**完全采用大模型推荐的招法**（对局方式 ``llm``）。

        流程（每一步都有明确理由）：

        1. 先跑一次本机引擎——**只为了给大模型一份候选着法与局面事实表**
           （提示词需要它，见 ``build_llm_snapshot``）。它选什么**不作数**；
        2. 问大模型要一手（同步，带超时）；
        3. **采用它的那一手**：
           * 在引擎候选里 → 直接采用，分数用候选里那一手的分数；
           * 不在候选里但**在我们规则下合法** → 照样采用，分数用
             ``Engine.score_move`` 现算（只跑一次浅层搜索，很便宜）；
           * 非法 / 没给着法 / 调用失败 → **退回本机引擎**并在棋谱与面板上
             明确标注"本步由本机引擎代走"。这一步是刻意的：让一个非法着法
             落盘会把棋局弄坏，比"这一手不是大模型给的"严重得多。

        返回的 ``SearchResult.notes`` 里必有来源说明，方便复盘时一眼看出
        "这一手是大模型下的还是本机引擎代下的"。
        """
        gs = self.gs
        self.comment_ply = int(getattr(gs, 'ply', 0) or 0)
        ctx = self._prepare(side)
        # 1) 本机引擎只当"候选与事实表的提供者"
        base = self.engine.analyse(
            ctx["board"], belief=None,
            history_counts=ctx["counts"],
            own_captured=ctx["known"],
            captured_hidden=ctx["hidden"],
            use_hidden_captures=self.use_hidden_captures,
            rule_facts=ctx["rule_facts"])
        if self.advisor is None or not self.llm_enabled:
            base.notes.append("大模型不可用（未装配或未开启），本步由本机引擎代走")
            self.last_source = "python"
            self._after_analyse(base)
            return base
        # 2) 问大模型
        try:
            snapshot = build_llm_snapshot(gs, base, ai_color=int(side))
        except Exception as exc:
            base.notes.append("大模型提示词组装失败（%r），本步由本机引擎代走" % exc)
            self.last_source = "python"
            self._after_analyse(base)
            return base
        self.last_snapshot = snapshot
        try:
            llm = self._ask_llm(snapshot, "suggest", timeout=self._llm_budget())
        except Exception as exc:
            base.notes.append("大模型调用失败（%r），本步由本机引擎代走" % exc)
            self.last_source = "python"
            self._after_analyse(base)
            return base
        self.last_llm_result = llm
        comment = self._comment_of(llm)
        # 3) 采用它那一手
        if llm is not None and getattr(llm, "ok", False) and getattr(llm, "move", None):
            adopted = self._adopt_llm_move(base, llm.move, legal=ctx["board"]
                                           .gen_moves(int(side)), side=int(side))
            if adopted:
                self.last_comment = (comment or "") + "（这一手由大模型决定）"
                self.last_source = "llm"
                self._after_analyse(base)
                return base
            base.notes.append("大模型的着法在本机规则下不合法或无法解析，"
                              "本步由本机引擎代走")
            self.last_comment = (comment or "") + "（它的着法不可用，本步由本机引擎代走）"
        else:
            error = getattr(llm, "error", "") if llm is not None else ""
            base.notes.append("大模型没有给出着法，本步由本机引擎代走")
            self.last_comment = (comment or error or "") + "（未给出着法，本步由本机引擎代走）"
        self.last_source = "python"
        self._after_analyse(base)
        return base

    def _after_analyse(self, result: SearchResult) -> None:
        """每次分析之后的公共收尾（历史记录 + 顶栏读数）。"""
        self.last_result = result
        self.history.append(result.summary())
        del self.history[:-50]
        self._emit(result)

    def analyse(self, comment: bool = True,
                side: Optional[int] = None) -> Optional[SearchResult]:
        """完成一次完整分析（含可选的大模型顾问）。

        ``comment=False`` 表示这次分析是**替人类**做的：仍然给出着法与评估，
        但**不请求大模型点评**——点评只针对"轮到谁走"那一方自己的招法，
        否则玩家会发现"我走棋，AI 却在点评"，看不出点评到底在说谁。

        ``side`` 是这次替**哪一方**想招（``RED`` / ``BLACK``）。不传就按
        ``gs.current_player`` 推（老调用点保持有效）。显式传它的意义：
        每方可以配不同的引擎（见 :mod:`jieqi.engines`），主循环按
        ``gs.side_engine(current_player)`` 挑入口，不再依赖"AI 只执一方"这个
        旧假设。
        """
        gs = self.gs
        turn = gui_color_to_side(gs.current_player) if side is None else int(side)
        self._analyse_side = int(turn)
        #: 记下这次分析是针对第几手的局面：点评/评估要挂到这个步数上，
        #: 因为异步点评常常在 AI 落子之后才回来（那时 gs.ply 已经变了）。
        self.comment_ply = int(getattr(gs, 'ply', 0) or 0)
        #: 这次分析是不是"AI 自己的招法"（决定要不要点评、点评怎么措辞）
        self.comment_for_ai = bool(comment)
        board = gui_board_to_engine(gs.board, turn=turn)
        known, hidden_count = captured_info(
            getattr(gs, "move_history", None),
            use_hidden_captures=self.use_hidden_captures)
        counts = repetition_counts(gs)
        #: 本局到当前为止的**规则事实**（长将 / 长捉要按"连续多少回合"判，
        #: 只看当前局面判不出来）。取不到就传 None，引擎行为与以前一致。
        rule_facts = self._rule_facts()

        # 本机引擎。曾经这里会先问一次外部引擎（``abjchess``），
        # 那个档位已**整体移除**，所以现在只有这一条路径。
        self.last_source = "python"
        result = self.engine.analyse(
            board, belief=None,
            history_counts=counts,
            own_captured=known,
            captured_hidden=hidden_count,
            use_hidden_captures=self.use_hidden_captures,
            rule_facts=rule_facts)
        self.last_result = result
        self.history.append(result.summary())
        del self.history[:-50]

        self._emit(result)

        # 只点评 AI 自己的招法；替人类分析（comment=False）时只算不评。
        if (comment and self.advisor is not None and self.llm_enabled
                and result.move is not None):
            self._consult_llm(gs, result)

        if self.settings.get("emit_decision_tree", True):
            self._write_tree(result)
        return result

    def get_best_move(self, comment: bool = True
                      ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """兼容旧接口：返回 ``((起行,起列),(止行,止列))``。"""
        result = self.analyse(comment=comment)
        if result is None or result.move is None:
            return None
        frm, to = result.move >> 7, result.move & 127
        return (frm // 9, frm % 9), (to // 9, to % 9)

    def get_best_move_verbose(self, comment: bool = True) -> Optional[SearchResult]:
        return self.analyse(comment=comment)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #: 质检发现问题又改不回来时，加在点评后面的显式警告（玩家一眼能看出
    #: "这句话不可信"，而不是被模型的红黑颠倒忽悠）。刻意用 ASCII 的 "!!"，
    #: 因为警告会走到控制台，Windows GBK 控制台打不出 ⚠ 之类的符号。
    LLM_TRUST_WARNING = "（!! 这段点评与局面归属矛盾，可能把红黑说反了，仅供参考）"

    def _comment_of(self, result) -> str:
        """从一次模型结果里取出点评，并做「归属质检 + 自动改判」。

        * 干净 → 原样返回；
        * 只是把"我方/对方"整体说反了 → 已由 ``suggest_checked`` 自动改回来；
        * 改不回来 → 返回原文 + 明确警告，绝不假装它是对的。
        """
        if result is None or not getattr(result, "ok", False):
            return ""
        text = (getattr(result, "comment", "") or "").strip()
        if not text:
            return ""
        if not self.llm_validate or self.advisor is None:
            return text
        try:
            check = self.advisor.validate_result(result, self.last_snapshot)
        except Exception:
            return text
        if check.get("ok"):
            return check.get("text") or text
        issues = "；".join(check.get("issues") or []) or "点评与事实表矛盾"
        _safe_print(f"[揭棋AI][大模型][质检] {issues}")
        notes = "；".join(check.get("notes") or [])
        if notes:
            _safe_print(f"[揭棋AI][大模型][质检] {notes}")
        fixed = (check.get("text") or "").strip()
        if fixed:
            return fixed
        return text + self.LLM_TRUST_WARNING

    def _ask_llm(self, snapshot: dict, style: str, timeout: float = 0.0):
        """问一次模型；开启质检时走 ``suggest_checked``（含自动改判/定向重问）。

        ``timeout > 0`` 时把整个"提问 + 重问"过程限制在 ``timeout`` 秒内，
        超时就不再重问，避免开启"采用大模型招法"后每步拖太久。
        """
        if self.advisor is None:
            return None
        if not self.llm_validate:
            return self.advisor.suggest(snapshot, style)
        budget = None
        if timeout > 0:
            import time as _time

            budget = _time.monotonic() + float(timeout)

        def remaining() -> float:
            if budget is None:
                return 1e9
            return budget - _time.monotonic()

        retries = 1 if remaining() >= 1.5 else 0
        try:
            return self.advisor.suggest_checked(snapshot, style=style,
                                                max_retries=retries)
        except TypeError:      # 兼容老版本签名
            return self.advisor.suggest(snapshot, style)

    def _consult_llm(self, gs, result: SearchResult) -> None:
        """问大模型。

        * ``apply_llm_move`` 开启时：**同步**拿它的推荐着法，合法就直接采用；
        * 否则 ``blend_weight <= 0`` 时只异步取点评，绝不影响选着。

        两条同步路径的点评同样要过「归属质检」，不能因为"顺便问了招法"
        就把一段红黑颠倒的点评直接贴到面板上。
        """
        try:
            snapshot = build_llm_snapshot(gs, result, ai_color=self.ai_side)
        except Exception as exc:  # 组装快照失败不能影响下棋
            self.last_comment = f"大模型提示词组装失败：{exc}"
            _safe_print(f"[揭棋AI][大模型] {self.last_comment}")
            return
        self.last_snapshot = snapshot

        # ---- 采用大模型招法（同步，会带来一次请求的延迟） ----
        if self.apply_llm_move:
            self.last_comment = "大模型思考中…（将采用它推荐的招法）"
            try:
                llm = self._ask_llm(snapshot, "suggest", timeout=self._llm_budget())
            except Exception as exc:  # pragma: no cover
                self.last_comment = f"大模型调用失败：{exc}"
                _safe_print(f"[揭棋AI][大模型] {self.last_comment}")
                return
            self.last_llm_result = llm
            comment = self._comment_of(llm)
            if llm is not None and llm.ok and llm.move:
                # ⚠ 一定要把**合法着法集合**传进去：这条路径（"完全采用大模型
                # 招法"的开关版）以前只认"引擎候选"，于是既漏掉了"合法但不在
                # 候选里"的好棋，又把"根本非法"的着法当成"不在候选"含糊过去。
                # 传了 legal 之后，"合法就采用 / 非法就拒绝"这条线才真正成立。
                # （``turn`` 是 ``analyse`` 里的局部量，这里按同一口径重算。）
                turn = int(getattr(self, "_analyse_side", None)
                           if getattr(self, "_analyse_side", None) is not None
                           else gui_color_to_side(gs.current_player))
                legal_moves = gui_board_to_engine(
                    gs.board, turn=turn).gen_moves(turn)
                if self._adopt_llm_move(result, llm.move, legal=legal_moves,
                                        side=turn):
                    self.last_comment = (comment or "") + "（已采用大模型招法）"
                else:
                    result.notes.append("大模型的着法不在合法着法里，已按引擎着法")
                    self.last_comment = (comment or "") + "（它的着法不合法，已按引擎着法）"
            else:
                result.notes.append("大模型没有给出着法，已按引擎着法")
                error = getattr(llm, "error", "") if llm is not None else ""
                self.last_comment = (comment or error or "") + "（未给出着法，按引擎着法）"
            _safe_print(f"[揭棋AI][大模型] {self.last_comment}")
            return

        if self.llm_blend_weight > 0 and blend_bonus is not None:
            self.last_comment = "大模型思考中…（会按权重影响本步选着）"
            try:
                llm = self._ask_llm(snapshot, "suggest", timeout=self._llm_budget())
            except Exception as exc:  # pragma: no cover
                self.last_comment = f"大模型调用失败：{exc}"
                _safe_print(f"[揭棋AI][大模型] {self.last_comment}")
                return
            self.last_llm_result = llm
            if llm is not None and llm.ok and llm.move:
                bonuses = blend_bonus(llm, snapshot.get("engine_candidates") or [],
                                      self.llm_blend_weight)
                self._apply_llm_bonus(result, bonuses)
            self.last_comment = (self._comment_of(llm)
                                 or getattr(llm, "error", "") or "")
            _safe_print(f"[揭棋AI][大模型] {self.last_comment}")
            return

        self.last_comment = "大模型思考中…（仅点评，不影响选着）"
        generation = self._llm_generation

        def _cb(res) -> None:
            if generation != self._llm_generation:
                return          # 局面已经回退/点评已关闭，丢弃这次结果
            self.last_llm_result = res
            if getattr(res, "ok", False):
                comment = self._comment_of(res)
                self.last_comment = comment or "（模型没有给出点评）"
                usage = getattr(res, "usage", None)
                print(f"[揭棋AI][大模型] {self.last_comment}"
                      + (f"  [latency={getattr(res, 'latency', 0):.2f}s usage={usage}]"
                         if usage else ""))
            else:
                self.last_comment = f"大模型调用失败：{getattr(res, 'error', '未知错误')}"
                _safe_print(f"[揭棋AI][大模型] {self.last_comment}")

        try:
            # 异步也走质检（在线程里重问，不阻塞主线程）
            self.advisor.suggest_async(snapshot, callback=_cb, style=self.llm_style,
                                       validate=self.llm_validate)
        except TypeError:      # 兼容老版本 suggest_async（没有 validate 参数）
            try:
                self.advisor.suggest_async(snapshot, callback=_cb, style=self.llm_style)
            except Exception as exc:
                self.last_comment = f"大模型请求发起失败：{exc}"
                _safe_print(f"[揭棋AI][大模型] {self.last_comment}")
        except Exception as exc:
            self.last_comment = f"大模型请求发起失败：{exc}"
            _safe_print(f"[揭棋AI][大模型] {self.last_comment}")

    def _llm_budget(self) -> float:
        """同步请求（含可能的定向重问）允许占用的秒数。"""
        try:
            timeout = float(getattr(self.llm_config, "timeout", 0) or 0)
        except (TypeError, ValueError):
            timeout = 0.0
        if timeout <= 0:
            timeout = 20.0
        return min(60.0, timeout * 2.0 + 3.0)

    def _adopt_llm_move(self, result: SearchResult, move,
                        legal: Optional[set] = None,
                        side: Optional[int] = None) -> bool:
        """把大模型推荐的着法设为最终着法。

        与旧版的关键区别：旧版**只接受"本来就在引擎候选里"的着法**，于是
        "完全采用大模型招法"实际上变成了"从引擎的小名单里挑一个"——大模型
        真正想下的那手只要不在短名单里就被丢掉。

        现在两档都收：

        * 在候选里 → 用候选里那一手的分数与主变；
        * **不在候选里但在我们规则下合法** → 照样采用，分数用
          ``Engine.score_move`` 现算一次浅搜（大模型的着法可能比引擎候选更好，
          没理由因为"引擎没想到"就否掉它）；

        传了 ``legal``（合法着法集合）时，**不合法一律拒绝**——绝不让非法的
        着法落到棋盘上。

        ``side`` 是这次替哪一方想招，用来把分数换算成红方视角
        （旧版读 ``gs.current_player``，两方都是引擎时会算错符号）。
        """
        try:
            fr, fc, tr, tc = (int(x) for x in move)
        except Exception:
            return False
        from jieqi.board import encode as _encode

        target = _encode(fr * 9 + fc, tr * 9 + tc)
        if legal is not None and target not in set(legal):
            # 关键一步：**非法着法绝不能落盘**。也一定要留一条 note——
            # 调用方与测试都靠它判断"这一手为什么不是大模型下的"，
            # 只 print 到控制台等于复盘时查无实据。
            print(f"[揭棋AI][大模型] 它推荐的 {move_str(target)} 不是合法着法，忽略")
            result.notes.append("大模型的着法 %s 在本机规则下不合法，已忽略"
                                % move_str(target))
            return False
        if side is None:
            side = getattr(self, "_analyse_side", None)
        if side is None:
            try:
                side = gui_color_to_side(self.gs.current_player)
            except Exception:
                side = RED
        side = int(side)
        sign = 1 if side == RED else -1
        stats = {st.move: st for st in result.stats}
        if target in stats:
            stat = stats[target]
            result.notes.append("采用大模型招法 %s" % move_str(target))
            result.move = target
            result.pv = stat.pv or [target]
            # ⚠ 符号口径：``MoveStat.mean`` 是**走子方视角**，而
            # ``SearchResult.score`` / ``verdict`` 是**红方视角**（界面与棋谱
            # 都用后者）。旧代码在这里乘的是"走子方符号"，等于把黑方的分数
            # 翻了两次（黑方走时显示成红方大优）。现在统一换算成红方视角。
            result.score = stat.mean * sign
            result.verdict = verdict_of(result.score)
            return True
        # 不在候选里：现算一个分数（浅搜），算不出来就按 0（不显示夸张数字）
        score = None
        try:
            score = self.engine.score_move(
                gui_board_to_engine(self.gs.board, turn=side), target, depth=1)
        except Exception:
            score = None
        result.notes.append("采用大模型招法 %s（不在引擎候选里，已现算评估）"
                            % move_str(target))
        result.move = target
        result.pv = [target]
        if score is not None:
            # ``Engine.score_move`` 返回的**已经是红方视角**（见它的文档），
            # 所以这里不再乘 sign。
            result.score = float(score)
            result.verdict = verdict_of(result.score)
        return True

    @staticmethod
    def _apply_llm_bonus(result: SearchResult,
                         bonuses: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]
                         ) -> None:
        if not bonuses:
            return
        side_sign = -1 if result.score < 0 and result.move is not None else 1
        touched = False
        for stat in result.stats:
            frm, to = stat.move >> 7, stat.move & 127
            key = ((frm // 9, frm % 9), (to // 9, to % 9))
            bonus = bonuses.get(key, 0.0)
            if bonus:
                stat.select_score += bonus
                touched = True
        if not touched:
            return
        result.stats.sort(key=lambda s: s.select_score, reverse=True)
        best = result.stats[0]
        if best.move != result.move:
            result.notes.append(f"参照大模型建议改选 {move_str(best.move)}")
            result.move = best.move
            result.pv = best.pv or [best.move]
            result.score = best.mean * side_sign

    # ------------------------------------------------------------------
    def _write_tree(self, result: SearchResult) -> None:
        path = self.settings.get("decision_tree_path") or "decision_tree.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(result.tree, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _emit(self, result: SearchResult) -> None:
        _safe_print(f"[揭棋AI] {result.summary()}")
        for note in result.notes:
            _safe_print(f"[揭棋AI] 备注：{note}")
        if self.settings.get("verbose"):
            for stat in result.stats[:6]:
                print(f"[揭棋AI]   {stat.describe()}")
        _safe_print(f"[揭棋AI] {result.belief_text}")


# --------------------------------------------------------------------------
# 兼容旧代码里可能用到的其他名字
# --------------------------------------------------------------------------
AIController = AdvancedAIController


def available_difficulties() -> List[str]:
    return list(DIFFICULTIES.keys())


def describe_settings() -> str:
    s = ai_settings()
    preset = DIFFICULTIES.get(str(s["difficulty"]).lower(), {})
    return (f"配置文件：{config_path()} | 难度：{s['difficulty']} | "
            f"思考时间：{s['time_limit'] or preset.get('time_limit')}s | "
            f"采样：{s['samples'] or preset.get('samples')}")
