# -*- coding: utf-8 -*-
"""揭棋（暗棋象棋）可选的大模型顾问模块。

设计目标
--------
* **默认关闭**：配置里 ``llm.enabled`` 为 ``false`` 时完全不联网，不产生任何影响。
* **绝不崩溃**：缺少网络、缺少 API Key、返回体畸形、超时……任何异常都被吞掉，
  统一转成 ``LLMResult(ok=False, error="中文错误说明")``；``suggest()`` /
  ``suggest_async()`` 永不向外抛异常。
* **纯标准库**：只使用 ``urllib.request`` / ``json`` / ``os`` / ``threading`` /
  ``dataclasses`` / ``typing`` / ``re`` / ``socket`` / ``time``，不导入 pygame、
  requests、openai 等第三方库。
* **导入零副作用**：模块导入时不读文件、不建连接、不起线程。

配置路径解析（``_default_config_path()``）
------------------------------------------
按顺序尝试，返回第一个"可用"的路径：

1. 环境变量 ``JIEQI_AI_CONFIG``（若该文件存在则直接采用；若被显式设置但文件
   不存在，则视为用户明确指定，同样直接采用，便于报出清晰错误）。
2. ``<本文件>/../ai_config.json`` —— 即 ``Game`` 包的父目录下的 ``ai_config.json``。
3. ``<本文件>/ai_config.json`` —— 即 ``jieqi`` 包内，作为只读环境的兜底。

"可用"的定义：文件存在；或文件不存在但其所在目录存在且可写（说明以后能写出来）。
若 2、3 都不可用，则返回第 2 项（首选路径）作为默认值。

.. note::
   任务书里给出的具体示例是 ``D:\\py\\Chinese Chess\\Game\\ai_config.json``
   （"Game 包的父目录"指的是生产工程 ``D:\\py\\Chinese Chess\\Game\\Game\\``？
   还是 ``Game`` 自身的父目录？两种读法不同）。本模块严格实现上面按 ``__file__``
   推导的规则，因此在当前交付布局下实际解析结果是
   ``D:\\py\\揭棋\\stage\\ai_config.json``；把本文件原样放到
   ``D:\\py\\Chinese Chess\\Game\\jieqi\\llm.py`` 时，解析结果就是
   ``D:\\py\\Chinese Chess\\Game\\ai_config.json``，与任务书示例一致。

配置优先级（已文档化的取舍）
----------------------------
* ``api_key``：文件内显式 ``api_key`` > 环境变量 ``api_key_env`` 指定的变量 > 空串。
* 其余字段：**环境变量优先于文件**（``JIEQI_LLM_ENABLED`` / ``JIEQI_LLM_BASE_URL`` /
  ``JIEQI_LLM_MODEL`` / ``JIEQI_LLM_TIMEOUT`` / ``JIEQI_LLM_MODE`` /
  ``JIEQI_LLM_BLEND_WEIGHT``）。这样运维可以在不改配置文件的前提下临时切换模型、
  打开开关或加大超时，符合"环境优先级更高"的常规约定。

快照（snapshot）契约假设
------------------------
所有字段都是可选的，缺失或为 ``None`` 时按"未知"处理，绝不报错。额外假设：

* ``board`` 为 10 行 × 9 列，行优先；格子为 ``None`` 或
  ``{"side": "red"/"black", "hidden": bool, "home": "炮"/"車"/..., "ident": str|None}``。
  未知键忽略，非 dict 的格子按"有子但信息未知"处理。
* 坐标形如 ``[row, col]``（也可写成 ``(row, col)`` 元组）。
* ``engine_candidates`` 按引擎评分从优到劣排列，每项可含
  ``from`` / ``to`` / ``score`` / ``winrate`` / ``pv`` / ``note``。
* ``blend_bonus`` 的键是 ``((fr, fc), (tr, tc))``；``weight=0`` 时全部为 0.0，
  即"纯建议、不影响搜索"。
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "POSITION_SYSTEM_PROMPT",
    "SIDE_ERROR_HINT",
    "STYLES",
    "RED_PIECE_CHARS",
    "BLACK_PIECE_CHARS",
    "LLMConfig",
    "LLMResult",
    "LLMAdvisor",
    "load_config",
    "render_position_text",
    "render_facts_text",
    "build_messages",
    "build_payload",
    "build_repair_hint",
    "parse_response",
    "piece_belongs_to",
    "piece_ownership",
    "own_pieces",
    "check_comment",
    "side_swap_error",
    "repair_comment",
    "validate_comment",
    "blend_bonus",
    "move_to_text",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

STYLES: Tuple[str, ...] = ("explain", "suggest", "review")
DEFAULT_STYLE = "explain"

#: 混权系数为 1.0、模型置信度为 1.0 时给予候选着的最大加分（厘兵）。
#: 数值刻意保守：LLM 只是顾问，不应压过引擎自身评分。
BLEND_MAX_BONUS_CP = 50.0

#: 历史记录上限。
HISTORY_LIMIT = 50

#: 网络返回体最大读取字节数（防止异常大响应撑爆内存）。
MAX_BODY_BYTES = 4 * 1024 * 1024

#: 线程池外的最小超时保护。
MIN_TIMEOUT = 0.5

TRUE_WORDS = frozenset({"1", "true", "yes", "y", "on", "t", "启用", "开"})
FALSE_WORDS = frozenset({"0", "false", "no", "n", "off", "f", "关闭", "关"})

SIDE_ZH = {"red": "红方", "black": "黑方"}

#: 棋盘渲染用的兜底列坐标（与方位列号一致）。
_FILE_NUMBERS = "１２３４５６７８９"
_RANK_NUMBERS = "０１２３４５６７８９"  # 索引 0 不用，1..9 取 1..9

POSITION_SYSTEM_PROMPT = """你是一位中国象棋（揭棋 / 暗棋）的资深教练，负责点评 **AI 一方自己即将走出的那一手**。

揭棋里未翻开的暗子身份未知，你必须把"已知"和"未知"分清楚，**绝不臆造事实**。
你会收到一份由本地引擎判定的「局面事实表」：归属、明暗、被吃、威胁都在里面算好了。
**凡是事实表里写明的，一律照抄，不要自己重新推断；事实表里没有的，不要编。**

【阵营对照（必须先背下来，点评里只允许用这套称呼）】
* 「我方」= AI 执的一方 = 事实表「本轮走棋方」里标着"本方"的那一方；
* 「对方」= 人类执的一方；
* 棋子的归属只能来自事实表，**绝不允许凭棋盘位置猜**。红方的子是 帅仕相马车炮兵，
  黑方的子是 将士象馬車砲卒，两套字形不同；混淆红黑是最严重的错误。
* 说"我方车"就必须真的是我方（= AI 那一方）的车。拿不准时写成"事实表里那枚 X"。

【写法对照（务必按这个读）】
* 坐标一律是「列字母 + 行数字」，例如 ``b3``：**列 a~i 从左到右，行 1~10 从下到上**，
  红方底线是第 1 行、黑方底线是第 10 行。一手棋写成 ``b3c3``（从 b3 走到 c3），
  与棋谱、走棋记录、引擎输出完全一致。
* 「红车 b4」：红方的车、已翻开、在 b4；「黑卒」同理。
* 「暗炮? b8」：**尚未翻开**的暗子，按"炮"的行棋规则走，真实兵种未知，只能推测。
  带问号 = 暗子；不带问号 = 已翻开的明子。绝不能说反，也不能说"暗子身份已确定"。
* 「我方吃掉对方：2 枚明子（黑卒、黑馬）+ 1 枚暗子」：被吃的暗子身份对双方都不可见，
  只能说"一枚暗子"，不许猜它是什么。
* 事实表里的「正被攻击」是引擎判定的事实，可以直接引用；不要凭空说"某子被攻击"。

【视角规则（错一条就算失败）】
* **点评对象**只有一个：**AI 一方自己即将走出的那一手**（事实表里候选 1 那一手）。
  不要去点评人类刚走的那一手，也不要泛泛评价整盘棋。
* 事实表里「本轮走棋方」若写着"本轮轮到本方走棋"，那就是 AI 要落子；
  若写着"上一手是对方走的；本轮轮到本方走棋"，说明人类刚走完，现在轮到 AI 应着。
* 解释 AI 这一手时用第一人称"我方"：想达到什么目的、冒什么风险、对方最可能怎么应。
* 绝对不要去指导人类一方该怎么下，也不要写成"红方应该如何如何"的泛泛建议。
* 事实表里「本次点评的这手棋（候选 N）」是引擎算出的招法，用自然语言注明了
  "我方/对方 + 兵种 + 起点 → 终点 + 吃子"。引用它时必须与这段注解完全一致。
* 不要复述棋盘，不要罗列事实表，落点放在"这步棋为什么好、之后会怎样"。

【上下文（让你记得住整盘棋）】
* 「走棋记录」是最近若干手（含翻子结果、吃子，并已标明"我方/对方"与（AI）），
  据此说明局势怎么走到现在；
* 「评估走势」是最近几步的引擎评分变化，指出形势转好还是转坏、拐点在哪一手；
* 「上一步点评」是你自己上次说的话。要保持前后一致；若改变判断，
  必须**明确说明改判的理由**，不要默不作声地自相矛盾。

【回答要求（必须严格遵守）】
一、只输出一个 JSON 对象，前后不要有任何解释文字、寒暄、Markdown 代码块或多余空行。
二、字段与类型固定如下：
    {
      "move": "b3c3",
      "comment": "用简体中文写 2 到 5 句点评",
      "confidence": 0.62,
      "tags": ["进攻", "弃子"]
    }
三、move 必须是事实表候选着法里的**第一手坐标**，格式严格为
    "起点+终点" 四个字符，例如 "b3c3"（从 b3 走到 c3）：
    列用字母 a~i（从左到右）、行用数字 0~9（红方底线是第 0 行）；
    **不要**写成 "(行,列)->(行,列)" 或 "b3-c3" 这类其它形式；
    如果所有候选着法都不值得推荐，move 必须为 null。
四、comment 用简体中文，说明推荐理由、局面要点（子力、攻势、暗子风险、防守漏洞）、
    对手最可能的应手，以及后续大致走向；若我方正被将军，必须先讲清楚怎么解将。
五、confidence 是 0.0 到 1.0 之间的数字。
六、tags 是 0 到 4 个简体中文短标签，例如 ["进攻","弃子","稳健"]，没有就给空数组。
七、输出必须能被 json.loads 直接解析。除这一个 JSON 对象外，不要输出任何其它内容。
八、写完自己复核一遍：每个出现的棋子都要能对上"红/黑"与"明/暗"；对不上就重写那句话。
"""

#: 点评被质检发现"红黑搞反"时，追加的系统提醒（用于定向重试）。
SIDE_ERROR_HINT = """【上一次回答被判定为不合格】
{issues}
请重写一次，只改这些错误，其它内容保持原有风格：
1. "我方"必须且只能是 AI 执的那一方；"对方"是另一方。归属以事实表为准。
2. 带问号的才是暗子，不带问号的是已翻开的明子。
3. 被吃的暗子身份未知，只能写"一枚暗子"。
仍然只输出那一个 JSON 对象。"""


_STYLE_HINTS = {
    "explain": "请讲解当前局面，给出推荐着法并说明理由。",
    "suggest": "请只给出推荐着法，并把 comment 写得极简（一句话以内）。",
    "review": "请按复盘的角度点评整体思路：双方的战略得失、关键转折点与后续改进建议。",
}

# ---------------------------------------------------------------------------
# 归属判定（治幻觉的核心工具）
# ---------------------------------------------------------------------------

#: 红方 / 黑方各兵种的汉字写法。两套字形**完全不重叠**，所以"某个字属于哪一方"
#: 是可以准确判定的——这正是下面"幻觉质检"能可靠工作的基础。
RED_PIECE_CHARS = frozenset({"帅", "仕", "相", "马", "车", "炮", "兵"})
BLACK_PIECE_CHARS = frozenset({"将", "士", "象", "馬", "車", "砲", "卒"})
PIECE_CHARS = RED_PIECE_CHARS | BLACK_PIECE_CHARS

#: 红方视角的说法在"我方 / 对方"里的两种叫法。
OWN_WORDS = ("我方", "己方", "本方", "我", "咱")
OPP_WORDS = ("对方", "敌方", "对手", "对面", "人家")

_SIDE_WORDS = OWN_WORDS + OPP_WORDS
#: 按长度降序，保证 "我方" 先于 "我" 命中。
_SIDE_WORDS_SORTED = tuple(sorted(_SIDE_WORDS, key=len, reverse=True))
_SIDE_CHARS = "红黑"
#: 归属词的各种写法 -> 规范写法（"我方" / "对方"）。
_SIDE_ALIASES = {
    "我": "我方", "咱": "我方", "己方": "我方", "本方": "我方", "我方": "我方",
    "对方": "对方", "敌方": "对方", "对手": "对方", "对面": "对方", "人家": "对方",
}
#: 对调归属词用的替换正则（按长度降序，先长后短）。
_SIDE_SPLIT_RE = re.compile("(?:%s)" % "|".join(_SIDE_WORDS_SORTED))
_CONTEXT_CHARS = frozenset("这那此该某俩两三四五各每每一的了着有是")
_PIECE_CLASS = "[" + "".join(sorted(PIECE_CHARS)) + "]"
#: 归属词与兵种字之间允许出现的黏连字（"我方的车" / "黑方的卒"）。
_GLUE_CHARS = frozenset("方的之了")
#: 「车可以吃」「卒顶上去」——靠动词判断主语的写法（动词字一定是单字，正则可靠）。
_ACTION = "吃|杀|将|攻|打|守|防|退|进|走|移|挪|动|挺|跳|飞|平|占|控|堵|拦|保|护|逃|躲|换|兑|弃|送|逼|压|牵|藏|亮|翻"
_VERB_CLAIM_RE = re.compile(
    r"(?<![%s])(%s)(?:可以|能|要|想|应该|会|被|将|可|正在|马上|直接|先|再)*(?:%s)"
    % ("".join(sorted(_CONTEXT_CHARS)) + _SIDE_CHARS, _PIECE_CLASS, _ACTION))


def _match_side_word(text: str, index: int) -> Optional[str]:
    """``text[index:]`` 开头是不是归属词（返回归一化写法，不是则 ``None``）。"""
    for word in _SIDE_WORDS_SORTED:
        if text.startswith(word, index):
            return _SIDE_ALIASES.get(word, "")
    return None


def _match_color_word(text: str, index: int) -> Optional[str]:
    """``text[index:]`` 开头是不是「红/黑（方）」；返回 ``"红"`` / ``"黑"``。"""
    ch = text[index:index + 1]
    if ch not in _SIDE_CHARS:
        return None
    if text[index + 1:index + 2] == "方":
        return ch
    # 「红车」「黑卒」这种紧贴兵种字的写法
    if text[index + 1:index + 2] in PIECE_CHARS:
        return ch
    return None


def scan_side_piece(text: Any) -> List[Dict[str, Any]]:
    """扫描点评文字，找出「归属词 + 兵种字」的每一次出现。

    返回每项形如::

        {"word": "我方"|"对方"|"", "color": "红"|"黑"|"",
         "piece": "车", "position": 5, "text": "我方的车"}

    为什么不用一个正则搞定？归属词紧贴兵种字的写法太多（"我方车""我方的车"
    "黑方卒""红車"），正则里的字符组在不同编码/写法下曾出现诡异的不匹配；
    逐字扫描虽然朴素，但**行为完全可预期**，而且归属判定这种事必须万无一失。
    """
    raw = _as_text(text)
    found: List[Dict[str, Any]] = []
    index = 0
    length = len(raw)
    while index < length:
        word = _match_side_word(raw, index)
        color = ""
        if word is None:
            color = _match_color_word(raw, index) or ""
            if not color:
                index += 1
                continue
        cursor = index + (len(word) if word else 1)
        # 跳过黏连字（的 / 方 / 之）
        while cursor < length and raw[cursor] in _GLUE_CHARS:
            cursor += 1
        ch = raw[cursor:cursor + 1]
        if ch in PIECE_CHARS:
            found.append({
                "word": word or "",
                "color": color,
                "piece": ch,
                "position": cursor,
                "text": raw[index:cursor + 1],
            })
            index = cursor + 1
        else:
            index += 1
    return found


def _color_side(color: str, own_side: Optional[int]) -> Optional[int]:
    """「红 / 黑」-> 引擎方；不知道我执哪方时返回 ``None``。"""
    if not color or own_side is None:
        return None
    return 0 if color == "红" else 1


def _claim_side(hit: Dict[str, Any], own_side: Optional[int]) -> Optional[int]:
    """把一次「归属词/颜色 + 兵种字」的出现翻译成"它声称这子属于哪一方"。

    注意「我方」本身不是颜色，必须结合 ``own_side`` 才知道是红还是黑——
    这正是模型最容易搞混的地方，也是这一层存在的意义。
    """
    word = _as_text(hit.get("word"))
    if word in OWN_WORDS:
        return own_side
    if word in OPP_WORDS:
        return None if own_side is None else 1 - own_side
    return _color_side(_as_text(hit.get("color")), own_side)


def _side_index(value: Any) -> Optional[int]:
    """``"red"``/``"红"``/``0`` -> 引擎方；识别不了返回 ``None``。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in (0, 1) else None
    text = str(value).strip().lower()
    if text in ("red", "r", "0", "红", "红方"):
        return 0
    if text in ("black", "b", "1", "黑", "黑方"):
        return 1
    return None


def _cn_word(side: Optional[int]) -> str:
    if side is None:
        return "未知"
    return "红" if side == 0 else "黑"


def piece_belongs_to(char: str) -> Optional[int]:
    """棋子汉字 -> 归属方（0 红 / 1 黑）；不是棋子字返回 ``None``。"""
    text = str(char or "")
    if not text:
        return None
    ch = text[0]
    if ch in RED_PIECE_CHARS:
        return 0
    if ch in BLACK_PIECE_CHARS:
        return 1
    return None


def piece_ownership(value: Any) -> List[Tuple[str, str, int]]:
    """从事实表（或棋子列表）里取出 ``(显示名, 兵种字, 归属方)`` 三元组列表。

    支持三种输入：快照 dict（取 ``llm_facts``）、``llm_facts`` 本身、或者
    ``["红车 b4", "暗炮? c8"]`` 这样的字符串列表。
    """
    if isinstance(value, (list, tuple, set, frozenset)):
        return _ownership_from_labels(value)
    data = _snapshot_view(value)
    facts = data.get("llm_facts") if isinstance(data.get("llm_facts"), dict) else data
    own_side = _side_index(facts.get("ai_color")) if isinstance(facts, dict) else None
    out: List[Tuple[str, str, int]] = []
    if isinstance(facts, dict):
        for key, side in (("own", own_side), ("opponent", None if own_side is None
                                              else 1 - own_side)):
            block = facts.get(key)
            if not isinstance(block, dict):
                continue
            resolved = _side_index(block.get("color"))
            if resolved is None:
                resolved = side
            labels = list(block.get("revealed") or ()) + list(block.get("hidden") or ())
            for label in labels:
                item = _ownership_from_label(str(label), resolved)
                if item:
                    out.append(item)
    if out:
        return out
    # 兜底：从棋盘快照的每一格自己推（事实表缺失时也还能工作）
    board = data.get("board") if isinstance(data, dict) else None
    if isinstance(board, (list, tuple)):
        for row in board:
            if not isinstance(row, (list, tuple)):
                continue
            for cell in row:
                if not isinstance(cell, dict):
                    continue
                side = _side_index(cell.get("side"))
                if side is None:
                    continue
                if cell.get("hidden"):
                    ch = str(cell.get("home") or "")
                    if ch:
                        out.append(("暗%s?" % ch, ch, side))
                else:
                    ch = str(cell.get("ident") or "")
                    if ch:
                        out.append(("%s%s" % (_cn_word(side), ch), ch, side))
    return out


def _ownership_from_labels(labels) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    for label in labels:
        item = _ownership_from_label(str(label), None)
        if item:
            out.append(item)
    return out


def _ownership_from_label(label: str, default_side: Optional[int]
                          ) -> Optional[Tuple[str, str, int]]:
    text = str(label or "")
    for ch in text:
        if ch in PIECE_CHARS:
            side = piece_belongs_to(ch)
            if side is None:
                side = default_side
            return (text, ch, side if side is not None else -1)
    return None


def own_pieces(value: Any):
    """``(我方 (名字,兵种) 集合, 对方 (名字,兵种) 集合)``。"""
    own_side = None
    data = _snapshot_view(value)
    facts = data.get("llm_facts") if isinstance(data.get("llm_facts"), dict) else data
    if isinstance(facts, dict):
        own_side = _side_index(facts.get("ai_color"))
    own, other = [], []
    for label, ch, side in piece_ownership(value):
        if own_side is None:
            # 没有 ai_color 时，退化成"按红黑分"
            (own if side == 0 else other).append((label, ch))
        else:
            (own if side == own_side else other).append((label, ch))
    return own, other


def _names(pairs) -> set:
    out = set()
    for label, ch in pairs:
        out.add(label)
        out.add(ch)
    return out

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """大模型顾问配置。所有字段都有安全默认值，缺失即用默认值。"""

    enabled: bool = False
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"
    timeout: float = 20.0
    temperature: float = 0.2
    max_tokens: int = 700
    mode: str = "explain"
    blend_weight: float = 0.0
    #: 点评是否做"归属质检 + 自动改判"（发现红黑搞反时自动换回或重问一次）。
    #: 默认开启：这是消灭大模型幻觉的关键一环，关掉只会让它胡说。
    comment_validation: bool = True
    extra_headers: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # -- 工具方法 ---------------------------------------------------------
    def redacted(self) -> "LLMConfig":
        """返回 ``api_key`` 已脱敏的副本，可安全打印 / 写日志。"""
        return replace(self, api_key=mask_secret(self.api_key))

    def to_dict(self, redact: bool = False) -> Dict[str, Any]:
        """转成可 JSON 序列化的 dict（便于 CLI 输出）。"""
        cfg = self.redacted() if redact else self
        return {
            "enabled": cfg.enabled,
            "base_url": cfg.base_url,
            "api_key": cfg.api_key,
            "api_key_env": cfg.api_key_env,
            "model": cfg.model,
            "timeout": cfg.timeout,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "mode": cfg.mode,
            "blend_weight": cfg.blend_weight,
            "comment_validation": cfg.comment_validation,
            "extra_headers": dict(cfg.extra_headers),
            "warnings": list(cfg.warnings),
        }


def mask_secret(secret: Any, keep: int = 4) -> str:
    """把密钥脱敏成 ``***abcd`` 形式（只保留末尾至多 4 个字符）。"""
    text = "" if secret is None else str(secret)
    if not text:
        return ""
    if len(text) <= keep:
        return "*" * len(text)
    return "*" * (len(text) - keep) + text[-keep:]


def _default_config_path() -> str:
    """按文档说明的顺序解析默认配置文件路径（绝对路径，已 normalize）。"""
    env_path = os.environ.get("JIEQI_AI_CONFIG")
    if env_path:
        # 用户显式指定：无条件采用，文件不存在时也报这个路径，错误信息更直白。
        return os.path.abspath(os.path.expanduser(env_path))

    module_dir = os.path.dirname(os.path.abspath(__file__))
    package_parent = os.path.dirname(module_dir)  # Game 包目录
    primary = os.path.normpath(os.path.join(package_parent, "ai_config.json"))
    fallback = os.path.normpath(os.path.join(module_dir, "ai_config.json"))

    if _path_usable(primary):
        return primary
    if _path_usable(fallback):
        return fallback
    return primary


def _path_usable(path: str) -> bool:
    """文件存在，或其所在目录存在且可写（说明将来可以写出来）。"""
    try:
        if os.path.isfile(path):
            return True
    except OSError:  # pragma: no cover - 极少数文件系统异常
        return False
    try:
        folder = os.path.dirname(path) or "."
        return os.path.isdir(folder) and os.access(folder, os.W_OK)
    except OSError:  # pragma: no cover
        return False


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_WORDS:
        return True
    if text in FALSE_WORDS:
        return False
    return default


def _as_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value).strip().replace("，", ".").replace(",", ".")
        try:
            return float(text)
        except (TypeError, ValueError):
            return default


def _as_int(value: Any, default: int) -> int:
    number = _as_float(value, None)
    if number is None:
        return default
    return int(number)


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalise_headers(value: Any, warnings: List[str]) -> Dict[str, str]:
    """``extra_headers`` 规整成 ``Dict[str, str]``，非法项丢弃并记警告。"""
    if not isinstance(value, dict):
        if value not in (None, ""):
            warnings.append("llm.extra_headers 不是对象，已忽略。")
        return {}
    headers: Dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            warnings.append("llm.extra_headers 中存在非法键，已忽略该项。")
            continue
        headers[key.strip()] = "" if item is None else str(item)
    return headers


def _normalise_mode(value: Any, warnings: List[str]) -> str:
    mode = _as_text(value, DEFAULT_STYLE).lower()
    if mode in STYLES:
        return mode
    if value not in (None, ""):
        warnings.append(
            "llm.mode=%r 不是合法取值（%s），已回退为 %r。"
            % (value, "/".join(STYLES), DEFAULT_STYLE)
        )
    return DEFAULT_STYLE


def load_config(path: Optional[str] = None) -> LLMConfig:
    """读取 JSON 配置里的 ``"llm"`` 段。

    * 文件不存在 / JSON 非法 / 缺少 ``llm`` 段 → 返回默认配置，并把中文警告写进
      ``cfg.warnings``，**不抛异常**。
    * 其它顶层键（例如另一个作者负责的 ``"ai"``）一律忽略。
    * 环境变量优先级见模块文档。
    """
    warnings: List[str] = []
    data: Dict[str, Any] = {}
    used_path: Optional[str] = None

    if path is None:
        path = _default_config_path()
    used_path = os.path.abspath(os.path.expanduser(str(path)))

    if not os.path.isfile(used_path):
        warnings.append("未找到配置文件：%s，已使用内置默认配置。" % used_path)
    else:
        try:
            with open(used_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except json.JSONDecodeError as exc:
            warnings.append("配置文件 JSON 解析失败（%s），已使用内置默认配置。" % exc)
        except OSError as exc:
            warnings.append("配置文件读取失败（%s），已使用内置默认配置。" % exc)
        except Exception as exc:  # pragma: no cover - 兜底
            warnings.append("配置文件读取出现未知错误（%s），已使用内置默认配置。" % exc)
        else:
            if isinstance(loaded, dict):
                data = loaded
            else:
                warnings.append("配置文件顶层不是 JSON 对象，已使用内置默认配置。")

    section = data.get("llm") if isinstance(data, dict) else None
    if section is None:
        if data or not warnings:
            warnings.append("配置缺少 llm 段，已使用内置默认配置。")
        section = {}
    elif not isinstance(section, dict):
        warnings.append("配置的 llm 段不是 JSON 对象，已使用内置默认配置。")
        section = {}

    defaults = LLMConfig()

    # --- 基本字段（文件） ---
    enabled = _as_bool(section.get("enabled"), defaults.enabled)
    base_url = _as_text(section.get("base_url"), defaults.base_url) or defaults.base_url
    api_key = _as_text(section.get("api_key"), "")
    api_key_env = _as_text(section.get("api_key_env"), defaults.api_key_env) or defaults.api_key_env
    model = _as_text(section.get("model"), defaults.model) or defaults.model
    temperature = _as_float(section.get("temperature"), defaults.temperature)
    max_tokens = _as_int(section.get("max_tokens"), defaults.max_tokens)
    mode = _normalise_mode(section.get("mode"), warnings)
    blend_weight = _as_float(section.get("blend_weight"), defaults.blend_weight)
    extra_headers = _normalise_headers(section.get("extra_headers"), warnings)

    timeout = _as_float(section.get("timeout"), None)
    if timeout is None:
        timeout = defaults.timeout
    if timeout <= 0:
        warnings.append("llm.timeout 不是正数，已改用默认值 %.1f 秒。" % defaults.timeout)
        timeout = defaults.timeout

    # --- 环境变量覆盖（环境优先于文件） ---
    env = os.environ.get
    raw_enabled = env("JIEQI_LLM_ENABLED")
    if raw_enabled not in (None, ""):
        enabled = _as_bool(raw_enabled, enabled)

    raw_base = env("JIEQI_LLM_BASE_URL")
    if raw_base:
        base_url = raw_base.strip()

    raw_model = env("JIEQI_LLM_MODEL")
    if raw_model:
        model = raw_model.strip()

    raw_timeout = env("JIEQI_LLM_TIMEOUT")
    if raw_timeout:
        env_timeout = _as_float(raw_timeout, None)
        if env_timeout is None or env_timeout <= 0:
            warnings.append("环境变量 JIEQI_LLM_TIMEOUT=%r 非法，已忽略。" % raw_timeout)
        else:
            timeout = env_timeout

    raw_mode = env("JIEQI_LLM_MODE")
    if raw_mode:
        env_mode = raw_mode.strip().lower()
        if env_mode in STYLES:
            mode = env_mode
        else:
            warnings.append("环境变量 JIEQI_LLM_MODE=%r 非法，已忽略。" % raw_mode)

    raw_weight = env("JIEQI_LLM_BLEND_WEIGHT")
    if raw_weight:
        env_weight = _as_float(raw_weight, None)
        if env_weight is None:
            warnings.append("环境变量 JIEQI_LLM_BLEND_WEIGHT=%r 非法，已忽略。" % raw_weight)
        else:
            blend_weight = env_weight

    # 点评质检默认开启；只有显式写 false / 0 才关闭（关掉就等于放任幻觉）。
    validate = _as_bool(env("JIEQI_LLM_VALIDATE"), _as_bool(
        section.get("comment_validation"), defaults.comment_validation))

    # --- API Key 优先级：文件显式 > api_key_env 指向的环境变量 > 空 ---
    if not api_key:
        env_key_name = api_key_env
        if env_key_name:
            env_key = os.environ.get(env_key_name, "")
            if env_key:
                api_key = env_key.strip()
    if not api_key:
        warnings.append(
            "未找到 API Key（可在配置 llm.api_key 填写，或设置环境变量 %s），大模型顾问不可用。"
            % api_key_env
        )

    # 数值保护
    if not (0.0 <= temperature <= 2.0):
        warnings.append("llm.temperature=%.3f 超出合理区间 [0,2]，已截断。" % temperature)
        temperature = min(2.0, max(0.0, temperature))
    if max_tokens <= 0:
        warnings.append("llm.max_tokens 不是正数，已改用默认值 %d。" % defaults.max_tokens)
        max_tokens = defaults.max_tokens

    return LLMConfig(
        enabled=enabled,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        model=model,
        timeout=float(timeout),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        mode=mode,
        blend_weight=float(blend_weight),
        comment_validation=bool(validate),
        extra_headers=extra_headers,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 坐标与棋盘渲染小工具
# ---------------------------------------------------------------------------


def _coord_text(value: Any) -> str:
    """``[row, col]`` / ``(row, col)`` / ``"b3"`` -> ``"b3"``；无法解析返回 ``"?"``。

    写法与棋谱、走棋记录一致（列字母 + 行数字，红方底线是第 1 行）。
    同时接受老记法 ``"(6,1)"``，这样老棋谱与老模型的输出都还读得懂。
    """
    from .notation import name_to_square, square_name

    if value is None:
        return "?"
    if isinstance(value, str):
        text = value.strip()
        got = name_to_square(text)
        if got is not None:
            return square_name(*got)
        numbers = re.findall(r"-?\d+", text)
        if len(numbers) >= 2:
            return _pair_name(numbers[0], numbers[1])
        return "?"
    if isinstance(value, dict):
        if "row" in value and "col" in value:
            return _pair_name(value.get("row"), value.get("col"))
        if "r" in value and "c" in value:
            return _pair_name(value.get("r"), value.get("c"))
        return "?"
    if isinstance(value, (list, tuple)):
        if len(value) < 2:
            return "?"
        return _pair_name(value[0], value[1])
    return "?"


def _pair_name(row: Any, col: Any) -> str:
    """``(row, col)`` -> ``"b3"``；不是合法格就退回 ``"r,c"`` 的字面写法。"""
    from .notation import square_name

    try:
        return square_name(int(row), int(col))
    except (TypeError, ValueError):
        return "%s,%s" % (_num_text(row), _num_text(col))


def _num_text(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def move_to_text(move: Any) -> Optional[str]:
    """把 ``((fr, fc), (tr, tc))`` / ``(fr, fc, tr, tc)`` 转成 ``"(fr,fc)->(tr,tc)"``。"""
    if move is None:
        return None
    if isinstance(move, (list, tuple)) and len(move) == 2:
        first, second = move
        if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
            if len(first) >= 2 and len(second) >= 2:
                return "(%s,%s)->(%s,%s)" % (
                    _num_text(first[0]),
                    _num_text(first[1]),
                    _num_text(second[0]),
                    _num_text(second[1]),
                )
        if isinstance(move[0], (int, float)) and isinstance(move[1], (int, float)):
            return None
    if isinstance(move, (list, tuple)) and len(move) >= 4:
        return "(%s,%s)->(%s,%s)" % (
            _num_text(move[0]),
            _num_text(move[1]),
            _num_text(move[2]),
            _num_text(move[3]),
        )
    if isinstance(move, str):
        text = move.strip()
        return text or None
    return None


def _cell_text(cell: Any) -> str:
    """把一个格子的值渲染成 1~2 个字符。"""
    if cell is None or cell == "" or cell == 0:
        return "．"
    if isinstance(cell, dict):
        side = _as_text(cell.get("side")).lower()
        side_char = {"red": "红", "black": "黑"}.get(side, "?")
        if cell.get("hidden"):
            return side_char + "暗"
        home = _as_text(cell.get("home") or cell.get("piece") or cell.get("name"))
        if home:
            return home[:2]
        return side_char + "明"
    if isinstance(cell, str):
        return cell.strip()[:2] or "？"
    return "？"


def render_board_text(board: Any, max_rows: int = 12) -> str:
    """把 10×9 的 ``board`` 渲染成简易中文棋盘图；非法输入返回说明文字。"""
    if not board:
        return "（无棋盘数据）"
    if not isinstance(board, (list, tuple)):
        return "（棋盘数据格式无法识别）"
    lines: List[str] = []
    for row_index, row in enumerate(board):
        if row_index >= max_rows:
            lines.append("……（略）")
            break
        if not isinstance(row, (list, tuple)):
            lines.append("%2d| ？" % row_index)
            continue
        cells = [_cell_text(cell) for cell in row]
        lines.append("%2d| %s" % (row_index, " ".join(cells)))
    if not lines:
        return "（无棋盘数据）"
    width = max(len(line) for line in lines)
    header = "   " + "".join(
        "%s " % (_FILE_NUMBERS[i] if i < len(_FILE_NUMBERS) else str(i)) for i in range(9)
    )
    rule = "   " + "-" * max(0, width - 3)
    return "\n".join([header.rstrip(), rule] + lines)


# ---------------------------------------------------------------------------
# 快照 → 提示词
# ---------------------------------------------------------------------------

_SNAPSHOT_KEYS: Tuple[str, ...] = (
    "side_to_move",
    "ai_color",
    "comment_target",
    "move_number",
    "last_move",
    "board_text",
    "board",
    "engine_candidates",
    "engine_depth",
    "engine_nodes",
    "engine_samples",
    "engine_verdict",
    "engine_warning",
    "repetition_warning",
    # 事实表（归属 / 明暗 / 被吃 / 威胁，由本地引擎判定）
    "llm_facts",
    "facts_text",
    "captured_pieces",
    # 上下文（让点评记得住整盘棋）
    "move_list",
    "eval_history",
    "previous_comments",
    "score_basis",
)


def _snapshot_view(snapshot: Any) -> Dict[str, Any]:
    """把任意输入规整成 dict（非 dict 输入返回空 dict，不抛异常）。"""
    if isinstance(snapshot, dict):
        return snapshot
    if snapshot is None:
        return {}
    try:
        return dict(snapshot)  # 兼容映射类对象
    except Exception:
        return {}


def render_position_text(snapshot: Any, include_board: bool = True) -> str:
    """把快照渲染成紧凑的简体中文文本块。

    ``include_board=False`` 时跳过棋盘图（快照的 ``board_text`` 仍然会被输出）。
    """
    snap = _snapshot_view(snapshot)
    lines: List[str] = []
    lines.append("【局面快照】")

    side = _as_text(snap.get("side_to_move")).lower()
    ai_color = _as_text(snap.get("ai_color")).lower()
    if side in SIDE_ZH:
        side_line = "轮到：%s" % SIDE_ZH[side]
    elif side:
        side_line = "轮到：%s" % side
    else:
        side_line = "轮到：未知"
    if ai_color in SIDE_ZH:
        side_line += "；AI 执：%s" % SIDE_ZH[ai_color]
    lines.append(side_line)

    target = _as_text(snap.get("comment_target"))
    if target:
        lines.append("点评对象：%s（请解释这步棋的意图，而不是泛泛评价局面）" % target)

    move_number = snap.get("move_number")
    if move_number is not None and move_number != "":
        lines.append("手数：第 %s 手" % _num_text(move_number))

    last_move = snap.get("last_move")
    if isinstance(last_move, dict):
        pieces = []
        if last_move.get("piece"):
            pieces.append("棋子 %s" % _as_text(last_move.get("piece")))
        if last_move.get("revealed_as"):
            pieces.append("翻开后是 %s" % _as_text(last_move.get("revealed_as")))
        if last_move.get("captured"):
            pieces.append("吃 %s" % _as_text(last_move.get("captured")))
        suffix = ("（%s）" % "，".join(pieces)) if pieces else ""
        lines.append(
            "上一手：%s -> %s%s"
            % (_coord_text(last_move.get("from")), _coord_text(last_move.get("to")), suffix)
        )
    elif last_move:
        lines.append("上一手：%s" % _as_text(last_move))
    else:
        lines.append("上一手：无（开局或未提供）")

    # --- 上下文：走棋记录 / 评估走势 / 上一步点评 ---
    move_list = snap.get("move_list")
    if isinstance(move_list, (list, tuple)) and move_list:
        lines.append("走棋记录（最近若干手）：")
        for item in move_list[-12:]:
            lines.append("  %s" % _as_text(item))
    eval_history = snap.get("eval_history")
    if isinstance(eval_history, (list, tuple)) and eval_history:
        lines.append("评估走势（引擎评分，最近在后）：")
        for item in eval_history[-6:]:
            lines.append("  %s" % _as_text(item))
    previous = snap.get("previous_comments")
    if isinstance(previous, (list, tuple)) and previous:
        lines.append("你上一步的点评（供前后对照，必要时请说明改判理由）：")
        for item in previous[-2:]:
            lines.append("  「%s」" % _as_text(item))

    if snap.get("repetition_warning"):
        lines.append("⚠ 引擎提示：局面可能出现重复/循环，注意变着。")

    # 棋盘
    external_board = _as_text(snap.get("board_text"))
    if external_board:
        lines.append("棋盘图（引擎提供）：")
        lines.append(external_board)
    elif include_board:
        lines.append("棋盘图（由快照 board 渲染）：")
        lines.append(render_board_text(snap.get("board")))

    # 局面事实表：归属 / 明暗 / 被吃 / 威胁，**由引擎判定后写死**
    facts_text = _as_text(snap.get("facts_text")) or render_facts_text(snap)
    if facts_text:
        lines.append("")
        lines.append(facts_text)

    # 候选着法
    candidates = snap.get("engine_candidates")
    lines.append("引擎候选着法（按优劣排序，坐标写法 b3c3 = 列字母 + 行数字）：")
    if isinstance(candidates, (list, tuple)) and candidates:
        for index, item in enumerate(candidates):
            if index >= 8:
                lines.append("  ……（其余候选已省略）")
                break
            if not isinstance(item, dict):
                lines.append("  %d. %s" % (index + 1, _as_text(item) or "未知候选"))
                continue
            from_text = _coord_text(item.get("from"))
            to_text = _coord_text(item.get("to"))
            parts = ["  %d. %s->%s" % (index + 1, from_text, to_text)]
            if item.get("note"):
                parts.append("名称=%s" % _as_text(item.get("note")))
            score = _as_float(item.get("score"), None)
            if score is not None:
                parts.append("评分=%.0f" % score)
            winrate = _as_float(item.get("winrate"), None)
            if winrate is not None:
                if winrate <= 1.0:
                    parts.append("胜率=%.0f%%" % (winrate * 100.0))
                else:
                    parts.append("胜率=%.0f%%" % winrate)
            pv = item.get("pv")
            if isinstance(pv, (list, tuple)) and pv:
                # 主变也统一成 ``b7b5`` 写法（老数据里可能是 ``(4,2)->(4,4)``）
                steps = [_coord_text(step) if not isinstance(step, str)
                         else (_as_text(step) or "?") for step in list(pv)[:6]]
                parts.append("主变=%s" % " ".join(steps))
            lines.append("，".join(parts))
    else:
        lines.append("  （引擎未提供候选着法）")

    # 引擎结论
    verdict = _as_text(snap.get("engine_verdict"))
    depth = snap.get("engine_depth")
    nodes = snap.get("engine_nodes")
    samples = snap.get("engine_samples")
    warning = _as_text(snap.get("engine_warning"))
    if warning:
        lines.append(warning)
    extra = []
    if depth not in (None, ""):
        extra.append("深度=%s" % _num_text(depth))
    if nodes not in (None, ""):
        extra.append("节点=%s" % _num_text(nodes))
    if samples not in (None, ""):
        extra.append("采样=%s" % _num_text(samples))
    if verdict:
        lines.append("引擎结论：%s%s" % (verdict, ("（%s）" % "，".join(extra)) if extra else ""))
    elif extra:
        lines.append("引擎结论：未提供（%s）" % "，".join(extra))
    else:
        lines.append("引擎结论：未提供")

    unknown = [key for key in snap.keys() if key not in _SNAPSHOT_KEYS]
    if unknown:
        lines.append("其它字段：%s" % "，".join(sorted(str(key) for key in unknown)))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 局面事实表 → 提示词（治幻觉的核心）
# ---------------------------------------------------------------------------


def _facts_view(value: Any) -> Dict[str, Any]:
    """从快照 / 事实表里取出事实表字典（取不到就返回空 dict）。"""
    data = _snapshot_view(value)
    facts = data.get("llm_facts")
    if isinstance(facts, dict):
        return facts
    if isinstance(data, dict) and ("own" in data or "opponent" in data):
        return data
    return {}


def _fact_move_text(candidate: Any, item: Any, board: Any, mover: Optional[int]) -> str:
    """把一条候选着法写成**不可能认错归属**的中文自然语言。

    这是消灭"我方走了对方棋子"这类幻觉的关键一步：坐标之外，还明确写出
    "我方(红方) 的「车」 从 a7 走到 a5，吃掉对方的「黑卒」"。
    """
    if not isinstance(item, dict):
        return ""
    move = _coerce_move((item.get("from"), item.get("to")))
    if move is None:
        move = _coerce_move(item.get("move"))
    if move is None:
        return ""
    fr, fc, tr, tc = move
    text = "本方(%s方，即 AI 执的一方)" % _cn_word(mover) if mover is not None else "本方"
    piece = _cell_at(board, fr, fc)
    if isinstance(piece, dict):
        side = _side_index(piece.get("side"))
        if side == mover:
            body = "本方(%s方)" % _cn_word(side)
        elif side is not None:
            body = "对方(%s方)" % _cn_word(side)
        else:
            body = "归属未知方"
        if piece.get("hidden"):
            home = _as_text(piece.get("home"))
            body += " 的暗子（按「%s」行棋，真实兵种未知）" % (home or "?")
        else:
            body += " 的「%s」" % (_as_text(piece.get("ident")) or "?")
        text = "%s 走 %s" % (text, body)
    # 坐标用 ``b7b5`` 这套（列字母 + 行号 0~9），与棋盘图、走棋记录、引擎一致
    text += " 从 %s 走到 %s" % (_pair_name(fr, fc), _pair_name(tr, tc))
    target = _cell_at(board, tr, tc)
    if isinstance(target, dict):
        side = _side_index(target.get("side"))
        if side is not None and side != mover:
            if target.get("hidden"):
                text += "，吃掉对方的一枚暗子（身份未知）"
            else:
                text += "，吃掉对方的「%s%s」" % (_cn_word(side),
                                                _as_text(target.get("ident")))
    return text


def _cell_at(board: Any, r: int, c: int) -> Any:
    if not isinstance(board, (list, tuple)):
        return None
    if not (0 <= r < len(board)):
        return None
    row = board[r]
    if not isinstance(row, (list, tuple)) or not (0 <= c < len(row)):
        return None
    return row[c]


def render_facts_text(value: Any) -> str:
    """把「局面事实表」（``gui_adapter.llm_facts`` 的产物）渲染成中文事实表。

    这份文本是**给模型照抄的底稿**：每一枚子的归属、明暗、位置都写死在里面，
    模型只需要组织语言。传入任意非法输入都返回空串，绝不抛异常。
    """
    facts = _facts_view(value)
    if not facts:
        return ""
    data = _snapshot_view(value)
    board = facts.get("board") or data.get("board")
    own_side = _side_index(facts.get("ai_color"))
    opp_side = None if own_side is None else 1 - own_side
    own_word = "我方(%s方)" % _cn_word(own_side)
    opp_word = "对方(%s方)" % _cn_word(opp_side)

    lines: List[str] = ["【局面事实表（引擎判定，全部照抄，不要自行推断）】"]
    lines.append(
        "* 本轮走棋方：%s。**AI（也就是「我」）执 %s方**，人类执 %s方；"
        "下文「我方」= %s方，「对方」= %s方。"
        % (_as_text(facts.get("ply_side")) or "未知",
           _cn_word(own_side), _cn_word(opp_side),
           _cn_word(own_side), _cn_word(opp_side)))
    lines.append(
        "* 归属铁律：事实表里写「我方」的一定是 %s 的子，写「对方」的一定是 %s 的子；"
        "红方子是 帅仕相马车炮兵，黑方子是 将士象馬車砲卒。凭位置猜归属＝错误。"
        % (_cn_word(own_side), _cn_word(opp_side)))

    def _block(prefix: str, block: Any) -> None:
        if not isinstance(block, dict):
            return
        revealed = list(block.get("revealed") or ())
        hidden = list(block.get("hidden") or ())
        lines.append("* %s的明子（**身份已确定**，%d 枚）：%s"
                     % (prefix, len(revealed),
                        "、".join(revealed) if revealed else "无"))
        lines.append("* %s的暗子（**尚未翻开，真实兵种未知**，%d 枚）：%s"
                     % (prefix, int(block.get("hidden_count") or len(hidden)),
                        "；".join(hidden) if hidden else "无"))
        note = _as_text(block.get("hidden_note"))
        if note:
            lines.append("    %s" % note)
        captured = list(block.get("captured") or ())
        if captured:
            # captured 里的条目已经自带「红/黑」前缀（明子）或写成「暗子」，直接用。
            lines.append("* %s已被吃掉的子（共 %d 枚）：%s"
                         % (prefix, len(captured),
                            "、".join(_as_text(item) for item in captured)))
        else:
            lines.append("* %s还没有棋子被吃。" % prefix)

    _block(own_word, facts.get("own"))
    _block(opp_word, facts.get("opponent"))

    check = facts.get("check")
    if isinstance(check, dict) and _as_text(check.get("text")):
        lines.append("* 将军情况：%s" % _as_text(check.get("text")))
    status = _as_text(facts.get("status"))
    if status:
        lines.append("* 子力对比：%s" % status)

    lines.append("* 被吃子总账：%s" % _captures_text(data, own_side))
    threats = facts.get("threats")
    if isinstance(threats, (list, tuple)) and threats:
        lines.append("* 攻击关系（引擎按规则算出的既成事实，可直接引用）：")
        for item in threats:
            lines.append("    - %s" % _as_text(item))
    else:
        lines.append("* 攻击关系：引擎没有发现互相攻击的棋子（不代表绝对安全）。")

    candidates = facts.get("engine_candidates") or data.get("engine_candidates")
    if isinstance(candidates, (list, tuple)) and candidates:
        lines.append("* 候选着法（按引擎评分从优到劣；每一条都已经写明是谁的子、吃什么）：")
        for index, item in enumerate(candidates[:8]):
            text = _fact_move_text(candidates, item, board, own_side)
            if not text and isinstance(item, dict):
                text = "%s->%s" % (_coord_text(item.get("from")), _coord_text(item.get("to")))
            if not text:
                continue
            score = _as_float(item.get("score"), None) if isinstance(item, dict) else None
            tail = "" if score is None else "，引擎评分 %+.0f" % score
            star = "（引擎首选，本次点评的这手棋）" if index == 0 else ""
            lines.append("    %d. %s%s%s" % (index + 1, text, tail, star))
    top = _top_move_text(data, own_side, board)
    if top:
        lines.append("* 本次点评的这手棋（候选 1）：%s" % top)
        lines.append("    点评 move 字段必须写成它的坐标写法，不得改动坐标。")
    return "\n".join(lines)


def _captures_text(data: Dict[str, Any], own_side: Optional[int]) -> str:
    """被吃子总账（从 ``captured_pieces`` 或事实表 own/opponent 里拼）。"""
    captured = data.get("captured_pieces")
    bits: List[str] = []
    if isinstance(captured, dict):
        for key, label in (("red", "红方"), ("black", "黑方")):
            items = captured.get(key)
            if not isinstance(items, (list, tuple)) or not items:
                continue
            side = _side_index(key)
            who = "我方" if (own_side is not None and side == own_side) else "对方"
            bits.append("%s（%s）损失：%s" % (label, who, "、".join(str(x) for x in items)))
    if not bits:
        bits.append("未被吃掉任何棋子，或没有提供被吃记录")
    return "；".join(bits)


def _top_move_text(data: Dict[str, Any], own_side: Optional[int],
                   board: Any = None) -> str:
    """把候选 1 写成"谁走哪枚子到哪里"的一句话。"""
    candidates = data.get("engine_candidates")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return ""
    item = candidates[0]
    if not isinstance(item, dict):
        return ""
    if board is None:
        board = data.get("board")
    text = _fact_move_text(candidates, item, board, own_side)
    move = _coerce_move((item.get("from"), item.get("to")))
    coord = ""
    if move is not None:
        from .notation import move_name

        fr, fc, tr, tc = move
        coord = move_name((fr, fc), (tr, tc))
    if text and coord:
        return "%s；坐标写法 %s" % (text, coord)
    return text or coord


# ---------------------------------------------------------------------------
# 点评质检：把"红黑搞反 / 明暗说错"这类幻觉抓出来
# ---------------------------------------------------------------------------


def check_comment(comment: Any, snapshot: Any = None) -> List[str]:
    """检查点评里有没有"把自己和对方的子搞混"这类硬伤。

    返回问题清单（空列表 = 没发现问题）。判定只依赖**可证明的矛盾**：
    点评里显式出现的归属说法与事实表冲突（"我方车"但我方没有车），
    或者某个兵种被挂到了错误的归属词下面。故意不猜语义，
    宁可漏报也不误报——误报会把好点评也丢掉。
    """
    text = _as_text(comment)
    if not text:
        return []
    own_side = None
    facts = _facts_view(snapshot)
    if facts:
        own_side = _side_index(facts.get("ai_color"))
    if own_side is None:
        own_side = _side_index(_snapshot_view(snapshot).get("ai_color"))
    if own_side is None:
        return []          # 连谁执哪方都不知道，就没有判定基准

    pairs, chars_by_side = _piece_index(snapshot)
    issues: List[str] = []

    for hit in scan_side_piece(text):
        ch = hit["piece"]
        side = _claim_side(hit, own_side)
        if side is None:
            continue
        # "我方车" -> 棋盘上必须真的存在"我方 + 车"这枚子，否则就是编的。
        # 这条判据同时覆盖两种情况：
        #   * 归属说反了（车是对方的）；
        #   * 该方根本没有这个兵种（AI 执黑却说"我方帅"）。
        if (side, ch) not in pairs:
            belongs = chars_by_side.get(ch)
            if belongs is None:
                issues.append(
                    "点评里说「%s」，但%s方根本没有这个兵种（编造）。"
                    % (hit["text"], "我" if side == own_side else "对"))
            else:
                issues.append(
                    "点评里说「%s」，但「%s」是%s方的子（%s）。"
                    % (hit["text"], ch, _cn_word(belongs),
                       "我方" if belongs == own_side else "对方"))

    for found in _VERB_CLAIM_RE.finditer(text):
        ch = found.group(1)
        # 棋子字**前面最近**的归属词说了算（找不到就说明只是泛泛而谈，不判）。
        label = _nearest_side_word(text, found.start())
        if not label:
            continue
        side = own_side if label == "我方" else 1 - own_side
        if (side, ch) in pairs:
            continue
        belongs = chars_by_side.get(ch)
        if belongs is None:
            issues.append(
                "点评把「%s」当作%s方行棋的主语，但%s方没有这个兵种。"
                % (ch, "我" if side == own_side else "对",
                   "我" if side == own_side else "对"))
        else:
            issues.append(
                "点评把「%s」当作%s方行棋的主语，但它是%s方的子。"
                % (ch, "我" if side == own_side else "对", _cn_word(belongs)))

    # 去重（同一个错误可能被上面两轮各抓一次）
    seen = set()
    unique: List[str] = []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            unique.append(issue)
    return unique


def _piece_index(snapshot: Any):
    """``(双方的 (归属方, 兵种字) 集合, {兵种字: 归属方})``。

    只在**棋盘上真的出现过的子**里判定归属：某方根本没有的兵种（例如 AI 执黑
    却写着"我方帅"），会由"这个组合不存在"直接判错，而不是靠全局字形表猜。
    """
    pairs: set = set()
    chars: Dict[str, int] = {}
    for _label, ch, side in piece_ownership(snapshot):
        if side not in (0, 1):
            continue
        pairs.add((side, ch))
        chars.setdefault(ch, side)
    if not pairs:
        # 快照里连棋盘都没有：退化成全局字形表（仍能抓住最明显的红黑互换）
        for ch in PIECE_CHARS:
            side = piece_belongs_to(ch)
            if side is not None:
                pairs.add((side, ch))
                chars.setdefault(ch, side)
    return pairs, chars


def side_swap_error(comment: Any, snapshot: Any = None) -> List[str]:
    """判断点评是不是"整体把红黑说反了"（我方 ↔ 对方 全线互换）。

    只有在**两个不同兵种**分别被贴上了错误归属词时（我方的子被说成对方的、
    对方的子被说成我方的）才敢断定是整句说反，这时把归属词与颜色字对调一次，
    句子就完全正确了。判据刻意保守，避免把正确的话改成错的。
    """
    text = _as_text(comment)
    if not text:
        return []
    facts = _facts_view(snapshot)
    own_side = _side_index(facts.get("ai_color")) if facts else None
    if own_side is None:
        own_side = _side_index(_snapshot_view(snapshot).get("ai_color"))
    if own_side is None:
        return []
    _pairs, chars = _piece_index(snapshot)

    own_mislabelled: set = set()
    opp_mislabelled: set = set()
    # 逐次扫描"归属词 + 兵种字"，同一个兵种可能两边都出现（红黑同型子），
    # 所以必须按出现位置判定，不能按兵种归并。
    for hit in scan_side_piece(text):
        ch = hit["piece"]
        belongs = chars.get(ch)
        if belongs is None:
            continue
        side = _claim_side(hit, own_side)
        if side is None:
            continue
        if side != belongs and belongs == own_side:
            own_mislabelled.add(ch)
        elif side != belongs and belongs != own_side:
            opp_mislabelled.add(ch)
    if own_mislabelled and opp_mislabelled:
        return ["点评把「我方」和「对方」整体说反了（我方的子被说成对方的，反之亦然）"]
    return []


def _nearest_side_word(text: str, position: int, window: int = 6) -> str:
    """``position`` 之前最近出现的归属词（``"我方"`` / ``"对方"`` / ``""``）。"""
    start = max(0, position - window)
    best = -1
    word = ""
    index = start
    while index < position:
        found = _match_side_word(text, index)
        if found:
            best = index
            word = found
            index += 1
        else:
            index += 1
    return word if best >= 0 else ""


def repair_comment(comment: Any, snapshot: Any = None) -> Tuple[str, List[str]]:
    """尝试**自动改正**点评里的红黑颠倒。

    返回 ``(新点评, 措施说明)``。能安全改就改（对调归属词与颜色字）；
    改不了就返回空串，调用方应当丢弃这段点评（宁可不点评，也不能误导玩家）。
    """
    text = _as_text(comment)
    if not text:
        return "", []
    issues = check_comment(text, snapshot)
    swap = side_swap_error(text, snapshot)
    if not issues and not swap:
        return text, []
    if not swap:
        return "", issues

    def swap_words(match) -> str:
        word = match.group(0)
        if word in ("我方", "己方", "本方", "咱"):
            return "对方"
        if word in ("对方", "敌方", "对手", "对面", "人家"):
            return "我方"
        return word

    fixed = _SIDE_SPLIT_RE.sub(swap_words, text)
    fixed = re.sub(r"红(?=[%s])" % "".join(sorted(PIECE_CHARS)), "黑", fixed)
    fixed = re.sub(r"黑(?=[%s])" % "".join(sorted(PIECE_CHARS)), "红", fixed)
    notes = ["已自动把点评里的「我方/对方」与红黑对调（原文把双方说反了）"]
    return fixed, notes


def validate_comment(comment: Any, snapshot: Any = None) -> Dict[str, Any]:
    """质检 + 自动修复的总入口（给 AI 控制器调用）。

    返回 ``{"text": 可用文本, "ok": 是否原样可用, "issues": [...], "notes": [...]}``；
    无法安全修复时 ``text`` 为空串，调用方应丢弃这段点评并（可选）重问一次。
    """
    text = _as_text(comment)
    if not text:
        return {"text": "", "ok": True, "issues": [], "notes": []}
    issues = check_comment(text, snapshot)
    if not issues:
        return {"text": text, "ok": True, "issues": [], "notes": []}
    repaired, notes = repair_comment(text, snapshot)
    return {"text": repaired, "ok": False, "issues": issues, "notes": notes}


def build_messages(snapshot: Any, style: str = "explain",
                   extra_system: str = "") -> List[Dict[str, str]]:
    """构造 OpenAI 兼容的 ``messages`` 列表。非法 ``style`` 回退到 ``explain``。

    ``extra_system`` 非空时追加一条 system 消息，用于"上一次点评被质检
    判定红黑搞反了，请重写"这种定向重试（比把批评塞进 user 消息更有效）。
    """
    chosen = _as_text(style).lower()
    if chosen not in STYLES:
        chosen = DEFAULT_STYLE
    user_text = "\n".join(
        [
            _STYLE_HINTS[chosen],
            "",
            render_position_text(snapshot),
            "",
            "请按系统提示的要求，只输出一个 JSON 对象。",
        ]
    )
    messages = [
        {"role": "system", "content": POSITION_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    hint = _as_text(extra_system)
    if hint:
        messages.append({"role": "system", "content": hint})
    return messages


def build_payload(
    snapshot: Any,
    cfg: "LLMConfig",
    style: Optional[str] = None,
    extra_system: str = "",
) -> Dict[str, Any]:
    """构造完整的请求体（``--dry-run`` 打印的就是它）。"""
    chosen = _as_text(style).lower() if style else ""
    if chosen not in STYLES:
        chosen = cfg.mode if cfg.mode in STYLES else DEFAULT_STYLE
    return {
        "model": cfg.model,
        "messages": build_messages(snapshot, chosen, extra_system=extra_system),
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }


def build_repair_hint(issues: Sequence[Any]) -> str:
    """把质检问题清单变成追加给模型的 system 提醒。"""
    items = [_as_text(issue) for issue in (issues or ()) if _as_text(issue)]
    if not items:
        return ""
    bullet = "\n".join("· %s" % item for item in items[:6])
    return SIDE_ERROR_HINT.format(issues=bullet)


# ---------------------------------------------------------------------------
# 返回解析
# ---------------------------------------------------------------------------

_MOVE_PATTERNS = (
    re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*(?:->|→|—>|⇒|=>|到|至|→)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"),
    re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*(?:->|→|⇒|=>|到|至)\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"),
    re.compile(r"(-?\d+)\s*,\s*(-?\d+)\s*(?:->|→|⇒|=>|到|至)\s*(-?\d+)\s*,\s*(-?\d+)"),
)
_MOVE_CANDIDATES = (
    re.compile(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*(?:->|→|⇒|=>|到|至)\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?"),
    re.compile(r"(-?\d+)\s*,\s*(-?\d+)\s*[-—]{1,2}\s*(-?\d+)\s*,\s*(-?\d+)"),
)
_COMMENT_RE = re.compile(r'"comment"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
_COMMENT_LOOSE_RE = re.compile(r"comment[^:\n]{0,10}[:：]\s*(.{1,400}?)(?:\n|$)", re.S)

#: 项目统一记法的一手（``b3c3``），用来从**散文**里抠坐标。
#: 行号只认 0~9（与引擎一致，不存在 ``a10``）。
_NOTATION_RE = re.compile(r"[a-i][0-9][a-i][0-9]")


def _coerce_move(move: Any) -> Optional[Tuple[int, int, int, int]]:
    """把多种写法的着法规整成 4 元组 ``(fr, fc, tr, tc)``；失败返回 ``None``。"""
    if move is None:
        return None
    if isinstance(move, dict):
        one = move.get("from") or move.get("start") or move.get("src")
        two = move.get("to") or move.get("end") or move.get("dst")
        if one is not None and two is not None:
            return _coerce_move((one, two))
        for key in ("move", "text", "coord", "coordinate"):
            if move.get(key):
                return _coerce_move(move.get(key))
        return None
    if isinstance(move, str):
        text = move.strip()
        # ① 先用**项目统一的记法**解析（``b3c3``）——这是模型现在被要求写的格式，
        #    也是唯一与棋盘/棋谱/引擎一致的口径。
        #    ⚠ 这里曾经只有下面那几条**老写法**的正则（``(r,c)->(r,c)``），
        #    于是模型老老实实按提示词写了 ``"move": "c6c5"``，解析结果却是
        #    ``None`` —— 表现为"大模型启动了却从不给招法"（每步都退回本机引擎）。
        parsed = _parse_notation_move(text)
        if parsed is not None:
            return parsed
        # ② 再退到老写法（兼容历史记录/旧提示词里模型仍然回坐标对的情况）
        for pattern in _MOVE_PATTERNS:
            found = pattern.search(text)
            if found:
                return tuple(int(part) for part in found.groups())  # type: ignore[return-value]
        return None
    if isinstance(move, (list, tuple)):
        if len(move) == 2:
            first, second = move
            if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
                if len(first) >= 2 and len(second) >= 2:
                    try:
                        return (
                            int(first[0]),
                            int(first[1]),
                            int(second[0]),
                            int(second[1]),
                        )
                    except (TypeError, ValueError):
                        return None
            if isinstance(first, dict) and isinstance(second, dict):
                return _coerce_move((first, second))
            if isinstance(move[0], str) and isinstance(move[1], str):
                return _coerce_move("%s->%s" % (move[0], move[1]))
            return None
        if len(move) >= 4:
            try:
                return (int(move[0]), int(move[1]), int(move[2]), int(move[3]))
            except (TypeError, ValueError):
                return None
    return None


def _parse_notation_move(text: str) -> Optional[Tuple[int, int, int, int]]:
    """按项目统一记法解析一手（``"b3c3"``），失败返回 ``None``。

    转调 :mod:`jieqi.notation` —— **不要在这里再写一套坐标换算**，
    "两份实现"就是"哪天只改了一边"的温床（这次的 bug 正是这么来的：
    提示词换了新记法，解析器还只认老写法）。

    两种输入都收：

    * **纯坐标**（JSON 里的 ``"move": "c6c5"``）——模型实际返回的样子；
    * **散文里夹着坐标**（``"我建议走 b9a7 这一步"``）——用一个宽松的
      正则把 ``[a-i][0-9][a-i][0-9]`` 抠出来再交给 notation 解析。
      （``notation.parse_move_name`` 是**严格**的：整串必须正好 4 个字符，
      所以散文必须先抠。）
    """
    try:
        from .notation import parse_move_name

        raw = str(text or "").strip()
        if len(raw) == 4:
            got = parse_move_name(raw)
        else:
            found = _NOTATION_RE.search(raw)
            got = parse_move_name(found.group(0)) if found else None
    except Exception:      # 记法模块缺失/异常时不要影响解析（退到老写法）
        return None
    if not got:
        return None
    (fr, fc), (tr, tc) = got
    if (fr, fc) == (tr, tc):
        return None        # 原地不动不是着法
    return int(fr), int(fc), int(tr), int(tc)


def _regex_fallback_move(text: str) -> Optional[Tuple[int, int, int, int]]:
    # 先试统一记法（可能出现在散文里，例如"我建议走 b3c3"）
    parsed = _parse_notation_move(text)
    if parsed is not None:
        return parsed
    for pattern in _MOVE_CANDIDATES:
        found = pattern.search(text)
        if found:
            try:
                return tuple(int(part) for part in found.groups())  # type: ignore[return-value]
            except (TypeError, ValueError):
                continue
    return None


def _extract_object(text: str) -> Optional[str]:
    """取出最外层 ``{...}`` 子串（按括号计数，跳过字符串内的括号）。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _clean_text(text: Any) -> str:
    """去掉 Markdown 代码围栏与首尾空白。"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.replace("\ufeff", "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_loose_number(text: str) -> Optional[float]:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    try:
        return float(numbers[0])
    except (TypeError, ValueError):
        return None


def _regex_fallback_comment(text: str) -> str:
    found = _COMMENT_RE.search(text)
    if found:
        try:
            return json.loads('"%s"' % found.group(1)).strip()
        except Exception:
            return found.group(1).strip()
    loose = _COMMENT_LOOSE_RE.search(text)
    if loose:
        value = loose.group(1).strip().strip('",').strip()
        if value:
            return value
    return ""


def parse_response(text: Any) -> Dict[str, Any]:
    """宽容解析模型返回。

    返回 ``{"move": (fr, fc, tr, tc) | None, "comment": str, "confidence": float | None,
    "tags": list, "raw": text}``。任何情况下都不抛异常。
    """
    raw = text if isinstance(text, str) else ""
    result: Dict[str, Any] = {
        "move": None,
        "comment": "",
        "confidence": None,
        "tags": [],
        "raw": raw,
    }
    if not raw.strip():
        # 空输入、None、以及非字符串输入（数字/对象等）都视为"无有效回复"。
        return result
    cleaned = _clean_text(raw)
    if not cleaned:
        return result

    data: Any = None
    candidates: List[str] = []
    obj_text = _extract_object(cleaned)
    if obj_text and obj_text != cleaned:
        candidates.append(obj_text)
    candidates.append(cleaned)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            data = parsed
            break

    if isinstance(data, dict):
        result["move"] = _coerce_move(
            data.get("move", data.get("best_move", data.get("suggested_move")))
        )
        comment = data.get("comment", data.get("explanation", data.get("reason", "")))
        result["comment"] = _as_text(comment)
        confidence = data.get("confidence", data.get("score", None))
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            result["confidence"] = float(confidence)
        elif isinstance(confidence, str) and confidence.strip():
            result["confidence"] = _parse_loose_number(confidence)
        tags = data.get("tags", data.get("labels", []))
        if isinstance(tags, (list, tuple)):
            result["tags"] = [_as_text(tag) for tag in tags if _as_text(tag)]
        elif isinstance(tags, str) and tags.strip():
            result["tags"] = [
                part.strip() for part in re.split(r"[,，、;；\s]+", tags) if part.strip()
            ]
    else:
        # 非 JSON：正则兜底
        result["move"] = _regex_fallback_move(cleaned)
        result["comment"] = _regex_fallback_comment(cleaned)
        confidence = re.search(
            r'"?confidence"?\s*[:：=]\s*(-?\d+(?:\.\d+)?)', cleaned, re.I
        )
        if confidence:
            try:
                result["confidence"] = float(confidence.group(1))
            except (TypeError, ValueError):
                result["confidence"] = None
        # 模型返回纯自然语言时，把原文（截断后）当作点评，避免点评丢失。
        if not result["comment"] and len(cleaned) <= 500 and not cleaned.startswith("{"):
            result["comment"] = cleaned

    if result["move"] is None and not isinstance(data, dict):
        result["move"] = _coerce_move(cleaned)

    # 置信度钳制到 [0, 1]（模型偶尔给出 60 之类的百分数）
    confidence = result["confidence"]
    if isinstance(confidence, float):
        if confidence > 1.0 and confidence <= 100.0:
            confidence = confidence / 100.0
        result["confidence"] = min(1.0, max(0.0, confidence))

    return result


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    """一次大模型调用的结果（失败时 ``ok=False`` 且 ``error`` 为中文说明）。"""

    ok: bool = False
    text: str = ""
    parsed: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency: float = 0.0
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def move(self) -> Optional[Tuple[int, int, int, int]]:
        """解析出的着法 4 元组 ``(fr, fc, tr, tc)``，没有则为 ``None``。"""
        if not isinstance(self.parsed, dict):
            return None
        return self.parsed.get("move")

    @property
    def comment(self) -> str:
        """模型的中文点评。"""
        if not isinstance(self.parsed, dict):
            return ""
        return _as_text(self.parsed.get("comment"))

    @property
    def confidence(self) -> Optional[float]:
        if not isinstance(self.parsed, dict):
            return None
        value = self.parsed.get("confidence")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def tags(self) -> List[str]:
        if not isinstance(self.parsed, dict):
            return []
        tags = self.parsed.get("tags")
        return list(tags) if isinstance(tags, (list, tuple)) else []

    def move_text(self) -> Optional[str]:
        """着法的可读写法 ``"b3c3"``（列字母 + 行数字，红方底线是第 1 行）。"""
        move = self.move
        if not move:
            return None
        from .notation import move_name

        fr, fc, tr, tc = move
        return move_name((fr, fc), (tr, tc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "move": self.move_text(),
            "move_tuple": list(self.move) if self.move else None,
            "comment": self.comment,
            "confidence": self.confidence,
            "tags": self.tags,
            "error": self.error,
            "latency": self.latency,
            "model": self.model,
            "usage": self.usage,
            "text": self.text,
        }


# ---------------------------------------------------------------------------
# 顾问主体
# ---------------------------------------------------------------------------


class LLMAdvisor:
    """可选的大模型顾问。默认关闭、永不抛异常、线程安全地记录历史。"""

    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg: LLMConfig = cfg if cfg is not None else load_config()
        self.last_result: Optional[LLMResult] = None
        self.history: List[LLMResult] = []
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []

    # -- 状态 -------------------------------------------------------------
    def available(self) -> bool:
        """是否已启用且配置了 API Key。"""
        return bool(self.cfg and self.cfg.enabled and self.cfg.api_key)

    def _record(self, result: LLMResult) -> LLMResult:
        with self._lock:
            self.last_result = result
            self.history.append(result)
            if len(self.history) > HISTORY_LIMIT:
                del self.history[: len(self.history) - HISTORY_LIMIT]
        return result

    # -- 请求构造 ---------------------------------------------------------
    def endpoint(self) -> str:
        """拼接 ``.../chat/completions`` 完整地址。"""
        base = (self.cfg.base_url or "").strip().rstrip("/")
        if not base:
            base = "https://api.openai.com/v1"
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def build_payload(self, snapshot: Any, style: Optional[str] = None,
                      extra_system: str = "") -> Dict[str, Any]:
        """构造请求体（``--dry-run`` 用的就是它）。"""
        return build_payload(snapshot, self.cfg, style, extra_system=extra_system)

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "jieqi-llm-advisor/1.0 (+stdlib urllib)",
        }
        extra = self.cfg.extra_headers or {}
        for key, value in extra.items():
            key_text = str(key).strip()
            if key_text:
                headers[key_text] = "" if value is None else str(value)
        if self.cfg.api_key:
            headers.setdefault("Authorization", "Bearer %s" % self.cfg.api_key)
        return headers

    # -- 同步调用 ---------------------------------------------------------
    def suggest(self, snapshot: Any, style: Optional[str] = None,
                extra_system: str = "") -> LLMResult:
        """阻塞调用大模型；任何失败都返回 ``ok=False`` 的 :class:`LLMResult`。

        ``extra_system`` 会作为**追加的 system 消息**塞进请求，用于定向重试
        （例如"上次点评把红黑说反了，请重写"）。
        """
        started = time.monotonic()

        def elapsed() -> float:
            return round(time.monotonic() - started, 4)

        if self.cfg is None:  # pragma: no cover - 防御
            return self._record(
                LLMResult(ok=False, error="内部错误：缺少配置对象。", latency=elapsed())
            )
        if not self.cfg.api_key:
            return self._record(
                LLMResult(
                    ok=False,
                    error="未配置 API Key（llm.api_key 或环境变量 %s），无法调用大模型。"
                    % self.cfg.api_key_env,
                    model=self.cfg.model,
                    latency=elapsed(),
                )
            )

        url = self.endpoint()
        payload = self.build_payload(snapshot, style, extra_system=extra_system)
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            return self._record(
                LLMResult(
                    ok=False,
                    error="请求体序列化失败：%s" % exc,
                    model=self.cfg.model,
                    latency=elapsed(),
                )
            )

        timeout = float(self.cfg.timeout) if self.cfg.timeout else 20.0
        if timeout < MIN_TIMEOUT:
            timeout = MIN_TIMEOUT

        request = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", None) or getattr(response, "code", 200)
                raw = response.read(MAX_BODY_BYTES)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read(4096).decode("utf-8", "replace").strip()
            except Exception:
                detail = ""
            if len(detail) > 400:
                detail = detail[:400] + "…"
            return self._record(
                LLMResult(
                    ok=False,
                    error="接口返回 HTTP %s：%s" % (exc.code, detail or exc.reason or "无响应体"),
                    model=self.cfg.model,
                    latency=elapsed(),
                    text=detail,
                )
            )
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, socket.timeout):
                message = "连接大模型超时（%.1f 秒）：%s" % (timeout, url)
            else:
                message = "无法连接大模型（%s）：%s" % (reason, url)
            return self._record(
                LLMResult(ok=False, error=message, model=self.cfg.model, latency=elapsed())
            )
        except (socket.timeout, TimeoutError):
            return self._record(
                LLMResult(
                    ok=False,
                    error="调用大模型超时（%.1f 秒）：%s" % (timeout, url),
                    model=self.cfg.model,
                    latency=elapsed(),
                )
            )
        except Exception as exc:  # 兜底：绝不让异常逃出 suggest()
            return self._record(
                LLMResult(
                    ok=False,
                    error="调用大模型失败（%s：%s）。" % (type(exc).__name__, exc),
                    model=self.cfg.model,
                    latency=elapsed(),
                )
            )

        decode_error = ""
        try:
            text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        except Exception as exc:  # pragma: no cover - decode 已给 errors
            text = ""
            decode_error = str(exc)

        if decode_error:
            return self._record(
                LLMResult(
                    ok=False,
                    error="响应体解码失败：%s" % decode_error,
                    model=self.cfg.model,
                    latency=elapsed(),
                )
            )

        if not text.strip():
            return self._record(
                LLMResult(
                    ok=False,
                    error="接口返回空响应体（HTTP %s）。" % status,
                    model=self.cfg.model,
                    latency=elapsed(),
                )
            )

        try:
            envelope = json.loads(text)
        except Exception as exc:
            return self._record(
                LLMResult(
                    ok=False,
                    error="接口返回的不是合法 JSON（%s）：%s" % (exc, text[:200]),
                    text=text,
                    model=self.cfg.model,
                    latency=elapsed(),
                )
            )

        if not isinstance(envelope, dict):
            return self._record(
                LLMResult(
                    ok=False,
                    error="接口返回的 JSON 顶层不是对象。",
                    text=text,
                    model=self.cfg.model,
                    latency=elapsed(),
                )
            )

        if isinstance(envelope.get("error"), dict):
            detail = _as_text(envelope["error"].get("message")) or json.dumps(
                envelope["error"], ensure_ascii=False
            )
            return self._record(
                LLMResult(
                    ok=False,
                    error="接口报错：%s" % detail,
                    text=text,
                    model=self.cfg.model,
                    latency=elapsed(),
                )
            )

        content, content_error = _extract_content(envelope)
        model_name = _as_text(envelope.get("model")) or self.cfg.model
        usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}

        if content_error:
            return self._record(
                LLMResult(
                    ok=False,
                    error=content_error,
                    text=text,
                    model=model_name,
                    usage=usage,
                    latency=elapsed(),
                )
            )

        parsed = parse_response(content)
        return self._record(
            LLMResult(
                ok=True,
                text=content,
                parsed=parsed,
                error="",
                latency=elapsed(),
                model=model_name,
                usage=usage,
            )
        )

    # -- 异步调用 ---------------------------------------------------------
    def suggest_async(
        self,
        snapshot: Any,
        callback: Optional[Callable[[LLMResult], Any]] = None,
        style: Optional[str] = None,
        validate: bool = False,
    ) -> threading.Thread:
        """在守护线程里调用 :meth:`suggest`，立刻返回已启动的线程。

        ``validate=True`` 时走 :meth:`suggest_checked`：点评先过"归属质检"，
        发现红黑搞反会自动改判或定向重问一次，**回调仍需立刻返回，不会更慢**。
        """

        def worker() -> None:
            result = (self.suggest_checked(snapshot, style=style)
                      if validate else self.suggest(snapshot, style))
            if callback is not None:
                try:
                    callback(result)
                except Exception:
                    # 回调是调用方代码，出错也不能影响顾问自身。
                    pass

        thread = threading.Thread(
            target=worker, name="jieqi-llm-advisor", daemon=True
        )
        with self._lock:
            self._threads.append(thread)
            if len(self._threads) > HISTORY_LIMIT:
                self._threads = self._threads[-HISTORY_LIMIT:]
        thread.start()
        return thread

    # -- 质检 / 定向重试 ---------------------------------------------------
    def validate_result(self, result: Optional[LLMResult],
                        snapshot: Any = None) -> Dict[str, Any]:
        """对一次结果做归属质检；``result`` 为空或失败时按"没问题"处理。"""
        comment = result.comment if isinstance(result, LLMResult) else ""
        if not comment:
            return {"text": "", "ok": True, "issues": [], "notes": []}
        return validate_comment(comment, snapshot)

    def comment_issues(self, comment: Any, snapshot: Any = None) -> List[str]:
        """点评里的归属问题清单（空列表 = 干净）。"""
        return check_comment(comment, snapshot)

    def suggest_checked(
        self,
        snapshot: Any,
        style: Optional[str] = None,
        max_retries: int = 1,
    ) -> LLMResult:
        """问一次 + 质检；发现"红黑搞反"时自动改判，必要时定向重问一次。

        无论走哪条分支，返回的都是同一个 :class:`LLMResult` 对象（可能已被
        ``parsed['comment']`` 就地修正），所以调用方照常读 ``comment`` 即可。
        """
        result = self.suggest(snapshot, style)
        if not isinstance(result, LLMResult) or not result.ok:
            return result
        if not getattr(self.cfg, "comment_validation", True):
            return result

        attempt = 0
        while True:
            issues = check_comment(result.comment, snapshot)
            if not issues:
                return result
            repaired, notes = repair_comment(result.comment, snapshot)
            if repaired and repaired != result.comment:
                # 能安全改判（整句红黑说反）：直接采用修正后的文字，不多花请求。
                self._patch_comment(result, repaired, notes)
                return result
            if attempt >= max(0, int(max_retries)):
                # 改不了也问不出干净的：把点评清空，宁可不说话也不误导玩家。
                self._patch_comment(result, "", ["模型点评与局面归属矛盾，已丢弃该点评"])
                return result
            attempt += 1
            hint = build_repair_hint(issues)
            try:
                retry = self.suggest(snapshot, style, extra_system=hint)
            except TypeError:
                # 兼容没有 extra_system 参数的旧实现/测试替身：退回不带提示的问法。
                retry = self.suggest(snapshot, style)
            if not isinstance(retry, LLMResult) or not retry.ok:
                self._patch_comment(result, "", ["重问失败，已丢弃有问题的点评"])
                return result
            result = retry

    @staticmethod
    def _patch_comment(result: LLMResult, text: str, notes: Sequence[Any]) -> None:
        """就地改写结果里的点评，并把质检过程记进 ``error`` 旁边（不覆盖 error）。"""
        if not isinstance(result.parsed, dict):
            result.parsed = {}
        original = result.parsed.get("comment")
        result.parsed["comment"] = text
        if original != text:
            result.parsed["comment_original"] = original
        kept = [str(note) for note in (notes or ()) if str(note)]
        if kept:
            result.parsed["comment_notes"] = kept


def _extract_content(envelope: Dict[str, Any]) -> Tuple[str, str]:
    """从 OpenAI 兼容响应里取出 ``choices[0].message.content``。"""
    choices = envelope.get("choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return "", "接口返回缺少 choices 字段，无法解析。"
    first = choices[0]
    if not isinstance(first, dict):
        return "", "接口返回的 choices[0] 格式异常。"
    message = first.get("message")
    content: Any = None
    if isinstance(message, dict):
        content = message.get("content")
        if content is None:
            content = message.get("reasoning_content")
    if content is None:
        content = first.get("text")
    if isinstance(content, (list, tuple)):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(_as_text(item.get("text") or item.get("content")))
            else:
                chunks.append(_as_text(item))
        content = "".join(chunks)
    if content is None:
        return "", "接口返回的 message.content 为空。"
    return str(content), ""


# ---------------------------------------------------------------------------
# 与引擎评分的融合
# ---------------------------------------------------------------------------


def _candidate_key(item: Any) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """把候选着法转成 ``((fr, fc), (tr, tc))`` 键。"""
    if not isinstance(item, dict):
        return None
    start = _coerce_move((item.get("from"), item.get("to")))
    if start is None:
        start = _coerce_move(item.get("move"))
    if start is None:
        return None
    return ((start[0], start[1]), (start[2], start[3]))


def _coerce_confidence(value: Any, default: float = 0.5) -> float:
    number = _as_float(value, None)
    if number is None:
        return default
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    return min(1.0, max(0.0, number))


def blend_bonus(
    result: Any,
    candidates: Any,
    weight: float,
    max_bonus: float = BLEND_MAX_BONUS_CP,
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
    """把大模型推荐的着法折算成候选着法的加分（厘兵）。

    * 返回的 dict 以 ``((fr, fc), (tr, tc))`` 为键，覆盖 ``candidates`` 里的每一项；
      不在候选中的一律 0.0。
    * 加分 = ``weight × confidence × max_bonus``，上限 ``max_bonus``。
    * **weight 为 0（或负数、非法值）时全部为 0.0**，也就是"顾问只看不插手"，
      完全不影响引擎搜索；只有显式给出正权重时才会产生微小倾斜。
    * 参数 ``result`` 可以是 :class:`LLMResult`，也可以是 ``parse_response`` 的 dict
      或形如 ``(fr, fc, tr, tc)`` / ``"1,2->3,4"`` 的着法本身；非法输入全部按 0.0 处理。
    """
    table: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}
    keys: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if isinstance(candidates, (list, tuple)):
        for item in candidates:
            key = _candidate_key(item)
            if key is not None:
                keys.append(key)
    for key in keys:
        table[key] = 0.0

    weight_value = _as_float(weight, 0.0) or 0.0
    if weight_value <= 0 or not keys:
        return table

    # 取出模型推荐的着法与置信度
    confidence: Any = None
    move: Any = None
    if isinstance(result, LLMResult):
        move = result.move
        confidence = result.confidence
    elif isinstance(result, dict):
        move = _coerce_move(result.get("move"))
        confidence = result.get("confidence")
    else:
        move = _coerce_move(result)
    if move is None:
        return table

    confidence_value = _coerce_confidence(confidence, 0.5)
    key = ((move[0], move[1]), (move[2], move[3]))
    if key not in table:
        return table

    bonus = min(float(max_bonus), weight_value * confidence_value * float(max_bonus))
    table[key] = float(bonus)
    return table
