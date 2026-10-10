# -*- coding: utf-8 -*-
"""对局方式（每方由谁出招）的**单一真相来源**。

需求原话："支持切换自己的引擎、对方的引擎、大模型三种对局方式，
切换到哪种就完全应用那个引擎的招法。"

所以这里定义的是**每方各自一个档位**（红方一个、黑方一个），共三档：

====================  ==================================================
``human``             人工输入：这一方由玩家自己点棋子（默认）
``python``            本机引擎：**完全采用**它的招法（默认）
``llm``               大模型：完全采用它推荐的招法
====================  ==================================================

三档的"定义 + 可用性 + 文案"都集中在这里，因为启动页（画 chip、画灰掉的原因）、
主循环（挑走哪条分支）、棋谱 meta（记档位）和测试都要用。分散写就会出现
"界面上说可用、点下去却没反应"这种自相矛盾（真实踩过：按钮的"选中"和
"能不能点"曾经混在同一个字段里）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

__all__ = [
    "HUMAN", "PYTHON", "LLM",
    "SIDES", "SOURCES", "CHIP_LABELS", "LABELS",
    "DEFAULT_FOR_SIDE", "normalize", "available", "reason", "describe",
    "chips_for", "cycle",
]

HUMAN = "human"
PYTHON = "python"
LLM = "llm"

#: 每方一个档位；顺序就是启动页上 chip 的顺序（人工在最左，最容易点）
SIDES: Tuple[str, ...] = ("red", "black")
SOURCES: Tuple[str, ...] = (HUMAN, PYTHON, LLM)

#: chip 上的短文案（启动页那几枚按钮）
CHIP_LABELS: Dict[str, str] = {
    HUMAN: "人工输入",
    PYTHON: "本机引擎",
    LLM: "大模型",
}

#: 完整名称（状态行、棋谱 meta、日志）
LABELS: Dict[str, str] = {
    HUMAN: "人工输入",
    PYTHON: "本机引擎",
    LLM: "大模型",
}

#: 新开一局时每方的默认档位：红方人工（你自己下）、黑方本机引擎。
#: 这就是原来的"人机对战"，行为与改动前一致。
DEFAULT_FOR_SIDE: Dict[str, str] = {"red": HUMAN, "black": PYTHON}


def normalize(source: Optional[str], *, fallback: str = HUMAN) -> str:
    """把任意输入归一成一个合法档位（认不出来就退回 ``fallback``）。

    注意**不检查可用性**：可用性由 :func:`available` 单独回答。
    把"规范化"和"能不能用"混在一起，会让"大模型模块没装"这种**临时**状态
    把配置里写好的档位永久改掉（下次装好就回不来了）。

    ⚠ 历史档位 ``abjchess`` / ``abjchess_then_python``（外部引擎）已经**整体移除**，
    配置里如果还留着会被归一到 ``python`` —— 这是刻意的：老配置文件不该让游戏
    起不来，但也不该悄悄变成别的语义。
    """
    text = str(source or "").strip().lower()
    if text in SOURCES:
        return text
    # 兼容几个常见别名；老的外部引擎写法一律折算回本机引擎
    alias = {"manual": HUMAN, "person": HUMAN, "engine": PYTHON,
             "local": PYTHON, "self": PYTHON, "model": LLM,
             "gpt": LLM, "bigmodel": LLM,
             "abj": PYTHON, "abjchess": PYTHON,
             "abjchess_then_python": PYTHON}
    return alias.get(text, fallback)


def available(source: str) -> bool:
    """这一档现在能不能真的用。

    * ``human`` / ``python``：永远可用；
    * ``llm``：**模块能导入**就认为可用。配没配 API Key 不在这里判定——
      换 key 需要重启，而启动页每次都会重读配置；真正的失败原因是
      "点了没反应"更糟，所以可用性只挡"根本不可能"的情况，
      key 的问题交给启动页下面那行说明文字（``describe``）说清楚。
    """
    source = normalize(source)
    if source == LLM:
        try:
            from . import llm as _llm  # noqa: F401

            return True
        except Exception:
            return False
    return True


def reason(source: str) -> str:
    """这一档为什么不可用（可用时返回空串）。给启动页画那行小字用。"""
    source = normalize(source)
    if available(source):
        return ""
    if source == LLM:
        return "大模型模块不可用（jieqi.llm 导入失败）"
    return ""


def describe(source: str, *, difficulty: str = "normal",
             note: str = "") -> str:
    """这一档做什么的一行中文说明（启动页 chip 下面那行）。

    ``note`` 用来插入运行时的补充信息（模型名、为什么灰掉…）。
    """
    source = normalize(source)
    if source == HUMAN:
        return "由你自己落子"
    if source == PYTHON:
        return "完全采用本机引擎的招法；难度「%s」" % (difficulty or "普通")
    if source == LLM:
        if not available(source):
            return reason(source)
        return ("完全采用大模型推荐的招法；%s"
                % (note or "每一步会同步等模型回复"))
    return ""


def chips_for(side: str) -> List[Tuple[str, str, bool, str]]:
    """某一方的几枚 chip：``[(档位, 文案, 是否可用, 不可用原因), …]``。"""
    side = side if side in SIDES else "red"
    out = []
    for source in SOURCES:
        out.append((source, CHIP_LABELS[source], available(source),
                    reason(source)))
    return out


def cycle(source: str, step: int = 1) -> str:
    """把档位沿 :data:`SOURCES` 循环移动（中局用 ``E``/``T`` 切档位）。

    **只落在可用的档位上**：某一档不可用时按 ``E`` 不该停在一个
    "灰掉、点了没反应"的档位上（那就是"按了没效果"的经典来源）。
    """
    usable = [s for s in SOURCES if available(s)]
    if not usable:
        return normalize(source)
    current = normalize(source)
    if current not in usable:
        current = usable[0]
    index = usable.index(current)
    return usable[(index + int(step)) % len(usable)]

