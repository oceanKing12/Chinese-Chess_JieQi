"""读取 ``Game/ai_config.json`` 里的 AI 设置（难度 / 思考时间 / 采样数 …）。

配置文件可以不存在，此时使用默认值。查找顺序：
1. 环境变量 ``JIEQI_AI_CONFIG``；
2. ``Game/ai_config.json``（与 ``config.py`` 同目录）；
3. ``jieqi/ai_config.json``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULT_AI_SETTINGS: Dict[str, Any] = {
    #: easy / normal / hard / expert
    "difficulty": "normal",
    #: 每步思考时间（秒）；None 表示用难度预设
    "time_limit": None,
    #: 定局化采样个数；None 表示用难度预设
    "samples": None,
    #: 最大搜索深度；None 表示用难度预设（通常 64 = 不限）
    "max_depth": None,
    #: 随机种子；None 表示每步随机
    "seed": None,
    #: 是否写出 decision_tree.json（tree.html / 1.py 可视化用）
    "emit_decision_tree": True,
    "decision_tree_path": "decision_tree.json",
    #: 是否把搜索信息打印到控制台
    "verbose": False,
    #: 是否使用被吃暗子的真实身份。
    #: 通行规则里被吃的暗子仍然未知（双方都不得翻看），所以默认 False，
    #: 保证 AI 与人类信息对称；True 只会让 AI 占便宜，请在明确知道后果时开启。
    "use_hidden_captures": False,
    #: 是否直接采用大模型的招法。
    #:
    #: ⚠ **现在这个开关被"对局方式"取代了**：把某一方的档位选成
    #: ``llm``（见 :mod:`jieqi.engines`）就是"完全采用它的招法"，
    #: 而且是**按方**生效的。这里保留字段只是为了兼容老配置，
    #: 在"某一方=llm"时它不再是必需的。
    "apply_llm_move": False,
    #: 每方各由谁出招（对局方式）：``human`` / ``python`` / ``llm``。
    #: **这是新的首选写法**；没写时按下面的 ``move_source`` 折算，保证老配置能用。
    "side_engines": None,
    #: 玩家执哪一方（``red`` 默认 / ``black``）。
    #: 启动页上**没有**阵营选项（用户要求去掉），所以换边就靠这一项
    #: （或者中局按 `C` 一键换边）。人机对战时另一方由 ``side_engines`` 决定。
    "player_side": None,
    #: 覆盖评估权重（见 jieqi/consts.py 的 EVAL_WEIGHTS）
    "weights": {},
    #: 谁出招：``python``（本机引擎，**默认**）/ ``llm``（大模型）。
    #:
    #: **首选写法是上面的 ``side_engines``**（每方一个档位）；这里保留是为了
    #: 老配置能用：没写 ``side_engines`` 时，``move_source`` 折算成
    #: "黑方用它、红方人工"。
    #:
    #: ⚠ 老配置里如果还写着 ``abjchess`` / ``abjchess_then_python``
    #: （外部引擎，已**整体移除**），会被 :func:`jieqi.engines.normalize`
    #: 折算回 ``python`` —— 老配置不该让游戏起不来。
    "move_source": "python",
    #: 揭棋特有的翻子搜索加成（见 jieqi/reveal.py）。
    #: 空 dict = 关闭（默认，行为与改动前逐位一致）。用 _bench/tune_reveal.py 标定。
    "reveal_weights": {},
    "reveal_quiet_weights": {},
}


def _candidates() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("JIEQI_AI_CONFIG")
    if env:
        out.append(Path(env).expanduser())
    here = Path(__file__).resolve()
    game_dir = here.parent.parent
    out.append(game_dir / "ai_config.json")
    out.append(here.parent / "ai_config.json")
    return out


def config_path() -> Path:
    for p in _candidates():
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return _candidates()[1]


def load_raw_config(path: Path | None = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else config_path()
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def ai_settings(path: Path | None = None) -> Dict[str, Any]:
    """返回合并后的 AI 设置（文件里的 ``ai`` 段覆盖默认值）。"""
    settings = dict(DEFAULT_AI_SETTINGS)
    raw = load_raw_config(path)
    section = raw.get("ai")
    if isinstance(section, dict):
        for key, value in section.items():
            if key in settings and value is not None:
                settings[key] = value
            elif key == "weights" and isinstance(value, dict):
                settings["weights"] = value
        # 允许文件里直接写顶层同名字段
    for key in ("difficulty", "time_limit"):
        if key in raw and raw[key] is not None and not isinstance(raw.get("ai"), dict):
            settings[key] = raw[key]
    if not isinstance(settings.get("weights"), dict):
        settings["weights"] = {}
    # 权重类字段允许直接写 dict（上面的通用分支只在 key 已存在时才复制）
    for key in ("reveal_weights", "reveal_quiet_weights"):
        value = (section or {}).get(key)
        if isinstance(value, dict) and value:
            settings[key] = value
    # 每方的对局方式：写在 ``ai.side_engines`` 里（``{"red": "human", ...}``）。
    # 值可能是 None（没配），此时保持 None，由调用方按 ``move_source`` 折算
    # ——**不要在这里塞默认值**，否则"老配置只有 move_source"这条兼容路径
    # 会被覆盖掉（玩家配的档位就不生效了）。
    value = (section or {}).get("side_engines")
    if isinstance(value, dict) and value:
        settings["side_engines"] = {
            str(k): str(v) for k, v in value.items() if v}
    # 玩家执哪一方（``ai.player_side``）：启动页没有阵营选项，靠它换边
    value = (section or {}).get("player_side")
    if value is not None:
        settings["player_side"] = str(value).strip().lower() or None
    return settings
