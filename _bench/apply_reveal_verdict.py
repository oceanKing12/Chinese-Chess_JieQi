# -*- coding: utf-8 -*-
"""把 reveal_weights 清空回关闭，并把标定结论写进配置注释。

标定结论（_bench/tune_reveal.py）：
  seed 8800 批测： base15/phase20 -> 0.818 [0.52, 0.95]（看着很好）
  换新局集 seed 13570 打 48 局： 0.433 [0.27, 0.61]（没复现）
  同批 base15/phase20/unknown30：0.429 [0.28, 0.59]
两组都跨 0.5 -> 判定为"小样本过拟合"，按计划清空，不留在配置里。
"""

import json
import sys
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
GAME = _paths.GAME

#: 两份配置都要清：**开发树**里的模板 + **部署树**里正在用的那份。
#: 用 ``_paths`` 认出来的 ``GAME`` 当"当前这棵树"，另一棵按老路径兜底
#: （哪份不存在就跳过，不再写死两个绝对路径）。
PATHS = [str(_paths.GAME / "ai_config.json")]
_OTHER = (Path(r"D:\py\揭棋\stage\Game") if _paths.STAGE.name != "stage"
          else Path(r"D:\py\Chinese Chess\Game"))
if (_OTHER / "ai_config.json").is_file():
    PATHS.append(str(_OTHER / "ai_config.json"))

NOTE_MAIN = (
    "揭棋特有的翻子搜索加成（见 jieqi/reveal.py，口径对齐 AB-JChess）。"
    "当前为关闭（空 dict）——这是标定的结论，不是没做。"
    "seed 8800 上 base15/phase20 拿到 0.818 [0.52,0.95]，但换到新局集 seed 13570 "
    "打 48 局只有 0.433 [0.27,0.61]，没有复现（小样本过拟合）；"
    "base15/phase20/unknown30 同批 0.429 [0.28,0.59]。两组都跨 0.5，"
    "都没有显著好过关闭，所以保持空：宁可不要这个加成，也不留没验证过的权重。"
    "将来要再试，先用 --batch 在新局集上量，按「区间下界大于 0.5」判定。")

NOTE_QUIET = (
    "只作用于不吃子的翻子着法。保持为空：quiet_safety 有已证实的误判"
    "（test_reveal.py 里两条 expectedFailure 标着），修好之前不能用。")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for path in PATHS:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        ai = data.setdefault("ai", {})
        ai["reveal_weights"] = {}
        ai["reveal_quiet_weights"] = {}
        ai["_reveal_weights"] = NOTE_MAIN
        ai["_reveal_quiet_weights"] = NOTE_QUIET
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        print("已清空 reveal_weights：%s -> %s" % (path, ai["reveal_weights"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
