# -*- coding: utf-8 -*-
"""验证：**带着老配置**（里面还留着已移除的外部引擎字段）游戏能正常起来，
而且大模型那一档仍然能出招。

这是"完全去掉外部引擎"最容易翻车的地方：配置文件是**受保护的**，
重新部署永远不会覆盖玩家的那一份 —— 所以线上那个 json 里一定还留着
``external_verbose`` 之类的字段，代码必须能忽略它们。
"""
from __future__ import annotations

import os
import sys

import _paths  # noqa: E402

_paths.add_to_path()
sys.stdout.reconfigure(encoding="utf-8")

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((1, 1))

from jieqi import engines as _engines  # noqa: E402
from jieqi import settings as _settings  # noqa: E402
from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402


def main() -> int:
    settings = _settings.ai_settings()
    print("=== 1) 配置读得进来（老字段被忽略）===")
    for key in ("move_source", "side_engines", "player_side",
                "abjchess_dir", "external_verbose", "external_verify"):
        print("  %-18s = %r" % (key, settings.get(key, "(未设置)")))
    print("")

    from game_state import GameState
    import AI as ai_module

    gs = GameState(None)
    gs.show_side_selection = False
    gs.apply_engine_settings(settings)
    print("=== 2) 对局方式 ===")
    print("  side_engines = %r" % (gs.side_engines,))
    print("  合法档位     = %r" % (_engines.SOURCES,))
    for side in _engines.SIDES:
        source = gs.side_engine(side)
        assert source in _engines.SOURCES, \
            "档位 %r 不在合法清单里（老配置没被归一）" % source
        print("  %s 方 = %s（%s）"
              % ("红" if side == "red" else "黑", source,
                 _engines.LABELS[source]))
    print("")

    print("=== 3) 大模型那一档还能出招吗（真的请求）===")
    gs.set_player_side("red")
    gs.set_side_engine("black", "llm")
    controller = ai_module.AdvancedAIController(gs)
    controller.settings["emit_decision_tree"] = False
    print("  advisor = %s" % ("已装配" if controller.advisor else "**没有**"))
    print("  status  = %s" % (controller.llm_status or "")[:110])
    if controller.advisor is None:
        print("  （没配 key 时这是正常结果，不算失败）")
        return 0
    side = gui_color_to_side(gs.current_player)
    result = controller.analyse_llm(side)
    move = None if result is None else result.move
    print("  source=%s move=%s" % (controller.last_source, move))
    for note in (result.notes if result else []):
        print("  备注：%s" % note)
    print("")
    if controller.last_source == "llm":
        print("结论：老配置不影响大模型出招 ✅")
    else:
        print("结论：大模型这一步退回了本机引擎（看上面备注的原因）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
