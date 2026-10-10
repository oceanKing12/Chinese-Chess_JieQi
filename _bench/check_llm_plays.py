# -*- coding: utf-8 -*-
"""验证「大模型」这一档对局方式在**真实配置**下能给出招法。

用户的疑问是"我启动大模型了，为什么没有给出招法"。可能的原因有三个，
这个脚本把三件事一起验掉：

1. **只开了点评、没选"大模型"这一档** —— 点评（``llm.enabled``）只让模型说话，
   要它**出招**必须把某一方的对局方式设成 ``llm``（启动页那一行 chip）；
2. **请求的 mode 不对** —— 点评用 ``explain``，出招要用 ``suggest``；
3. **着法不合法/解析不出来** —— 这时会退回本机引擎并留下说明。

⚠ 这个脚本会**真的调用大模型**（走用户的 api_key）。默认只问 1 手，加 ``--plies N``
可以多问几手。
"""
from __future__ import annotations

import argparse
import os
import sys

#: 路径探测：开发树与部署树都能跑（与其它 _bench 脚本同一套）
import _paths  # noqa: E402

_paths.add_to_path()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.stdout.reconfigure(encoding="utf-8")

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((1, 1))

from jieqi import engines as _engines  # noqa: E402
from jieqi import settings as _settings  # noqa: E402
from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402
from jieqi.llm import load_config  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="验「大模型」这一档能不能出招")
    parser.add_argument("--plies", type=int, default=1)
    parser.add_argument("--side", default="black", choices=("red", "black"))
    args = parser.parse_args(argv)

    print("=== 1) 配置检查 ===")
    cfg = load_config()
    print("  llm.enabled   = %s" % getattr(cfg, "enabled", None))
    print("  llm.mode      = %r（点评用 explain，出招要用 suggest）"
          % getattr(cfg, "mode", None))
    print("  llm.model     = %r" % getattr(cfg, "model", None))
    key = getattr(cfg, "api_key", "") or ""
    print("  llm.api_key   = %s" % ("已填（%d 字符）" % len(key) if key else "**空**"))
    settings = _settings.ai_settings()
    print("  ai.side_engines = %r" % settings.get("side_engines"))
    print("  ai.player_side  = %r" % settings.get("player_side"))
    print("")
    print("  说明：`llm.enabled=true` 只让模型**点评**；要它**出招**，")
    print("        必须把某一方的对局方式设成 `llm`（启动页那一行 chip，")
    print("        或 ai_config.json 的 ai.side_engines）。")
    print("")

    from game_state import GameState
    import AI as ai_module

    gs = GameState(None)
    gs.show_side_selection = False
    gs.set_player_side("red" if args.side == "black" else "black")
    gs.set_side_engine(args.side, "llm")
    controller = ai_module.AdvancedAIController(gs)
    controller.settings["emit_decision_tree"] = False
    print("=== 2) 装配结果 ===")
    print("  llm_enabled = %s" % controller.llm_enabled)
    print("  advisor     = %s" % ("已装配" if controller.advisor else "**没有**"))
    print("  模型        = %r" % getattr(controller, "llm_model_label", ""))
    print("  状态        = %s" % getattr(controller, "llm_status", ""))
    if controller.advisor is None:
        print("")
        print("!! 顾问没装配起来：上面那行状态就是原因（多半是 api_key 为空）。")
        return 2
    print("")
    print("=== 3) 让它出招（真的请求模型）===")
    for i in range(max(1, args.plies)):
        side = gui_color_to_side(gs.current_player)
        ours = sorted(legal_moves(gs.board, side))
        if not ours:
            print("  本机无合法着法，停")
            break
        result = controller.analyse_llm(side)
        mv = None if result is None else result.move
        print("  第 %d 手（%s方）：source=%s move=%s"
              % (i + 1, "红" if side == 0 else "黑",
                 controller.last_source,
                 "（拿不到）" if mv is None else str(mv)))
        if result is not None:
            for note in result.notes:
                print("      备注：%s" % note)
        if mv is None:
            break
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.current_player = "black" if side == 0 else "red"
    print("")
    print("=== 结论 ===")
    print("  last_source = %s（llm = 采用了大模型给的招法；"
          "python = 退回了本机引擎）" % controller.last_source)
    print("  last_comment = %s" % (controller.last_comment or "")[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
