# -*- coding: utf-8 -*-
"""打印真实棋局上的大模型提示词（不发网络请求），用来人工检查"归属"是否清楚。

用法::

    python _bench/preview_llm_prompt.py            # 走 12 步后的局面
    python _bench/preview_llm_prompt.py 30         # 走 30 步
    python _bench/preview_llm_prompt.py 20 --ply 8 # 浏览到第 8 手

输出顺序：系统提示词 → 质检规则自检 → 事实表 → 完整 user 消息。
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
GAME = _paths.GAME
STAGE = _paths.STAGE
for path in (str(STAGE), str(GAME)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((1, 1))

from jieqi import llm as llm_mod  # noqa: E402
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    build_llm_snapshot, gui_board_to_engine, gui_color_to_side, legal_moves,
)


def play(plies: int, seed: int = 11):
    from game_state import GameState

    random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.enable_ai("black")
    rng = random.Random(seed)
    for k in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.set_ply_comment(gs.ply, f"第{k + 1}手：我在抢中路。", f"AI 评估：+{k * 10}")
        gs.current_player = "black" if side == 0 else "red"
    return gs


def main(argv) -> int:
    plies = int(argv[0]) if argv and not argv[0].startswith("-") else 12
    ply = None
    if "--ply" in argv:
        ply = int(argv[argv.index("--ply") + 1])
    out_path = None
    if "--out" in argv:
        out_path = argv[argv.index("--out") + 1]
    lines = []

    def emit(text: str = "") -> None:
        lines.append(text)

    gs = play(plies)
    if ply is not None:
        gs.goto_ply(ply)
    turn = gui_color_to_side(gs.current_player)
    # 引擎结果必须针对"轮到走棋的一方"来算，否则快照里的归属注解会张冠李戴。
    engine = Engine("easy", time_limit=0.3, samples=1, seed=3)
    result = engine.analyse(gui_board_to_engine(gs.board, turn=turn))
    snap = build_llm_snapshot(gs, result, ai_color=turn)

    emit("=" * 78)
    emit("事实表（本地判定，模型只许照抄）")
    emit("=" * 78)
    emit(snap.get("facts_text") or "(空)")
    emit()
    emit("=" * 78)
    emit("完整 user 消息")
    emit("=" * 78)
    emit(llm_mod.render_position_text(snap))

    emit()
    emit("=" * 78)
    emit("质检自检（故意写错的点评应当被抓出来）")
    emit("=" * 78)
    samples = [
        "我方车从中路杀出，吃掉对方的兵。",
        "我方帅已经安全了。",
        "黑方車正吃我的卒，需要小心。",
        "这步棋稳住中路。",
        "我方的卒顶上去，对方的车被牵制。",
    ]
    for text in samples:
        issues = llm_mod.check_comment(text, snap)
        fixed, notes = llm_mod.repair_comment(text, snap)
        emit("- 原文：%s" % text)
        emit("  问题：%s" % ("；".join(issues) if issues else "无"))
        emit("  修复：%s %s" % (fixed or "(丢弃)", "；".join(notes) if notes else ""))

    text = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("已写入 %s" % out_path)
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
