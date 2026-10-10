# -*- coding: utf-8 -*-
"""直接问大模型要一手，把**原始回复**打印出来（定位"为什么不给着法"）。

走的就是 :meth:`LLMAdvisor.suggest_checked` —— ``analyse_llm`` 内部用的同一条路径。
"""
from __future__ import annotations

import os
import sys

import _paths  # noqa: E402

_paths.add_to_path()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.stdout.reconfigure(encoding="utf-8")

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((1, 1))

from jieqi.gui_adapter import (  # noqa: E402
    build_llm_snapshot, gui_color_to_side, legal_moves,
)
from jieqi.llm import LLMAdvisor, load_config  # noqa: E402


def main() -> int:
    from game_state import GameState
    from jieqi.engine import Engine

    cfg = load_config()
    print("cfg.enabled=%s mode=%r model=%r key=%s"
          % (cfg.enabled, cfg.mode, cfg.model, bool(cfg.api_key)))
    advisor = LLMAdvisor(cfg)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.current_player = "red"
    side = gui_color_to_side(gs.current_player)
    engine = Engine("easy", time_limit=0.5, samples=1, seed=7)
    from jieqi.gui_adapter import captured_info, gui_board_to_engine, repetition_counts

    board = gui_board_to_engine(gs.board, turn=side)
    known, hidden = captured_info(gs.move_history)
    result = engine.analyse(board, belief=None,
                            history_counts=repetition_counts(gs),
                            own_captured=known, captured_hidden=hidden)
    snap = build_llm_snapshot(gs, result, ai_color=side)
    print("快照里候选数 = %d" % len(snap.get("engine_candidates") or []))
    print("")
    print("=== 用 style=suggest 问一次 ===")
    llm = advisor.suggest_checked(snap, style="suggest", max_retries=0)
    print("ok      = %s" % llm.ok)
    print("error   = %r" % llm.error)
    print("parsed  = %r" % (llm.parsed,))
    print("move    = %r" % (llm.move,))
    print("comment = %s" % (llm.comment or "")[:300])
    print("")
    print("原始文本 text =")
    print((llm.text or "")[:1200])
    legal = sorted(legal_moves(gs.board, side))
    print("")
    print("本机合法着法前 10：%s"
          % " ".join("%s" % m for m in legal[:10]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
