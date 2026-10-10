# -*- coding: utf-8 -*-
"""引擎"看杀"体检：**从真实棋谱里**找出它漏掉的强制杀。

做法（对棋谱的每一手）：

1. 站在走子方的角度，用**暴力深搜**（只找吃将的线，深度可控）判断"我方是否有
   在 N 层内吃对方将的强制着法"；
2. 同时让引擎正常分析这个局面；
3. 如果暴力搜到了强制杀、而引擎的评估**不是杀分**（``|score| < MATE_BOUND``），
   记一个"漏杀"。

这就是用户反馈的核心：被叫杀时引擎还在说"我方大优"。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.stdout.reconfigure(encoding="utf-8")
import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from jieqi import board as B  # noqa: E402
from jieqi import record as R  # noqa: E402
from jieqi.consts import RED, BLACK  # noqa: E402
from jieqi.engine import Engine, MATE_BOUND  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    captured_info, gui_board_to_engine, gui_color_to_side, repetition_counts,
)


def build_state(record, ply):
    from game_state import GameState

    moves, reveals, _meta = R.replay_plan(record)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    for index, (sr, sc, er, ec) in enumerate(moves[:ply], start=1):
        setup = reveals.get(index)
        if setup is not None:
            cell = gs.board[setup[0]][setup[1]]
            if cell is not None and not cell.get("revealed"):
                cell["actual"] = setup[2]
        gs.move_piece(sr, sc, er, ec)
        gs.current_player = "black" if gs.current_player == "red" else "red"
    gs.animations.clear()
    return gs


def forced_capture(board, side, depth):
    """``side`` 是否能在 ``depth`` 层内强制吃掉对方的将。

    只有"吃将"才算赢，所以这里只搜"能不能吃到将"，不做评估——
    完全独立于引擎的评估函数，用来当"标准答案"。
    返回 ``True/False``。
    """
    return _fc(board, side, depth, _cache={})


def _fc(board, side, depth, _cache):
    """自己这一层：只要有一个着法能吃将（或子层能强制）就成功。"""
    if board.kings[1 - side] < 0:
        return True
    if depth <= 0:
        return False
    moves = board.gen_moves(side)
    if not moves:
        return False
    for mv in moves:
        board.make(mv)
        if board.kings[1 - side] < 0:
            board.unmake()
            return True
        # 对手这一层：对手所有着法都必须无法逃过（即每一手之后我们仍能强制）
        ok = _all(board, side, depth - 1, _cache)
        board.unmake()
        if ok:
            return True
    return False


def _all(board, side, depth, _cache):
    """对手所有着法之后，我们都还能强制吃将。"""
    opp = 1 - side
    replies = board.gen_moves(opp)
    if not replies:
        # 对手无棋可走（被将死）：也算我们赢
        return True
    for mv in replies:
        board.make(mv)
        if board.kings[side] < 0:      # 我们反而被吃：这条线失败
            board.unmake()
            return False
        ok = _fc(board, side, depth - 1, _cache)
        board.unmake()
        if not ok:
            return False
    return True


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    parser.add_argument("--depth", type=int, default=3,
                        help="暴力搜几层（3 层 ≈ 两步杀）")
    parser.add_argument("--time", type=float, default=1.5)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args(argv)

    total_positions = 0
    missed = 0
    for path in args.records:
        record = R.load_record(path)
        if record is None:
            print("读不到：%s" % path)
            continue
        moves, _r, meta = R.replay_plan(record)
        print("== %s（%d 手）==" % (meta.get("name") or path, len(moves)))
        for ply in range(args.start, len(moves)):
            gs = build_state(record, ply)
            side = gui_color_to_side(gs.current_player)
            board = gui_board_to_engine(gs.board, turn=side)
            if not forced_capture(board, side, args.depth):
                continue
            total_positions += 1
            known, hidden = captured_info(gs.move_history)
            engine = Engine("normal", time_limit=args.time,
                            samples=args.samples, seed=11)
            result = engine.analyse(
                gui_board_to_engine(gs.board, turn=side),
                history_counts=repetition_counts(gs),
                own_captured=known, captured_hidden=hidden)
            side_name = "红" if side == RED else "黑"
            seen = abs(result.score) >= MATE_BOUND
            if not seen:
                missed += 1
            print("  第 %2d 手（%s走）**有 %d 层内强制吃将**；"
                  "引擎 %s  深度 %d  评估 %+8.0f（%s）最佳 %s"
                  % (ply + 1, side_name, args.depth,
                     "看到了 ✓" if seen else "漏了 ✗", result.depth,
                     result.score, result.verdict,
                     B.move_str(result.move) if result.move is not None else "(无)"))
        print("")
    print("有强制杀的局面共 %d 个，引擎漏掉 %d 个" % (total_positions, missed))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
