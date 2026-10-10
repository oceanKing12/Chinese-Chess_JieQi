# -*- coding: utf-8 -*-
"""对比"暗子估值"两种口径的棋力：**各自对上同一个固定基准**。

不搞 A/B 同场对打（那需要在一盘棋里让两边用不同的评估函数，边界容易出错），
而是让两种口径分别跟**同一批随机对手/同一批局面**下同样的盘，比较：

* 胜率（对基准）；
* 平均"子力净收益"——用引擎自己的材料口径统计（更灵敏，样本少也能看出差别）。

用法::

    python _bench/ab_hidden_value.py --games 12 --time 0.2
"""
from __future__ import annotations

import argparse
import os
import random
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

from jieqi import evaluate as EV  # noqa: E402
from jieqi.consts import HOME_IDENT, PIECE_VALUE, PST_FLAT_SIDE  # noqa: E402
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    captured_info, gui_board_to_engine, gui_color_to_side, legal_moves,
    repetition_counts,
)

_NEW = EV._piece_value


def _old_piece_value(board, ctx, sq, side):
    """旧口径：暗子按剩余暗子池的后验期望估值。"""
    if board.hidden[sq]:
        v = ctx.dark_total.get(side, {}).get(sq)
        if v is not None:
            return v
        return ctx.mean_value.get(side, 0.0) + PST_FLAT_SIDE[side][HOME_IDENT[sq]][sq]
    ident = board.ident[sq]
    return PIECE_VALUE[ident] + PST_FLAT_SIDE[side][ident][sq]


def material_balance(gui_board):
    """红方视角的"明面材料差"（暗子按摆位身份算），用来当灵敏指标。"""
    from jieqi.consts import CHAR_TO_PIECE, PIECE_VALUE as PV

    total = 0
    for r in range(10):
        for c in range(9):
            piece = gui_board[r][c]
            if not piece:
                continue
            ident = (piece.get("actual") if piece.get("revealed")
                     else piece.get("original"))
            side = CHAR_TO_PIECE.get(str(ident))
            if side is None:
                continue
            value = PV.get(side[1], 0)
            total += value if piece.get("color") == "red" else -value
    return total


def play(seed, time_limit, samples, max_plies, mode):
    """引擎（黑方）对随机走子的红方；返回 (结果, 手数, 材料差)。"""
    from game_state import GameState

    EV._piece_value = _old_piece_value if mode == "old" else _NEW
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.enable_ai("black")
    gs.record_saved = "(ab)"
    engine = Engine("normal", time_limit=time_limit, samples=samples, seed=seed)
    rng = random.Random(seed * 31 + 7)
    plies = 0
    while not gs.game_over and plies < max_plies:
        if gs.current_player == "black":
            known, hidden = captured_info(gs.move_history)
            board = gui_board_to_engine(gs.board, turn=1)
            result = engine.analyse(board, history_counts=repetition_counts(gs),
                                    own_captured=known, captured_hidden=hidden)
            move = result.move
            if move is None:
                break
            frm, to = move >> 7, move & 127
        else:
            moves = legal_moves(gs.board, 0)
            if not moves:
                break
            mv = rng.choice(moves)
            frm, to = mv >> 7, mv & 127
        gs.move_piece(frm // 9, frm % 9, to // 9, to % 9)
        gs.current_player = "black" if gs.current_player == "red" else "red"
        gs.check_game_state()
        plies += 1
    EV._piece_value = _NEW
    winner = gs.winner if gs.game_over else None
    return winner, plies, material_balance(gs.board)


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--time", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    summary = {}
    for mode in ("old", "new"):
        wins = losses = draws = 0
        mats = []
        for index in range(args.games):
            winner, plies, mat = play(args.seed + index, args.time, args.samples,
                                      args.max_plies, mode)
            mats.append(mat)
            if winner == "black":
                wins += 1
            elif winner == "red":
                losses += 1
            else:
                draws += 1
        summary[mode] = (wins, losses, draws, sum(mats) / max(1, len(mats)))
        print("%s：引擎(黑) %d 胜 / %d 负 / %d 和；平均材料差(红视角) %+.0f"
              % ("旧口径（后验期望）" if mode == "old" else "新口径（摆位身份）",
                 wins, losses, draws, summary[mode][3]))
    print("")
    old, new = summary["old"], summary["new"]
    print("对比：胜场 %d -> %d，材料差 %+.0f -> %+.0f"
          % (old[0], new[0], old[3], new[3]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
