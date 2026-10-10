# -*- coding: utf-8 -*-
"""体检：引擎是不是还在"开局就翻暗车 / 随手弃暗马"。

做法：自对弈若干盘，统计前 N 手里

* 走的是**暗车/暗砲**（摆位身份是车/炮的未翻开子）的着法数量；
* 走暗车之后**翻出来的不是车**（= 子力下降）的次数；
* 前 N 手一共走了多少手。

用法::

    python _bench/check_hidden_piece_play.py --games 4 --time 0.15 --plies 24
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

from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    captured_info, gui_board_to_engine, gui_color_to_side, repetition_counts,
)

#: 摆位身份 -> 字（只关心车/炮这两个"翻了可能亏"的）
HOME_NAMES = {5: "车", 6: "炮"}


def run_game(seed, time_limit, samples, max_plies):
    from game_state import GameState

    random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.enable_ai("black")
    gs.record_saved = "(probe)"      # 不要往默认 records/ 写文件
    engines = {"red": Engine("normal", time_limit=time_limit, samples=samples,
                             seed=seed * 2 + 1),
               "black": Engine("normal", time_limit=time_limit, samples=samples,
                               seed=seed * 2 + 2)}
    stats = {"plies": 0, "flip_home_car": 0, "flip_home_cannon": 0,
             "flip_downgrade": 0, "flip_any": 0}
    while not gs.game_over and stats["plies"] < max_plies:
        side = gs.current_player
        known, hidden = captured_info(gs.move_history)
        board = gui_board_to_engine(gs.board, turn=gui_color_to_side(side))
        result = engines[side].analyse(
            board, history_counts=repetition_counts(gs),
            own_captured=known, captured_hidden=hidden)
        if result.move is None:
            break
        frm, to = result.move >> 7, result.move & 127
        fr, fc = frm // 9, frm % 9
        tr, tc = to // 9, to % 9
        piece = gs.board[fr][fc] or {}
        was_hidden = bool(piece) and not piece.get("revealed")
        home_name = None
        if was_hidden:
            # 摆位身份 = 该子起点格子的标准摆位
            from jieqi.consts import HOME_IDENT
            home = HOME_IDENT[fr * 9 + fc]
            home_name = HOME_NAMES.get(home)
            stats["flip_any"] += 1
        gs.move_piece(fr, fc, tr, tc)
        if was_hidden:
            revealed = (gs.board[tr][tc] or {}).get("actual")
            if home_name == "车":
                stats["flip_home_car"] += 1
                if revealed != "车" and revealed != "車":
                    stats["flip_downgrade"] += 1
            elif home_name == "炮":
                stats["flip_home_cannon"] += 1
                if revealed not in ("炮", "砲"):
                    stats["flip_downgrade"] += 1
        gs.current_player = "black" if side == "red" else "red"
        gs.check_game_state()
        stats["plies"] += 1
    return stats


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--time", type=float, default=0.15)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--plies", type=int, default=24,
                        help="每盘只看前几手（开局阶段）")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args(argv)

    total = {"plies": 0, "flip_home_car": 0, "flip_home_cannon": 0,
             "flip_downgrade": 0, "flip_any": 0}
    for index in range(args.games):
        stats = run_game(args.seed + index, args.time, args.samples, args.plies)
        for key in total:
            total[key] += stats[key]
        print("第 %d 盘前 %d 手：翻暗车 %d、翻暗炮 %d、总翻子 %d、翻亏 %d"
              % (index + 1, stats["plies"], stats["flip_home_car"],
                 stats["flip_home_cannon"], stats["flip_any"],
                 stats["flip_downgrade"]))
    print("")
    plies = max(1, total["plies"])
    print("合计 %d 盘 / %d 手（开局阶段）：" % (args.games, total["plies"]))
    print("  翻暗车（摆位身份=车）%d 次，占全部着法 %.1f%%，占翻子 %.1f%%"
          % (total["flip_home_car"], 100.0 * total["flip_home_car"] / plies,
             100.0 * total["flip_home_car"] / max(1, total["flip_any"])))
    print("  翻暗炮 %d 次；翻子共 %d 次；其中翻亏（翻出来不是原来那个大子）%d 次"
          % (total["flip_home_cannon"], total["flip_any"],
             total["flip_downgrade"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
