# -*- coding: utf-8 -*-
"""复盘一局棋：重放到指定手数，用引擎重新分析，看"当时有没有看到杀"。

用法::

    python _bench/diagnose_game.py <棋谱> [--from 38] [--to 47] [--time 2] [--samples 4]

对每一手（默认最后几手）打印：

* 引擎现在的评估（走子方视角 / 红方视角）与深度；
* 引擎最佳着法、以及**实际走的着法**在引擎候选里的得分；
* 若引擎报告"杀"，打印几步杀；
* 该局面下对方是否有**一步杀**（直接暴力枚举：走一步看对方能不能吃将）。

最后一条是关键：如果对手存在一步杀而我们没看到，那就是搜索没搜到杀。
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
from jieqi.consts import BLACK, RED  # noqa: E402
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    captured_info, gui_board_to_engine, gui_color_to_side, legal_moves,
    repetition_counts,
)


def build_state(record, ply):
    """重放棋谱到第 ply 手，返回 GameState（公开信息视角）。"""
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


def opponent_one_move_mate(board, side_int):
    """对方（side_int 的对手）是否有一手能吃将？返回那些着法。"""
    eng = gui_board_to_engine(board, turn=side_int)
    opp = 1 - side_int
    out = []
    for mv in eng.gen_moves(opp):
        eng.make(mv)
        if eng.kings[side_int] < 0:
            out.append(mv)
        eng.unmake()
    return out


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("--from", dest="start", type=int, default=None,
                        help="从第几手之后开始分析（默认最后 10 手）")
    parser.add_argument("--to", dest="end", type=int, default=None)
    parser.add_argument("--time", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args(argv)

    record = R.load_record(args.record)
    if record is None:
        print("读不到棋谱：%s" % args.record)
        return 2
    moves, _reveals, meta = R.replay_plan(record)
    total = len(moves)
    start = args.start if args.start is not None else max(0, total - 10)
    end = args.end if args.end is not None else total
    print("棋谱 %s：%d 手，结果 %s；分析第 %d~%d 手（每手 %.1fs / %d 采样）"
          % (meta.get("name"), total, meta.get("result"), start + 1, end,
             args.time, args.samples))
    print("")

    for ply in range(start, end):
        gs = build_state(record, ply)
        side = gui_color_to_side(gs.current_player)
        side_name = "红" if side == RED else "黑"
        known, hidden = captured_info(gs.move_history)
        board = gui_board_to_engine(gs.board, turn=side)
        engine = Engine("normal", time_limit=args.time, samples=args.samples,
                        seed=11)
        result = engine.analyse(board, history_counts=repetition_counts(gs),
                                own_captured=known, captured_hidden=hidden)
        played = moves[ply]
        played_mv = (played[0] * 9 + played[1]) << 7 | (played[2] * 9 + played[3])
        mate_now = opponent_one_move_mate(gs.board, side)
        line = ("第 %2d 手（%s走）深度 %d  %+6.0f（红方视角 %+6.0f）  %s"
                % (ply + 1, side_name, result.depth, result.score,
                   result.score, result.verdict))
        print(line)
        print("     引擎最佳：%s" % B.move_str(result.move))
        stats = {st.move: st for st in result.stats}
        if played_mv in stats:
            print("     实际走的：%s（走子方视角 %+.0f）  落后最佳 %+.0f"
                  % (B.move_str(played_mv), stats[played_mv].mean,
                     stats[result.move].mean - stats[played_mv].mean
                     if result.move in stats else float("nan")))
        else:
            print("     实际走的：%s（不在候选里）" % B.move_str(played_mv))
        if mate_now:
            print("     ⚠ 对方有 **一步吃将** 的着法：%s"
                  % "、".join(B.move_str(m) for m in mate_now[:4]))
        # 引擎实际选择的着法走完之后，对方是否一步将死我们
        after_mate = None
        if result.move is not None:
            test = gui_board_to_engine(gs.board, turn=side)
            test.make(result.move)
            reply = test.gen_moves(1 - side)
            for mv in reply:
                test.make(mv)
                if test.kings[side] < 0:
                    after_mate = mv
                test.unmake()
                if after_mate:
                    break
        if after_mate is not None:
            print("     ✗ 引擎选的 %s 走完会被对方一步吃将！"
                  % B.move_str(result.move))
        print("")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
