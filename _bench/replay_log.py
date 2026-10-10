# -*- coding: utf-8 -*-
"""把"日志里读到的一串着法"重建成局面，并逐手查"这一手是不是崩盘点"。

适用场景：某局已经打完了、没有留下棋谱文件，但控制台日志里留下了每一步
``最佳：(6,8)->(5,8)`` 以及评估值。把着法抄进 ``--moves`` 或一个文本文件，
这个脚本就能：

1. 回放每一步（用项目自己的规则引擎校验，走不通会明确指出第几手）；
2. 打印每一步之后的棋盘与"红方视角评估"；
3. 对指定手数（默认最后几手）用引擎重新分析，列出当时的候选与最佳着法，
   于是能看出"当时有没有更好的选择、落差多大"。

用法::

    python _bench/replay_log.py --moves-file moves.txt --analyse-last 6
    python _bench/replay_log.py --moves "(6,8)->(5,8),(9,2)->(7,4)" --time 1.0

``moves`` 文本里每行一个着法，``(...)->(...)`` 或 ``(r,c)-(r,c)`` 都认；
``#`` 开头的行与空行会被忽略。
"""
from __future__ import annotations

import argparse
import os
import re
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

from jieqi import record as R  # noqa: E402
from jieqi.board import move_str  # noqa: E402
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    captured_info, gui_board_to_engine, gui_color_to_side, legal_moves,
    repetition_counts,
)

_MOVE_RE = re.compile(r"\(\s*(\d)\s*,\s*(\d)\s*\)\s*(?:->|→|-|到|至)\s*"
                      r"\(\s*(\d)\s*,\s*(\d)\s*\)")


def parse_moves(text: str):
    out = []
    for line in str(text).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for found in _MOVE_RE.finditer(line):
            out.append(tuple(int(x) for x in found.groups()))
    return out


def make_state(seed: int = 1):
    from game_state import GameState

    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.enable_ai("black")
    gs.animations.clear()
    return gs


def reveal_map(record):
    """从棋谱里取出"第几手会把哪一格翻开成什么"。

    棋谱**不写**暗子的真身（那是隐私信息），但"某一步在某格翻出了 X"是公开
    事实，所以回放时按这个表把该格的身份补上，才能用规则引擎校验每一步。
    返回 ``{手数: ((r,c), side, 兵种字)}``。
    """
    out = {}
    for move in record.get("moves") or []:
        if not move.get("hidden"):
            continue
        revealed = move.get("revealed_as")
        if not revealed:
            continue
        to = move.get("to") or [None, None]
        try:
            out[int(move.get("index") or 0)] = ((int(to[0]), int(to[1])),
                                                move.get("side"), str(revealed))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def record_moves(record):
    out = []
    for move in record.get("moves") or []:
        frm, to = move.get("from"), move.get("to")
        try:
            out.append((int(frm[0]), int(frm[1]), int(to[0]), int(to[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def red_pov(score_side: float, side: str) -> float:
    """走子方视角 -> 红方视角。"""
    return score_side if side == "red" else -score_side


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default=None,
                        help="棋谱名/路径：直接回放整盘（最可靠，推荐）")
    parser.add_argument("--moves", default=None, help="着法串")
    parser.add_argument("--moves-file", default=None, help="每行一个着法的文件")
    parser.add_argument("--analyse-last", type=int, default=0,
                        help="对最后 N 手重新分析（0 = 只回放不分析）")
    parser.add_argument("--first-side", choices=("red", "black"), default="red",
                        help="第一步是谁走的（默认红先；从对局中途开始抄日志时"
                             "要按实际填）")
    parser.add_argument("--time", type=float, default=1.0, help="每手分析秒数")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--top", type=int, default=6, help="每手列出前几个候选")
    parser.add_argument("--quiet", action="store_true", help="不打印每步棋盘")
    args = parser.parse_args(argv)

    record = None
    if args.record:
        record = R.load_record(args.record)
        if record is None:
            print("找不到棋谱：%s" % args.record, file=sys.stderr)
            return 2

    raw = args.moves or ""
    if args.moves_file:
        raw += "\n" + Path(args.moves_file).read_text(encoding="utf-8")
    if record is not None:
        moves = record_moves(record)
        reveals = reveal_map(record)
    else:
        moves = parse_moves(raw)
        reveals = {}
    if not moves:
        print("没有可回放的着法。用 --record 给棋谱，或用 --moves / --moves-file "
              "给一串 (4,2)->(4,4)。", file=sys.stderr)
        return 2

    gs = make_state()
    gs.current_player = args.first_side
    print("回放 %d 步（第一步 %s 走）%s"
          % (len(moves), args.first_side,
             "，来源棋谱 %s" % record.get("name") if record else ""))
    for index, (sr, sc, er, ec) in enumerate(moves, start=1):
        setup = reveals.get(index)
        if setup and gs.board[setup[0][0]][setup[0][1]] is not None:
            cell = gs.board[setup[0][0]][setup[0][1]]
            if not cell.get("revealed"):
                cell["actual"] = setup[2]      # 回放时补上"这一步翻出了什么"
        side = gs.current_player
        side_int = gui_color_to_side(side)
        legal = legal_moves(gs.board, side_int)
        mv = (sr * 9 + sc) << 7 | (er * 9 + ec)
        if mv not in legal:
            print("!! 第 %d 手 %s %s 不是合法着法（可能日志里抄漏了一步）"
                  % (index, side, move_str(mv)))
            print("   当前局面的合法着法数：%d" % len(legal))
            return 3
        piece = gs.board[sr][sc] or {}
        label = ("%s%s" % ("红" if piece.get("color") == "red" else "黑",
                           piece.get("actual") if piece.get("revealed")
                           else "暗" + str(piece.get("original"))))
        target = gs.board[er][ec]
        gs.move_piece(sr, sc, er, ec)
        gs.current_player = "black" if side == "red" else "red"
        note = ""
        if target:
            note = " 吃%s" % ("红" if target.get("color") == "red" else "黑") + \
                   (str(target.get("actual")) if target.get("revealed") else "暗子")
        print("%3d. %-8s %s%s" % (index, label, move_str(mv), note))
        if not args.quiet and index % 10 == 0:
            print(R.board_to_text(gs.board))
    print("")
    print("回放完，最后一手之后的棋盘：")
    print(R.board_to_text(gs.board))

    if args.analyse_last > 0:
        count = min(args.analyse_last, len(moves))
        print("")
        print("重新分析最后 %d 手（每手 %.1fs，引擎只能看公开信息）："
              % (count, args.time))
        # 回到倒数第 count 手之前
        gs2 = make_state()
        gs2.current_player = args.first_side
        for index, (sr, sc, er, ec) in enumerate(moves[:len(moves) - count], start=1):
            gs2.move_piece(sr, sc, er, ec)
            gs2.current_player = "black" if gs2.current_player == "red" else "red"
        for offset in range(count):
            index = len(moves) - count + offset + 1
            sr, sc, er, ec = moves[index - 1]
            side = gs2.current_player
            side_int = gui_color_to_side(side)
            known, hidden = captured_info(gs2.move_history)
            board = gui_board_to_engine(gs2.board, turn=side_int)
            engine = Engine("normal", time_limit=args.time, samples=args.samples,
                            seed=1)
            result = engine.analyse(board, history_counts=repetition_counts(gs2),
                                    own_captured=known, captured_hidden=hidden)
            played = (sr * 9 + sc) << 7 | (er * 9 + ec)
            stats = {st.move: st for st in result.stats}
            print("")
            print("第 %d 手（%s走，深度 %d）：实际 %s"
                  % (index, side, result.depth, move_str(played)))
            best = result.move
            if best is not None and best in stats:
                print("   引擎现在认为最佳：%s（走子方视角 %.0f，红方视角 %.0f）"
                      % (move_str(best), stats[best].mean,
                         red_pov(stats[best].mean, side)))
            if played in stats:
                print("   实际这手的评分：走子方视角 %.0f，红方视角 %.0f"
                      % (stats[played].mean, red_pov(stats[played].mean, side)))
                if best is not None and best in stats:
                    print("   落差：%.0f（走子方视角）"
                          % (stats[best].mean - stats[played].mean))
            else:
                print("   实际这手不在引擎候选里（可能是搜索没覆盖到的着法）")
            ranked = sorted(result.stats, key=lambda s: s.mean, reverse=True)
            for rank, st in enumerate(ranked[:args.top], start=1):
                print("     %d. %-14s 走子方视角 %+6.0f  红方视角 %+6.0f"
                      % (rank, move_str(st.move), st.mean, red_pov(st.mean, side)))
            # 推进一手
            gs2.move_piece(sr, sc, er, ec)
            gs2.current_player = "black" if side == "red" else "red"
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
