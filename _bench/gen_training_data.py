# -*- coding: utf-8 -*-
"""批量自对弈生成训练数据（JSONL，一行一个局面）。

为什么要单独一个脚本：``record_tool.py train`` 是"把已有棋谱转样本"，
而训练真正需要的是**成规模的对局**。这个脚本直接让引擎自己跟自己下 N 盘、
边下边落盘，所以：

* 断了也能接着跑（``--skip-existing`` 跳过已经生成的盘）；
* 每盘都写一份棋谱（``records/``）**和**一份训练样本（``*.jsonl``），
  棋谱可以随时用 ``record_tool.py analyse`` 复盘，样本直接喂训练脚本；
* 进度、耗时、样本数都打出来，方便估算"要跑一晚上能出多少数据"。

用法::

    # 先小规模试跑，确认格式和速度
    python _bench/gen_training_data.py --games 20 --time 0.15

    # 正式生成（放后台跑）
    python _bench/gen_training_data.py --games 1000 --time 0.3 --samples 1 \\
        --out-records Game/records --out-jsonl Game/records/training.jsonl

**重要**：样本里默认**不带暗子真身**（``hidden_truth``）——直接用它训练会让
模型学会偷看暗子。要"事后打标签"才加 ``--with-hidden-truth``。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
GAME = _paths.GAME
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.stdout.reconfigure(encoding="utf-8")
import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from game_state import GameState  # noqa: E402
from jieqi import record as R  # noqa: E402
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    captured_info, gui_board_to_engine, gui_color_to_side, repetition_counts,
)


def play_one_game(seed: int, time_limit: float, samples: int) -> GameState:
    """让引擎自己跟自己下一整盘，返回终局的 GameState。"""
    random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.enable_ai("black")
    # 屏蔽"终局自动存盘"：批量生成时由本脚本统一命名保存，
    # 否则每个 game over 都会往默认 records/ 里再写一份，白占空间。
    gs.record_saved = "(selfplay)"
    engines = {
        "red": Engine("normal", time_limit=time_limit, samples=samples,
                      seed=seed * 2 + 1),
        "black": Engine("normal", time_limit=time_limit, samples=samples,
                        seed=seed * 2 + 2),
    }
    while not gs.game_over:
        side = gs.current_player
        known, hidden = captured_info(gs.move_history)
        board = gui_board_to_engine(gs.board, turn=gui_color_to_side(side))
        result = engines[side].analyse(
            board, history_counts=repetition_counts(gs),
            own_captured=known, captured_hidden=hidden)
        if result.move is None:
            gs.game_over = True
            gs.winner = "black" if side == "red" else "red"
            break
        frm, to = result.move >> 7, result.move & 127
        gs.move_piece(frm // 9, frm % 9, to // 9, to % 9)
        # 把引擎自己的判断记进棋谱：以后复盘能看出"当时引擎怎么想的"
        gs.set_ply_comment(gs.ply, "",
                           "评估 %+.0f（%s）深度 %d" % (result.score, result.verdict,
                                                     result.depth))
        gs.current_player = "black" if side == "red" else "red"
        gs.check_game_state()
        if gs.total_plies >= 300:      # 极端情况兜底，防止死循环
            gs.game_over = True
            gs.winner = "draw"
            gs.draw_reason = "超过 300 手"
    gs.animations.clear()
    return gs


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20, help="下多少盘")
    parser.add_argument("--time", type=float, default=0.2, help="每步思考秒数")
    parser.add_argument("--samples", type=int, default=1, help="暗子采样个数")
    parser.add_argument("--seed", type=int, default=1, help="起始随机种子")
    parser.add_argument("--out-records", default=str(GAME / "records"),
                        help="棋谱输出目录")
    parser.add_argument("--out-jsonl", default=None,
                        help="训练样本输出（默认 <out-records>/training.jsonl）")
    parser.add_argument("--skip-existing", action="store_true",
                        help="已有同名棋谱就跳过（断点续跑）")
    parser.add_argument("--with-hidden-truth", action="store_true",
                        help="样本里带暗子真身（只适合事后打标签，慎用）")
    parser.add_argument("--quiet", action="store_true", help="只打进度不打印棋谱")
    args = parser.parse_args(argv)

    out_records = Path(args.out_records)
    out_records.mkdir(parents=True, exist_ok=True)
    jsonl = Path(args.out_jsonl) if args.out_jsonl else out_records / "training.jsonl"

    mode = "a" if args.skip_existing else "w"
    total_games = total_samples = total_plies = 0
    t_start = time.monotonic()
    handle = open(jsonl, mode, encoding="utf-8")
    try:
        for index in range(args.games):
            seed = args.seed + index
            name = "selfplay-%05d" % seed
            if args.skip_existing and (out_records / (name + ".json")).is_file():
                continue
            t0 = time.monotonic()
            gs = play_one_game(seed, args.time, args.samples)
            record = gs.build_record({"name": name, "difficulty": "normal",
                                      "notes": "自对弈 seed=%d" % seed})
            R.save_record(record, directory=str(out_records), name=name)
            # 逐盘导出样本再拼进总文件：不重复实现格式，也便于单盘检查
            tmp = str(out_records / (name + ".part.jsonl"))
            info = R.export_training(record, tmp,
                                     include_hidden_truth=args.with_hidden_truth)
            if info.get("ok") and info.get("samples"):
                with open(tmp, "r", encoding="utf-8") as part:
                    handle.write(part.read())
                handle.flush()
            try:
                os.remove(tmp)
            except OSError:
                pass
            total_games += 1
            total_samples += int(info.get("samples") or 0)
            total_plies += int(record.get("total_plies") or 0)
            elapsed = time.monotonic() - t0
            spent = time.monotonic() - t_start
            eta = spent / max(1, total_games) * (args.games - total_games)
            if not args.quiet:
                print("%-16s %3d 手  %s  %.1fs  累计 %d 盘 / %d 样本  预计还需 %.0f 分"
                      % (name, record.get("total_plies", 0),
                         record.get("result", "*"), elapsed,
                         total_games, total_samples, eta / 60.0))
    finally:
        handle.close()

    spent = time.monotonic() - t_start
    print("")
    print("完成：%d 盘 / %d 手 / %d 个训练样本，用时 %.1f 分"
          % (total_games, total_plies, total_samples, spent / 60.0))
    print("样本文件：%s" % jsonl)
    print("棋谱目录：%s" % out_records)
    if total_games:
        print("平均每盘 %.1f 秒、%.0f 个样本；按这个速度，"
              "10 万样本大约需要 %.1f 小时"
              % (spent / total_games, total_samples / total_games,
                 spent / max(1, total_samples) * 100000 / 3600.0))
    if args.with_hidden_truth:
        print("注意：样本带了 hidden_truth（暗子真身），别拿它直接训练策略/价值网。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
