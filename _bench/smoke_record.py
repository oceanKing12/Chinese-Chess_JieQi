# -*- coding: utf-8 -*-
"""棋谱：导出 → 读回 → 训练样本 → 复盘找失着（自检用，不联网）。"""
import json
import os
import random
import sys
import tempfile
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

from game_state import GameState  # noqa: E402
from jieqi import record as R  # noqa: E402
from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402

gs = GameState(None)
gs.show_side_selection = False
gs.player_side = "red"
gs.enable_ai("black")
rng = random.Random(3)
for _ in range(24):
    side = gui_color_to_side(gs.current_player)
    moves = legal_moves(gs.board, side)
    if not moves:
        break
    mv = rng.choice(moves)
    gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9, (mv & 127) // 9, (mv & 127) % 9)
    gs.set_ply_comment(gs.ply, "第%d手：测试点评。" % gs.ply, "AI 评估：+%d" % (gs.ply * 7))
    gs.current_player = "black" if side == 0 else "red"

out = Path(tempfile.mkdtemp(prefix="jieqi-rec-"))
rec = gs.build_record()
print("moves:", rec["total_plies"], "positions:", len(rec["positions"]))
info = R.save_record(rec, directory=str(out), name="smoke")
print("save:", info["ok"], Path(info["path"]).name, Path(info["text_path"]).name)
loaded = R.load_record(info["path"])
text = R.record_to_text(loaded)
print("---- 记谱前 9 行 ----")
print("\n".join(text.splitlines()[:9]))

train = R.export_training(loaded, str(out / "train.jsonl"))
print("training:", train["ok"], train["samples"])
with open(out / "train.jsonl", encoding="utf-8") as fh:
    first = json.loads(fh.readline())
print("sample keys:", sorted(first.keys()))
print("board row0:", first["board"][0])
print("move:", first["move"], "|", first["move_chinese"],
      "| legal:", len(first["legal_moves"]), "| eval_red:", first["eval_red"])
print("hidden leak:", "actual" in json.dumps(loaded))

print("---- blunder scan (0.15s/ply) ----")
res = R.analyse_record(loaded, time_limit=0.15, samples=1, threshold=60, top=5)
print("checked:", res["checked"], "flagged:", len(res["items"]))
for item in res["items"]:
    print("  ply %2d %s 实际 %s(%s) 最佳 %s(%s) 落差 %.0f"
          % (item["ply"], item["side"], item["played"], item["played_score"],
             item["best"], item["best_score"], item["loss"]))
