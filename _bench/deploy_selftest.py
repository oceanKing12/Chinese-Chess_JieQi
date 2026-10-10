# -*- coding: utf-8 -*-
"""在部署目录里跑一遍棋谱全链路：下棋 → 自动导出 → CLI list/show/analyse/train。

⚠ **写盘位置**：棋谱与训练样本都写到**临时目录**，不碰玩家的 ``Game/records``。
这个脚本是自检工具，跑一次就往玩家棋谱列表里塞一份"测试对局"是不对的
（真实教训：跑几次自检，玩家的 records 里就混进了好几份 0-1 的对局，
他自己都分不清哪些是自己的）。
"""
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
GAME = _paths.GAME
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.chdir(GAME)
sys.stdout.reconfigure(encoding="utf-8")
import pygame

pygame.init()
pygame.display.set_mode((1, 1))
from game_state import GameState  # noqa: E402
from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402

#: 自检专用的棋谱目录（命令行可以用第一个参数覆盖）
RECORDS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    tempfile.mkdtemp(prefix="jieqi-selftest-"))
RECORDS.mkdir(parents=True, exist_ok=True)
print("自检棋谱目录：%s（玩家的 records 不会被写入）" % RECORDS)

gs = GameState(None, record_dir=str(RECORDS))
gs.show_side_selection = False
gs.player_side = "red"
gs.enable_ai("black")
rng = random.Random(7)
for _ in range(30):
    side = gui_color_to_side(gs.current_player)
    moves = legal_moves(gs.board, side)
    if not moves:
        break
    capturing = [mv for mv in moves if gs.board[(mv & 127) // 9][(mv & 127) % 9]]
    mv = rng.choice(capturing or moves)
    gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9, (mv & 127) // 9, (mv & 127) % 9)
    gs.current_player = "black" if side == 0 else "red"
    gs.set_ply_comment(gs.ply, "部署自检第 %d 手点评。" % gs.ply,
                       "AI 评估：%+d（测试）" % (gs.ply * 11 - 100))
gs.game_over = True
gs.winner = "red"
gs._auto_export_record()
print("自动导出得到：", gs.record_saved)
info = gs.export_record(name="deploy-selftest")
print("手动导出：", info.get("ok"), info.get("path"))

tool = str(GAME / "record_tool.py")


def run(*args):
    out = subprocess.run([sys.executable, tool, *args], capture_output=True,
                         text=True, encoding="utf-8", errors="replace",
                         cwd=str(GAME))
    print("$ record_tool %s -> %s" % (" ".join(args), out.returncode))
    print("\n".join((out.stdout or "").splitlines()[:14]))
    if out.stderr:
        print("[stderr]", out.stderr.strip()[:300])
    return out.returncode


#: ``--dir`` 是 record_tool 的**全局**选项，必须放在子命令前面
run("--dir", str(RECORDS), "list")
run("--dir", str(RECORDS), "show", "deploy-selftest", "--board", "12")
run("--dir", str(RECORDS), "train", "deploy-selftest",
    "-o", str(RECORDS / "train-selftest.jsonl"))
run("--dir", str(RECORDS), "analyse", "deploy-selftest",
    "--time", "0.2", "--samples", "1", "--threshold", "80", "--top", "4")
p = RECORDS / "train-selftest.jsonl"
if p.is_file():
    lines = p.read_text(encoding="utf-8").splitlines()
    print("训练样本行数：", len(lines))
    print("第一行前 200 字：", lines[0][:200])
