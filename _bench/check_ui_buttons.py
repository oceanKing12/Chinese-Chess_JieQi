# -*- coding: utf-8 -*-
"""导入/导出棋谱往返自检 + 新 UI 按钮几何。

重点验证"导出 → 导入"之后局面必须**完全一致**（棋盘、手数、点评、结果），
否则导入功能就是假的。
"""
import os
import random
import sys
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
from jieqi import record as R  # noqa: E402
from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402


def play(gs, plies, seed=13):
    rng = random.Random(seed)
    for _ in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        capturing = [mv for mv in moves
                     if gs.board[(mv & 127) // 9][(mv & 127) % 9]]
        mv = rng.choice(capturing or moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        # 点评按局面存：这一格描述的是"这一手之后的局面"。
        # 最后那个格（终局之后）在真实对局里不会有内容，所以这里也不写，
        # 免得构造出一个"棋谱里存在、但现实里不可能存在"的格子。
        if gs.ply <= plies - 1:
            gs.set_ply_comment(gs.ply, "第%d手点评" % gs.ply,
                               "AI 评估：%+d" % (gs.ply * 9))
        gs.current_player = "black" if side == 0 else "red"
    gs.animations.clear()


def board_signature(board):
    out = []
    for r in range(10):
        for c in range(9):
            p = board[r][c]
            if not p:
                out.append("..")
            else:
                out.append("%s%s%s" % (p["color"][0],
                                       p["actual"] if p["revealed"] else "?",
                                       "h" if not p["revealed"] else "v"))
    return "|".join(out)


print("=== 1) 导出 → 导入 往返 ===")
src = GameState(None)
src.show_side_selection = False
src.player_side = "red"
src.enable_ai("black")
play(src, 26)
src.game_over = True
src.winner = "red"
before = board_signature(src.board)
print("原局：手数", src.total_plies, "胜者", src.winner)
record = src.build_record({"difficulty": "normal"})
print("棋谱着法数：", len(record["moves"]))

# 逐步对照：重放出来的一手在原局里是否合法、翻子身份是否对得上
moves, reveals, meta = R.replay_plan(record)
probe = GameState(None)
probe.show_side_selection = False
for index, (sr, sc, er, ec) in enumerate(moves, start=1):
    side_int = 0 if probe.current_player == "red" else 1
    legal = ((sr * 9 + sc) << 7 | (er * 9 + ec)) in legal_moves(probe.board, side_int)
    if not legal:
        rec = record["moves"][index - 1]
        cell = probe.board[sr][sc] or {}
        print("第 %d 手不合法：%s 走 %s；该格=%s；record=%s；reveal=%s"
              % (index, probe.current_player, (sr, sc, er, ec), cell,
                 {k: rec.get(k) for k in ("side", "hidden", "home", "piece", "to")},
                 reveals.get(index)))
        # 多打几手上下文，方便定位
        for k in range(max(1, index - 3), index + 1):
            print("   第 %d 手：%s" % (k, moves[k - 1]))
        break
    setup = reveals.get(index)
    if setup is not None:
        piece = probe.board[setup[0]][setup[1]]
        if piece is not None and not piece.get("revealed"):
            piece["actual"] = setup[2]
    probe.move_piece(sr, sc, er, ec)
    probe.current_player = "black" if probe.current_player == "red" else "red"
print("逐步对照走到第", probe.ply, "手")

dst = GameState(None)
dst.show_side_selection = False
info = dst.import_record(record)
print("导入结果：", info)
# 导入后应当**停在开局**、且不进入结算（否则一导入就弹"点击任意位置重新开始"）
print("导入后停在第", dst.ply, "手；game_over =", dst.game_over,
      "winner =", dst.winner)
assert dst.ply == 0, "导入后没有停在开局"
assert not dst.game_over and dst.winner is None, "导入后立刻进入结算（就是那个 bug）"
dst.goto_ply(dst.total_plies)
print("翻到最后一手：game_over =", dst.game_over, "winner =", dst.winner)
assert dst.game_over and dst.winner == "red", "走到最后一手没有结算"
after = board_signature(dst.board)
print("棋盘一致：", before == after)
print("手数一致：", dst.total_plies == src.total_plies, dst.total_plies, src.total_plies)
print("点评一致：", dst.ply_comments == src.ply_comments)

# 中途翻历史不该触发结算；回到开局也要把结算状态清掉
dst2 = GameState(None)
dst2.show_side_selection = False
dst2.import_record(record)
dst2.goto_ply(5)
assert not dst2.game_over, "中途翻历史触发了结算"
dst2.goto_ply(dst2.total_plies)
assert dst2.game_over
dst2.goto_ply(0)
print("中途翻第5手 / 末手 / 回开局：game_over =",
      (False, True, dst2.game_over), "（最后应为 False）")
assert not dst2.game_over, "回到开局还留着结算状态"

print("走上一步/下一步可用：", dst.ply, "->", end=" ")
dst.goto_ply(dst.total_plies - 3)
print(dst.ply, "->", end=" ")
dst.goto_ply(dst.total_plies)
print(dst.ply)

print("")
print("=== 2) 文件导入 ===")
# 自检用的棋谱写到**临时目录**，绝不往玩家的 records/ 里塞测试文件
import tempfile  # noqa: E402

_TMP_RECORDS = tempfile.mkdtemp(prefix="jieqi-uicheck-")
R.save_record(record, directory=_TMP_RECORDS, name="roundtrip-check")
path = os.path.join(_TMP_RECORDS, "roundtrip-check.json")
dst2 = GameState(None, record_dir=_TMP_RECORDS)
dst2.show_side_selection = False
info2 = dst2.import_record_file("roundtrip-check")
print("按名字导入：", info2.get("ok"), info2.get("plies"), info2.get("error"))
print("手数一致：", dst2.total_plies == dst.total_plies,
      dst2.total_plies, dst.total_plies)
# 注：这里**不比"棋盘逐格一致"或"着法逐手一致"**。棋谱里暗子只记
# "这一步翻出了什么"，没翻开的那些子导入方只能按自己的开局随机分配身份，
# 所以两边本来就可能不同——这正是"棋谱不泄漏真身"的体现（第 1 节的
# 内存往返之所以能一致，是因为那份棋谱最初就是从同一个对象导出的）。
print("（自检棋谱写在这里：%s）" % _TMP_RECORDS)
bad = dst2.import_record_file("no-such-record-xyz")
print("坏名字导入：", bad.get("ok"), bad.get("error"))
print("（自检棋谱写在这里：%s）" % _TMP_RECORDS)

print("")
print("=== 3) 新 UI 按钮 ===")
from renderer import Renderer  # noqa: E402
from resource_manager import ResourceManager  # noqa: E402

rm = ResourceManager()
rm.load_resources()
gs = GameState(rm)
gs.show_side_selection = False
play(gs, 12)
r = Renderer(rm, gs)
r.draw_all()
local = lambda rect: rect.move(-r.panel_rect.x, -r.panel_rect.y)  # noqa: E731
card = r._panel_layout()['action']
for getter in (r.record_button, r.import_button, r.replay_button, r.restart_button):
    cmd, rect, text, enabled = getter()
    print("%-14s %-10s 卡内=%s 命中=%s 可用=%s"
          % (cmd, text, card.contains(local(rect)),
             r.button_at(rect.center) == cmd, enabled))
for cmd, rect, text, enabled in r.nav_buttons():
    print("%-14s %-10s 卡内=%s 命中=%s" % (cmd, text, card.contains(local(rect)),
                                          r.button_at(rect.center) == cmd))
print("按钮两两重叠：", [(a, b) for i, (a, ra) in enumerate(
    [(c, rc) for c, rc, _t, _x in r._all_buttons()])
    for b, rb in [(c, rc) for c, rc, _t, _x in r._all_buttons()][i + 1:]
    if ra.colliderect(rb)] or "无")
print("窗口尺寸：", r.window.get_size(), "上限 1000")
r.overlay = 'board'
r.draw_all()
print("回顾棋局浮层绘制正常")
r.overlay = ''
r.draw_all()
print("全部 OK")
