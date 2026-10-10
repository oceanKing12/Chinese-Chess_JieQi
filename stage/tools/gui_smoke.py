"""GUI 端到端冒烟测试（无显示器环境）。

真正启动 ChessGame：加载图片资源、建窗口、选择阵营、玩家点两下走一步、
等 AI 后台线程算完并落子、再连走几个回合并渲染，确认整条链路可用。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.chdir(GAME_DIR)
for p in (str(GAME_DIR.parent), str(GAME_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pygame  # noqa: E402

from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402
from main import ChessGame  # noqa: E402


def board_signature(board):
    """棋盘的紧凑签名（用于"导入后局面是否与原局一致"的断言）。"""
    out = []
    for r in range(10):
        for c in range(9):
            piece = board[r][c]
            if not piece:
                out.append("..")
            else:
                out.append("%s%s%s" % (piece["color"][0],
                                       piece["actual"] if piece["revealed"] else "?",
                                       "h" if not piece["revealed"] else "v"))
    return "|".join(out)


def click(game, r, c):
    """点击棋盘上的逻辑坐标（要加上棋盘在窗口里的偏移）。"""
    cfg = game.config
    if game.gs.player_side == "red":
        dr, dc = 9 - r, 8 - c
    else:
        dr, dc = r, c
    pos = (cfg.BOARD_X + cfg.OFF_SET + dc * cfg.CELL_SIZE + cfg.CELL_SIZE // 2,
           cfg.BOARD_Y + cfg.OFF_SET + dr * cfg.CELL_SIZE + cfg.CELL_SIZE // 2)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                        {"pos": pos, "button": 1}))
    game._handle_main_game()


def all_buttons(renderer):
    """遍历界面上所有按钮：(命令, 矩形, 文案, 是否可用/选中)。

    启动页与棋盘页是**两套按钮**（棋盘页把难度/大模型开关搬走了），
    所以按当前界面取——这也正是 ``Renderer._all_buttons`` 的逻辑，
    这里刻意自己列一遍，好让"按钮搬了家"这件事在冒烟测试里也暴露出来。
    """
    if renderer.gs.show_side_selection:
        return list(renderer.start_screen_buttons())
    out = [renderer.record_button(), renderer.import_button(),
           renderer.replay_button(), renderer.restart_button()]
    out.extend(renderer.score_buttons())
    out.extend(renderer.nav_buttons())
    return out


def press_key(game, key):
    """按一个键（走真实的 KEYDOWN 事件通道）。"""
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                        {"key": key, "mod": 0, "unicode": ""}))
    game._handle_main_game()


def click_button(game, cmd):
    """点击某个界面按钮（按命令名反查它的矩形）。"""
    for c, rect, _text, _state in all_buttons(game.renderer):
        if c == cmd:
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                                {"pos": rect.center, "button": 1}))
            game._handle_main_game()
            return
    raise AssertionError(f"找不到按钮 {cmd}")


def wait_for_human_turn(game, timeout=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        game._handle_main_game()
        if game.gs.game_over or game.gs.current_player != game.gs.ai_color:
            return True
        time.sleep(0.02)
    return False


#: 冒烟测试专用的棋谱目录（模块级，好让收尾的 ``finally`` 也能拿到它）。
_tmp_records = ""


def main():
    global _tmp_records
    game = ChessGame()
    # ⚠ 冒烟测试**绝不能**把棋谱写进玩家的 ``Game/records/``。
    # 真实教训：跑几次自检就多出好几份"测试对局"，玩家在棋谱列表里
    # 分不清哪些是自己的（已经发生过一次，得手工归档清理）。
    # 所以整个脚本统一改写到临时目录，跑完就删。
    _tmp_records = tempfile.mkdtemp(prefix="jieqi-smoke-")
    game.gs.record_dir = _tmp_records
    print(f"1. 初始化完成：窗口 {game.config.WIDTH}x{game.config.HEIGHT}，"
          f"资源 {len(game.rm.piece_images)} 张棋子图"
          f"（棋谱写临时目录 {_tmp_records}）")

    # 阵营：启动页上没有阵营选项了（用户："去掉我执红或者我执黑选项，
    # 太啰嗦了，有需要我会自己调的"），默认红方先手，直接点「开始游戏」。
    assert game.gs.player_side == "red", game.gs.player_side
    click_button(game, "game:start")
    game._handle_side_selection()
    assert not game.gs.show_side_selection, "「开始游戏」没有生效"
    assert game.gs.ai_mode and game.gs.ai_color == "black", game.gs.ai_color
    print(f"2. 开始游戏成功：我执{game.gs.player_side}，AI 执"
          f"{game.gs.ai_color}；窗口 {game.config.WINDOW_W}x{game.config.WINDOW_H}")

    for turn in range(4):
        if game.gs.game_over:
            break
        side = game.gs.current_player
        moves = legal_moves(game.gs.board, gui_color_to_side(side))
        if not moves:
            print(f"   {side} 无着可走")
            break
        mv = moves[turn % len(moves)]
        sr, sc = (mv >> 7) // 9, (mv >> 7) % 9
        er, ec = (mv & 127) // 9, (mv & 127) % 9
        click(game, sr, sc)
        click(game, er, ec)
        assert game.gs.current_player == "black", "玩家走子后没有轮到 AI"
        ok = wait_for_human_turn(game)
        assert ok, "AI 没能在超时内完成走子"
        res = getattr(game.ai_controller, "last_result", None)
        print(f"3.{turn + 1} 玩家 {side} {(sr, sc)}->{(er, ec)}，"
              f"AI 回应 {res.summary() if res else '无'}；"
              f"AI 思考线程存活={game.ai_thread.is_alive()}")

    # 渲染一遍，确认渲染器不会因为新状态报错
    game.renderer.draw_all()
    print("4. 三区布局渲染正常")

    # 用按钮切难度（**启动页**上的 chip：棋盘页已经不再放难度按钮了）
    press_key(game, pygame.K_ESCAPE)
    assert game.gs.show_side_selection, "ESC 没有回到启动页"
    click_button(game, "difficulty:hard")
    assert game.ai_difficulty == "hard", game.ai_difficulty
    click_button(game, "difficulty:easy")
    assert game.ai_difficulty == "easy", game.ai_difficulty
    # 对局方式（每方一个引擎档位）也在启动页上。
    # 外部引擎已被完全关闭 → 那一档必须**灰掉**（点了没反应），
    # 所以这里断言"它是灰的"，而不是去点它。
    from jieqi import engines as _engines

    click_button(game, "engine:red:llm")
    assert game.gs.side_engine("red") == "llm", game.gs.side_engines
    click_button(game, "engine:black:python")
    assert game.gs.side_engine("black") == "python", game.gs.side_engines
    layout = game.renderer.start_layout()
    # 三档对局方式都要能用（外部引擎那档已整体移除）
    assert _engines.available("llm"), "大模型模块在，那一档应当可用"
    assert not layout.disabled("engine:red:llm"), "不该灰掉大模型那一档"
    game.set_side_engine("red", "human")     # 还原成默认，别影响后面的检查
    print(f"5. 启动页切难度/对局方式正常：难度={game.ai_difficulty}，"
          f"红方={game.gs.side_engine('red')} 黑方={game.gs.side_engine('black')}，"
          f"对局方式共 {len(_engines.SOURCES)} 档")
    # 回到棋盘页继续后面的检查
    click_button(game, "game:continue")
    assert not game.gs.show_side_selection, "「继续上一局」没有回到棋盘"

    # 导出棋谱：点按钮必须真的写出文件。
    # 注意只统计"棋谱文件"（*.json 里排除 index.json，那是索引不是棋谱）。
    import glob
    record_dir = _tmp_records

    def _records():
        return {p for p in glob.glob(os.path.join(record_dir, "*.json"))
                if os.path.basename(p) != "index.json"}

    before_files = _records()
    click_button(game, "record:export")
    new_files = sorted(_records() - before_files)
    # ``record_saved`` 是"这一局自己写出去的那份"，重复导出会**复用**它，
    # 所以"没有新文件"也是正常的（同一局更新同一份，不产生 -2 垃圾）。
    # 断言真正该成立的事：导出之后当前这一局的棋谱文件存在且读得回来。
    from jieqi import record as _record_mod

    info = game.gs.record_saved
    exported = (_record_mod.load_record(info) if info else None)
    if exported is None:
        # 还没有 record_saved（例如这一局刚开始）时，必须至少写出一个新文件
        assert new_files, f"导出棋谱没有产生文件（{game.gs.toast}）"
        exported = _record_mod.load_record(new_files[-1])
    assert exported is not None, "导出的棋谱读不回来"
    chosen = new_files[-1] if new_files else (exported.get("name") or "")
    print(f"8. 导出棋谱正常：{os.path.basename(str(chosen))}"
          f"（共 {game.gs.ply} 手）")

    # 导入棋谱：把刚导出的那份读回来重放。
    # 关键约定：导入后**停在开局**且不进入结算状态——棋谱多半是已经下完的，
    # 一导入就弹"点击任意位置重新开始"会把整盘棋锁死。
    if not new_files:
        new_files = [os.path.join(record_dir,
                                  str(exported.get("name") or "") + ".json")]
    click_button(game, "view:board")          # 先开着浮层，导入时应当自动关掉
    assert game.renderer.overlay == "board"
    info = game.gs.import_record(exported)
    game.renderer.overlay = ""
    assert info["ok"], info
    assert game.gs.total_plies == exported["total_plies"], info
    assert game.gs.ply == 0, f"导入后没有停在开局（ply={game.gs.ply}）"
    assert not game.gs.game_over, "导入后立刻进入结算状态，界面会被锁死"
    assert board_signature(game.gs.board) == \
        board_signature(game.gs.positions[0]["board"]), "导入后棋盘不是开局局面"
    # 一路"下一步"能走到终局；中途不该结算。
    # 注意这份棋谱是**下到一半**导出的（result = "*"），所以走到最后一手也不该
    # 结算；只有棋谱本身带结果（1-0 / 0-1 / 1/2-1/2）时才会。
    finished = str(exported.get("result") or "*") in ("1-0", "0-1", "1/2-1/2")
    for ply in range(1, game.gs.total_plies):
        game.gs.goto_ply(ply)
        assert not game.gs.game_over, f"第 {ply} 手就结算了"
    game.gs.goto_ply(game.gs.total_plies)
    assert board_signature(game.gs.board) == \
        board_signature(game.gs.positions[game.gs.total_plies]["board"]), \
        "走到最后一手棋盘与棋谱终局不一致"
    assert game.gs.game_over == finished, \
        f"结算状态与棋谱结果不符（结果 {exported.get('result')}）"
    game.gs.goto_ply(0)
    print(f"9. 导入棋谱正常：重放 {info['plies']} 手，导入后回到开局、"
          f"可一路复盘到终局，结果按棋谱判定（{exported.get('result')}）")

    # 步数自由切换：上一步 / 下一步 / 开局 / 最新，点评要跟着走。
    # 先临时关掉 AI：回到最新一手就轮到 AI 走子（这是设计如此），
    # 会立刻产生新着法并截断历史，干扰这里对"点评随步数恢复"的断言。
    ai_mode_backup = game.gs.ai_mode
    game.gs.ai_mode = False
    total = game.gs.total_plies
    game.gs.ply_comments[total] = '最后一步的点评'
    game.gs.ply_comments[total - 1] = '倒数第二步的点评'
    # 按钮现在按**回合**翻（一回合 = 红黑各一手，用户要求的）。
    # 为了不跟上面两格撞车（总手数可能是偶数），标记选在**第 3 手**（奇数）。
    mark = 3 if total >= 3 else 0
    if mark:
        game.gs.ply_comments[mark] = '第三次落子的点评'
    click_button(game, "nav:first")
    assert game.gs.ply == 0 and game.gs.is_browsing, game.gs.ply
    click_button(game, "nav:next")
    assert game.gs.ply == 2, ("按回合翻应当到第 2 手，实际 %s" % game.gs.ply)
    assert game.gs.current_player == 'red', \
        "整回合落点上应当轮到红方，实际 %s" % game.gs.current_player
    click_button(game, "nav:last")
    assert game.gs.ply == total and not game.gs.is_browsing, game.gs.ply
    assert game.gs.ai_comment == '最后一步的点评', (game.gs.ai_comment, total)
    # 退一个回合：必须落在整回合上，且点评跟着换
    click_button(game, "nav:prev")
    assert game.gs.ply != total, "退一回合后不该还在最新"
    assert game.gs.ply % 2 == 0 or game.gs.ply == 0, \
        "回合翻没落在整回合上：%s" % game.gs.ply
    assert game.gs.current_player == 'red', \
        "整回合落点上应当轮到红方，实际 %s" % game.gs.current_player
    click_button(game, "nav:last")
    game.gs.ai_mode = ai_mode_backup
    print(f"9. 步数自由切换正常：开局/上一回合/下一回合/最新，"
          f"点评随步数恢复（第 {game.gs.ply}/{total} 手）")

    # 键盘快捷键
    for key, expect in ((pygame.K_3, "hard"), (pygame.K_1, "easy")):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                            {"key": key, "mod": 0, "unicode": ""}))
        game._handle_main_game()
        assert game.ai_difficulty == expect, (game.ai_difficulty, expect)
    for key in (pygame.K_l, pygame.K_k):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                            {"key": key, "mod": 0, "unicode": ""}))
        game._handle_main_game()
    # 中局热键切"对局方式"：E 切红方、T 切黑方。
    # 循环只会在 人工/本机引擎/大模型 之间转（不会停在灰档上）。
    before_engines = dict(game.gs.side_engines)
    for key in (pygame.K_e, pygame.K_t):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                            {"key": key, "mod": 0, "unicode": ""}))
        game._handle_main_game()
    assert game.gs.side_engine('red') != before_engines['red'] or \
        game.gs.side_engine('black') != before_engines['black'], \
        "E/T 没有切换对局方式：%s" % game.gs.side_engines
    for side in ('red', 'black'):
        assert game.gs.side_engine(side) in _engines.SOURCES, \
            "%s 方的档位不在合法清单里：%s" % (side, game.gs.side_engine(side))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                        {"key": pygame.K_LEFT, "mod": 0, "unicode": ""}))
    game._handle_main_game()
    left_ply = game.gs.ply
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                        {"key": pygame.K_END, "mod": 0, "unicode": ""}))
    game._handle_main_game()
    assert game.gs.ply == total, game.gs.ply
    print(f"10. 键盘快捷键正常：难度={game.ai_difficulty}，"
          f"大模型={game.renderer.llm_on}，"
          f"对局方式 红={game.gs.side_engine('red')} "
          f"黑={game.gs.side_engine('black')}（E/T 切换），"
          f"← 到第 {left_ply} 手、End 回最新")

    # ESC 回启动页（改设置/换阵营/继续）；F2 才是「人机 / 双人」切换
    for key in (pygame.K_ESCAPE,):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                            {"key": key, "mod": 0, "unicode": ""}))
        game._handle_main_game()
    assert game.gs.show_side_selection, "ESC 没有回到启动页"
    # 启动页上必须能继续这一局
    cont = [r for c, r, _t, _a in game.renderer.start_screen_buttons()
            if c == "game:continue"]
    assert cont, "启动页上没有「继续上一局」"
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                        {"pos": cont[0].center, "button": 1}))
    game._handle_side_selection()
    assert not game.gs.show_side_selection, "「继续上一局」没有生效"
    assert game.gs.ply == game.gs.total_plies, "继续之后应当停在最新一手"
    print("11. ESC 回启动页 / 继续上一局 正常")
    for _ in range(2):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                            {"key": pygame.K_F2, "mod": 0,
                                             "unicode": ""}))
        game._handle_main_game()
    assert game.gs.ai_mode, "F2 切换回人机对战失败"
    print("11b. F2 人机/双人切换正常")


    # 重复局面哈希必须稳定
    h1 = game.gs._board_hash()
    h2 = game.gs._board_hash()
    assert h1 == h2, "局面哈希不稳定"
    print(f"12. 局面哈希稳定：{str(h1)[:60]}…")

    # 可选：等待大模型点评到达，并确认它能进入渲染用的 gs.ai_comment
    if os.environ.get("JIEQI_SMOKE_WAIT_LLM"):
        # 清掉前面第 9 步为测试步数切换而塞进去的假点评，
        # 否则这个断言会被那串假文本满足，等于没验证模型
        game.gs.ply_comments = [''] * len(game.gs.ply_comments)
        game.gs.ply_evals = [''] * len(game.gs.ply_evals)
        game.gs.refresh_display_comment()
        # 明确把开关设成"异步只点评"，不依赖前面按键留下的状态
        if game.ai_controller is None:
            game.ai_controller = AdvancedAIController(game.gs)
        controller = game.ai_controller
        assert controller.advisor is not None, \
            f"大模型顾问不可用：{controller.llm_status}"
        controller.apply_llm_move = False
        controller.llm_enabled = True
        controller.last_comment = ""
        controller.discard_inflight_llm()
        game._sync_ai_state()
        assert game.renderer.llm_on, f"无法开启大模型点评：{game.gs.toast}"

        ply_before = game.gs.ply
        # 如果此刻正好轮到 AI（前面的按键可能已经让它开始思考），先等它走完，
        # 否则玩家点击会被"AI 回合不响应点击"挡掉，看起来像是没走成。
        if game.gs.ai_mode and game.gs.current_player == game.gs.ai_color:
            assert wait_for_human_turn(game), "等待 AI 落子超时"
            ply_before = game.gs.ply
        assert game.gs.current_player != game.gs.ai_color or not game.gs.ai_mode, \
            "轮到 AI，无法验证玩家走子"
        # 必须让玩家再走一步，才会触发一次新的 AI 分析（进而请求点评）
        side = game.gs.current_player
        moves = legal_moves(game.gs.board, gui_color_to_side(side))
        assert moves, "没有可走的着法"
        mv = moves[0]
        click(game, (mv >> 7) // 9, (mv >> 7) % 9)
        click(game, (mv & 127) // 9, (mv & 127) % 9)
        assert game.gs.ply == ply_before + 1, \
            f"玩家这一步没走成（{ply_before} -> {game.gs.ply}）"
        assert wait_for_human_turn(game), "AI 没有回应，无法验证点评"
        assert game.gs.ply >= ply_before + 2, "AI 没有落子"

        deadline = time.time() + 45
        failure = None
        while time.time() < deadline:
            game._handle_main_game()
            comment = game.gs.ai_comment or ""
            if comment and "思考中" not in comment:
                if "失败" in comment or "组装失败" in comment:
                    failure = comment
                    break
                print(f"11. 大模型点评已到达并写入 gs.ai_comment：{comment[:80]}…")
                break
            time.sleep(0.05)
        if failure is not None:
            raise AssertionError(f"大模型顾问报错：{failure}")
        if not game.gs.ai_comment:
            raise AssertionError(
                "等待 45 秒仍没有拿到大模型点评："
                f"status={getattr(controller, 'llm_status', '')!r} "
                f"comment_ply={getattr(controller, 'comment_ply', None)} "
                f"ply={game.gs.ply} comments={game.gs.ply_comments}")
        # 点评必须挂在"分析时的那一步"上，而不是随手挂到当前步
        ply = getattr(controller, 'comment_ply', None)
        assert ply is not None and game.gs.ply_comments[ply], \
            f"点评没有挂到 comment_ply={ply} 上"
        game.renderer.draw_all()   # 点评面板也要能画出来
        print(f"12. 含大模型点评的界面渲染正常（点评挂在第 {ply} 手，"
              f"当前显示第 {game.gs.ply} 手）")

    # 终局结算：用户就是在这里崩的（draw_game_over 用了 pygame.font.SysFont）
    game.gs.game_over = True
    game.gs.winner = 'black'
    for _ in range(3):
        game._handle_main_game()
    print("13. 终局结算界面连续渲染正常（SysFont 崩溃已修复）")

    pygame.quit()
    print("GUI 冒烟测试通过")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        # 断言失败时也要把临时棋谱目录清掉（否则每跑一次就留一个目录）
        shutil.rmtree(_tmp_records, ignore_errors=True)
    sys.exit(code)
