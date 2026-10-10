"""把三区布局渲染成 PNG，方便肉眼检查（不需要显示器）。

用法：python tools/ui_preview.py [输出目录]
生成：01_启动页.png / 01b_启动页_大模型执黑.png / 02_对局中.png / 03_被吃棋子.png
      / 04_回顾棋局.png / 05_结算.png / 06_AI思考中.png
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.chdir(GAME_DIR)
for p in (str(GAME_DIR.parent), str(GAME_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pygame  # noqa: E402

from resource_manager import ResourceManager  # noqa: E402


def save(surface, path):
    pygame.image.save(surface, str(path))
    print(f"  已保存 {path}")


def play(gs, plies, rng, prefer_capture=False):
    """随机走子；``prefer_capture`` 时优先吃子（用来造出"被吃棋子"展示条的内容）。"""
    from jieqi.gui_adapter import gui_color_to_side, legal_moves

    for _ in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        if prefer_capture:
            capturing = [mv for mv in moves
                         if gs.board[(mv & 127) // 9][(mv & 127) % 9]]
            if capturing:
                moves = capturing
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9, (mv & 127) // 9, (mv & 127) % 9)
        gs.current_player = 'black' if side == 0 else 'red'
        gs.move_count += 1
    gs.animations.clear()


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GAME_DIR.parent / "_ui_preview"
    out.mkdir(parents=True, exist_ok=True)

    from game_state import GameState
    from renderer import Renderer

    pygame.init()
    # convert_alpha() 需要先有显示模式（真实游戏里 main.py 也是先 set_mode 再加载资源）
    pygame.display.set_mode((1, 1))
    rm = ResourceManager()
    rm.load_resources()
    gs = GameState(rm)
    renderer = Renderer(rm, gs)

    # 1) 启动页（默认状态：还没下过棋，"继续上一局"灰掉）
    gs.show_side_selection = True
    renderer.mouse_pos = renderer.start_layout().difficulty_chips['hard'].center
    renderer.draw_all()
    save(renderer.window, out / "01_启动页.png")

    # 1b) 启动页（点评打开、有未完成的对局、"黑方引擎=大模型"）
    rng = random.Random(7)
    play(gs, 6, rng)
    gs.ply = gs.total_plies
    gs.player_side = 'red'
    gs.enable_ai('black')
    gs.set_side_engine('black', 'llm')          # 黑方交给大模型
    renderer.llm_on = True
    renderer.llm_note = "已装配 deepseek-chat"
    renderer.mouse_pos = (-1, -1)
    renderer.draw_all()
    save(renderer.window, out / "01b_启动页_大模型执黑.png")
    gs.show_side_selection = False

    # 2) 对局中：先走十几步，让走棋记录里有内容
    gs.player_side = 'red'
    gs.enable_ai('black')
    rng = random.Random(11)
    play(gs, 10, rng)
    # AI 走棋信息现在分两处显示：第一行是"用时/深度"，第二行是"评估"
    gs.ai_eval_text = ("AI 用时 2031ms\n深度 4　2 采样　40960 节点\n"
                       "AI 评估：+83（红方稍优）")
    gs.toast = "AI 用时 2031ms | 深度 4 | 评估 +83（红方稍优）"
    gs.ai_comment = ("当前红方稍优。推荐炮二平五，把暗炮调到中路，既瞄准黑方中卒，"
                     "又为后续车马出动让路。黑方最可能应以卒５进１或马８进７；"
                     "需要注意黑方右翼还有车马未翻开，存在被偷袭的风险，"
                     "因此不宜过早深入，保持阵型弹性更稳妥。")
    gs.selected = (4, 4)
    renderer.difficulty = 'normal'
    renderer.llm_on = True
    renderer.llm_label = 'deepseek-chat'
    renderer.draw_all()
    save(renderer.window, out / "02_对局中.png")

    # 2b) AI 思考中：顶栏应当显示**实时秒表**（旧的显示上一手用时会误导）
    import time as _time
    gs.ai_thinking = True
    gs.ai_think_started = _time.monotonic() - 3.4
    gs.toast = 'AI 思考中…'
    renderer.draw_all()
    save(renderer.window, out / "02b_AI思考中.png")
    gs.ai_thinking = False
    gs.ai_think_started = 0.0

    # 3) 多打出一些吃子，专门检查上下两条「被吃棋子」展示条
    gs.reset()
    gs.show_side_selection = False
    gs.player_side = 'red'
    gs.enable_ai('black')
    rng = random.Random(2024)
    play(gs, 60, rng, prefer_capture=True)
    gs.ai_eval_text = ("AI 用时 1876ms\n深度 4　2 采样　38112 节点\n"
                       "AI 评估：-64（黑方稍优）")
    gs.ai_comment = "双方已经吃掉了不少子；被吃的暗子仍然保持未知身份。"
    renderer.draw_all()
    save(renderer.window, out / "03_被吃棋子.png")
    known = sum(1 for rec in gs.move_history if rec.get('captured'))
    print(f"  共 {known} 手吃子；"
          f"红被吃 {sum(1 for r in gs.move_history if (r.get('captured') or {}).get('color') == 'red')}，"
          f"黑被吃 {sum(1 for r in gs.move_history if (r.get('captured') or {}).get('color') == 'black')}")

    # 4) 「回顾棋局」浮层（大棋盘 + 被吃之子）
    renderer.overlay = 'board'
    renderer.draw_all()
    save(renderer.window, out / "04_回顾棋局.png")
    renderer.overlay = ''

    # 5) 结算
    gs.game_over = True
    gs.winner = 'red'
    renderer.draw_all()
    save(renderer.window, out / "05_结算.png")

    pygame.quit()
    print("预览图已生成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
