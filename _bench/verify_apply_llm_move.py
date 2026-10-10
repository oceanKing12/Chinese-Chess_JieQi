"""实机验证「应用大模型招法」：真的让 DeepSeek 决定 AI 走哪一步。"""
import os
import sys
import time
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
GAME = _paths.GAME
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.chdir(GAME)
sys.path.insert(0, str(GAME))
sys.path.insert(0, str(GAME.parent))

import pygame  # noqa: E402

from game_state import GameState  # noqa: E402
from AI import AdvancedAIController  # noqa: E402
from jieqi.board import move_str  # noqa: E402
from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = 'red'
    gs.current_player = 'red'
    gs.enable_ai('black')

    # 先让红方随便走一步，轮到 AI
    mv = legal_moves(gs.board, gui_color_to_side('red'))[0]
    gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9, (mv & 127) // 9, (mv & 127) % 9)
    gs.current_player = 'black'

    controller = AdvancedAIController(gs)
    print("顾问状态：", controller.llm_status)
    assert controller.advisor is not None, "顾问没装配上"

    enabled = controller.toggle_apply_llm_move()
    print("应用大模型招法开关：", enabled)
    assert enabled, "开关没打开"

    t0 = time.time()
    result = controller.analyse()
    print(f"AI 最终着法：{move_str(result.move)}  用时 {time.time() - t0:.1f}s")
    print("引擎/大模型备注：")
    for note in result.notes:
        print("   ", note)
    llm = controller.last_llm_result
    if llm is not None and llm.move:
        fr, fc, tr, tc = llm.move
        print(f"大模型推荐：({fr},{fc})->({tr},{tc})")
        from jieqi.board import encode
        same = encode(fr * 9 + fc, tr * 9 + tc) == result.move
        print("是否采用了它的招法：", "是" if same else "否")
        assert same, "开了开关却没有采用大模型招法"
    else:
        print("大模型没有给出可用的着法（已按引擎着法），备注里应有说明")
    print("大模型点评：", (controller.last_comment or '')[:120])
    print("OK")
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
