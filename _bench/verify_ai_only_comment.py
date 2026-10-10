"""实机验证：点评只针对 AI 自己的招法。

1. AI 自己走棋 → 会请求大模型，且 prompt 里有「点评对象：AI 自己的招法」；
2. 玩家点「让 AI 替我走这步」→ 同一套引擎，但**一次请求都不发**。
"""
import os
import sys
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
from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402


def step(gs):
    side = gs.current_player
    moves = legal_moves(gs.board, gui_color_to_side(side))
    mv = moves[0]
    gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9, (mv & 127) // 9, (mv & 127) % 9)
    gs.current_player = 'black' if side == 'red' else 'red'


def main():
    pygame.init()
    pygame.display.set_mode((1, 1))
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = 'red'
    gs.current_player = 'red'
    gs.enable_ai('black')
    step(gs)                                   # 红方先走，轮到 AI

    controller = AdvancedAIController(gs)
    print("顾问状态：", controller.llm_status)
    assert controller.advisor is not None, "顾问没装配上"

    # ---- 1) AI 自己走棋：应该请求点评 ----
    sent = []
    real = controller.advisor.suggest_async

    def spy(snapshot, **kwargs):
        sent.append(snapshot)
        return real(snapshot, **kwargs)

    controller.advisor.suggest_async = spy
    result = controller.analyse()              # comment=True
    print(f"1) AI 走棋：着法={result.move}  发起请求={len(sent)}")
    assert sent, "AI 自己走棋却没有请求点评"
    prompt = sent[0]
    print("   comment_target =", prompt.get("comment_target"))
    assert prompt.get("comment_target") == "AI 自己的招法"
    import time
    for _ in range(60):
        time.sleep(0.2)
        if controller.last_comment and "思考中" not in controller.last_comment:
            break
    print("   点评：", (controller.last_comment or "(空)")[:110])

    # ---- 2) 替玩家走棋：一次请求都不该发 ----
    sent.clear()
    controller.last_comment = ""
    controller.apply_llm_move = True           # 连"应用招法"也必须被排除
    controller.apply_llm_move = False
    controller.apply_llm_move = True
    res2 = controller.analyse(comment=False)
    print(f"2) 替玩家走棋：着法={res2.move}  发起请求={len(sent)}")
    assert not sent, f"替玩家走棋时不该访问大模型，却发了 {len(sent)} 次"
    assert controller.last_comment == "", controller.last_comment

    # ---- 3) 同步路径（应用大模型招法）也不能被触发 ----
    sent.clear()
    controller.apply_llm_move = True
    res3 = controller.analyse(comment=False)
    print(f"3) 替玩家+应用招法：着法={res3.move}  发起请求={len(sent)}")
    assert not sent, "替玩家走棋时连同步请求都不该发"

    print("OK：点评只针对 AI 自己的招法")
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
