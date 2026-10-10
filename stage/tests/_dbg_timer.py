# -*- coding: utf-8 -*-
"""临时调试：把 AI 思考时间的每一段打出来，定位"到第二秒归零"。"""
import os
import sys
import time
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.chdir(GAME_DIR)
sys.path.insert(0, str(GAME_DIR.parent))
sys.path.insert(0, str(GAME_DIR))

import pygame  # noqa: E402

from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402
from main import ChessGame  # noqa: E402

game = ChessGame()

# 打点：_ai_think_start / _finish_ai_think / _ai_think_stop / _process_ai_move
orig_start = game._ai_think_start
orig_finish = game._finish_ai_think
orig_stop = game._ai_think_stop
orig_process = game._process_ai_move
T0 = time.monotonic()


def stamp(tag):
    return "[%7.3fs] %s" % (time.monotonic() - T0, tag)


def start():
    print(stamp("_ai_think_start() -> began=%.3f" % time.monotonic()))
    orig_start()


def finish():
    print(stamp("_finish_ai_think() began=%.3f ai_think_ms(before)=%s"
                % (game._ai_think_began, game.ai_think_ms)))
    orig_finish()
    print(stamp("   after: ai_think_ms=%s gs.ai_think_ms=%s gs.ai_thinking=%s"
                % (game.ai_think_ms, game.gs.ai_think_ms, game.gs.ai_thinking)))


def stop():
    print(stamp("_ai_think_stop()  (began=%.3f)" % game._ai_think_began))
    orig_stop()


def process():
    print(stamp("_process_ai_move()  thread_alive=%s"
                % (game.ai_thread.is_alive() if game.ai_thread else None)))
    orig_process()


game._ai_think_start = start
game._finish_ai_think = finish
game._ai_think_stop = stop
game._process_ai_move = process

# 执红开局
rects = game.renderer.side_selection_rects()
pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                    {"pos": rects['red'].center, "button": 1}))
game._handle_side_selection()
print(stamp("开局：show_side_selection=%s ai_color=%s"
            % (game.gs.show_side_selection, game.gs.ai_color)))


def click(r, c):
    cfg = game.config
    dr, dc = (9 - r, 8 - c) if game.gs.player_side == "red" else (r, c)
    pos = (cfg.BOARD_X + cfg.OFF_SET + dc * cfg.CELL_SIZE + cfg.CELL_SIZE // 2,
           cfg.BOARD_Y + cfg.OFF_SET + dr * cfg.CELL_SIZE + cfg.CELL_SIZE // 2)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                        {"pos": pos, "button": 1}))
    game._handle_main_game()


for turn in range(2):
    moves = legal_moves(game.gs.board, gui_color_to_side(game.gs.current_player))
    mv = moves[0]
    click((mv >> 7) // 9, (mv >> 7) % 9)
    click((mv & 127) // 9, (mv & 127) % 9)
    n = 0
    while game.gs.current_player == game.gs.ai_color and not game.gs.game_over:
        game._handle_main_game()
        n += 1
        if n % 100 == 0:
            print(stamp("   ...第 %d 帧：ai_thinking=%s ai_think_ms=%s"
                        " gs.ai_think_ms=%s ai_eval_text=%r"
                        % (n, game.gs.ai_thinking, game.ai_think_ms,
                           game.gs.ai_think_ms,
                           (game.gs.ai_eval_text or "").split("\n")[0])))
        if n > 3000:
            break
        time.sleep(0.01)
    print(stamp("第 %d 回合结束：toast=%r" % (turn + 1, game.gs.toast)))
    print(stamp("   ai_eval_text=%r" % (game.gs.ai_eval_text or "")))
    print(stamp("   ai_think_ms=%s _ai_think_ms_at_result=%s"
                % (game.ai_think_ms, game._ai_think_ms_at_result)))
