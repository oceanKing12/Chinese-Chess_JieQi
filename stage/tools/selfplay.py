"""自对弈 / 对抗测试：新引擎 vs 旧 AI vs 随机走子。

用法（在 stage 目录下）：
    python tools/selfplay.py --new-vs-legacy --games 4 --new-time 0.6
    python tools/selfplay.py --new-vs-random --games 20 --new-time 0.3
    python tools/selfplay.py --legacy-vs-random --games 6

新引擎按 ``--new-time`` 秒/步、``--samples`` 个采样运行；旧 AI 沿用原实现。
两个对手共用同一套 GUI ``GameState``，走子流程与实际游戏完全一致。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import sys
import time
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
GAME_DIR = STAGE / "Game"
for p in (str(STAGE), str(GAME_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from jieqi.board import encode, move_str  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    gui_board_to_engine,
    gui_color_to_side,
    is_checkmate,
    legal_moves,
)

PROJECT_ROOT = STAGE.parent if STAGE.name == "stage" else STAGE
#: 旧 AI 的备份位置（部署时会自动生成），拿它作为"旧版基线"对照
LEGACY_CANDIDATES = [
    PROJECT_ROOT / "_backup_before_ai_optimize" / "AI.py",
    Path(r"D:\py\Chinese Chess\_backup_before_ai_optimize\AI.py"),
    Path(r"D:\py\Chinese Chess\Game\AI.py"),
    STAGE / "_bench" / "legacy_AI_original.py",
]
DEFAULT_AI_SETTINGS = {
    "difficulty": "normal",
    "time_limit": 0.6,
    "samples": 2,
    "max_depth": None,
    "seed": None,
    "emit_decision_tree": False,
    "decision_tree_path": "decision_tree.json",
    "verbose": False,
    "weights": {},
}


# --------------------------------------------------------------------------
# 选手
# --------------------------------------------------------------------------
class RandomPlayer:
    name = "random"

    def __init__(self, gs, side, **kwargs):
        self.gs = gs

    def get_best_move(self):
        side = gui_color_to_side(self.gs.current_player)
        moves = legal_moves(self.gs.board, side)
        if not moves:
            return None
        mv = random.choice(moves)
        return ((mv >> 7) // 9, (mv >> 7) % 9), ((mv & 127) // 9, (mv & 127) % 9)

    def stats(self):
        return {}


class NewEnginePlayer:
    name = "new"

    def __init__(self, gs, side, settings=None, **kwargs):
        import AI as ai_module

        opts = dict(DEFAULT_AI_SETTINGS)
        opts.update(settings or {})
        ai_module.ai_settings = lambda path=None: dict(opts)  # 测试时覆盖配置
        gs.ai_color = side
        self.controller = ai_module.AdvancedAIController(gs)

    def get_best_move(self):
        return self.controller.get_best_move()

    def stats(self):
        res = self.controller.last_result
        if res is None:
            return {}
        return {"depth": res.depth, "nodes": res.nodes, "samples": res.samples,
                "score": round(res.score, 1), "elapsed": round(res.elapsed, 3)}


class LegacyPlayer:
    name = "legacy"

    def __init__(self, gs, side, move_time=None, **kwargs):
        cls = load_legacy_class()
        gs.ai_color = side
        self.controller = cls(gs)

    def get_best_move(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            move = self.controller.get_best_move()
        return move

    def stats(self):
        return {}


_LEGACY_CLASS = None


def load_legacy_class():
    """加载旧版 AI（从备份里）。找不到就明确报错，而不是悄悄用新版冒充基线。"""
    global _LEGACY_CLASS
    if _LEGACY_CLASS is not None:
        return _LEGACY_CLASS
    path = next((p for p in LEGACY_CANDIDATES if p.is_file()), None)
    if path is None:
        raise RuntimeError(
            "找不到旧版 AI 备份（_backup_before_ai_optimize\\AI.py）。"
            "旧版基线对照不可用，请改用 --new-vs-random / --asym。")
    spec = importlib.util.spec_from_file_location("legacy_ai_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载旧版 AI：{path}")
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    cls = mod.AdvancedAIController
    if not hasattr(cls, "PIECE_VALUES"):
        raise RuntimeError(
            f"{path} 不是旧版 AI（更像新版引擎），旧版基线对照不可用。")
    _LEGACY_CLASS = cls
    return _LEGACY_CLASS


PLAYERS = {"new": NewEnginePlayer, "legacy": LegacyPlayer, "random": RandomPlayer}


# --------------------------------------------------------------------------
# 对局
# --------------------------------------------------------------------------
def new_gamestate(seed=None):
    from game_state import GameState

    if seed is not None:
        random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.ai_mode = True
    gs.ai_color = "black"
    return gs


def play_game(kind_red, kind_black, settings=None, settings_red=None,
              settings_black=None, max_plies=220, seed=None, verbose=False):
    gs = new_gamestate(seed)
    players = {
        "red": PLAYERS[kind_red](gs, "red",
                                 settings=settings_red or settings),
        "black": PLAYERS[kind_black](gs, "black",
                                     settings=settings_black or settings),
    }
    log = []
    illegal = None
    t0 = time.monotonic()

    for ply in range(max_plies):
        if gs.game_over:
            break
        side = gs.current_player
        player = players[side]
        move = player.get_best_move()
        engine_board = gui_board_to_engine(gs.board, turn=gui_color_to_side(side))
        if move is None:
            if not legal_moves(gs.board, gui_color_to_side(side)):
                gs.game_over = True
                gs.winner = "black" if side == "red" else "red"
                log.append(f"{side} 无着可走，判负")
            break
        (sr, sc), (er, ec) = move
        mv = encode(sr * 9 + sc, er * 9 + ec)
        if not engine_board.is_legal(mv):
            illegal = f"{player.name} 走出非法着法 {move_str(mv)}"
            gs.game_over = True
            gs.winner = "black" if side == "red" else "red"
            log.append(illegal)
            break
        gs.move_piece(sr, sc, er, ec)
        gs.current_player = "black" if side == "red" else "red"
        gs.check_game_state()
        if verbose:
            log.append(f"{ply + 1:3d}. {side} {move_str(mv)}")

    elapsed = time.monotonic() - t0
    if not gs.game_over:
        gs.winner = "draw"
        gs.draw_reason = f"超过 {max_plies} 手"
    return {
        "winner": gs.winner,
        "winner_player": players.get(gs.winner).name if gs.winner in players else gs.winner,
        "reason": gs.draw_reason,
        "plies": gs.move_count,
        "illegal": illegal,
        "elapsed": round(elapsed, 2),
        "red": kind_red,
        "black": kind_black,
        "log": log,
        "final_board": gs.board,
    }


def main():
    ap = argparse.ArgumentParser(description="揭棋 AI 对抗测试")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--new-time", type=float, default=0.6, help="新引擎每步秒数")
    ap.add_argument("--samples", type=int, default=2, help="新引擎采样数")
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--max-plies", type=int, default=220)
    ap.add_argument("--seed", type=int, default=20240101)
    ap.add_argument("--new-vs-legacy", action="store_true")
    ap.add_argument("--new-vs-random", action="store_true")
    ap.add_argument("--legacy-vs-random", action="store_true")
    ap.add_argument("--new-vs-new", action="store_true")
    ap.add_argument("--asym", action="store_true",
                    help="新引擎自战：浅搜索(弱) vs 深搜索(强)，用来验证搜索深度的确带来棋力")
    ap.add_argument("--weak-time", type=float, default=0.15)
    ap.add_argument("--weak-samples", type=int, default=1)
    ap.add_argument("--strong-time", type=float, default=1.5)
    ap.add_argument("--strong-samples", type=int, default=6)
    ap.add_argument("--red-json", default=None,
                    help="红方引擎设置的 JSON，例如 '{\"time_limit\":1.5,\"samples\":1}'")
    ap.add_argument("--black-json", default=None,
                    help="黑方引擎设置的 JSON")
    ap.add_argument("--no-swap", action="store_true",
                    help="不交换先后手（默认每局交换，以抵消先手优势）")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.new_vs_legacy:
        pair = ("new", "legacy")
    elif args.new_vs_random:
        pair = ("new", "random")
    elif args.legacy_vs_random:
        pair = ("legacy", "random")
    elif args.new_vs_new:
        pair = ("new", "new")
    elif args.asym or args.red_json or args.black_json:
        pair = ("new", "new")
    else:
        pair = ("new", "legacy")

    settings = {"difficulty": args.difficulty, "time_limit": args.new_time,
                "samples": args.samples, "emit_decision_tree": False}
    weak = {"difficulty": "normal", "time_limit": args.weak_time,
            "samples": args.weak_samples, "max_depth": 4,
            "emit_decision_tree": False}
    strong = {"difficulty": "normal", "time_limit": args.strong_time,
              "samples": args.strong_samples, "emit_decision_tree": False}
    if args.red_json:
        weak = {**weak, **json.loads(args.red_json)}
    if args.black_json:
        strong = {**strong, **json.loads(args.black_json)}
    results = []
    for i in range(args.games):
        # 默认交替先后手，抵消先手优势
        red, black = (pair if i % 2 == 0 else pair[::-1])
        if args.no_swap:
            red, black = pair
        sr = sb = settings
        flipped = (i % 2 == 1) and not args.no_swap and pair[0] == pair[1]
        if (args.asym or args.red_json or args.black_json) and pair[0] == pair[1]:
            sr, sb = (weak, strong) if not flipped else (strong, weak)
        res = play_game(red, black, settings=settings, settings_red=sr,
                        settings_black=sb, max_plies=args.max_plies,
                        seed=args.seed + i, verbose=args.verbose)
        if sr is not sb:
            res["strong_side"] = "black" if not flipped else "red"
        results.append(res)
        if not args.json:
            tag = (f" 强={res['strong_side']}" if "strong_side" in res else "")
            print(f"[{i + 1}/{args.games}] 红={red} 黑={black} "
                  f"结果={res['winner']}({res['winner_player']}) "
                  f"手数={res['plies']} 用时={res['elapsed']}s{tag} "
                  f"{'非法:' + res['illegal'] if res['illegal'] else ''}")
            if args.verbose and res["log"]:
                print("    " + " ".join(res["log"][-12:]))

    summary = {}
    for res in results:
        for name in (res["red"], res["black"]):
            summary.setdefault(name, {"wins": 0, "losses": 0, "draws": 0})
        if res["winner"] == "draw":
            summary[res["red"]]["draws"] += 1
            summary[res["black"]]["draws"] += 1
        else:
            winner_name = res["winner_player"]
            loser = res["black"] if winner_name == res["red"] else res["red"]
            summary[winner_name]["wins"] += 1
            summary[loser]["losses"] += 1

    if args.json:
        print(json.dumps({"games": [{k: v for k, v in r.items() if k != "final_board"}
                                    for r in results],
                          "summary": summary}, ensure_ascii=False, indent=2))
    else:
        print("--- 汇总 ---")
        for name, rec in summary.items():
            print(f"{name:8s} 胜{rec['wins']} 负{rec['losses']} 和{rec['draws']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
