# -*- coding: utf-8 -*-
"""给 ``jieqi/reveal.py`` 的翻子权重做**自动标定**（简化版 SPSA / 爬山）。

为什么要标定：``reveal_weights`` 全 0 时等于关闭，随便填一组数字只是猜。
AB-JChess 那些 ``Reveal*`` 参数（``RevealBonusBase`` / ``RevealBonusPhase``…）
就是被自动调参训出来的，所以量纲是可参考的，但**具体数值必须用本项目自己的
自对弈结果说话**——两边的评估函数完全不同。

方法：分组爬山（比逐参数全网格省 ~5 倍时间）
--------------------------------------------
1. 从一组起点权重出发（``SEED``）；
2. 每轮对**每个参数**生成"调大 / 调小"两个候选（步长逐步减半）；
3. 每个候选和**当前最优**打 N 局（A = 候选，B = 当前最优，交替先后手）；
4. 取本轮得分最高的候选作为新的当前最优，进入下一轮；
5. 轮数用完就输出历史最优，并写进 ``ai_config.json``（可选）。

统计上的注意点（都在代码里落实了）
----------------------------------
* **黑白轮换**：同一局面开局固定，靠交替先后手抵消先手优势；
* **同一个种子集**：所有候选打的是同一批开局，比较才公平；
* **固定思考时间**：A/B 两边时间预算相同，只比权重；
* 每轮结束打印置信区间，**分不出高下时（区间跨 0）不升级**——宁可不动，
  也不要被噪声推着走（这是这类标定最常见的翻车方式）。

用法::

    python _bench/tune_reveal.py --rounds 4 --games 16 --time 0.5
    python _bench/tune_reveal.py --rounds 3 --games 12 --apply   # 写进 ai_config
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from jieqi.engine import Engine  # noqa: E402
from jieqi.gui_adapter import (  # noqa: E402
    captured_info,
    gui_board_to_engine,
    gui_color_to_side,
    repetition_counts,
)

#: 爬山起点。
#:
#: ⚠ 这组数值是**批测测出来的**，不是拍脑袋的：一开始我想当然地用了
#: ``base 30 / phase 40``（= 下面这个的两倍），批测显示它得分率只有 0.500
#: （区间跨 0.5，等于没用），而 ``base 15 / phase 20`` 拿到 0.818 [0.52, 0.95]
#: 显著更好；更激进的 ``phase 80`` 反而只有 0.182（明显更差）。
#: 也就是说：**翻子加成这个方向是对的，但量级差一倍就从"有用"变成"有害"**。
SEED = {
    "base": 15,
    "phase": 20,
    "unknown": 0,
    "comeback": 0,
    "reduction": 0,
}
#: 静着（不吃子的翻子）权重：奖励翻在安全位置
SEED_QUIET = {
    "base": -12,
    "safety": 24,
    "phase": 0,
    "high_value": 0,
    "diversity": 0,
}

#: 每个参数的可调范围（防止爬到荒谬值上）与初始步长
BOUNDS = {
    "base": (-60, 120, 30),
    "phase": (-40, 160, 40),
    "unknown": (-40, 120, 30),
    "comeback": (-80, 80, 40),
    "reduction": (-2, 4, 1),
}
BOUNDS_QUIET = {
    "base": (-60, 60, 20),
    "safety": (-48, 48, 16),
    "phase": (-40, 40, 20),
    "high_value": (-40, 40, 20),
    "diversity": (-40, 40, 20),
}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def play_one(job):
    """下一局（供并行调用）。``job`` 是 ``(index, a_side, w_a, w_qa, w_b, w_qb,
    movetime, max_plies, seed)``，返回 ``('a'|'b'|'draw', 深度列表)``。"""
    (index, a_side, w_a, w_qa, w_b, w_qb, movetime, max_plies, seed) = job
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = a_side
    # 标定要下几百局，**绝不能让它们全写进 records/**（否则棋谱目录被灌满、
    # 而且每局写盘还会拖慢对局）。``_auto_export_record`` 看到 ``record_saved``
    # 非空就直接返回，所以这一行就够了。
    gs.record_saved = "(标定对局，不保存)"
    engines = {
        "red": Engine("normal", time_limit=movetime, samples=1,
                      seed=seed, reveal_weights=(w_a if a_side == "red" else w_b),
                      reveal_quiet_weights=(w_qa if a_side == "red" else w_qb)),
        "black": Engine("normal", time_limit=movetime, samples=1,
                        seed=seed + 977,
                        reveal_weights=(w_b if a_side == "red" else w_a),
                        reveal_quiet_weights=(w_qb if a_side == "red" else w_qa)),
    }
    depths: List[int] = []
    for _ in range(max_plies):
        if gs.game_over:
            break
        side = gs.current_player
        turn = gui_color_to_side(side)
        known, hidden = captured_info(gs.move_history)
        board = gui_board_to_engine(gs.board, turn=turn)
        try:
            res = engines[side].analyse(
                board, history_counts=repetition_counts(gs),
                own_captured=known, captured_hidden=hidden)
        except Exception:
            res = None                    # 引擎内部异常不该毁掉整轮标定
        if res is None or res.move is None:
            gs.game_over = True
            gs.winner = "black" if side == "red" else "red"
            break
        if res.depth:
            depths.append(int(res.depth))
        frm, to = res.move >> 7, res.move & 127
        gs.move_piece(frm // 9, frm % 9, to // 9, to % 9)
        gs.current_player = "black" if side == "red" else "red"
        gs.check_game_state()
    winner = gs.winner or "draw"
    if winner == "draw":
        return "draw", depths
    if (winner == "red") == (a_side == "red"):
        return "a", depths
    return "b", depths


def play_match(w_a: Dict, w_qa: Dict, w_b: Dict, w_qb: Dict,
               games: int, movetime: float, max_plies: int,
               seed: int, workers: int = 1) -> Tuple[int, int, int, float]:
    """A（候选） vs B（当前最优）打 ``games`` 局，交替先后手。

    返回 ``(A 胜, B 胜, 和, 平均每步深度)``。

    ``workers > 1`` 时多局并行。**每局用各自的 Engine 实例**，所以没有共享
    状态；并行只影响速度、不影响结果（随机种子是每局固定的）。
    """
    jobs = []
    for g in range(games):
        a_side = "red" if g % 2 == 0 else "black"
        jobs.append((g, a_side, w_a, w_qa, w_b, w_qb, movetime, max_plies,
                     seed + g))
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(play_one, jobs))
    else:
        results = [play_one(job) for job in jobs]
    wins = {"a": 0, "b": 0, "draw": 0}
    depths: List[float] = []
    for outcome, got in results:
        wins[outcome] += 1
        depths.extend(got)
    avg_depth = sum(depths) / len(depths) if depths else 0.0
    return wins["a"], wins["b"], wins["draw"], avg_depth


def _score(win_a: int, win_b: int) -> float:
    """A 的得分率（和棋算半分），范围 0..1。"""
    total = win_a + win_b
    if total == 0:
        return 0.5
    return (win_a + 0.5 * (total - win_a - win_b)) / total if total else 0.5


def _wilson(win_a: int, win_b: int) -> Tuple[float, float]:
    """胜率的 95% 置信区间（Wilson，比正态近似在小样本下稳）。"""
    n = win_a + win_b
    if n == 0:
        return 0.0, 1.0
    p = win_a / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def candidates(current: Dict[str, int], bounds: Dict[str, tuple],
               step_scale: float) -> List[Tuple[str, int, Dict[str, int]]]:
    """为每个参数生成"加一步 / 减一步"两个候选。"""
    out: List[Tuple[str, int, Dict[str, int]]] = []
    for key, (low, high, step) in bounds.items():
        delta = max(1, int(round(step * step_scale)))
        for direction in (1, -1):
            new_value = _clamp(current.get(key, 0) + direction * delta, low, high)
            if new_value == current.get(key, 0):
                continue
            trial = dict(current)
            trial[key] = new_value
            out.append((key, direction, trial))
    return out


#: 一批"直接和全关比"的候选权重。爬山之前先跑这个：如果**没有任何一组**
#: 明显好过全关，那说明"给翻子加成"这个方向在当前评估函数下不成立，
#: 再花几十分钟爬山也是白搭。
BATCH = {
    "小加成": {"base": 15, "phase": 20},
    "小加成+未知": {"base": 15, "phase": 20, "unknown": 30},
    "仅常数": {"base": 30},
    "SEED原值": {"base": 30, "phase": 40},
    "强阶段": {"phase": 80},
    "多翻少削": {"base": 30, "phase": 40, "unknown": 20, "reduction": 2},
    "保守(负)": {"base": -30, "phase": -40},
}


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--games", type=int, default=16,
                        help="每个候选打多少局（建议 ≥12，否则噪声很大）")
    parser.add_argument("--time", type=float, default=0.5, help="每步思考秒数")
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--apply", action="store_true",
                        help="把最优权重写进 ai_config.json")
    parser.add_argument("--quiet", action="store_true",
                        help="也标定 reveal_quiet_weights（默认只标 main）")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行下几局（每局独立引擎实例，不影响结果）")
    parser.add_argument("--compare", action="store_true",
                        help="只做一件小事：SEED 权重 vs 全关，直接看有没有用")
    parser.add_argument("--batch", action="store_true",
                        help="把 BATCH 里几组候选分别和全关比一遍（同一局集）")
    parser.add_argument("--only", default="",
                        help="只跑 BATCH 里名字包含这些关键字的组（逗号分隔）")
    args = parser.parse_args(argv)

    best = dict(SEED)
    best_quiet = dict(SEED_QUIET) if args.quiet else {}
    history: List[dict] = []
    step_scale = 1.0
    t_start = time.monotonic()

    # ``--batch``：把几组候选分别和"全关"比一遍。**同一批开局**（同一个
    # args.seed），所以组与组之间可以横向比较——比逐轮爬山公平也快得多。
    if args.batch:
        keys = list(BATCH)
        if args.only:
            wanted = [w.strip() for w in args.only.split(",") if w.strip()]
            picked = [k for k in keys if any(w in k for w in wanted)]
            if not picked:
                print("--only 没匹配到任何组；可选：%s" % "、".join(keys))
                return 2
            keys = picked
        print("=" * 78)
        print("批测：每组候选 vs 全关（各 %d 局、并行 %d、同一局集 seed=%d）"
              % (args.games, args.workers, args.seed))
        print("=" * 78)
        rows = []
        t_all = time.monotonic()
        for label in keys:
            weights = BATCH[label]
            t0 = time.monotonic()
            wa, wb, dr, depth = play_match(weights, {}, {}, {}, args.games,
                                           args.time, args.max_plies, args.seed,
                                           workers=args.workers)
            low, high = _wilson(wa, wb)
            rate = _score(wa, wb)
            rows.append((rate, low, high, label, wa, wb, dr, depth))
            print("  %-10s 启用 %2d 胜 / 关闭 %2d 胜 / 和 %2d   得分率 %.3f "
                  "[%.2f, %.2f]  深度 %.2f  (%.1f 分)"
                  % (label, wa, wb, dr, rate, low, high, depth,
                     (time.monotonic() - t0) / 60.0))
        print("-" * 78)
        print("总用时 %.1f 分钟" % ((time.monotonic() - t_all) / 60.0))
        rows.sort(reverse=True)
        winners = [r for r in rows if r[1] > 0.5]
        if winners:
            rate, low, high, label, wa, wb, dr, depth = winners[0]
            print("最好的一组：%s（得分率 %.3f，下界 %.2f）" % (label, rate, low))
            print("  权重 = %s" % json.dumps(BATCH[label], ensure_ascii=False))
            print("  → 可以拿它当爬山起点：加 --seed-weights 重跑标定")
        else:
            print("**没有任何一组显著好过全关**（都跨 0.5）。")
            print("  → 说明「给翻子加成」这个方向在现在的评估函数下不成立，")
            print("    保持 reveal_weights 全 0（关闭）才是对的，别再爬山了。")
        return 0

    # ``--compare``：先回答"这组权重到底有没有用"，再决定要不要花几十分钟爬山。
    # 实测教训：不加这一步就埋头爬山，很可能在噪声里爬了半天、结论啥也没有。
    if args.compare:
        print("=" * 74)
        print("对比：SEED 权重 vs 全关（各 %d 局并行 %d）" % (args.games, args.workers))
        print("  SEED = %s" % best)
        print("=" * 74)
        t0 = time.monotonic()
        wa, wb, dr, depth = play_match(best, best_quiet, {}, {}, args.games,
                                       args.time, args.max_plies, args.seed,
                                       workers=args.workers)
        low, high = _wilson(wa, wb)
        print("  启用 %d 胜 / 关闭 %d 胜 / 和 %d   得分率 %.3f [%.2f, %.2f]  深度 %.2f"
              % (wa, wb, dr, _score(wa, wb), low, high, depth))
        print("  用时 %.1f 分钟" % ((time.monotonic() - t0) / 60.0))
        if low > 0.5:
            print("  结论：*** 显著更好 ***（可以写进 ai_config 的 reveal_weights）")
        elif high < 0.5:
            print("  结论：显著更差，保持关闭")
        else:
            print("  结论：分不出高下——多打几局再看，别急着启用")
        return 0

    print("=" * 74)
    print("翻子权重标定：起点 %s" % best)
    if args.quiet:
        print("静着权重：      %s" % best_quiet)
    print("每候选 %d 局，每步 %.2fs，最多 %d 轮"
          % (args.games, args.time, args.rounds))
    print("=" * 74)

    for round_index in range(1, args.rounds + 1):
        print("\n--- 第 %d 轮（步长系数 %.2f）---" % (round_index, step_scale))
        results: List[Tuple[float, int, int, str, Dict[str, int]]] = []
        todos = candidates(best, BOUNDS, step_scale)
        for key, direction, trial in todos:
            t0 = time.monotonic()
            wa, wb, dr, depth = play_match(trial, best_quiet, best, best_quiet,
                                           args.games, args.time,
                                           args.max_plies, args.seed,
                                           workers=args.workers)
            rate = _score(wa, wb)
            low, high = _wilson(wa, wb)
            elapsed = time.monotonic() - t0
            arrow = "+" if direction > 0 else "-"
            print("  %-10s %s%-4d  候选 %d 胜 / 当前 %d 胜 / 和 %d"
                  "   得分率 %.3f [%.2f, %.2f]  深度 %.2f  (%.0fs)"
                  % (key, arrow, trial[key] - best.get(key, 0),
                     wa, wb, dr, rate, low, high, depth, elapsed))
            if rate > 0.5 and low > 0.5:
                # 只有"整个置信区间都在 0.5 以上"才算真的更好
                results.append((rate, low, wa, key, trial))
        if not results:
            print("  → 本轮没有任何候选显著更好，步长减半继续")
            step_scale *= 0.5
            if step_scale < 0.2:
                print("  → 步长已经很小，收工")
                break
            continue
        results.sort(key=lambda x: (x[0], x[1]), reverse=True)
        rate, low, wa, key, trial = results[0]
        print("  → 采纳 %s = %d（得分率 %.3f，下界 %.2f）"
              % (key, trial[key], rate, low))
        best = trial
        history.append({"round": round_index, "key": key, "value": trial[key],
                        "rate": rate, "wl": wa})
        step_scale = max(0.25, step_scale * 0.5)

    total_min = (time.monotonic() - t_start) / 60.0
    print("\n" + "=" * 74)
    print("最优 reveal_weights = %s" % json.dumps(best, ensure_ascii=False))
    if args.quiet:
        print("最优 quiet 权重     = %s" % json.dumps(best_quiet, ensure_ascii=False))
    print("采纳轨迹：%s" % json.dumps(history, ensure_ascii=False))
    print("总用时 %.1f 分钟" % total_min)
    print("=" * 74)

    # 最后做一次"冠军 vs 全关"的收官验证
    print("\n收官验证：最优权重 vs 关闭（各 %d 局）…" % args.games)
    wa, wb, dr, depth = play_match(best, best_quiet, {}, {},
                                   args.games, args.time, args.max_plies,
                                   args.seed + 5000, workers=args.workers)
    low, high = _wilson(wa, wb)
    print("  启用 %d 胜 / 关闭 %d 胜 / 和 %d   得分率 %.3f [%.2f, %.2f]  深度 %.2f"
          % (wa, wb, dr, _score(wa, wb), low, high, depth))
    verdict = ("*** 显著更好，建议启用 ***" if low > 0.5 else
               ("似乎更差，建议保持关闭" if high < 0.5 else
                "分不出高下（区间跨 0.5）——不要启用，多打几局再说"))
    print("  结论：%s" % verdict)

    if args.apply and low > 0.5:
        path = GAME / "ai_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config.setdefault("ai", {})["reveal_weights"] = best
        if args.quiet:
            config["ai"]["reveal_quiet_weights"] = best_quiet
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print("  已写入 %s（重启游戏生效）" % path)
    elif args.apply:
        print("  未写入配置：收官验证没有显著优势，保持默认关闭更安全")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
