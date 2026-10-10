#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""揭棋大模型顾问命令行工具（可选功能，默认关闭）。

用法示例（可在任意工作目录下运行）::

    python Game/llm_advisor.py --demo --dry-run
    python Game/llm_advisor.py --demo --json --dry-run
    python Game/llm_advisor.py --demo
    python Game/llm_advisor.py --config D:/path/ai_config.json --style review --demo

退出码::

    0  成功
    2  配置缺失 / 非法
    3  网络或接口错误
    4  未能从回复中解析出着法
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 允许从任意 cwd 运行：把 stage\Game 目录塞进 sys.path，使 `import jieqi` 生效。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from jieqi.llm import (  # noqa: E402  (必须在 sys.path 调整之后导入)
    LLMAdvisor,
    load_config,
    parse_response,
    render_position_text,
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_NETWORK = 3
EXIT_NO_MOVE = 4

#: 内置示例快照：让 CLI 在没有任何对局运行时也能演示 / 自检。
DEMO_SNAPSHOT = {
    "side_to_move": "black",
    "ai_color": "black",
    "move_number": 17,
    "last_move": {"from": [3, 4], "to": [4, 4], "piece": "红兵", "captured": "黑卒"},
    "board_text": (
        "     １ ２ ３ ４ ５ ６ ７ ８ ９\n"
        " 9 | 車 馬 相 士 將 士 相 馬 車\n"
        " 8 | ． ． ． ． 暗 ． ． ． ．\n"
        " 7 | ． 炮 ． ． ． ． ． 炮 ．\n"
        " 6 | 卒 ． 卒 ． 紅卒 ． 卒 ． 卒\n"
        " 5 | ． ． ． ． ． ． ． ． ．\n"
        " 4 | ． ． ． ． 卒 ． ． ． ．\n"
        " 3 | 兵 ． 兵 ． ． ． 兵 ． 兵\n"
        " 2 | ． 炮 ． ． ． ． ． 炮 ．\n"
        " 1 | 車 馬 相 士 帥 士 相 馬 車\n"
        " 0 | ． ． ． ． ． ． ． ． ．\n"
    ),
    "board": [
        [
            None,
            {"side": "red", "hidden": True, "home": "炮", "ident": None},
            None,
            None,
            None,
            None,
            None,
            {"side": "black", "hidden": False, "home": "馬", "ident": "黑馬"},
            None,
        ],
        [
            None,
            None,
            None,
            None,
            {"side": "black", "hidden": False, "home": "卒", "ident": "黑卒"},
            None,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            {"side": "red", "hidden": False, "home": "兵", "ident": "红兵"},
            None,
            None,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            None,
            {"side": "black", "hidden": True, "home": "車", "ident": None},
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    ],
    "engine_candidates": [
        {
            "from": [4, 2],
            "to": [4, 4],
            "score": 18,
            "winrate": 0.54,
            "pv": ["(4,2)->(4,4)", "(6,4)->(5,4)", "(7,1)->(7,4)"],
            "note": "炮二平五",
        },
        {
            "from": [7, 1],
            "to": [7, 4],
            "score": 12,
            "winrate": 0.52,
            "pv": ["(7,1)->(7,4)"],
            "note": "炮八平五",
        },
        {
            "from": [0, 6],
            "to": [2, 5],
            "score": -6,
            "winrate": 0.48,
            "pv": ["(0,6)->(2,5)"],
            "note": "相三进五",
        },
    ],
    "engine_depth": 6,
    "engine_nodes": 41000,
    "engine_samples": 8,
    "engine_verdict": "红方稍优",
    "repetition_warning": False,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_advisor.py",
        description="揭棋大模型顾问（可选、默认关闭；缺少网络或 API Key 时不会影响对局）。",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="配置文件路径，默认按 jieqi.llm._default_config_path() 的顺序解析。",
    )
    parser.add_argument(
        "--style",
        default=None,
        choices=["explain", "suggest", "review"],
        help="输出风格：explain 讲解 / suggest 只给着法 / review 复盘。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印脱敏配置、局面文本与请求体，不发起任何网络请求。",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="使用内置示例局面（无需运行对局即可体验）。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以机器可读的 JSON 输出结果。",
    )
    return parser


def _print_json(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _print_human(result, style: str) -> None:
    print("=== 大模型顾问结果 ===")
    print("风格：%s" % style)
    print("状态：%s" % ("成功" if result.ok else "失败"))
    if result.error:
        print("错误：%s" % result.error)
    move_text = result.move_text()
    if move_text:
        print("推荐着法：%s" % move_text)
    if result.comment:
        print("点评：%s" % result.comment)
    if result.confidence is not None:
        print("置信度：%.2f" % result.confidence)
    if result.tags:
        print("标签：%s" % "、".join(result.tags))
    print("耗时：%.3f 秒　模型：%s" % (result.latency, result.model or "(未知)"))
    if result.usage:
        print("用量：%s" % json.dumps(result.usage, ensure_ascii=False))
    if result.text:
        print("--- 原始回复 ---")
        print(result.text)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    cfg = load_config(args.config)

    # 配置若完全无法加载（文件存在但读不了 / JSON 坏了），load_config 会在 warnings
    # 里说明；这里把"路径不存在"也当作提示而非致命错误，因为默认配置仍可 dry-run。
    for warning in cfg.warnings:
        print("[配置警告] %s" % warning, file=sys.stderr)

    style = args.style or cfg.mode or "explain"

    # 读取失败且用户显式指定了路径 → 视为配置错误。
    if args.config and not os.path.isfile(os.path.abspath(os.path.expanduser(args.config))):
        print("[错误] 指定的配置文件不存在：%s" % args.config, file=sys.stderr)
        return EXIT_CONFIG

    advisor = LLMAdvisor(cfg)
    snapshot = DEMO_SNAPSHOT if args.demo else {}

    # ---------------- dry-run ----------------
    if args.dry_run:
        payload = advisor.build_payload(snapshot, style)
        if args.json:
            _print_json(
                {
                    "dry_run": True,
                    "demo": bool(args.demo),
                    "style": style,
                    "endpoint": advisor.endpoint(),
                    "config": cfg.redacted().to_dict(redact=True),
                    "position_text": render_position_text(snapshot),
                    "payload": payload,
                }
            )
        else:
            print("=== 配置（密钥已脱敏） ===")
            print(json.dumps(cfg.redacted().to_dict(redact=True), ensure_ascii=False, indent=2))
            print("=== 局面文本 ===")
            print(render_position_text(snapshot))
            print("=== 请求地址 ===")
            print(advisor.endpoint())
            print("=== 请求体（不会发送） ===")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    # ---------------- 真实调用 ----------------
    if not advisor.available():
        print(
            "[错误] 大模型顾问不可用：请设置 llm.enabled=true 并提供 API Key"
            "（llm.api_key 或环境变量 %s）。" % cfg.api_key_env,
            file=sys.stderr,
        )
        return EXIT_CONFIG

    result = advisor.suggest(snapshot, style)

    if args.json:
        _print_json({"dry_run": False, "style": style, "result": result.to_dict()})
    else:
        _print_human(result, style)

    if not result.ok:
        return EXIT_NETWORK
    if result.move is None and parse_response(result.text)["move"] is None:
        return EXIT_NO_MOVE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
