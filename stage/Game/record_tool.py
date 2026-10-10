#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""棋谱工具：导出 / 列表 / 画棋盘 / 找失着 / 生成训练样本。

用法（在 ``Game`` 目录下，或任意目录用完整路径）::

    # 1) 把某个棋谱画出来看（棋盘 + 每手记谱）
    python record_tool.py show 20261007-153012-red-vs-ai

    # 2) 列表
    python record_tool.py list

    # 3) 复盘找失着：逐手用引擎复查，按"落差"排序列出（默认 0.5 秒/手）
    python record_tool.py analyse 20261007-153012-red-vs-ai
    python record_tool.py analyse 20261007-153012-red-vs-ai --time 1.0 --samples 2

    # 4) 生成训练样本（JSONL，一行一个局面）
    python record_tool.py train 20261007-153012-red-vs-ai -o train.jsonl
    python record_tool.py train --all -o training-all.jsonl

    # 5) 只导出中文记谱文本
    python record_tool.py text 20261007-153012-red-vs-ai

退出码：0 成功；2 参数/找不到棋谱；3 引擎或文件出错。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from jieqi import record as R  # noqa: E402  (必须在 sys.path 调整之后)

EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_ERROR = 3


def _safe_print(text: str = "") -> None:
    """往控制台打印，遇到编码不支持的字符自动降级。

    Windows 控制台默认 GBK，打不出 "−"（U+2212）之类的字符；如果直接 print
    会抛 ``UnicodeEncodeError``，把一份好端端的棋谱报告变成"出错"。宁可少几个
    好看符号，也不能让工具崩。
    """
    try:
        print(text)
        return
    except Exception:
        pass
    try:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, "replace").decode(encoding, "replace"))
    except Exception:
        try:
            print(text.encode("ascii", "replace").decode("ascii"))
        except Exception:
            pass


def _parse_board(record: dict, ply: int):
    """棋谱里第 ``ply`` 手的棋盘（``ply=0`` 是开局）。"""
    positions = record.get("positions") or []
    if not positions:
        return None
    ply = max(0, min(int(ply), len(positions) - 1))
    return positions[ply].get("board")


def cmd_list(args) -> int:
    items = R.list_records(args.dir)
    if not items:
        print("还没有棋谱。下完一局会自动保存到 %s" % R.record_dir(args.dir))
        return EXIT_OK
    print("棋谱目录：%s" % R.record_dir(args.dir))
    print("%-34s %-8s %5s  %-6s %s" % ("名字", "结果", "手数", "我执", "时间"))
    for item in items:
        print("%-34s %-8s %5d  %-6s %s"
              % (item["name"], item["result"], item["total_plies"],
                 "红" if item["player_side"] == "red" else "黑",
                 item["created"]))
    print("共 %d 份" % len(items))
    return EXIT_OK


def cmd_show(args) -> int:
    record = R.load_record(args.name if os.path.isabs(args.name)
                           else os.path.join(R.record_dir(args.dir), args.name))
    if record is None:
        print("找不到棋谱：%s" % args.name, file=sys.stderr)
        return EXIT_BAD_ARGS
    print(R.record_to_text(record))
    if args.board is not None:
        print("第 %d 手之后的棋盘：" % args.board)
        print(R.board_to_text(_parse_board(record, args.board)))
    return EXIT_OK


def cmd_text(args) -> int:
    return cmd_show(args)


def cmd_analyse(args) -> int:
    record = R.load_record(args.name if os.path.isabs(args.name)
                           else os.path.join(R.record_dir(args.dir), args.name))
    if record is None:
        print("找不到棋谱：%s" % args.name, file=sys.stderr)
        return EXIT_BAD_ARGS
    print("复盘：%s（%d 手）　引擎 %.2fs/手，samples=%d"
          % (record.get("name"), record.get("total_plies", 0),
             args.time, args.samples))
    result = R.analyse_record(record, time_limit=args.time, samples=args.samples,
                              threshold=args.threshold, top=args.top)
    print("已复查 %d 手；落差 >= %d 的有 %d 手（只列出前 %d 手）"
          % (result["checked"], args.threshold, len(result["items"]), args.top))
    if not result["items"]:
        print("没有发现明显失着。")
        return EXIT_OK
    print("")
    print("%4s %-4s %-22s %8s  %-22s %8s %8s"
          % ("手数", "方", "实际着法", "得分", "引擎最佳", "得分", "落差"))
    for item in result["items"]:
        _safe_print("%4d %-4s %-22s %8s  %-22s %8s %8.0f"
                    % (item["ply"], "红" if item["side"] == "red" else "黑",
                       item["played"], _fmt(item["played_score"]),
                       item["best"], _fmt(item["best_score"]), item["loss"]))
    print("")
    _safe_print("提示：得分是「走子方视角」（正数 = 走子方占优），"
                "落差 = 最佳分 - 实际分。")
    print("      暗子身份未知，深度也受时间限制，落差大不代表一定是错棋，")
    print("      但连续几手大落差通常就是崩盘的起点。")
    if args.json:
        _safe_print(json.dumps(result["items"], ensure_ascii=False, indent=1))
    return EXIT_OK


def _fmt(value) -> str:
    return "-" if value is None else "%.0f" % float(value)


def cmd_train(args) -> int:
    names = []
    if args.all:
        names = [item["name"] for item in R.list_records(args.dir)]
    elif args.name:
        names = [args.name]
    if not names:
        print("请给一个棋谱名，或用 --all", file=sys.stderr)
        return EXIT_BAD_ARGS
    out = args.output or os.path.join(R.record_dir(args.dir), "training.jsonl")
    total = 0
    missing = []
    with open(out, "w", encoding="utf-8") as handle:
        for name in names:
            record = R.load_record(os.path.join(R.record_dir(args.dir), name))
            if record is None:
                # 顺手再试一次"当路径/当名字"的写法，仍然读不到才算缺失
                record = R.load_record(name)
            if record is None:
                missing.append(name)
                print("跳过（读不到）：%s" % name, file=sys.stderr)
                continue
            # 逐盘写到同一个文件：先导到临时文件再拼接，避免重复实现格式
            tmp = out + ".part"
            info = R.export_training(record, tmp,
                                     include_hidden_truth=args.with_hidden_truth)
            if not info.get("ok"):
                print("导出失败 %s：%s" % (name, info.get("error")), file=sys.stderr)
                continue
            with open(tmp, "r", encoding="utf-8") as part:
                handle.write(part.read())
            try:
                os.remove(tmp)
            except OSError:
                pass
            total += int(info.get("samples") or 0)
            print("%-34s -> %d 个样本" % (name, info.get("samples") or 0))
    print("共写出 %d 个样本 -> %s" % (total, out))
    if missing:
        print("有 %d 份棋谱读不到：%s（用 list 看看确切名字）"
              % (len(missing), "、".join(missing)), file=sys.stderr)
        return EXIT_BAD_ARGS
    if args.with_hidden_truth:
        print("注意：这次带上了 hidden_truth（暗子真身），只适合事后打标签，"
              "直接训练会让模型学会偷看暗子。")
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record_tool.py",
        description="揭棋棋谱工具：列表 / 查看 / 复盘找失着 / 生成训练样本。")
    parser.add_argument("--dir", default=None,
                        help="棋谱目录（默认 Game/records）")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("list", help="列出全部棋谱")
    p.set_defaults(func=cmd_list)

    for name, help_text in (("show", "打印一份棋谱（含终局棋盘）"),
                            ("text", "同 show")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("name", help="棋谱名（可以带 .json）")
        p.add_argument("--board", type=int, default=None,
                       help="额外打印第 N 手之后的棋盘")
        p.set_defaults(func=cmd_show if name == "show" else cmd_text)

    p = sub.add_parser("analyse", help="逐手复查，找失着")
    p.add_argument("name", help="棋谱名")
    p.add_argument("--time", type=float, default=0.5, help="每手思考秒数")
    p.add_argument("--samples", type=int, default=1, help="采样个数")
    p.add_argument("--threshold", type=int, default=120, help="落差阈值（厘兵）")
    p.add_argument("--top", type=int, default=20, help="最多列几手")
    p.add_argument("--json", action="store_true", help="额外输出 JSON")
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("train", help="生成训练样本 JSONL")
    p.add_argument("name", nargs="?", default=None, help="棋谱名（或用 --all）")
    p.add_argument("--all", action="store_true", help="导出全部棋谱")
    p.add_argument("-o", "--output", default=None, help="输出文件（默认 records/training.jsonl）")
    p.add_argument("--with-hidden-truth", action="store_true",
                   help="带上暗子真身（慎用，只适合事后打标签）")
    p.set_defaults(func=cmd_train)
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_BAD_ARGS
    try:
        return int(args.func(args) or EXIT_OK)
    except KeyboardInterrupt:
        return EXIT_ERROR
    except Exception as exc:      # 工具而已，出错给中文说明即可
        print("出错：%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
