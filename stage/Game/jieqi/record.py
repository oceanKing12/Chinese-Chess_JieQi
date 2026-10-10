# -*- coding: utf-8 -*-
"""棋谱记录 / 导出 / 复盘（给"让 AI 学棋谱"做数据底座）。

设计目标
--------
1. **一次对局 = 一个自包含的 JSON**：里面的 ``positions[0..n]`` 是开局到终局的
   全部局面快照，**逐手都在**，所以哪怕只拿到这一个文件，也能完整重放整盘棋、
   重新分析每一步（不需要当时的进程状态）。
2. **不泄漏暗子真身**：写盘时，未翻开的暗子只写它的摆位身份（``home``）和
   ``hidden=True``，**绝不写 ``actual``**——否则这份棋谱就成了"AI 作弊资料"。
   翻开的明子写真实身份（对双方本来就可见）。
3. **训练样本一行一条**：``export_training`` 输出 JSONL，每行是一个"局面 ->
   实际着法 + 引擎评估 + 最终胜负"的样本，后续接训练脚本不用再清洗。
4. **出问题不影响下棋**：所有函数对缺字段/坏数据一律宽容，绝不抛异常；
   ``save`` 失败只返回错误说明。

文件布局（``Game/records/``）::

    20260817-153012-red-vs-ai.json   一盘棋
    20260817-153012-red-vs-ai.txt    同一盘棋的中文记谱（给人看）
    index.json                       全部棋谱的索引（列表页用）
    training-20260817.jsonl          训练样本（由 CLI 生成）

对外主入口
----------
* :func:`build_record` —— 由 GameState 组装棋谱字典（纯函数，可单测）
* :func:`save_record` / :func:`load_record` / :func:`record_path`
* :func:`record_to_text` —— 中文记谱文本
* :func:`analyse_record` —— 逐手用引擎复查，找出"实际着法 vs 最佳着法"的落差
* :func:`export_training` —— 训练样本 JSONL
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .consts import BLACK, RED, ROWS, COLS

__all__ = [
    "RECORD_VERSION",
    "DEFAULT_RECORD_DIR",
    "record_dir",
    "new_record_name",
    "unique_record_name",
    "build_record",
    "save_record",
    "save_text",
    "load_record",
    "list_records",
    "record_to_text",
    "board_to_text",
    "board_to_codes",
    "replay_plan",
    "export_training",
    "analyse_record",
    "describe_record",
    "RATINGS",
    "RATING_BY_CODE",
    "RATING_BY_KEY",
    "rating_label",
    "rating_good",
    "valid_rating",
]

#: 棋谱格式版本（以后加字段时用它区分）。
RECORD_VERSION = 1

#: 默认棋谱目录：``<Game 包目录>/records``。
DEFAULT_RECORD_DIR = "records"

_SIDE_CN = {RED: "红", BLACK: "黑"}
_SIDE_EN = {RED: "red", BLACK: "black"}

# ---------------------------------------------------------------------------
# 人工打分（五档中文等级）
# ---------------------------------------------------------------------------
#
# 为什么是五档而不是"让用户填个数"：
#   * 界面上一键完成（键盘 5~9），几百手也标得动；
#   * 与棋界通用的标注法一致，你自己看棋谱时一眼就懂；
#   * 同时给出 ``good``（+1/0/−1）用于训练加权，**并保留原始档位**——
#     五档的信息量大于三档，将来换损失函数不用重新标一遍。
#
# 顺序即强度：好棋 > 正着 > 疑问手 > 错着 > 漏着。
RATINGS: "List[Dict[str, Any]]" = [
    {"code": "best", "label": "好棋", "good": 1, "key": "5"},
    {"code": "good", "label": "正着", "good": 1, "key": "6"},
    {"code": "inaccuracy", "label": "疑问手", "good": 0, "key": "7"},
    {"code": "mistake", "label": "错着", "good": -1, "key": "8"},
    {"code": "blunder", "label": "漏着", "good": -1, "key": "9"},
]

#: 档位代码 -> 档位字典（查表用）
RATING_BY_CODE: Dict[str, Dict[str, Any]] = {r["code"]: r for r in RATINGS}
#: 键盘按键（"5"…"9"）-> 档位代码；"0" 是"清除"
RATING_BY_KEY: Dict[str, str] = {r["key"]: r["code"] for r in RATINGS}


def rating_label(code: str) -> str:
    """档位代码 -> 中文标签（未知代码返回空串，绝不抛异常）。"""
    item = RATING_BY_CODE.get(str(code or "").strip())
    return str(item["label"]) if item else ""


def rating_good(code: str) -> Optional[int]:
    """档位代码 -> 训练用数值（+1 好 / 0 中性 / −1 差）；未知返回 ``None``。"""
    item = RATING_BY_CODE.get(str(code or "").strip())
    return int(item["good"]) if item else None


def valid_rating(code: str) -> bool:
    """这个代码是不是合法档位（写盘前用它挡掉脏数据）。"""
    return str(code or "").strip() in RATING_BY_CODE


# ---------------------------------------------------------------------------
# 目录与文件名
# ---------------------------------------------------------------------------


def record_dir(base: Optional[str] = None) -> str:
    """棋谱目录（不存在就建）。默认 ``<Game 包目录>/records``。"""
    if base:
        path = os.path.abspath(os.path.expanduser(str(base)))
    else:
        path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         DEFAULT_RECORD_DIR))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


def new_record_name(player_side: str = "red", stamp: Optional[float] = None,
                    suffix: str = "") -> str:
    """``20260817-153012-red-vs-ai`` 这样的棋谱名（不含扩展名）。"""
    when = time.localtime(stamp if stamp is not None else time.time())
    side = "black" if str(player_side).lower() == "black" else "red"
    name = "%s-%s-vs-ai" % (time.strftime("%Y%m%d-%H%M%S", when), side)
    if suffix:
        safe = re.sub(r"[^0-9A-Za-z_-]+", "-", str(suffix)).strip("-")
        if safe:
            name += "-" + safe
    return name


def _safe_name(name: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "-", str(name or "").strip())
    return text or new_record_name()


#: 同名防撞时最多往后试多少个后缀（``-2`` … ``-999``）
_MAX_UNIQUE_TRIES = 999


def unique_record_name(name: str, folder: Optional[str] = None,
                       mine: str = "") -> str:
    """给 ``name`` 找一个**不会覆盖**的棋谱名。

    为什么必须有这个：文件名只精确到秒（``new_record_name``），而
    ``save_record`` 原本直接 ``open(path, "w")``——两局棋在同一秒结束、
    或者"导入棋谱后接着下"，就会把已有的棋谱**悄悄覆盖掉**。

    判定时 **json 和 txt 一起看**：这两个文件是一对，只判 json 会出现
    "json 是新名字、txt 把旧的盖了"的错配。

    ``mine`` 是"这一局自己已经写出过的文件名"——它**不算被占用**，
    于是同一局重复导出会更新自己那份，而不是产生 ``-2`` 垃圾。
    """
    folder = record_dir(folder)
    base = _safe_name(name)
    already_mine = bool(mine) and base == _safe_name(mine)
    if already_mine or not _taken(folder, base):
        return base
    for index in range(2, _MAX_UNIQUE_TRIES + 1):
        candidate = "%s-%d" % (base, index)
        if not _taken(folder, candidate):
            return candidate
    # 极端情况下（上千个同名）退回带时间戳的名字，总比覆盖强
    return new_record_name(suffix=time.strftime("%H%M%S"))


def _taken(folder: str, base: str) -> bool:
    """这个名字（json 或 txt）已经被占了吗？"""
    return (os.path.exists(os.path.join(folder, base + ".json"))
            or os.path.exists(os.path.join(folder, base + ".txt")))


# ---------------------------------------------------------------------------
# 局面序列化（**注意：暗子不写真身**）
# ---------------------------------------------------------------------------


def _piece_cell(piece: Optional[dict], reveal_hidden: bool = False) -> Optional[dict]:
    """把一枚 GUI 棋子写成棋谱里的格子。

    ``reveal_hidden=False``（默认、也是写盘时的取值）时，暗子只写摆位身份；
    传 ``True`` 才会带上真实兵种——**只有本地研究/训练标签才需要**。
    """
    if not piece:
        return None
    hidden = not bool(piece.get("revealed"))
    cell: Dict[str, Any] = {
        "side": _SIDE_EN.get(0 if str(piece.get("color")) == "red" else 1, "red"),
        "hidden": hidden,
        "home": str(piece.get("original") or ""),
    }
    if hidden:
        if reveal_hidden:
            cell["actual"] = str(piece.get("actual") or "")
    else:
        cell["ident"] = str(piece.get("actual") or "")
    return cell


def snapshot_board(board, reveal_hidden: bool = False) -> Optional[List[List[Any]]]:
    """GUI 棋盘 -> 10×9 的可 JSON 化快照（None 表示空格）。"""
    if not board:
        return None
    out: List[List[Any]] = []
    for r in range(ROWS):
        row = board[r] if r < len(board) else None
        cells = []
        for c in range(COLS):
            piece = row[c] if row and c < len(row) else None
            cells.append(_piece_cell(piece, reveal_hidden))
        out.append(cells)
    return out


def board_to_codes(board) -> List[str]:
    """棋盘 -> 10 行字符串（每格 2~3 字符），用于 JSONL 训练样本。

    ``R`` 红 / ``B`` 黑 / ``.`` 空；暗子在兵种前加 ``*``（如 ``B卒*``）；
    明子直接写兵种（如 ``R车``）。这种写法紧凑、无歧义，方便以后喂模型。
    """
    rows: List[str] = []
    for r in range(ROWS):
        row = board[r] if board and r < len(board) else None
        cells: List[str] = []
        for c in range(COLS):
            piece = row[c] if row and c < len(row) else None
            if not piece:
                cells.append("..")
                continue
            mark = "R" if str(piece.get("color")) == "red" else "B"
            ch = str(piece.get("actual") if piece.get("revealed")
                     else piece.get("original") or "?")
            cells.append("%s%s%s" % (mark, ch, "*" if not piece.get("revealed") else ""))
        rows.append(" ".join(cells))
    return rows


def board_to_text(board) -> str:
    """棋盘 -> 给人看的中文棋盘（坐标口径与提示词 / 棋谱 / 引擎一致）。

    列号用字母 ``a``~``i``（左到右）、行号用数字 ``10``~``1``（自上而下，
    因为红方在下、第 1 行是红方底线）。
    """
    from .notation import FILE_LETTERS

    lines = ["     " + " ".join(FILE_LETTERS)]
    # 行号自上而下打印（9 -> 0），因为内部 row 0 画在最上面；rank 0 是红方底线。
    for rank in range(ROWS - 1, -1, -1):
        r = ROWS - 1 - rank
        row = board[r] if board and r < len(board) else None
        cells: List[str] = []
        for c in range(COLS):
            piece = row[c] if row and c < len(row) else None
            if not piece:
                cells.append(" · ")
                continue
            ch = str(piece.get("actual") if piece.get("revealed")
                     else piece.get("original") or "?")
            cells.append("%s%s" % (ch, "?" if not piece.get("revealed") else " "))
        lines.append("行%-2d  %s" % (rank, " ".join(cells)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 组装棋谱
# ---------------------------------------------------------------------------


def _move_entry(rec: dict, index: int, evals: Sequence[str] = (),
                comments: Sequence[str] = (),
                side: Optional[str] = None,
                scores: Sequence[str] = (),
                notes: Sequence[str] = ()) -> Dict[str, Any]:
    """一条走棋记录 -> 棋谱里的着法条目（含翻子结果、吃子、点评、人工打分）。

    ``side`` 是"这一步是谁走的"：优先用记录里存的 ``side``，其次用调用方给的
    （按手数奇偶推：红先），最后才退回棋子颜色。之所以要显式传，是因为
    ``move_history`` 存的是**落子前**的快照，正常流程里棋子颜色就是走子方颜色，
    但将来若有人拿"替对方走"的记录来组装棋谱，按奇偶推才不会写错归属。
    """
    piece = rec.get("piece") or {}
    stored = rec.get("side")
    if stored in (0, 1):
        side = _SIDE_EN[int(stored)]
    elif stored in ("red", "black"):
        side = str(stored)
    if side not in ("red", "black"):
        side = "red" if str(piece.get("color")) == "red" else "black"
    try:
        (sr, sc), (er, ec) = rec["start"], rec["end"]
        sr, sc, er, ec = int(sr), int(sc), int(er), int(ec)
    except Exception:
        sr = sc = er = ec = -1
    #: 坐标用 ``"b3"`` 写法（列字母 + 行数字，红方底线是第 1 行）。
    #: 与走棋记录、提示词的着法**逐字一致**，不再出现两套坐标。
    entry: Dict[str, Any] = {
        "index": int(index),
        "side": side,
        "from": _coord((sr, sc)) if sr >= 0 else None,
        "to": _coord((er, ec)) if er >= 0 else None,
        "hidden": not bool(piece.get("revealed")),
        "home": str(piece.get("original") or ""),
    }
    if piece.get("revealed"):
        entry["piece"] = str(piece.get("actual") or "")
    revealed_as = rec.get("revealed_as")
    if revealed_as:
        entry["revealed_as"] = str(revealed_as)
    cap = rec.get("captured")
    if cap:
        cap_hidden = not bool(cap.get("revealed"))
        entry["captured"] = {
            "side": "red" if str(cap.get("color")) == "red" else "black",
            "hidden": cap_hidden,
            "home": str(cap.get("original") or ""),
        }
        # 记下"被吃在哪个格子"：暗子的摆位身份要靠它反推（暗卒/暗兵的
        # 行棋方向不同，只有知道它原来在哪一排才能正确还原规则身份）。
        where = rec.get("captured_sq") or rec.get("end")
        if where:
            try:
                entry["captured"]["square"] = _coord((int(where[0]),
                                                      int(where[1])))
            except (TypeError, ValueError, IndexError):
                pass
        if not cap_hidden:
            entry["captured"]["piece"] = str(cap.get("actual") or "")
        # 被吃的暗子**不写 actual**（规则上双方都不得翻看）
    # 点评/评估都是**按局面**存的，并且"挂在这一手之后"：
    # ``ply_comments[k]`` / ``ply_evals[k]`` 描述的是"第 k 手走完之后的局面"。
    # 所以第 index 手的走子方看到的是第 (index-1) 手之后的局面，取 slot=index-1。
    slot = index - 1
    if 0 <= slot < len(evals) and evals[slot]:
        entry["eval"] = str(evals[slot])
    if 0 <= slot < len(comments) and comments[slot]:
        entry["comment"] = str(comments[slot])
    # 人工打分同样挂在 slot=index-1（"走这一步之前的局面"），与 eval/comment
    # 完全同规则——这样"站在第 k 手打分"就是在给第 k+1 手打分，不会错位。
    human: Dict[str, Any] = {}
    if 0 <= slot < len(scores):
        code = str(scores[slot] or "").strip()
        if valid_rating(code):
            human["rating"] = code
            human["label"] = rating_label(code)
            human["rating_good"] = rating_good(code)
    if 0 <= slot < len(notes) and notes[slot]:
        human["note"] = str(notes[slot])[:500]
    if human:
        entry["human"] = human
    return entry


def _result_of(gs) -> str:
    if not getattr(gs, "game_over", False):
        return "*"
    winner = getattr(gs, "winner", None)
    if winner == "draw":
        return "1/2-1/2"
    if winner == "red":
        return "1-0"
    if winner == "black":
        return "0-1"
    return "*"


def build_record(gs, meta: Optional[dict] = None,
                 reveal_hidden: bool = False) -> Dict[str, Any]:
    """由 :class:`GameState` 组装一份完整棋谱。

    ``meta`` 可以放 ``difficulty`` / ``engine`` / ``notes`` 等附加信息；
    ``result``（最后一次搜索结果）可以放在 ``meta["engine"]`` 里一起存下来。
    """
    meta = dict(meta or {})
    positions = getattr(gs, "positions", None) or []
    history = getattr(gs, "move_history", None) or []
    evals = list(getattr(gs, "ply_evals", None) or [])
    comments = list(getattr(gs, "ply_comments", None) or [])
    scores = list(getattr(gs, "ply_scores", None) or [])
    notes = list(getattr(gs, "ply_notes", None) or [])

    moves = [_move_entry(rec, i + 1, evals, comments,
                         side="red" if i % 2 == 0 else "black",
                         scores=scores, notes=notes)
             for i, rec in enumerate(history)]

    snapshots: List[Dict[str, Any]] = []
    for k, state in enumerate(positions):
        try:
            board = state.get("board")
            snapshots.append({
                "ply": k,
                "current_player": str(state.get("current_player") or "red"),
                "in_check": bool(state.get("in_check")),
                "capture_count": int(state.get("capture_count") or 0),
                "board": snapshot_board(board, reveal_hidden=reveal_hidden),
            })
        except Exception:
            continue

    record = {
        "version": RECORD_VERSION,
        "kind": "jieqi-game-record",
        "name": str(meta.get("name") or ""),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_ts": time.time(),
        "meta": {
            "player_side": str(getattr(gs, "player_side", "red") or "red"),
            "ai_color": str(getattr(gs, "ai_color", "") or ""),
            "ai_mode": bool(getattr(gs, "ai_mode", False)),
            "difficulty": str(meta.get("difficulty") or ""),
            "engine": meta.get("engine") or {},
            "notes": str(meta.get("notes") or ""),
            "reveal_hidden": bool(reveal_hidden),
        },
        "result": _result_of(gs),
        "winner": str(getattr(gs, "winner", "") or ""),
        "draw_reason": str(getattr(gs, "draw_reason", "") or ""),
        "total_plies": len(moves),
        "moves": moves,
        "positions": snapshots,
    }
    # 分支棋谱（``variations``）功能已**整体移除**：只写主线。
    # 老棋谱里如果还留着 ``variations`` 字段，导入时会被直接忽略（不报错）。
    return record


def save_record(record: Dict[str, Any], directory: Optional[str] = None,
                name: Optional[str] = None,
                also_text: bool = True,
                overwrite: bool = False,
                unique: bool = False,
                mine: str = "") -> Dict[str, Any]:
    """把棋谱写进 ``records/``；返回 ``{ok, path, text_path, requested, error}``。

    **默认绝不覆盖已有棋谱**（``overwrite=False``）：

    * ``unique=False``：优先沿用请求的名字；若该名字已被占用，自动加
      ``-2`` / ``-3`` 后缀（见 :func:`unique_record_name`）。
      传 ``mine``（"这一局自己已经写出过的名字"）时它**不算被占用**，
      于是"同一局重复导出"会更新自己那份、不产生 ``-2`` 垃圾。
    * ``unique=True``：**一定**开新文件（终局自动保存用这个，保证导入的
      棋谱永远不会被续下的一局改写）。
    * ``overwrite=True``：退回旧行为（直接用请求的名字覆盖），只给明确
      知道后果的调用方用。

    绝不抛异常：磁盘满 / 没权限 / 数据不可序列化都只返回 ``ok=False`` + 中文原因。
    """
    folder = record_dir(directory)
    requested = _safe_name(
        name or record.get("name")
        or new_record_name(record.get("meta", {}).get("player_side", "red")))
    if overwrite:
        chosen = requested
    elif unique:
        # 一定开新文件：连"这一局自己写过的"也不算数，直接找一个空闲名
        chosen = unique_record_name(requested, folder)
    else:
        chosen = unique_record_name(requested, folder, mine=mine)
    record = dict(record or {})
    record["name"] = chosen
    path = os.path.join(folder, chosen + ".json")
    text_path = os.path.join(folder, chosen + ".txt")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=1)
    except (OSError, TypeError, ValueError) as exc:
        return {"ok": False, "path": path, "text_path": "",
                "requested": requested, "error": str(exc)}
    written_text = ""
    if also_text:
        try:
            with open(text_path, "w", encoding="utf-8") as handle:
                handle.write(record_to_text(record))
            written_text = text_path
        except OSError:
            written_text = ""
    _touch_index(folder, record, path)
    return {"ok": True, "path": path, "text_path": written_text,
            "requested": requested, "error": ""}


def save_text(record: Dict[str, Any], directory: Optional[str] = None,
              name: Optional[str] = None) -> str:
    """只写中文记谱文本；返回写出的路径（失败返回空串）。

    同样**默认不覆盖**——单独调用它时也不该把别人的记谱盖掉。
    """
    folder = record_dir(directory)
    chosen = unique_record_name(
        _safe_name(name or record.get("name") or new_record_name()), folder)
    path = os.path.join(folder, chosen + ".txt")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(record_to_text(record))
        return path
    except OSError:
        return ""


def _touch_index(folder: str, record: Dict[str, Any], path: str) -> None:
    """维护 ``index.json``（列表页/学习脚本用；坏了也不影响棋谱本身）。"""
    index_path = os.path.join(folder, "index.json")
    items: List[Dict[str, Any]] = []
    try:
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                items = [item for item in loaded if isinstance(item, dict)]
    except (OSError, ValueError):
        items = []
    entry = {
        "name": record.get("name") or os.path.splitext(os.path.basename(path))[0],
        "file": os.path.basename(path),
        "created": record.get("created", ""),
        "result": record.get("result", "*"),
        "total_plies": record.get("total_plies", 0),
        "player_side": (record.get("meta") or {}).get("player_side", ""),
        "difficulty": (record.get("meta") or {}).get("difficulty", ""),
    }
    items = [item for item in items if item.get("name") != entry["name"]]
    items.append(entry)
    items.sort(key=lambda item: str(item.get("created") or ""), reverse=True)
    try:
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(items, handle, ensure_ascii=False, indent=1)
    except OSError:
        pass


def load_record(path: str) -> Optional[Dict[str, Any]]:
    """读一份棋谱（``path`` 可以是名字、相对路径或绝对路径）。

    还会自动补齐 ``.json`` 后缀，以及到 ``records/`` 目录里找同名文件——
    这几种写法都要能用，否则 CLI 和脚本里很容易出现"文件明明在却读不到"。
    """
    text = str(path)
    folder = record_dir()
    candidates = [text]
    if not text.endswith(".json"):
        candidates.append(text + ".json")
    if not os.path.isabs(text):
        # 相对路径：既可能相对当前工作目录，也可能相对 records/ 目录
        candidates.append(os.path.abspath(text))
        if not text.endswith(".json"):
            candidates.append(os.path.abspath(text) + ".json")
        candidates.append(os.path.join(folder, text))
        if not text.endswith(".json"):
            candidates.append(os.path.join(folder, text + ".json"))
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        try:
            if os.path.isfile(item):
                with open(item, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    data.setdefault("name",
                                    os.path.splitext(os.path.basename(item))[0])
                    return data
        except (OSError, ValueError):
            continue
    return None


def list_records(directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """列出全部棋谱（按时间倒序）。读不动的文件会被跳过。"""
    folder = record_dir(directory)
    out: List[Dict[str, Any]] = []
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json") or name == "index.json":
            continue
        record = load_record(os.path.join(folder, name))
        if record is None:
            continue
        meta = record.get("meta") or {}
        out.append({
            "name": record.get("name") or name[:-5],
            "file": name,
            "created": record.get("created", ""),
            "created_ts": record.get("created_ts", 0),
            "result": record.get("result", "*"),
            "total_plies": record.get("total_plies", 0),
            "player_side": meta.get("player_side", ""),
            "difficulty": meta.get("difficulty", ""),
            "winner": record.get("winner", ""),
        })
    out.sort(key=lambda item: str(item.get("created") or ""), reverse=True)
    return out


def describe_record(record: Dict[str, Any]) -> str:
    """一行摘要，CLI ``list`` 用。"""
    meta = record.get("meta") or {}
    return "%s  共 %d 手  结果 %s  我执%s  难度%s" % (
        record.get("name") or "?",
        int(record.get("total_plies") or 0),
        record.get("result") or "*",
        "红" if meta.get("player_side") == "red" else "黑",
        meta.get("difficulty") or "-",
    )


# ---------------------------------------------------------------------------
# 中文记谱
# ---------------------------------------------------------------------------

_FILE_CN = "一二三四五六七八九"


def _coord(pos: Any) -> str:
    """``[row, col]`` / ``"b3"`` -> ``"b3"``（列字母 + 行数字）。

    宽容三种输入：``[6, 1]``（棋谱里的写法）、``"b3"``（新记法）、
    ``"(6,1)"``（老记法）。读得回来才不会因为换了记法把老棋谱变成废纸。
    """
    from .notation import name_to_square, square_name

    if isinstance(pos, str):
        got = name_to_square(pos)
        if got is not None:
            return square_name(*got)
    try:
        return square_name(int(pos[0]), int(pos[1]))
    except Exception:
        return "??"


def _move_coords(move: Any):
    """一条着法条目 -> ``(sr, sc, er, ec)``；读不出来返回 ``None``。

    宽容新旧两种记法：``{"from": "b3", "to": "b4"}``（新）
    与 ``{"from": [6, 1], "to": [5, 1]}``（2026-10 之前的老棋谱）。
    """
    from .notation import name_to_square

    if not isinstance(move, dict):
        return None
    out = []
    for key in ("from", "to"):
        value = move.get(key)
        pos = None
        if isinstance(value, str):
            pos = name_to_square(value)
            if pos is None:
                import re

                nums = re.findall(r"\d+", value)
                if len(nums) >= 2:
                    pos = (int(nums[0]), int(nums[1]))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                pos = (int(value[0]), int(value[1]))
            except (TypeError, ValueError):
                pos = None
        if pos is None:
            return None
        out.extend(pos)
    return tuple(out)


def move_to_chinese(move: Dict[str, Any], board_before=None) -> str:
    """一条着法 -> 中文记谱（``红 暗炮(2,1)->(4,1) 翻出车`` 这种可读写法）。

    刻意不做"炮二平五"式的传统记谱：揭棋里暗子身份未知，传统记谱会把暗子
    当成明子写，反而容易误读。这里用"归属 + 兵种（暗子带问号）+ 坐标"。
    """
    side = _SIDE_CN.get(0 if move.get("side") == "red" else 1, "?")
    hidden = bool(move.get("hidden"))
    home = str(move.get("home") or "?")
    piece = ("暗" + home + "?") if hidden else str(move.get("piece") or home)
    text = "%s%s %s->%s" % (side, piece, _coord(move.get("from")),
                            _coord(move.get("to")))
    if move.get("revealed_as"):
        text += "（翻开后是 %s%s）" % (
            side, move.get("revealed_as"))
    cap = move.get("captured")
    if isinstance(cap, dict):
        cap_side = _SIDE_CN.get(0 if cap.get("side") == "red" else 1, "?")
        if cap.get("hidden"):
            text += " 吃%s暗子(未知)" % cap_side
        else:
            text += " 吃%s%s" % (cap_side, cap.get("piece") or cap.get("home") or "?")
    return text


def record_to_text(record: Dict[str, Any]) -> str:
    """棋谱 -> 完整中文文本（含逐手记录、终局棋盘、双方损失、点评）。"""
    record = record or {}
    meta = record.get("meta") or {}
    lines: List[str] = []
    lines.append("揭棋棋谱　%s" % (record.get("name") or ""))
    lines.append("生成时间：%s" % record.get("created", ""))
    lines.append("我执：%s　AI：%s　难度：%s　引擎：%s"
                 % ("红" if meta.get("player_side") == "red" else "黑",
                    meta.get("ai_color") or "-",
                    meta.get("difficulty") or "-",
                    json.dumps(meta.get("engine") or {}, ensure_ascii=False)
                    if meta.get("engine") else "-"))
    lines.append("结果：%s%s" % (record.get("result", "*"),
                                ("（%s）" % record.get("draw_reason"))
                                if record.get("draw_reason") else ""))
    lines.append("总手数：%s" % record.get("total_plies", 0))
    lines.append("")

    engine = meta.get("engine") or {}
    if engine:
        lines.append("【每步引擎读数（红方视角，正数=红方好）】")
        for key, label in (("score", "终局评估"), ("depth", "深度"),
                           ("nodes", "节点"), ("samples", "采样")):
            if engine.get(key) not in (None, ""):
                lines.append("  %s：%s" % (label, engine[key]))
        if engine.get("best_move"):
            lines.append("  末手最佳：%s" % engine["best_move"])
        lines.append("")

    lines.append("【棋谱】")
    for move in record.get("moves") or []:
        text = "%3s. %s" % (move.get("index", "?"), move_to_chinese(move))
        extra = []
        if move.get("eval"):
            extra.append(str(move["eval"]).replace("\n", "　"))
        line = text
        if extra:
            line += "　｜ " + "　".join(extra)
        lines.append(line)
        if move.get("comment"):
            comment = str(move["comment"]).strip().replace("\n", " ")
            lines.append("      点评：%s" % comment)
    lines.append("")

    positions = record.get("positions") or []
    if positions:
        last = positions[-1]
        lines.append("【终局棋盘】")
        lines.append(board_to_text(last.get("board")))
        lines.append("")

    lines.append("【被吃之子】")
    lines.append("  红方损失：%s" % _loss_text(record, "red"))
    lines.append("  黑方损失：%s" % _loss_text(record, "black"))
    lines.append("")
    lines.append("说明：暗子在被吃掉时身份对双方都未知，所以只记数量，不记兵种。")
    return "\n".join(lines) + "\n"


def _loss_text(record: Dict[str, Any], side: str) -> str:
    pieces: List[str] = []
    hidden = 0
    for move in record.get("moves") or []:
        cap = move.get("captured")
        if not isinstance(cap, dict) or cap.get("side") != side:
            continue
        if cap.get("hidden"):
            hidden += 1
        else:
            pieces.append(str(cap.get("piece") or cap.get("home") or "?"))
    bits = []
    if pieces:
        bits.append("%d 枚明子（%s）" % (len(pieces), "、".join(pieces)))
    if hidden:
        bits.append("%d 枚暗子（身份未知）" % hidden)
    return " + ".join(bits) if bits else "无"


# ---------------------------------------------------------------------------
# 训练样本
# ---------------------------------------------------------------------------


def _final_value(record: Dict[str, Any], side: str) -> float:
    """终局结果 -> 该方视角的值（胜 1 / 和 0 / 负 -1）。"""
    result = str(record.get("result") or "*")
    if result == "1/2-1/2":
        return 0.0
    if result == "1-0":
        return 1.0 if side == "red" else -1.0
    if result == "0-1":
        return 1.0 if side == "black" else -1.0
    return 0.0


def _parse_eval(text: str) -> Optional[float]:
    """从"AI 评估：+123（红方稍优）"里抠出红方视角的分值。"""
    found = re.search(r"([+-]?\d+(?:\.\d+)?)", str(text or ""))
    if not found:
        return None
    try:
        return float(found.group(1))
    except ValueError:
        return None


def export_training(record: Dict[str, Any], path: str,
                    include_hidden_truth: bool = False) -> Dict[str, Any]:
    """把一盘棋导出成训练样本 JSONL（一行一个局面）。

    每行字段：

    * ``name`` / ``ply`` / ``side``：棋谱名、第几手、轮到谁走；
    * ``board``：10 行紧凑棋盘（暗子带 ``*``，**默认不含真身**）；
    * ``hidden_truth``：``[[r,c,兵种], …]``——**只有**棋谱本身是用
      ``build_record(..., reveal_hidden=True)`` 生成、并且这里显式传
      ``include_hidden_truth=True`` 时才会有（适合"事后复盘打标签"，
      直接拿去训练会让模型学会偷看暗子，慎用）；
    * ``move``：实际走法 ``[fr,fc,tr,tc]`` 与 ``move_chinese``；
    * ``legal_moves``：该局面的全部合法着法（引擎按公开信息生成）；
    * ``eval_red``：该手前后的引擎评分（红方视角，缺就是 ``null``）；
    * ``human``：**人工打分**（``rating`` / ``label`` / ``rating_good`` / ``note``），
      没打分就没有这个字段。训练时用 ``rating_good ∈ {+1,0,−1}`` 做优势加权，
      同时保留 ``rating`` 原始档位（五档信息量大于三档，换损失函数不用重标）；
    * ``line``：固定是 ``"main"``（分支棋谱功能已移除）；
    * ``value``：终局胜负（走子方视角，胜 1 和 0 负 -1）；
    * ``result``：棋谱结果。
    """
    from .board import encode, move_str  # 局部导入，避免模块导入期的循环

    lines: List[str] = []
    positions = record.get("positions") or []
    moves = record.get("moves") or []
    skipped = 0
    for move in moves:
        try:
            index = int(move.get("index") or 0)
        except (TypeError, ValueError):
            continue
        snapshot = positions[index - 1] if 0 < index <= len(positions) else None
        if not snapshot:
            skipped += 1
            continue
        board = snapshot.get("board")
        side = str(move.get("side") or snapshot.get("current_player") or "red")
        try:
            legal = _legal_moves_from_snapshot(board, side)
        except Exception:
            legal = []
        sample: Dict[str, Any] = {
            "name": record.get("name") or "",
            "ply": index,
            "side": side,
            "board": board_to_codes(_snapshot_to_gui(board)),
            #: 着法统一用 ``b7b5`` 写法（与走棋记录 / 棋谱 / 引擎一致）
            "move": _move_text_of(move),
            "move_chinese": move_to_chinese(move),
            "legal_moves": [move_str(m) for m in legal],
            "eval_red": _parse_eval(move.get("eval", "")),
            "human": _human_of(move),
            "line": "main",
            "value": _final_value(record, side),
            "result": record.get("result") or "*",
        }
        if sample["human"] is None:
            del sample["human"]          # 没打分就不写这个字段（老样本形状不变）
        if include_hidden_truth:
            truth = []
            for r in range(ROWS):
                row = board[r] if board and r < len(board) else None
                for c in range(COLS):
                    cell = row[c] if row and c < len(row) else None
                    if isinstance(cell, dict) and cell.get("hidden") and cell.get("actual"):
                        truth.append([r, c, cell.get("actual")])
            sample["hidden_truth"] = truth
        if move.get("comment"):
            sample["comment"] = str(move["comment"])[:400]
        lines.append(json.dumps(sample, ensure_ascii=False))

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))
    except OSError as exc:
        return {"ok": False, "path": path, "samples": 0, "error": str(exc),
                "skipped": skipped}
    return {"ok": True, "path": path, "samples": len(lines), "error": "",
            "skipped": skipped}


def _human_of(move: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从着法条目里取出"人工打分"（没打分返回 ``None``）。"""
    if not isinstance(move, dict):
        return None
    item = move.get("human")
    if not isinstance(item, dict):
        return None
    out: Dict[str, Any] = {}
    code = str(item.get("rating") or "").strip()
    if valid_rating(code):
        out["rating"] = code
        out["label"] = rating_label(code)
        out["rating_good"] = rating_good(code)
    if item.get("note"):
        out["note"] = str(item["note"])[:500]
    return out or None




def _apply_move_to_snapshot(board, mv, move: Dict[str, Any]):
    """在棋谱快照上走一步（返回**新的**棋盘，不改原数据）。

    只做数据搬运：把起点格子挪到终点、按 ``revealed_as`` 翻开身份、
    吃掉终点原有的子。规则合法性由引擎另行保证（导出样本只需要局面形状对）。
    """
    frm, to = (mv[0], mv[1]), (mv[2], mv[3])
    if not board:
        return board
    copy_board = [[dict(cell) if isinstance(cell, dict) else None
                   for cell in row] for row in board]
    try:
        piece = copy_board[frm[0]][frm[1]]
        if piece is None:
            return board
        if move.get("revealed_as"):
            piece["ident"] = str(move["revealed_as"])
            piece["hidden"] = False
        copy_board[to[0]][to[1]] = piece
        copy_board[frm[0]][frm[1]] = None
    except (IndexError, TypeError, KeyError):
        return board
    return copy_board


def _snapshot_to_gui(board) -> Optional[list]:
    """棋谱里的棋盘快照 -> GUI 棋盘字典（供走法生成/渲染复用）。"""
    if not board:
        return None
    out = [[None] * COLS for _ in range(ROWS)]
    for r in range(min(ROWS, len(board))):
        row = board[r]
        if not isinstance(row, (list, tuple)):
            continue
        for c in range(min(COLS, len(row))):
            cell = row[c]
            if not isinstance(cell, dict):
                continue
            side = str(cell.get("side") or "red")
            hidden = bool(cell.get("hidden"))
            home = str(cell.get("home") or "")
            actual = str(cell.get("actual") or cell.get("ident") or home)
            out[r][c] = {
                "color": "red" if side == "red" else "black",
                "original": home or actual,
                "actual": actual,
                "revealed": not hidden,
                "moved": not hidden,
            }
    return out


def _flip_hidden_ident(board, r: int, c: int, ident: str) -> bool:
    """把 ``(r,c)`` 上那枚**暗子**的摆位身份改成 ``ident``（用于复盘时按新证据修正）。

    揭棋里暗子只可能待在初始格子上，所以"这一格原本该放什么"是确定的；
    如果棋谱里记错了（早期格式没有记录被吃位置），用真实格子反推更可信。
    返回是否真的改了。
    """
    try:
        cell = board[r][c]
    except (TypeError, IndexError):
        return False
    if not isinstance(cell, dict) or not cell.get("hidden"):
        return False
    if cell.get("home") == ident:
        return False
    cell["home"] = ident
    cell["original"] = ident
    return True


def _legal_moves_from_snapshot(board, side: str) -> List[int]:
    from .gui_adapter import gui_board_to_engine

    gui = _snapshot_to_gui(board)
    if gui is None:
        return []
    engine_board = gui_board_to_engine(
        gui, turn=RED if str(side) == "red" else BLACK)
    return engine_board.gen_moves(RED if str(side) == "red" else BLACK)



# ---------------------------------------------------------------------------
# 复盘：找出"漏着 / 失着"
# ---------------------------------------------------------------------------


def replay_plan(record: Dict[str, Any]):
    """为"把棋谱重放成棋局/复盘"准备 ``(moves, reveals, meta)``。

    * ``moves``：每一手的 ``(sr, sc, er, ec)``；
    * ``reveals``：``{手数: (r, c, 兵种字)}``——棋谱**不写**暗子真身（隐私），
      但"某一步在某格翻出了 X"是公开事实，补上它才能用规则引擎校验每一步；
    * ``meta``：名字/结果/执子方等。

    重放方（``GameState.load_record`` 与 ``_bench/replay_log.py``）拿到这三样
    就能从开局一步一步走出来，既不需要棋谱里存真身，也不会算错规则身份
    （暗卒/暗兵的走法方向取决于摆位格）。
    """
    moves: List[Tuple[int, int, int, int]] = []
    reveals: Dict[int, Tuple[int, int, str]] = {}
    for move in record.get("moves") or []:
        # 新旧两种记法都认（``"b3"`` / ``[6, 1]``），老棋谱照样能重放
        item = _move_coords(move)
        if item is None:
            continue
        index = len(moves) + 1
        moves.append(item)
        # "这一步把某枚暗子翻成了什么"是公开事实（记在 revealed_as 里）。
        # 重放时必须照着补上该子的真实身份，否则重放出来的局面与原局不同，
        # 后续着法会因为"这枚子其实是什么"不同而变成不合法。
        if move.get("hidden") and move.get("revealed_as"):
            try:
                reveals[index] = (int(item[0]), int(item[1]),
                                  str(move["revealed_as"]))
            except (TypeError, ValueError, IndexError):
                pass
    meta = {
        "name": str(record.get("name") or ""),
        "result": str(record.get("result") or "*"),
        "winner": str(record.get("winner") or ""),
        "draw_reason": str(record.get("draw_reason") or ""),
        "player_side": str((record.get("meta") or {}).get("player_side") or "red"),
        "difficulty": str((record.get("meta") or {}).get("difficulty") or ""),
    }
    return moves, reveals, meta


def _prepare_replay(record: Dict[str, Any]):
    """为"用引擎复查某一手"准备"可解析的棋盘 + 走法列表"。

    1. 把棋谱里的**开局**棋盘快照转成 GUI 棋盘字典，并补上被吃暗子的摆位身份
       （被吃暗子保持未知，但"它原本该在哪一排"是公开信息）；
    2. 返回走法列表，供"引擎当初选了别的"这类对比使用。

    返回 ``(gui_board, moves)``；数据不可用时返回 ``(None, [])``。
    """
    from .consts import HOME_IDENT, IDENT_CHARS

    positions = record.get("positions") or []
    moves = record.get("moves") or []
    if not positions:
        return None, []
    gui = _snapshot_to_gui(positions[0].get("board"))
    if gui is None:
        return None, []
    # 按手数顺序推进，一边走一边修：只有"被吃的那一刻"才知道它原本在哪一格
    for move in moves:
        cap = move.get("captured")
        if not isinstance(cap, dict) or not cap.get("hidden"):
            continue
        square = cap.get("square")
        if not square:
            continue
        # ``square`` 现在是 ``"b3"`` 写法（老棋谱里是 ``[r, c]``），两种都认
        from .notation import name_to_square

        pos = name_to_square(square) if isinstance(square, str) else None
        if pos is None:
            try:
                pos = (int(square[0]), int(square[1]))
            except (TypeError, ValueError, IndexError):
                continue
        r, c = pos
        sq = r * COLS + c
        home = HOME_IDENT[sq] if 0 <= sq < ROWS * COLS else 0
        side = RED if cap.get("side") == "red" else BLACK
        ident = IDENT_CHARS.get((side, home))
        if ident:
            _flip_hidden_ident(gui, r, c, ident)
    return gui, moves


def analyse_record(record: Dict[str, Any], engine_factory=None,
                   time_limit: float = 0.6, samples: int = 1,
                   threshold: int = 120, top: int = 20) -> Dict[str, Any]:
    """逐手复查棋谱，找出"实际着法 vs 引擎最佳着法"落差最大的几手。

    ``engine_factory`` 是一个返回"类 Engine 对象"的可调用对象（默认用真实引擎）；
    单测里可以传替身，避免每次都真的搜 2 秒。

    返回 ``{"items": [...], "engine": {...}, "checked": n}``，
    每项含 ``ply`` / ``side`` / ``played`` / ``best`` / ``loss``（走子方视角的落差）。
    """
    from .board import move_str

    rows: List[Dict[str, Any]] = []
    positions = record.get("positions") or []
    moves = record.get("moves") or []
    engine = None
    for move in moves:
        try:
            index = int(move.get("index") or 0)
        except (TypeError, ValueError):
            continue
        snapshot = positions[index - 1] if 0 < index <= len(positions) else None
        if not snapshot:
            continue
        side = str(move.get("side") or snapshot.get("current_player") or "red")
        side_int = RED if side == "red" else BLACK
        try:
            gui = _snapshot_to_gui(snapshot.get("board"))
            if gui is None:
                continue
            from .gui_adapter import gui_board_to_engine

            board = gui_board_to_engine(gui, turn=side_int)
            if engine_factory is None:
                from .engine import Engine

                engine = Engine("normal", time_limit=time_limit, samples=samples,
                                seed=1)
            else:
                engine = engine_factory()
            result = engine.analyse(board)
        except Exception:
            continue
        if result is None or not getattr(result, "stats", None):
            continue
        wanted = _encode_move(move)
        stats = {st.move: st for st in result.stats}
        if wanted is None:
            continue
        best = result.move
        played_score = stats[wanted].mean if wanted in stats else None
        best_score = stats[best].mean if best in stats else None
        loss = 0.0
        if played_score is not None and best_score is not None:
            loss = float(best_score - played_score)
        rows.append({
            "ply": index,
            "side": side,
            "played": move_str(wanted),
            "played_chinese": move_to_chinese(move),
            "best": move_str(best) if best is not None else "?",
            "best_score": round(best_score, 1) if best_score is not None else None,
            "played_score": round(played_score, 1) if played_score is not None else None,
            "loss": round(loss, 1),
            "is_best": (best == wanted),
            "depth": getattr(result, "depth", 0),
        })
    flagged = [row for row in rows if row["loss"] >= threshold]
    flagged.sort(key=lambda row: row["loss"], reverse=True)
    return {
        "checked": len(rows),
        "threshold": threshold,
        "items": flagged[:max(1, int(top))],
        "all": rows,
        "engine": {"time_limit": time_limit, "samples": samples},
    }


def _move_text_of(move: Any) -> str:
    """一条着法条目 -> ``"b7b5"``（写进训练样本的 ``move`` 字段）。"""
    item = _move_coords(move)
    if item is None:
        return ""
    from .notation import move_name

    return move_name((item[0], item[1]), (item[2], item[3]))


def _encode_move(move: Dict[str, Any]) -> Optional[int]:
    """一条着法条目 -> 内部着法编码；读不出来返回 ``None``（新旧记法都认）。"""
    item = _move_coords(move)
    if item is None:
        return None
    sr, sc, er, ec = item
    return ((sr * COLS + sc) << 7) | (er * COLS + ec)
