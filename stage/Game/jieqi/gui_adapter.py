"""GUI（棋子字典棋盘）<-> 引擎（位棋盘）之间的转换。

GUI 的棋子是一个 dict::

    {'original': '炮', 'actual': '车', 'color': 'red',
     'revealed': False, 'moved': False}

* ``original`` 是**摆位身份**（该格子初始放什么），暗子按它行棋；
* ``actual`` 是真实身份，只有翻开后才可以看（引擎不会看）；
* ``revealed``/``moved`` 在正常流程里同步变化。

这一层刻意做得很薄，方便单测；同时承担三件“合法信息”的提取工作：
1. 已翻开棋子的真实身份（双方都看得见）；
2. 已被吃掉棋子的身份（吃子对双方可见）；
3. 公开局面签名序列（用于三次重复判和）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .board import Board, encode, move_str
from .consts import (
    BLACK,
    CHAR_TO_PIECE,
    HOME_IDENT,
    IDENT_CHARS,
    K,
    NAME_TO_SIDE,
    PIECE_VALUE,
    RED,
    ROWS,
    COLS,
)

#: GUI 里颜色字符串 -> 引擎方
GUI_COLOR = {"red": RED, "black": BLACK}
SIDE_COLOR = {RED: "red", BLACK: "black"}


def gui_color_to_side(color: str) -> int:
    return GUI_COLOR.get(str(color).lower(), RED)


def side_to_gui_color(side: int) -> str:
    return SIDE_COLOR[side]


def parse_gui_piece(piece: dict, sq: int) -> Tuple[int, int, bool, Optional[int]]:
    """把一枚 GUI 棋子解析成 ``(side, ident, hidden, home_ident)``。"""
    side = gui_color_to_side(piece.get("color", "red"))
    hidden = not bool(piece.get("revealed", False))
    actual = piece.get("actual")
    ident = CHAR_TO_PIECE.get(actual, (side, K))[1]
    home = None
    if hidden:
        orig = piece.get("original")
        if orig in CHAR_TO_PIECE:
            home = CHAR_TO_PIECE[orig][1]
        else:
            home = HOME_IDENT[sq]
    return side, ident, hidden, home


def gui_board_to_engine(gui_board, turn: int = RED,
                        trust_revealed: bool = True) -> Board:
    """把 GUI 的 10x9 棋盘转成引擎棋盘。

    ``trust_revealed=True`` 时会把暗子的真实身份一起写进 ``ident``，
    调用方（引擎）随后会用 ``clone_revealed_only`` 抹掉，因此不会泄漏。
    """
    from .consts import Z_SIDE

    b = Board.empty()
    b.turn = turn
    if turn == BLACK:
        b.key ^= Z_SIDE
    for r in range(ROWS):
        row = gui_board[r]
        for c in range(COLS):
            piece = row[c]
            if not piece:
                continue
            sq = r * 9 + c
            side, ident, hidden, home = parse_gui_piece(piece, sq)
            if hidden and not trust_revealed:
                ident = home if home is not None else HOME_IDENT[sq]
            b.put(sq, side, ident, hidden=hidden, home_ident=home)
    return b


def engine_board_to_gui(board: Board) -> List[List[Optional[dict]]]:
    """把引擎棋盘转回 GUI 的 10x9 棋子字典。"""
    out: List[List[Optional[dict]]] = [[None] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            sq = r * 9 + c
            if not board.ident[sq]:
                continue
            side = board.side[sq]
            if board.hidden[sq]:
                ch = IDENT_CHARS[(side, HOME_IDENT[sq])]
                out[r][c] = {"original": ch, "actual": ch,
                             "color": SIDE_COLOR[side],
                             "revealed": False, "moved": False}
            else:
                ch = IDENT_CHARS[(side, board.ident[sq])]
                out[r][c] = {"original": ch, "actual": ch,
                             "color": SIDE_COLOR[side],
                             "revealed": True, "moved": True}
    return out


def captured_info(move_history, use_hidden_captures: bool = False):
    """统计被吃掉的棋子。

    返回 ``(captured_known, captured_hidden_count)``：

    * ``captured_known[side]``：**明子**被吃的身份列表（身份对双方都可见）；
    * ``captured_hidden_count[side]``：**暗子**被吃的数量。按通行规则，
      被吃的暗子保持未知（"临局双方及旁观者皆不得揭看棋子"），
      所以只记数量、不记身份——这样 AI 的信息与人类完全对称。
    * ``use_hidden_captures=True`` 时把被吃暗子的真值也算进 ``captured_known``，
      信息多于人类，仅供调试/研究。
    """
    known: Dict[int, List[int]] = {RED: [], BLACK: []}
    hidden_count: Dict[int, int] = {RED: 0, BLACK: 0}
    for rec in move_history or ():
        cap = rec.get("captured")
        if not cap:
            continue
        side = gui_color_to_side(cap.get("color", "red"))
        actual = cap.get("actual")
        if actual not in CHAR_TO_PIECE:
            hidden_count[side] += 1
            continue
        if cap.get("revealed") or use_hidden_captures:
            known[side].append(CHAR_TO_PIECE[actual][1])
        else:
            hidden_count[side] += 1
    return known, hidden_count


def captured_identities(move_history) -> Dict[int, List[int]]:
    """兼容旧接口：只返回确定的被吃明子身份。"""
    known, _ = captured_info(move_history)
    return known


def _capture_entry(cap: dict, owner: int) -> Tuple[str, str]:
    """把一枚被吃棋子转成事实表里的两个写法 ``(真相, 公开说法)``。

    * 明子被吃：``("黑卒", "黑卒")`` —— 身份对双方可见；
    * 暗子被吃：``("", "一枚暗子")`` —— 按通行规则**不得翻看**，
      所以只记数量，不写身份。
    """
    color = str(cap.get("color", "red"))
    side = _side_of_color(color)
    cn = "红" if side == RED else "黑"
    if cap.get("revealed"):
        return ("%s%s" % (cn, cap.get("actual", "")), "%s%s" % (cn, cap.get("actual", "")))
    return ("", "一枚暗子（身份未知，双方都不得翻看）")


def track_captures(move_history, limit: Optional[int] = None) -> Dict[str, Any]:
    """按手数重放走棋记录，统计"谁被吃了什么"。

    返回 ``{"lost": {RED: {...}, BLACK: {...}}, "taken": {...}}``，每方形如::

        {"known": 1, "hidden": 2, "items": [{"label": "黑卒", "known": True}],
         "text": "1 枚明子(黑卒) + 2 枚暗子(身份未知)"}

    之所以要重放而不是读 ``gs.capture_piece``：那个列表在 ``goto_ply`` 回退时
    不会跟着恢复，浏览历史时它会比当前局面多出后面几步的子。按手数重放才与
    "当前显示的这一手"严格一致。``limit`` 表示"算到第 limit 手为止"（即 ``ply``）。
    """
    per_side: Dict[int, Dict[str, Any]] = {
        side: {"known": 0, "hidden": 0, "items": []} for side in (RED, BLACK)}
    history = list(move_history or ())
    if limit is not None:
        try:
            history = history[:max(0, int(limit))]
        except (TypeError, ValueError):
            pass
    for rec in history:
        cap = rec.get("captured")
        if not cap:
            continue
        owner = _side_of_color(cap.get("color", "red"))
        bucket = per_side[owner]
        if cap.get("revealed"):
            cn = "红" if owner == RED else "黑"
            label = "%s%s" % (cn, cap.get("actual", ""))
            bucket["known"] += 1
            bucket["items"].append({"label": label, "known": True})
        else:
            bucket["hidden"] += 1
            bucket["items"].append({"label": "暗子", "known": False})

    for bucket in per_side.values():
        bits = []
        if bucket["known"]:
            names = "、".join(item["label"] for item in bucket["items"] if item["known"])
            bits.append("%d 枚明子（%s）" % (bucket["known"], names))
        if bucket["hidden"]:
            bits.append("%d 枚暗子（被吃时未翻开，身份对双方都未知）" % bucket["hidden"])
        bucket["text"] = " + ".join(bits) if bits else "无"

    taken: Dict[int, List[Dict[str, Any]]] = {RED: [], BLACK: []}
    for owner in (RED, BLACK):
        taken[1 - owner] = list(per_side[owner]["items"])
    return {"lost": per_side, "taken": taken}


def capture_counts(move_history, limit: Optional[int] = None) -> Dict[int, Dict[str, int]]:
    """统计到第 ``limit`` 手为止，每方被吃的 **明子 / 暗子** 数量。"""
    return {side: {"known": int(data["known"]), "hidden": int(data["hidden"])}
            for side, data in track_captures(move_history, limit)["lost"].items()}


def captured_gui_pieces(gs, side: str, limit: Optional[int] = None
                        ) -> List[Dict[str, Any]]:
    """给界面用：某一方（``'red'`` / ``'black'``）**被吃**的棋子列表。

    每项是 ``{"piece": <GUI 棋子字典或 None>, "unknown": bool}``：

    * 明子被吃 → ``piece`` 是带 ``actual``/``color`` 的字典，界面照原样显示；
    * 暗子被吃 → ``piece=None``、``unknown=True``，界面上用"暗子图标"表示
      （被吃的暗子身份对双方都不可见，绝不能显示真身）。

    ``limit`` 表示"只算到第 limit 手为止"：浏览历史时必须传当前 ``ply``，
    否则展示条会把后面几步的吃子也算进来（回退时 ``capture_piece`` 不会恢复，
    这正是它不能直接拿来显示的原因）。
    """
    target = _side_of_color(side)
    if limit is None:
        limit = getattr(gs, "ply", None)
    history = list(getattr(gs, "move_history", None) or ())
    if limit is not None:
        try:
            history = history[:max(0, int(limit))]
        except (TypeError, ValueError):
            pass
    out: List[Dict[str, Any]] = []
    for rec in history:
        cap = rec.get("captured")
        if not cap:
            continue
        if _side_of_color(cap.get("color", "red")) != target:
            continue
        if cap.get("revealed"):
            out.append({"piece": dict(cap), "unknown": False})
        else:
            out.append({"piece": None, "unknown": True})
    return out


def _gui_piece_from_capture(cap) -> Optional[dict]:
    """走棋记录里保存的被吃棋子 -> GUI 棋子字典（缺字段时返回 None）。"""
    if not isinstance(cap, dict):
        return None
    color = cap.get("color")
    actual = cap.get("actual")
    if not color or not actual:
        return None
    return {"original": actual, "actual": actual, "color": color,
            "revealed": True, "moved": True}


def _side_facts(board, side: int, color_text: str, captures) -> LlmFacts:
    """组装某一方的事实（本方 / 对方共用同一套算法）。"""
    revealed, hidden, value = _scan_pieces(board)
    lost = captures.get("lost", {}).get(side, {})
    known = int(lost.get("known", 0))
    unknown = int(lost.get("hidden", 0))

    shown, missing = _clip_entries(list(revealed[side]))
    revealed_labels = [
        "%s%s%s" % (color_text, ch, square_name(sq)) for sq, ch in shown]
    if missing:
        revealed_labels.append("……另有 %d 枚明子未列出" % missing)

    # 暗子：同一种摆位身份合并成一行（"暗卒? ×3：位置…"），否则 15 行全是废话。
    # 「真实兵种未知」由渲染层统一在段末声明一次，不逐条重复。
    grouped: Dict[str, List[str]] = {}
    for sq, home in sorted(hidden[side], key=lambda item: item[0]):
        grouped.setdefault(home, []).append(square_name_cn(sq))
    hidden_labels = [
        "暗%s? ×%d：%s" % (home, len(squares), "、".join("(%s)" % s for s in squares))
        for home, squares in grouped.items()]

    captured_labels = [item["label"] for item in lost.get("items", [])]

    return LlmFacts(
        color=color_text,
        revealed=revealed_labels,
        revealed_missing=missing,
        hidden_count=len(hidden[side]),
        hidden=hidden_labels,
        hidden_missing=0,
        captured=captured_labels,
        captured_known=known,
        captured_hidden=unknown,
        value=value[side],
    )


#: 每方初始的暗子枚数（摆位身份格去掉将帅）。
DARK_PER_SIDE = 15

#: 事实表里最多罗列多少枚棋子（防止超长提示词），超出的只给数量。
FACTS_LIST_LIMIT = 24

#: 事实表里最多列出多少条威胁（再多也没什么信息量，只会把提示词撑长）。
THREAT_LIMIT = 8


def _clip_entries(entries, limit: int = FACTS_LIST_LIMIT):
    items = list(entries)
    if len(items) <= limit:
        return items, 0
    return items[:limit], len(items) - limit


def _threat_lines(board: Board, ai_side: int) -> List[str]:
    """用引擎的攻击判定算出"谁在被谁打"，写成给模型看的中文。

    这是纯事实（``Board.is_attacked`` 是保守判定：只多不少，绝不少算），
    正好用来消灭"我方车正被攻击"这类凭感觉编出来的句子。
    """
    lines: List[str] = []
    if board is None:
        return lines
    other = 1 - ai_side
    ai_cn = "红" if ai_side == RED else "黑"
    op_cn = "红" if other == RED else "黑"

    ki = board.king_sq(ai_side)
    if ki is not None and board.in_check(ai_side):
        lines.append("⚠ 本方(%s方)正在被将军，必须解将（解将优先于一切进攻）" % ai_cn)
    ko = board.king_sq(other)
    if ko is not None and board.in_check(other):
        lines.append("对方(%s方)正在被将军" % op_cn)

    own_hit, own_unknown = [], 0
    for sq in range(ROWS * COLS):
        if not board.ident[sq] or board.side[sq] != ai_side:
            continue
        if not board.is_attacked(sq, other):
            continue
        if board.hidden[sq]:
            own_unknown += 1
        else:
            ch = IDENT_CHARS.get((ai_side, board.ident[sq]), "")
            own_hit.append("%s%s%s" % (ai_cn, ch, square_name(sq)))
    if own_hit or own_unknown:
        bits = list(own_hit[:THREAT_LIMIT])
        if len(own_hit) > THREAT_LIMIT:
            bits.append("……等共 %d 枚" % len(own_hit))
        if own_unknown:
            bits.append("%d 枚暗子（身份未知）" % own_unknown)
        lines.append("本方(%s方)正被对方攻击的子：%s" % (ai_cn, "、".join(bits)))

    op_hit, op_unknown = [], 0
    for sq in range(ROWS * COLS):
        if not board.ident[sq] or board.side[sq] != other:
            continue
        if not board.is_attacked(sq, ai_side):
            continue
        if board.hidden[sq]:
            op_unknown += 1
        else:
            ch = IDENT_CHARS.get((other, board.ident[sq]), "")
            op_hit.append("%s%s%s" % (op_cn, ch, square_name(sq)))
    if op_hit or op_unknown:
        bits = list(op_hit[:THREAT_LIMIT])
        if len(op_hit) > THREAT_LIMIT:
            bits.append("……等共 %d 枚" % len(op_hit))
        if op_unknown:
            bits.append("%d 枚暗子（身份未知）" % op_unknown)
        lines.append("对方(%s方)正被本方攻击的子（可以吃/可以继续压）：%s"
                     % (op_cn, "、".join(bits)))
    return lines


def llm_facts(gs, board: Optional[Board] = None, ai_color: int = BLACK,
              turn: Optional[int] = None, ply: Optional[int] = None) -> Dict[str, Any]:
    """组装「局面事实表」——给大模型看的、**已经判定好的事实**。

    设计意图：把"谁的车、是明是暗、被吃了多少、谁在被攻击"这些**模型最容易
    编错**的信息在本地算准，用不含歧义的中文写死；模型只做解读与表达。
    """
    turn = gui_color_to_side(getattr(gs, "current_player", "red")) if turn is None else turn
    ai_color = int(ai_color)
    ply_value = getattr(gs, "ply", None) if ply is None else ply
    try:
        ply_value = int(ply_value) if ply_value is not None else None
    except (TypeError, ValueError):
        ply_value = None

    board_cells = getattr(gs, "board", None)
    history = getattr(gs, "move_history", None) or []
    captures = track_captures(history, ply_value)
    if board is None:
        try:
            board = gui_board_to_engine(board_cells, turn=turn)
        except Exception:
            board = None

    own_color = side_to_gui_color(ai_color)
    op_color = side_to_gui_color(1 - ai_color)
    own = _side_facts(board_cells, ai_color, "红" if ai_color == RED else "黑", captures)
    opponent = _side_facts(board_cells, 1 - ai_color, "红" if ai_color != RED else "黑",
                           captures)

    own.hidden_note = _hidden_note(own, ai_color)
    opponent.hidden_note = _hidden_note(opponent, 1 - ai_color)

    status = _status_line(own, opponent, ai_color)

    check: Dict[str, Any] = {
        "side_to_move_in_check": False,
        "ai_in_check": False,
        "opponent_in_check": False,
        "text": "双方都没有被将军",
    }
    threats: List[str] = []
    if board is not None:
        try:
            check["ai_in_check"] = bool(board.in_check(ai_color))
            check["opponent_in_check"] = bool(board.in_check(1 - ai_color))
            check["side_to_move_in_check"] = bool(board.in_check(turn))
            if check["ai_in_check"]:
                check["text"] = "本方(%s方)正在被将军" % ("红" if ai_color == RED else "黑")
            elif check["opponent_in_check"]:
                check["text"] = "对方(%s方)正在被将军" % ("红" if ai_color != RED else "黑")
        except Exception:
            pass
        try:
            threats = _threat_lines(board, ai_color)
        except Exception:
            threats = []

    if ply_value is None:
        ply_side = ""
    elif ply_value == 0:
        ply_side = "开局，还没有走过任何一手"
    else:
        rec = history[ply_value - 1] if 0 <= ply_value - 1 < len(history) else None
        prev = None
        if isinstance(rec, dict):
            prev = _side_of_color((rec.get("piece") or {}).get("color", "red"))
        if prev is None:
            ply_side = "上一手是谁走的未知；本轮轮到%s方走棋" % _cn(ai_color if turn == ai_color else 1 - ai_color)
        else:
            who = "本方" if prev == ai_color else "对方"
            ply_side = ("上一手是%s(%s方)走的；本轮轮到%s(%s方)走棋"
                        % (who, _cn(prev),
                           "本方" if turn == ai_color else "对方", _cn(turn)))

    return {
        "side_to_move": side_to_gui_color(turn),
        "ai_color": own_color,
        "opponent_color": op_color,
        "line": "%s方" % _cn(ai_color),
        "opponent_line": "%s方" % _cn(1 - ai_color),
        "ply": ply_value,
        "ply_side": ply_side,
        "check": check,
        "own": own.to_dict(),
        "opponent": opponent.to_dict(),
        "status": status,
        "threats": threats,
    }


def _cn(side: int) -> str:
    return "红" if int(side) == RED else "黑"


def _hidden_note(facts: LlmFacts, side: int) -> str:
    """用"明子清单 + 被吃明细"反推暗子还剩哪些可能，写成人话。"""
    cn = _cn(side)
    if facts.hidden_count == 0:
        return "%s方已经没有任何暗子，全部身份都已明牌。" % cn
    possible = DARK_PER_SIDE - len(facts.revealed) - facts.captured_known - facts.captured_hidden
    possible = max(0, possible)
    return ("%s方还有 %d 枚暗子（位置见上），它们的真实兵种未知；"
            "按「15 枚暗子 - 已翻开 - 已被吃」推算，剩余未知身份池里还有 %d 枚。"
            % (cn, facts.hidden_count, possible))


def _status_line(own: LlmFacts, opponent: LlmFacts, ai_side: int) -> str:
    diff = int(own.value) - int(opponent.value)
    if diff > 0:
        balance = "本方多 %d 厘兵" % diff
    elif diff < 0:
        balance = "本方少 %d 厘兵" % abs(diff)
    else:
        balance = "双方明子价值相同"
    return ("明子价值：本方可用明子共 %d 厘兵，对方可用明子共 %d 厘兵 → %s"
            "（只统计已翻开的明子，暗子与将帅不计入）"
            % (own.value, opponent.value, balance))




def engine_board_text(board: Board) -> str:
    return board.to_text()


def _record_side(rec: dict, ply_index: int) -> Optional[int]:
    """走棋记录里这一步是哪一方走的。

    优先用 ``history`` 里存下来的 ``side`` 字段；没有就按手数奇偶推
    （第 0 手是红方，之后红黑交替）。
    """
    if isinstance(rec, dict):
        side = rec.get("side")
        if side in (RED, BLACK):
            return int(side)
    if ply_index < 0:
        return None
    return RED if ply_index % 2 == 0 else BLACK


def _home_char(piece: dict, sq: int, side: int) -> str:
    """取棋子的"摆位身份"汉字。

    GUI 的棋子字典里本来就带 ``original``，优先用它；
    只有当它缺失时才回退到按格子查表——**必须用 .get 兜底**：
    棋子完全可以走到没有摆位身份的空行（例如第 1/4/5/8 行），
    那里的 ``HOME_IDENT`` 是 0，直接索引 ``IDENT_CHARS[(side, 0)]`` 会 KeyError。
    """
    original = piece.get("original")
    if original:
        return str(original)
    home = HOME_IDENT[sq]
    return IDENT_CHARS.get((side, home), "") if home else ""


def board_text(gui_board, turn: int = RED, flip: bool = False) -> str:
    """直接由 GUI 棋盘渲染中文棋盘文本（不依赖引擎）。

    格式（**这是大模型读棋盘图的唯一入口，不要随意改动**）::

        列号:  列0 列1 列2 列3 列4 列5 列6 列7 列8
        行0    车? 马? 相? 仕? 帅  仕? 相? 马? 车?
        行1     ·  ·  ·  ·  ·  ·  ·  ·  ·

    每一行都自带坐标，模型不需要回头去数格子。约定：

    * 带 ``?`` 的是**未翻开的暗子**，前面的字是它的摆位身份（按它行棋）；
    * 不带 ``?`` 的是**已经翻开的明子**，身份确定；
    * **列用字母 a~i（左到右）、行用数字 1~10（下到上）**，红方底线是第 1 行
      ——与走棋记录、棋谱、提示词的着法写法完全一致；
    * 行号写在最左边（``10`` .. ``1``），与真实棋盘的观感一致（红方在下）。
    """
    from .notation import FILE_LETTERS

    lines = ["列号:    " + "  ".join(FILE_LETTERS)]
    # 行号自上而下打印（9 -> 0），因为内部 row 0 画在最上面；rank 0 是红方底线。
    ranks = range(0, ROWS) if flip else range(ROWS - 1, -1, -1)
    for rank in ranks:
        row = ROWS - 1 - rank
        cells = []
        for c in range(COLS):
            piece = gui_board[row][c]
            if not piece:
                cells.append(" · ")
                continue
            side = gui_color_to_side(piece.get("color", "red"))
            if not piece.get("revealed"):
                cells.append(_home_char(piece, row * 9 + c, side) + "?")
            else:
                cells.append(str(piece.get("actual", "?")) + " ")
        lines.append("行%-2d   " % rank + " ".join(cells))
    lines.append(f"轮到：{'红方' if turn == RED else '黑方'}")
    return "\n".join(lines)


def build_llm_snapshot(gs, result=None, ai_color: int = RED) -> dict:
    """组装给大模型看的局面快照（见 jieqi/llm.py 的 snapshot 契约）。

    除了当前局面，还会带上一段**上下文**，这样点评才能"记得住"整盘棋、
    给出动态评价，而不是每一步都像第一次看到：
    * ``move_list``：最近若干手的走棋记录（含翻子结果、吃子）；
    * ``eval_history``：最近几步的引擎评分与结论，用来看形势怎么变化；
    * ``previous_comments``：上一步的点评，便于前后自洽或明确指出改判理由。
    * ``llm_facts``：本地算好的**事实表**（归属 / 明暗 / 被吃 / 威胁），
      提示词里原样输出。模型不再需要（也不允许）自己从棋盘图上猜归属。
    """
    turn = gui_color_to_side(gs.current_player)
    board = getattr(gs, "board", None)
    ply = int(getattr(gs, "ply", 0) or 0)
    snap: dict = {
        "side_to_move": SIDE_COLOR.get(turn, "red"),
        "ai_color": SIDE_COLOR.get(ai_color, "black"),
        # 明确的"点评对象"：这次点评讲的是 AI 自己即将走出的那一手，
        # 而不是泛泛地评价局面。提示词里会据此要求模型用"我/黑方"的口吻
        # 解释这步棋的意图，避免它跑去指点人类一方该怎么走。
        "comment_target": "AI 自己的招法",
        "move_number": getattr(gs, "move_count", 0),
        "repetition_warning": bool(getattr(gs, "toast", "").find("重复") >= 0),
        "board_text": board_text(board, turn=turn),
        "board": _snapshot_board(board),
    }
    # ---- 事实表：把"谁的车、是明是暗、谁被谁打"在本地算准 ----
    # 注意顺序：事实表里的"候选着法自然语言注解"要用到引擎结果，所以必须在
    # engine_candidates 写进快照之后再渲染 facts_text（早期的实现顺序反了，
    # 结果事实表里完全没有候选着法，模型只能自己猜"这步是谁在走"）。
    try:
        engine_board = gui_board_to_engine(board, turn=turn)
    except Exception:
        engine_board = None
    try:
        facts = llm_facts(gs, board=engine_board, ai_color=ai_color,
                          turn=turn, ply=ply)
        snap["llm_facts"] = facts
    except Exception:
        pass
    try:
        snap["captured_pieces"] = {
            SIDE_COLOR[side]: [
                ("%s%s" % ("红" if side == RED else "黑", item["label"]))
                if item.get("known") else "暗子"
                for item in data.get("items", [])]
            for side, data in track_captures(
                getattr(gs, "move_history", None) or [], ply)["lost"].items()
        }
    except Exception:
        pass
    # 只取"到当前显示的这一手为止"的记录：如果正在浏览历史，
    # 提示词必须描述**看到的那一步**，不能把后面的棋也塞进去。
    all_hist = getattr(gs, "move_history", None) or []
    hist = all_hist[:ply] if ply else []
    if hist:
        rec = hist[-1]
        try:
            (sr, sc), (er, ec) = rec["start"], rec["end"]
            piece = rec.get("piece") or {}
            cap = rec.get("captured")
            entry = {
                "from": [sr, sc], "to": [er, ec],
                "piece": _piece_label(piece),
                "captured": _piece_label(cap) if cap else None,
            }
            # 关键：move_history 存的是**落子前**的棋子快照，所以一步"翻子"
            # 在这里仍然显示为"暗子"，会和大模型看到的棋盘图自相矛盾
            # （棋盘上那一格已经是明子了）。这里补上翻开后的真实身份。
            revealed_as = rec.get("revealed_as")
            if not revealed_as and piece and not piece.get("revealed"):
                post = board[er][ec] if board else None
                if post and post.get("revealed"):
                    revealed_as = _piece_label(post)
            if revealed_as:
                entry["revealed_as"] = (revealed_as if str(revealed_as).startswith(
                    ("红", "黑")) else _side_prefix(piece, revealed_as))
            snap["last_move"] = entry
        except Exception:
            pass

    # ---- 上下文：走棋记录 / 评分走势 / 上一步点评 ----
    try:
        recent = hist[-12:]
        first_index = len(hist) - len(recent) + 1
        # 每条记录后面标上"我方/对方"：模型不必自己去推"这一步是谁走的"，
        # 少一个推理环节就少一处幻觉。
        start_ply = ply - len(recent)
        snap["move_list"] = [
            "%s　[%s]%s" % (
                move_record_text(rec, first_index + i),
                "我方" if _record_side(rec, start_ply + i) == ai_color else "对方",
                "（AI）" if _record_side(rec, start_ply + i) == ai_color else "")
            for i, rec in enumerate(recent)]
    except Exception:
        pass
    try:
        evals = []
        for text in (getattr(gs, 'ply_evals', None) or [])[:ply]:
            if text:
                evals.append(str(text).replace("\n", "　"))
        if evals:
            snap["eval_history"] = evals[-6:]
    except Exception:
        pass
    try:
        comments = [c for c in (getattr(gs, 'ply_comments', None) or [])[:ply] if c]
        if comments:
            snap["previous_comments"] = comments[-3:]
    except Exception:
        pass

    if result is not None:
        # 视角统一成「轮到走棋的一方」——这是棋类惯例，也避免模型搞混谁占优。
        # （引擎内部的 st.mean 本来就是走棋方视角；result.score/verdict 是红方视角。）
        #
        # 另外做一次"归属自检"：候选着法的起点必须是**走棋方自己的子**。正常流程
        # 一定成立（controller 只在该方走棋时分析），但外部工具传错 turn 时能立刻
        # 发现，避免把"红方的车"写成"我方的车"——那正是要治的幻觉。
        try:
            owner = gui_board_to_engine(board, turn=turn)
        except Exception:
            owner = None
        mismatched = 0
        cands = []
        for st in result.stats[:8]:
            frm, to = st.move >> 7, st.move & 127
            if owner is not None and owner.ident[frm] and owner.side[frm] != turn:
                mismatched += 1
            cands.append({
                "from": [frm // 9, frm % 9], "to": [to // 9, to % 9],
                "score": round(st.mean, 1),
                "winrate": round(_winrate(st.mean), 4),
                "pv": [move_str(m) for m in (st.pv or [st.move])][:6],
                "note": "",
            })
        snap["engine_candidates"] = cands
        snap["engine_depth"] = result.depth
        snap["engine_nodes"] = result.nodes
        snap["engine_samples"] = result.samples
        snap["engine_verdict"] = _side_verdict(result.stats[0].mean if result.stats
                                              else 0.0, turn)
        snap["score_basis"] = "所有 score/winrate 均为「轮到走棋一方」的视角，正数表示对走棋方有利"
        if mismatched:
            snap["engine_warning"] = (
                "⚠ 有 %d 条候选着法的起点不是「轮到走棋的一方」的子："
                "说明这份引擎结果不是针对当前局面算的，"
                "请**不要**为它们编造归属，只作泛泛参考。" % mismatched)

    # 现在候选着法已经在快照里了，再渲染事实表文字——这样它才能把每一步
    # 写成"我方(黑方，即 AI 执的一方) 的「車」 从 (9,8) 走到 (7,8)"这种
    # 不可能认错归属的句子。
    try:
        from .llm import render_facts_text

        facts = snap.get("llm_facts")
        if isinstance(facts, dict):
            facts.setdefault("board", snap.get("board"))
        snap["facts_text"] = render_facts_text(snap)
    except Exception:
        pass
    return snap


def _side_prefix(piece: Optional[dict], text) -> str:
    """给身份文字补上"红/黑"前缀，例如 车 -> 红车。"""
    if not piece:
        return str(text)
    prefix = "红" if str(piece.get("color")) == "red" else "黑"
    text = str(text)
    return text if text.startswith(("红", "黑")) else prefix + text


def _side_verdict(score_side_pov: float, turn: int) -> str:
    """把「走棋方视角」的分值翻译成中文结论。

    ``verdict_of`` 本身就会按分值的正负说成"红方/黑方…"，
    所以这里只需要先把走棋方视角换算成红方视角，**不要再去替换文字**
    （替换会把"黑方落后"说成"黑方胜势"）。
    """
    from .engine import verdict_of

    red_pov = score_side_pov if turn == RED else -score_side_pov
    return verdict_of(red_pov)


def repetition_counts(gs) -> Dict[str, int]:
    """统计到当前显示的这一手为止，每个公开局面出现了几次（三次重复判和用）。

    直接遍历 ``gs.positions``（每一步的完整快照），比重放走棋记录更快也更可靠：
    走棋记录一旦被回退/开新分支，重放很容易和真实局面脱节。
    """
    counts: Dict[str, int] = {}
    positions = getattr(gs, "positions", None)
    ply = getattr(gs, "ply", None)
    if positions:
        limit = len(positions) if ply is None else min(int(ply) + 1, len(positions))
        for k in range(limit):
            state = positions[k]
            try:
                b = gui_board_to_engine(
                    state["board"],
                    turn=gui_color_to_side(state.get("current_player", "red")))
            except Exception:
                continue
            sig = b.public_signature()
            counts[sig] = counts.get(sig, 0) + 1
        return counts

    # 兜底：老式重放（没有 positions 的情况，例如外部构造的 GameState 替身）
    states = getattr(gs, "state_history", None) or []
    history = getattr(gs, "move_history", None) or []
    if not states:
        return {}
    first = states[0]
    try:
        b = gui_board_to_engine(first["board"],
                                turn=gui_color_to_side(first.get("current_player", "red")))
    except Exception:
        return {}
    counts[b.public_signature()] = 1
    for rec in history[:ply] if ply is not None else history:
        try:
            (sr, sc), (er, ec) = rec["start"], rec["end"]
        except Exception:
            break
        mv = encode(sr * 9 + sc, er * 9 + ec)
        if not b.is_legal(mv):
            break
        b.make(mv)
        sig = b.public_signature()
        counts[sig] = counts.get(sig, 0) + 1
    return counts


def _piece_label(piece: Optional[dict]) -> str:
    if not piece:
        return ""
    if not piece.get("revealed"):
        return "暗子"
    return ("红" if str(piece.get("color")) == "red" else "黑") + str(piece.get("actual", ""))


def move_record_text(rec: dict, index: int) -> str:
    """把一条走棋记录格式化成面板里显示的一行中文。

    翻子会写成 ``暗砲(b3)-(b2)→車``，让人一眼看出翻出来是什么；
    坐标就是 ``b3`` 这套（列字母 + 行数字，红方底线是第 1 行），
    与棋谱 JSON、提示词的着法**逐字一致**。
    """
    from .notation import move_name

    piece = rec.get("piece") or {}
    side = "红" if str(piece.get("color")) == "red" else "黑"
    if piece.get("revealed"):
        label = str(piece.get("actual", "?"))
    else:
        label = "暗" + str(piece.get("original", "?"))
        revealed_as = rec.get("revealed_as")
        if revealed_as:
            label += "→" + str(revealed_as)
    try:
        (sr, sc), (er, ec) = rec["start"], rec["end"]
        coords = move_name((int(sr), int(sc)), (int(er), int(ec)))
    except Exception:
        coords = "??"
    text = f"{index:>3}. {side}{label} {coords}"
    cap = rec.get("captured")
    if cap:
        if cap.get("revealed"):
            text += "  吃" + ("红" if str(cap.get("color")) == "red" else "黑") \
                + str(cap.get("actual", ""))
        else:
            text += "  吃暗子"
    return text


def _snapshot_board(board):
    out = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            piece = board[r][c] if board else None
            if not piece:
                row.append(None)
                continue
            side = gui_color_to_side(piece.get("color", "red"))
            hidden = not piece.get("revealed")
            row.append({
                "side": SIDE_COLOR[side],
                "hidden": hidden,
                "home": _home_char(piece, r * 9 + c, side),
                "ident": None if hidden else str(piece.get("actual", "")),
            })
        out.append(row)
    return out


def _winrate(score: float) -> float:
    """把厘兵分折成 0~1 的胜率，方便大模型理解。"""
    import math

    if abs(score) > 100_000:
        return 0.99 if score > 0 else 0.01
    return 1.0 / (1.0 + math.exp(-score / 260.0))


# --------------------------------------------------------------------------
# 给 GUI 用的快速规则查询（替换旧版 O(90×90) 深拷贝的实现）
# --------------------------------------------------------------------------
def legal_moves(gui_board, turn: int) -> List[int]:
    b = gui_board_to_engine(gui_board, turn=turn)
    return b.gen_moves(turn)


def has_legal_moves(gui_board, turn: int) -> bool:
    b = gui_board_to_engine(gui_board, turn=turn)
    return bool(b.gen_moves(turn))


def in_check(gui_board, turn: int) -> bool:
    b = gui_board_to_engine(gui_board, turn=turn)
    return b.in_check(turn)


def is_checkmate(gui_board, turn: int) -> bool:
    """被将死或困毙（象棋里无棋可走即判负）。"""
    return not has_legal_moves(gui_board, turn)


def apply_move_to_gui(gui_board, mv: int) -> None:
    """在 GUI 棋盘上执行一步（供测试 / 工具使用，不走动画）。"""
    frm, to = mv >> 7, mv & 127
    sr, sc = frm // 9, frm % 9
    er, ec = to // 9, to % 9
    piece = gui_board[sr][sc]
    gui_board[er][ec] = piece
    gui_board[sr][sc] = None
    if piece is not None:
        piece["revealed"] = True
        piece["moved"] = True


def public_signature(gui_board, turn: int) -> str:
    return gui_board_to_engine(gui_board, turn=turn).public_signature()


# ==========================================================================
# 局面事实表（llm_facts）
# ==========================================================================
#
# 这一节是"治幻觉"的核心：**归属、明暗、被吃、威胁全部由引擎算好后写死给模型**，
# 模型只负责组织语言，不再需要自己从棋盘图上猜"这枚车是谁的"。
#
# 结构（键名即契约，``jieqi.llm`` 只按这些键取值，缺键一律按"未知"处理）::
#
#     {
#       "side_to_move": "red", "ai_color": "black", "line": "黑方",
#       "opponent_line": "红方", "ply": 17,
#       "ply_side": "上一手是对方(红方)走的；本轮轮到本方(黑方)走棋",
#       "check": {"side_to_move_in_check": False, "ai_in_check": False,
#                 "opponent_in_check": False, "text": "双方都没有被将军"},
#       "own": {"color": "red", "revealed": ["红车(行4,列4)"],
#               "hidden_count": 8, "hidden": ["暗炮?(行2,列1)"],
#               "captured": ["黑卒", "暗子"], "captured_count": 2, "value": 1200},
#       "opponent": {... 同上 ...},
#       "status": "本方(红方)净多 100 厘兵…",
#       "threats": ["⚠ 本方(红方)正在被将军", "…"],
#       "text": "…整段中文事实表…"
#     }
#
# ``text`` 由 :func:`jieqi.llm.render_facts_text` 生成；快照里也可以只带
# ``llm_facts`` 而不带 ``text``，``jieqi.llm`` 会自己渲染（避免同一段文字算两次）。


@dataclass
class LlmFacts:
    """一方的公开事实（明子 / 暗子 / 被吃 / 剩余可能性）。"""

    color: str = "red"
    revealed: List[str] = field(default_factory=list)
    revealed_missing: int = 0
    hidden_count: int = 0
    hidden: List[str] = field(default_factory=list)
    hidden_missing: int = 0
    captured: List[str] = field(default_factory=list)
    captured_known: int = 0
    captured_hidden: int = 0
    captured_unknown: int = 0
    value: int = 0
    hidden_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color": self.color,
            "revealed": list(self.revealed),
            "revealed_missing": self.revealed_missing,
            "hidden_count": self.hidden_count,
            "hidden": list(self.hidden),
            "hidden_missing": self.hidden_missing,
            "captured": list(self.captured),
            "captured_known": self.captured_known,
            "captured_hidden": self.captured_hidden,
            "captured_unknown": self.captured_unknown,
            "value": self.value,
            "hidden_note": self.hidden_note,
        }


def square_name(sq: int) -> str:
    """``sq``（``row * 9 + col``）-> ``"b3"``。

    坐标口径见 :mod:`jieqi.notation`：列 ``a``~``i`` 左到右、行 ``0``~``9``
    下到上（红方底线是第 0 行）。所以棋谱、走棋记录、提示词、引擎输出
    可以逐字对照。旧写法是 ``"(行4,列4)"``，与引擎对不上，换算一次就容易
    把上下搞反。
    """
    from .notation import square_to_xy
    from .notation import square_name as _name

    try:
        sq = int(sq)
    except (TypeError, ValueError):
        return "??"
    return _name(*square_to_xy(sq))


def square_name_cn(sq: int) -> str:
    """``sq`` -> ``"b3"``（与 :func:`square_name` 相同；中文语境下也用它）。

    保留这个别名是为了不改动一堆调用点；两种写法现在已经统一。
    """
    return square_name(sq)


def _coord_name(coord) -> str:
    """``(row, col)`` -> ``"b3"``。"""
    from .notation import square_name as _name

    try:
        r, c = int(coord[0]), int(coord[1])
    except (TypeError, ValueError, IndexError):
        return "??"
    return _name(r, c)


def _side_of_color(color: Any) -> int:
    return NAME_TO_SIDE.get(str(color).strip().lower(), RED)


def _scan_pieces(board):
    """扫描 GUI 棋盘，返回 ``(明子, 暗子, 材料分)`` 三份按方分组的统计。"""
    revealed: Dict[int, List[Tuple[int, str]]] = {RED: [], BLACK: []}
    hidden: Dict[int, List[Tuple[int, str]]] = {RED: [], BLACK: []}
    value: Dict[int, int] = {RED: 0, BLACK: 0}
    if not board:
        return revealed, hidden, value
    for r in range(ROWS):
        row = board[r] if r < len(board) else None
        for c in range(COLS):
            piece = row[c] if row and c < len(row) else None
            if not piece:
                continue
            sq = r * COLS + c
            side = _side_of_color(piece.get("color", "red"))
            if piece.get("revealed"):
                ch = str(piece.get("actual") or "")
                if not ch:
                    continue
                revealed[side].append((sq, ch))
                ident = CHAR_TO_PIECE.get(ch, (side, 0))[1]
                value[side] += int(PIECE_VALUE.get(ident, 0))
            else:
                home = _home_char(piece, sq, side)
                if not home:
                    continue
                hidden[side].append((sq, home))
    return revealed, hidden, value



