"""揭棋比赛规则：长将 / 长捉 / 判和条件（**纯函数 + 一个跟踪器**）。

规则依据（棋者象棋《揭棋玩法规则》 https://qizhe.org.cn/home/?p=848）：

1. **长捉禁招**：与象棋一致。单子长捉不超过 6 回合，6 回合后禁着；
   双子长捉不超过 12 回合。**兵（卒）、帅（将）、士（仕）、象（相）可以长捉**。
2. **长将禁招**：一子长将允许连续 6 回合；两子长将连续允许 12 回合
   （其中任一子不能连续 6 次）；两子以上连续允许 18 回合（其中任一子不能连续 6 次）。
3. **局面循环判和**（次数见 ``RULE['repetition_draw']``，本项目沿用 6 次）。
4. **空步判和**：双方无吃子（闲着）步数达 80 步判和。
   翻暗子要清空计数；将军和应将不计数、也不清空。
5. **无进攻子力判和**：这里的口径是**双方都没有"可以过河"的子力**
   （见 :func:`insufficient_material` 的说明，与规则原文有差异，已注明）。
6. **最大步数 200 步判和**。

设计约定
--------
* 本模块**不导入 pygame**，可以直接单测（``stage/tests/test_rules.py``）。
* **不读暗子的真实身份**：暗子按摆位身份（``piece['original']``）参与一切判断。
  这是揭棋的一贯口径——暗子按摆位行棋，所以"按摆位算出来的威胁"是公开信息，
  双方都能算出来，不构成偷看。
* 一步的客观事实收敛成 :class:`MoveFacts`；所有判定都是"事实序列 -> 结论"的
  纯函数，因此**幂等**（重复调用不会改变任何状态）。
* :class:`GameState` 只负责"每走一步就把事实追加进列表"，检查侧只读。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .consts import (
    COLS,
    HOME_IDENT,
    IDENT_CHARS,
    PIECE_VALUE,
    ROWS,
    SQUARES,
)

# --------------------------------------------------------------------------
# 规则阈值
# --------------------------------------------------------------------------
RULE = {
    # 长将：按"参与将军的**不同棋子数**"选上限（回合）
    'perpetual_check_single': 6,    # 一子长将上限（回合）
    'perpetual_check_double': 12,   # 两子长将上限
    'perpetual_check_multi': 18,    # 两子以上长将上限
    # 长捉：只有单子 / 双子两档
    'perpetual_chase_single': 6,    # 单子长捉上限
    'perpetual_chase_double': 12,   # 双子长捉上限
    #: 任一枚棋子自己的连续长将 / 长捉上限。规则原文只对"两子长将"写了这一句
    #: （"其中任一子不能连续 6 次"），但它表达的是长将的本质——**不能让同一枚
    #: 棋子无限反复**，所以这里统一施加到长将（1/2/3 子）与长捉（1/2 子）上。
    #: 对一子长将、单子长捉来说它与 6 回合的总上限重合，不会改变结论。
    'perpetual_piece_streak': 6,
    #: 局面循环几次判和。本项目**沿用 6 次**（权威规则文本写 5，用户确认按 6）。
    'repetition_draw': 6,
    #: 第几次之后开始每步提醒。取 4 与 6 自洽：从第 4 次起连续提醒三次
    #: （第 4、5 次提醒"还差 2 次 / 1 次"，第 6 次直接判和），
    #: 与原文"已达到第 4 次后，开始每步提醒"完全一致。
    'repetition_warn': 4,
    'idle_draw_plies': 80,          # 无吃子步数判和
    'max_plies': 200,               # 最大步数判和
}

#: 这些兵种"**可以长捉**"，不受长捉限制（规则 1）。
#: 注意用的是汉字（与 ``piece['original']`` / ``piece['actual']`` 一致），
#: 红黑两种写法都要列上：兵/卒、帅/将、仕/士、相/象。
CHASE_EXEMPT = {'兵', '卒', '帅', '将', '仕', '士', '相', '象'}


# --------------------------------------------------------------------------
# 一步的客观事实
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MoveFacts:
    """一步的客观事实（**从棋盘算出来**，不看暗子真身）。

    ``chased`` 里存的是**对方棋子所在的格子**（``(row, col)``，按行主序），
    而不是棋子本身：棋盘坐标对双方都是公开信息，用它做"是不是同一枚子"的
    比较不会泄漏任何暗子身份。
    """

    mover: str            # 'red' / 'black'
    piece: str            # 走的这枚子的汉字（暗子用 original 摆位身份）
    checked: bool         # 走完是否将军
    captured: bool
    flipped: bool         # 这一步是否翻开了暗子
    chased: Tuple[Tuple[int, int], ...] = ()
    in_check: bool = False      # 走这一步**之前**，走子方自己是否正被将军
    ply: int = 0
    sentinel: bool = False      # 见 sentinel_fact()


def sentinel_fact(mover: str = "red", checked: bool = False,
                  chased: Sequence[Sequence[int]] = (), in_check: bool = False,
                  piece: str = "", **kwargs) -> MoveFacts:
    """构造一个"标记步"事实。

    用途：检测**下一步就会越界**的临界局面。规则是"连续 6 回合后禁着"，
    所以"已经连续 6 次"本身还没违规，再走第 7 次才违规。把标记步接在事实
    序列末尾再判一次，就能分辨这两种状态：

    * 序列末尾是将军，追加标记步后仍然是 6 次连续 -> 此时才是"再走就违规"。

    引擎用它来剔除"会让长将 / 长捉越界"的候选着法（见
    :func:`Engine.forbidden_moves`）。
    """
    return MoveFacts(mover=str(mover), piece=str(piece), checked=bool(checked),
                     captured=False, flipped=False,
                     chased=tuple((int(r), int(c)) for r, c in chased),
                     in_check=bool(in_check), ply=int(kwargs.get('ply', 0)),
                     sentinel=True)


# --------------------------------------------------------------------------
# 内部工具：把事实序列规范化
# --------------------------------------------------------------------------
def _facts_list(facts) -> List[MoveFacts]:
    """接受 ``MoveFacts`` / ``dict`` / 元组，统一成 ``MoveFacts`` 列表。

    为什么要这么宽容：``GameState`` 会把事实存成普通元组（写进棋谱 JSON），
    而单测里更愿意直接写 ``MoveFacts``；判定逻辑不该关心这两种写法的差异。
    """
    out: List[MoveFacts] = []
    for item in facts or ():
        f = _as_fact(item)
        if f is not None:
            out.append(f)
    return out


def _as_fact(item) -> Optional[MoveFacts]:
    if isinstance(item, MoveFacts):
        return item
    if isinstance(item, dict):
        return MoveFacts(
            mover=str(item.get('mover') or 'red'),
            piece=str(item.get('piece') or ''),
            checked=bool(item.get('checked')),
            captured=bool(item.get('captured')),
            flipped=bool(item.get('flipped')),
            chased=_norm_coords(item.get('chased')),
            in_check=bool(item.get('in_check')),
            ply=_int(item.get('ply'), 0),
            sentinel=bool(item.get('sentinel')),
        )
    if isinstance(item, (tuple, list)):
        # 允许 ``(mover, piece, checked, captured, flipped, chased)`` 这种紧凑写法
        vals = list(item) + [None] * 8
        return MoveFacts(
            mover=str(vals[0] or 'red'),
            piece=str(vals[1] or ''),
            checked=bool(vals[2]),
            captured=bool(vals[3]),
            flipped=bool(vals[4]),
            chased=_norm_coords(vals[5]),
            in_check=bool(vals[6]),
            ply=_int(vals[7], 0),
            sentinel=False,
        )
    return None


def _norm_coords(value) -> Tuple[Tuple[int, int], ...]:
    out: List[Tuple[int, int]] = []
    for coord in value or ():
        try:
            r, c = int(coord[0]), int(coord[1])
        except (TypeError, ValueError, IndexError):
            continue
        out.append((r, c))
    return tuple(out)


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _other(side: str) -> str:
    return 'black' if str(side) == 'red' else 'red'


# --------------------------------------------------------------------------
# 长将 / 长捉：连续段与上限
# --------------------------------------------------------------------------
def _trailing_run(facts: Sequence[MoveFacts], hit) -> List[MoveFacts]:
    """取事实序列**末尾**那段"每一步都满足 ``hit`` 且由同一方连续走出"的着法。

    ``hit`` 为 ``None`` 时只按"同一方连续"分段（长捉用它自己数 chased）。

    "连续"的定义（本模块的统一口径）：

    * 必须是**同一方**接连走出的着法——中间只要插了一步对方的棋，就算断了。
      对长将来说对方每步都得应将，但"应将"本身不是将军，所以段自然就断了；
      对长捉也一样，对方一跑子，下一手就捉不到了。
    * 段必须在**当前末尾**结束。历史中段出现过 8 次连续将军、但现在已经不将了，
      不构成"正在长将"，不该判违规（判定只回答"**现在**还该不该变招"）。
    """
    run: List[MoveFacts] = []
    if not facts:
        return run
    last = facts[-1]
    if last.sentinel:
        # 标记步只用来"延续"末尾的段；它自己不算一次将军 / 捉子
        if len(facts) < 2:
            return run
        last = facts[-2]
    mover = last.mover
    for f in reversed(facts):
        if f.sentinel:
            continue
        if f.mover != mover:
            break
        if hit is not None and not hit(f):
            break
        run.append(f)
    run.reverse()
    return run


def _chase_run(facts: Sequence[MoveFacts]) -> List[MoveFacts]:
    """末尾那段"每一步都在捉"的着法。

    与长将的 :func:`_trailing_run` 同一套"连续"口径：同方接连走出的着法，
    中途断了就重数；差别只在于这里的"命中"是"这一步确实捉到了子"
    （``chased`` 非空）。**捉的对象可以中途换人**——一直追着同一个子反倒不是
    长捉的常见形态（对方会把子跑掉）。

    **豁免兵种在这里也一并跳掉**（规则 1："兵（卒），帅（将），士（仕），
    象（相）可长捉"）：由兵/卒/帅/将/仕/士/相/象走出的捉子着法不算长捉，
    所以它们既不计入"连续"次数，也不参与"几枚棋子"的档位判定。
    这一条与"被捉的子是豁免兵种就不算被捉"（在 :func:`chase_targets` 里）
    是两个方向，缺一不可。
    """
    run: List[MoveFacts] = []
    if not facts:
        return run
    last = facts[-1]
    if last.sentinel:
        if len(facts) < 2:
            return run
        last = facts[-2]
    mover = last.mover
    for f in reversed(facts):
        if f.sentinel:
            continue
        if f.mover != mover or not f.chased:
            break
        if f.piece in CHASE_EXEMPT:
            break                       # 豁免兵种捉子 -> 不算长捉，连续段到此为止
        run.append(f)
    run.reverse()
    return run


def _chase_limit(pieces: int) -> int:
    """长捉上限：单子 6 / 双子 12 / 三子及以上也给 12。

    规则原文只写了"单子 6、双子 12"两档。三子以上同时长捉同一个目标在象棋里
    几乎不可能（每一步只能有一枚子真的吃到它），真出现时按最宽的一档处理，
    宁可漏判也不误判。
    """
    if pieces <= 1:
        return int(RULE['perpetual_chase_single'])
    return int(RULE['perpetual_chase_double'])


def detect_perpetual_check(facts, side: Optional[str] = None
                           ) -> Tuple[bool, str]:
    """长将判定：``(是否违规, 中文原因)``。

    * **连续**按 :func:`_trailing_run` 的定义：同方接连将军，中间断了就重数；
      段必须在**当前末尾**结束（"现在还在长将"才算违规）。
    * 上限按"参与将军的**不同棋子数**"选：1 子 6 / 2 子 12 / ≥3 子 18。
    * 附加条件：**任一枚单子自己的连续将军次数不得达到 6**
      （规则原文写在"两子长将"那一条里，理由见 ``RULE['perpetual_piece_streak']``）。
    """
    seq = _facts_list(facts)
    if not seq:
        return False, ''
    sides = [side] if side else ['red', 'black']
    #: 事实序列末尾挂着"标记步"（见 :func:`sentinel_fact`）时，问的是
    #: "**再走一手**会不会越界"，所以把段长按 +1 折算（第 5 次将军本身合规、
    #: 第 6 次才违规 —— 折算之后"再来一次"正好落到上限上）。
    extra = 1 if seq[-1].sentinel else 0
    for who in sides:
        who = str(who)
        run = _trailing_run(seq, lambda f: f.checked)
        if not run or run[0].mover != who:
            continue
        # 末尾只挂了个"标记步" -> 它按定义不是将军，末手其实是别的着法，
        # 这一段并没有真的延续到当前 -> 不判违规。
        if seq[-1].sentinel and (len(seq) < 2 or run[-1] is not seq[-2]):
            continue
        pieces = len({f.piece for f in run})
        limit = int(RULE['perpetual_check_single'])
        if pieces == 2:
            limit = int(RULE['perpetual_check_double'])
        elif pieces >= 3:
            limit = int(RULE['perpetual_check_multi'])
        if len(run) + extra >= limit:
            return True, ('%s方长将违规：连续将军 %d 回合已达上限（%d 枚棋子参与，'
                          '上限 %d 回合），必须变招'
                          % (_side_cn(who), len(run) + extra, pieces, limit))
        streak_piece, streak = _piece_streak(run)
        # "任一枚棋子不能连续 6 次"的落地口径：
        # 同一枚子自己已经连续将军 ``streak`` 次、而**再凑一次**就能到
        # ``streak + extra + 1`` 次时，只要还能从段里别处补上足够的次数
        # （``len(run) + extra`` 已经到总上限 ``limit``），就该变招。
        # 对"一子长将"来说这一条与 6 回合上限重合（不会提前判违规）；
        # 它真正起作用的是"两子长将"那种总上限宽到 12 的情形——原文的
        # "其中任一子不能连续 6 次"说的就是这一种。
        if streak_piece and streak + extra + 1 > int(RULE['perpetual_piece_streak']):
            return True, ('%s方的「%s」连续将军 %d 次（任一单子不得连续 %d 次），'
                          '必须变招'
                          % (_side_cn(who), streak_piece, streak + extra,
                             RULE['perpetual_piece_streak']))
        if streak_piece and streak + extra >= int(RULE['perpetual_piece_streak']) - 1 \
                and len(run) + extra >= limit:
            return True, ('%s方的「%s」连续将军 %d 次（任一单子不得连续 %d 次），'
                          '必须变招'
                          % (_side_cn(who), streak_piece, streak + extra,
                             RULE['perpetual_piece_streak']))
    return False, ''


def detect_perpetual_chase(facts, side: Optional[str] = None
                           ) -> Tuple[bool, str]:
    """长捉判定：``(是否违规, 中文原因)``。

    * 上限按"参与长捉的**不同棋子数**"选：1 子 6 / 2 子 12。
    * **豁免兵种**（兵/卒/帅/将/仕/士/相/象）在 :func:`chase_targets` 里就已经
      被剔出 ``chased``，所以这里不需要再判一次——见 ``CHASE_EXEMPT`` 的说明。
    """
    seq = _facts_list(facts)
    if not seq:
        return False, ''
    sides = [side] if side else ['red', 'black']
    #: 末尾挂着"标记步"时问的是"**再走一手**会不会越界"，段长折算 +1
    #: （第 5 次捉子本身合规、第 6 次才违规）。见 :func:`sentinel_fact`。
    extra = 1 if seq[-1].sentinel else 0
    for who in sides:
        who = str(who)
        run = _chase_run(seq)
        if not run or run[0].mover != who:
            continue
        if seq[-1].sentinel and (len(seq) < 2 or run[-1] is not seq[-2]):
            continue                # 同上：末尾的标记步不算"还在捉"
        pieces = len({f.piece for f in run})
        limit = _chase_limit(pieces)
        if len(run) + extra >= limit:
            return True, ('%s方长捉违规：连续捉子 %d 回合已达上限（%d 枚棋子参与，'
                          '上限 %d 回合），必须变招'
                          % (_side_cn(who), len(run) + extra, pieces, limit))
        streak_piece, streak = _piece_streak(run)
        # 同长将：同一枚子连续捉到"再加一次就到 6"、且总回合数也到上限时变招
        # （单子长捉时这一条与 6 回合上限重合，不会提前判）。
        if streak_piece and streak + extra + 1 > int(RULE['perpetual_piece_streak']):
            return True, ('%s方的「%s」连续捉子 %d 次（任一单子不得连续 %d 次），'
                          '必须变招'
                          % (_side_cn(who), streak_piece, streak + extra,
                             RULE['perpetual_piece_streak']))
        if streak_piece and streak + extra >= int(RULE['perpetual_piece_streak']) - 1 \
                and len(run) + extra >= limit:
            return True, ('%s方的「%s」连续捉子 %d 次（任一单子不得连续 %d 次），'
                          '必须变招'
                          % (_side_cn(who), streak_piece, streak + extra,
                             RULE['perpetual_piece_streak']))
    return False, ''


def _piece_streak(run: Sequence[MoveFacts]) -> Tuple[str, int]:
    """段**末尾**同一枚棋子连续行动的次数 -> ``(棋子汉字, 次数)``。"""
    if not run:
        return '', 0
    piece = run[-1].piece
    count = 0
    for f in reversed(run):
        if f.piece != piece:
            break
        count += 1
    return piece, count


def _side_cn(side: str) -> str:
    return '黑' if str(side) == 'black' else '红'


# --------------------------------------------------------------------------
# 被捉（被威胁）判定
# --------------------------------------------------------------------------
def _without(board, squares: Iterable[int]):
    """返回一份"``squares`` 这些格子全部清空"的棋盘副本（其余原样）。

    ⚠ **必须只清空真的有子的格子**：``Board.remove()`` 会无条件异或掉
    "某方某子在此格"的哈希、并清空 ``occ`` 位，对一个**空格**调用它会把位棋盘
    与哈希算坏（那个格子会变成"占着但没子"的幽灵格，之后走法生成整片漏子）。
    真实踩过这个坑，排查了很久。
    """
    probe = board.copy()
    for sq in squares:
        try:
            sq = int(sq)
        except (TypeError, ValueError):
            continue
        if 0 <= sq < SQUARES and probe.ident[sq]:
            probe.remove(sq)
    return probe

def _only_side_keeps(board, side: int, clear_side: int):
    """返回一份"只剩 ``side`` 方自己的子"的棋盘副本。

    用来回答"**把对方棋子都拿掉之后**，``side`` 还能不能踏进某一格"。

    ⚠ 连**对方的将帅也要拿掉**（不只是"除将帅外"）。第一版留着对方的将，
    结果探针上到处触发"**将帅照面**"：红帅 (2,2) 想走到 (1,3) 被
    ``legal_filter`` 判成自将（走完与黑将同列无阻挡），于是"红帅保护邻格"
    被判反。探针问的是"这一格在不在我方的控制 / 可达范围里"，本来就不该
    受对方将帅位置的干扰。
    """
    probe = board.copy()
    for sq in range(SQUARES):
        if probe.ident[sq] and probe.side[sq] == clear_side:
            probe.remove(sq)
    return probe


def _landing_squares(board, sq: int, side: int) -> List[int]:
    """``side`` 方**能落到 ``sq`` 这一格**的己方起点格列表。

    语义是"谁**能踏进这一格**"，所以用普通伪合法着法（含非吃子着法）：
    格子空着时"吃子着法"一条都不会生成，用它判会一律得到"没人能来"。
    （``Board.gen_pseudo`` 的语义也确实是"目标格上有敌子才算吃"。）

    滑行子要求路径通畅、炮对空格是平移（不需要炮架）、马相要求不蹩腿不塞眼，
    全是引擎现成的正确实现，绝不自己重写一遍走法。
    """
    out: List[int] = []
    if not (0 <= sq < SQUARES):
        return out
    try:
        moves = board.gen_pseudo(side)
    except Exception:               # pragma: no cover - 防御
        return out
    for mv in moves:
        if (mv & 127) == sq:
            out.append(mv >> 7)
    return out


def _capture_squares(board, sq: int, side: int) -> List[int]:
    """``side`` 方**能吃掉 ``sq`` 上那枚子**的起点格列表（``sq`` 上必须有子）。"""
    if not (0 <= sq < SQUARES) or not board.ident[sq]:
        return []
    try:
        moves = board.gen_pseudo(side, captures_only=True)
    except Exception:               # pragma: no cover - 防御
        return []
    return [mv >> 7 for mv in moves if (mv & 127) == sq]


def _is_defended(board, sq: int, side: int) -> bool:
    """``sq`` 留空之后，``side`` 方**还能不能踏进这一格**（= 有保护）。

    做法：把这一格清空、把对方棋子全部拿掉，再问"我方还有子能走到这里吗"。
    这样"保护"就是纯粹的占位控制，与"这一格上现在站着谁"无关——正是长捉判据
    要的那半句话，也顺带堵掉了"被吃的那枚子自己算不算保护者"的歧义
    （它已经不在盘上了，自然不能保护自己）。

    ⚠ **不能**用 ``is_attacked``：它的语义是"这一格上的子能不能被吃掉"，包含
    **将帅对脸**那种只对"将帅本身"成立的伪攻击（同列无阻挡即算），会把一堆
    根本走不进去的格子误判成"有保护"。真实踩过：红车 (1,4) 的落点明明被红帅
    保护，却因为它身后黑将照面而判反。
    """
    probe = _without(board, (sq,))
    probe = _only_side_keeps(probe, side, 1 - side)
    return bool(_landing_squares(probe, sq, side))


def _safe_capture(board, frm: int, sq: int) -> bool:
    """走子方一枚子从 ``frm`` 吃 ``sq`` 之后，**对方吃回来不划算** -> True。

    做法：把 ``frm`` 与 ``sq`` 两格都清空，再问"走子方**别的**子还控制着
    ``sq`` 吗"。控制着就说明对方吃回来会亏子（我方有保护），所以这算"捉"；
    没人保护就只是"白送一个子换一个子"（等值交换），不判长捉。
    """
    if not board.ident[frm]:
        return False
    mover = board.side[frm]
    probe = _without(board, (sq, frm))
    return _is_defended(probe, sq, mover)


def _attacker_hanging(board, frm: int, target: int) -> bool:
    """走子方这枚子（在 ``frm``）**正被对方白吃** -> True。

    ``target`` 是它想去吃的那一格：判"有没有保护"时要把 ``frm`` 与 ``target``
    两格都清掉——棋子走到 ``target`` 之后，它原来站的 ``frm`` 已经空了，不该再
    替它挡车 / 挡炮（否则会把"对方沿着空出来的那条线打过来"误判成"我有保护"）。

    为什么要多这一条：**互捉不算捉**。"我捉你的车"和"你的车正捉着我的车"
    同时成立时，谁也不欠谁——对方先吃我就是了。反过来，如果对方吃我会亏子
    （我的落点有保护、或者换子我赚），那我是**真威胁**，才算捉。

    判据用"换子是否划算"而不是"落点有没有保护"：对方吃回来要拿它自己的子
    去换，只有换得起才叫白吃。例如黑卒能吃红车、但红帅保护着落点，黑卒吃回来
    是拿 100 换 900，亏得更多 —— 那一步不算化解威胁。
    """
    if not board.ident[frm]:
        return False
    mine = board.side[frm]
    other = 1 - mine
    # 看对方能不能真踏进"这一格"（清空它，其余原样）
    probe = _without(board, (frm,))
    incoming = _landing_squares(probe, frm, other)
    if not incoming:
        return False                    # 没人能真踏进来 -> 谈不上"被白吃"
    # 两格必须在**同一次**调用里清——分两次的话第二次是在空格上调用
    # ``remove()``，那会把位棋盘算坏（见 :func:`_without` 的警告）。
    if _is_defended(_without(board, (frm, target)), frm, mine):
        return False                    # 我有保护 -> 对方来吃是亏的
    # 没保护，再看对方吃这一手**划不划算**：它进来的那枚子价值 ``src``，
    # 吃掉的是我（``value``）。只有 ``src <= value``（换得起、或换得赚）才算真敢吃。
    # 例：黑卒 (100) 来吃红车 (900)，吃下去白丢 800，红方根本不用理它 ——
    # 那一步不该被当成"我的车正被白吃"。
    value = PIECE_VALUE.get(board.ident[frm], 0)
    for src in incoming:
        if PIECE_VALUE.get(probe.ident[src], 0) <= value:
            return True
    return False


def chase_targets(board_or_gamestate, after_move: bool = True,
                  side: Optional[str] = None) -> Tuple[Tuple[int, int], ...]:
    """走完这一步后，走子方**威胁（捉）**到的对方棋子坐标集合。

    判据（与象棋"捉"的口径对齐的**简化版**）：

    * 某枚对方子的落点被我方攻击；
    * **且**我方有**别的**子保护该落点（对方吃回来会亏子）；
    * **豁免兵种跳过**——兵/卒/帅/将/仕/士/相/象"可以长捉"，
      它们**被**捉时不算被捉（它们去捉别人也不受长捉限制）。

    为什么不把"被捉的子自己"算成保护者：它正被吃，不能保护自己。
    为什么不看暗子真身：判断只用引擎的公开信息（暗子按摆位身份行棋），
    所以这个结论对双方都是可见的。

    ``board_or_gamestate`` 可以是引擎 ``Board``，也可以是 ``GameState``；
    ``side`` 不传时按走子方取（``Board.turn`` / 当前 ``current_player``——
    ``GameState.move_piece`` 里"走完这一步之后、切换回合之前"调用它才是对的）。
    """
    if side is None:
        side = _side_of(board_or_gamestate)
    board = _engine_board(board_or_gamestate, side)
    if board is None:
        return ()
    me = 1 if str(side) == 'black' else 0
    opponent = 1 - me

    out: List[Tuple[int, int]] = []
    for sq in range(SQUARES):
        if board.ident[sq] == 0 or board.side[sq] != opponent:
            continue
        if _chase_exempt(board, sq):
            continue
        if not board.is_attacked(sq, me):
            continue                # 根本打不到 -> 谈不上捉
        for frm in _capture_squares(board, sq, me):
            if _attacker_hanging(board, frm, sq):
                continue            # 互捉：我自己也正被白吃 -> 谁也不欠谁
            if _safe_capture(board, frm, sq):
                out.append((sq // COLS, sq % COLS))
                break
    out.sort()
    return tuple(out)


def _chase_exempt(board, sq: int) -> bool:
    """这枚子是否属于"可以长捉"的豁免兵种（按**公开身份**判）。

    暗子用摆位身份（``HOME_IDENT``，引擎里就是 ``rule_ident``），明子用真实身份；
    两者都能在 ``IDENT_CHARS`` 里查到汉字，再与 ``CHASE_EXEMPT`` 比对。
    """
    if board.hidden[sq]:
        ident = HOME_IDENT[sq] or board.ident[sq]
    else:
        ident = board.ident[sq]
    ch = IDENT_CHARS.get((board.side[sq], ident))
    return bool(ch) and ch in CHASE_EXEMPT


def _side_of(board_or_gamestate) -> str:
    """取走子方（``'red'`` / ``'black'``）。"""
    turn = getattr(board_or_gamestate, 'turn', None)
    if isinstance(turn, int):
        return 'black' if turn else 'red'
    player = getattr(board_or_gamestate, 'current_player', None)
    if player:
        return str(player)
    return 'red'


def _engine_board(board_or_gamestate, side: str):
    """把 ``Board`` 或 ``GameState`` 统一成引擎 ``Board``（失败返回 ``None``）。"""
    from .board import Board

    if isinstance(board_or_gamestate, Board):
        return board_or_gamestate
    gui = getattr(board_or_gamestate, 'board', None)
    if gui is None:
        return None
    try:
        from .gui_adapter import gui_color_to_side, gui_board_to_engine

        turn = gui_color_to_side(side)
        return gui_board_to_engine(gui, turn=turn)
    except Exception:               # pragma: no cover - 构造失败就当"没有威胁"
        return None


# --------------------------------------------------------------------------
# 无吃子（闲着）步数
# --------------------------------------------------------------------------
def idle_ply_count(facts) -> int:
    """从事实序列算"无吃子步数"（规则 4）。

    * **吃子 -> 清零**；
    * **翻暗子 -> 清零**（规则原文"翻暗子，翻子清空计算"：翻子不是闲着，
      它改变了局面信息量，所以既不算数、也把之前的计数一起抹掉）；
    * **将军 / 应将 -> 不计数、也不清零**（规则原文"将和应将，不计数，不清空"）；
    * 其余**闲着（不含将军的普通着法）-> +1**。

    "应将"的识别：走子方**走这一步之前**正被将军（``MoveFacts.in_check``），
    那么这一步就是应将。
    """
    count = 0
    for f in _facts_list(facts):
        if f.captured or f.flipped:
            count = 0
        elif f.checked or f.in_check:
            continue
        else:
            count += 1
    return count


def move_limit_reached(ply_count: int) -> bool:
    """是否达到最大步数（200 步判和）。"""
    return _int(ply_count, 0) >= int(RULE['max_plies'])


def idle_draw_reached(count: int) -> bool:
    """无吃子步数是否达到判和线（80 步）。"""
    return _int(count, 0) >= int(RULE['idle_draw_plies'])


# --------------------------------------------------------------------------
# 无进攻子力（双方都没有"可以过河"的子力）
# --------------------------------------------------------------------------
def _enemy_half(piece_side: str) -> Tuple[int, ...]:
    """对方半场的行号。红方在 row 0..4、黑方在 row 5..9，所以：
    红子过河 = 走到 row>=5，黑子过河 = 走到 row<=4。"""
    if str(piece_side) == 'red':
        return tuple(range(5, ROWS))
    return tuple(range(0, 5))


def _has_crossing_move(board, side: int) -> bool:
    """``side`` 方是否**至少有一枚子存在能落到对方半场的合法着法**。

    这里直接用引擎的走法生成（``gen_moves`` = 已过滤自将 / 送将 / 对面笑），
    所以"过河"是**真能走**，不是纸上谈兵。
    """
    rows = set(_enemy_half('red' if side == 0 else 'black'))
    for mv in board.gen_moves(side):
        if (mv & 127) // COLS in rows:
            return True
    return False


def can_cross_river(game_state_or_board) -> Dict[str, bool]:
    """双方各自"还有没有可以过河的子力" -> ``{'red': bool, 'black': bool}``。

    * **暗子按摆位身份**判走法（引擎的 ``MOVES_HIDDEN`` 就是干这个的）：
      它虽然身份未知，但"按摆位现在能不能过河"是公开信息；
    * **明士 / 明象能过河**（权威规则：明子照中国象棋规则走子，
      揭开后的士、象可以过河），暗士 / 暗象按中国象棋规则不能过河——
      这个区别正是 ``board.py`` 里 ``MOVES`` / ``MOVES_HIDDEN`` 两张表的差异，
      所以这里直接复用引擎走法生成，绝不另写一套走法。
    """
    board = _as_engine_board_any(game_state_or_board)
    if board is None:
        # 拿不到局面：保守地当"还能过河"（**不判和**），绝不因为算不出来就把棋判死
        return {'red': True, 'black': True}
    return {'red': _has_crossing_move(board, 0),
            'black': _has_crossing_move(board, 1)}


def _as_engine_board_any(game_state_or_board):
    """把 ``GameState`` / ``Board`` / GUI 棋盘统一成引擎 ``Board``。"""
    from .board import Board

    if isinstance(game_state_or_board, Board):
        return game_state_or_board
    gui = getattr(game_state_or_board, 'board', None)
    if gui is None:
        gui = game_state_or_board
    try:
        from .gui_adapter import gui_board_to_engine

        return gui_board_to_engine(gui)
    except Exception:               # pragma: no cover
        return None


def insufficient_material(game_state) -> bool:
    """**双方都没有"可以过河"的子力** -> 判和。

    口径说明（用户确认的口径）
    --------------------------
    权威规则原文写的是"无进攻子力判和，**除将帅外都算进攻子力**"，即"任一方
    除将帅外无子即判和"。本项目按用户确认的口径实现为更强的一条：
    **双方都再也没有任何一枚棋子能走到对方半场**（走法按公开信息算，暗子按
    摆位身份）时才判和。理由：过不了河的士相只能在自己半场打转，谁也攻不到
    对方将帅，棋局实际上已经不可能再有进展；而"除将帅外都算进攻子力"的字面
    口径会在"双方各剩一枚士"这种明显和棋的局面下继续空转几百步。

    * 双方都过不了河 -> True（判和）；
    * **任意一方**还有子能过河 -> False（不判和，继续下）；
    * 只剩双方将帅 -> 九宫内走不出去，自然过不了河 -> True。

    注意与"**已经**过河"的区别：这里看的是"**还能不能**过河"，所以一枚已经
    在对方半场的子一定算能过河（它横走一步仍然在对方半场）；一枚卡在对方
    底线的兵（只能向前、已到底）则过不了河。
    """
    can = can_cross_river(game_state)
    return not can.get('red', True) and not can.get('black', True)


# --------------------------------------------------------------------------
# 重复局面
# --------------------------------------------------------------------------
def repetition_verdict(count: int) -> Dict[str, Any]:
    """同一公开局面出现过 ``count`` 次时该怎么办（**纯函数**）。

    返回 ``{'draw': bool, 'warn': bool, 'message': str}``：

    * ``draw``：达到 ``RULE['repetition_draw']``（本项目 = 6）-> 判和；
    * ``warn``：达到 ``RULE['repetition_warn']``（= 4）-> 每步提醒，
      正好与"6 次判和"自洽：第 4、5 次提醒"还差 2 次 / 1 次"，第 6 次判和。
    """
    count = max(0, _int(count, 0))
    draw_at = int(RULE['repetition_draw'])
    warn_at = int(RULE['repetition_warn'])
    if count >= draw_at:
        return {'draw': True, 'warn': True,
                'message': '局面重复 %d 次判和' % draw_at}
    if count >= warn_at:
        return {'draw': False, 'warn': True,
                'message': '局面已重复 %d 次，再重复 %d 次判和，请注意变招'
                           % (count, draw_at - count)}
    return {'draw': False, 'warn': False, 'message': ''}


def repetition_count(position_counter, signature: str) -> int:
    """``position_counter`` 里某个签名的出现次数（**只读**，不会写回）。"""
    try:
        return len((position_counter or {}).get(signature) or ())
    except Exception:               # pragma: no cover
        return 0


# --------------------------------------------------------------------------
# 判和条件汇总
# --------------------------------------------------------------------------
def evaluate_draws(game_state, facts=None) -> Optional[Dict[str, str]]:
    """按顺序检查全部判和条件，返回 ``{'rule', 'reason'}`` 或 ``None``。

    顺序即优先级：**无过河子力 -> 无吃子 80 步 -> 重复局面 -> 最大步数**。
    每条判据都是纯函数，重复调用结果完全一致（幂等），所以可以每帧安全调用。
    """
    if game_state is None:
        return None
    if insufficient_material(game_state):
        return {'rule': 'insufficient_material',
                'reason': '双方均无可以过河的子力判和'}

    seq = _facts_list(facts) if facts is not None else facts_from_game_state(game_state)
    idle = idle_ply_count(seq)
    if idle_draw_reached(idle):
        return {'rule': 'idle_plies',
                'reason': '连续 %d 步未吃子（闲着）判和' % int(RULE['idle_draw_plies'])}

    signature = getattr(game_state, '_board_hash', None)
    sig = ''
    if callable(signature):
        try:
            sig = str(signature() or '')
        except Exception:           # pragma: no cover
            sig = ''
    count = repetition_count(getattr(game_state, 'position_counter', None), sig)
    verdict = repetition_verdict(count)
    if verdict['draw']:
        return {'rule': 'repetition', 'reason': verdict['message']}

    if move_limit_reached(getattr(game_state, 'total_plies', 0)):
        return {'rule': 'max_plies',
                'reason': '达到最大步数 %d 步判和' % int(RULE['max_plies'])}
    return None


# --------------------------------------------------------------------------
# 从棋局取"事实序列"
# --------------------------------------------------------------------------
def fact_from_record(record: Dict[str, Any], ply: int = 0) -> MoveFacts:
    """从一条走棋记录（``GameState.move_history[i]``）还原 :class:`MoveFacts`。

    ``chased`` **无法**从记录里还原（那要看当时的棋盘），所以一律留空——
    这个函数只用于"没有现成事实"的兜底（例如导入的老棋谱、外部构造的替身），
    长捉判定在这种情况下会偏保守（少判，不误判）。
    """
    rec = record if isinstance(record, dict) else {}
    piece = rec.get('piece') if isinstance(rec.get('piece'), dict) else {}
    side_raw = rec.get('side')
    if str(side_raw) in ('0', 'red'):
        mover = 'red'
    elif str(side_raw) in ('1', 'black'):
        mover = 'black'
    else:
        mover = 'red' if int(ply) % 2 == 0 else 'black'
    revealed_before = bool(piece.get('revealed'))
    if revealed_before:
        ch = str(piece.get('actual') or '')
    else:
        ch = str(piece.get('original') or '')
    return MoveFacts(
        mover=mover,
        piece=ch,
        checked=_rec_flag(rec, 'checked', 'in_check'),
        captured=bool(rec.get('captured')),
        # 落子前的快照是暗子，且这一步真的翻开了它，才算"翻子"。
        # 旧记录没有 ``revealed`` 字段时按"没有翻子"处理（宁可少清一次计数，
        # 也不要因为字段缺失把"无吃子步数"反复清零）。
        flipped=(not revealed_before) and bool(piece),
        chased=(),
        in_check=_rec_flag(rec, 'self_check', 'was_in_check'),
        ply=int(ply),
    )


def _rec_flag(rec: Dict[str, Any], *names: str) -> bool:
    for name in names:
        if name in rec:
            return bool(rec.get(name))
    return False


def facts_from_game_state(game_state, upto: Optional[int] = None) -> List[MoveFacts]:
    """取 ``GameState`` 到第 ``upto`` 手为止的**事实序列**（默认当前 ``ply``）。

    优先用 ``move_piece`` 当时算好并存下来的事实（``gs.rule_facts``）；
    缺了（老棋谱 / 外部替身）就退回 :func:`fact_from_record` 的粗略还原。
    这样即使事实不全，判和与"最大步数"这类只看计数的规则也不会失效。
    """
    if game_state is None:
        return []
    history = list(getattr(game_state, 'move_history', None) or ())
    if upto is None:
        upto = getattr(game_state, 'ply', None)
        if upto is None:
            upto = len(history)
    upto = max(0, min(int(upto), len(history)))
    stored = getattr(game_state, 'rule_facts', None) or []
    out: List[MoveFacts] = []
    for i in range(upto):
        if i < len(stored) and stored[i] is not None:
            f = _as_fact(stored[i])
            if f is not None:
                out.append(f)
                continue
        out.append(fact_from_record(history[i], i))
    return out


# --------------------------------------------------------------------------
# 违规状态 + 跟踪器
# --------------------------------------------------------------------------
def rule_violation(facts, side: Optional[str] = None) -> Dict[str, Any]:
    """长将 / 长捉违规状态 -> ``{'forbidden', 'offender', 'reason'}``。

    ``side=None`` 时**两方都查**，但"触犯禁招"的语义是"该谁变招"，
    所以只有**正在连续将军 / 捉子**的那一方可能被判违规（一段早已停下的
    长将历史不该让任何一方背锅）。
    """
    seq = _facts_list(facts)
    for who in ([str(side)] if side else ['red', 'black']):
        for detect in (detect_perpetual_check, detect_perpetual_chase):
            violated, reason = detect(seq, who)
            if violated:
                return {'forbidden': True, 'offender': who, 'reason': reason}
    return {'forbidden': False, 'offender': None, 'reason': ''}


class RuleTracker:
    """按手累积 :class:`MoveFacts`，随时回答"现在谁违规、哪些着法被禁"。

    刻意做成"**喂进去才更新**"的小对象，而不是每帧重算：

    * ``GameState`` 每走一步喂一次（在 ``move_piece`` 里，顺手就算好了）；
    * 检查侧（``rule_status`` / ``check_game_state``）只读，**幂等**。

    缓冲区上限取"长将 18 回合 + 长捉 12 回合 + 余量"——判据最多回看 18 手，
    再老的着法对结论毫无影响，所以这里用定长 ``deque`` 自动丢弃。
    """

    #: 需要回看的最大手数（够 18 回合长将 + 一点余量）
    WINDOW = 24

    def __init__(self, facts: Optional[Iterable[MoveFacts]] = None,
                 window: int = WINDOW) -> None:
        self._facts: deque = deque(maxlen=max(2, int(window)))
        if facts:
            self.extend(facts)

    # ------------------------------------------------------------ 写入
    def reset(self) -> None:
        self._facts.clear()

    def feed(self, fact) -> None:
        f = _as_fact(fact)
        if f is not None:
            self._facts.append(f)

    def extend(self, facts: Iterable) -> None:
        for fact in facts or ():
            self.feed(fact)

    def truncate(self, length: int) -> None:
        """截断到前 ``length`` 步（分叉 / 回退时调用）。"""
        length = max(0, int(length))
        while len(self._facts) > length:
            self._facts.pop()

    # ------------------------------------------------------------ 只读
    def facts(self) -> List[MoveFacts]:
        return list(self._facts)

    def status(self, side: Optional[str] = None) -> Dict[str, Any]:
        return rule_violation(list(self._facts), side)

    def forbidden_now(self, side: Optional[str] = None) -> Dict[str, Any]:
        """现在是否已经"必须变招"（序列末尾再加一手就会越界）。"""
        seq = list(self._facts) + [sentinel_fact()]
        if side is None:
            # 标记步归**最后一手那一方**：长将 / 长捉是"谁在连续"的事，
            # 标记步按对方算的话就会把一段早已停下的旧将军误判成持续违规。
            side = seq[-2].mover if len(seq) >= 2 else 'red'
        return rule_violation(seq, side)


def forbidden_moves(facts, board, side: str) -> List[int]:
    """在 ``board`` 上，哪些着法会让 ``side`` 触犯长将 / 长捉禁招。

    引擎在**根节点**用它把这类着法从候选里剔除（``Engine.analyse`` 的
    ``rule_facts`` 参数）。对每个候选着法，把"它造成的事实"接到历史事实后面
    再判一次；判成违规就返回该着法。

    说明：这里只看**一手**的后果，不做"对方被迫应将 / 被迫逃子"的完整闭环
    （那需要递归证明，成本远超收益，而且规则原文也只给了回合数上限）。
    """
    seq0 = _facts_list(facts)
    me = 1 if str(side) == 'black' else 0
    who = 'black' if me else 'red'
    out: List[int] = []
    for mv in board.gen_moves(me):
        fact = fact_after_move(board, mv, who)
        if not fact.checked and not fact.chased:
            continue                # 既不将军也不捉子 -> 不可能触犯这两条
        probe = seq0 + [fact]
        violated = (detect_perpetual_check(probe, who)[0]
                    or detect_perpetual_chase(probe, who)[0])
        if violated:
            out.append(int(mv))
    return out


def fact_after_move(board, move: int, side: str,
                    was_hidden: Optional[bool] = None) -> MoveFacts:
    """在 ``board``（**走子前**的局面）上模拟 ``move``，算出这一步的事实。

    ``board`` 必须是**走子之前**的局面，而且这只是"模拟"——不会改动它。
    原因是 ``chased`` / ``checked`` / ``captured`` 全都要求知道"走之前是什么样"：

    * 「被吃的那枚子」只在走之前还在目标格上；
    * 「翻子」看的是**起点**上那枚子走之前是不是暗的。

    所以调用方要么趁早调用，要么用 ``was_hidden`` 把"走之前它是不是暗子"
    显式传进来（``GameState`` 就是这么做的：它在落子之后才记事实，
    但起点已经空了，只能靠调用方把当时的 ``revealed`` 状态带过来）。
    """
    from .board import move_from, move_to

    me = 1 if str(side) == 'black' else 0
    frm, to = move_from(move), move_to(move)
    # ⚠ "吃子"必须在 make **之前**读：走完之后目标格上已经是走子方自己的子了，
    # 再问"这格有没有敌子"永远是 False（第一版就是这么写错的）。
    captured = bool(board.ident[to]) and board.side[to] != me
    # ⚠ "翻子"看的是**起点**：调用方如果在落子之后才算，起点已经空了，
    # 这里必须靠 ``was_hidden`` 兜底（否则 flipped 永远是 False，
    # 于是"翻子清零"这条规则完全失效）。
    if was_hidden is None:
        was_hidden = bool(board.hidden[frm]) if 0 <= frm < SQUARES else False
    piece = board.piece_char(frm) or ''
    for prefix in ('暗', '红', '黑'):
        if piece.startswith(prefix):
            piece = piece[1:]
            break
    in_check = bool(board.in_check(me)) if board.kings[me] >= 0 else False
    probe = board.copy()
    probe.make(move)
    checked = False
    if probe.kings[1 - me] >= 0:
        checked = bool(probe.in_check(1 - me))
    chased = chase_targets(probe, side=side)
    return MoveFacts(mover=str(side), piece=piece, checked=checked,
                     captured=captured, flipped=bool(was_hidden), chased=chased,
                     in_check=in_check, ply=int(board.ply) + 1)


__all__ = [
    'RULE', 'CHASE_EXEMPT', 'MoveFacts', 'RuleTracker',
    'sentinel_fact', 'fact_after_move', 'fact_from_record',
    'facts_from_game_state', 'forbidden_moves', 'rule_violation',
    'chase_targets', 'can_cross_river', 'insufficient_material',
    'idle_ply_count', 'idle_draw_reached', 'move_limit_reached',
    'detect_perpetual_check', 'detect_perpetual_chase',
    'repetition_verdict', 'repetition_count', 'evaluate_draws',
]
