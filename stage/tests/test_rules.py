# -*- coding: utf-8 -*-
"""揭棋比赛规则（``jieqi/rules.py``）的单元测试。

覆盖范围（对应权威规则 https://qizhe.org.cn/home/?p=848）：

* 长将：6 / 12 / 18 三档上限、"任一单子连续 6 次"的附加条件、连续段的中断；
* 长捉：6 / 12 两档、豁免兵种（兵卒帅将仕士相象）、互捉不算捉、捉的判据；
* 空步判和：吃子 / **翻子**清零，将军与应将不计数也不清零，其余累加；
* 无过河子力判和：**双方都没有可以过河的子力**才判和（明士/明象能过河、
  暗士/暗象不能），边界（只剩将帅、卡在底线的兵、九宫里的士相）；
* 重复局面：6 次判和、第 4 次起提醒、**幂等**（重复调用不改变计数）；
* 最大步数 200 判和；
* 引擎层：``forbidden_moves`` / ``Engine.analyse(rule_facts=...)`` 剔除违规着法，
  且**不传 ``rule_facts`` 时行为与以前完全一致**。

测试全程使用 ``tempfile`` 临时目录，绝不碰用户的 ``records/``。
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
for _p in (str(GAME_DIR.parent), str(GAME_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pygame  # noqa: E402

from jieqi import board as B  # noqa: E402
from jieqi import rules as R  # noqa: E402
from jieqi.consts import BLACK, RED  # noqa: E402

#: 整数形式的红黑（``rules`` 里对外用字符串，引擎里用 0/1）
R_ = RED
B_ = BLACK


def setUpModule():
    """``GameState`` 会 import pygame，所以必须先初始化（用假驱动，不开窗口）。"""
    pygame.init()
    pygame.display.set_mode((1, 1))


# --------------------------------------------------------------------------
# 造事实 / 造棋盘的小工具
# --------------------------------------------------------------------------
def fact(mover, piece='车', checked=False, chased=(), captured=False,
         flipped=False, in_check=False, sentinel=False):
    """造一条 :class:`MoveFacts`（默认是"闲着"）。"""
    return R.MoveFacts(mover=mover, piece=piece, checked=checked,
                       captured=captured, flipped=flipped,
                       chased=tuple((int(r), int(c)) for r, c in chased),
                       in_check=in_check, sentinel=sentinel)


def checks(mover, times, piece='车'):
    """造"某一方连着将军 ``times`` 次"的事实段。"""
    return [fact(mover, piece=piece, checked=True) for _ in range(times)]


def chases(mover, times, piece='车', target=(5, 4)):
    """造"某一方连着捉子 ``times`` 次"的事实段。"""
    return [fact(mover, piece=piece, chased=(target,)) for _ in range(times)]


def sq_of(r, c):
    return r * 9 + c


def build_board(pieces):
    """``[((r, c), side, ident, hidden), ...]`` -> 引擎 ``Board``。"""
    board = B.Board.empty()
    for (r, c), side, ident, hidden in pieces:
        board.put(sq_of(r, c), side, ident, hidden=hidden)
    return board


#: 一对将帅，摆在默认位置（红帅 (0,4)、黑将 (9,4)）。
KINGS = [((0, 4), R_, B.K, False), ((9, 4), B_, B.K, False)]


def gui_board_blank():
    return [[None] * 9 for _ in range(10)]


def gui_piece(color, original, actual=None, revealed=True):
    return {'color': color, 'original': original,
            'actual': actual or original, 'revealed': revealed,
            'moved': revealed}


# ==========================================================================
# 1. 长将
# ==========================================================================
class TestPerpetualCheck(unittest.TestCase):
    """长将禁招：一子 6 / 两子 12 / 两子以上 18，且任一单子不得连续 6 次。"""

    def test_five_checks_by_one_piece_is_allowed(self):
        """第 5 次连续将军还不违规。"""
        violated, reason = R.detect_perpetual_check(checks('red', 5))
        self.assertFalse(violated, reason)

    def test_six_checks_by_one_piece_is_a_violation(self):
        """单子连续将军 6 次 -> 违规（上限 6 回合）。"""
        violated, reason = R.detect_perpetual_check(checks('red', 6))
        self.assertTrue(violated)
        self.assertIn('长将', reason)
        self.assertIn('红', reason)

    def test_black_offender_is_reported_as_black(self):
        violated, reason = R.detect_perpetual_check(checks('black', 6))
        self.assertTrue(violated)
        self.assertIn('黑', reason)

    def test_five_checks_by_one_piece_is_the_forbidden_boundary(self):
        """单子已连续将军 5 次 -> "再走一次就违规"（``forbidden_now``）。"""
        tracker = R.RuleTracker(checks('red', 5))
        status = tracker.forbidden_now()
        self.assertTrue(status['forbidden'], status)
        self.assertEqual(status['offender'], 'red')

    def test_four_checks_is_not_yet_the_boundary(self):
        """只连续 4 次时还不该报"必须变招"（第 5 次将军本身是允许的）。"""
        tracker = R.RuleTracker(checks('red', 4))
        self.assertFalse(tracker.forbidden_now()['forbidden'])

    def test_interrupted_sequence_does_not_violate(self):
        """中途被对方走了一步 -> 连续断了，重新计数，不违规。"""
        seq = []
        for _ in range(6):
            seq += checks('red', 1)
            seq.append(fact('black'))          # 对方走一步（不是将军）
        violated, _ = R.detect_perpetual_check(seq)
        self.assertFalse(violated, '连续段被对方打断后不该累计')

    def test_alternating_two_pieces_may_check_twelve_times(self):
        """两子交替将军：12 次到上限 -> 违规。"""
        seq = []
        for i in range(6):
            seq.append(fact('red', piece='车A', checked=True))
            seq.append(fact('red', piece='车B', checked=True))
        self.assertEqual(len(seq), 12)
        violated, reason = R.detect_perpetual_check(seq)
        self.assertTrue(violated, reason)
        self.assertIn('2 枚棋子', reason)

    def test_alternating_two_pieces_eleven_times_is_allowed(self):
        """同样是两子交替，但只有 11 次 -> 未到 12 回合上限。"""
        seq = []
        for i in range(6):
            seq.append(fact('red', piece='车A', checked=True))
            if i < 5:
                seq.append(fact('red', piece='车B', checked=True))
        seq.append(fact('red', piece='车A', checked=True))
        self.assertEqual(len(seq), 12)          # 稳一稳，先确认构造
        # 去掉最后一次 -> 11 次
        violated, _ = R.detect_perpetual_check(seq[:-1])
        self.assertFalse(violated)

    def test_single_piece_streak_caps_two_piece_rotation(self):
        """两子轮换里，**同一枚子**连续超过 5 次就违规，哪怕总回合还没到 12。

        规则原文："两子长将连续允许 12 回合（**其中任一子不能连续 6 次**）"。
        这里构造"车A 1 次 + 车B 6 次"：总回合只有 7（远不到 12），
        但车B 自己连续 6 次将军已经触犯附加条件。
        """
        seq = ([fact('red', piece='车A', checked=True)]
               + [fact('red', piece='车B', checked=True)] * 6)
        self.assertEqual(len({f.piece for f in seq}), 2, '前提：两枚棋子参与')
        self.assertLess(len(seq), R.RULE['perpetual_check_double'],
                        '前提：总回合还没到 12')
        violated, reason = R.detect_perpetual_check(seq)
        self.assertTrue(violated, reason)
        self.assertIn('任一单子', reason)
        self.assertIn('车B', reason)

    def test_single_piece_streak_alone_does_not_fire_early(self):
        """一子长将时"任一单子连续"这一条与 6 回合上限重合，不会提前判。"""
        seq = checks('red', 5, piece='车A')
        violated, _ = R.detect_perpetual_check(seq)
        self.assertFalse(violated, '整段只有 5 次，还没到 6 回合上限')
        violated, _ = R.detect_perpetual_check(
            seq + [fact('red', piece='车A', checked=True)])
        self.assertTrue(violated, '到第 6 次就该违规')


    def test_three_pieces_get_the_widest_limit(self):
        """三子轮换：上限 18 回合，17 次不违规、18 次违规。"""
        seq = [fact('red', piece='车%d' % (i % 3), checked=True)
               for i in range(17)]
        violated, _ = R.detect_perpetual_check(seq)
        self.assertFalse(violated, '三子 17 次不该违规')
        seq.append(fact('red', piece='车0', checked=True))
        violated, reason = R.detect_perpetual_check(seq)
        self.assertTrue(violated, reason)
        self.assertIn('18', reason)

    def test_stale_history_is_not_a_violation(self):
        """历史里有过 8 次连续将军、但**现在已经不将了** -> 不判违规。"""
        seq = checks('red', 8) + [fact('black'), fact('red', chased=((5, 4),))]
        violated, _ = R.detect_perpetual_check(seq)
        self.assertFalse(violated, '段必须在当前末尾结束才算"正在长将"')


# ==========================================================================
# 2. 长捉
# ==========================================================================
class TestPerpetualChase(unittest.TestCase):
    """长捉禁招：单子 6 / 双子 12；兵卒帅将仕士相象**可以长捉**。"""

    def test_five_chases_by_one_piece_is_allowed(self):
        violated, reason = R.detect_perpetual_chase(chases('red', 5))
        self.assertFalse(violated, reason)

    def test_six_chases_by_one_piece_is_a_violation(self):
        violated, reason = R.detect_perpetual_chase(chases('red', 6))
        self.assertTrue(violated)
        self.assertIn('长捉', reason)

    def test_twelve_chases_by_two_pieces_is_a_violation(self):
        seq = []
        for _ in range(6):
            seq.append(fact('red', piece='车A', chased=((5, 4),)))
            seq.append(fact('red', piece='车B', chased=((5, 4),)))
        violated, reason = R.detect_perpetual_chase(seq)
        self.assertTrue(violated, reason)
        self.assertIn('12', reason)

    def test_two_pieces_eleven_chases_is_allowed(self):
        seq = []
        for _ in range(5):
            seq.append(fact('red', piece='车A', chased=((5, 4),)))
            seq.append(fact('red', piece='车B', chased=((5, 4),)))
        seq.append(fact('red', piece='车A', chased=((5, 4),)))
        self.assertEqual(len(seq), 11)
        violated, _ = R.detect_perpetual_chase(seq)
        self.assertFalse(violated)

    def test_empty_chased_breaks_the_run(self):
        seq = chases('red', 5) + [fact('red')] + chases('red', 5)
        violated, _ = R.detect_perpetual_chase(seq)
        self.assertFalse(violated, '中间有一手没捉到子 -> 连续段断了')

    def test_chase_by_different_pieces_uses_the_two_piece_limit(self):
        """换了**捉的子**就换档：车 3 次 + 马 3 次 = 双子上限 12，还没到。

        （捉的对象换人不影响"连续在捉"，但参与长捉的**棋子数**决定上限。）
        """
        seq = ([fact('red', piece='车', chased=((5, 4),))] * 3
               + [fact('red', piece='马', chased=((6, 4),))] * 3)
        violated, reason = R.detect_perpetual_chase(seq)
        self.assertFalse(violated, reason)
        # 把它续到 12 次就该违规了
        long_seq = seq + [fact('red', piece='车', chased=((5, 4),))] * 3 \
            + [fact('red', piece='马', chased=((6, 4),))] * 3
        violated, reason = R.detect_perpetual_chase(long_seq)
        self.assertTrue(violated, reason)
        self.assertIn('12', reason)


class TestChaseExemption(unittest.TestCase):
    """豁免兵种：兵/卒/帅/将/仕/士/相/象可以长捉。"""

    def test_chase_exempt_set_is_exactly_the_rule_list(self):
        self.assertEqual(R.CHASE_EXEMPT,
                         {'兵', '卒', '帅', '将', '仕', '士', '相', '象'})

    def test_pawn_is_not_treated_as_chased(self):
        """红车能吃黑卒、也有保护，但卒**豁免** -> 不算捉。"""
        board = build_board(KINGS + [
            ((1, 3), R_, B.R, False),       # 红车
            ((1, 4), B_, B.P, False),       # 黑卒（被瞄准，豁免）
            ((0, 3), R_, B.P, False),       # 红兵保护 (1,3) 落点
        ])
        self.assertEqual(R.chase_targets(board, side='red'), ())

    def test_rook_is_chased_in_the_same_position(self):
        """同样的盘面，把卒换成車就**算**捉 —— 证明豁免是唯一差别。"""
        board = build_board(KINGS + [
            ((1, 3), R_, B.R, False),
            ((1, 4), B_, B.R, False),       # 换成黑車
            ((0, 3), R_, B.P, False),
        ])
        self.assertEqual(R.chase_targets(board, side='red'), ((1, 4),))

    def test_exempt_piece_can_chase_without_limit(self):
        """兵卒长捉**自己**不违规：连续 8 次捉子也不判长捉。"""
        seq = chases('red', 8, piece='兵')
        violated, reason = R.detect_perpetual_chase(seq)
        self.assertFalse(violated, reason)

    def test_every_exempt_char_can_chase_without_limit(self):
        """八个豁免兵种每一个都试一遍（车/马/炮不豁免，作为对照）。"""
        for ch in sorted(R.CHASE_EXEMPT):
            seq = chases('red', 8, piece=ch)
            violated, reason = R.detect_perpetual_chase(seq)
            self.assertFalse(violated, '%s 长捉不该判违规：%s' % (ch, reason))
        for ch in ('车', '馬', '炮', '砲'):
            seq = chases('red', 8, piece=ch)
            violated, _ = R.detect_perpetual_chase(seq)
            self.assertTrue(violated, '%s 长捉 8 次应当判违规' % ch)

    def test_all_exempt_chars_are_skipped_on_board(self):
        """八个豁免兵种逐一看盘面判定都返回"豁免"。"""
        pairs = [(B.P, R_, 'red'), (B.K, None, 'red'), (B.A, R_, 'red'),
                 (B.E, R_, 'red')]
        for ident, _, _ in pairs:
            board = build_board(KINGS + [((1, 3), R_, ident, False)])
            self.assertTrue(R._chase_exempt(board, sq_of(1, 3)),
                            '红方 %s 应当豁免长捉' % B.IDENT_CHARS[(R_, ident)])
        for ident in (B.P, B.K, B.A, B.E):
            board = build_board(KINGS + [((8, 3), B_, ident, False)])
            self.assertTrue(R._chase_exempt(board, sq_of(8, 3)),
                            '黑方 %s 应当豁免长捉' % B.IDENT_CHARS[(B_, ident)])

    def test_hidden_pawn_uses_home_identity_for_exemption(self):
        """暗子按**摆位身份**判豁免：摆在卒位上的暗子按卒算（豁免），
        摆在車位上的暗子按車算（不豁免）——证明用的是摆位身份。"""
        board_pawn = build_board(KINGS + [((6, 0), B_, B.R, True)])   # 暗卒在卒位
        self.assertEqual(B.HOME_IDENT[sq_of(6, 0)], B.P)
        self.assertTrue(R._chase_exempt(board_pawn, sq_of(6, 0)))
        # 黑方車位在底排（第 9 行）：找出那一格，确认它不豁免
        rook_sq = next(s for s in range(90)
                       if B.HOME_IDENT[s] == B.R and s // 9 == 9)
        board_rook = build_board(KINGS + [((rook_sq // 9, rook_sq % 9), B_,
                                           B.E, True)])
        self.assertFalse(R._chase_exempt(board_rook, rook_sq),
                         '(行%d,列%d) 是車位，暗子按車算，不豁免'
                         % (rook_sq // 9, rook_sq % 9))


class TestChaseJudgement(unittest.TestCase):
    """捉的判据本身：要"能吃到 + 落点有人保护"，互捉不算捉。"""

    def test_mutual_attack_is_not_a_chase(self):
        """我在吃你的车、你的车同时也能白吃我 -> 谁也不欠谁，不算捉。"""
        board = build_board(KINGS + [
            ((1, 2), R_, B.R, False),       # 红车
            ((1, 3), B_, B.R, False),       # 黑車（红车能吃它）
        ])
        self.assertEqual(R.chase_targets(board, side='red'), ())

    def test_unprotected_target_is_not_a_chase(self):
        """能吃但**没有保护**（换子而已）-> 不算捉。"""
        board = build_board(KINGS + [
            ((1, 2), R_, B.R, False),       # 红车
            ((1, 3), B_, B.R, False),       # 黑車
            ((1, 7), B_, B.R, False),       # 黑車来保护（红落点没人保护）
        ])
        self.assertEqual(R.chase_targets(board, side='red'), ())

    def test_protected_capture_is_a_chase(self):
        """红车吃黑車、落点有红兵保护（对方吃回来亏）-> 算捉。"""
        board = build_board(KINGS + [
            ((1, 3), R_, B.R, False),
            ((1, 4), B_, B.R, False),
            ((0, 3), R_, B.P, False),
        ])
        self.assertEqual(R.chase_targets(board, side='red'), ((1, 4),))

    def test_hanging_attacker_is_not_a_chase(self):
        """攻击者自己被对方白吃 -> 不算"捉"（它得先顾自己）。

        盘面：红车 (1,3) 想吃黑車 (2,3)，但红车 (1,3) 同时被**黑卒 (1,5)**
        （卒 = 100，比红车便宜得多）白吃，而红方没有任何子能保护 (1,3)
        —— 这就是"互捉"，谁也不欠谁，所以 (2,3) 不算被捉。

        ⚠ 黑車刻意摆在 (2,3) 而不是 (1,4)：摆在 (1,4) 的话**红帅 (0,4)**
        自己就能吃它，而那一步是安全的（红车保护着 (0,4)），"捉"会由红帅成立，
        把要测的东西遮住。所以这里只断言 (2,3) 不在被捉名单里——(1,4) 那一枚
        是不是被捉与"红车自身难保"无关（红帅盯着它）。
        """
        board = build_board(KINGS + [
            ((1, 3), R_, B.R, False),       # 红车（自己正被白吃）
            ((1, 4), B_, B.R, False),       # 黑車（红帅盯着它）
            ((2, 3), B_, B.R, False),       # 黑車（红车想吃它，但红车自身难保）
            ((1, 5), B_, B.P, False),       # 黑卒白吃红车：100 换 900
        ])
        self.assertNotIn((2, 3), R.chase_targets(board, side='red'))

    def test_protected_attacker_does_chase(self):
        """同一个盘面 + 红马保护红车 -> 红车不再被白吃，(2,3) 的黑車成了被捉。

        （(1,4) 的黑車本来就被红帅盯着，与"红车自身难保"无关，所以两次都在。）
        """
        board = build_board(KINGS + [
            ((1, 3), R_, B.R, False),
            ((1, 4), B_, B.R, False),
            ((2, 3), B_, B.R, False),
            ((1, 5), B_, B.P, False),
            ((4, 2), R_, B.H, False),       # 红马保护 (2,3)（红车走进去的地方）
        ])
        self.assertEqual(R.chase_targets(board, side='red'), ((1, 4), (2, 3)))

    def test_chase_targets_accepts_gamestate(self):
        """``chase_targets`` 也要能吃 ``GameState``（走 GUI 坐标那一路）。"""

        class FakeGameState:
            pass

        gs = FakeGameState()
        gs.board = gui_board_blank()
        gs.board[0][4] = gui_piece('red', '帅')
        gs.board[9][4] = gui_piece('black', '将')
        gs.board[1][3] = gui_piece('red', '车')
        gs.board[1][4] = gui_piece('black', '車')
        gs.board[0][3] = gui_piece('red', '兵')
        gs.current_player = 'red'
        self.assertEqual(R.chase_targets(gs), ((1, 4),))

    def test_after_move_flag_is_accepted(self):
        """``after_move`` 参数保留（接口兼容），不影响结论。"""
        board = build_board(KINGS + [
            ((1, 3), R_, B.R, False),
            ((1, 4), B_, B.R, False),
            ((0, 3), R_, B.P, False),
        ])
        self.assertEqual(R.chase_targets(board, after_move=True, side='red'),
                         R.chase_targets(board, after_move=False, side='red'))


# ==========================================================================
# 3. 无吃子（闲着）步数
# ==========================================================================
class TestIdlePlyCount(unittest.TestCase):
    """规则 4：吃子清零、翻子清零、将军与应将不计数也不清零、其余累加。"""

    def test_quiet_moves_accumulate(self):
        self.assertEqual(R.idle_ply_count([fact('red')] * 7), 7)

    def test_capture_resets(self):
        seq = [fact('red')] * 5 + [fact('black', captured=True)] + [fact('red')] * 3
        self.assertEqual(R.idle_ply_count(seq), 3)

    def test_flip_resets(self):
        """**翻暗子清零**：翻子改变了信息量，不算闲着，且把之前的计数抹掉。"""
        seq = [fact('red')] * 5 + [fact('red', flipped=True)] + [fact('black')] * 2
        self.assertEqual(R.idle_ply_count(seq), 2)

    def test_check_neither_counts_nor_resets(self):
        seq = [fact('red')] * 5 + [fact('red', checked=True)] + [fact('black')] * 2
        self.assertEqual(R.idle_ply_count(seq), 7)

    def test_responding_to_check_neither_counts_nor_resets(self):
        """应将（走这一步之前正被将军）同样不计数、不清零。"""
        seq = [fact('red')] * 5 + [fact('black', in_check=True)] + [fact('red')] * 2
        self.assertEqual(R.idle_ply_count(seq), 7)

    def test_capture_after_checks_still_resets(self):
        seq = ([fact('red')] * 5 + [fact('red', checked=True)]
               + [fact('black', in_check=True)]
               + [fact('red', captured=True)] + [fact('red')] * 2)
        self.assertEqual(R.idle_ply_count(seq), 2)

    def test_eighty_idle_plies_reaches_the_draw_line(self):
        self.assertFalse(R.idle_draw_reached(79))
        self.assertTrue(R.idle_draw_reached(80))
        self.assertEqual(R.RULE['idle_draw_plies'], 80)

    def test_flip_clears_a_long_streak(self):
        seq = [fact('red')] * 79 + [fact('red', flipped=True)]
        self.assertEqual(R.idle_ply_count(seq), 0)


# ==========================================================================
# 4. 无过河子力判和
# ==========================================================================
class TestInsufficientMaterial(unittest.TestCase):
    """双方都没有"可以过河"的子力 -> 判和（用户确认的口径）。"""

    def test_only_kings_is_a_draw(self):
        board = build_board(KINGS)
        self.assertTrue(R.insufficient_material(board))
        self.assertEqual(R.can_cross_river(board), {'red': False, 'black': False})

    def test_pieces_on_own_half_only_may_still_cross(self):
        """红兵在 (4,4)（未过河）**能**向前过河 -> 不判和。"""
        board = build_board(KINGS + [((4, 4), R_, B.P, False)])
        self.assertTrue(R.can_cross_river(board)['red'])
        self.assertFalse(R.insufficient_material(board))

    def test_pawn_on_own_home_row_cannot_cross(self):
        """红兵还在底线 (0,0)：兵只能向前，下一步仍在自家半场 -> 过不了河。"""
        board = build_board(KINGS + [((0, 0), R_, B.P, False)])
        self.assertFalse(R.can_cross_river(board)['red'])
        self.assertTrue(R.insufficient_material(board))

    def test_pawn_past_the_river_still_can_cross(self):
        """已经过河的红兵当然还"能过河"（它横一步仍在对方半场）。"""
        board = build_board(KINGS + [((5, 3), R_, B.P, False)])
        self.assertTrue(R.can_cross_river(board)['red'])
        self.assertFalse(R.insufficient_material(board))

    def test_hidden_advisor_cannot_cross(self):
        """**暗**仕按摆位身份走中国象棋规则：出不了九宫 -> 过不了河。"""
        board = build_board(KINGS + [((0, 3), R_, B.A, True)])
        self.assertFalse(R.can_cross_river(board)['red'])
        self.assertTrue(R.insufficient_material(board))

    def test_revealed_advisor_can_cross(self):
        """**明**仕可以过河（权威规则："揭开后的士、象可以过河"）-> 不判和。

        摆在 (4,4) 的明仕能斜走到 (5,3)/(5,5)，那两格已经在黑方半场。
        ⚠ 这里刻意把将帅摆在不同纵线上：揭棋里两将同线且中间无子会触发
        "对面笑"（将帅照面），那种局面下引擎会把几乎所有着法都判成自将
        （这是引擎**正确**的行为），拿来测"能不能过河"只会得到假阴性。
        """
        board = build_board([
            ((0, 3), R_, B.K, False), ((9, 7), B_, B.K, False),
            ((4, 4), R_, B.A, False),
        ])
        self.assertTrue(R.can_cross_river(board)['red'])
        self.assertFalse(R.insufficient_material(board))

    def test_revealed_elephant_can_cross(self):
        """**明**象同样可以过河。"""
        board = build_board([
            ((0, 3), R_, B.K, False), ((9, 7), B_, B.K, False),
            ((4, 2), R_, B.E, False),
        ])
        self.assertTrue(R.can_cross_river(board)['red'])
        self.assertFalse(R.insufficient_material(board))

    def test_hidden_elephant_needs_a_crossing_target(self):
        """**暗**象在 (0,2)：能到的四个田字点全在自家半场 -> 过不了河。"""
        board = build_board(KINGS + [((0, 2), R_, B.E, True)])
        self.assertFalse(R.can_cross_river(board)['red'])

    def test_black_pawn_on_own_baseline_cannot_cross(self):
        """黑卒在自家底线 (9,0)：只能向前，下一步还在自家半场。"""
        board = build_board(KINGS + [((9, 0), B_, B.P, False)])
        self.assertFalse(R.can_cross_river(board)['black'])
        self.assertTrue(R.insufficient_material(board))

    def test_black_pawn_past_the_river_can_cross(self):
        board = build_board(KINGS + [((4, 3), B_, B.P, False)])
        self.assertTrue(R.can_cross_river(board)['black'])

    def test_hidden_black_pawn_not_yet_across(self):
        """摆位在 (6,0) 的暗子按卒走：只能走到 (5,0)，还在自家半场。"""
        board = build_board(KINGS + [((6, 0), B_, B.E, True)])
        self.assertEqual(B.HOME_IDENT[sq_of(6, 0)], B.P, '前提：(6,0) 是卒位')
        self.assertFalse(R.can_cross_river(board)['black'])

    def test_one_side_crossing_is_enough_to_continue(self):
        """只要**任意一方**还能过河就不判和。"""
        board = build_board(KINGS + [((0, 3), R_, B.A, True),
                                     ((4, 3), B_, B.P, False)])
        self.assertFalse(R.can_cross_river(board)['red'])
        self.assertTrue(R.can_cross_river(board)['black'])
        self.assertFalse(R.insufficient_material(board))

    def test_hidden_pawn_can_cross_using_home_identity(self):
        """暗子按**摆位身份**判走法：摆在車位上的暗子按車走（能过河），
        摆在卒位上的暗子按卒走（还不能过河）——证明用的是摆位身份。

        (9,0) 是黑方車位、(6,0) 是黑方卒位；两盘都放在**不同纵线**的将帅下，
        避免"对面笑"把着法全滤掉。
        """
        kings = [((0, 3), R_, B.K, False), ((9, 7), B_, B.K, False)]
        board_rook = build_board(kings + [((9, 0), B_, B.E, True)])   # 暗車
        self.assertEqual(B.HOME_IDENT[sq_of(9, 0)], B.R)
        self.assertTrue(R.can_cross_river(board_rook)['black'])
        board_pawn = build_board(kings + [((6, 0), B_, B.R, True)])   # 暗卒
        self.assertEqual(B.HOME_IDENT[sq_of(6, 0)], B.P)
        self.assertFalse(R.can_cross_river(board_pawn)['black'])

    def test_unavailable_board_is_treated_as_playable(self):
        """拿不到局面时保守认为"还能过河"，绝不因为算不出来把棋判死。"""
        self.assertFalse(R.insufficient_material(None))


# ==========================================================================
# 5. 重复局面 / 最大步数
# ==========================================================================
class TestRepetitionRule(unittest.TestCase):
    """局面循环：本项目沿用 **6 次**判和，第 4 次起提醒。"""

    def test_thresholds_match_configuration(self):
        self.assertEqual(R.RULE['repetition_draw'], 6)
        self.assertEqual(R.RULE['repetition_warn'], 4)

    def test_below_warn_threshold_is_silent(self):
        verdict = R.repetition_verdict(3)
        self.assertFalse(verdict['draw'])
        self.assertFalse(verdict['warn'])
        self.assertEqual(verdict['message'], '')

    def test_fourth_occurrence_warns(self):
        verdict = R.repetition_verdict(4)
        self.assertFalse(verdict['draw'], '第 4 次只提醒，不判和')
        self.assertTrue(verdict['warn'])
        self.assertIn('变招', verdict['message'])
        self.assertIn('2', verdict['message'])          # 距判和还差 2 次

    def test_fifth_occurrence_still_only_warns(self):
        verdict = R.repetition_verdict(5)
        self.assertFalse(verdict['draw'])
        self.assertTrue(verdict['warn'])

    def test_sixth_occurrence_is_a_draw(self):
        verdict = R.repetition_verdict(6)
        self.assertTrue(verdict['draw'])
        self.assertIn('6', verdict['message'])

    def test_repetition_count_is_read_only(self):
        counter = {'abc': [1, 3, 5]}
        self.assertEqual(R.repetition_count(counter, 'abc'), 3)
        self.assertEqual(counter, {'abc': [1, 3, 5]}, '读计数不能改变它')
        self.assertEqual(R.repetition_count(counter, 'nope'), 0)


class TestMoveLimit(unittest.TestCase):
    def test_boundary(self):
        self.assertFalse(R.move_limit_reached(199))
        self.assertTrue(R.move_limit_reached(200))
        self.assertTrue(R.move_limit_reached(201))

    def test_rule_constant(self):
        self.assertEqual(R.RULE['max_plies'], 200)


# ==========================================================================
# 6. GameState 集成
# ==========================================================================
class TestGameStateIntegration(unittest.TestCase):
    """``GameState``：判和条件接上、``rule_status`` 暴露违规方、检查必须幂等。"""

    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        os.chdir(GAME_DIR)
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        cls._tmp.cleanup()

    def setUp(self):
        from unittest import mock

        from game_state import GameState

        self.gs = GameState(None)
        self.gs.auto_save_record = lambda: False        # 不写盘（records/ 绝不碰）
        self._patch = mock.patch("jieqi.settings.load_raw_config",
                                 return_value={"record": {"auto_save": False}})
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _play(self, plies=6, seed=7):
        from jieqi.gui_adapter import gui_color_to_side, legal_moves

        rng = random.Random(seed)
        for _ in range(plies):
            side = gui_color_to_side(self.gs.current_player)
            moves = legal_moves(self.gs.board, side)
            if not moves:
                break
            mv = rng.choice(moves)
            self.gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                               (mv & 127) // 9, (mv & 127) % 9)
            self.gs.current_player = ('black' if self.gs.current_player == 'red'
                                      else 'red')
            self.gs.check_game_state()
        return self.gs

    # ---------------------------------------------------------- 幂等
    def test_check_game_state_is_idempotent(self):
        """连续检查 100 次，``position_counter`` 与判和状态都不能变（原 bug）。"""
        gs = self._play(6)
        before = {k: list(v) for k, v in gs.position_counter.items()}
        for _ in range(100):
            gs.check_game_state()
        after = {k: list(v) for k, v in gs.position_counter.items()}
        self.assertEqual(before, after, '判和检测有副作用（position_counter 被改了）')
        self.assertFalse(gs.game_over, '正常开中局不该被判和')

    def test_fold_repetition_check_is_read_only(self):
        gs = self._play(4)
        sig = gs._board_hash()
        before = list(gs.position_counter.get(sig, []))
        for _ in range(50):
            gs._check_fold_repetition()
        self.assertEqual(list(gs.position_counter.get(sig, [])), before)

    def test_move_piece_只累积一次计数(self):
        """计数只在**真的走了一步**时增加（一次一手）。"""
        gs = self.gs
        from jieqi.gui_adapter import gui_color_to_side, legal_moves

        side = gui_color_to_side(gs.current_player)
        mv = legal_moves(gs.board, side)[0]
        total_before = sum(len(v) for v in gs.position_counter.values())
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9, (mv & 127) // 9, (mv & 127) % 9)
        total_after = sum(len(v) for v in gs.position_counter.values())
        self.assertEqual(total_after, total_before + 1)

    # ---------------------------------------------------- 判和条件
    def test_idle_draw_reason_is_chinese(self):
        gs = self.gs
        gs.rule_facts = [fact('red') for _ in range(R.RULE['idle_draw_plies'])]
        gs.move_history = [{} for _ in range(R.RULE['idle_draw_plies'])]
        reason = R.evaluate_draws(gs, gs.rule_facts)
        self.assertIsNotNone(reason)
        self.assertEqual(reason['rule'], 'idle_plies')
        self.assertIn('未吃子', reason['reason'])
        self.assertIn('80', reason['reason'])

    def test_insufficient_material_draw_reason(self):
        gs = self.gs
        gs.board = gui_board_blank()
        gs.board[0][4] = gui_piece('red', '帅')
        gs.board[9][4] = gui_piece('black', '将')
        gs.move_history = []
        gs.rule_facts = []
        reason = R.evaluate_draws(gs)
        self.assertIsNotNone(reason)
        self.assertEqual(reason['rule'], 'insufficient_material')
        self.assertIn('过河', reason['reason'])

    def test_repetition_draw_reason(self):
        gs = self.gs
        sig = gs._board_hash()
        gs.position_counter = {sig: list(range(R.RULE['repetition_draw']))}
        reason = R.evaluate_draws(gs)
        self.assertIsNotNone(reason)
        self.assertEqual(reason['rule'], 'repetition')
        self.assertIn('重复', reason['reason'])

    def test_max_plies_reason(self):
        gs = self.gs
        gs.board = gui_board_blank()
        # 双方各留一枚"还能过河"的子，避免先撞上"无过河子力判和"；
        # 将帅放在**不同纵线**上，免得"对面笑"把着法全滤掉。
        gs.board[0][3] = gui_piece('red', '帅')
        gs.board[9][5] = gui_piece('black', '将')
        gs.board[0][0] = gui_piece('red', '仕', revealed=False)   # 暗仕在仕位
        gs.board[9][0] = gui_piece('black', '車', revealed=False)  # 暗車在車位
        gs.position_counter = {}
        # ⚠ 事实用"将军"而不是"闲着"：判和条件有优先级，"无吃子 80 步"排在
        # "最大步数"前面，喂 200 个闲着会先撞上 80 步判和。
        gs.rule_facts = [fact('red', checked=True)
                         for _ in range(R.RULE['max_plies'])]
        gs.move_history = [{} for _ in range(R.RULE['max_plies'])]
        gs.positions = [gs._snapshot_state()] * (R.RULE['max_plies'] + 1)
        self.assertFalse(R.insufficient_material(gs), '前提：还有能过河的子力')
        reason = R.evaluate_draws(gs, gs.rule_facts)
        self.assertIsNotNone(reason)
        self.assertEqual(reason['rule'], 'max_plies')
        self.assertIn('200', reason['reason'])

    def test_idle_draw_triggers_check_game_state(self):
        """把闲着计数推到 80 步，``check_game_state`` 应当真的判和。"""
        gs = self.gs
        # 先造一个"双方都还有能过河的子力"的局面（否则会先撞上无过河子力判和），
        # 再将帅摆在不同纵线上，避免"对面笑"。
        gs.board = gui_board_blank()
        gs.board[0][3] = gui_piece('red', '帅')
        gs.board[9][5] = gui_piece('black', '将')
        gs.board[0][0] = gui_piece('red', '仕', revealed=False)
        gs.board[9][0] = gui_piece('black', '車', revealed=False)
        # 直接喂事实：从开局到现在全是闲着（每手都不吃子不翻子不将军）
        gs.rule_facts = [fact('red') for _ in range(R.RULE['idle_draw_plies'])]
        gs.move_history = [{} for _ in range(R.RULE['idle_draw_plies'])]
        gs.ply = len(gs.rule_facts)
        gs.check_game_state()
        self.assertGreaterEqual(R.idle_ply_count(gs.rule_facts),
                                R.RULE['idle_draw_plies'])
        self.assertTrue(gs.game_over)
        self.assertEqual(gs.winner, 'draw')
        self.assertIn('未吃子', gs.draw_reason)

    # ---------------------------------------------------- rule_status
    def test_rule_status_shape_on_normal_position(self):
        gs = self._play(6)
        status = gs.rule_status()
        self.assertEqual(set(status), {'forbidden', 'offender', 'reason'})
        self.assertFalse(status['forbidden'])
        self.assertIsNone(status['offender'])

    def test_rule_status_reports_offender(self):
        """注入"红方连续将军 6 次" -> 必须指出红方违规并给出中文原因。"""
        gs = self.gs
        gs.rule_facts = checks('red', 6, piece='车')
        gs.move_history = [{} for _ in range(6)]
        gs.ply = 6
        status = gs.rule_status()
        self.assertTrue(status['forbidden'])
        self.assertEqual(status['offender'], 'red')
        self.assertIn('长将', status['reason'])
        self.assertIn('红', status['reason'])

    def test_rule_status_toast_is_loud_and_deduped(self):
        """违规要大声提醒，但**同一句话不反复刷**（每帧都会调）。"""
        gs = self.gs
        gs.rule_facts = chases('black', 6, piece='車')
        gs.move_history = [{} for _ in range(6)]
        gs.ply = 6
        gs.check_game_state()
        self.assertTrue(gs.toast_loud)
        self.assertIn('黑', gs.toast)
        gs.set_toast('别的提示')
        gs.check_game_state()
        self.assertEqual(gs.toast, '别的提示',
                         '同一句规则提醒不该每帧重新弹一次')

    def test_rule_facts_are_recorded_每走一步(self):
        gs = self._play(4)
        self.assertEqual(len(gs.rule_facts), len(gs.move_history))
        for f, rec in zip(gs.rule_facts, gs.move_history):
            self.assertEqual(f.mover, 'red' if rec.get('side') in (0, 'red')
                             else 'black')
            self.assertEqual(f.captured, bool(rec.get('captured')))
            self.assertEqual(f.flipped, not rec['piece']['revealed'])

    def test_facts_from_game_state_falls_back_to_records(self):
        """没有存事实时（老棋谱）退回按走棋记录还原，不抛异常。"""
        gs = self._play(4)
        saved, gs.rule_facts = gs.rule_facts, None
        try:
            facts = R.facts_from_game_state(gs)
        finally:
            gs.rule_facts = saved
        self.assertEqual(len(facts), 4)
        self.assertTrue(all(f.mover in ('red', 'black') for f in facts))


# ==========================================================================
# 7. 引擎层：剔除违规着法
# ==========================================================================
class TestEngineForbiddenMoves(unittest.TestCase):
    """AI 在根节点必须避开会导致自己长将 / 长捉违规的着法。"""

    def _check_loop_board(self):
        """红車 (0,3)、红車 (1,3) 都能横走到 (0,4)/(1,4) 将军黑将。

        黑帅位在 (9,4)，两条通道都空着，所以红方"平车"就是一步将军。
        """
        return build_board(KINGS + [
            ((0, 3), R_, B.R, False),
            ((1, 3), R_, B.R, False),
        ])

    def test_forbidden_moves_detects_the_seventh_check(self):
        board = self._check_loop_board()
        side = 'red'
        # 已经连续将军 5 次（同一枚车）-> 再将军就违规
        history = checks('red', 5, piece='车')
        forbidden = R.forbidden_moves(history, board, side)
        self.assertTrue(forbidden, '应当能识别出"再将军就违规"的着法')
        # 这些着法必须真的是将军
        for mv in forbidden:
            after = board.copy()
            after.make(mv)
            self.assertTrue(after.in_check(B_), B.move_str(mv))

    def test_no_history_means_nothing_is_forbidden(self):
        board = self._check_loop_board()
        self.assertEqual(R.forbidden_moves([], board, 'red'), [])

    def test_non_checking_move_is_never_forbidden_by_check_rule(self):
        """不将军也不捉子的着法不可能触犯这两条 -> 一条都不该被禁。"""
        board = build_board(KINGS + [((5, 0), R_, B.R, False)])
        history = checks('red', 5, piece='车')
        for mv in R.forbidden_moves(history, board, 'red'):
            after = board.copy()
            after.make(mv)
            self.assertTrue(after.in_check(B_), B.move_str(mv))

    def test_fact_after_move_matches_board(self):
        board = self._check_loop_board()
        mv = B.encode(sq_of(0, 3), sq_of(0, 4))
        f = R.fact_after_move(board, mv, 'red')
        self.assertEqual(f.mover, 'red')
        self.assertTrue(f.checked)
        self.assertFalse(f.captured)
        self.assertFalse(f.flipped)


class TestEngineAnalyseRuleFacts(unittest.TestCase):
    """``Engine.analyse`` 的新入参 ``rule_facts``：传了才起作用，不传完全不变。"""

    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def _engine(self, **kw):
        from jieqi.engine import Engine

        return Engine(difficulty='easy', time_limit=0.4, samples=1,
                      max_depth=2, seed=1234, **kw)

    def _board(self, plies=0, seed=5):
        pub = B.Board.new_game(random.Random(seed))
        rng = random.Random(seed)
        for _ in range(plies):
            moves = pub.gen_moves(pub.turn)
            if not moves:
                break
            pub.make(rng.choice(moves))
        return pub

    def test_analyse_without_rule_facts_is_unchanged(self):
        """不传 ``rule_facts`` 时行为与以前完全一致（可复现的着法）。"""
        board = self._board(6)
        eng = self._engine()
        first = eng.analyse(board.clone_revealed_only())
        second = self._engine().analyse(board.clone_revealed_only())
        self.assertIsNotNone(first.move)
        self.assertEqual(first.move, second.move, '同一 seed 应当给出同一着法')
        self.assertNotIn('规则', ' '.join(first.notes))

    def test_analyse_accepts_rule_facts(self):
        """传 ``rule_facts`` 时正常出招，且能读出 notes。"""
        board = self._board(6)
        facts = [fact('red') for _ in range(4)]
        result = self._engine().analyse(board.clone_revealed_only(),
                                       rule_facts=facts)
        self.assertIsNotNone(result.move, '给规则事实后也必须给出着法')
        self.assertIsInstance(result.notes, list)

    def test_analyse_drops_forbidden_rule_moves(self):
        """让红方"再将军就违规"时，引擎选出的着法不能是那个违规着法。"""
        board = build_board(KINGS + [
            ((0, 3), R_, B.R, False),
            ((1, 3), R_, B.R, False),
        ])
        board.turn = R_
        history = checks('red', 5, piece='车')
        forbidden = set(R.forbidden_moves(history, board, 'red'))
        self.assertTrue(forbidden, '前提：确实存在会被禁的着法')
        pub = board.clone_revealed_only()
        result = self._engine().analyse(pub, rule_facts=history)
        self.assertIsNotNone(result.move)
        if result.move in forbidden:
            self.fail('引擎选了会导致长将违规的着法 %s' % B.move_str(result.move))
        self.assertTrue(any('规则' in n or '长将' in n or '禁' in n
                            for n in result.notes),
                        '剔除违规着法后应当留下 notes 说明：%r' % result.notes)

    def test_analyse_keeps_forbidden_move_when_it_is_the_only_one(self):
        """若无着法可选，被禁的那一步**照样要能走**（总不能让 AI 停在那儿）。

        构造：红车 (2,3) 被两枚黑車 (2,1) / (2,4) 夹住，红方唯一的**机动**就是
        吃 (2,4)（另两步是帅挪窝）；而吃将在"红方已连续将军 5 次"的历史下会被
        判成长将禁招。
        """
        board = build_board([
            ((0, 4), R_, B.K, False), ((9, 4), B_, B.K, False),
            ((2, 1), B_, B.R, False), ((2, 3), R_, B.R, False),
            ((2, 4), B_, B.R, False),
        ])
        board.turn = R_
        legal = board.gen_moves(R_)
        history = checks('red', 5, piece='车')
        forbidden = set(R.forbidden_moves(history, board, 'red'))
        self.assertIn(B.encode(sq_of(2, 3), sq_of(2, 4)), forbidden,
                      '前提：吃将这步会被判成长将禁招')
        result = self._engine().analyse(board.clone_revealed_only(),
                                        rule_facts=history)
        self.assertIsNotNone(result.move, '没有别的着法时必须还能走')
        self.assertIn(result.move, legal, '走出来的必须是合法着法')
        self.assertTrue(any('禁' in n or '长将' in n for n in result.notes),
                        '剔除过违规着法就该在 notes 里说明：%r' % result.notes)

    def test_analyse_keeps_the_rule_move_when_every_move_is_forbidden(self):
        """**全部**合法着法都被禁时也得照样出招（这一步只验证不崩、不返回 None）。"""
        board = build_board([
            ((0, 4), R_, B.K, False), ((9, 4), B_, B.K, False),
            ((2, 1), B_, B.R, False), ((2, 3), R_, B.R, False),
            ((2, 4), B_, B.R, False),
        ])
        board.turn = R_
        legal = set(board.gen_moves(R_))
        # 用"连续将军 5 次"+"每一步都是将军"的历史把所有着法都逼成禁招
        history = checks('red', 5, piece='车')
        blocked = set(R.forbidden_moves(history, board, 'red'))
        result = self._engine().analyse(board.clone_revealed_only(),
                                        rule_facts=history)
        self.assertIsNotNone(result.move)
        self.assertIn(result.move, legal)
        # 只要存在没被禁的着法，选出来的就必须是其中之一
        if legal - blocked:
            self.assertNotIn(result.move, blocked,
                             '还有不被禁的着法时不该走禁招')


if __name__ == '__main__':
    unittest.main(verbosity=2)
