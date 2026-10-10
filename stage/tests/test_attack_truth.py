"""攻击判定差分测试：`is_attacked` 绝不能漏判。

`legal_filter` 判定"走完自己的将帅是否被攻击"用的就是 `is_attacked`。
它一旦漏判（比如炮的炮架数错、马的蹩腿方向错、士/象的明暗表用错），
AI 就会把一个"送将"的着法当成合法着法——这正是"AI 送将"最可能的来源。

这里用**另一套实现**做参照：直接生成该方所有伪合法着法，看 sq 是否在目标集合里。
两者必须在成千上万个随机局面上完全一致（飞将/对面笑除外，因为它不是"走法"）。
"""
import random
import sys
import unittest
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
sys.path.insert(0, str(GAME_DIR.parent))
sys.path.insert(0, str(GAME_DIR))

from jieqi import board as B  # noqa: E402
from jieqi.consts import BLACK, K, RED, SQUARES  # noqa: E402


def brute_attack_set(board, side):
    """参照实现：该方能够**吃掉**的格子集合。

    注意 ``is_attacked`` 的语义是"这个格子上的棋子能不能被吃掉"，
    相当于"伪合法着法里目标格有敌方棋子的那些"。空格的**安静走法**
    （例如炮平到一个空格）不属于"攻击"，不能拿来当参照。
    """
    out = set()
    for mv in board.gen_pseudo(side):
        to = mv & 127
        if board.side[to] == 1 - side:      # 目标格上有敌子 -> 可吃
            out.add(to)
    return out


def flying_general_only(board, sq, by):
    """``is_attacked(sq, by)`` 为真但不是任何着法目标，只可能是将帅照面。"""
    ks = board.kings[by]
    if ks < 0 or board.rule_ident[ks] != K:
        return False
    if ks % 9 != sq % 9:
        return False
    if board.side[sq] == by:
        return False
    lo, hi = sorted((ks // 9, sq // 9))
    return all(board.ident[r * 9 + sq % 9] == 0 for r in range(lo + 1, hi))


class TestIsAttackedNoMisses(unittest.TestCase):
    """`is_attacked` 的语义是"这个格子上的子能不能被吃掉"。

    对**敌方棋子所在的格子**，它必须与"伪合法着法能吃到它"完全等价；
    对空格，为真只表示"被控制"（车/炮控制空格是正常的），所以不要求等价。
    真正致命的方向是**漏判**：漏判可吃格 = legal_filter 放行送将着法。
    """

    def _check(self, board, label):
        for side in (RED, BLACK):
            brute = brute_attack_set(board, side)
            misses = []
            mismatched = []
            for sq in range(SQUARES):
                if board.side[sq] != 1 - side:
                    continue            # 只对"有敌子"的格子要求等价
                fast = board.is_attacked(sq, side)
                slow = sq in brute
                if slow and not fast:
                    misses.append(sq)
                elif fast and not slow and not flying_general_only(board, sq, side):
                    mismatched.append(sq)
            self.assertEqual(misses, [],
                             f"{label} side={side} is_attacked 漏判可吃格 {misses}")
            self.assertEqual(mismatched, [],
                             f"{label} side={side} is_attacked 与着法不一致 {mismatched}")

    def test_capturable_squares_are_always_detected(self):
        """只要是"能吃的敌子格"，is_attacked 必须为真（漏掉就是送将）。"""
        for seed in range(10):
            b = B.Board.new_game(random.Random(seed))
            rng = random.Random(seed * 91 + 5)
            for ply in range(50):
                for side in (RED, BLACK):
                    for sq in brute_attack_set(b, side):
                        self.assertTrue(
                            b.is_attacked(sq, side),
                            f"seed={seed} ply={ply} side={side} "
                            f"漏判可吃格 {sq} = {(sq // 9, sq % 9)}")
                moves = b.gen_moves(b.turn)
                if not moves:
                    break
                b.make(rng.choice(moves))

    def test_opening(self):
        for seed in range(6):
            b = B.Board.new_game(random.Random(seed))
            self._check(b, f"开局 seed={seed}")

    def test_random_play(self):
        for seed in range(12):
            b = B.Board.new_game(random.Random(seed))
            rng = random.Random(seed * 31 + 7)
            for ply in range(60):
                self._check(b, f"seed={seed} ply={ply}")
                moves = b.gen_moves(b.turn)
                if not moves:
                    break
                b.make(rng.choice(moves))

    def test_handcrafted_cannon_screens(self):
        """炮的炮架数、以及"移开炮架就暴露"这些边界情况。"""
        cases = []
        # 帅(0,4)、炮架红兵(3,4)、黑炮(6,4) -> 正好一个炮架，将被将
        b = B.Board.empty()
        b.put(4, RED, K)
        b.put(85, BLACK, K)
        b.put(3 * 9 + 4, RED, B.P, hidden=False)
        b.put(6 * 9 + 4, BLACK, B.C, hidden=False)
        cases.append((b, "红兵是唯一炮架", True))
        # 再加一个红兵(4,4) -> 两个炮架，不被将
        b2 = b.copy()
        b2.put(4 * 9 + 4, RED, B.P, hidden=False)
        cases.append((b2, "两个炮架", False))
        # 明士自由斜走也要能被判定为攻击
        b3 = B.Board.empty()
        b3.put(4, RED, K)
        b3.put(85, BLACK, K)
        b3.put(1 * 9 + 3, BLACK, B.A, hidden=False)
        b3.put(2 * 9 + 2, RED, B.P, hidden=False)
        cases.append((b3, "明士斜走攻击", None))
        # 暗士限九宫内
        b4 = B.Board.empty()
        b4.put(4, RED, K)
        b4.put(85, BLACK, K)
        b4.put(0 * 9 + 3, BLACK, B.A, hidden=True)
        cases.append((b4, "暗士限九宫", None))
        # 象蹩腿
        b5 = B.Board.empty()
        b5.put(4, RED, K)
        b5.put(85, BLACK, K)
        b5.put(0 * 9 + 6, BLACK, B.E, hidden=False)
        b5.put(2 * 9 + 4, RED, B.P, hidden=False)
        cases.append((b5, "象眼被塞", None))
        # 马蹩腿
        b6 = B.Board.empty()
        b6.put(4, RED, K)
        b6.put(85, BLACK, K)
        b6.put(2 * 9 + 4, BLACK, B.H, hidden=False)
        b6.put(1 * 9 + 4, RED, B.P, hidden=False)
        cases.append((b6, "马腿被蹩", None))

        for board, label, expect in cases:
            with self.subTest(label):
                self._check(board, label)
                if expect is not None:
                    self.assertEqual(board.is_attacked(4, BLACK), expect, label)

    def test_cannon_screen_rule_precisely(self):
        """炮打将：炮架恰好一个才算将军（0 个或 2 个都不算）。"""
        def build(extra_screen):
            b = B.Board.empty()
            b.put(4, RED, K)
            b.put(85, BLACK, K)
            b.put(6 * 9 + 4, BLACK, B.C, hidden=False)
            if extra_screen == 1:
                b.put(3 * 9 + 4, RED, B.P, hidden=False)
            elif extra_screen == 2:
                b.put(3 * 9 + 4, RED, B.P, hidden=False)
                b.put(4 * 9 + 4, RED, B.P, hidden=False)
            return b

        self.assertFalse(build(0).is_attacked(4, BLACK), "0 个炮架不该被将")
        self.assertTrue(build(1).is_attacked(4, BLACK), "1 个炮架应该被将")
        self.assertFalse(build(2).is_attacked(4, BLACK), "2 个炮架不该被将")


class TestLegalFilterMatchesBruteForce(unittest.TestCase):
    """optimized `legal_filter` 必须与"逐个 make/unmake 验证"完全一致。

    `legal_filter` 用了"关键格"优化：只要移动的不是将帅、也不在从将帅出发的
    前两个己方子之列，就直接判为合法。这个优化一旦有漏洞，AI 就会送将。
    """

    def _brute_legal(self, board, side):
        out = []
        for mv in board.gen_pseudo(side):
            board.make(mv)
            ok = not board.is_attacked(board.kings[side], 1 - side)
            board.unmake()
            if ok:
                out.append(mv)
        return out

    def test_matches_on_random_positions(self):
        for seed in range(14):
            b = B.Board.new_game(random.Random(seed))
            rng = random.Random(seed * 17 + 3)
            for ply in range(40):
                for side in (RED, BLACK):
                    if b.turn != side:
                        continue
                    fast = sorted(b.gen_moves(side))
                    slow = sorted(self._brute_legal(b, side))
                    self.assertEqual(fast, slow,
                                     f"seed={seed} ply={ply} side={side} 合法性过滤不一致")
                moves = b.gen_moves(b.turn)
                if not moves:
                    break
                b.make(rng.choice(moves))

    def test_matches_on_constructed_pin_positions(self):
        """构造牵制/暴露局面，重点验证"关键格"优化。"""
        cases = []
        # 车牵制：帅(0,4)、黑车(9,4)、红马(5,4) 是唯一挡子
        b = B.Board.empty()
        b.put(4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(9 * 9 + 4, BLACK, B.R, hidden=False)
        b.put(5 * 9 + 4, RED, B.H, hidden=False)
        cases.append(b)
        # 炮的两层炮架：帅(0,4)、红兵(2,4)、红车(3,4)、黑炮(6,4)
        b2 = B.Board.empty()
        b2.put(4, RED, K)
        b2.put(9 * 9 + 3, BLACK, K)
        b2.put(2 * 9 + 4, RED, B.P, hidden=False)
        b2.put(3 * 9 + 4, RED, B.R, hidden=False)
        b2.put(6 * 9 + 4, BLACK, B.C, hidden=False)
        cases.append(b2)
        # 对面笑：帅(0,4)、黑将(9,4)、红兵(5,4) 唯一挡子
        b3 = B.Board.empty()
        b3.put(4, RED, K)
        b3.put(9 * 9 + 4, BLACK, K)
        b3.put(5 * 9 + 4, RED, B.P, hidden=False)
        cases.append(b3)
        # 将在九宫角落，多个方向都可能暴露
        b4 = B.Board.empty()
        b4.put(2 * 9 + 3, RED, K)
        b4.put(9 * 9 + 5, BLACK, K)
        b4.put(4 * 9 + 3, BLACK, B.R, hidden=False)
        b4.put(4 * 9 + 0, RED, B.R, hidden=False)
        b4.put(8 * 9 + 3, BLACK, B.C, hidden=False)
        b4.put(5 * 9 + 3, RED, B.P, hidden=False)
        cases.append(b4)
        # 蹩马腿：黑马能踩到红帅，但马腿被红仕占着
        b5 = B.Board.empty()
        b5.put(0 * 9 + 4, RED, K)
        b5.put(9 * 9 + 3, BLACK, K)
        b5.put(2 * 9 + 3, BLACK, B.H, hidden=False)   # 马在 (2,3)
        b5.put(1 * 9 + 3, RED, B.A, hidden=False)     # 马腿 (1,3) 是红仕
        b5.put(6 * 9 + 0, RED, B.P, hidden=False)
        cases.append(b5)

        for i, board in enumerate(cases):
            for side in (RED, BLACK):
                fast = sorted(board.gen_moves(side))
                slow = sorted(self._brute_legal(board, side))
                self.assertEqual(fast, slow, f"构造局面 {i} side={side} 不一致")


class TestPawnScreenRegression(unittest.TestCase):
    """回归：走进将帅射线、给敌方炮当炮架的着法必须被过滤掉。

    这是真实对局里"AI 送将"的根因：黑将 (8,4)、红炮 (6,4)、(7,4) 空着时
    并不将军；黑象 (9,2)->(7,4) 之后正好成为唯一炮架，红炮立刻打死黑将。
    """

    def test_walking_into_cannon_line_is_illegal(self):
        b = B.Board.empty()
        b.put(8 * 9 + 4, BLACK, K)                  # 黑将 (8,4)
        b.put(0 * 9 + 4, RED, K)                    # 红帅 (0,4)
        b.put(6 * 9 + 4, RED, B.C, hidden=False)    # 红炮 (6,4)
        b.put(9 * 9 + 2, BLACK, B.E, hidden=True)   # 黑暗象 (9,2)
        b.put(9 * 9 + 6, BLACK, B.E, hidden=True)   # 黑暗象 (9,6)
        b.turn = BLACK
        self.assertFalse(b.is_attacked(8 * 9 + 4, RED), "此刻不该被将")
        legal = b.gen_moves(BLACK)
        self.assertNotIn(B.encode(9 * 9 + 2, 7 * 9 + 4), legal,
                         "(9,2)->(7,4) 会给红炮当炮架，属于送将，必须被过滤")
        self.assertNotIn(B.encode(9 * 9 + 6, 7 * 9 + 4), legal,
                         "(9,6)->(7,4) 同理")
        # 走到别处（不挡在炮与将之间）仍然合法
        self.assertIn(B.encode(9 * 9 + 2, 7 * 9 + 0), legal)

    def test_leaper_leg_blocker_move_is_illegal(self):
        """移开蹩马腿的己方子 -> 敌方马立刻踩到将帅，必须被过滤。

        注意：马腿一旦被自己占着，**这颗子的任何走法都会放开马腿**，
        所以它应该一步都走不了（"困子"），而不是只有某一步不合法。
        """
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(2 * 9 + 3, BLACK, B.H, hidden=False)   # 黑马 (2,3)，踩 (0,4) 的帅
        b.put(1 * 9 + 3, RED, B.A, hidden=False)     # 马腿 (1,3) 被红仕占着
        b.put(1 * 9 + 5, RED, B.A, hidden=False)
        b.put(6 * 9 + 0, RED, B.P, hidden=False)     # 一枚能正常走的子
        b.turn = RED
        self.assertFalse(b.is_attacked(0 * 9 + 4, BLACK), "此刻马腿被蹩，帅安全")
        legal = b.gen_moves(RED)
        from_leg = [m for m in legal if (m >> 7) == 1 * 9 + 3]
        self.assertEqual(from_leg, [],
                         f"蹩马腿的仕不该有合法走法，却有 {[B.move_str(m) for m in from_leg]}")
        # 其它子照常能走，说明不是整个局面没着法
        self.assertTrue([m for m in legal if (m >> 7) == 6 * 9 + 0])


if __name__ == "__main__":
    unittest.main()
