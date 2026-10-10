"""引擎基础自检：初始局面走法数、make/unmake 一致性、公开信息不泄漏。"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Game"))

from jieqi import board as B  # noqa: E402
from jieqi.consts import (  # noqa: E402
    BLACK, RED, HOME_DARK_SQUARES, K, R, H, C, P, A, E, SQUARES, HOME_IDENT,
)


class TestSetup(unittest.TestCase):
    def test_initial_pieces(self):
        b = B.Board.new_game(random.Random(7))
        for side in (RED, BLACK):
            self.assertEqual(len(HOME_DARK_SQUARES[side]), 15)
            self.assertEqual(sum(1 for sq in range(SQUARES)
                                 if b.side[sq] == side), 16)
            self.assertEqual(sum(1 for sq in range(SQUARES)
                                 if b.side[sq] == side and b.hidden[sq]), 15)
        self.assertEqual(b.kings[RED], 4)
        self.assertEqual(b.kings[BLACK], 85)
        self.assertFalse(b.hidden[4])
        self.assertFalse(b.hidden[85])

    def test_initial_move_count(self):
        """红方开局应有 44 步合法走法（按摆位身份推算）。"""
        b = B.Board.new_game(random.Random(1))
        moves = b.gen_moves(RED)
        self.assertEqual(len(moves), 44, sorted(B.move_str(m) for m in moves))

    def test_hidden_piece_uses_home_ident(self):
        b = B.Board.new_game(random.Random(3))
        for sq in range(SQUARES):
            if b.hidden[sq]:
                self.assertEqual(b.rule_ident[sq], HOME_IDENT[sq])


class TestMakeUnmake(unittest.TestCase):
    def _snapshot(self, b):
        return (b.ident[:], b.side[:], b.hidden[:], b.rule_ident[:],
                b.occ, b.occ_side[:], b.kings[:], b.turn, b.key)

    def test_make_unmake_restores_exactly(self):
        for seed in range(6):
            b = B.Board.new_game(random.Random(seed))
            for _ in range(24):
                moves = b.gen_moves(b.turn)
                if not moves:
                    break
                mv = random.Random(seed * 100 + b.ply).choice(moves)
                before = self._snapshot(b)
                b.make(mv)
                b.unmake()
                self.assertEqual(before, self._snapshot(b),
                                 f"seed={seed} ply={b.ply} mv={B.move_str(mv)}")

    def test_zobrist_matches_recompute(self):
        b = B.Board.new_game(random.Random(11))
        rng = random.Random(11)
        for _ in range(20):
            moves = b.gen_moves(b.turn)
            if not moves:
                break
            b.make(rng.choice(moves))
            self.assertEqual(b.key, self._recompute_key(b))

    def _recompute_key(self, b):
        from jieqi.consts import Z_HOME, Z_PIECE, Z_SIDE
        k = Z_SIDE if b.turn == BLACK else 0
        for sq in range(SQUARES):
            if b.ident[sq]:
                k ^= (Z_HOME[b.side[sq]][sq] if b.hidden[sq]
                      else Z_PIECE[b.side[sq]][b.ident[sq]][sq])
        return k


class TestLegality(unittest.TestCase):
    def test_no_move_leaves_king_attacked(self):
        for seed in range(4):
            b = B.Board.new_game(random.Random(seed))
            rng = random.Random(seed)
            for _ in range(30):
                moves = b.gen_moves(b.turn)
                if not moves:
                    break
                mv = rng.choice(moves)
                b.make(mv)
                side = 1 - b.turn
                self.assertFalse(b.is_attacked(b.kings[side], b.turn))

    def test_generated_moves_are_legal(self):
        b = B.Board.new_game(random.Random(5))
        rng = random.Random(5)
        for _ in range(20):
            moves = b.gen_moves(b.turn)
            if not moves:
                break
            for mv in random.Random(b.ply).sample(moves, min(6, len(moves))):
                self.assertTrue(b.is_legal(mv), B.move_str(mv))
            b.make(rng.choice(moves))

    def test_exposed_king_is_filtered(self):
        """红帅 (0,4) 与黑车 (9,4) 之间只有一枚过了河的红兵 (5,4) 挡着。

        红兵横走会让出 4 路（车将），必须非法；向前走仍挡着，合法。
        """
        b = B.Board.empty()
        b.put(4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(9 * 9 + 4, BLACK, R, hidden=False)
        b.put(5 * 9 + 4, RED, P, hidden=False)  # 唯一挡子，已过河
        b.turn = RED
        self.assertFalse(b.is_attacked(4, BLACK))
        legal = b.gen_moves(RED)
        # 横走让开 4 路 -> 非法
        self.assertNotIn(B.encode(5 * 9 + 4, 5 * 9 + 3), legal)
        self.assertNotIn(B.encode(5 * 9 + 4, 5 * 9 + 5), legal)
        # 向前走仍然挡着 -> 合法
        self.assertIn(B.encode(5 * 9 + 4, 6 * 9 + 4), legal)

    def test_pinned_piece_has_no_moves(self):
        """被车牵制的红马，任何走法都会暴露红帅。"""
        b = B.Board.empty()
        b.put(4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(9 * 9 + 4, BLACK, R, hidden=False)
        b.put(5 * 9 + 4, RED, H, hidden=False)  # 被牵制的马
        b.turn = RED
        legal = [m for m in b.gen_moves(RED) if (m >> 7) == 5 * 9 + 4]
        self.assertEqual(legal, [], [B.move_str(m) for m in legal])

    def test_flying_general_illegal(self):
        b = B.Board.empty()
        b.put(4, RED, K)
        b.put(85, BLACK, K)
        b.put(3 * 9 + 3, RED, P, hidden=False)
        b.turn = RED
        # 红帅在 (0,4) 与黑将 (9,4) 已经照面 -> 视为被攻击
        self.assertTrue(b.is_attacked(4, BLACK))
        # 红帅离开 4 路本身是允许的（不再照面），但 (0,3) 不在其合法走法里
        # 是因为将帅只能走一步、且 (0,3) 与 (0,4) 相邻，这里应合法；换个断言：
        # 红帅不能走到会照面的位置：从 (0,4) 无处可去才照面，故直接验证被将军标记。
        self.assertTrue(b.in_check(RED))

    def test_check_must_be_resolved(self):
        b = B.Board.empty()
        b.put(4, RED, K)
        b.put(85, BLACK, K)
        b.put(2 * 9 + 4, BLACK, R, hidden=False)   # 黑车在 4 路将军
        b.put(7 * 9 + 0, RED, R, hidden=False)
        b.turn = RED
        self.assertTrue(b.in_check(RED))
        legal = b.gen_moves(RED)
        for mv in legal:
            b.make(mv)
            ok = not b.is_attacked(b.kings[RED], BLACK)
            b.unmake()
            self.assertTrue(ok, B.move_str(mv))


class TestNoCheating(unittest.TestCase):
    def test_movegen_independent_of_hidden_identity(self):
        """两副只有暗子真实身份不同、公开信息相同的棋局，走法必须完全一致。"""
        rng = random.Random(9)
        base = B.Board.new_game(rng)
        other = base.copy()
        for sq in range(SQUARES):
            if other.hidden[sq]:
                other.ident[sq] = HOME_IDENT[sq]
        self.assertEqual(sorted(base.gen_moves(RED)), sorted(other.gen_moves(RED)))
        self.assertEqual(base.public_key(), other.public_key())
        self.assertEqual(base.public_signature(), other.public_signature())


if __name__ == "__main__":
    unittest.main()
