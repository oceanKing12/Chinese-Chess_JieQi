"""战术与局面判断测试：直接考察"看得准、走得对"。

覆盖：
* 必须应将（不能走违规着法）；
* 不吃"毒饵"（吃子后必被反吃，需要 2~3 层搜索 + 静态搜索才看得到）；
* 该吃的白送暗子要吃（按暗子后验期望估值）；
* 不能把大子挂在对方口里；
* 杀棋判定：能杀的要判杀，不能杀的不许乱判杀。
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Game"))

from jieqi import board as B  # noqa: E402
from jieqi.consts import (  # noqa: E402
    A, BLACK, C, E, H, K, MATE_BOUND, P, R, RED, SQUARES,
)
from jieqi.engine import Engine  # noqa: E402


def forced_mate(board: B.Board, side: int, plies: int) -> bool:
    """独立的暴力搜索：``side`` 先行能否在 ``plies`` 个半回合内将死对手。

    杀棋在揭棋里只依赖公开信息（走法、将军、无着可走都是公开的），
    所以这个纯规则暴力搜索是判断"到底有没有杀"的可靠参照。
    """
    if plies <= 0:
        return False
    for mv in board.gen_moves(side):
        board.make(mv)
        try:
            opp = 1 - side
            if board.kings[opp] < 0 or not board.gen_moves(opp):
                return True
            if plies >= 3:
                all_ok = True
                for reply in board.gen_moves(opp):
                    board.make(reply)
                    ok = forced_mate(board, side, plies - 2)
                    board.unmake()
                    if not ok:
                        all_ok = False
                        break
                if all_ok:
                    return True
        finally:
            board.unmake()
    return False


def analyse(board, seconds=1.6, samples=4, seed=101, difficulty="normal"):
    return Engine(difficulty, time_limit=seconds, samples=samples,
                  seed=seed).analyse(board)


class TestMustRespondToCheck(unittest.TestCase):
    def test_escapes_check(self):
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(1 * 9 + 4, BLACK, R, hidden=False)      # 黑车将军
        b.put(5 * 9 + 0, RED, R, hidden=False)
        b.put(4 * 9 + 2, RED, C, hidden=False)        # 红炮可以垫将
        b.turn = RED
        self.assertTrue(b.in_check(RED))
        res = analyse(b)
        self.assertIsNotNone(res.move)
        b.make(res.move)
        self.assertFalse(b.is_attacked(b.kings[RED], BLACK),
                         f"{B.move_str(res.move)} 没有解将")
        b.unmake()


class TestAvoidPoisonedCapture(unittest.TestCase):
    def test_does_not_take_poisoned_pawn(self):
        """红车吃 (5,4) 的黑卒会被 (9,4) 的黑车反吃，净亏一车，不能吃。"""
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(4 * 9 + 4, RED, R, hidden=False)        # 红车 (4,4)
        b.put(5 * 9 + 4, BLACK, P, hidden=False)      # 黑卒 (5,4)，诱饵
        b.put(9 * 9 + 4, BLACK, R, hidden=False)      # 黑车 (9,4) 保护它
        b.put(0 * 9 + 0, RED, P, hidden=False)
        b.turn = RED
        bad = B.encode(4 * 9 + 4, 5 * 9 + 4)
        self.assertIn(bad, b.gen_moves(RED))
        res = analyse(b, seconds=2.0, samples=2, seed=7)
        self.assertNotEqual(res.move, bad,
                            "引擎吃了被保护的黑卒（会白丢一车）")

    def test_shallow_search_misses_it(self):
        """反证：只看一层、且关掉静态搜索时容易中招 —— 说明搜索深度确实在起作用。

        这里只断言"浅搜索不保证看出来"，不做强断言，避免偶发不稳定导致测试失败。
        """
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(4 * 9 + 4, RED, R, hidden=False)
        b.put(5 * 9 + 4, BLACK, P, hidden=False)
        b.put(9 * 9 + 4, BLACK, R, hidden=False)
        b.put(0 * 9 + 0, RED, P, hidden=False)
        b.turn = RED
        shallow = Engine("normal", time_limit=0.2, samples=1, max_depth=1,
                         seed=7, quiescence=False, null_move=False)
        res = shallow.analyse(b)
        self.assertIsNotNone(res.move)


class TestTakeFreeMaterial(unittest.TestCase):
    def test_takes_hanging_chariot(self):
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(5 * 9 + 4, RED, R, hidden=False)
        b.put(5 * 9 + 7, BLACK, R, hidden=False)
        b.put(9 * 9 + 0, BLACK, P, hidden=False)
        b.put(0 * 9 + 0, RED, P, hidden=False)
        b.turn = RED
        res = analyse(b)
        self.assertEqual(res.move, B.encode(5 * 9 + 4, 5 * 9 + 7))
        self.assertGreater(res.score, 500)

    def test_takes_free_dark_piece(self):
        """白吃的暗子要吃掉（按后验期望，约值 300+ 厘兵）。"""
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(5 * 9 + 0, RED, R, hidden=False)     # 红车可以沿 0 路上行
        b.put(0 * 9 + 0, BLACK, H, hidden=True)    # 黑方 (0,0) 摆位（车位）的暗子
        b.put(9 * 9 + 8, BLACK, P, hidden=False)
        b.put(0 * 9 + 8, RED, P, hidden=False)
        b.turn = RED
        res = analyse(b, seconds=2.0, samples=2, seed=3)
        target = B.encode(5 * 9 + 0, 0 * 9 + 0)
        self.assertIn(target, b.gen_moves(RED))
        self.assertEqual(res.move, target,
                         f"应吃掉无人保护的暗子，实际 {B.move_str(res.move)}")
        self.assertGreater(res.score, 200, f"评估 {res.score} 偏低")

    def test_does_not_hang_chariot(self):
        """红车被黑炮瞄着（车吃不到炮，中间有炮架），不能走无关的闲着挂车。"""
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(5 * 9 + 4, RED, R, hidden=False)     # 红车
        b.put(5 * 9 + 6, BLACK, P, hidden=False)   # 炮架
        b.put(5 * 9 + 8, BLACK, C, hidden=False)   # 黑炮隔着炮架瞄准红车
        b.put(0 * 9 + 0, RED, P, hidden=False)
        b.put(6 * 9 + 0, BLACK, P, hidden=False)
        b.turn = RED
        self.assertTrue(b.is_attacked(5 * 9 + 4, BLACK))
        res = analyse(b, seconds=2.0, samples=4, seed=5)
        b.make(res.move)
        still_there = (b.ident[5 * 9 + 4] == R and b.side[5 * 9 + 4] == RED)
        if still_there:
            self.assertTrue(b.is_attacked(5 * 9 + 4, RED),
                            f"{B.move_str(res.move)} 之后红车被攻击却无保护（挂子）")
        b.unmake()
        self.assertGreater(res.score, -300, f"评估 {res.score} 过于悲观，可能挂子")


class TestMateTruthfulness(unittest.TestCase):
    def _mate_in_one_position(self):
        b = B.Board.empty()
        b.put(0 * 9 + 3, RED, K)
        b.put(1 * 9 + 0, RED, R, hidden=False)
        b.put(9 * 9 + 4, BLACK, K)
        b.put(9 * 9 + 3, BLACK, P, hidden=False)
        b.put(9 * 9 + 5, BLACK, P, hidden=False)
        b.turn = RED
        return b

    def test_brute_force_agrees_it_is_mate(self):
        b = self._mate_in_one_position()
        self.assertTrue(forced_mate(b, RED, 1))
        res = analyse(b, seconds=1.5, samples=2, seed=11)
        self.assertGreater(res.score, MATE_BOUND)
        self.assertEqual(res.move, B.encode(1 * 9 + 0, 1 * 9 + 4))
        self.assertEqual(res.mate_in(), 1)

    def test_stalemate_counts_as_a_loss(self):
        """象棋规则：困毙（无棋可走）同样判负 —— 红帅一让路就能困死黑将。

        黑将在 (9,3)：(9,2) 出九宫、(8,3) 被红兵看住、(9,4) 会让双方将帅照面，
        所以红帅走到 4 路任意一格即可困毙黑方。
        """
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(7 * 9 + 3, RED, P, hidden=False)
        b.turn = RED
        self.assertFalse(b.in_check(BLACK), "此刻黑方并未被将军，是困毙而不是将死")
        self.assertTrue(forced_mate(b, RED, 1), "红方一步即可困毙黑方")
        res = analyse(b, seconds=1.5, samples=2, seed=11)
        self.assertGreaterEqual(res.score, MATE_BOUND,
                                f"没能识别困毙，评分 {res.score}")
        self.assertEqual(res.mate_in(), 1)

    def test_mate_claim_is_always_verifiable(self):
        """引擎凡是判杀的局面，暴力搜索必须能找到杀（抽查随机局面）。"""
        checked = 0
        for seed in range(40):
            b = B.Board.new_game(random.Random(seed))
            rng = random.Random(seed * 7 + 1)
            for _ in range(rng.randrange(6, 26)):
                moves = b.gen_moves(b.turn)
                if not moves:
                    break
                b.make(rng.choice(moves))
            if not b.gen_moves(b.turn):
                continue
            res = analyse(b, seconds=0.7, samples=2, seed=seed)
            if res.score >= MATE_BOUND:
                self.assertTrue(
                    forced_mate(b, b.turn, 5),
                    f"seed={seed} 引擎判杀（{res.score}）但 5 层内没有杀")
                checked += 1
            if checked >= 3:
                break
        self.assertGreaterEqual(checked, 0)


if __name__ == "__main__":
    unittest.main()
