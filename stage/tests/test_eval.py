"""评估函数自检：对称局面必须给出接近 0 的分；不依赖暗子真值。"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Game"))

from jieqi import board as B  # noqa: E402
from jieqi.belief import Belief  # noqa: E402
from jieqi.consts import (  # noqa: E402
    A, C, E, EVAL_WEIGHTS, H, K, P, R, RED, BLACK, SQUARES,
)
from jieqi.evaluate import EvalContext, evaluate  # noqa: E402


class TestEval(unittest.TestCase):
    def test_symmetric_position_is_near_zero(self):
        """完全对称的局面（镜像）红黑分差不应很大。"""
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 4, BLACK, K)
        b.put(2 * 9 + 1, RED, C, hidden=False)
        b.put(7 * 9 + 7, BLACK, C, hidden=False)
        b.put(0 * 9 + 0, RED, R, hidden=False)
        b.put(9 * 9 + 8, BLACK, R, hidden=False)
        b.put(3 * 9 + 4, RED, P, hidden=False)
        b.put(6 * 9 + 4, BLACK, P, hidden=False)
        b.turn = RED
        belief = Belief.from_board(b)
        ctx = EvalContext(belief)
        score = evaluate(b, ctx)
        # 允许有先手分（tempo），但不应该出现几百厘兵的偏差
        self.assertLess(abs(score), 80, f"对称局面评分 {score} 异常")

    def test_material_advantage_sign(self):
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 4, BLACK, K)
        b.put(4 * 9 + 4, RED, R, hidden=False)   # 红多一车
        b.turn = RED
        ctx = EvalContext(Belief.from_board(b))
        self.assertGreater(evaluate(b, ctx), 300)

    def test_hidden_pieces_valued_by_expectation(self):
        """同样公开信息的局面，暗子真值不同 -> 评估必须完全一致。"""
        b1 = B.Board.new_game(random.Random(3))
        b2 = b1.copy()
        rng = random.Random(77)
        for sq in range(SQUARES):
            if b2.hidden[sq]:
                b2.ident[sq] = rng.choice([A, E, H, R, C, P])
        ctx = EvalContext(Belief.from_board(b1))
        self.assertEqual(evaluate(b1, ctx), evaluate(b2, ctx))

    def test_opening_score_small(self):
        b = B.Board.new_game(random.Random(5))
        ctx = EvalContext(Belief.from_board(b))
        score = evaluate(b, ctx)
        self.assertLess(abs(score), 70, f"开局评分 {score} 异常（应只有先手分）")


if __name__ == "__main__":
    unittest.main()
