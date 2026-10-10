"""引擎测试：合法性、时间预算、不偷看暗子、能找出一步杀。"""
import random
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Game"))

from jieqi import board as B  # noqa: E402
from jieqi.belief import Belief  # noqa: E402
from jieqi.consts import (  # noqa: E402
    BLACK, RED, K, R, P, MATE_BOUND, HOME_IDENT, SQUARES, A, C, H, E,
)
from jieqi.engine import Engine, verdict_of  # noqa: E402


def random_plies(board, n, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        moves = board.gen_moves(board.turn)
        if not moves:
            break
        board.make(rng.choice(moves))
    return board


class TestHiddenPieceValuation(unittest.TestCase):
    """暗子必须按**摆位身份**估值（它能干什么），不能按"暗子池的期望"估值。

    这是真实反馈里的两个问题（"引擎开局就翻暗车""随手弃暗马"）的根因：
    把一枚暗车算成"半辆车"（期望只有 529 而不是 900），翻开就看着不亏、
    丢掉也不心疼。锁死"暗车 == 车价"这条不变量。
    """

    def _ctx(self, board):
        from jieqi.evaluate import EvalContext
        return EvalContext(Belief.from_board(board))

    def test_hidden_chariot_valued_as_chariot(self):
        from jieqi.evaluate import _piece_value

        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 4, BLACK, K)
        # (0,0) 是红车的摆位格：暗车放在这里，价值必须等于车
        b.put(0 * 9 + 0, RED, R, hidden=True)
        ctx = self._ctx(b)
        value = _piece_value(b, ctx, 0 * 9 + 0, RED)
        from jieqi.consts import PIECE_VALUE, PST_FLAT_SIDE
        expect = PIECE_VALUE[R] + PST_FLAT_SIDE[RED][R][0]
        self.assertAlmostEqual(value, expect, places=6,
                               msg="暗车没有按车价估值")
        # 明显高于"暗子池期望"（车/马/炮/兵混在一起的均值）
        self.assertGreater(value, ctx.mean_value[RED] + 200)

    def test_hidden_piece_keeps_home_identity_after_moving(self):
        """暗子的摆位身份跟着它走：同一枚暗车，估价不随所在格子变。

        正常对局里暗子一移动就会翻开，所以这条只在"人工摆出来的暗子"上成立；
        它锁的是 ``rule_ident``（行棋身份）与摆位身份的一致性——
        引擎的 ``_piece_value`` 就是靠这个把暗车当车估的。
        """
        from jieqi.evaluate import _piece_value
        from jieqi.consts import PIECE_VALUE

        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 4, BLACK, K)
        # (2,0) 的本格摆位身份是"相"；这里故意放一枚"摆位身份是车"的暗子
        b.put(2 * 9 + 0, RED, R, hidden=True, home_ident=R)
        self.assertEqual(b.rule_ident[2 * 9 + 0], R,
                         "home_ident 没有被记进 rule_ident")
        ctx = self._ctx(b)
        value = _piece_value(b, ctx, 2 * 9 + 0, RED)
        from jieqi.consts import PST_FLAT_SIDE
        self.assertAlmostEqual(
            value, PIECE_VALUE[R] + PST_FLAT_SIDE[RED][R][2 * 9 + 0], places=6)
        self.assertGreater(value, PIECE_VALUE[C] + 300,
                           "暗车被当成小兵估价了")

    def test_revealing_a_hidden_chariot_can_cost_material(self):
        """翻暗车时如果翻出来不是车，评估要**真的下降**（不能无所谓）。"""
        from jieqi.evaluate import evaluate

        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 4, BLACK, K)
        b.put(0 * 9 + 0, RED, R, hidden=True)
        ctx = self._ctx(b)
        hidden_score = evaluate(b, ctx)
        b.make(B.encode(0 * 9 + 0, 2 * 9 + 0))       # 翻开成 P（走 2 格的是车？不）
        # 直接构造"翻出来是个兵"的同一局面，比较分数
        b2 = B.Board.empty()
        b2.put(0 * 9 + 4, RED, K)
        b2.put(9 * 9 + 4, BLACK, K)
        b2.put(2 * 9 + 0, RED, P, hidden=False)
        revealed_score = evaluate(b2, self._ctx(b2))
        self.assertGreater(hidden_score, revealed_score + 400,
                           "暗车翻开变兵，评估居然没怎么掉（会鼓励乱翻）")


class TestMateGuard(unittest.TestCase):
    """"送杀体检"：不能把好棋当送杀否掉，也不能放过明显的送杀。"""

    def test_does_not_reject_a_winning_mate(self):
        """红方一步杀必须照走（体检早期版本把这一步当成"送杀"改掉了）。"""
        b = B.Board.empty()
        b.put(0 * 9 + 3, RED, K)
        b.put(1 * 9 + 0, RED, R, hidden=False)
        b.put(9 * 9 + 4, BLACK, K)
        b.put(9 * 9 + 3, BLACK, P, hidden=False)
        b.put(9 * 9 + 5, BLACK, P, hidden=False)
        b.turn = RED
        mate_move = B.encode(1 * 9 + 0, 1 * 9 + 4)
        for guard in (True, False):
            res = Engine("normal", time_limit=1.2, samples=2, seed=11,
                         mate_guard=guard).analyse(b)
            self.assertEqual(res.move, mate_move,
                             f"mate_guard={guard} 时把一步杀改掉了")
            self.assertGreater(res.score, MATE_BOUND * 0.5)

    def test_board_is_intact_after_the_guard(self):
        """体检超时后必须把棋盘回滚干净（曾经留下"走到一半"的格子，
        紧接着 gen_pseudo 就 KeyError: 0）。"""
        from jieqi.belief import Belief
        from jieqi.evaluate import EvalContext
        from jieqi.search import Searcher, TimeUp

        b = random_plies(B.Board.new_game(random.Random(8)), 26, seed=2)
        pub = b.clone_revealed_only()
        searcher = Searcher(pub.copy(), EvalContext(Belief.from_board(b)))
        root = searcher.board.gen_moves(searcher.board.turn)
        before = searcher.board.key
        searcher.guard_deadline(0.001)          # 故意极小，逼它超时
        try:
            searcher.forced_loss_moves(root, plies=4)
        except TimeUp:
            pass
        board = searcher.board
        self.assertEqual(board.key, before, "体检之后棋盘哈希变了")
        self.assertEqual(len(board._undo), 0, "体检之后撤销栈没清空")
        corrupt = [s for s in range(SQUARES)
                   if board.side[s] >= 0 and board.rule_ident[s] == 0]
        self.assertEqual(corrupt, [], f"体检留下非法格子：{corrupt}")
        # 还能正常生成着法（KeyError 的现场）
        self.assertTrue(board.gen_moves(board.turn))


class TestEngineBasics(unittest.TestCase):
    def test_returns_legal_move_opening(self):
        b = B.Board.new_game(random.Random(4))
        eng = Engine("easy", time_limit=0.4, seed=1)
        res = eng.analyse(b)
        self.assertIsNotNone(res.move)
        self.assertIn(res.move, b.gen_moves(RED))
        self.assertLess(res.elapsed, 3.0)
        self.assertTrue(res.summary())

    def test_returns_legal_move_middlegame(self):
        b = random_plies(B.Board.new_game(random.Random(8)), 30, seed=2)
        eng = Engine("normal", time_limit=1.5, samples=2, seed=3)
        res = eng.analyse(b)
        self.assertIn(res.move, b.gen_moves(b.turn))
        self.assertGreater(res.nodes, 0)

    def test_full_game_all_moves_legal(self):
        b = B.Board.new_game(random.Random(21))
        eng = Engine("easy", time_limit=0.25, samples=1, seed=5)
        for _ in range(14):
            moves = b.gen_moves(b.turn)
            if not moves:
                break
            res = eng.analyse(b)
            self.assertIn(res.move, moves)
            b.make(res.move)

    def test_time_limit_respected(self):
        b = random_plies(B.Board.new_game(random.Random(12)), 16, seed=6)
        eng = Engine("normal", time_limit=1.0, samples=3, seed=7)
        t0 = time.monotonic()
        eng.analyse(b)
        self.assertLess(time.monotonic() - t0, 2.5)

    def test_no_legal_move_reports_none(self):
        """红帅被黑车贴脸将军且无处可逃时，引擎应报告无着可走。"""
        b = B.Board.empty()
        b.put(4, RED, K)
        b.put(85, BLACK, K)
        b.put(1 * 9 + 4, BLACK, R, hidden=False)   # 黑车在 (1,4) 将军
        b.put(1 * 9 + 3, BLACK, R, hidden=False)   # 再看住 (0,3)
        b.put(1 * 9 + 5, BLACK, R, hidden=False)   # 看住 (0,5)
        b.turn = RED
        self.assertEqual(b.gen_moves(RED), [])
        res = Engine("easy", time_limit=0.3, seed=1).analyse(b)
        self.assertIsNone(res.move)


class TestNoCheating(unittest.TestCase):
    def test_hidden_identity_does_not_change_result(self):
        """只有暗子真实身份不同（公开信息完全相同）的两局，AI 决策必须一致。"""
        b1 = B.Board.new_game(random.Random(31))
        b2 = b1.copy()
        rng = random.Random(99)
        for sq in range(SQUARES):
            if b2.hidden[sq]:
                # 打乱真实身份，但保持“同色同格暗子”这一公开信息不变
                b2.ident[sq] = rng.choice([A, E, H, R, C, P])
        self.assertEqual(b1.public_signature(), b2.public_signature())

        #: 必须用**固定深度**而不是固定时间：迭代加深是按墙上时钟决定搜到第几层的，
        #: 用 time_limit 的话两次运行的深度可能不同，这个测试就会随机失败
        #: （那是计时抖动，不是真的发现了信息泄漏）。
        eng1 = Engine("normal", time_limit=30.0, samples=2, max_depth=3, seed=1234)
        eng2 = Engine("normal", time_limit=30.0, samples=2, max_depth=3, seed=1234)
        r1 = eng1.analyse(b1)
        r2 = eng2.analyse(b2)
        self.assertEqual(r1.move, r2.move)
        self.assertAlmostEqual(r1.score, r2.score, places=6)
        self.assertEqual([s.mean for s in r1.stats], [s.mean for s in r2.stats])
        self.assertEqual([s.samples for s in r1.stats], [s.samples for s in r2.stats])

    def test_pool_matches_hidden_count(self):
        b = random_plies(B.Board.new_game(random.Random(17)), 20, seed=4)
        belief = Belief.from_board(b)
        for side in (RED, BLACK):
            self.assertEqual(len(belief.pools[side]), len(belief.dark[side]))

    def test_captured_are_removed_from_pool(self):
        b = B.Board.new_game(random.Random(23))
        # 模拟：红方一枚暗子（车）被吃掉，该格清空
        victim = next(sq for sq in range(SQUARES)
                      if b.hidden[sq] and b.side[sq] == RED)
        b.ident[victim] = R          # 被吃的其实是一枚车
        b.remove(victim)
        belief = Belief.from_board(b, {RED: [R]})
        # 两枚车里被吃一枚，池中应只剩一枚
        self.assertEqual(belief.pools[RED].count(R), 1)
        self.assertEqual(len(belief.pools[RED]), len(belief.dark[RED]))
        self.assertEqual(len(belief.pools[RED]), 14)


class TestTactics(unittest.TestCase):
    def test_finds_mate_in_one(self):
        """红车 (1,0)->(1,4) 造成将死（黑将 (9,4)，两卒堵住 (9,3)/(9,5)）。"""
        b = B.Board.empty()
        b.put(0 * 9 + 3, RED, K)
        b.put(1 * 9 + 0, RED, R, hidden=False)
        b.put(9 * 9 + 4, BLACK, K)
        b.put(9 * 9 + 3, BLACK, P, hidden=False)
        b.put(9 * 9 + 5, BLACK, P, hidden=False)
        b.turn = RED
        mate_move = B.encode(1 * 9 + 0, 1 * 9 + 4)
        self.assertIn(mate_move, b.gen_moves(RED))
        # 确认这就是杀
        b.make(mate_move)
        self.assertEqual(b.gen_moves(BLACK), [])
        b.unmake()

        res = Engine("normal", time_limit=1.5, samples=2, seed=11).analyse(b)
        self.assertIsNotNone(res.move)
        self.assertGreater(res.score, MATE_BOUND * 0.5,
                           f"局面分 {res.score} 应判定为杀棋")
        self.assertAlmostEqual(res.score, 1_000_000 - 1, delta=4)
        self.assertEqual(res.move, mate_move)

    def test_takes_free_chariot(self):
        """黑车白送：红车可以无代价吃掉它，引擎应吃。"""
        b = B.Board.empty()
        b.put(0 * 9 + 4, RED, K)
        b.put(9 * 9 + 3, BLACK, K)
        b.put(5 * 9 + 4, RED, R, hidden=False)
        b.put(5 * 9 + 7, BLACK, R, hidden=False)
        b.put(9 * 9 + 0, BLACK, P, hidden=False)
        b.turn = RED
        res = Engine("normal", time_limit=1.2, samples=2, seed=13).analyse(b)
        self.assertEqual(res.move, B.encode(5 * 9 + 4, 5 * 9 + 7),
                         f"应吃黑车，实际 {B.move_str(res.move)}")


class TestVerdict(unittest.TestCase):
    def test_verdict_text(self):
        self.assertEqual(verdict_of(0), "均势")
        self.assertEqual(verdict_of(500), "红方大优")
        self.assertEqual(verdict_of(-500), "黑方大优")


if __name__ == "__main__":
    unittest.main()
