"""规则一致性：GUI 自己的走法验证器必须与引擎走法生成完全一致。

这是最容易出隐蔽 bug 的地方——如果 AI 与玩家对规则的理解不同：
* AI 会走出玩家认为非法的棋；
* AI 会把玩家其实能解的将误判成杀棋。
所以这里对随机局面做全盘 90×90 逐格比对。
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Game"))

from config import Config  # noqa: E402
from jieqi import board as B  # noqa: E402
from jieqi.consts import RED, BLACK  # noqa: E402
from jieqi.gui_adapter import engine_board_to_gui, gui_board_to_engine  # noqa: E402


def gui_pseudo_legal(cfg, gui_board, sr, sc, er, ec):
    """完全照抄 GameState.validate_move 的逻辑（不含自将检查）。"""
    piece = gui_board[sr][sc]
    target = gui_board[er][ec]
    if not piece:
        return False
    if target and target['color'] == piece['color']:
        return False
    if piece['actual'] in ['将', '帅']:
        return cfg._validate_king(sr, sc, er, ec, gui_board)
    rule_type = piece['original'] if not piece['revealed'] else piece['actual']
    validator_name = cfg.MOVE_VALIDATORS.get(rule_type, '_invalid_move')
    validator = getattr(cfg, validator_name, cfg._invalid_move)
    return validator(sr, sc, er, ec, gui_board)


def engine_pseudo_set(board, side):
    return set(board.gen_pseudo(side))


class TestRuleConsistency(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def _compare(self, board, side, label=""):
        gui = engine_board_to_gui(board)
        engine_moves = engine_pseudo_set(board, side)
        for sr in range(10):
            for sc in range(9):
                piece = gui[sr][sc]
                if not piece or piece['color'] != ('red' if side == RED else 'black'):
                    continue
                for er in range(10):
                    for ec in range(9):
                        mv = B.encode(sr * 9 + sc, er * 9 + ec)
                        gui_ok = gui_pseudo_legal(self.cfg, gui, sr, sc, er, ec)
                        eng_ok = mv in engine_moves
                        if gui_ok != eng_ok:
                            self.fail(
                                f"{label} 规则不一致：{B.move_str(mv)} "
                                f"棋子={piece['original']}/{'明' if piece['revealed'] else '暗'} "
                                f"GUI={gui_ok} 引擎={eng_ok}")

    def test_opening_position(self):
        b = B.Board.new_game(random.Random(2))
        self._compare(b, RED, "开局红")
        self._compare(b, BLACK, "开局黑")

    def test_random_middlegame_positions(self):
        for seed in (11, 12, 13):
            b = B.Board.new_game(random.Random(seed))
            rng = random.Random(seed)
            for _ in range(28):
                moves = b.gen_moves(b.turn)
                if not moves:
                    break
                b.make(rng.choice(moves))
            self._compare(b, b.turn, f"seed={seed}")

    def test_revealed_advisor_can_cross_river(self):
        """明士可以过河（"士相可以过河"），引擎与 GUI 都必须允许。"""
        b = B.Board.empty()
        b.put(4, RED, B.K)
        b.put(85, BLACK, B.K)
        b.put(6 * 9 + 4, RED, B.A, hidden=False)   # 红明士已过河到 (6,4)
        b.turn = RED
        moves = b.gen_pseudo(RED)
        self.assertIn(B.encode(6 * 9 + 4, 5 * 9 + 3), moves)
        self.assertIn(B.encode(6 * 9 + 4, 7 * 9 + 5), moves)
        gui = engine_board_to_gui(b)
        self.assertTrue(gui_pseudo_legal(self.cfg, gui, 6, 4, 5, 3))
        self.assertTrue(gui_pseudo_legal(self.cfg, gui, 6, 4, 7, 5))

    def test_hidden_advisor_stays_in_palace(self):
        b = B.Board.empty()
        b.put(4, RED, B.K)
        b.put(85, BLACK, B.K)
        b.put(0 * 9 + 3, RED, B.A, hidden=True)    # 暗仕在 (0,3)
        b.turn = RED
        moves = b.gen_pseudo(RED)
        self.assertIn(B.encode(0 * 9 + 3, 1 * 9 + 4), moves)      # 进中心，九宫内
        self.assertNotIn(B.encode(0 * 9 + 3, 1 * 9 + 2), moves)   # 出九宫，禁止
        gui = engine_board_to_gui(b)
        self.assertFalse(gui_pseudo_legal(self.cfg, gui, 0, 3, 1, 2))

    def test_hidden_elephant_uses_eye_rule(self):
        """暗象按田字走且塞象眼；暗子只可能在初始格子上。"""
        b = B.Board.empty()
        b.put(4, RED, B.K)
        b.put(85, BLACK, B.K)
        b.put(0 * 9 + 2, RED, B.E, hidden=True)     # 红暗象在初始象位
        b.turn = RED
        moves = b.gen_pseudo(RED)
        self.assertIn(B.encode(0 * 9 + 2, 2 * 9 + 0), moves)
        self.assertIn(B.encode(0 * 9 + 2, 2 * 9 + 4), moves)
        # 塞象眼
        b.put(1 * 9 + 1, RED, B.P, hidden=False)
        moves = b.gen_pseudo(RED)
        self.assertNotIn(B.encode(0 * 9 + 2, 2 * 9 + 0), moves)
        gui = engine_board_to_gui(b)
        self.assertFalse(gui_pseudo_legal(self.cfg, gui, 0, 2, 2, 0))

    def test_revealed_elephant_can_cross_river(self):
        """明象可过河攻杀（"象、士等原限的棋子在揭开后可过河"）。"""
        b = B.Board.empty()
        b.put(4, RED, B.K)
        b.put(85, BLACK, B.K)
        b.put(4 * 9 + 2, RED, B.E, hidden=False)
        b.turn = RED
        moves = b.gen_pseudo(RED)
        self.assertIn(B.encode(4 * 9 + 2, 6 * 9 + 0), moves)
        self.assertIn(B.encode(4 * 9 + 2, 6 * 9 + 4), moves)
        gui = engine_board_to_gui(b)
        self.assertTrue(gui_pseudo_legal(self.cfg, gui, 4, 2, 6, 0))


if __name__ == "__main__":
    unittest.main()
