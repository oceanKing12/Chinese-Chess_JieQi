"""走棋步数自由切换 / 点评随步数恢复 / 开新分支。

这些是"回退后能退回那一步的点评、可以自由切换走棋步数"的底层保证：
``GameState.positions`` 保存每一步之后的完整局面，切换只是移动指针。
"""
import os
import random
import sys
import unittest
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(GAME_DIR.parent))
sys.path.insert(0, str(GAME_DIR))

import pygame  # noqa: E402

from jieqi.gui_adapter import (  # noqa: E402
    gui_color_to_side, legal_moves, repetition_counts,
)


def setUpModule():
    pygame.init()
    pygame.display.set_mode((1, 1))


def make_gs(seed=0, plies=8):
    from game_state import GameState

    random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = 'red'
    gs.enable_ai('black')
    rng = random.Random(seed)
    for k in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.set_ply_comment(gs.ply, f'第{k + 1}手点评', f'评分{k}')
        gs.current_player = 'black' if side == 0 else 'red'
    return gs


class TestPlyNavigation(unittest.TestCase):
    def test_positions_align_with_move_history(self):
        gs = make_gs()
        self.assertEqual(len(gs.positions), gs.total_plies + 1)
        self.assertEqual(len(gs.move_history), gs.total_plies)
        self.assertEqual(gs.ply, gs.total_plies)
        self.assertFalse(gs.is_browsing)

    def test_goto_restores_board_and_comment(self):
        gs = make_gs()
        total = gs.total_plies
        for target in (3, 6, 0, total):
            self.assertTrue(gs.goto_ply(target))
            self.assertEqual(gs.ply, target)
            self.assertEqual(gs.move_count, target)
            if target == 0:
                self.assertEqual(gs.ai_comment, '')
            else:
                self.assertEqual(gs.ai_comment, f'第{target}手点评')
                self.assertEqual(gs.ai_eval_text, f'评分{target - 1}')
            # 只要不是最新一手就算"浏览历史"（包括开头）
            self.assertEqual(gs.is_browsing, target < total)

    def test_goto_is_idempotent_and_clamped(self):
        gs = make_gs()
        self.assertFalse(gs.goto_ply(gs.ply), "跳到当前手不该报告发生了移动")
        gs.goto_ply(0)
        self.assertTrue(gs.goto_ply(999))
        self.assertEqual(gs.ply, gs.total_plies)
        self.assertTrue(gs.goto_ply(-5))
        self.assertEqual(gs.ply, 0)
        self.assertFalse(gs.goto_ply(0))

    def test_board_content_differs_between_plies(self):
        """真的切到了不同局面，而不是只改了个数字。"""
        gs = make_gs()
        gs.goto_ply(0)
        first = [[None if not p else (p['color'], p['original'], p['actual'])
                  for p in row] for row in gs.board]
        gs.goto_ply(gs.total_plies)
        last = [[None if not p else (p['color'], p['original'], p['actual'])
                 for p in row] for row in gs.board]
        self.assertNotEqual(first, last)

    def test_undo_redo_round_trip(self):
        gs = make_gs()
        total = gs.total_plies
        self.assertTrue(gs.undo_round())
        self.assertEqual(gs.ply, total - 2)
        self.assertTrue(gs.is_browsing)
        self.assertTrue(gs.redo_move(2))
        self.assertEqual(gs.ply, total)
        self.assertFalse(gs.is_browsing)

    def test_undo_does_not_delete_history(self):
        """回退不丢历史 —— 这正是"可以再前进回去"的前提。"""
        gs = make_gs()
        total = gs.total_plies
        gs.goto_ply(2)
        self.assertEqual(len(gs.move_history), total)
        self.assertEqual(gs.total_plies, total)

    def test_new_move_creates_branch(self):
        gs = make_gs()
        gs.goto_ply(3)
        side = gui_color_to_side(gs.current_player)
        mv = legal_moves(gs.board, side)[0]
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        self.assertEqual(gs.total_plies, 4)
        self.assertEqual(gs.ply, 4)
        self.assertEqual(len(gs.move_history), 4)
        self.assertEqual(len(gs.positions), 5)

    def test_state_history_is_derived_and_consistent(self):
        gs = make_gs()
        gs.goto_ply(4)
        self.assertEqual(len(gs.state_history), 4)
        self.assertIs(gs.state_history[0], gs.positions[0])

    def test_undo_at_start_is_safe(self):
        gs = make_gs(plies=0)
        self.assertEqual(gs.total_plies, 0)
        self.assertFalse(gs.undo_move())
        self.assertFalse(gs.undo_round())
        self.assertEqual(gs.ply, 0)

    def test_repetition_counts_uses_displayed_ply(self):
        """重复局面统计只看"到当前这一手为止"，不能把后面的棋算进去。"""
        gs = make_gs()
        full = repetition_counts(gs)
        self.assertEqual(sum(full.values()), gs.total_plies + 1)
        gs.goto_ply(3)
        partial = repetition_counts(gs)
        self.assertEqual(sum(partial.values()), 4)


class TestCommentPersistence(unittest.TestCase):
    def test_comment_is_stored_by_ply(self):
        gs = make_gs()
        self.assertEqual(gs.ply_comments[3], '第3手点评')
        self.assertEqual(gs.ply_evals[3], '评分2')
        self.assertEqual(len(gs.ply_comments), gs.total_plies + 1)

    def test_set_ply_comment_only_touches_that_ply(self):
        gs = make_gs()
        gs.set_ply_comment(gs.ply, '最新点评', '最新评分')
        self.assertEqual(gs.ply_comments[gs.total_plies], '最新点评')
        self.assertEqual(gs.ply_comments[1], '第1手点评')
        gs.set_ply_comment(999, '越界')
        self.assertNotIn('越界', gs.ply_comments)
        gs.set_ply_comment(-1, '越界')
        self.assertNotIn('越界', gs.ply_comments)

    def test_comment_falls_back_to_previous_ply(self):
        """玩家自己走的那一步没有点评时，显示上一步的点评解释来龙去脉。"""
        gs = make_gs()
        gs.ply_comments[2] = ''
        gs.goto_ply(2)
        self.assertEqual(gs.ai_comment, '第1手点评')

    def test_async_comment_lands_on_its_own_ply(self):
        """异步点评在 AI 落子之后才回来，必须挂到分析时的那一步上。"""
        gs = make_gs()
        target_ply = gs.ply
        # 模拟 AI 落子（ply 前进）
        side = gui_color_to_side(gs.current_player)
        mv = legal_moves(gs.board, side)[0]
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        self.assertEqual(gs.ply, target_ply + 1)
        # 点评回来时挂到旧步数
        gs.set_ply_comment(target_ply, '针对上一步局面的点评')
        self.assertEqual(gs.ply_comments[target_ply], '针对上一步局面的点评')
        self.assertEqual(gs.ply_comments[gs.ply], '')
        # 当前显示：本步没有点评，退回上一步的（正好解释刚走的这步）
        gs.refresh_display_comment()
        self.assertEqual(gs.ai_comment, '针对上一步局面的点评')

    def test_branch_truncates_comments_too(self):
        gs = make_gs()
        gs.goto_ply(3)
        side = gui_color_to_side(gs.current_player)
        mv = legal_moves(gs.board, side)[0]
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.set_ply_comment(gs.ply, '新分支点评')
        self.assertEqual(len(gs.move_history), 4)
        # 截断的**意图**是"旧分支（第 5 手之后）的点评必须不复存在"，
        # 同时这一手自己新留的空位要保留：
        #   * 长度 = 当前手数 + 2（0..ply 各一格，再加这一手的新格）；
        #   * 索引 5（旧分支第 5 手之后）必须已经空了。
        self.assertEqual(len(gs.ply_comments), gs.ply + 2)
        self.assertEqual(gs.ply_comments[4], '新分支点评')
        self.assertEqual(gs.ply_comments[5], '',
                         "旧分支第 5 手之后的点评没被截掉")
        gs.goto_ply(4)
        self.assertEqual(gs.ai_comment, '新分支点评')
        # 回退再前进，点评要还在（回退不该丢掉后面的点评）
        gs.goto_ply(3)
        gs.goto_ply(4)
        self.assertEqual(gs.ai_comment, '新分支点评')


class TestImportedRecordNavigation(unittest.TestCase):
    """导入棋谱之后的导航约定。

    棋谱多半是**已经下完**的。如果一导入就进结算状态，界面会立刻弹出
    "红方胜利 / 点击任意位置重新开始"把整盘棋锁死，连上一步/下一步都点不了。
    所以约定是：导入后停在**开局**，只有玩家自己翻到最后一手才结算。
    """

    def setUp(self):
        from jieqi import record as R

        self.R = R
        self.dir = __import__("tempfile").mkdtemp(prefix="jieqi-nav-")
        self.src = make_gs(seed=5, plies=20)
        self.src.game_over = True
        self.src.winner = 'black'

    def _import(self):
        from game_state import GameState

        record = self.R.build_record(self.src)
        gs = GameState(None)
        gs.show_side_selection = False
        info = gs.import_record(record)
        self.assertTrue(info["ok"], info.get("error"))
        return gs

    def test_import_lands_on_opening_position(self):
        gs = self._import()
        self.assertEqual(gs.ply, 0)
        self.assertFalse(gs.game_over)
        self.assertIsNone(gs.winner)
        # 棋盘必须是真正的开局（而不是重放后残留的终局）
        opening = gs.positions[0]["board"]
        for r in range(10):
            for c in range(9):
                a, b = gs.board[r][c], opening[r][c]
                self.assertEqual(a is None, b is None, f"({r},{c}) 空/非空不一致")
                if a and b:
                    self.assertEqual(a["actual"], b["actual"], f"({r},{c}) 身份不一致")

    def test_result_applies_only_at_last_ply(self):
        gs = self._import()
        total = gs.total_plies
        for ply in range(total + 1):
            gs.goto_ply(ply)
            if ply < total:
                self.assertFalse(gs.game_over, f"第 {ply} 手就结算了")
            else:
                self.assertTrue(gs.game_over)
                self.assertEqual(gs.winner, "black")

    def test_back_to_opening_clears_result(self):
        gs = self._import()
        gs.goto_ply(gs.total_plies)
        self.assertTrue(gs.game_over)
        gs.goto_ply(0)
        self.assertFalse(gs.game_over, "退回开局后仍处于结算状态")
        self.assertIsNone(gs.winner)


if __name__ == "__main__":
    unittest.main()
