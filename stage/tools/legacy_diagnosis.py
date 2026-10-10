"""诊断旧版 AI（D:\\py\\Chinese Chess\\Game\\AI.py）的问题，供迁移说明使用。

不改动旧文件，只把它作为模块加载进来观察行为。

已知问题（本脚本用于验证）：
1. ``loss()`` 把"移动后自己被将军"的巨额惩罚以**正数**返回，而
   ``calculate_piece_move`` 是 ``scores + moved_score``、排序又是降序，
   于是"送将"的着法反而排到最前面被选中 —— 旧 AI 会主动走违规着法。
2. ``_get_ordered_moves`` 只调用 ``validate_move``（走法形状），
   完全没有过滤"自将 / 送将 / 对面笑"。
3. ``main.py`` 应用 AI 着法时不校验，所以这些违规着法会真的落到棋盘上。
"""
import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
for p in (str(STAGE), str(STAGE / "Game")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from jieqi import board as B  # noqa: E402
from jieqi.gui_adapter import engine_board_to_gui, gui_board_to_engine  # noqa: E402

LEGACY_PATH = Path(r"D:\py\Chinese Chess\Game\AI.py")


class FakeGS:
    """给旧 AI 用的最小 GameState 替身。"""

    def __init__(self, gui_board, current_player="red", ai_color="red"):
        from config import Config

        self.board = gui_board
        self.current_player = current_player
        self.ai_color = ai_color
        self.config = Config()
        self.in_check = False
        self.selected = None
        self.move_count = 0
        self.move_history = []
        self.position_counter = {}
        self.toast = ""
        self.game_over = False

    # 旧 AI 只会用到这两个
    def validate_move(self, sr, sc, er, ec, board=None):
        board = board or self.board
        piece = board[sr][sc]
        target = board[er][ec]
        if target and target['color'] == piece['color']:
            return False
        if piece['actual'] in ['将', '帅']:
            return self.config._validate_king(sr, sc, er, ec, board)
        rule_type = piece['original'] if not piece['revealed'] else piece['actual']
        name = self.config.MOVE_VALIDATORS.get(rule_type, '_invalid_move')
        return getattr(self.config, name)(sr, sc, er, ec, board)

    def _get_king_positions(self, board):
        return (next(((r, c) for r in range(10) for c in range(9)
                      if (p := board[r][c]) and p['actual'] == '帅'), None),
                next(((r, c) for r in range(10) for c in range(9)
                      if (p := board[r][c]) and p['actual'] == '将'), None))

    def _check_check(self, board, color, red_king, black_king):
        target = red_king if color == 'red' else black_king
        opp = 'black' if color == 'red' else 'red'
        for r in range(10):
            for c in range(9):
                p = board[r][c]
                if p and p['color'] == opp and self.validate_move(r, c, target[0], target[1], board):
                    return True
        return False

    def _check_face_to_face(self, board, red_king, black_king):
        if red_king[1] != black_king[1]:
            return False
        return not any(board[r][red_king[1]] for r in
                       range(min(red_king[0], black_king[0]) + 1, max(red_king[0], black_king[0])))

    def ai_check_fold_repetition(self):
        return None

    def check_game_state(self):
        """旧 AI 在 _simulate_move 里会调用它；这里不做任何事。"""

    def is_checkmate(self, color):
        return False

    def _has_legal_moves(self):
        return True


def load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_ai_module", LEGACY_PATH)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod.AdvancedAIController


def build_position():
    """红方被黑车将（必须应将），同时红方另有一枚车可以随便走（送将）。"""
    b = B.Board.empty()
    b.put(0 * 9 + 4, B.RED, B.K)                       # 红帅 (0,4)
    b.put(9 * 9 + 3, B.BLACK, B.K)                     # 黑将 (9,3)
    b.put(1 * 9 + 4, B.BLACK, B.R, hidden=False)       # 黑车 (1,4) 将军
    b.put(5 * 9 + 0, B.RED, B.R, hidden=False)         # 红车 (5,0)
    b.put(3 * 9 + 0, B.RED, B.P, hidden=False)         # 红兵 (3,0)
    b.turn = B.RED
    return b


def main():
    Legacy = load_legacy()
    board = build_position()
    eng = board.clone_revealed_only()
    legal_moves = {B.move_str(m) for m in eng.gen_moves(B.RED)}
    print("引擎认可的合法着法（应将）：")
    for m in sorted(legal_moves):
        print("   ", m)

    gs = FakeGS(engine_board_to_gui(board), current_player="red", ai_color="red")
    ai = Legacy(gs)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ordered = ai._get_ordered_moves(gs)
    print("\n旧 AI 排序后的前 5 个候选着法：")
    for mv in ordered[:5]:
        text = B.move_str(B.encode(mv[0][0] * 9 + mv[0][1], mv[1][0] * 9 + mv[1][1]))
        print(f"    {text}  合法={'是' if text in legal_moves else '否（送将/违规）'}")

    with contextlib.redirect_stdout(buf):
        chosen = ai.get_best_move()
    if chosen is None:
        print("\n旧 AI 未给出着法")
        return 0
    text = B.move_str(B.encode(chosen[0][0] * 9 + chosen[0][1],
                               chosen[1][0] * 9 + chosen[1][1]))
    print(f"\n旧 AI 最终选择：{text}")
    print(f"该着法在引擎看来是否合法：{'是' if text in legal_moves else '否（送将/违规）'}")

    # 旧 AI 用 game_score 做最终取舍（而不是排序用的 calculate_piece_move），
    # 这里把两者的分数都打出来对比。
    print("\n各候选着法在旧 AI 的两套评分下的表现：")
    print(f"    {'着法':<18}{'排序分':>14}{'最终评估分':>14}   合法")
    for mv in ordered[:6]:
        (sr, sc), (er, ec) = mv
        text_m = B.move_str(B.encode(sr * 9 + sc, er * 9 + ec))
        with contextlib.redirect_stdout(buf):
            order_score = ai.calculate_piece_move(gs, mv)
            sim = ai._simulate_move(gs, mv)
            final_score = ai.game_score(sim)
        print(f"    {text_m:<18}{order_score:>14.0f}{final_score:>14.0f}"
              f"   {'是' if text_m in legal_moves else '否'}")

    if text not in legal_moves:
        print("\n=> 旧 AI 走出了「不应对将军」的违规着法；main.py 会直接把它落到棋盘上。")
    else:
        print("\n（本局面下旧 AI 恰好选了合法着法，但它的最终取舍函数对'被将军'"
              "只扣 15 分，见上表排序分与最终评估分的巨大差异。）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
