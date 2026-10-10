"""检查 tests 里的 forced_mate 参照实现是否正确：把"杀棋线路"打出来人工核对。"""
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / "Game"))
sys.path.insert(0, str(STAGE))

from jieqi import board as B  # noqa: E402
from jieqi.consts import BLACK, K, P, RED  # noqa: E402


def find_line(board, side, plies, line=None):
    """返回一条强制杀线路（list of move），找不到返回 None。"""
    line = line if line is not None else []
    if plies <= 0:
        return None
    for mv in board.gen_moves(side):
        board.make(mv)
        try:
            opp = 1 - side
            if board.kings[opp] < 0:
                return line + [mv]
            replies = board.gen_moves(opp)
            if not replies:
                return line + [mv]
            if plies >= 3:
                sub = None
                for reply in replies:
                    board.make(reply)
                    got = find_line(board, side, plies - 2, line + [mv, reply])
                    board.unmake()
                    if got is None:
                        break
                    sub = got
                else:
                    return sub
        finally:
            board.unmake()
    return None


def main():
    b = B.Board.empty()
    b.put(0 * 9 + 4, RED, K)
    b.put(9 * 9 + 3, BLACK, K)
    b.put(7 * 9 + 3, RED, P, hidden=False)
    b.turn = RED
    print("初始局面：")
    print(b.to_text())
    line = find_line(b, RED, 7)
    print("forced_mate 声称的杀棋线路：", line)
    if line:
        for mv in line:
            print(f"  {B.move_str(mv)}  {b.piece_char(mv >> 7)}")
            b.make(mv)
        print("走完之后：")
        print(b.to_text())
        print("黑方合法着法：", [B.move_str(m) for m in b.gen_moves(BLACK)])
        print("黑将被吃？", b.kings[BLACK] < 0)


if __name__ == "__main__":
    main()
