"""定位 legal_filter 的漏判：找出"优化版认为合法、暴力版认为非法"的着法。"""
import random
import sys
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
sys.path.insert(0, str(GAME_DIR.parent))
sys.path.insert(0, str(GAME_DIR))

from jieqi import board as B  # noqa: E402
from jieqi.consts import BLACK, RED, SQ_BIT  # noqa: E402


def brute_legal(board, side):
    out = []
    for mv in board.gen_pseudo(side):
        board.make(mv)
        ok = not board.is_attacked(board.kings[side], 1 - side)
        board.unmake()
        if ok:
            out.append(mv)
    return out


def find_first_mismatch():
    for seed in range(40):
        b = B.Board.new_game(random.Random(seed))
        rng = random.Random(seed * 17 + 3)
        for ply in range(40):
            side = b.turn
            fast = set(b.gen_moves(side))
            slow = set(brute_legal(b, side))
            extra = sorted(fast - slow)
            missing = sorted(slow - fast)
            if extra or missing:
                return b, seed, ply, side, extra, missing
            moves = b.gen_moves(side)
            if not moves:
                break
            b.make(rng.choice(moves))
    return None


def explain(board, side, mv):
    frm, to = mv >> 7, mv & 127
    ks = board.kings[side]
    print(f"    着法 {B.move_str(mv)}  棋子={board.piece_char(frm)} "
          f"(hidden={board.hidden[frm]}, rule_ident={board.rule_ident[frm]})")
    print(f"    该方将帅在 {ks} = {(ks // 9, ks % 9)}")
    print(f"    _critical_squares = {sorted(board._critical_squares(side))} "
          f"-> {[(s // 9, s % 9) for s in sorted(board._critical_squares(side))]}")
    print(f"    frm={frm} 在关键格里？ {frm in board._critical_squares(side)}")
    print(f"    frm == 将帅格？ {frm == ks}")
    print(f"    当前是否被将（走子前）？ {board.is_attacked(ks, 1 - side)}")
    board.make(mv)
    after_ks = board.kings[side]
    attacked = board.is_attacked(after_ks, 1 - side)
    print(f"    走完后将帅在 {after_ks} = {(after_ks // 9, after_ks % 9)}，"
          f"被攻击？ {attacked}")
    board.unmake()
    # 找出是谁在攻击
    board.make(mv)
    ks2 = board.kings[side]
    attackers = []
    for sq in range(90):
        if board.side[sq] == 1 - side and board.rule_ident[sq]:
            if board.is_attacked(ks2, 1 - side):
                # 逐个测试：把该子拿掉后是否还被攻击
                ident = board.ident[sq]
                side0 = board.side[sq]
                hidden = board.hidden[sq]
                rule0 = board.rule_ident[sq]
                board.ident[sq] = 0
                board.side[sq] = -1
                board.rule_ident[sq] = 0
                bit = SQ_BIT[sq]
                board.occ ^= bit
                board.occ_side[side0] ^= bit
                still = board.is_attacked(ks2, 1 - side)
                board.ident[sq] = ident
                board.side[sq] = side0
                board.hidden[sq] = hidden
                board.rule_ident[sq] = rule0
                board.occ |= bit
                board.occ_side[side0] |= bit
                if not still:
                    attackers.append(sq)
    print(f"    攻击者格子：{attackers} -> "
          f"{[(s // 9, s % 9, board.piece_char(s)) for s in attackers]}")
    board.unmake()


def main():
    found = find_first_mismatch()
    if not found:
        print("没有找到不一致")
        return 0
    board, seed, ply, side, extra, missing = found
    print(f"seed={seed} ply={ply} side={side}")
    print(f"  优化版多出的着法: {[B.move_str(m) for m in extra]}")
    print(f"  优化版漏掉的着法: {[B.move_str(m) for m in missing]}")
    print("局面：")
    print(board.to_text())
    for mv in extra:
        print("  --- 分析多出的着法 ---")
        explain(board, side, mv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
