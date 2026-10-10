'''诊断：拆解 evaluate() 各项，并列出受攻明细。'''
import random, sys
from pathlib import Path
#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
G = _paths.GAME
sys.path.insert(0, str(G.parent)); sys.path.insert(0, str(G))
from jieqi import board as B
from jieqi.belief import Belief
from jieqi.consts import IDENT_CHARS, PIECE_VALUE, P, PST_FLAT_SIDE, RED, BLACK, SQUARES, SQ_BIT
from jieqi.evaluate import EvalContext, evaluate

def bd(b):
    ctx = EvalContext(Belief.from_board(b)); w = ctx.weights
    v = {}
    for sq in range(SQUARES):
        s = b.side[sq]
        if s < 0: continue
        if b.hidden[sq]:
            pool = ctx.belief.pools[s]
            v[sq] = sum(PIECE_VALUE[i] + PST_FLAT_SIDE[s][i][sq] for i in pool) / max(1, len(pool))
        else:
            v[sq] = PIECE_VALUE[b.ident[sq]] + PST_FLAT_SIDE[s][b.ident[sq]][sq]
    att = (b.attack_map(RED), b.attack_map(BLACK)); occ = b.occ_side
    mob = bin(att[0] & ~occ[RED]).count('1') - bin(att[1] & ~occ[BLACK]).count('1')
    mat = sum(v[sq] if b.side[sq] == RED else -v[sq] for sq in v)
    tot, det = 0.0, []
    for sq in v:
        s = b.side[sq]
        if not (att[1-s] & SQ_BIT[sq]): continue
        d = bool(att[s] & SQ_BIT[sq])
        loss = v[sq] * (w['attacked_defended'] if d else w['hanging'])
        tot += loss if s == BLACK else -loss
        det.append((sq, s, b.hidden[sq], b.rule_ident[sq], v[sq], d, loss))
    ev = evaluate(b, ctx)
    return {'材料+位置': mat, '机动性': w['mobility']*mob, '受攻': tot,
            '其余(将帅/将军/先手)': ev - mat - w['mobility']*mob - tot}, ev, det

def show(b, lab):
    t, ev, det = bd(b)
    print('=== %s | evaluate=%+d (黑方 %+d)' % (lab, ev, -ev))
    for k, x in t.items():
        if abs(x) > 1e-9: print('      %-18s %+9.1f' % (k, x))
    det.sort(key=lambda z: -abs(z[6]))
    for sq, s, hid, rule, val, d, loss in det[:6]:
        cn = IDENT_CHARS[(s, rule)] if rule else '?'
        print('      [受攻](%d,%d)%s%s 值%.0f %s 罚%+.1f'
              % (sq//9, sq%9, '红' if s==RED else '黑', cn + ('?' if hid else ''),
                 val, '有护' if d else '无护', loss))

def main():
    show(B.Board.new_game(random.Random(7)), '开局红先')
    b1 = B.Board.new_game(random.Random(7)); sq = 3*9+4; b1.ident[sq] = P
    b1.make(B.encode(sq, 4*9+4)); show(b1, '红兵(3,4)->(4,4)')
    for lab, mv in [('黑将(9,4)->(8,4)', B.encode(9*9+4, 8*9+4)),
                    ('黑卒(6,4)->(5,4)', B.encode(6*9+4, 5*9+4)),
                    ('黑卒(6,6)->(5,6)', B.encode(6*9+6, 5*9+6)),
                    ('黑車?(9,8)->(8,8)', B.encode(9*9+8, 8*9+8))]:
        b = b1.copy()
        if not b.is_legal(mv):
            print('=== %s 非法' % lab); continue
        b.make(mv); show(b, lab)
    return 0

sys.exit(main())
