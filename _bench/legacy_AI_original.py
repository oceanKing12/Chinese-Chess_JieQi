import random
from game_state import GameState
import copy
from position_tables import Position_Value
from typing import Tuple
from Game.config import Config
from typing import Tuple, Dict, Any
import json
class AdvancedAIController:
    PIECE_VALUES = {
        '将': 1000, '帅': 1000,
        '车': 90, '車': 90,
        '马': 40, '馬': 40,
        '炮': 45, '砲': 45,
        '士': 20, '仕': 20,
        '象': 20, '相': 20,
        '卒': 10, '兵': 10,
        '暗': 10
    }

    def __init__(self, game_state: GameState):
        self.gs = game_state
        self.ai_color = game_state.ai_color

        self.max_depth = 1
        self.transposition_table = {}

        self.new_board = None
        self.config = Config()
        self.dark_value_ai = None
        self.dark_value_player = None
        self.SET_piece_scores = 0.6
        self.SET_piece_position_scores = 0.1
        self.SET_revealed = 1
        self.check_reward = 15
        self.check_punish = 20
        self.move_weight = 0.2
        self.position_value_weight = 0.2
        self.piece_in_guarded = 1.25#被保护加成
        self.piece_in_attacked=0.5#被攻击价值打折

        self.death_value=9999999
    def get_best_move(self) -> tuple:
        """带置换表的多步Minimax搜索"""
        self.transposition_table = {}
        self.decision_tree = {
            'type': 'root',
            'depth': self.max_depth,
            'alpha': -999999999,
            'beta': 999999999,
            'children': [],
            'value': None
        }
        #避免一子多动
        repeat_move = self.gs.ai_check_fold_repetition()
        best_move = None
        alpha = -999999999
        beta = 999999999
        # 获取所有合法移动并按历史得分排序（启发式排序）
        game_score = self.game_score(self.gs)  # 当前移动方的得分
        print(f'初始ai局面的评分：{game_score}')
        legal_moves = self._get_ordered_moves(self.gs)
        if repeat_move and len(legal_moves)>1 and (repeat_move in legal_moves):
            print('重复局面，变招')
            legal_moves.remove(repeat_move)
        pruned = False
        print("==================================")
        print(f'ai合法移动步数：{len(legal_moves)}')
        for idx, move in enumerate(legal_moves):
            type = self.get_move_piece_type(self.gs.board, move[0])
            if pruned:
                # 记录被剪枝的移动
                self._add_pruned_move(self.decision_tree, move,type)
                continue
            (r,c),(nr,nc)=move
            print(f'模拟移动：{(r,c)}→{nr,nc}')
            new_state = self._simulate_move(self.gs, move)
            # 递归搜索
            value, subtree = self._minimax(new_state, self.max_depth - 1, alpha, beta, False)
            print(f'{self.gs.current_player}玩家{(r,c)}→{nr,nc}移动最终ai得分：{value}')
            # 添加当前移动节点到决策树
            print(alpha, value, beta)
            print("==================================")
            move_node = self._create_move_node(move, type,value, subtree)
            self.decision_tree['children'].append(move_node)

            if value > alpha:
                alpha = value
                best_move = move

            if alpha >= beta:
                print(f"Beta剪枝: Beta({beta}) <= Alpha({alpha})")
                pruned = True
                # 添加后续剪枝的移动
                self._add_pruned_move(self.decision_tree, move)
                break

        self.decision_tree['value'] = alpha

        with open('decision_tree.json', 'w', encoding='utf-8') as f:
            json.dump(self.decision_tree, f, ensure_ascii=False, indent=2)
        return best_move

    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float,
                 maximizing_player: bool) -> Tuple[float, Dict[str, Any]]:
        # 检查置换表
        # 创建当前节点数据结构
        node = {
            'depth': depth,
            'alpha': alpha,
            'beta': beta,
            'value': None,
            'maximizing': maximizing_player,
            'children': [],
            'pruned_moves': []
        }
        # 终止条件：达到最大深度或游戏结束
        print(f'当前深度:{depth}')
        legal_moves = self._get_ordered_moves(state)
        if len(legal_moves)==0:
            print('当前玩家没有合法移动')
            node['value'] = round(self.game_score(state),2)
            return node['value'], node
        if depth == 0 or state.game_over:
            if state.game_over:print('game over!return')
            node['value'] = round(self.game_score(state),2)
            return node['value'], node
        # 生成有序移动列表（提高剪枝效率）

        pruned = False
        print(f'{'ai玩家的合法步数：'if maximizing_player else '人类玩家的合法步数：'}{len(legal_moves)}')
        print('\n'.join(map(str, legal_moves)))
        if maximizing_player:
            node['value'] = -999999999
            for idx, move in enumerate(legal_moves):
                type = self.get_move_piece_type(state.board, move[0])
                (r, c), (nr, nc) = move
                if pruned:
                    self._add_pruned_move(node, move,type)
                    continue
                new_state = self._simulate_move(state, move)
                child_value, child_node = self._minimax(new_state, depth-1, alpha, beta, False)
                print(f'{state.current_player}玩家{(r,c)}→{nr,nc}移动最终ai得分：{child_value}')

                # 更新最大值
                if child_value > node['value']:
                    node['value'] = child_value
                alpha = max(alpha, node['value'])
                # 添加子节点

                move_node = self._create_move_node(move,type, child_value, child_node)
                node['children'].append(move_node)

                if beta <= alpha:
                    print(f"Beta剪枝: Beta({beta}) <= Alpha({alpha})")
                    pruned = True
                    self._add_pruned_move(node, move,type)
                    break
        else:
            node['value'] =999999999
            for idx, move in enumerate(legal_moves):
                type = self.get_move_piece_type(state.board, move[0])
                if pruned:
                    self._add_pruned_move(node, move,type)
                    continue
                (r, c), (nr, nc) = move
                print(f'模拟人类玩家移动：{(r, c)}→{nr, nc}')
                new_state = self._simulate_move(state, move)
                child_value, child_node = self._minimax(new_state, depth-1, alpha, beta, True)
                print(f'{state.current_player}玩家{(r, c)}→{nr, nc}移动最终ai得分：{child_value}')

                # 更新最小值
                if child_value < node['value']:
                    node['value'] = child_value
                beta = min(beta, node['value'])

                #添加子节点

                move_node = self._create_move_node(move, type,child_value, child_node)
                node['children'].append(move_node)

                # Alpha-Beta剪枝
                if beta <= alpha:
                    print(f"Beta剪枝: Beta({beta}) <= Alpha({alpha})")
                    pruned = True
                    self._add_pruned_move(node, move,type)
                    break
        return node['value'], node
    def get_move_piece_type(self,board,pos):
        move_piece = board[pos[0]][pos[1]]
        type = (f'{'红' if move_piece['color']=='red'else '黑'}' if move_piece['revealed'] else '暗')+move_piece[f'{'actual' if move_piece['revealed'] else 'original'}']
        return type
    def _create_move_node(self, move: Tuple[Tuple[int, int], Tuple[int, int]],type,
                         value: float, subtree: Dict) -> Dict[str, Any]:
        """创建移动节点数据结构"""
        return {
            'type': 'move',
            'move': {
                'from': move[0],
                'to': move[1],
                'piece': type
            },
            'value': value,
            'children': subtree['children'],
            'pruned': False,
            'alpha': subtree.get('alpha', None),
            'beta': subtree.get('beta', None)
        }

    def _add_pruned_move(self, parent_node: Dict, move: Tuple[Tuple[int, int], Tuple[int, int]],type):
        """添加被剪枝的移动记录"""
        parent_node['children'].append({
            'type': 'move',
            'move': {
                'from': move[0],
                'to': move[1],
                'piece': type
            },
            'value': None,
            'pruned': True,
            'children': []
        })
    def _get_state_hash(self, state: GameState) -> str:
        """生成唯一局面标识（Zobrist哈希简化版）"""
        hash_str = []
        for r in range(10):
            for c in range(9):
                piece = state.board[r][c]
                if piece:
                    hash_str.append(f"{r}{c}{piece['color'][0]}{piece['actual']}{int(piece['revealed'])}")
        return '|'.join(sorted(hash_str))  # 排序消除顺序影响

    def _simulate_move(self, state: GameState, move: tuple) -> GameState:
        """深度拷贝并执行移动"""
        new_state = copy.deepcopy(state)
        sr, sc = move[0]
        er, ec = move[1]
        # 执行移动逻辑（需与实际游戏规则一致）
        new_state.board[er][ec] = new_state.board[sr][sc]
        new_state.board[er][ec]['revealed']=True
        new_state.board[sr][sc] = None
        new_state.current_player = 'black' if new_state.current_player == 'red' else 'red'
        new_state.check_game_state()
        return new_state

    def _get_ordered_moves(self, state: GameState) -> list:
        moves = []
        for r in range(10):
            for c in range(9):
                piece = state.board[r][c]
                if piece and piece['color'] == state.current_player:
                    for nr in range(10):
                        for nc in range(9):
                            if self.validate_move(r, c, nr, nc,state.board):
                                move = ((r, c), (nr, nc))
                                score = self.calculate_piece_move(state, move)
                                moves.append((move, score))
                                #print(f'{(r,c)}→{(nr,nc)}，得分：{score}\n')
                                #if score>0 or len(moves)==0:moves.append((move, score))#去掉负分移动并至少保留一个行动
        # 按评分降序排序
        moves.sort(key=lambda x: x[1], reverse=True)
        #moves=moves[0:2]
        return [m[0] for m in moves]

    def calculate_piece_move(self,state,move):#计算棋子的移动分：吃子，将军，揭开暗子，是否会导致自己被吃
        scores=0
        (r,c),(er,ec)=move
        piece=state.board[r][c]
        target_piece=state.board[er][ec]
        capture_score=0
        if target_piece and target_piece['color']!= piece['color']:
            if target_piece['revealed']:
                #明子的价值=自身价值+位置价值
                capture_score=self._piece_total_light(target_piece,(er,ec))
                #print(f'吃明子奖励:{capture_score}')
            else:
                #暗子的价值=暗子的概率价值+位置价值
                capture_score=self._piece_total_dark(target_piece,(er,ec),state.board,target_piece['color'])
                #print(f'吃暗子奖励:{capture_score}')
        scores+=capture_score
        #揭示棋子奖励,随机
        reveal_score=0
        if not piece['revealed']:
            reveal_score = self.dark_piece_value(state.board,piece['color'])
            #print(f'揭示暗子棋子奖励:{reveal_score}')
            # 移动后新棋子的位置分,移动前并不知道新棋子是什么，应该为概率分
            dark_piece_firstmove_score=self.dark_piece_firstmove_score(self.gs.board,self.ai_color,er, ec)
            target_position_value = dark_piece_firstmove_score-self._piece_position_value(piece['original'], (r, c))
            #print(f'移动后新棋子的位置分差值:{target_position_value}')
            # 该移动造成的新分数=揭示奖励*系数+位置分*系数
            scores += self.Jiaquan(reveal_score, target_position_value)
            #print(f'移动明棋的位置分差值:{target_position_value}')
        moved_score=self.piece_moved(r,c,er,ec,state)
        return round(scores+moved_score, 2)
    def piece_moved(self,r,c,er,ec,state):
        #移动后的情况
        board=copy.deepcopy(state.board)
        player =state.current_player
        #print(player)
        s_piece=board[r][c]
        s_revealed=s_piece['revealed']
        target_piece = board[er][ec]
        board[er][ec]=s_piece
        board[er][ec]['revealed']=True
        board[r][c]=None

        red_king,black_king=self._get_king_positions(board)
        attack=self.attack(board,(er,ec),s_revealed)
        #print(f'{'移动后导致自身被将军，-1000' if my_in else ''},{'移动后将军，+50' if op_in else ''}')
        #3.移动后是否导致自己被吃，er,ec现在是移动后的位置
        loss=self.loss(board,red_king, black_king,player) #移动后失子分
        #移动后解除我方的威胁。防守
        scores=attack-loss
        #print(f'移动后总分数：{scores}')
        return scores
    def attack(self,board,position,revealed):
        score=0
        piece=board[position[0]][position[1]]#piece为移动后的子
        #1.捉子
        if revealed:#明子才知道移动后的威胁分
            for r in range(10):
                for c in range(9):
                    current_piece = board[r][c]
                    if current_piece:
                        if current_piece['color'] == self.ai_color:
                            pass
                        # 这个棋子是ai的
                        else:
                            # 检查这个棋子是否处于当前棋子的攻击范围
                            if self.gs.validate_move(position[0], position[1], r, c, board):
                                if current_piece['actual'] == '帅' or current_piece['actual'] == '将':
                                    #print(f'将军得分：{self.check_reward}')
                                    score += self.check_reward
                                    return score
                                attack_value=0
                                is_guarded=self.is_guarded(board,current_piece['color'],(r,c))
                                if current_piece['revealed']:
                                    attack_value=self._piece_total_light(current_piece,(r,c))
                                else:
                                    attack_value=self._piece_total_dark(current_piece,(r,c),board,current_piece['color'])
                                attack_value-=self.piece_in_guarded if is_guarded else 0
                                #print(f'威胁{'有根'if is_guarded else '无根'}棋子{r}{c},加分{attack_value}')
                                score += attack_value
                            pass
                            # 这个棋子是玩家的
        #print(f'攻击总分:{score}')
        return score
    def loss(self,board,red_king,black_king,player):#target是移动后的位置
        score=0
        king_death = self._check_in(board, player, red_king, black_king) or self._check_face_to_face(board,red_king,black_king)
        if king_death:
            #print(f'移动后导致被将或对面笑，扣分：{9999999}')
            score+=99999999
        #print(f'失分:{score}')
        return score
    def _check_in(self, board, color: str, red_king_pos: Tuple[int, int], black_king_pos: Tuple[int, int]) -> bool:
        """检查color是否被将军状态"""
        if not (red_king_pos and black_king_pos): return False
        target_pos = red_king_pos if color == 'red' else black_king_pos#待检验是否被吃的子的位置
        opponent = 'black' if color == 'red' else 'red'
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece and piece['color'] == opponent:
                    if self.gs.validate_move(r, c, target_pos[0], target_pos[1], board):
                        #print(r, c, target_pos[0], target_pos[1])
                        return True
        return False
    def pre_move_score(self, board,position):
        score = 0
        start_value = 0
        (r,c)=position
        piece=board[r][c]
        print(piece)
        is_attacked = self.is_guarded(board, f'red'if piece['color']=='black' else 'red', (r,c))
        print(f'移动前该棋子是否被攻击{is_attacked}')
        if is_attacked:
            if piece['actual'] == '将' or piece['actual'] == '帅':
                print('被将军，加分：100')
                score += 100
                return score
            if piece['revealed']:
                start_value = self._piece_total_light(piece, piece['position'])
                print(f'明子的初始移动权重{start_value}')
            else:
                start_value = self._piece_total_dark(piece, piece['position'], self.gs.board, self.ai_color)
                print(f'暗子的初始移动权重{start_value}')
            score += start_value

        '''is_guarded = self.is_guarded(self.gs.board, self.ai_color, piece['position'])
        print(is_attacked, is_guarded)
        if is_attacked:
            if not is_guarded:  # 被攻击并且没有保护
                print(f'不移动的话失去分数：{start_value}')
                score += start_value
            else:
                score += self.in_guard_score
        print(f'棋子移动总权重:{score}')
        return round(score * self.move_weight, 2)'''
    def is_guarded(self,board,color,position):#color方的棋子能不能到position这个位置
        for r in range(10):
            for c in range(9):
                piece=board[r][c]
                if piece and piece['color']==color:
                    if self.validate_move(r,c,position[0],position[1],board):
                        return True
        return False
    def validate_move(self, sr: int, sc: int, er: int, ec: int, board) -> bool:
        """验证移动合法性"""
        board = board
        piece = board[sr][sc]
        target = board[er][ec]
        if target and target['color'] == piece['color']:
            return False
        if piece['actual'] in ['将', '帅']:
            return self.config._validate_king(sr, sc, er, ec, board)
        # 根据棋子类型调用对应的验证器
        rule_type = piece['original'] if not piece['revealed'] else piece['actual']
        validator_name = self.config.MOVE_VALIDATORS.get(rule_type, '_invalid_move')
        validator = getattr(self.config, validator_name, self.config._invalid_move)
        return validator(sr, sc, er, ec, board)

    def game_score(self,state):
        #color是当前玩家
        print(f'判断{state.current_player}局面')
        ai_value=0
        player_value=0
        current_player=state.current_player
        opponent='black' if current_player=='red' else 'red'
        pieces_scores=[[{} for i in range(9)] for s in range(10)]
        red_king,black_king=self._get_king_positions(state.board)
        if red_king and black_king:
            face_to_face=self._check_face_to_face(state.board, red_king, black_king)
            if face_to_face:#对面笑，当前玩家的回合开始时对面笑，当前玩家赢
               if current_player==state.ai_color:ai_value+=self.death_value
               else:player_value+=self.death_value
        moves_scores=0
        for r in range(10):
            for c in range(9):
                piece = state.board[r][c]
                #print(piece)
                is_attacked,is_guarded=False,False
                if piece:
                    moves_scores = self._get_piece_moves(state.board, r, c)
                    if piece['revealed']:#棋子的基本价值
                       piece_value = self._piece_total_light(piece, (r, c))*self.SET_revealed
                    else:
                       piece_value = self._piece_total_dark(piece, (r, c), state.board, piece['color'])
                    is_attacked=self.is_guarded(state.board,opponent,(r,c))
                    if is_attacked:
                       if piece['actual']==('将'or '帅'):
                           print('被将军')
                           if piece['color']==state.ai_color:ai_value-=(self.check_punish if state.current_player==state.ai_color else self.death_value)
                           else:player_value-=(self.death_value if state.current_player==state.ai_color else self.check_punish)
                           continue
                       piece_value=self.piece_in_attacked*piece_value
                       is_guarded=self.is_guarded(state.board,piece['color'],(r,c))
                       if is_guarded:
                            piece_value = self.piece_in_guarded * piece_value
                    #print(piece_value)
                    #pieces_scores[r][c]={'value':piece_value,'is_guarded':is_guarded,'is_attacked':is_attacked}
                    if piece['color']==state.ai_color:
                        ai_value+=piece_value+moves_scores
                    else:
                        player_value+=piece_value+moves_scores
        return moves_scores+ai_value-player_value

    def _piece_total_light(self,piece,piece_position):#明子的价值=自身价值+位置价值
        piece_scores = self._piece_value(piece['actual'])
        piece_position_scores = self._piece_position_value(piece['actual'], piece_position)
        return self.Jiaquan(piece_scores,piece_position_scores)
    def _piece_total_dark(self,piece,piece_position,board,color):#暗子的价值=暗子的期望价值+初始位置价值
        piece_scores =self.dark_piece_value(board,color)
        #print(piece_scores)
        piece_position_scores = self._piece_position_value(piece['original'], piece_position)
        #print(piece_position_scores)
        return self.Jiaquan(piece_scores,piece_position_scores)
    def _piece_value(self, piece_type: str) -> float:
        """获取棋子基础价值"""
        return self.PIECE_VALUES.get(piece_type, 0)
    def dark_piece_value(self,board,color):#输出暗子的期望价值
        darks=self.all_darks(board,color)
        if darks:
            dark_value=sum(self._piece_value(item['actual']) for item in darks)/len(darks)
            #print(f'暗子的期望价值:{dark_value}')
            return dark_value
        else:return 0
    def dark_piece_firstmove_score(self,board,color,er,ec):#暗子移动后获得新子的位置分，因为未知，所以为概率分
        darks = self.all_darks(board, color)
        if darks:
            score=sum(self._piece_position_value(item['actual'],(er,ec)) for item in darks)/len(darks)
            #print(f'暗子首次移动获得的期望位置分:{score}')
            return score
        else:return 0
    def all_darks(self,board,color):
        darks=[]#所有暗子
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece and piece['color'] == color and not piece['revealed']:darks.append(piece)
        return darks
    def _piece_position_value(self,type,position):
        return round(Position_Value.position_tables[0][type][position[0]][position[1]]*self.position_value_weight,2)
    def Jiaquan(self,piece_scores,piece_position_scores):
        return round(self.SET_piece_scores*piece_scores+piece_position_scores*self.SET_piece_position_scores,2)
    def _check_face_to_face(self, board, red_king_pos: Tuple[int, int], black_king_pos: Tuple[int, int]) -> bool:
        """检查将帅照面"""
        if not (red_king_pos and black_king_pos):return False
        if red_king_pos[1] != black_king_pos[1]:
            return False
        start = min(red_king_pos[0], black_king_pos[0]) + 1
        end = max(red_king_pos[0], black_king_pos[0])
        return not any(board[r][red_king_pos[1]] for r in range(start, end))
    def _get_king_positions(self, board):
        return (
            next(((r, c) for r in range(10) for c in range(9) if (p := board[r][c]) and p['actual'] == '帅'), None),
            next(((r, c) for r in range(10) for c in range(9) if (p := board[r][c]) and p['actual'] == '将'), None)
        )
    def _get_piece_moves(self,board,sr,sc):
        cc_piece=board[sr][sc]
        score=0
        for er in range(10):
            for ec in range(9):
                if self.validate_move(sr,sc,er,ec,board):
                    piece_type=cc_piece['actual'] if cc_piece['revealed'] else cc_piece['original']
                    score+=self.config.PIECE_MOVE_VALUE.get(piece_type,0)
        return round(score*self.move_weight,2)



