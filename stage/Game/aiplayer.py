import random
from game_state import GameState
import copy
from position_tables import Position_Value
from typing import Tuple
from Game.config import Config
class AdvancedAIController:
    PIECE_VALUES = {
        '将': 1000, '帅': 1000,
        '车': 90, '車': 90,
        '马': 40, '馬': 40,
        '炮': 45, '砲': 45,
        '士': 20, '仕': 20,
        '象': 20, '相': 20,
        '卒': 10, '兵': 10,#m每个暗子价值默认为
        '暗':10
    }

    def __init__(self, game_state: GameState):
        self.gs = game_state
        self.ai_color = game_state.ai_color
        self.max_depth = game_state.config.AI['ai_search_depth']  # 搜索深度
        self.transposition_table = {}
        self.new_board=None#临时棋盘
        self.Ai_grads=0# 置换表缓存

        self.config=Config()
        self.dark_value_ai=None
        self.dark_value_player=None
        self.SET_piece_scores = 0.2
        self.SET_piece_position_scores = 0.1
        self.check_reward=15
        self.check_punish=1000
        self.move_weight=0.2
        self.position_value_weight=0.3#位置分系数
        self.in_guard_score=5#有根子被攻击时默认的移动分

    def _piece_value(self, piece_type: str) -> float:
        """获取棋子基础价值"""
        return self.PIECE_VALUES.get(piece_type, 0)
    def dark_piece_value(self,board,color):#输出暗子的期望价值
        darks=self.all_darks(board,color)
        if darks:
            score=sum(self._piece_value(item['actual']) for item in darks)/len(darks)
            print(f'暗子的期望价值:{score}')
            return score
        else:return 0
    def dark_piece_firstmove_score(self,board,color,er,ec):#暗子移动后获得新子的位置分，因为未知，所以为概率分
        darks = self.all_darks(board, color)
        if darks:
            score=sum(self._piece_position_value(item['actual'],(er,ec)) for item in darks)/len(darks)
            print(f'暗子首次移动获得的新位置分:{score}')
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
    def _piece_total_light(self,piece,piece_position):#明子的价值=自身价值+位置价值
        piece_scores = self._piece_value(piece['actual'])
        piece_position_scores = self._piece_position_value(piece['actual'], piece_position)
        return self.Jiaquan(piece_scores,piece_position_scores)
    def _piece_total_dark(self,piece,piece_position,board,color):#暗子的价值=暗子的期望价值+初始位置价值
        piece_scores =self.dark_piece_value(board,color) if (board and color) else self.dark_value_ai
        piece_position_scores = self._piece_position_value(piece['original'], piece_position)
        return self.Jiaquan(piece_scores,piece_position_scores)
    def Jiaquan(self,piece_scores,piece_position_scores):
        return round(self.SET_piece_scores*piece_scores+piece_position_scores*self.SET_piece_position_scores,2)
    def dark_random(self):
        #返回暗子系数，应该根据目前的暗子数量及可能子力调整，初始为1-9的整数
        return random.randint(1,9)
    def get_best_move(self) -> tuple:
        """带调试输出的决策方法"""
        #当前AI方的局面评分
        #Ai_grads=self.calculate_Ai_grads(self.gs)
        scores=0
        #创建一个棋盘拷贝
        self.dark_value_ai = self.dark_piece_value(self.gs.board, self.gs.ai_color)  # 当前ai方暗子的期望价值
        self.dark_value_player = self.dark_piece_value(self.gs.board, self.gs.player_side)  # 当前人类方暗子的期望价值
        self.new_board =copy.deepcopy(self.gs.board)
        my_board=self._get_my_board(self.gs)
        moves=self.get_piece_scores(my_board)
        #print(f'移动前我方的局面总分：{piece_value}')
        best_mmove = max(moves, key=lambda x: x[2])
        print("-------------------------------------------")
        print('\n'.join(map(str, sorted(moves, key=lambda x: x[2], reverse=True))))
        print(f'最佳移动：{best_mmove[0]}→{best_mmove[1]},scores={best_mmove[2]}')
        print("-------------------------------------------")
        return (best_mmove[0],best_mmove[1])
    def get_piece_scores(self,board):
        moves = []
        before_move_game_score = self.game_score(self.gs.board, self.ai_color)
        print(f'移动前我方的被攻击分：{ before_move_game_score}')
        for piece in board:
            targets=piece['targets']
            #print(f'{piece['position'][0]}行{piece['position'][1]}列的{'明'+piece['actual'] if piece['revealed'] else '暗'+piece['original']}的合法移动：{len(targets)}')
            for target in targets:
                _type=piece['actual'] if piece['revealed'] else piece['original']
                before_move_score=self.calculate_before_piece_move(piece)
                in_move_score=self.calculate_piece_move(piece['actual'] if piece['revealed'] else piece['original'],piece['position'],target)
                moved_score=self.moved(piece['position'][0],piece['position'][1],target[0],target[1])
                game_changed=moved_score[1][1]-before_move_game_score
                print(f'局面改变:{game_changed}')
                move_total_score = round(in_move_score[0] + moved_score[0] + before_move_score+game_changed, 2)
                print(f'{_type}：{piece['position']}→{target},scores={move_total_score}')
                print("=======================")
                moves.append((piece['position'],target,move_total_score,('before_move_score',before_move_score),in_move_score,moved_score))
        return moves
    def calculate_before_piece_move(self,piece):
        score=0
        start_value=0
        print(piece['revealed'],piece['actual'])
        is_attacked = self.is_guarded(self.gs.board, self.gs.player_side, piece['position'])
        if piece['actual']=='将' or piece['actual']=='帅':
            if is_attacked:
                print('被将军，加分：100')
                score += 100
            return score
        if piece['revealed']:
            start_value=self._piece_total_light(piece,piece['position'])
            print(f'明子的初始移动权重{start_value}')
        else:
            start_value =self._piece_total_dark(piece,piece['position'],self.gs.board,self.ai_color)
            print(f'暗子的初始移动权重{start_value}')
        score+=start_value
        is_guarded=self.is_guarded(self.gs.board,self.ai_color,piece['position'])
        print(is_attacked,is_guarded)
        if is_attacked:
            if not is_guarded:#被攻击并且没有保护
                print(f'不移动的话失去分数：{start_value}')
                score+=start_value
            else:
                score+=self.in_guard_score
        print(f'棋子移动总权重:{score}')
        return round(score*self.move_weight,2)
    def is_guarded(self,board,color,position):#color方的棋子能不能到position这个位置
        for r in range(10):
            for c in range(9):
                piece=board[r][c]
                if piece and piece['color']==color and (r,c)!=position:
                    if self.validate_move(r,c,position[0],position[1],board):
                        return True
        return False
    def calculate_piece_move(self,position,target):#计算棋子的移动分：吃子，将军，揭开暗子，是否会导致自己被吃
        scores=0
        r,c,er,ec=position[0],position[1],target[0],target[1]
        piece=self.gs.board[r][c]
        target_piece=self.gs.board[er][ec]
        capture_score=0
        if target_piece and target_piece['color']!= self.ai_color:
            if target_piece['revealed']:
                #明子的价值=自身价值+位置价值
                _type=target_piece['actual']
                capture_score=self.Jiaquan(self._piece_value(_type),self._piece_position_value(_type,(er,ec)))
                print(f'吃明子奖励:{self.Jiaquan(self._piece_value(_type),self._piece_position_value(_type,(er,ec)))}')
            else:
                #暗子的价值=暗子的概率价值+位置价值
                capture_score=self.Jiaquan(self.dark_value_player,self._piece_position_value(target_piece['original'],(er,ec)))
                print(f'吃暗子奖励:{self.Jiaquan(self.dark_value_player,self._piece_position_value(target_piece['original'],(er,ec)))}')
        scores+=capture_score
        #揭示棋子奖励,随机
        reveal_score=0
        target_position_value=0
        if not piece['revealed']:
            reveal_score = self.dark_value_ai
            print(f'揭示暗子棋子奖励:{reveal_score}')
            # 移动后新棋子的位置分,移动前并不知道新棋子是什么，应该为概率分
            dark_piece_firstmove_score=self.dark_piece_firstmove_score(self.gs.board,self.ai_color,er, ec)
            target_position_value = dark_piece_firstmove_score-self._piece_position_value(piece['original'], (r, c))
            print(f'移动后新棋子的位置分差值:{target_position_value}')
            # 该移动造成的新分数=揭示奖励*系数+位置分*系数
            scores += self.Jiaquan(reveal_score, target_position_value)
        else:
            target_position_value = self._piece_position_value(piece['actual'], (er, ec))-self._piece_position_value(piece['actual'], (r, c))
            scores +=target_position_value*self.SET_piece_position_scores
            print(f'移动明棋的位置分差值:{target_position_value}')
        print(f'吃子、揭示、移动总分:{scores}')
        return round(scores, 2),('capture_score:', capture_score),('reveal_score',reveal_score),('target_position_value',target_position_value)
    def calculate_Ai_grads(self):
        #2.明子价值+位置分数3.对将帅的威胁分数4.防守分数
        state=self.gs
        for r in range(10):
            for c in range(9):
                piece = state.board[r][c]
                if piece and piece['color'] == self.ai_color:
                    print(piece)
        # 1.暗子分数
    def _get_legal_moves(self, state: GameState) -> list:
        """生成所有合法移动（确保始终返回列表）"""
        moves = []
        for r in range(10):
            for c in range(9):
                piece = state.board[r][c]
                if piece and piece['color'] == self.ai_color:
                    for nr in range(10):
                        for nc in range(9):
                            if state.validate_move(r, c, nr, nc):
                                if not self.try_logical_move(state,r, c, nr, nc):
                                    moves.append(((r, c), (nr, nc)))
        return moves
    def _get_my_board(self, state: GameState) -> list:
        """生成所有合法移动（确保始终返回列表）"""
        my_board=[]
        for r in range(10):
            for c in range(9):
                piece = state.board[r][c]
                if piece and piece['color'] == self.ai_color:
                    targets=[]
                    for nr in range(10):
                        for nc in range(9):
                            if state.validate_move(r, c, nr, nc):
                                #输出符合移动规则的移动，但未校验移动后的局面
                                #if not self.try_logical_move(state,r, c, nr, nc):
                                targets.append((nr, nc))
                    piece={**piece,'position':(r,c),'targets':targets}
                    my_board.append(piece)
        return my_board
    def try_logical_move(self,state: GameState,r,c,nr,nc):
        # 模拟移动并验证状态
        temp_board = copy.deepcopy(state.board)
        temp_board[nr][nc] = copy.deepcopy(temp_board[r][c])
        temp_board[r][c] = None
        temp_red, temp_black = state._get_king_positions(temp_board)
        # 将军和送将检测
        if state.in_check and state._check_check(temp_board, state.current_player, temp_red, temp_black):
            state.toast = f'{state.current_player}需要应将'
            state.selected = None
            return True

        if state._check_check(temp_board, state.current_player, temp_red, temp_black):
            state.toast = f'{state.current_player}不可送将'
            state.selected = None
            return True

        if state._check_face_to_face(temp_board, temp_red, temp_black):
            state.toast = '移动后导致对面笑，行动失败'
            state.selected = None
            return True
        return False

    def moved(self,r,c,er,ec):
        #移动后的情况
        scores=0
        board=self.new_board
        #dark_value_ = self.dark_piece_value(board, self.gs.ai_color)
        s_piece=copy.deepcopy(board[r][c])
        s_revealed=s_piece['revealed']
        e_piece = board[er][ec]
        board[er][ec]=s_piece
        board[er][ec]['revealed']=True
        board[r][c]=None
        red_king,black_king=self._get_king_positions(board)
        attack=self.attack(board,red_king,black_king,(er,ec),s_revealed)
        #print(f'{'移动后导致自身被将军，-1000' if my_in else ''},{'移动后将军，+50' if op_in else ''}')
        #3.移动后是否导致自己被吃，er,ec现在是移动后的位置
        loss=self.loss(board,s_piece,(r,c),(er,ec),red_king, black_king) #移动后失子分
        #移动后解除我方的威胁。防守
        after_move_game_score = self.game_score(board, self.ai_color)
        print(f'移动后我方的被攻击分：{after_move_game_score}')
        scores=attack-loss
        #判断结束后还原
        board[r][c]=s_piece
        board[r][c]['revealed']=s_revealed
        board[er][ec] =e_piece
        print(f'移动后总分数：{scores}')
        return round(scores,2),('after_move_game_score',after_move_game_score),('loss',loss),('attack',attack)
    def attack(self,board,red_king,black_king,position,revealed):
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
                                    print(f'将军得分：{self.check_reward}')
                                    score += self.check_reward
                                    return score
                                attack_value=0
                                is_guarded=self.is_guarded(board,current_piece['color'],(r,c))
                                if current_piece['revealed']:
                                    attack_value=self._piece_total_light(current_piece,(r,c))
                                else:
                                    attack_value=self._piece_total_dark(current_piece,(r,c),board,current_piece['color'])
                                attack_value-=self.in_guard_score if is_guarded else 0
                                print(f'威胁{'有根'if is_guarded else '无根'}棋子{r}{c},加分{attack_value}')
                                score += attack_value
                            pass
                            # 这个棋子是玩家的
        print(f'攻击总分:{score}')
        return score
    def guard(self):
        score=0
        #1.移动后的被攻击分-移动前的被攻击分#2.解除将军加分
        print(f'防守得分:{score}')
        return score
    #将军检查：
    def loss(self,board,piece,position,target,red_king,black_king):#target是移动后的位置
        score=0
        king_death = self._check_in(board, self.gs.ai_color, red_king, black_king) or self._check_face_to_face(board,red_king,black_king)
        if king_death:
            print(f'移动后导致被将或对面笑，扣分：{self.check_punish}')
            score=self.check_punish
        print(f'失分:{score}')
        return score
    def _get_king_positions(self, board):
        return (
            next(((r, c) for r in range(10) for c in range(9) if (p := board[r][c]) and p['actual'] == '帅'), None),
            next(((r, c) for r in range(10) for c in range(9) if (p := board[r][c]) and p['actual'] == '将'), None)
        )

    def _check_in(self, board, color: str, red_king_pos: Tuple[int, int], black_king_pos: Tuple[int, int]) -> bool:
        """检查color是否被将军状态"""
        target_pos = red_king_pos if color == 'red' else black_king_pos#待检验是否被吃的子的位置
        opponent = 'black' if color == 'red' else 'red'
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece and piece['color'] == opponent:
                    if self.gs.validate_move(r, c, target_pos[0], target_pos[1], board):
                        print(r, c, target_pos[0], target_pos[1])
                        return True
        return False

    def _check_face_to_face(self, board, red_king_pos: Tuple[int, int], black_king_pos: Tuple[int, int]) -> bool:
        """检查将帅照面"""
        if red_king_pos[1] != black_king_pos[1]:
            return False
        start = min(red_king_pos[0], black_king_pos[0]) + 1
        end = max(red_king_pos[0], black_king_pos[0])
        return not any(board[r][red_king_pos[1]] for r in range(start, end))
    def validate_move(self, sr: int, sc: int, er: int, ec: int, board) -> bool:
        """验证移动合法性"""
        board = board
        piece = board[sr][sc]
        target = board[er][ec]
        '''if target and target['color'] == piece['color']:
            return False'''
        if piece['actual'] in ['将', '帅']:
            return self.config._validate_king(sr, sc, er, ec, board)
        # 根据棋子类型调用对应的验证器
        rule_type = piece['original'] if not piece['moved'] else piece['actual']
        validator_name = self.config.MOVE_VALIDATORS.get(rule_type, '_invalid_move')
        validator = getattr(self.config, validator_name, self.config._invalid_move)
        return validator(sr, sc, er, ec, board)
    def game_score(self,board,color):#color的被攻击分，攻击分
        score=0
        opponent='black' if color=='red' else 'red'
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece and piece['color'] == color:
                   is_attacked=self.is_guarded(board,opponent,(r,c))
                   is_guarded=self.is_guarded(board,color,(r,c))
                   if piece['actual'] == '将' or piece['actual'] == '帅':
                       if is_attacked:
                           print('当前被将军，减分：100')
                           score -= 100
                   elif is_attacked and is_guarded:
                       if piece['revealed']:
                           start_value = self._piece_total_light(piece, (r, c))
                           start_value-=self.in_guard_score
                           print(f'明子有根棋子被攻击:{start_value}')
                       else:
                           start_value = self._piece_total_dark(piece, (r, c), board, color)
                           start_value -= self.in_guard_score
                           print(f'暗子有根棋子被攻击:{start_value}')
                       score -= start_value
                   elif is_attacked and not is_guarded:
                       if piece['revealed']:
                           start_value = self._piece_total_light(piece, (r,c))
                           print(f'明子无根棋子被攻击:{start_value}')
                       else:
                           start_value = self._piece_total_dark(piece, (r,c), board, color)
                           print(f'暗子无根棋子被攻击:{start_value}')
                       score -= start_value
                    #我方的棋子
        return score


