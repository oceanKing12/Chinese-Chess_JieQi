# config.py
from typing import List

try:  # 规则阈值只有一处真相来源（见下面 DRAW_CONDITIONS 的说明）
    from jieqi.rules import RULE as _RULE
except ImportError:  # pragma: no cover - 兼容"直接 python main.py"的导入方式
    try:
        from Game.jieqi.rules import RULE as _RULE
    except ImportError:
        _RULE = {'idle_draw_plies': 80, 'repetition_draw': 6,
                 'repetition_warn': 4, 'max_plies': 200}


class Config:
    #: 棋盘格子边长。整窗高度 = 顶栏 + 上下展示条 48×2 + 棋盘高度，
    #: 所以格子直接决定窗口能不能塞进屏幕：这里按"整窗不超过 1000px"倒推
    #: （见下面的 ``_MAX_WINDOW_H`` 与文件末尾的自检）。
    #:
    #: 64px：棋盘 656×720 -> 整窗 1004×952。被吃棋子条收成一行小图标 +
    #: 顶栏去掉提示行之后，比原来（980×996）矮了 44px，棋盘反而更大。
    CELL_SIZE = 64
    OFF_SET = 40
    #: 棋盘区域尺寸（下面的 WIDTH/HEIGHT 一直指棋盘，不要混用）
    WIDTH = CELL_SIZE * 9 + OFF_SET * 2
    HEIGHT = CELL_SIZE * 10 + OFF_SET * 2

    # ------------------------------------------------------------------
    # 窗口尺寸上限：布局做完以后会自检（见文件末尾的 assert）。
    # 改 CELL_SIZE / TOP_BAR_H / BOARD_STRIP_H 时，只要不超这个上限就安全。
    # ------------------------------------------------------------------
    _MAX_WINDOW_H = 1000
    _MAX_WINDOW_W = 1060

    # ------------------------------------------------------------------
    # 三区布局：
    #   ┌───────────────────────────┬──────────┐
    #   │ 顶部配置区                 │          │
    #   ├───────────────────────────┤ 右侧信息 │
    #   │ 棋盘（下棋区）             │ 面板     │
    #   └───────────────────────────┴──────────┘
    TOP_BAR_H = 104         # 顶部配置区高度（两行控件 + 状态行）
    PANEL_W = 324           # 右侧信息面板宽度
    BOARD_X = 0             # 棋盘左上角在窗口中的位置

    # ------------------------------------------------------------------
    # 棋盘上下的「被吃棋子」展示条
    #   上方：对方（人类）吃掉的子      下方：本方（下方那一方）吃掉的子
    # 明子直接显示真身图标；被吃的暗子按规则身份未知，只画"暗子"图标。
    # 同种棋子（含暗子）合并成一个图标 + 右上角数量角标，所以**一行就够**。
    # ------------------------------------------------------------------
    BOARD_STRIP_H = 48          # 单条高度（一行 34px 图标 + 上下留白）
    STRIP_ICON = 34             # 被吃棋子的图标边长
    STRIP_GAP = 2               # 图标间距
    #: 展示条里标签预留的宽度（标签放在棋盘外侧，与图标永不相交）
    STRIP_LABEL_W = 150
    #: 棋盘本身（不含展示条）在窗口里的纵向位置
    BOARD_Y = TOP_BAR_H + BOARD_STRIP_H

    #: 整窗尺寸 = 棋盘 + 右侧面板（横向）、顶栏 + 上下展示条 + 棋盘（纵向）。
    WINDOW_W = WIDTH + PANEL_W
    WINDOW_H = BOARD_Y + HEIGHT + BOARD_STRIP_H

    #: 统一的间距 / 圆角令牌。所有排版都从这里取值，避免各处随手写数字
    #: 导致边距不齐（旧版就是按钮 x 坐标各写各的，看起来"没排过版"）。
    PAD_XS = 4
    PAD_S = 8
    PAD = 12
    PAD_L = 16
    PAD_XL = 24
    RADIUS_S = 6
    RADIUS = 10
    RADIUS_L = 16
    #: 顶部配置区**两行**各自的 y 与高度。
    #: AI 的走棋信息（用时/深度/评估）现在放在第一行右侧，不再单占一行；
    #: 键盘快捷键提示也不再显示（键位写在文档和 tooltip 文案里）。
    TOP_ROW1_Y = 10
    TOP_ROW1_H = 36
    TOP_ROW2_Y = 56
    TOP_ROW2_H = 26

    #: 面板/配置区的字号
    FONT_BRAND = 24
    FONT_TITLE = 20
    FONT_BODY = 18
    FONT_SMALL = 16
    FONT_CHIP = 15
    FONT_MOVE = 15
    FONT_TINY = 13
    FONT_OVER = 64
    FONT_OVER_HINT = 30

    #: 判和条件。
    #:
    #: ⚠ **真相来源只有一处**：具体阈值与判据全部在 ``jieqi/rules.py`` 的
    #: ``RULE`` 里（长将 / 长捉 / 无吃子 80 步 / 无过河子力 / 重复局面 / 200 步）。
    #: 下面这份字典是**从 ``RULE`` 派生的别名**，不再手写第二份数字——旧版本这里是
    #: ``{'max_no_capture': 40, 'threefold_repetition': 6}``，与规则模块各说各话，
    #: 改一处忘一处就会两边不一致（"两个真相来源"正是这类 bug 的温床）。
    #:
    #: 键名保留历史写法，方便老代码/老测试继续读：
    #: ``max_no_capture``（旧名，单位是**手** = 80 步）、``threefold_repetition``
    #: （旧名，现在等于 ``repetition_draw`` = 6）。
    DRAW_CONDITIONS = {
        'max_no_capture': int(_RULE['idle_draw_plies']),      # 旧名 = 无吃子手数
        'threefold_repetition': int(_RULE['repetition_draw']),  # 旧名 = 重复次数
        'idle_draw_plies': int(_RULE['idle_draw_plies']),
        'repetition_draw': int(_RULE['repetition_draw']),
        'repetition_warn': int(_RULE['repetition_warn']),
        'max_plies': int(_RULE['max_plies']),
    }
    AI={
        'ai_mode' :True , # 是否启用AI对战
        'ai_search_depth' : 1 # AI搜索深度
    }
    #: 配色令牌。键名保持向后兼容（旧代码/测试会直接取），
    #: 新增的是排版用的层次色，让面板看起来是"卡片"而不是一块大灰板。
    COLORS = {
        # ---- 棋子 / 棋盘基础 ----
        'red': (198, 66, 54),
        'black': (44, 46, 52),
        'board_top': (243, 216, 172),
        'board_bottom': (218, 176, 120),
        'selected_glow': (255, 214, 102),
        'grid': (118, 88, 54),
        'check': (214, 52, 44),
        'text_red': (255, 236, 214),
        'text_black': (214, 222, 240),
        'river_text': (120, 88, 52),
        'selection_primary': (255, 226, 158),   # 淡金色
        'selection_highlight': (255, 250, 226),  # 珍珠白
        'selection_secondary': (198, 142, 74),  # 古铜色
        # ---- 深色 UI ----
        'window_bg': (24, 22, 26),
        'bar_bg': (32, 30, 36),
        'bar_bg_alt': (26, 24, 30),
        'panel_bg': (28, 26, 32),
        'panel_card': (36, 34, 41),
        'panel_card_alt': (43, 40, 48),
        'panel_line': (58, 54, 64),
        'panel_line_soft': (45, 42, 51),
        'card_shadow': (14, 13, 16),
        # ---- 按钮 ----
        'btn_bg': (46, 43, 52),
        'btn_bg_hover': (62, 58, 70),
        'btn_bg_active': (214, 158, 74),
        'btn_text': (228, 224, 234),
        'btn_text_active': (26, 22, 16),
        'btn_disabled': (37, 35, 41),
        'btn_border': (66, 62, 74),
        # ---- 文字 ----
        'text_main': (234, 230, 240),
        'text_dim': (150, 145, 160),
        'text_faint': (106, 102, 116),
        'text_llm': (150, 200, 255),
        'text_warn': (255, 190, 110),
        'text_ok': (140, 214, 158),
        # ---- 强调 / 阵营 ----
        'accent': (214, 158, 74),
        'accent_soft': (122, 94, 50),
        'red_side': (214, 92, 78),
        'black_side': (130, 158, 214),
        # ---- 棋盘叠加标记 ----
        'last_move': (255, 196, 90),
        'hint_dot': (74, 134, 94),
        'hint_capture': (196, 88, 74),
        'coord_text': (150, 116, 72),
        # ---- 被吃棋子展示条 ----
        'strip_bg': (30, 28, 34),
        'strip_line': (52, 48, 58),
        #: 展示条的标题字色（红方 / 黑方各一），让人一眼看出这行是谁的子。
        'strip_red': (226, 128, 116),
        'strip_black': (146, 172, 222),
        # ---- 启动页背景（深色底 + 中心冷光 + 暖色晕染）----
        'start_bg': (17, 16, 20),
        'start_glow': (50, 54, 70),
        'start_warm': (82, 58, 38),
        #: 阵营徽记（启动页标题与按钮上的红/黑标记）
        'brand_red': (198, 66, 54),
        'brand_black': (58, 62, 74),
    }

    PIECE_TYPES = ['帅', '将', '仕', '士', '相', '象', '马', '馬', '车', '車', '炮', '砲', '兵', '卒']
    PIECE_MOVE_VALUE={
        '车': 4, '馬':5, '相': 2,
        '仕': 2, '炮': 3, '兵': 2,
        '卒': 2, '砲': 3, '马': 5,
        '象': 2, '士': 2, '車': 4
    }
    TRADITIONAL_LAYOUT = [
        ['车', '马', '相', '仕', '帅', '仕', '相', '马', '车'],
        [None] * 9,
        [None, '炮', None, None, None, None, None, '炮', None],
        ['兵', None, '兵', None, '兵', None, '兵', None, '兵'],
        [None] * 9,
        [None] * 9,
        ['卒', None, '卒', None, '卒', None, '卒', None, '卒'],
        [None, '砲', None, None, None, None, None, '砲', None],
        [None] * 9,
        ['車', '馬', '象', '士', '将', '士', '象', '馬', '車']
    ]

    MOVE_VALIDATORS = {
        '车': 'validate_chariot', '馬': 'validate_horse', '相': 'validate_elephant',
        '仕': 'validate_advisor', '炮': 'validate_cannon', '兵': 'validate_soldier',
        '卒': 'validate_soldier', '砲': 'validate_cannon', '马': 'validate_horse',
        '象': 'validate_elephant', '士': 'validate_advisor', '車': 'validate_chariot'
    }
    def _invalid_move(self, *args) -> bool:
        """无效移动处理"""
        return False
    def _validate_king(self, sr: int, sc: int, er: int, ec: int, board) -> bool:
        """将/帅移动验证"""
        piece = board[sr][sc]
        valid_zone = (0 <= er <= 2) if piece['color'] == 'red' else (7 <= er <= 9)
        return valid_zone and 3 <= ec <= 5 and (abs(er - sr) + abs(ec - sc) == 1)
    def _check_face_to_face(self, board, red_king_pos, black_king_pos) -> bool:
        """检查将帅照面"""
        if red_king_pos[1] != black_king_pos[1]:
            return False
        start = min(red_king_pos[0], black_king_pos[0]) + 1
        end = max(red_king_pos[0], black_king_pos[0])
        return not any(board[r][red_king_pos[1]] for r in range(start, end))
    def validate_chariot(self, sr: int, sc: int, er: int, ec: int, board) -> bool:
        """车移动验证"""
        if sr != er and sc != ec:
            return False
        step = 1 if er > sr or ec > sc else -1
        if sr == er:  # 横向移动
            return not any(board[sr][c] for c in range(sc + step, ec, step))
        else:  # 纵向移动
            return not any(board[r][sc] for r in range(sr + step, er, step))

    def validate_cannon(self, sr: int, sc: int, er: int, ec: int, board) -> bool:
        """炮移动验证"""
        if sr != er and sc != ec:
            return False

        obstacles = 0
        step = 1 if er > sr or ec > sc else -1

        if sr == er:  # 横向移动
            for c in range(sc + step, ec, step):
                if board[sr][c]:
                    obstacles += 1
        else:  # 纵向移动
            for r in range(sr + step, er, step):
                if board[r][sc]:
                    obstacles += 1

        # 炮需要恰好跨越一个障碍（吃子时）或没有障碍（移动时）
        return obstacles == (1 if board[er][ec] else 0)

    def validate_horse(self, sr: int, sc: int, er: int, ec: int, board) -> bool:
        """马移动验证"""
        dx = abs(er - sr)
        dy = abs(ec - sc)
        if (dx, dy) not in [(2, 1), (1, 2)]:
            return False

        # 检查蹩马腿
        if dx == 2:
            block_r = sr + (1 if er > sr else -1)
            if board[block_r][sc]:
                return False
        else:
            block_c = sc + (1 if ec > sc else -1)
            if board[sr][block_c]:
                return False
        return True

    def validate_elephant(self, sr: int, sc: int, er: int, ec: int, board) -> bool:
        """象/相移动验证：田字 + 塞象眼。

        揭棋里**士相可以过河**（"士相可以过河，增加了杀伤力"），
        所以这里刻意不加过河限制，明暗一致，与引擎 jieqi/consts.py 相同。
        """
        dx, dy = abs(er - sr), abs(ec - sc)
        if dx != 2 or dy != 2:
            return False
        return not board[(sr + er) // 2][(sc + ec) // 2]

    def validate_advisor(self, sr: int, sc: int, er: int, ec: int, board) -> bool:
        """仕/士移动验证。

        揭棋规则（国内通行 + 象棋程序维基）：
        * **明士可以过河、可以出九宫**，只保留“斜走一格”的形状限制；
        * **暗士**的第一步限在自家九宫内斜走一格。

        旧实现把暗士限制成“只有四个角进中心”（漏了中心回角），
        且明士没有任何限制，与本项目引擎的规则不一致，这里统一改正。
        """
        if abs(er - sr) != 1 or abs(ec - sc) != 1:
            return False
        piece = board[sr][sc]
        if piece['revealed']:
            return True  # 明士：自由斜走一格（可过河、可出九宫）
        if piece['color'] == 'red':
            return 0 <= er <= 2 and 3 <= ec <= 5
        return 7 <= er <= 9 and 3 <= ec <= 5

    def validate_soldier(self, sr: int, sc: int, er: int, ec: int, board) -> bool:
        """兵/卒移动验证"""
        piece = board[sr][sc]
        dx, dy = er - sr, abs(ec - sc)
        if piece['color'] == 'red':
            return dx == 1 and dy == 0 if sr <= 4 else (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
        return dx == -1 and dy == 0 if sr >= 5 else (dx == -1 and dy == 0) or (dx == 0 and dy == 1)
    def _get_king_positions(self, board):
        return (
            next(((r, c) for r in range(10) for c in range(9) if (p := board[r][c]) and p['actual'] == '帅'), None),
            next(((r, c) for r in range(10) for c in range(9) if (p := board[r][c]) and p['actual'] == '将'), None)
        )
    def repate(self,moves):
        result = (moves[-1] if len(moves) >= 3 and all(d == moves[-1] for d in moves[-3:]) else None)


# ---------------------------------------------------------------------------
# 布局自检：整窗必须放得进屏幕，否则棋盘下沿会被切掉（真实遇到过）。
# 断言放在类定义之后，改尺寸时只要 import 一次就会立刻报出来。
# ---------------------------------------------------------------------------
assert Config.WINDOW_H <= Config._MAX_WINDOW_H, (
    "整窗高度 %d 超过上限 %d：把 CELL_SIZE 调小，或压缩 TOP_BAR_H / BOARD_STRIP_H"
    % (Config.WINDOW_H, Config._MAX_WINDOW_H))
assert Config.WINDOW_W <= Config._MAX_WINDOW_W, (
    "整窗宽度 %d 超过上限 %d：把 CELL_SIZE 或 PANEL_W 调小"
    % (Config.WINDOW_W, Config._MAX_WINDOW_W))
