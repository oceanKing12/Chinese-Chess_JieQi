# renderer.py
"""三区布局渲染（深色卡片式风格）。

    ┌──────────────────────────────────────────┬────────────────┐
    │ 顶部配置区：品牌 / 难度 / 两个大模型开关   │                │
    │            轮到谁 / 手数 / 将军 / AI 执哪方│  右侧信息面板： │
    │            提示消息            快捷键说明  │  状态·评估·操作 │
    ├──────────────────────────────────────────┤  走棋记录      │
    │ 棋盘（下棋区，含坐标、上一步标记、          │  大模型点评    │
    │ 可走点提示、动画与结算覆盖层）             │                │
    └──────────────────────────────────────────┴────────────────┘

实现要点：

1. 所有棋盘绘制代码继续使用「棋盘本地坐标」，靠 :meth:`_on_board` 临时把
   ``self.screen`` 指到一块 WIDTH×HEIGHT 的离屏 Surface 上，画完再整块 blit
   到窗口的棋盘位置。这样棋盘坐标代码一行都不用改。
2. 排版全部走 ``config.py`` 里的间距/圆角令牌，按钮与卡片共用同一套几何
   （``button_at`` 与绘制取的是同一批 Rect），不会出现"画在这里、点在那里"。
3. 面板按「卡片」组织：状态卡 / 评估卡 / 操作卡 / 记录区 / 点评卡，
   每块都有标题、圆角、细边框与投影，视觉上有明确层次。
"""
from __future__ import annotations

import contextlib
import math
import time
from collections import OrderedDict

import pygame

try:  # 兼容两种运行方式（PYTHONPATH 含项目根，或直接 python main.py）
    from Game.game_state import GameState
    from Game.resource_manager import ResourceManager, load_font
    from Game.jieqi.gui_adapter import (
        captured_gui_pieces, gui_color_to_side, legal_moves, move_record_text,
    )
    from Game.start_screen import DIFFICULTY_ITEMS, StartLayout
except ImportError:  # pragma: no cover
    from game_state import GameState
    from resource_manager import ResourceManager, load_font
    from jieqi.gui_adapter import (
        captured_gui_pieces, gui_color_to_side, legal_moves, move_record_text,
    )
    from start_screen import DIFFICULTY_ITEMS, StartLayout

#: 点评卡最多占面板高度的比例（再长就用滚轮翻看，**不再截断文字**）
_COMMENT_MAX_RATIO = 0.45

def _wrap(font, text: str, max_width: int):
    """按字符宽度折行（中文按字折行最自然）。"""
    lines = []
    for raw in str(text).splitlines() or [""]:
        if not raw:
            lines.append("")
            continue
        current = ""
        for ch in raw:
            if font.size(current + ch)[0] <= max_width:
                current += ch
            else:
                lines.append(current)
                current = ch
        lines.append(current)
    return lines


def _fit(font, text: str, max_width: int) -> str:
    """把一行文字裁到 ``max_width`` 内（超长加省略号）。

    ``font.size()`` 只是估计值，中文标点/全角括号的**实际渲染宽度**偶尔会略大，
    所以裁完再用 ``render(...).get_width()`` 复核一遍并逐字回退。
    贴着边界排版时（顶栏第二行右边紧挨着"点评"徽标）差这几像素就会擦上去。
    """
    text = str(text)
    if font.size(text)[0] <= max_width:
        return text
    while text and font.size(text + "…")[0] > max_width:
        text = text[:-1]
    out = text + "…"
    while out and font.render(out, True, (0, 0, 0)).get_width() > max_width:
        out = out[:-2] + "…"
    return out


def _center_offset(box_height: int, content_height: int) -> int:
    """内容在 ``box_height`` 里垂直居中时的顶部偏移（负数按 0 处理）。"""
    return max(0, int((box_height - content_height) / 2))


class Renderer:
    def __init__(self, resource_manager: ResourceManager, game_state: GameState):
        self.rm = resource_manager
        self.gs = game_state
        self.config = self.rm.config
        cfg = self.config
        self.window = pygame.display.set_mode((cfg.WINDOW_W, cfg.WINDOW_H))
        pygame.display.set_caption("揭棋 · 人机对战")
        self.board_surface = pygame.Surface((cfg.WIDTH, cfg.HEIGHT))
        #: 绘制方法的当前目标；棋盘层会临时切到 board_surface
        self.screen = self.window
        self.board_rect = pygame.Rect(cfg.BOARD_X, cfg.BOARD_Y, cfg.WIDTH, cfg.HEIGHT)
        #: 上下的「被吃棋子」展示条（不参与点击命中，纯展示）
        self.capture_top_rect = pygame.Rect(
            cfg.BOARD_X, cfg.TOP_BAR_H, cfg.WIDTH, cfg.BOARD_STRIP_H)
        self.capture_bottom_rect = pygame.Rect(
            cfg.BOARD_X, cfg.BOARD_Y + cfg.HEIGHT, cfg.WIDTH, cfg.BOARD_STRIP_H)
        self.top_rect = pygame.Rect(0, 0, cfg.WINDOW_W, cfg.TOP_BAR_H)
        self.panel_rect = pygame.Rect(cfg.WIDTH, cfg.TOP_BAR_H, cfg.PANEL_W,
                                      cfg.WINDOW_H - cfg.TOP_BAR_H)
        self.mouse_pos = (-1, -1)
        self.difficulty = "normal"
        #: 对局设置（启动页上改，AI 控制器与棋谱保存都从这里读）
        self.llm_on = False
        self.apply_llm_move = False
        self.llm_note = ""
        self.auto_save = True
        self.ai_note = ""
        self.player_side = "red"
        #: 小缓存：可走点 / 折行后的点评，避免每帧重复计算
        self._hint_cache_key = None
        self._hint_cache: set = set()
        self._wrap_cache_key = None
        self._wrap_cache: list = []
        #: 被吃棋子的缓存（按"手数 + 记录长度"作废，避免每帧深拷贝棋盘）
        self._capture_cache_key = None
        self._capture_cache: dict = {"red": [], "black": []}
        #: 当前打开的浮层（``''`` / ``'board'``=回顾棋局）
        self.overlay = ''
        #: 走棋记录的滚动位置（0 = 贴到最新一手，正数 = 往前翻）
        self.log_scroll = 0
        self.log_scroll_max = 0
        self._log_state_key = None
        #: 大模型点评卡的滚动位置（点评很长时用滚轮翻看）
        self.comment_scroll = 0
        #: 启动页布局缓存（设置没变就复用，避免每帧重建）
        self._start_layout = None
        self._start_layout_key = None
        self._record_count_cache = None

    # ================================================================ 布局
    @contextlib.contextmanager
    def _on_board(self):
        prev = self.screen
        self.screen = self.board_surface
        try:
            yield
        finally:
            self.screen = prev

    def draw_all(self):
        """主绘制方法。"""
        if self.gs.show_side_selection:
            self.draw_side_selection()
            return
        self.window.fill(self.config.COLORS['window_bg'])
        # 先画上下两条"被吃棋子"，再把棋盘贴上去（两者区域互不重叠）
        self.draw_capture_strips()
        with self._on_board():
            self.draw_board()
            self._draw_last_move_marker()
            self._draw_move_hints()
            self.draw_pieces()
            self._draw_selection_glow()
            self.draw_animations()
            if self.gs.game_over:
                self.draw_game_over()
        self.window.blit(self.board_surface, self.board_rect.topleft)
        self.draw_top_bar()
        self.draw_info_panel()
        # toast 压在棋盘顶部（顶栏已经没有第三行放它了）；浮层打开时让位给浮层
        if self.overlay != 'board':
            self._draw_toast()
        if self.overlay == 'board':
            self.draw_board_overlay()

    # ------------------------------------------------- 回顾棋局（大棋盘浮层）
    def review_rects(self):
        """浮层里的按钮矩形（窗口坐标）。"""
        cfg = self.config
        w, h = int(cfg.WINDOW_W * 0.88), int(cfg.WINDOW_H * 0.9)
        left = (cfg.WINDOW_W - w) // 2
        top = (cfg.WINDOW_H - h) // 2
        panel = pygame.Rect(left, top, w, h)
        btn_h = 34
        y = panel.bottom - cfg.PAD - btn_h
        nav_w = 96
        gap = 8
        total = nav_w * 4 + gap * 3
        x0 = panel.x + (w - total) // 2
        nav = {}
        for i, cmd in enumerate(('nav:first', 'nav:prev', 'nav:next', 'nav:last')):
            nav[cmd] = pygame.Rect(x0 + i * (nav_w + gap), y, nav_w, btn_h)
        close_w = 110
        close = pygame.Rect(panel.right - cfg.PAD - close_w, y, close_w, btn_h)
        return {'panel': panel, 'nav': nav, 'close': close}

    def _all_buttons(self):
        """当前界面上的所有可点按钮 ``(cmd, rect, text, active)``。

        分两种界面：**启动页**（设置 + 阵营 + 继续/载入）与**棋盘页**
        （面板操作卡 + 翻手）。两套按钮的 Rect 与各自 ``draw_*`` 用的是同一批，
        所以"画在这里、点在那里"不会发生。
        """
        if self.gs.show_side_selection:
            return list(self.start_screen_buttons())
        out = [self.record_button(), self.import_button(), self.replay_button(),
               self.restart_button()]
        out.extend(self.score_buttons())
        out.extend(self.nav_buttons())
        return out

    def button_at(self, pos):
        """命中测试：返回命令字符串（按钮优先于落子）。"""
        if self.gs.show_side_selection:
            return self.start_layout().hit(pos)
        if self.overlay == 'board':
            rects = self.review_rects()
            for cmd, rect in rects['nav'].items():
                if rect.collidepoint(pos):
                    return cmd
            if rects['close'].collidepoint(pos):
                return 'view:close'
            return None
        for cmd, rect, _text, _active in self._all_buttons():
            if rect.collidepoint(pos):
                return cmd
        return None

    def draw_board_overlay(self):
        """「回顾棋局」浮层：大棋盘 + 双方被吃之子 + 翻手按钮。

        刻意**不放走棋记录**——记录一直在右侧面板里，浮层的作用只是"把棋看得更大"，
        配合上一步/下一步复盘。ESC 或「关闭」退出。
        """
        cfg = self.config
        colors = cfg.COLORS
        rects = self.review_rects()
        panel = rects['panel']
        shroud = pygame.Surface((cfg.WINDOW_W, cfg.WINDOW_H), pygame.SRCALPHA)
        shroud.fill((0, 0, 0, 180))
        self.window.blit(shroud, (0, 0))

        sheet = pygame.Surface((panel.width, panel.height))
        sheet.fill(colors['panel_bg'])
        pygame.draw.rect(sheet, colors['panel_line'], sheet.get_rect(), 2,
                         border_radius=cfg.RADIUS)
        local = lambda r: r.move(-panel.x, -panel.y)  # noqa: E731

        title = load_font(cfg.FONT_TITLE).render("回顾棋局", True, colors['text_main'])
        sheet.blit(title, (cfg.PAD_L, cfg.PAD))
        sub = ("第 %d/%d 手　%s" % (getattr(self.gs, 'ply', 0),
                                   getattr(self.gs, 'total_plies', 0),
                                   "浏览历史（点「最新」回到当前）"
                                   if self.gs.is_browsing else "当前局面"))
        font_small = load_font(cfg.FONT_SMALL)
        sheet.blit(font_small.render(sub, True, colors['text_dim']),
                   (cfg.PAD_L, cfg.PAD + title.get_height() + 4))

        # 棋盘：尽量占满（先按可用高度算宽度，再用面板宽封顶）。
        # 底部要留出：按钮行 34 + 间隙 + 被吃之子两行 + 下边距。
        lost_h = (font_small.get_linesize() + 2) * 2
        foot = 34 + cfg.PAD + lost_h + cfg.PAD
        board_top = cfg.PAD + title.get_height() + font_small.get_linesize() + cfg.PAD
        avail_h = max(80, panel.height - board_top - foot)
        board_w = min(int(panel.width - cfg.PAD_L * 2),
                      int(avail_h * cfg.WIDTH / cfg.HEIGHT))
        board_h = int(board_w * cfg.HEIGHT / cfg.WIDTH)
        board_left = (panel.width - board_w) // 2
        board_top += max(0, (avail_h - board_h) // 2)
        scaled = pygame.transform.smoothscale(self.board_surface, (board_w, board_h))
        board_rect = pygame.Rect(board_left, board_top, board_w, board_h)
        sheet.blit(scaled, board_rect)
        # 缩放比例：浮层里的棋盘是缩放过的，一切叠加标记都要按它换算。
        sx, sy = board_w / float(cfg.WIDTH), board_h / float(cfg.HEIGHT)
        # 复盘的"上一步四角标记"也要画在**放大后**的位置上（棋盘页那份是给
        # 原尺寸棋盘画的），否则缩放之后标记会落在错误的格子上。
        squares = self._last_move_squares()
        if squares:
            for (r, c) in squares:
                dr, dc = self.gs._convert_coords(r, c)
                self._draw_corner_marks(
                    dr, dc, cfg.COLORS['last_move'], surface=sheet,
                    origin=board_rect.topleft, scale=(sx, sy))

        # 被吃之子：棋盘下方两行纯文字（浮层里不重复画图标，省地方也更清楚），
        # 用 _fit 裁到面板宽度内，长棋谱也不会把字画到面板外面去。
        lost_font = load_font(cfg.FONT_TINY)
        lost_y = max(board_top + board_h + cfg.PAD_XS,
                     panel.height - 34 - cfg.PAD - lost_h)
        captured = self._captured_lists()
        for side, label in (('black', "黑方被吃"), ('red', "红方被吃")):
            items = captured.get(side) or []
            text = _fit(lost_font, "%s：%s" % (label, self._capture_summary(items)),
                        panel.width - cfg.PAD_L * 2)
            sheet.blit(lost_font.render(text, True, colors['text_dim']),
                       (cfg.PAD_L, lost_y))
            lost_y += lost_font.get_linesize() + 2

        # 底部按钮：**必须画在 sheet 上**再整块贴出去（画到窗口上会被 sheet 盖住）
        saved = self.window
        self.window = sheet
        try:
            for cmd, label in (('nav:first', '开局'), ('nav:prev', '上一回合'),
                               ('nav:next', '下一回合'), ('nav:last', '最新')):
                enabled = (self.gs.ply > 0 if cmd in ('nav:first', 'nav:prev')
                           else self.gs.ply < self.gs.total_plies)
                self._draw_button(local(rects['nav'][cmd]), label, cfg.FONT_SMALL,
                                  enabled=enabled)
            self._draw_button(local(rects['close']), "关闭（ESC）", cfg.FONT_SMALL,
                              enabled=True)
        finally:
            self.window = saved
        self.window.blit(sheet, panel.topleft)

    @staticmethod
    def _capture_summary(items) -> str:
        """被吃之子的一句话摘要（明子列名字，暗子只数数量）。"""
        names = []
        hidden = 0
        for item in items or ():
            piece = item.get('piece') if isinstance(item, dict) else None
            if item.get('unknown') or not isinstance(piece, dict):
                hidden += 1
                continue
            color = "红" if str(piece.get('color')) == 'red' else "黑"
            names.append(color + str(piece.get('actual') or '?'))
        bits = []
        if names:
            bits.append("、".join(names[:14]) + ("…" if len(names) > 14 else ""))
        if hidden:
            bits.append("%d 枚暗子（身份未知）" % hidden)
        return "　".join(bits) if bits else "无"

    # ------------------------------------------------- 被吃棋子展示条
    def _captured_lists(self):
        """``{'red': [...], 'black': [...]}`` = 每一方**被吃**的棋子（带缓存）。

        每项是 ``{"piece": dict|None, "unknown": bool}``：明子给真身（照原样显示），
        被吃的暗子按规则身份未知，只给 ``unknown=True``，界面画"暗子"图标。
        """
        gs = self.gs
        key = (int(getattr(gs, 'ply', 0) or 0),
               len(getattr(gs, 'move_history', None) or ()))
        if key != self._capture_cache_key:
            self._capture_cache_key = key
            try:
                self._capture_cache = {
                    'red': captured_gui_pieces(gs, 'red'),
                    'black': captured_gui_pieces(gs, 'black'),
                }
            except Exception:
                self._capture_cache = {'red': [], 'black': []}
        return self._capture_cache

    def draw_capture_strips(self):
        """棋盘上下的被吃棋子条（**上=我方被吃的，下=对方被吃的**）。

        为什么是"上=我方"：棋盘是"我方在下方"的视角，所以**靠近自己的那一条
        （下面）摆自己吃到的战利品**，离自己远的那一条（上面）摆自己损失掉的子
        ——这与中国象棋软件的惯例一致。旧版正好反过来（上=对方被吃），
        用户看的时候就会觉得"下面明明是我吃的子，怎么写成我方被吃"。

        * 明子直接显示真身图标，被吃的暗子只显示"暗子"背面图标——
          规则上被吃的暗子双方都不得翻看，界面也不能画出来；
        * **同种棋子合并成一枚图标 + 右上角数量角标**（如「卒2」= 两个卒被吃），
          所以一行就能排完，条高只有 CELL 的一半；
        * 每条靠外缘画一道我方/对方的颜色，再配上同色的「我方被吃 / 对方被吃」
          标签，红黑一眼可辨（只靠图标颜色是不够的，暗子图标两边都是深色）。
        """
        cfg = self.config
        colors = cfg.COLORS
        mine = 'red' if getattr(self.gs, 'player_side', 'red') == 'red' else 'black'
        theirs = 'black' if mine == 'red' else 'red'
        captured = self._captured_lists()

        self.screen = self.window
        # 上方 = 我方被吃（对方吃到的子），靠右排；下方 = 对方被吃（我吃到的子），
        # 靠左排。标签都在另一侧，与棋盘中央离得最远。
        self._draw_one_capture_strip(
            self.capture_top_rect, captured[mine], "我方被吃", label_side='left',
            accent=colors['strip_red' if mine == 'red' else 'strip_black'])
        self._draw_one_capture_strip(
            self.capture_bottom_rect, captured[theirs], "对方被吃",
            label_side='right',
            accent=colors['strip_red' if theirs == 'red' else 'strip_black'])
        # 与棋盘之间的细分隔线，明确"这不是棋盘的一部分"
        for y in (self.capture_top_rect.bottom - 3,
                  self.capture_bottom_rect.top + 1):
            pygame.draw.line(self.window, colors['strip_line'],
                             (self.capture_top_rect.left + cfg.PAD, y),
                             (self.capture_top_rect.right - cfg.PAD, y), 2)

    #: 同种棋子归并后的排序权重：大子在前，暗子垫底（未知身份，别抢眼）
    _CAPTURE_ORDER = ('车', '車', '炮', '砲', '马', '馬', '相', '象',
                      '仕', '士', '兵', '卒')

    @classmethod
    def _capture_groups(cls, items):
        """把被吃棋子按"同种"归并成 ``[(key, 图标项, 数量), …]``。

        ``key`` 用 ``(颜色, 兵种或 '暗')``：明子按真身归并，暗子统一归并成一组
        （因为它们的身份本来就不可知，界面上也只能画同一个背面）。
        排序：先按兵种权重（大子在前），再按数量多的在前——这样展示条的顺序
        是稳定的，不会因为吃子顺序不同而跳来跳去。
        """
        groups: "OrderedDict[tuple, list]" = OrderedDict()
        for item in items or ():
            piece = item.get('piece') if isinstance(item, dict) else None
            if item.get('unknown') or not isinstance(piece, dict):
                color = 'black'
                if isinstance(piece, dict) and piece.get('color'):
                    color = 'red' if str(piece['color']) == 'red' else 'black'
                elif isinstance(item, dict) and item.get('color'):
                    color = ('red' if str(item['color']) == 'red' else 'black')
                key = (color, '暗')
            else:
                color = 'red' if str(piece.get('color')) == 'red' else 'black'
                key = (color, str(piece.get('actual') or '暗'))
            groups.setdefault(key, []).append(item)

        def sort_key(entry):
            key, bucket = entry
            ident = key[1]
            rank = (cls._CAPTURE_ORDER.index(ident)
                    if ident in cls._CAPTURE_ORDER else len(cls._CAPTURE_ORDER))
            return (rank, -len(bucket), key)

        return [(key, bucket[0], len(bucket))
                for key, bucket in sorted(groups.items(), key=sort_key)]

    def _draw_one_capture_strip(self, rect, items, label, label_side='left',
                                accent=None):
        """画一条被吃棋子展示条（一行图标，同种带数量角标）。

        布局约定（**这是"标签永远不会被棋子盖住"的关键**）：

        * 图标排在条的一侧，标签排在**另一侧**（上方条：图标靠右、标签靠左；
          下方条：图标靠左、标签靠右）；
        * 能放几枚按"条宽 - 左右内边距 - 标签预留宽"算，放不下的只在标签里
          报数量，绝不会画到条外面去。

        ``items`` 里的每一项都是 ``{"piece": dict|None, "unknown": bool}``；
        ``accent`` 是这条所属一方的颜色（外缘色条 + 标签字色）。
        """
        cfg = self.config
        colors = cfg.COLORS
        pygame.draw.rect(self.window, colors['strip_bg'], rect)
        if accent is not None:
            # 外缘一条 4px 的方色：上条画在顶边、下条画在底边（都是"外侧"）
            edge = pygame.Rect(rect.x, rect.y if label_side == 'left'
                               else rect.bottom - 3, rect.width, 3)
            pygame.draw.rect(self.window, accent, edge)
        icon = cfg.STRIP_ICON
        step = icon + cfg.STRIP_GAP
        label_w = cfg.STRIP_LABEL_W
        grid_w = max(step, rect.width - cfg.PAD * 2 - label_w)
        per_row = max(1, grid_w // step)
        y = rect.y + (rect.height - icon) // 2

        groups = self._capture_groups(items)
        shown = groups[:per_row]
        hidden = max(0, len(groups) - len(shown))
        # 图标区靠哪一侧：标签在左则图标靠右，反之靠左。
        if label_side == 'left':
            grid_left = rect.right - cfg.PAD - len(shown) * step + cfg.STRIP_GAP
        else:
            grid_left = rect.x + cfg.PAD
        for index, (key, sample, count) in enumerate(shown):
            image = self._captured_image(key)
            if image is None:
                continue
            x = grid_left + index * step
            self.window.blit(image, (x, y))
            if count > 1:
                self._draw_count_badge(x + icon, y, count)

        total = len(items) if items else 0
        text = label
        if total:
            text = "%s %d 枚" % (label, total)
            if hidden > 0:
                text += "（另 %d 种未画出）" % hidden
        font = load_font(cfg.FONT_TINY)
        surf = font.render(text, True, accent or colors['text_faint'])
        if label_side == 'left':
            x = rect.x + cfg.PAD
        else:
            x = max(rect.x + cfg.PAD, rect.right - cfg.PAD - surf.get_width())
        self.window.blit(surf, (x, rect.y + (rect.height - surf.get_height()) // 2))

    def _draw_count_badge(self, right_x, top_y, count):
        """在图标右上角画一个数量角标（``卒2`` = 两个卒被吃）。

        位置特意压在图标**右上角外沿**：角标小、颜色亮，不遮棋子本身的脸。
        """
        cfg = self.config
        colors = cfg.COLORS
        text = str(count)
        font = load_font(cfg.FONT_TINY)
        w = max(16, font.size(text)[0] + 8)
        h = 16
        rect = pygame.Rect(right_x - w + 4, top_y - 5, w, h)
        pygame.draw.rect(self.window, colors['card_shadow'], rect.inflate(2, 2),
                         border_radius=8)
        pygame.draw.rect(self.window, colors['btn_bg_active'], rect,
                         border_radius=8)
        surf = font.render(text, True, colors['bar_bg'])
        self.window.blit(surf, surf.get_rect(center=rect.center))

    def _captured_image(self, key_or_item):
        """被吃棋子 -> 要贴的图标。

        参数可以是 ``(颜色, 兵种)`` 元组（归并后的分组键），也可以是单项
        ``{"piece":…, "unknown":…}``（向后兼容单测里的旧调用）。

        明子用 ``rm.piece_images[(color, 实际身份)]``（与棋盘上同一套图标）；
        被吃的暗子用暗子背面图标 ``(color, '暗')``，绝不用真身。
        """
        cfg = self.config
        if isinstance(key_or_item, tuple) and len(key_or_item) == 2:
            color, ident = key_or_item
        else:
            item = key_or_item
            if not isinstance(item, dict):
                return None
            piece = item.get('piece')
            color = 'black'
            if isinstance(piece, dict):
                color = 'red' if str(piece.get('color')) == 'red' else 'black'
            if item.get('unknown') or not isinstance(piece, dict):
                ident = '暗'
            else:
                ident = str(piece.get('actual') or '暗')
        image = self.rm.piece_images.get((color, ident))
        if image is None:
            image = self.rm.piece_images.get((color, '暗'))
        if image is None:
            return None
        if image.get_width() != cfg.STRIP_ICON:
            image = pygame.transform.smoothscale(image, (cfg.STRIP_ICON,
                                                         cfg.STRIP_ICON))
        return image

    # ------------------------------------------------------------ 棋盘层
    def draw_board(self):
        """绘制棋盘基础（木纹底 + 网格 + 九宫 + 楚河汉界 + 坐标）。"""
        self.screen.blit(self.rm.background, (0, 0))
        self._draw_grid_lines()
        self._draw_palace_diagonals()
        self._draw_river_text()
        self._draw_board_coords()

    def _line_x(self, col: float) -> float:
        return self.config.OFF_SET + (col + 0.5) * self.config.CELL_SIZE

    def _line_y(self, row: float) -> float:
        return self.config.OFF_SET + (row + 0.5) * self.config.CELL_SIZE

    def _draw_grid_lines(self):
        """绘制棋盘网格线。"""
        cfg = self.config
        grid = cfg.COLORS['grid']
        x0, x1 = self._line_x(0), self._line_x(8)
        for i in range(10):
            y = self._line_y(i)
            pygame.draw.line(self.screen, grid, (x0, y), (x1, y),
                             2 if i in (0, 9) else 1)
        y_top, y_mid_hi = self._line_y(0), self._line_y(4)
        y_mid_lo, y_bot = self._line_y(5), self._line_y(9)
        for i in range(9):
            x = self._line_x(i)
            thickness = 2 if i in (0, 8) else 1
            pygame.draw.line(self.screen, grid, (x, y_top), (x, y_mid_hi), thickness)
            pygame.draw.line(self.screen, grid, (x, y_mid_lo), (x, y_bot), thickness)
            if i in (0, 8):
                # 最左/最右两条竖线在河界处连通，其余留空
                pygame.draw.line(self.screen, grid, (x, y_mid_hi), (x, y_mid_lo),
                                 thickness)

    def _draw_palace_diagonals(self):
        """绘制九宫格斜线。"""
        cfg = self.config
        for top in (True, False):
            if top:
                r0, r1 = 0, 2
            else:
                r0, r1 = 7, 9
            c0, c1 = 3, 5
            pygame.draw.line(self.screen, cfg.COLORS['grid'],
                             (self._line_x(c0), self._line_y(r0)),
                             (self._line_x(c1), self._line_y(r1)), 1)
            pygame.draw.line(self.screen, cfg.COLORS['grid'],
                             (self._line_x(c1), self._line_y(r0)),
                             (self._line_x(c0), self._line_y(r1)), 1)

    def _draw_river_text(self):
        """楚河汉界：分别居中在河界左右两半，比原来一行到底更规整。"""
        cfg = self.config
        font = load_font(cfg.FONT_BRAND)
        cy = int(self._line_y(4) + (self._line_y(5) - self._line_y(4)) / 2)
        for text, cx in (("楚 河", self.config.WIDTH * 0.28),
                         ("漢 界", self.config.WIDTH * 0.72)):
            surf = font.render(text, True, cfg.COLORS['river_text'])
            self.screen.blit(surf, surf.get_rect(center=(int(cx), cy)))

    def _draw_board_coords(self):
        """棋盘外围的坐标：**列用 a~i、行用 0~9**，四边都标一份。

        为什么四边都标：走棋记录与棋谱都用 ``b7b5`` 这套写法（列字母 + 行号，
        红方底线是第 0 行），玩家要能**直接照着读**。四边同标的好处是不管
        现在红方在下还是黑方在下，抬头看一眼就有对应的行列号，不用心算翻转。

        口径与 :mod:`jieqi.notation` 一致（``rank = 9 - row``）；显示位置则跟
        玩家视角走（黑方视角整盘 180° 翻转，见 ``_convert_coords``）。
        """
        from jieqi.notation import FILE_LETTERS

        cfg = self.config
        font = load_font(cfg.FONT_TINY)
        color = cfg.COLORS['coord_text']
        top_y = cfg.OFF_SET // 2
        bot_y = cfg.HEIGHT - cfg.OFF_SET // 2
        left_x = cfg.OFF_SET // 2
        right_x = cfg.WIDTH - cfg.OFF_SET // 2
        for row in range(10):
            display_r, _ = self.gs._convert_coords(row, 0)
            rank = 9 - row                       # 红方底线（内部 row 9）是第 0 行
            y = self._line_y(display_r)
            for x in (left_x, right_x):
                surf = font.render(str(rank), True, color)
                self.screen.blit(surf, surf.get_rect(center=(int(x), int(y))))
        for col in range(9):
            _, display_c = self.gs._convert_coords(0, col)
            letter = FILE_LETTERS[col]
            x = self._line_x(display_c)
            for y in (top_y, bot_y):
                surf = font.render(letter, True, color)
                self.screen.blit(surf, surf.get_rect(center=(int(x), int(y))))

    def draw_pieces(self):
        """绘制所有棋子。"""
        animated_positions = {a['end_pos'] for a in self.gs.animations}
        for r in range(10):
            for c in range(9):
                if (r, c) not in animated_positions and self.gs.board[r][c]:
                    self._draw_piece(r, c, self.gs.board[r][c])

    #: 兼容旧调用名
    def _draw_pieces_local(self):
        self.draw_pieces()

    def _piece_center(self, logic_r, logic_c):
        display_r, display_c = self.gs._convert_coords(logic_r, logic_c)
        return (self.config.OFF_SET + display_c * self.config.CELL_SIZE
                + self.config.CELL_SIZE // 2,
                self.config.OFF_SET + display_r * self.config.CELL_SIZE
                + self.config.CELL_SIZE // 2)

    def _display_center(self, logic_r, logic_c):
        """逻辑坐标 -> 棋盘 Surface 上的显示中心（不取整，供动画插值用）。"""
        display_r, display_c = self.gs._convert_coords(logic_r, logic_c)
        return (self.config.OFF_SET + (display_c + 0.5) * self.config.CELL_SIZE,
                self.config.OFF_SET + (display_r + 0.5) * self.config.CELL_SIZE)

    def _blit_piece_image(self, piece_data, center):
        if piece_data['revealed']:
            img_key = (piece_data['color'], piece_data['actual'])
        else:
            img_key = (piece_data['color'], '暗')
        image = self.rm.piece_images.get(img_key)
        if image:
            self.screen.blit(image, image.get_rect(center=center))
            return
        text = piece_data['actual'] if piece_data['revealed'] else '暗'
        color = (self.config.COLORS['text_red'] if piece_data['color'] == 'red'
                 else self.config.COLORS['text_black'])
        surface = self.rm.font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=center))

    def _draw_piece(self, logic_r: int, logic_c: int, piece_data: dict):
        center = self._piece_center(logic_r, logic_c)
        radius = self.config.CELL_SIZE // 2 - 3
        shadow = pygame.Surface((radius * 2 + 8, radius * 2 + 8), pygame.SRCALPHA)
        pygame.draw.circle(shadow, (0, 0, 0, 90),
                           (radius + 4, radius + 5), radius)
        self.screen.blit(shadow, (center[0] - radius - 4, center[1] - radius - 4))
        self._blit_piece_image(piece_data, center)

    def draw_animations(self):
        current_time = pygame.time.get_ticks()
        for anim in self.gs.animations[:]:
            elapsed = current_time - anim['start_time']
            progress = min(elapsed / anim['duration'], 1.0)
            if anim['type'] == 'move':
                sr, sc = anim['start_pos']
                er, ec = anim['end_pos']
                center = self._display_center(sr + (er - sr) * progress,
                                              sc + (ec - sc) * progress)
                center = (int(center[0]), int(center[1]))
                flip_progress = min(progress * 2, 1.0) if anim['flip'] else 0.0
                if anim['flip'] and flip_progress < 1.0:
                    self._draw_flipping_piece(anim['piece'], center, flip_progress)
                else:
                    self._blit_piece_image(anim['piece'], center)
                if progress >= 1.0:
                    self.gs.animations.remove(anim)

    def _draw_piece_image(self, piece_data, center):
        """兼容旧调用名。"""
        self._blit_piece_image(piece_data, center)

    def _draw_flipping_piece(self, piece_data, center, progress):
        angle = 180 * progress
        img_key = ((piece_data['color'], '暗') if progress < 0.5
                   else (piece_data['color'], piece_data['actual']))
        image = self.rm.piece_images.get(img_key)
        if image:
            rotated = pygame.transform.rotate(image, angle)
            self.screen.blit(rotated, rotated.get_rect(center=center))
            return
        text = '暗' if progress < 0.5 else piece_data['actual']
        color = (self.config.COLORS['text_red'] if piece_data['color'] == 'red'
                 else self.config.COLORS['text_black'])
        surface = self.rm.font.render(text, True, color)
        scale = 1 - abs(progress - 0.5)
        scaled = pygame.transform.scale(
            surface, (max(1, int(surface.get_width() * scale)),
                      max(1, surface.get_height())))
        self.screen.blit(scaled, scaled.get_rect(center=center))

    # ---------------------------------------------------- 棋盘上的提示层
    def _last_move_squares(self):
        """当前显示的那一手是哪两个格子（用于高亮）。"""
        ply = int(getattr(self.gs, 'ply', 0) or 0)
        hist = getattr(self.gs, 'move_history', None) or []
        if ply <= 0 or ply > len(hist):
            return None
        rec = hist[ply - 1]
        start, end = rec.get('start'), rec.get('end')
        if start and end:
            return start, end
        return None

    def _draw_corner_marks(self, display_r, display_c, color, surface=None,
                           origin=(0, 0), scale=(1.0, 1.0), width=3):
        """在格子的四个角画短直角括号（比整块高亮更轻、不挡棋子）。

        ``surface`` / ``origin`` / ``scale`` 是给「回顾棋局」浮层用的：那里
        棋盘被缩放后贴到另一块表面上，标记也得按同样比例换算，否则会画到
        错位的格子上（棋盘页那份是原尺寸，用默认值即可）。
        """
        cfg = self.config
        cell = cfg.CELL_SIZE * float(scale[0])
        x = origin[0] + (cfg.OFF_SET + display_c * cfg.CELL_SIZE) * scale[0]
        y = origin[1] + (cfg.OFF_SET + display_r * cfg.CELL_SIZE) * scale[1]
        target = surface if surface is not None else self.screen
        inset = 4
        arm = 13 * float(scale[0])
        line_w = max(2, int(round(width * float(scale[0]))))
        for cx, cy, dx, dy in (
                (x + inset, y + inset, 1, 1),
                (x + cell - inset, y + inset, -1, 1),
                (x + inset, y + cell - inset, 1, -1),
                (x + cell - inset, y + cell - inset, -1, -1)):
            pygame.draw.line(target, color, (cx, cy), (cx + dx * arm, cy), line_w)
            pygame.draw.line(target, color, (cx, cy), (cx, cy + dy * arm), line_w)

    def _draw_last_move_marker(self):
        squares = self._last_move_squares()
        if not squares:
            return
        color = self.config.COLORS['last_move']
        for (r, c) in squares:
            dr, dc = self.gs._convert_coords(r, c)
            self._draw_corner_marks(dr, dc, color)

    def _legal_targets(self):
        """选中的棋子能走到哪些格（带缓存，避免每帧重算）。"""
        sel = getattr(self.gs, 'selected', None)
        if not sel:
            return set()
        key = (tuple(sel), int(getattr(self.gs, 'ply', 0) or 0),
               getattr(self.gs, 'current_player', 'red'))
        if key == self._hint_cache_key:
            return self._hint_cache
        targets = set()
        try:
            side = gui_color_to_side(self.gs.current_player)
            frm = sel[0] * 9 + sel[1]
            for mv in legal_moves(self.gs.board, side):
                if (mv >> 7) == frm:
                    to = mv & 127
                    targets.add((to // 9, to % 9))
        except Exception:
            targets = set()
        self._hint_cache_key = key
        self._hint_cache = targets
        return targets

    def _draw_move_hints(self):
        """选中棋子后，在可落点上画小圆点（吃子画空心圈）。"""
        targets = self._legal_targets()
        if not targets:
            return
        cfg = self.config
        cell = cfg.CELL_SIZE
        for (r, c) in targets:
            cx, cy = self._piece_center(r, c)
            cx, cy = int(cx), int(cy)
            occupied = self.gs.board[r][c] is not None
            if occupied:
                pygame.draw.circle(self.screen, cfg.COLORS['hint_capture'],
                                   (cx, cy), cell // 2 - 4, 3)
            else:
                pygame.draw.circle(self.screen, cfg.COLORS['hint_dot'],
                                   (cx, cy), 7)

    def _draw_selection_glow(self):
        """选中棋子的流光边框（四角旋转的渐隐描边）。"""
        if not self.gs.selected:
            return
        logic_r, logic_c = self.gs.selected
        display_r, display_c = self.gs._convert_coords(logic_r, logic_c)
        cell = self.config.CELL_SIZE
        x = self.config.OFF_SET + display_c * cell
        y = self.config.OFF_SET + display_r * cell
        phase = pygame.time.get_ticks() % 1600 / 1600
        alpha = int(170 + 85 * math.sin(phase * 2 * math.pi))
        glow = pygame.Surface((cell, cell), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*self.config.COLORS['selection_primary'], alpha),
                         glow.get_rect(), 3, border_radius=8)
        self.screen.blit(glow, (x, y))
        self._draw_corner_marks(display_r, display_c,
                                self.config.COLORS['selection_secondary'])

    # ============================================================ 绘制原语
    def _card(self, surface, rect: pygame.Rect, title=None):
        """画一张卡片（投影 + 圆角 + 细边框 + 可选标题），返回内容区 Rect。"""
        cfg = self.config
        pygame.draw.rect(surface, cfg.COLORS['card_shadow'], rect.move(0, 2),
                         border_radius=cfg.RADIUS)
        pygame.draw.rect(surface, cfg.COLORS['panel_card'], rect,
                         border_radius=cfg.RADIUS)
        pygame.draw.rect(surface, cfg.COLORS['panel_line_soft'], rect, 1,
                         border_radius=cfg.RADIUS)
        y = rect.y + cfg.PAD_S
        if title:
            font = load_font(cfg.FONT_TINY)
            surf = font.render(title, True, cfg.COLORS['text_dim'])
            surface.blit(surf, (rect.x + cfg.PAD, y))
            y += surf.get_height() + 4
        return pygame.Rect(rect.x + cfg.PAD, y,
                           rect.width - 2 * cfg.PAD, rect.bottom - cfg.PAD_S - y)

    def _text(self, surface, text, pos, font_size, color, bold=False):
        font = load_font(font_size)
        surf = font.render(str(text), True, color)
        surface.blit(surf, pos)
        return surf.get_width()

    def _draw_button(self, rect, text, font_size, active=False, enabled=True):
        """通用按钮：未选中态描边、选中态实心强调色、禁用态压暗。"""
        cfg = self.config
        hover = enabled and rect.collidepoint(self.mouse_pos)
        if not enabled:
            bg, fg, border = (cfg.COLORS['btn_disabled'], cfg.COLORS['text_faint'],
                              cfg.COLORS['panel_line_soft'])
        elif active:
            bg, fg, border = (cfg.COLORS['btn_bg_active'],
                              cfg.COLORS['btn_text_active'],
                              cfg.COLORS['btn_bg_active'])
        elif hover:
            bg, fg, border = (cfg.COLORS['btn_bg_hover'], cfg.COLORS['btn_text'],
                              cfg.COLORS['accent_soft'])
        else:
            bg, fg, border = (cfg.COLORS['btn_bg'], cfg.COLORS['btn_text'],
                              cfg.COLORS['btn_border'])
        pygame.draw.rect(self.window, bg, rect, border_radius=cfg.RADIUS_S)
        pygame.draw.rect(self.window, border, rect, 1, border_radius=cfg.RADIUS_S)
        font = load_font(font_size)
        surface = font.render(str(text), True, fg)
        self.window.blit(surface, surface.get_rect(center=rect.center))

    def _draw_pill(self, surface, rect, text, font_size, bg, fg, border=None):
        """小圆角标签（状态徽标）。"""
        cfg = self.config
        pygame.draw.rect(surface, bg, rect, border_radius=rect.height // 2)
        if border:
            pygame.draw.rect(surface, border, rect, 1, border_radius=rect.height // 2)
        font = load_font(font_size)
        surf = font.render(str(text), True, fg)
        surface.blit(surf, surf.get_rect(center=rect.center))

    # ========================================================== 顶部配置区
    def _difficulty_start_x(self) -> int:
        cfg = self.config
        brand_w = load_font(cfg.FONT_BRAND).size("揭棋")[0]
        return cfg.PAD_L + 4 + cfg.PAD_S + brand_w + cfg.PAD_XL

    @property
    def ai_info_rect(self) -> pygame.Rect:
        """第一行，品牌右边留给"AI 用时/深度"的矩形。

        难度与大模型开关都搬到**启动页**之后，这一行整条都空出来了；
        右边界与第二行的「AI 评估」**对齐到同一条竖线**（见 ``ai_eval_rect``），
        这样两行右边齐平，看起来是排过版的。
        """
        cfg = self.config
        left = self._difficulty_start_x()
        # 右边与第二行的「AI 评估」对齐到同一条竖线（见 ai_eval_rect），
        # 两行右边界齐平才像"排过版"；「点评」徽标本来就在更右边，不会撞。
        right = self._content_right()
        return pygame.Rect(left, cfg.TOP_ROW1_Y, max(0, right - left),
                           cfg.TOP_ROW1_H)

    def _content_right(self) -> int:
        """顶栏两行共用的右边界（= 第一枚翻手按钮左边再留一个间距）。"""
        cfg = self.config
        nav = self.nav_buttons()
        if nav:
            return nav[0][1].x - cfg.PAD
        return cfg.WINDOW_W - cfg.PAD_L

    #: 第二行"状态区"（徽标 + 手数 + toast）的总宽。
    #: 这一段 + toast + AI 评估 + 四个翻手按钮要一起塞进 980px 的窗口，
    #: 所以这几个数是**一起定的**（改一个就要看另外两个）。
    STATUS_AREA_W = 150
    #: toast 在状态区**内部**占的那一段（状态文字用它左边的部分）
    TOAST_W = 62

    @property
    def status_text_max_w(self) -> int:
        """状态文字（"第 12/34 手 · AI 执黑"）能占的最大宽度。

        它右边紧挨着 toast 的落点，两者必须在**同一个数**上对齐——分开写死
        两组数字的话，改一处忘一处就会让状态文字压到 toast 上。
        """
        return max(120, self.STATUS_AREA_W - self.TOAST_W)

    def _status_area_right(self) -> int:
        """第二行左侧"状态区"的右边界（徽标 + 手数 + toast 的可用宽度）。

        徽标宽度是**量出来的**：写死"轮到 红方"四个字的宽度，一旦文案改成
        "红方回合"或者字号变了，算出来的矩形就会与实际画的字错开。

        ``STATUS_AREA_W`` 取 150 而不是 260：第二行右边还有 AI 评估和四个翻手
        按钮，留给状态区的越宽、评估被截得越狠（截到"AI 评估：+83（红方稍优）"
        只剩前几个字就没意义了）。状态文字本身也会被 ``_fit`` 截断，
        所以宁可两边都刚好够用。
        """
        cfg = self.config
        font = load_font(cfg.FONT_SMALL)
        badge = font.size("轮到 红方")[0] + 24
        return (cfg.PAD_L + badge + cfg.PAD + self.STATUS_AREA_W + cfg.PAD_S)

    @property
    def ai_eval_rect(self) -> pygame.Rect:
        """第二行，"上一回合"按钮左边留给"AI 评估"的矩形。"""
        cfg = self.config
        left = self._status_area_right() + cfg.PAD_S
        right = self._content_right()
        return pygame.Rect(left, cfg.TOP_ROW2_Y, max(0, right - left),
                           cfg.TOP_ROW2_H)

    @property
    def toast_rect(self) -> pygame.Rect:
        """toast 提示的落点：状态文字与"AI 评估"之间的空档。

        **必须留出宽度**：旧版这里算出来是 0px（状态区把整段吃光了），
        于是 AI 思考中/将军这类提示在顶栏上根本看不见。
        """
        cfg = self.config
        left = self._status_area_right() - self.TOAST_W
        return pygame.Rect(left, cfg.TOP_ROW2_Y,
                           max(0, self.ai_eval_rect.x - cfg.PAD_S - left),
                           cfg.TOP_ROW2_H)

    @property
    def llm_badge_rect(self) -> pygame.Rect | None:
        """第一行最右端的"点评"小徽标（纯展示，开关在启动页）。

        放在**第一行**而不是第二行：第二行右边已经被「AI 评估」和四个翻手按钮
        占满，硬塞一个小徽标进去就会擦到评估文字（真实截图里撞上过）。
        第一行右边本来就只有 AI 用时/深度，空得很。

        没开点评时返回 ``None``：不开就不显示（"默认开启ai点评关了，
        后续继续保持"——界面也不该老提它）。
        """
        cfg = self.config
        if not self.llm_on:
            return None
        font = load_font(cfg.FONT_TINY)
        label = "点评"
        w = font.size(label)[0] + 20
        x = cfg.WINDOW_W - cfg.PAD_L - w
        return pygame.Rect(x, cfg.TOP_ROW1_Y + 4, w, cfg.TOP_ROW1_H - 8)

    def draw_top_bar(self):
        """顶部配置区：第一行品牌 + AI 走棋信息；第二行对局状态 + 评估 + 翻手。

        难度与大模型开关已经搬到**启动页**（用户需求："一些配置选项可以放在
        这里…棋盘页也可以精简一点"），顶栏因此从"挤满按钮"变成"只报对局状态"。
        翻手按钮（开局/上一回合/下一回合/最新）从右侧面板移到第二行右侧：
        它们是每局都要按的，放在顶栏比缩在面板里的小按钮好按得多。
        """
        cfg = self.config
        colors = cfg.COLORS
        bar = pygame.Surface((cfg.WINDOW_W, cfg.TOP_BAR_H))
        bar.fill(colors['bar_bg'])
        # 底部一条强调色分隔线，让配置区和棋盘分开
        pygame.draw.line(bar, colors['bar_bg_alt'],
                         (0, cfg.TOP_BAR_H - 6), (cfg.WINDOW_W, cfg.TOP_BAR_H - 6), 6)
        pygame.draw.line(bar, colors['accent_soft'],
                         (0, cfg.TOP_BAR_H - 3), (cfg.WINDOW_W, cfg.TOP_BAR_H - 3), 2)
        self.window.blit(bar, (0, 0))

        # ---- 第一行：品牌 + AI 走棋信息（用时/深度）----
        x = cfg.PAD_L
        pygame.draw.rect(self.window, colors['accent'],
                         pygame.Rect(x, cfg.TOP_ROW1_Y + 4, 4, cfg.TOP_ROW1_H - 8),
                         border_radius=2)
        x += 4 + cfg.PAD_S
        brand_font = load_font(cfg.FONT_BRAND)
        surf = brand_font.render("揭棋", True, colors['text_main'])
        self.window.blit(surf, (x, cfg.TOP_ROW1_Y + (cfg.TOP_ROW1_H - surf.get_height()) // 2))
        self._draw_ai_info()

        # ---- 第二行：状态 + AI 评估 + toast + 翻手 ----
        self._draw_status_line()
        self._draw_ai_eval()
        self._draw_toast()
        self._draw_llm_badge()
        for cmd, rect, text, active in self.nav_buttons():
            self._draw_button(rect, text, cfg.FONT_CHIP, enabled=active,
                              active=False)

    def _ai_text_parts(self):
        """把 ``gs.ai_eval_text``（多行）拆成"用时/深度/评估"三段。

        **AI 正在思考时优先显示实时秒表**，而且返回的是一个**全新列表**——
        不能去改 ``gs.ai_eval_text``（那是渲染线程只读的字段），也不能被
        后面的评估文案更新覆盖掉。

        修过一个真 bug：秒表本来只是"文本层面"的替换，而
        ``_sync_ai_state`` 每帧都会把当前步的评估文案写进
        ``gs.ai_eval_text``；``eval_for_ply`` 又有"这一步还没有评估就退回
        上一步的"的兜底逻辑，于是 AI 一想棋，显示的立刻变回**上一步**的
        "AI 用时 2066ms"——玩家看到的正是"从 0 开始涨到 2 秒，然后又变回
        2000ms"，像是计时器被重置了。
        """
        if getattr(self.gs, 'ai_thinking', False):
            started = float(getattr(self.gs, 'ai_think_started', 0.0) or 0.0)
            if started > 0:
                return ["AI 思考中　%.1fs" % max(0.0, time.monotonic() - started)]
            return ["AI 思考中…"]
        raw = getattr(self.gs, 'ai_eval_text', '') or ''
        if not raw and getattr(self.gs, 'ai_think_ms', 0):
            raw = "AI 用时 %dms" % self.gs.ai_think_ms
        return [piece.strip() for piece in str(raw).split('\n') if piece.strip()]

    def _draw_llm_badge(self):
        """第二行的「点评」小徽标（开关本身在启动页上）。

        只在开着的时候显示，也是一条状态提示："这局的点评是开着的"。
        """
        rect = self.llm_badge_rect
        if rect is None:
            return
        cfg = self.config
        colors = cfg.COLORS
        pygame.draw.rect(self.window, colors['panel_card'], rect,
                         border_radius=rect.height // 2)
        pygame.draw.rect(self.window, colors['text_llm'], rect, 1,
                         border_radius=rect.height // 2)
        font = load_font(cfg.FONT_TINY)
        surf = font.render("点评", True, colors['text_llm'])
        self.window.blit(surf, surf.get_rect(center=rect.center))

    def _draw_ai_info(self):
        """第一行：AI 用时/深度（思考中则显示实时秒表）。

        "用时"取的是**真实挂钟时间**（从开始思考到落子落定），而不是引擎自报的
        搜索耗时——玩家感知到的是前者（真实反馈："明明过了很久，却显示才 2000ms"）。
        第一行现在整条都空着（难度/大模型开关已经搬到启动页），所以三段并排
        也放得下，不用再挑着显示。
        """
        cfg = self.config
        colors = cfg.COLORS
        parts = self._ai_text_parts()
        others = [p for p in parts if '评估' not in p]
        if not others:
            return
        font = load_font(cfg.FONT_SMALL)
        rect = self.ai_info_rect
        thinking = bool(getattr(self.gs, 'ai_thinking', False))
        text = _fit(font, "　".join(others), rect.width)
        if not text:
            return
        surf = font.render(text, True,
                           colors['text_warn'] if thinking else colors['text_llm'])
        self.window.blit(surf, (rect.right - surf.get_width(),
                                rect.y + (rect.height - surf.get_height()) // 2))

    def _draw_ai_eval(self):
        """第二行右侧：AI 评估（原来在右侧面板的「引擎评估」卡片里）。"""
        cfg = self.config
        colors = cfg.COLORS
        parts = [p for p in self._ai_text_parts() if '评估' in p]
        if not parts:
            return
        font = load_font(cfg.FONT_SMALL)
        rect = self.ai_eval_rect
        # 右边留 8px：``_fit`` 是拿 ``font.size()`` 估宽的，而**实际渲染**出来的
        # 宽度偶尔会略大于估计值（中文标点+全角括号），贴着边界就会擦到"点评"徽标。
        text = _fit(font, "  ".join(parts), max(0, rect.width - 8))
        if not text:
            return
        color = colors['text_warn'] if self.gs.in_check else colors['text_llm']
        surf = font.render(text, True, color)
        self.window.blit(surf, (rect.right - surf.get_width(),
                                rect.y + (rect.height - surf.get_height()) // 2))

    def _draw_toast(self):
        """toast 提示：**操作反馈放大显示在棋盘上**，几秒后缩回顶栏小字。

        为什么要有"大提示"：导出/导入棋谱这类操作点了之后界面上什么都没变，
        玩家根本不知道成没成功（真实反馈）。所以 ``loud`` 的提示会在棋盘上方
        弹一条带边框的横幅（写明存到哪个文件），几秒后自动消失；
        状态类提示（将军、思考中）仍然只走顶栏那一行小字。
        """
        text = getattr(self.gs, 'toast', '') or ''
        if not text:
            return
        if self._toast_is_loud():
            self._draw_toast_banner(text)
            return
        cfg = self.config
        colors = cfg.COLORS
        font = load_font(cfg.FONT_SMALL)
        rect = self.toast_rect
        if rect.width < 40:
            return
        shown = _fit(font, "· " + text, rect.width)
        if not shown:
            return
        surf = font.render(shown, True, colors['text_warn'])
        self.window.blit(surf, (rect.x,
                                rect.y + (rect.height - surf.get_height()) // 2))

    def _toast_is_loud(self) -> bool:
        """这条 toast 现在是否应当放大显示（loud 且还没过期）。"""
        if not getattr(self.gs, 'toast_loud', False):
            return False
        until = float(getattr(self.gs, 'toast_until', 0.0) or 0.0)
        return until > time.monotonic()

    def _draw_toast_banner(self, text):
        """操作反馈横幅：成功用强调色、失败用暖色，最后 1 秒淡出。"""
        cfg = self.config
        colors = cfg.COLORS
        until = float(getattr(self.gs, 'toast_until', 0.0) or 0.0)
        left = until - time.monotonic()
        alpha = 235 if left > 1.0 else max(60, int(235 * max(0.0, left)))

        body = load_font(cfg.FONT_BODY)
        shown = _fit(body, text, cfg.WIDTH - cfg.PAD * 6)
        surf = body.render(shown, True, colors['text_main'])
        rect = pygame.Rect(0, 0,
                           min(cfg.WIDTH - cfg.PAD * 2,
                               surf.get_width() + cfg.PAD * 3),
                           surf.get_height() + cfg.PAD * 2)
        rect.midtop = (self.board_rect.centerx, self.board_rect.y + cfg.PAD * 3)

        plate = pygame.Surface(rect.size, pygame.SRCALPHA)
        plate.fill((*colors['panel_card'], alpha))
        self.window.blit(plate, rect.topleft)
        bad = any(word in text for word in ('失败', '错误', '无法', '读不到'))
        edge = colors['text_warn'] if bad else colors['accent']
        pygame.draw.rect(self.window, edge, rect, 2, border_radius=cfg.RADIUS)
        self.window.blit(surf, surf.get_rect(center=rect.center))

    def _engines_status_text(self) -> str:
        """状态行里的"每方由谁出招"：``红:人工  黑:引擎``（都是人工时返回空）。

        用短名（"引擎"而不是"本机引擎"）：状态行宽度有限，两个全名加上手数/
        难度会被 ``_fit`` 截掉后半截，反而看不出黑方用什么——那这段信息就白加了。
        """
        gs = self.gs
        try:
            from jieqi import engines as _engines
        except ImportError:      # pragma: no cover
            return ""
        if all(gs.side_engine(s) == _engines.HUMAN for s in ("red", "black")):
            return ""            # 双人对下：不用占位置说"两边都是人工"
        short = {_engines.HUMAN: "人工", _engines.PYTHON: "引擎",
                 _engines.LLM: "大模型"}
        return " ".join("%s:%s" % (cn, short.get(gs.side_engine(side), ""))
                        for side, cn in (("red", "红"), ("black", "黑")))

    def _draw_status_line(self):
        """第二行：轮到谁（徽标）+ 手数 + 每方由谁出招 + 难度 + 将军/浏览/结束。"""
        cfg = self.config
        colors = cfg.COLORS
        turn = "红方" if self.gs.current_player == 'red' else "黑方"
        turn_color = colors['red_side'] if turn == "红方" else colors['black_side']
        badge_font = load_font(cfg.FONT_SMALL)
        label = f"轮到 {turn}"
        w = badge_font.size(label)[0] + 24
        rect = pygame.Rect(cfg.PAD_L, cfg.TOP_ROW2_Y, w, cfg.TOP_ROW2_H)
        ai_thinking = '思考中' in (getattr(self.gs, 'toast', '') or '')
        pulse = 0.55 + 0.45 * math.sin(pygame.time.get_ticks() / 220.0)
        bg = tuple(int(c * (0.35 + 0.25 * pulse)) for c in turn_color)
        self._draw_pill(self.window, rect, label, cfg.FONT_SMALL, bg, colors['text_main'])
        x = rect.right + cfg.PAD

        font = load_font(cfg.FONT_SMALL)
        bits = [f"第 {getattr(self.gs, 'ply', 0)}/{getattr(self.gs, 'total_plies', 0)} 手"]
        # 每方各一段"谁在走这一方的棋"：'红:人工 黑:本机引擎'。
        # 两方可以配不同的引擎（见 jieqi.engines），一行里都说清楚，
        # 免得玩家看着大模型在替自己走棋却以为是本机引擎。
        engines_bit = self._engines_status_text()
        if engines_bit:
            bits.append(engines_bit)
        bits.append(f"难度 {self._difficulty_cn()}")
        if self.gs.is_browsing:
            bits.append("浏览历史中（点「最新」继续）")
        elif self.gs.game_over:
            winner = getattr(self.gs, 'winner', None)
            bits.append("平局" if winner == 'draw'
                        else ("红方胜" if winner == 'red' else "黑方胜"))
        elif ai_thinking:
            bits.append("AI 思考中")
        elif self.gs.in_check:
            bits.append("将军！")
        color = (colors['text_warn'] if (self.gs.in_check or self.gs.is_browsing)
                 else colors['text_dim'])
        # 右边依次是 toast 提示和"AI 评估"，所以状态文字按固定上限截断，
        # 保证三段互不重叠（长提示不会撞上去）。
        text = _fit(font, "  ·  ".join(bits), self.status_text_max_w)
        self.window.blit(font.render(text, True, color),
                         (x, cfg.TOP_ROW2_Y + (cfg.TOP_ROW2_H - font.get_linesize()) // 2))

    # ============================================================ 右侧面板
    def _panel_layout(self):
        """面板各区块的**本地**矩形（相对面板左上角）。

        绘制与命中测试共用同一套几何，保证"看到的按钮"就是"点得到的按钮"。

        面板**只保留操作卡 + 走棋记录 + 大模型点评**：
        「对局状态」「引擎评估」两张卡都去掉了——轮到谁、第几手、AI 执哪方、
        难度、AI 用时/深度/评估在顶栏都已经有了，右侧再来一遍纯属重复。
        省下来的高度全给走棋记录（放不下时可以滚轮翻看）。
        """
        cfg = self.config
        w = self.panel_rect.width
        h = self.panel_rect.height
        inner = w - 2 * cfg.PAD
        y = cfg.PAD
        # 操作卡三行：导出/导入 + 回顾/研究/重开 + 评分。
        # 高度按内容算出来的（标题 30 + 32 + 8 + 30 + 8 + 26 = 134），
        # 少 1px 就会把最后一行按钮画到卡片外面。
        action = pygame.Rect(cfg.PAD, y, inner, 136)
        y = action.bottom + cfg.PAD
        log_title = pygame.Rect(cfg.PAD, y, inner, 22)
        y = log_title.bottom + 2
        comment_h = self._comment_height()
        log_bottom = h - comment_h - (cfg.PAD_S if comment_h else cfg.PAD)
        log = pygame.Rect(cfg.PAD, y, inner, max(40, log_bottom - y))
        comment = (pygame.Rect(cfg.PAD, h - comment_h, inner, comment_h)
                   if comment_h else None)
        return {'action': action, 'log_title': log_title, 'log': log,
                'comment': comment}

    # ---- 面板里的按钮（窗口坐标，供 button_at 命中） ----
    def _panel_rect_to_window(self, local: pygame.Rect) -> pygame.Rect:
        return local.move(self.panel_rect.x, self.panel_rect.y)

    def _action_rows(self):
        """操作卡内部各行的本地矩形。

        两行：导出/导入 → 回顾·重开，外加**人工打分**那一行（键盘 5~9 / 0）。
        高度按内容算（标题 30 + 32 + 8 + 30 + 8 + 26 = 134），
        少 1px 就会把最后一行按钮画到卡片外面。

        「开局/上一回合/下一回合/最新」四个翻手按钮已经搬到**顶栏第二行**：
        它们是每局都要按的高频操作，塞在面板里又小又挤（用户要的是"精简"）。
        「分支研究」那个按钮已随分支棋谱功能一起移除，所以第二行只剩两个。
        """
        cfg = self.config
        lay = self._panel_layout()
        card = lay['action']
        x = card.x + cfg.PAD
        w = card.width - 2 * cfg.PAD
        y = card.y + 30
        half = (w - 8) // 2
        row_a1 = pygame.Rect(x, y, half, 32)                      # 导出棋谱
        row_a2 = pygame.Rect(x + half + 8, y, w - half - 8, 32)   # 导入棋谱
        y = row_a1.bottom + cfg.PAD_S
        row_c1 = pygame.Rect(x, y, half, 30)                      # 回顾棋局
        row_c2 = pygame.Rect(x + half + 8, y, w - half - 8, 30)   # 重新开始
        y = row_c1.bottom + cfg.PAD_S
        # 评分行：5 档 + 清除，共 6 个小按钮
        gap = 4
        score_w = (w - 5 * gap) // 6
        scores = [pygame.Rect(x + i * (score_w + gap), y, score_w, 26)
                  for i in range(6)]
        return row_a1, row_a2, row_c1, row_c2, scores

    def record_button(self):
        """导出棋谱（有棋可导出时才可用）。"""
        row_a1 = self._action_rows()[0]
        enabled = bool(getattr(self.gs, 'move_history', None))
        return ('record:export', self._panel_rect_to_window(row_a1),
                "导出棋谱", enabled)

    def import_button(self):
        """导入棋谱（读回一份保存过的棋谱并重放出来）。"""
        row_a2 = self._action_rows()[1]
        return ('record:import', self._panel_rect_to_window(row_a2),
                "导入棋谱", True)

    def replay_button(self):
        """打开「回顾棋局」浮层（大棋盘 + 被吃之子，配合上一步/下一步复盘）。"""
        c1 = self._action_rows()[2]
        return ('view:board', self._panel_rect_to_window(c1),
                "回顾棋局", True)

    def nav_buttons(self):
        """走棋步数切换：开局 / 上一回合 / 下一回合 / 最新（顶栏第二行右侧）。

        **终局后也保持可用**：结算画面会盖住棋盘，如果连翻记录都点不动，玩家就
        被"点击任意位置重新开始"困住了（真实反馈）。翻到历史会自动退出结算状态。
        """
        cfg = self.config
        gs = self.gs
        font = load_font(cfg.FONT_CHIP)
        # 宽度按文案量出来（"上一回合"比"开局"宽得多），间距紧凑一点塞进顶栏
        labels = (('nav:first', '开局', gs.ply > 0),
                  ('nav:prev', '上一回合', gs.ply > 0),
                  ('nav:next', '下一回合', gs.ply < gs.total_plies),
                  ('nav:last', '最新', gs.ply < gs.total_plies))
        widths = [font.size(text)[0] + 20 for _c, text, _e in labels]
        gap = 6
        total = sum(widths) + gap * (len(widths) - 1)
        right = cfg.WINDOW_W - cfg.PAD_L
        x = right - total
        y = cfg.TOP_ROW2_Y
        out = []
        for (cmd, text, enabled), w in zip(labels, widths):
            out.append((cmd, pygame.Rect(x, y, w, cfg.TOP_ROW2_H), text, enabled))
            x += w + gap
        return out

    def restart_button(self):
        c2 = self._action_rows()[3]
        return ('game:restart', self._panel_rect_to_window(c2),
                "重新开始", True)

    #: 界面上的评分档：``(代码, 标签, 快捷键)``。
    #: 这里**只放展示需要的东西**（标签/键位），数值语义（``rating_good``）留在
    #: ``jieqi/record.py`` 的 ``RATINGS`` 里——UI 不依赖数据层的常量，
    #: 两边靠"代码字符串"对齐（``score:best`` 这样的命令）。
    SCORE_ITEMS = (
        ('best', '好棋', '5'),
        ('good', '正着', '6'),
        ('inaccuracy', '疑问手', '7'),
        ('mistake', '错着', '8'),
        ('blunder', '漏着', '9'),
    )

    def score_buttons(self):
        """人工打分：五档 + 清除（键盘 5~9 / 0 同效）。

        打分挂在**当前局面**上（= 给"从这里走出的那一手"打分），与棋谱里
        ``moves[i].human`` 的 slot 规则一致。开局没有可评的着法，所以灰掉。
        """
        _a1, _a2, _c1, _c2, rects = self._action_rows()
        has_move = int(getattr(self.gs, 'ply', 0)) > 0
        out = []
        for (code, label, _key), rect in zip(self.SCORE_ITEMS, rects):
            out.append(('score:' + code, self._panel_rect_to_window(rect),
                        label, has_move))
        out.append(('score:', self._panel_rect_to_window(rects[-1]),
                    "清除", has_move))
        return out

    # ---- 面板绘制 ----
    def draw_info_panel(self):
        cfg = self.config
        lay = self._panel_layout()
        panel = pygame.Surface((self.panel_rect.width, self.panel_rect.height))
        panel.fill(cfg.COLORS['panel_bg'])
        pygame.draw.line(panel, cfg.COLORS['panel_line'],
                         (0, 0), (0, panel.get_height()), 2)
        self._draw_action_card(panel, lay['action'])
        self._draw_move_log(panel, lay['log_title'], lay['log'])
        if lay['comment'] is not None:
            self._draw_comment_card(panel, lay['comment'])
        self.window.blit(panel, self.panel_rect.topleft)

    def _draw_action_card(self, panel, card):
        cfg = self.config
        self._card(panel, card, "操作")
        saved = self.window
        self.window = panel
        try:
            for getter in (self.record_button, self.import_button):
                _c, grect, gtext, genabled = getter()
                self._draw_button(grect.move(-self.panel_rect.x,
                                             -self.panel_rect.y),
                                  gtext, cfg.FONT_CHIP, enabled=genabled)
            for getter in (self.replay_button, self.restart_button):
                _c, grect, gtext, genabled = getter()
                self._draw_button(grect.move(-self.panel_rect.x,
                                             -self.panel_rect.y),
                                  gtext, cfg.FONT_CHIP,
                                  enabled=genabled)
            # 评分行：当前档位高亮，其余普通
            current = str(getattr(self.gs, 'ai_score_code', '') or '')
            for cmd, srect, stext, senabled in self.score_buttons():
                self._draw_button(srect.move(-self.panel_rect.x,
                                             -self.panel_rect.y),
                                  stext, cfg.FONT_CHIP,
                                  active=(cmd == 'score:' + current
                                          if current else cmd == 'score:'),
                                  enabled=senabled)
        finally:
            self.window = saved

    def _wrapped_comment(self, max_lines=None):
        """折行后的点评（带缓存，一帧只折一次）。"""
        comment = getattr(self.gs, 'ai_comment', '') or ''
        if not comment:
            return []
        key = (comment, self.panel_rect.width)
        if key != self._wrap_cache_key:
            small = load_font(self.config.FONT_SMALL)
            self._wrap_cache = _wrap(small, comment, self.panel_rect.width - 24)
            self._wrap_cache_key = key
        lines = self._wrap_cache
        return lines[:max_lines] if max_lines else lines

    def _comment_height(self):
        """点评卡的高度（没有点评时为 0，记录区就占满剩余空间）。

        点评很长时**不无限长高**：卡最高占面板的 45%（``_COMMENT_MAX_RATIO``），
        超出的部分用滚轮翻看。早前版本把卡按内容撑满，长点评要么把走棋记录
        挤没了、要么被硬截断（真实反馈："ai点评太长被截断了"）。
        """
        if not (getattr(self.gs, 'ai_comment', '') or ''):
            return 0
        cfg = self.config
        small = load_font(cfg.FONT_SMALL)
        lines = self._wrapped_comment()
        title_h = load_font(cfg.FONT_TINY).get_linesize() + 6
        want = cfg.PAD + title_h + len(lines) * small.get_linesize() + cfg.PAD
        cap = int(self.panel_rect.height * _COMMENT_MAX_RATIO)
        return max(cfg.PAD * 2 + title_h + small.get_linesize(), min(want, cap))

    def _comment_view_lines(self):
        """点评卡当前能显示的那几行（带滚动偏移）。"""
        cfg = self.config
        small = load_font(cfg.FONT_SMALL)
        card = self._panel_layout()['comment']
        if card is None:
            return [], 0, 0
        title_h = load_font(cfg.FONT_TINY).get_linesize() + 6
        body_h = max(0, card.height - cfg.PAD * 2 - title_h)
        per_page = max(1, body_h // small.get_linesize())
        lines = self._wrapped_comment()
        max_offset = max(0, len(lines) - per_page)
        offset = max(0, min(int(getattr(self, 'comment_scroll', 0)), max_offset))
        return lines[offset:offset + per_page], offset, max_offset

    def scroll_comment(self, delta: int) -> int:
        """滚轮滚动大模型点评卡；返回滚动后的偏移。"""
        _lines, _offset, max_offset = self._comment_view_lines()
        self.comment_scroll = max(0, min(max_offset,
                                        int(getattr(self, 'comment_scroll', 0))
                                        + int(delta)))
        return self.comment_scroll

    def _draw_move_log(self, panel, title_rect, log_rect):
        cfg = self.config
        colors = cfg.COLORS
        title = load_font(cfg.FONT_TITLE).render("走棋记录", True, colors['text_main'])
        panel.blit(title, (title_rect.x, title_rect.y))
        # 坐标图例：记录里的 ``b7b0`` 是"列字母 + 行号(0~9)"，
        # 与棋盘左/上那圈行号列号是同一套；不写一句玩家得自己猜。
        # 只写 ``a0~i9`` 这么短，免得把右边的"显示到第 N 手"挤掉。
        legend = load_font(cfg.FONT_TINY).render("a0~i9", True,
                                                 colors['text_faint'])
        panel.blit(legend, (title_rect.x + title.get_width() + cfg.PAD_S,
                            title_rect.y + 5))

        move_font = load_font(cfg.FONT_MOVE)
        line_h = self._log_line_height()
        max_lines = max(1, log_rect.height // line_h)
        # 手数或记录长度变了（落子 / 导入棋谱 / 切换步数）：滚回贴住当前手，
        # 否则玩家会看到"记录停在旧位置"的错位画面。
        state_key = (int(getattr(self.gs, 'ply', 0) or 0),
                     len(getattr(self.gs, 'move_history', None) or ()))
        if state_key != self._log_state_key:
            self._log_state_key = state_key
            self._clamp_log_scroll(resync=True)
        offset = self._clamp_log_scroll(resync=False)
        max_offset = int(getattr(self, 'log_scroll_max', 0))
        # 标题右侧的提示：滚过历史时说明"现在看到哪一段"，并提示能不能滚
        if max_offset <= 0:
            hint_text = f"显示到第 {getattr(self.gs, 'ply', 0)} 手"
        else:
            top_ply = max(1, int(getattr(self.gs, 'ply', 0)) - offset)
            hint_text = f"显示到第 {top_ply} 手　滚轮翻看"
        hint = load_font(cfg.FONT_TINY).render(hint_text, True, colors['text_faint'])
        panel.blit(hint, (title_rect.right - hint.get_width(), title_rect.y + 4))

        lines = self._move_lines(max_lines, offset=offset)
        y = log_rect.y
        for text, color, is_current, is_ai in lines:
            if is_current:
                pygame.draw.rect(panel, colors['panel_card_alt'],
                                 pygame.Rect(log_rect.x - 4, y - 2,
                                             log_rect.width + 8, line_h),
                                 border_radius=cfg.RADIUS_S)
            # AI 的着法加一条左侧色条，一眼能看出"这步是 AI 走的"
            if is_ai:
                pygame.draw.rect(panel, colors['accent'],
                                 pygame.Rect(log_rect.x - 4, y, 3, line_h - 3),
                                 border_radius=2)
            panel.blit(move_font.render(_fit(move_font, text, log_rect.width - 8),
                                        True, color),
                       (log_rect.x + (6 if is_ai else 0), y))
            y += line_h

        # 滚动条：只在"一屏放不下"时出现，宽度 3px，位置贴着记录区右缘
        if max_offset > 0:
            track = pygame.Rect(log_rect.right - 1, log_rect.y, 3, log_rect.height)
            pygame.draw.rect(panel, colors['panel_line_soft'], track,
                             border_radius=2)
            total = max_offset + max_lines
            thumb_h = max(24, int(log_rect.height * max_lines / max(1, total)))
            # offset 越大 = 看得越早 = 滑块越靠上
            travel = max(0, log_rect.height - thumb_h)
            thumb_y = log_rect.y + int(travel * (1 - offset / max(1, max_offset)))
            thumb = pygame.Rect(track.x, thumb_y, 3, thumb_h)
            pygame.draw.rect(panel, colors['btn_bg_active'], thumb, border_radius=2)

    def _draw_comment_card(self, panel, card):
        cfg = self.config
        colors = cfg.COLORS
        content = self._card(panel, card)
        title_font = load_font(cfg.FONT_TINY)
        title = "大模型点评 · AI 自己的招法"
        if self.gs.is_browsing:
            title = f"大模型点评 · 第 {getattr(self.gs, 'ply', 0)} 手 · AI 招法"
        lines, offset, max_offset = self._comment_view_lines()
        tag = title
        if max_offset > 0:
            tag += f"（{offset + 1}-{offset + len(lines)}/{max_offset + len(lines)}　滚轮翻看）"
        panel.blit(title_font.render(_fit(title_font, tag, content.width),
                                     True, colors['text_llm']),
                   (content.x, content.y))
        small = load_font(cfg.FONT_SMALL)
        y = content.y + title_font.get_linesize() + 4
        bottom = card.bottom - cfg.PAD
        for line in lines:
            if y + small.get_linesize() > bottom:
                break
            panel.blit(small.render(line, True, colors['text_main']), (content.x, y))
            y += small.get_linesize()
        # 长点评：卡片右侧一条细滚动条，提示"下面还有"
        if max_offset > 0:
            total = max_offset + len(lines)
            thumb_h = max(18, int(card.height * len(lines) / max(1, total)))
            travel = max(0, card.height - thumb_h)
            thumb_y = card.y + int(travel * offset / max(1, max_offset))
            pygame.draw.rect(panel, colors['panel_line_soft'],
                             pygame.Rect(card.right - 3, card.y, 3, card.height),
                             border_radius=2)
            pygame.draw.rect(panel, colors['btn_bg_active'],
                             pygame.Rect(card.right - 3, thumb_y, 3, thumb_h),
                             border_radius=2)

    #: 兼容旧调用名
    def draw_llm_panel(self):
        self.draw_info_panel()

    def _info_lines(self):
        """面板信息行（保留旧接口，返回 ``(文案, 颜色)``）。"""
        colors = self.config.COLORS
        rows = []
        turn = "红方" if self.gs.current_player == 'red' else "黑方"
        rows.append((f"轮到：{turn}", colors['text_main']))
        if self.gs.game_over:
            winner = getattr(self.gs, 'winner', None)
            text = "平局" if winner == 'draw' else ("红方胜" if winner == 'red' else "黑方胜")
            rows.append((f"结果：{text}", colors['text_warn']))
            if getattr(self.gs, 'draw_reason', None):
                rows.append((self.gs.draw_reason, colors['text_dim']))
        rows.append((f"手数：{getattr(self.gs, 'ply', 0)}", colors['text_dim']))
        for piece in str(getattr(self.gs, 'ai_eval_text', '') or '').split('\n'):
            if piece.strip():
                rows.append((piece.strip(), colors['text_llm']))
        rows.append((f"难度：{self._difficulty_cn()}", colors['text_dim']))
        state = "开" if self.llm_on else "关"
        rows.append((f"大模型点评：{state}",
                     colors['text_llm'] if self.llm_on else colors['text_dim']))
        return rows

    def _difficulty_cn(self):
        return {'easy': '简单', 'normal': '普通', 'hard': '困难',
                'expert': '专家'}.get(str(self.difficulty).lower(),
                                     str(self.difficulty))

    def _ai_plies(self):
        """**由引擎/大模型走出**的是哪几手（用于给记录加色条）。

        改成按"走这一手的那一刻、那一方的档位"判定，而不是按"棋子颜色等于
        ``ai_color``"。因为现在两方可以各配不同的引擎，甚至中途换档位
        ——记录条的颜色要如实反映"这手是谁下的"。
        没有对局信息（例如导入的棋谱）时返回空集。
        """
        out = set()
        hist = getattr(self.gs, 'move_history', None) or []
        if not hist:
            return out
        try:
            from jieqi import engines as _engines
        except ImportError:      # pragma: no cover
            return out
        for i, rec in enumerate(hist):
            piece = rec.get('piece') or {}
            color = piece.get('color')
            if color not in ('red', 'black'):
                continue
            # 记录里存了当时那一刻的档位（换过档位的对局才看得出差别）；
            # 老棋谱没有这个字段，就按当前档位判断。
            source = rec.get('engine') or self.gs.side_engine(color)
            if source != _engines.HUMAN:
                out.add(i + 1)
        return out

    def _move_lines(self, limit, offset=0):
        """走棋记录：显示到**当前浏览的那一手**为止，并标出是哪一手。

        ``offset`` 是"再往前翻几手"（滚动条用）：0 = 贴到最新一手，
        正数 = 往前看更早的记录。

        返回 ``[(文案, 颜色, 是否当前手, 是否 AI 走的), ...]``。
        """
        from jieqi.gui_adapter import move_record_text

        cfg = self.config
        history = getattr(self.gs, 'move_history', None) or []
        ply = int(getattr(self.gs, 'ply', len(history)) or 0)
        if not history or ply <= 0:
            return [("（还没有走子）", cfg.COLORS['text_faint'], False, False)]
        ai_plies = self._ai_plies()
        visible = min(ply, len(history))
        end = max(1, visible - max(0, int(offset)))    # 显示到第 end 手（含）
        start = max(0, end - limit)
        lines = []
        for i in range(start, end):
            text = move_record_text(history[i], i + 1)
            is_current = (i + 1) == ply
            color = (cfg.COLORS['text_main'] if is_current
                     else cfg.COLORS['text_dim'])
            lines.append((text, color, is_current, (i + 1) in ai_plies))
        return lines

    def _max_log_offset(self, log_rect) -> int:
        """走棋记录最多能往前翻几手（保证不会翻过开局）。"""
        history = getattr(self.gs, 'move_history', None) or []
        ply = int(getattr(self.gs, 'ply', 0) or 0)
        visible = min(ply, len(history))
        per_page = max(1, log_rect.height // self._log_line_height())
        return max(0, visible - per_page)

    def _log_line_height(self) -> int:
        return load_font(self.config.FONT_MOVE).get_linesize() + 3

    def _clamp_log_scroll(self, delta=0, resync=False):
        """调整/校正走棋记录的滚动位置。

        走棋记录变长以后一屏放不下，所以要能翻。规则：

        * 新建对局、导入棋谱、切换步数时 ``resync=True``：**滚到最新一手**；
        * 玩家滚轮翻记录时 ``delta`` 加减，范围限制在 ``[0, 最大偏移]``。
        """
        cfg = self.config
        lay = self._panel_layout()
        max_offset = self._max_log_offset(lay['log'])
        if resync:
            self.log_scroll = 0
        else:
            self.log_scroll = max(0, min(max_offset,
                                         int(self.log_scroll) + int(delta)))
        self.log_scroll_max = max_offset
        return self.log_scroll

    def scroll_move_log(self, delta: int) -> bool:
        """滚轮滚动走棋记录；返回是否真的滚动了（用于决定要不要让棋盘吃事件）。"""
        before = int(getattr(self, 'log_scroll', 0))
        self._clamp_log_scroll(delta=delta)
        return int(self.log_scroll) != before

    # ============================================================== 结算
    def draw_game_over(self):
        """结算覆盖层（画在**当前绘制目标**上，棋盘内/窗口上都能用）。"""
        cfg = self.config
        colors = cfg.COLORS
        area_w, area_h = self.screen.get_size()
        overlay = pygame.Surface((area_w, area_h), pygame.SRCALPHA)
        overlay.fill((8, 8, 12, 190))
        self.screen.blit(overlay, (0, 0))

        if self.gs.winner == 'draw':
            title, accent = "平 局", colors['text_main']
        elif self.gs.winner == 'red':
            title, accent = "红方 胜利", colors['red_side']
        else:
            title, accent = "黑方 胜利", colors['black_side']

        card = pygame.Rect(0, 0, 420, 200)
        card.center = (area_w // 2, area_h // 2)
        pygame.draw.rect(self.screen, colors['card_shadow'], card.move(0, 6),
                         border_radius=cfg.RADIUS_L)
        pygame.draw.rect(self.screen, colors['panel_card'], card,
                         border_radius=cfg.RADIUS_L)
        pygame.draw.rect(self.screen, accent, card, 2, border_radius=cfg.RADIUS_L)

        big = load_font(cfg.FONT_OVER)
        surf = big.render(title, True, accent)
        self.screen.blit(surf, surf.get_rect(center=(card.centerx, card.y + 74)))

        small = load_font(cfg.FONT_SMALL)
        reason = getattr(self.gs, 'draw_reason', None)
        sub = reason or f"共 {getattr(self.gs, 'total_plies', 0)} 手"
        surf = small.render(sub, True, colors['text_dim'])
        self.screen.blit(surf, surf.get_rect(center=(card.centerx, card.y + 122)))

        hint = load_font(cfg.FONT_BODY).render("点击任意位置重新开始", True,
                                               colors['text_main'])
        self.screen.blit(hint, hint.get_rect(center=(card.centerx, card.y + 164)))

    @property
    def start_kwargs(self) -> dict:
        """启动页需要的全部状态（矩形与绘制共用同一份，避免两处取值不一致）。"""
        gs = self.gs
        #: 「继续上一局」：有一局走到一半（有棋、没终局、也不是导入的别人的棋谱）
        unfinished = bool(getattr(gs, 'move_history', None)) and not gs.game_over
        return {
            'difficulty': self.difficulty,
            'llm_on': self.llm_on,
            'apply_llm': self.apply_llm_move,
            'auto_save': self.auto_save,
            'llm_note': self.llm_note,
            'ai_note': self.ai_note,
            'has_unfinished': unfinished,
            'record_count': self._record_count(),
            #: 每方当前的对局方式（启动页那两行 chip 靠它决定谁被选中）
            'side_engines': dict(getattr(gs, 'side_engines', None) or {}),
            #: 玩家执哪一方（启动页的"我执红/我执黑"chip 与开始按钮文案）
            'player_side': getattr(gs, 'player_side', 'red') or 'red',
            'mouse_pos': self.mouse_pos,
        }

    def _record_count(self) -> int:
        """``records/`` 里已保存几份棋谱（只数一次，失败就当 0）。

        为什么缓存：``draw_all`` 每帧都会走启动页这条路，不能每帧去列目录。
        """
        if self._record_count_cache is not None:
            return self._record_count_cache
        count = 0
        try:
            from jieqi import record as _rec

            count = len(_rec.list_records())
        except Exception:
            count = 0
        self._record_count_cache = count
        return count

    def start_layout(self) -> StartLayout:
        """构造（并缓存）启动页布局。设置一变就重建，保证几何与文案同步。"""
        kwargs = self.start_kwargs
        key = tuple(sorted((k, v) for k, v in kwargs.items() if k != 'mouse_pos'))
        if key != self._start_layout_key or self._start_layout is None:
            self._start_layout_key = key
            self._start_layout = StartLayout(self.config, **kwargs)
        else:
            self._start_layout.mouse_pos = self.mouse_pos
        self._start_layout._unfinished_plies = len(
            getattr(self.gs, 'move_history', None) or ())
        return self._start_layout

    def start_screen_buttons(self):
        """启动页上的所有按钮 ``[(cmd, rect, text, active), …]``。"""
        return self.start_layout().all_buttons()

    def draw_side_selection(self):
        """启动页（整窗）：标题 + 对局设置 + 阵营选择 + 继续/载入。

        名字保留 ``draw_side_selection``（老代码/测试都用它），但它现在不只是
        "选红还是选黑"，而是**开局前的全部设置**：难度、大模型开关、自动保存
        都在这里改，棋盘页因此能瘦下来（用户需求："一些配置选项可以放在这里"）。
        """
        self.start_layout().draw(self.window)

    # ---- 启动页控件（与 ``start_screen_buttons`` 同一批矩形） ----
    def difficulty_buttons(self):
        """难度 chip（在**启动页**上，不再是顶栏）。"""
        layout = self.start_layout()
        out = []
        for name, _label, _hint in DIFFICULTY_ITEMS:
            entry = layout.difficulty_button(name)
            if entry is not None:
                out.append(entry)
        return out

    def engine_buttons(self):
        """每方四枚"对局方式"chip（``engine:<side>:<source>``）。

        这两个方法原来叫 ``llm_button`` / ``apply_llm_button``（两个独立开关），
        现在并进了"每方一个引擎档位"里：选"大模型"就等于"完全采用它的招法"，
        不再需要单独的"应用招法"开关。
        """
        out = []
        for cmd, rect, text, active in self.start_layout().all_buttons():
            if str(cmd).startswith('engine:'):
                out.append((cmd, rect, text, active))
        return out
