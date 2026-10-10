# start_screen.py
"""启动页（阵营选择 / 对局设置）的几何与绘制。

**为什么单独一个模块**：启动页和棋盘页是两块完全不同的界面，几何却必须
"画在这里 = 点在这里"。把这些矩形集中到一个模块，``Renderer.draw_start_screen``
与 ``Renderer.button_at`` 取的是**同一批 Rect**，不会出现"看着有按钮、点不到"
的情况（旧版启动页的按钮几何散在 renderer 里，加一个控件就得改三处）。

布局（窗口坐标系，整窗铺满）::

    ┌──────────────────────────────────────────────┐
    │                                              │
    │        ┌────────────────────────────┐        │
    │        │ 揭 棋                      │        │
    │        │ 中国象棋 · 暗子对局         │        │
    │        ├──────────────┬─────────────┤        │
    │        │ 难度  [简][普]│ 我的阵营     │        │
    │        │ 红方引擎     │ [执红][执黑] │        │
    │        │ 黑方引擎     │ 继续对局     │        │
    │        │ 自动保存     │ 导入棋谱     │        │
    │        └──────────────┴─────────────┘        │
    └──────────────────────────────────────────────┘

两列布局（而不是竖着堆一列）是为了**塞进屏幕**：整窗高度上限 1000px
（``config._MAX_WINDOW_H``），竖排一行一个控件会直接超出去。

**"红方引擎 / 黑方引擎"两行**是"对局方式"的入口：每方各挑一个
（人工输入 / 本机引擎 / 大模型），选谁就完全采用谁的招法。
原来的"大模型点评 / 应用招法"两个开关**并掉了**——它们现在由"某方引擎=
大模型"直接表达，少两个开关也少两处能配错的地方。
"""
from __future__ import annotations

from collections import OrderedDict

import pygame

from resource_manager import load_font

try:  # 对局方式的定义（纯常量模块）
    from jieqi import engines as _engines
except ImportError:  # pragma: no cover
    from Game.jieqi import engines as _engines

#: 启动页文案里出现的难度档（顺序 = 显示顺序，与 ``jieqi.engine.DIFFICULTIES`` 对齐）
DIFFICULTY_ITEMS = (
    ('easy', '简单', '快棋，一步约 1 秒'),
    ('normal', '普通', '默认档，一步约 2 秒'),
    ('hard', '困难', '一步约 4 秒'),
    ('expert', '专家', '一步约 8 秒'),
)

#: 难度副标题（鼠标悬停时显示在标题下方的那一行）
DIFFICULTY_HINT = {name: hint for name, _label, hint in DIFFICULTY_ITEMS}


def _blend(color, target, t: float):
    """在两个颜色之间线性插值（``t`` 超出 [0,1] 会被夹住）。"""
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(color, target))


class StartLayout:
    """启动页所有矩形 + 绘制。

    一次构建、一次使用：构造时按当前 ``(difficulty, llm_on, apply_llm, auto_save)``
    算出文案宽度与整卡尺寸，``draw()`` 再按同一批矩形作画。
    """

    CARD_MAX_W = 860
    CARD_SIDE_MIN = 24          # 卡片左右至少留这么多（窗口再窄也不贴边）

    def __init__(self, config, *, difficulty='normal', llm_on=False,
                 apply_llm=False, auto_save=True, llm_note='',
                 ai_note='', has_unfinished=False, record_count=0,
                 side_engines=None, player_side='red', mouse_pos=(-1, -1)):
        self.cfg = config
        self.difficulty = str(difficulty or 'normal')
        self.llm_on = bool(llm_on)
        self.apply_llm = bool(apply_llm)
        self.auto_save = bool(auto_save)
        self.llm_note = str(llm_note or '')
        self.ai_note = str(ai_note or '')
        self.has_unfinished = bool(has_unfinished)
        self.record_count = int(record_count or 0)
        #: 玩家执哪一方（红/黑）。阵营现在只是**选择**，开局要点"开始游戏"。
        self.player_side = 'black' if str(player_side).lower() == 'black' else 'red'
        #: 每方当前的对局方式（``{'red': 'human', 'black': 'python'}``）。
        #: 传 ``None`` 就用默认（红方人工 / 黑方本机引擎）。
        self.side_engines = dict(side_engines or _engines.DEFAULT_FOR_SIDE)
        for _side in _engines.SIDES:
            self.side_engines[_side] = _engines.normalize(
                self.side_engines.get(_side),
                fallback=_engines.DEFAULT_FOR_SIDE[_side])
        self.mouse_pos = tuple(mouse_pos)

        w = config.WINDOW_W
        h = config.WINDOW_H
        self.window = pygame.Rect(0, 0, w, h)

        card_w = max(560, min(self.CARD_MAX_W, w - 2 * self.CARD_SIDE_MIN))
        #: 卡片高度：左列要放下 5 行控件（难度 / 红方引擎 / 黑方引擎 /
        #: 大模型点评 / 自动保存），每行"chip + 下面一行小字"占 84px。
        #: 按实际内容算而不是写死常数——写死过一次，加一行就overflow。
        rows = 5
        card_h = 132 + rows * 84 + 22
        #: 窗口比卡片矮时（首次启动窗口是 720 高）整体压缩，绝不画到窗口外
        card_h = min(card_h, max(360, h - 24))
        self.card = pygame.Rect(0, 0, card_w, card_h)
        self.card.center = (w // 2, h // 2 + 4)
        self.title_pos = (self.card.centerx, self.card.y + 40)
        self.sub_pos = (self.card.centerx, self.card.y + 74)
        self.divider_y = self.card.y + 100
        self.footer_pos = (self.card.centerx, self.card.bottom - 28)

        self._label_font = load_font(config.FONT_SMALL)
        left_label_w = max(self._label_font.size(t)[0]
                           for t in ("AI 难度", "红方引擎", "黑方引擎",
                                     "大模型点评", "自动保存"))
        self._col_gap = 24
        self._left_x = self.card.x + 30
        self._left_w = 430
        self._left_pitch = 84
        self._col_x = self._left_x + self._left_w + self._col_gap
        self._col_w = self.card.right - 30 - self._col_x
        self._label_w = left_label_w + 12

        self._build_left()
        self._build_right()

    # ------------------------------------------------------------------ 左列
    def _left_x_in(self) -> int:
        return self._left_x + self._label_w

    def _build_left(self):
        cfg = self.cfg
        x = self._left_x_in()
        w = self._left_w - self._label_w

        # 难度：四枚 chip 平分一行（放不下就缩，绝不画到列外）
        chip_font = load_font(cfg.FONT_CHIP)
        natural = [chip_font.size(label)[0] + 26
                   for _n, label, _h in DIFFICULTY_ITEMS]
        gap = 6
        total = sum(natural) + gap * (len(natural) - 1)
        if total > w:                       # 窄窗口：整体等比缩小
            scale = max(0.5, w / float(total))
            natural = [max(44, int(n * scale)) for n in natural]
        # 左右分别对齐到列的边界：把富余宽度**平分**给每枚 chip，
        # 而不是全塞给最后一枚（那样"专家"会比别的宽一倍，看着像排版错了）。
        share = max(0, (w - sum(natural) - gap * (len(natural) - 1))) \
            // len(natural)
        widths = [n + share for n in natural]
        widths[-1] += w - (sum(widths) + gap * (len(widths) - 1))
        self.difficulty_chips = OrderedDict()
        cx = x
        for (name, _label, _hint), cw in zip(DIFFICULTY_ITEMS, widths):
            self.difficulty_chips[name] = pygame.Rect(
                cx, self.card.y + 132, cw, 34)
            cx += cw + gap

        # 每方一行几枚 chip（人工 / 本机引擎 / 大模型）。
        # 原来的"大模型点评""应用招法"两行开关在这里并掉了，行数不变。
        self.engine_chips = {}
        for row_index, side in enumerate(_engines.SIDES):
            row_y = self.card.y + 216 + row_index * 84
            self.engine_chips[side] = OrderedDict()
            chips = _engines.chips_for(side)
            chip_font = load_font(cfg.FONT_CHIP)
            natural = [chip_font.size(label)[0] + 22
                       for _src, label, _ok, _why in chips]
            gap_e = 5
            total_e = sum(natural) + gap_e * (len(natural) - 1)
            if total_e > w:                 # 窄窗口：整体等比缩小
                scale = max(0.5, w / float(total_e))
                natural = [max(40, int(n * scale)) for n in natural]
            share = max(0, (w - sum(natural) - gap_e * (len(natural) - 1))) \
                // len(natural)
            widths_e = [n + share for n in natural]
            widths_e[-1] += w - (sum(widths_e) + gap_e * (len(widths_e) - 1))
            ex = x
            for (src, _label, _ok, _why), cw in zip(chips, widths_e):
                self.engine_chips[side][src] = pygame.Rect(ex, row_y, cw, 34)
                ex += cw + gap_e

        self.auto_toggle = pygame.Rect(x, self.card.y + 384 + 84, w, 34)
        self.llm_toggle = pygame.Rect(x, self.card.y + 384, w, 34)

        # 行标签（与左边的行垂直居中对齐）
        self.row_labels = {
            'difficulty': pygame.Rect(self._left_x, self.card.y + 139,
                                      self._label_w - 12, 20),
            'engine_red': pygame.Rect(self._left_x, self.card.y + 223,
                                      self._label_w - 12, 20),
            'engine_black': pygame.Rect(self._left_x, self.card.y + 307,
                                        self._label_w - 12, 20),
            'llm': pygame.Rect(self._left_x, self.card.y + 391,
                               self._label_w - 12, 20),
            'auto': pygame.Rect(self._left_x, self.card.y + 475,
                                self._label_w - 12, 20),
        }
        # 每行下面一小行状态说明（模型名 / 配置提示 / 这一档做什么）
        self.row_notes = {
            'difficulty': pygame.Rect(x, self.card.y + 170, w, 18),
            'engine_red': pygame.Rect(x, self.card.y + 254, w, 18),
            'engine_black': pygame.Rect(x, self.card.y + 338, w, 18),
            'llm': pygame.Rect(x, self.card.y + 422, w, 18),
            'auto': pygame.Rect(x, self.card.y + 506, w, 18),
        }

    # ------------------------------------------------------------------ 右列
    def _build_right(self):
        x = self._col_x
        w = self._col_w
        top = self.card.y + 132
        #: **一个**开始按钮。阵营不再占界面位置（用户："去掉我执红或者我执黑
        #: 选项，太啰嗦了，有需要我会自己调的"），所以右列只留三个动作按钮。
        self.start_button = pygame.Rect(x, top, w, 56)
        btn_h = 42
        y = self.start_button.bottom + 16
        self.continue_button = pygame.Rect(x, y, w, btn_h)
        self.import_button = pygame.Rect(x, y + btn_h + 10, w, btn_h)

    # ------------------------------------------------------------------ 查询
    def difficulty_button(self, name):
        rect = self.difficulty_chips.get(str(name).lower())
        if rect is None:
            return None
        label = next((lab for n, lab, _h in DIFFICULTY_ITEMS
                      if n == str(name).lower()), str(name))
        return ('difficulty:' + str(name).lower(), rect, label,
                str(name).lower() == self.difficulty)

    def all_buttons(self):
        """``[(命令, Rect, 文案, 选中/开启), …]``（绘制与命中测试共用）。

        第四个字段的语义与 ``renderer._all_buttons`` 一致 = **"当前是否选中/开启"**
        （难度当前档、开关是否打开、自动保存是否开）。**它不表示"能不能点"**：
        "继续上一局"没有棋可继续时也是返回 False，但那种"灰掉"要靠
        :meth:`disabled` 判断——把两件事混在一个字段里，会让"当前难度"被
        误当成"不可点"，点难度就永远没反应（真的踩过）。
        """
        out = []
        for name, _label, _hint in DIFFICULTY_ITEMS:
            entry = self.difficulty_button(name)
            if entry is not None:
                out.append(entry)
        # 每方四枚"对局方式"chip。第四个字段 = 是否是当前选中的那一档
        # （**不表示能不能点**，能不能点看 disabled()）。
        for side, cn in (('red', '红'), ('black', '黑')):
            chips = self.engine_chips.get(side) or {}
            for source in _engines.SOURCES:
                rect = chips.get(source)
                if rect is None:
                    continue
                label = "%s方 %s" % (cn, _engines.CHIP_LABELS[source])
                out.append(('engine:%s:%s' % (side, source), rect, label,
                            self.side_engines.get(side) == source))
        out.append(('llm:toggle', self.llm_toggle,
                    "大模型点评：%s" % ('开' if self.llm_on else '关'),
                    self.llm_on))
        out.append(('setting:autosave', self.auto_toggle,
                    "自动保存棋谱：%s" % ('开' if self.auto_save else '关'),
                    self.auto_save))
        # 阵营不再占界面位置：默认红方先手（人机对战时 AI 执黑）。
        # 想换边：`Game/ai_config.json` 的 `ai.player_side`，或中局按 `C`。
        out.append(('game:start', self.start_button, "开始游戏", True))
        out.append(('game:continue', self.continue_button,
                    "继续上一局（%d 手）" % self._unfinished_plies,
                    self.has_unfinished))
        out.append(('record:import', self.import_button, "载入棋谱…", True))
        return out

    #: 「继续上一局」按钮上的手数由 ``renderer`` 注入（避免这里依赖 GameState）
    _unfinished_plies = 0

    def disabled(self, cmd: str) -> bool:
        """按钮是否是"灰掉、点了没反应"的状态。

        两类：一是"继续上一局"没有棋可继续；二是**这一档引擎用不了**
        （大模型模块不可用）。第二种必须也在这里灰掉，否则"点了没反应"
        会被当成 bug。
        """
        text = str(cmd or '')
        if text == 'game:continue':
            return not self.has_unfinished
        if text.startswith('engine:'):
            parts = text.split(':')
            if len(parts) == 3:
                return not _engines.available(parts[2])
        return False

    def engine_note(self, side: str) -> str:
        """某一方当前档位下面那行说明（含"为什么是灰的"）。"""
        side = side if side in _engines.SIDES else 'red'
        source = self.side_engines.get(side, _engines.HUMAN)
        extra = ''
        if source == _engines.LLM and self.llm_note:
            extra = self.llm_note
        return _engines.describe(source, difficulty=self.difficulty, note=extra)

    def side_engine_note(self, side: str, source: str) -> str:
        """某一方**某一档**的说明（鼠标悬停时用，讲"这一档会做什么"）。"""
        return _engines.describe(source, difficulty=self.difficulty,
                                 note=self.llm_note)

    def hit(self, pos):
        for cmd, rect, _text, _active in self.all_buttons():
            if self.disabled(cmd):
                continue
            if rect.collidepoint(pos):
                return cmd
        return None

    def hovered(self):
        """当前鼠标悬停的控件（用来显示副标题提示）。"""
        for _cmd, rect, _text, _active in self.all_buttons():
            if rect.collidepoint(self.mouse_pos):
                return rect
        return None

    # ------------------------------------------------------------------ 绘制
    def draw(self, surface):
        colors = self.cfg.COLORS
        self._draw_background(surface)
        self._draw_card(surface)
        self._draw_header(surface)
        self._draw_left(surface)
        self._draw_right(surface)
        self._draw_footer(surface)

    # ---- 背景：深色底 + 中心冷光 + 右上暖色晕染 + 一条强调色带 ----
    def _draw_background(self, surface):
        cfg = self.cfg
        colors = cfg.COLORS
        surface.fill(colors['start_bg'])
        glow = colors['start_glow']
        # 中心冷光：从卡片中心向外 60 步径向渐变（每步一圈矩形，够快也够柔）
        steps = 60
        max_r = int(math_hypot(cfg.WINDOW_W, cfg.WINDOW_H) * 0.55)
        for i in range(steps, 0, -1):
            r = int(max_r * i / steps)
            t = 1.0 - (i / float(steps))
            color = _blend(colors['start_bg'], glow, 0.42 * t * t)
            pygame.draw.circle(surface, color, (cfg.WINDOW_W // 2,
                                                cfg.WINDOW_H // 2 + 30), r)
        # 右上角暖色晕染：让画面不只是一块冷灰
        warm = colors['start_warm']
        for i in range(18, 0, -1):
            r = int(420 * i / 18)
            t = 1.0 - (i / 18.0)
            pygame.draw.circle(surface, _blend(colors['start_bg'], warm,
                                               0.28 * t * t),
                               (cfg.WINDOW_W - 60, 40), r)
        # 顶部强调带（与棋盘页顶栏呼应）
        band = pygame.Surface((cfg.WINDOW_W, 4))
        for x in range(cfg.WINDOW_W):
            t = x / max(1, cfg.WINDOW_W - 1)
            band.fill(_blend(colors['accent_soft'], colors['accent'], t),
                      pygame.Rect(x, 0, 1, 4))
        surface.blit(band, (0, 0))

    # ---- 卡片：投影 + 圆角 + 细边 ----
    def _draw_card(self, surface):
        cfg = self.cfg
        colors = cfg.COLORS
        pygame.draw.rect(surface, colors['card_shadow'], self.card.move(0, 10),
                         border_radius=cfg.RADIUS_L)
        pygame.draw.rect(surface, colors['panel_bg'], self.card,
                         border_radius=cfg.RADIUS_L)
        pygame.draw.rect(surface, colors['panel_line'], self.card, 1,
                         border_radius=cfg.RADIUS_L)

    # ---- 标题 / 副标题 / 分隔线 ----
    def _draw_header(self, surface):
        cfg = self.cfg
        colors = cfg.COLORS
        tfont = load_font(cfg.FONT_OVER_HINT + 14)
        sfont = load_font(cfg.FONT_BODY)
        # 标题：红黑两半，视觉上就是"一方一色"
        left = tfont.render("揭", True, colors['strip_red'])
        right = tfont.render(" 棋", True, colors['strip_black'])
        total = left.get_width() + right.get_width()
        x = self.title_pos[0] - total // 2
        y = self.title_pos[1] - left.get_height() // 2
        surface.blit(left, (x, y))
        surface.blit(right, (x + left.get_width(), y))

        sub = sfont.render("中国象棋 · 暗子对局 · 每方各挑一个引擎", True,
                           colors['text_dim'])
        surface.blit(sub, sub.get_rect(center=self.sub_pos))

        # 分隔线：中间亮、两端淡
        line = pygame.Surface((self.card.width - 60, 2), pygame.SRCALPHA)
        for i in range(line.get_width()):
            edge = min(i, line.get_width() - 1 - i) / (line.get_width() / 2.0)
            alpha = int(150 * min(1.0, edge * 2.2))
            line.fill((*colors['accent'], alpha), pygame.Rect(i, 0, 1, 2))
        surface.blit(line, (self.card.x + 30, self.divider_y))

    # ---- 左列：难度 + 每方的对局方式 + 自动保存 ----
    def _draw_left(self, surface):
        cfg = self.cfg
        colors = cfg.COLORS
        tfont = load_font(cfg.FONT_SMALL)
        nfont = load_font(cfg.FONT_TINY)

        for key, text, target in (
                ('difficulty', "AI 难度", self.difficulty_chips['easy']),
                ('engine_red', "红方引擎",
                 (self.engine_chips.get('red') or {}).get(_engines.HUMAN)),
                ('engine_black', "黑方引擎",
                 (self.engine_chips.get('black') or {}).get(_engines.HUMAN)),
                ('llm', "大模型点评", self.llm_toggle),
                ('auto', "自动保存", self.auto_toggle)):
            rect = self.row_labels[key]
            surf = tfont.render(text, True, colors['text_dim'])
            # 与右侧控件**垂直居中**对齐（标签右对齐，紧贴控件）
            y = (target.centery - surf.get_height() // 2) if target is not None \
                else rect.y
            surface.blit(surf, (rect.right - surf.get_width(), y))

        hover_rect = self.hovered()
        for name, _label, _hint in DIFFICULTY_ITEMS:
            cmd, rect, label, active = self.difficulty_button(name)
            self._chip(surface, rect, label, active=active,
                       hover=rect is hover_rect)

        # 难度副标题：优先显示"鼠标悬停的那一档"，否则显示当前档
        hovered_name = None
        for name, _l, _h in DIFFICULTY_ITEMS:
            if self.difficulty_chips[name] is hover_rect:
                hovered_name = name
        hint = DIFFICULTY_HINT.get(hovered_name or self.difficulty, '')
        surf = nfont.render("当前：%s　%s" % (
            next((lab for n, lab, _h in DIFFICULTY_ITEMS
                  if n == self.difficulty), self.difficulty), hint),
            True, colors['text_faint'])
        surface.blit(surf, (self.row_notes['difficulty'].x,
                            self.row_notes['difficulty'].y))

        # 每方几枚 chip：人工输入 / 本机引擎 / 大模型。
        # 用不了的档位**画成灰的**并在下面一行说明原因——"点了没反应"是最
        # 让人困惑的状态，宁可明说原因。
        for side, cn in (('red', '红'), ('black', '黑')):
            chips = self.engine_chips.get(side) or {}
            current = self.side_engines.get(side, _engines.HUMAN)
            note_rect = self.row_notes['engine_%s' % side]
            note_text = self.engine_note(side)
            note_bad = not _engines.available(current)
            for source in _engines.SOURCES:
                rect = chips.get(source)
                if rect is None:
                    continue
                usable = _engines.available(source)
                hover = rect is hover_rect
                # 悬停某一档时，下面那行说明就改讲**那一档**（鼠标移开再还原）
                if hover:
                    note_text = self.side_engine_note(side, source)
                    note_bad = not usable
                self._chip(surface, rect, _engines.CHIP_LABELS[source],
                           active=(source == current), hover=hover,
                           enabled=usable)
            surface.blit(nfont.render(_clip(nfont, note_text, note_rect.width),
                                      True,
                                      colors['text_warn'] if note_bad
                                      else colors['text_faint']),
                         note_rect.topleft)

        # 大模型点评：与"对局方式=大模型"是**两件事**，都要有：
        #   * 点评 = 让模型对每一步说几句话（不影响选着），可以单独开；
        #   * 对局方式 = 让某一方**完全采用**模型给的招法。
        self._chip(surface, self.llm_toggle,
                   "大模型点评：%s" % ('开' if self.llm_on else '关'),
                   active=self.llm_on, hover=self.llm_toggle is hover_rect)
        note = self.llm_note or "点评只针对轮到走棋那一方的招法"
        surface.blit(nfont.render(_clip(nfont, note,
                                        self.row_notes['llm'].width),
                                  True,
                                  colors['text_ok'] if self.llm_on
                                  else colors['text_faint']),
                     self.row_notes['llm'].topleft)

        self._chip(surface, self.auto_toggle,
                   "自动保存棋谱：%s" % ('开' if self.auto_save else '关'),
                   active=self.auto_save, hover=self.auto_toggle is hover_rect)
        save_note = ("对局结束时自动存进 Game/records/"
                     if self.auto_save else "已关闭：仍可用 S 键或「导出棋谱」手动存")
        surface.blit(nfont.render(_clip(nfont, save_note,
                                        self.row_notes['auto'].width),
                                  True, colors['text_faint']),
                     self.row_notes['auto'].topleft)

    # ---- 右列：阵营（选）+ 开始游戏 + 继续 / 载入 ----
    def _draw_right(self, surface):
        cfg = self.cfg
        colors = cfg.COLORS
        bfont = load_font(cfg.FONT_BODY)
        nfont = load_font(cfg.FONT_TINY)
        hover_rect = self.hovered()

        # 阵营：两枚 chip（选"我执哪一方"），选中的那枚描金
        # 开始游戏：**一个**按钮。阵营不占界面位置（默认红方先手），
        # 想换边就改配置里的 `ai.player_side` 或中局按 `C`。
        self._wide_button(surface, self.start_button, "开始游戏", enabled=True,
                          hover=self.start_button is hover_rect,
                          primary=True)

        # 继续上一局（没有未完成的对局时灰掉，并把原因写在按钮右边）
        enabled = self.has_unfinished
        self._wide_button(surface, self.continue_button,
                          "继续上一局%s" % ("（%d 手）" % self._unfinished_plies
                                            if enabled else ""),
                          enabled=enabled,
                          hover=self.continue_button is hover_rect,
                          trailing='' if enabled else "暂时没有未完成的对局")
        self._wide_button(surface, self.import_button,
                          "载入棋谱…（%d 份已保存）" % self.record_count,
                          enabled=True, hover=self.import_button is hover_rect)

    # ---- 底部提示 ----
    def _draw_footer(self, surface):
        colors = self.cfg.COLORS
        font = load_font(self.cfg.FONT_TINY)
        text = "红/黑两方各挑一个引擎：选谁就完全采用谁的招法　·　1~4 切难度　C 换边　S 导出　I 导入"
        surf = font.render(text, True, colors['text_faint'])
        surface.blit(surf, surf.get_rect(center=self.footer_pos))

    # ---- 控件绘制 ----
    def _chip(self, surface, rect, label, active=False, hover=False,
              enabled=True):
        """圆角开关/chip：选中描金，未选中是深灰描边。"""
        cfg = self.cfg
        colors = cfg.COLORS
        if not enabled:
            bg = colors['btn_disabled']
        elif active:
            bg = colors['btn_bg_active']
        elif hover:
            bg = colors['btn_bg_hover']
        else:
            bg = colors['btn_bg']
        pygame.draw.rect(surface, bg, rect, border_radius=cfg.RADIUS_S)
        pygame.draw.rect(surface, colors['btn_border'], rect, 1,
                         border_radius=cfg.RADIUS_S)
        fg = colors['btn_text_active'] if active else colors['btn_text']
        if not enabled:
            fg = colors['text_faint']
        font = load_font(cfg.FONT_CHIP)
        surf = font.render(label, True, fg)
        surface.blit(surf, surf.get_rect(center=rect.center))

    def _wide_button(self, surface, rect, label, enabled=True, hover=False,
                     trailing='', primary=False):
        """宽按钮（继续 / 载入 / 开始游戏）。

        ``primary=True`` 时用**主色**并把字居中：那是这一页最重要的动作
        （"开始游戏"），要比旁边两个次要按钮显眼。
        """
        cfg = self.cfg
        colors = cfg.COLORS
        if not enabled:
            bg, fg = colors['btn_disabled'], colors['text_faint']
        elif primary:
            bg = colors['accent'] if not hover else colors['accent_soft']
            fg = colors['text_main']
        elif hover:
            bg, fg = colors['btn_bg_hover'], colors['text_main']
        else:
            bg, fg = colors['panel_card_alt'], colors['text_main']
        pygame.draw.rect(surface, bg, rect, border_radius=cfg.RADIUS_S)
        pygame.draw.rect(surface, colors['btn_border'], rect, 1,
                         border_radius=cfg.RADIUS_S)
        font = load_font(cfg.FONT_BODY if primary else cfg.FONT_SMALL)
        surf = font.render(label, True, fg)
        if primary and not trailing:
            surface.blit(surf, surf.get_rect(center=rect.center))
        else:
            surface.blit(surf, (rect.x + 14,
                                rect.centery - surf.get_height() // 2))
        if trailing:
            note = load_font(cfg.FONT_TINY).render(trailing, True,
                                                   colors['text_faint'])
            surface.blit(note, (rect.right - 14 - note.get_width(),
                                rect.centery - note.get_height() // 2))
        elif enabled:
            arrow = font.render("→", True, colors['accent'])
            surface.blit(arrow, (rect.right - 14 - arrow.get_width(),
                                 rect.centery - arrow.get_height() // 2))


def math_hypot(a: float, b: float) -> float:
    return (a * a + b * b) ** 0.5


def _clip(font, text: str, max_width: int) -> str:
    """把一行文字裁到 ``max_width`` 内（超长加省略号）。"""
    text = str(text)
    if font.size(text)[0] <= max_width:
        return text
    while text and font.size(text + "…")[0] > max_width:
        text = text[:-1]
    return text + "…"
