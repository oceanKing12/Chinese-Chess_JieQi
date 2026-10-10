"""界面重做后的排版 / 配色 / 按钮几何回归测试（无显示器）。

这些断言锁的是"排版不再退回到随手写坐标"这件事：

* 配色令牌齐全（重做调色板时最容易漏掉某个键，让某一处变成黑块）；
* 三个区域仍然严丝合缝地铺满窗口；
* 顶部配置区三行**互不重叠**（旧版就是行与行叠字）；
* 面板里每张卡片都完整落在面板内，卡片之间也不重叠；
* 每个按钮都能被 ``button_at`` 命中，而且命中的命令和画出来的文案一致——
  "画在这里、点在那里"是这类手写坐标界面最典型的 bug。
"""
import os
import sys
import unittest
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(GAME_DIR.parent))
sys.path.insert(0, str(GAME_DIR))

import pygame  # noqa: E402

from resource_manager import ResourceManager, load_font  # noqa: E402

_ORIGINAL_CWD = os.getcwd()

#: 重做后的配色令牌必须全部存在（渲染代码会直接下标取用）
REQUIRED_COLORS = (
    "red", "black", "board_top", "board_bottom", "grid", "check",
    "text_red", "text_black", "river_text",
    "selection_primary", "selection_highlight", "selection_secondary",
    "window_bg", "bar_bg", "bar_bg_alt", "panel_bg", "panel_card",
    "panel_card_alt", "panel_line", "panel_line_soft", "card_shadow",
    "btn_bg", "btn_bg_hover", "btn_bg_active", "btn_text",
    "btn_text_active", "btn_disabled", "btn_border",
    "text_main", "text_dim", "text_faint", "text_llm", "text_warn",
    "text_ok", "accent", "accent_soft", "red_side", "black_side",
    "last_move", "hint_dot", "hint_capture", "coord_text",
)


class TestUiStyle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(_ORIGINAL_CWD)

    def setUp(self):
        from game_state import GameState
        from renderer import Renderer

        pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        pygame.display.set_mode((1, 1))
        rm = ResourceManager()
        rm.load_resources()
        self.gs = GameState(rm)
        self.gs.show_side_selection = False
        self.gs.enable_ai('black')
        self.r = Renderer(rm, self.gs)
        self.cfg = self.r.config

    # ---------------------------------------------------------- 配色 / 令牌
    def test_palette_has_every_required_key(self):
        missing = [k for k in REQUIRED_COLORS if k not in self.cfg.COLORS]
        self.assertEqual(missing, [], f"配色令牌缺失：{missing}")
        for key, value in self.cfg.COLORS.items():
            self.assertEqual(len(value), 3, f"{key} 不是 RGB 三元组")
            for channel in value:
                self.assertIsInstance(channel, int, f"{key} 分量不是整数")
                self.assertTrue(0 <= channel <= 255, f"{key} 分量越界：{channel}")

    def test_layout_tokens_are_ordered(self):
        """间距令牌必须是递增的，否则排版会自相矛盾。"""
        pads = [self.cfg.PAD_XS, self.cfg.PAD_S, self.cfg.PAD,
                self.cfg.PAD_L, self.cfg.PAD_XL]
        self.assertEqual(pads, sorted(pads), f"间距令牌未递增：{pads}")
        radii = [self.cfg.RADIUS_S, self.cfg.RADIUS, self.cfg.RADIUS_L]
        self.assertEqual(radii, sorted(radii), f"圆角令牌未递增：{radii}")

    def test_font_sizes_are_usable(self):
        """每个字号都要能真的渲染出中文，且高度在合理范围。"""
        sizes = [self.cfg.FONT_BRAND, self.cfg.FONT_TITLE, self.cfg.FONT_BODY,
                 self.cfg.FONT_SMALL, self.cfg.FONT_CHIP, self.cfg.FONT_MOVE,
                 self.cfg.FONT_TINY, self.cfg.FONT_OVER, self.cfg.FONT_OVER_HINT]
        for size in sizes:
            font = load_font(size)
            surf = font.render("揭棋难度评估", True, (255, 255, 255))
            self.assertGreater(surf.get_width(), size, f"字号 {size} 渲染异常")
            self.assertGreater(surf.get_height(), 0)

    # ------------------------------------------------------------ 三区几何
    def test_top_bar_rows_do_not_overlap(self):
        """顶部配置区两行必须依次往下排，不能叠字（旧版的核心毛病）。

        顶栏现在是**两行**：第一行控件 + AI 走棋信息（用时/深度），第二行对局状态
        + AI 评估。原来那第三行「键盘快捷键提示」已经去掉，toast 改压在棋盘底部。
        """
        cfg = self.cfg
        self.assertFalse(hasattr(cfg, 'TOP_ROW3_Y'),
                         "顶栏第三行已删除，不该再有 TOP_ROW3_* 常量")
        r1 = pygame.Rect(0, cfg.TOP_ROW1_Y, cfg.WINDOW_W, cfg.TOP_ROW1_H)
        r2 = pygame.Rect(0, cfg.TOP_ROW2_Y, cfg.WINDOW_W, cfg.TOP_ROW2_H)
        self.assertLessEqual(r1.bottom, r2.top, "第一行和第二行重叠")
        self.assertLessEqual(r2.bottom, cfg.TOP_BAR_H, "第二行溢出了配置区")

    def test_ai_info_never_collides_with_buttons(self):
        """顶栏两行的文字区都必须夹在**没有任何按钮**的空档里。

        难度与大模型开关搬到启动页之后，顶栏第一行整条都空出来了；第二行右侧
        是四个翻手按钮。所以这里改成量"文字区不与任何按钮相交"，比写死
        "夹在专家和应用招法之间"更贴近真正要保证的事（不叠字、不压按钮）。
        """
        r = self.r
        info = r.ai_info_rect
        self.assertGreater(info.width, 60, "AI 信息区太窄，什么都显示不了")
        self.assertLessEqual(info.right, r.config.WINDOW_W - r.config.PAD_L)
        eval_rect = r.ai_eval_rect
        self.assertGreater(eval_rect.width, 80, "AI 评估区太窄")
        buttons = [(cmd, rect) for cmd, rect, _t, _a in r.nav_buttons()]
        for cmd, rect in buttons:
            self.assertFalse(info.colliderect(rect), f"AI 信息压到了 {cmd}")
            self.assertFalse(eval_rect.colliderect(rect), f"AI 评估压到了 {cmd}")
        # 第二行从左到右：状态区 → toast → 评估 → 翻手按钮，互不重叠
        self.assertLessEqual(r.toast_rect.right, eval_rect.x + 1,
                             "toast 与 AI 评估重叠")
        self.assertLessEqual(eval_rect.right, buttons[0][1].x,
                             "AI 评估与翻手按钮重叠")

    def test_top_bar_has_no_config_buttons(self):
        """顶栏只报状态：难度 chip 与"对局方式"chip 必须已经搬到启动页。

        用户要求"棋盘页也可以精简一点"，所以这两组控件**不该**再出现在顶栏；
        它们在启动页上仍然可用（``test_start_screen_*`` 覆盖）。
        """
        r = self.r
        for _cmd, rect, _t, _a in r.difficulty_buttons():
            self.assertFalse(r.top_rect.contains(rect),
                             "难度按钮还在顶栏上（应当只在启动页）")
        for _cmd, rect, _t, _a in r.engine_buttons():
            self.assertFalse(r.top_rect.contains(rect),
                             "对局方式 chip 还在顶栏上（应当只在启动页）")
        # 棋盘页的按钮清单里也不该再出现它们
        board_cmds = [cmd for cmd, _r, _t, _a in r._all_buttons()]
        self.assertNotIn('llm:toggle', board_cmds)
        self.assertNotIn('llm:apply', board_cmds)
        self.assertFalse(any(cmd.startswith('engine:') for cmd in board_cmds))
        self.assertFalse(any(cmd.startswith('difficulty:') for cmd in board_cmds))

    def test_nav_buttons_live_in_the_top_bar(self):
        """四个翻手按钮在顶栏第二行（从面板搬上来，给面板瘦身）。"""
        r = self.r
        labels = [text for _c, _r, text, _a in r.nav_buttons()]
        self.assertEqual(labels, ['开局', '上一回合', '下一回合', '最新'])
        for cmd, rect, _t, _a in r.nav_buttons():
            self.assertTrue(r.top_rect.contains(rect), f"{cmd} 不在顶栏：{rect}")
        # 两两不重叠（否则会有一个点不到）
        rects = [(c, rect) for c, rect, _t, _a in r.nav_buttons()]
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                self.assertFalse(rects[i][1].colliderect(rects[j][1]),
                                 f"{rects[i][0]} 与 {rects[j][0]} 重叠")

    def test_panel_cards_fit_and_do_not_overlap(self):
        """面板里每张卡片都要完整落在面板内，且互不重叠。

        面板现在只有三块：操作卡 + 走棋记录 + 大模型点评。
        「对局状态」「引擎评估」两张卡已删除（顶栏已经写了同样的信息）。
        """
        panel = self.r.panel_rect
        self.gs.ai_comment = "测试点评" * 10
        self.gs.ai_eval_text = "AI 用时 100ms\n深度 4　2 采样　3 节点\nAI 评估：+12（均势）"
        lay = self.r._panel_layout()
        self.assertNotIn('status', lay, "「对局状态」卡应该已经删除")
        self.assertNotIn('eval', lay, "「引擎评估」卡应该已经删除")
        self.assertNotIn('header', lay, "「棋局信息」标题应该已经删除")
        cards = [lay['action'], lay['log']]
        self.assertIsNotNone(lay['comment'], "有点评时必须给出点评卡")
        cards.append(lay['comment'])
        for card in cards:
            self.assertGreaterEqual(card.left, 0, f"卡片越出左边界：{card}")
            self.assertGreaterEqual(card.top, 0, f"卡片越出上边界：{card}")
            self.assertLessEqual(card.right, panel.width, f"卡片越出右边界：{card}")
            self.assertLessEqual(card.bottom, panel.height, f"卡片越出下边界：{card}")
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                self.assertFalse(cards[i].colliderect(cards[j]),
                                 f"卡片重叠：{cards[i]} vs {cards[j]}")

    def test_action_card_contains_all_its_buttons(self):
        """操作卡里的每一行按钮都必须真的在卡内（少算 1px 就会溢出去）。

        这一版操作卡**两行**：导出/导入 → 回顾/重开，外加人工打分那一行。
        翻手按钮已搬到顶栏（``test_nav_buttons_live_in_the_top_bar`` 覆盖），
        「回退一回合」和「让 AI 替我走这步」两个按钮按需求去掉，
        「研究 / 保存分支」随分支棋谱功能一起移除。"""
        card = self.r._panel_layout()['action']
        local = lambda rect: rect.move(-self.r.panel_rect.x, -self.r.panel_rect.y)
        buttons = [self.r.record_button(), self.r.import_button(),
                   self.r.replay_button(), self.r.restart_button()]
        buttons.extend(self.r.score_buttons())
        for cmd, rect, _text, _active in buttons:
            self.assertTrue(card.contains(local(rect)),
                            f"{cmd} 不在操作卡内：{local(rect)} vs {card}")

    def test_action_card_rows_do_not_overlap(self):
        """各行按钮**互不重叠**（自己算坐标最容易出这种事）。"""
        a1, a2, c1, c2, scores = self.r._action_rows()
        boxes = [a1, a2, c1, c2] + list(scores)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                self.assertFalse(boxes[i].colliderect(boxes[j]),
                                 "第 %d 个和第 %d 个按钮重叠：%s / %s"
                                 % (i, j, boxes[i], boxes[j]))

    def test_score_buttons_cover_five_levels_plus_clear(self):
        """评分行 = 五档 + 清除，命令形如 ``score:blunder`` / ``score:``。"""
        buttons = self.r.score_buttons()
        self.assertEqual(len(buttons), 6)
        cmds = [cmd for cmd, _r, _t, _a in buttons]
        self.assertEqual(cmds, ["score:best", "score:good", "score:inaccuracy",
                                "score:mistake", "score:blunder", "score:"])
        labels = [text for _c, _r, text, _a in buttons]
        self.assertEqual(labels, ["好棋", "正着", "疑问手", "错着", "漏着", "清除"])

    def test_score_buttons_disabled_at_opening(self):
        """开局没有可评的着法 → 评分按钮要灰掉（不能让人打一个不存在的分）。"""
        self.gs.ply = 0
        for _cmd, _rect, _text, active in self.r.score_buttons():
            self.assertFalse(active, "开局时评分按钮不该可用")

    def test_comment_height_shrinks_to_zero_without_comment(self):
        self.gs.ai_comment = ""
        self.assertEqual(self.r._comment_height(), 0)
        self.gs.ai_comment = "一句点评。"
        self.assertGreater(self.r._comment_height(), 0)

    # --------------------------------------------------------- 按钮命中测试
    def test_every_button_is_clickable(self):
        """画出来的每个按钮都必须能被命中，且命令与文案对得上。

        棋盘页与启动页是两套按钮，各自在**自己那个界面**上测。
        """
        r = self.r
        gs = self.gs
        gs.game_over = True            # 禁用态也要能走一遍命中
        gs.game_over = False
        cases = [r.record_button(), r.import_button(), r.replay_button(),
                 r.restart_button()]
        cases.extend(r.nav_buttons())
        for cmd, rect, text, _state in cases:
            self.assertIsNotNone(text, f"{cmd} 没有文案")
            self.assertTrue(rect.width > 0 and rect.height > 0, f"{cmd} 尺寸异常")
            self.assertEqual(r.button_at(rect.center), cmd,
                             f"{cmd} 在自己的中心点命不中（文案：{text}）")

    def test_start_screen_buttons_are_clickable(self):
        """启动页上的每个控件（含新加的难度/大模型/自动保存）都要能点中。"""
        r = self.r
        gs = self.gs
        gs.show_side_selection = True
        try:
            layout = r.start_layout()
            for cmd, rect, text, _active in r.start_screen_buttons():
                self.assertIsNotNone(text, f"{cmd} 没有文案")
                self.assertTrue(rect.width > 0 and rect.height > 0,
                                f"{cmd} 尺寸异常")
                if layout.disabled(cmd):
                    continue        # 灰掉的控件点了不生效，这是有意的
                self.assertEqual(r.button_at(rect.center), cmd,
                                 f"{cmd} 在启动页中心点命不中（文案：{text}）")
            # 启动页上的控件必须都在窗口里（放不下就会被切掉，点不到）
            window = pygame.Rect(0, 0, r.config.WINDOW_W, r.config.WINDOW_H)
            for cmd, rect, _t, _a in r.start_screen_buttons():
                self.assertTrue(window.contains(rect), f"{cmd} 超出窗口：{rect}")
        finally:
            gs.show_side_selection = False

    def test_start_screen_offers_the_new_settings(self):
        """启动页要提供用户点名的那几项：难度、每方的对局方式、载入棋谱、阵营。

        "三种对局方式"现在是**每方一行四档 chip**（人工/本机引擎/外部引擎/
        大模型），命令是 ``engine:<side>:<source>``；原来独立的
        ``llm:toggle`` / ``llm:apply`` 两个开关已经并进"大模型"这一档。
        """
        r = self.r
        self.gs.show_side_selection = True
        try:
            cmds = [cmd for cmd, _r, _t, _a in r.start_screen_buttons()]
            for want in ('difficulty:easy', 'difficulty:expert', 'setting:autosave',
                         'game:start', 'record:import', 'game:continue'):
                self.assertIn(want, cmds, f"启动页缺少 {want}")
            # 阵营选项已按用户要求去掉（"太啰嗦了，有需要我会自己调的"）：
            # 换边走 ai_config.json 的 ai.player_side 或中局热键 C。
            self.assertFalse(any(c.startswith('side:') for c in cmds),
                             "启动页不该再有阵营按钮：%s" % cmds)
            for side in ('red', 'black'):
                for source in ('human', 'python', 'llm'):
                    want = f'engine:{side}:{source}'
                    self.assertIn(want, cmds, f"启动页缺少 {want}")
        finally:
            self.gs.show_side_selection = False

    def test_start_screen_disables_unusable_engines(self):
        """用不了的档位必须**灰掉**并给得出原因。

        外部引擎那一档已经整体移除，所以这里只剩"大模型模块不可用"这一种
        灰档情形。灰掉逻辑本身仍然必须有：它保证"点了没反应"不会发生。
        """
        from jieqi import engines as _engines

        r = self.r
        self.gs.show_side_selection = True
        try:
            layout = r.start_layout()
            self.assertEqual(layout.disabled('engine:red:human'), False)
            self.assertEqual(layout.disabled('engine:red:python'), False)
            self.assertEqual(layout.disabled('engine:red:llm'), False,
                             "大模型模块在，这一档应当可用")
            # 可用档位不该给出"为什么灰掉"的理由
            self.assertEqual(_engines.reason('human'), '')
            self.assertEqual(_engines.reason('python'), '')
        finally:
            self.gs.show_side_selection = False

    def test_removed_buttons_are_gone(self):
        """按需求删掉的功能：不许再出现「回退一回合」「让 AI 替我走这步」按钮。"""
        for name in ('undo_button', 'ai_move_for_me_button',
                     'ai_move_for_me_half_button'):
            self.assertFalse(hasattr(self.r, name), f"{name} 应该已经删除")
        texts = [text for _c, _r, text, _a in self.r._all_buttons()]
        self.assertNotIn("回退一回合（不访问模型）", texts)
        self.assertNotIn("让 AI 替我走这步", texts)
        self.assertIn("回顾棋局", texts)
        self.assertIn("导入棋谱", texts)

    def test_board_center_never_hits_a_button(self):
        """棋盘正中间不该命中任何按钮，否则玩家没法落子。"""
        for pos in (self.r.board_rect.center,
                    (self.r.board_rect.centerx, self.r.board_rect.y + 40),
                    (self.r.board_rect.x + 40, self.r.board_rect.centery)):
            self.assertIsNone(self.r.button_at(pos), pos)

    def test_buttons_inside_their_regions(self):
        """棋盘页的按钮在面板/顶栏里；启动页的控件在整个窗口里。"""
        r = self.r
        for getter in (r.record_button, r.import_button,
                       r.replay_button, r.restart_button):
            _c, rect, _t, _a = getter()
            self.assertTrue(r.panel_rect.contains(rect), rect)
        for _c, rect, _t, _a in r.nav_buttons():
            self.assertTrue(r.top_rect.contains(rect), rect)
        self.gs.show_side_selection = True
        try:
            window = pygame.Rect(0, 0, r.config.WINDOW_W, r.config.WINDOW_H)
            for _c, rect, _t, _a in r.engine_buttons():
                self.assertTrue(window.contains(rect), rect)
            for _c, rect, _t, _a in r.difficulty_buttons():
                self.assertTrue(window.contains(rect), rect)
            for _c, rect, _t, _a in r.start_screen_buttons():
                self.assertTrue(window.contains(rect), rect)
        finally:
            self.gs.show_side_selection = False

    # ------------------------------------------------------------- 渲染冒烟
    def test_all_screens_render_without_error(self):
        """阵营选择 / 对局中 / 结算（含点评与长记录）都要能画出来。"""
        gs = self.gs
        gs.show_side_selection = True
        self.r.draw_all()
        gs.show_side_selection = False
        gs.toast = "提示" * 40          # 超长提示必须被裁剪，不能压到快捷键
        gs.ai_eval_text = "AI 评估：+12（均势）\n深度 4　2 采样　3 节点\nAI 用时 100ms"
        gs.ai_comment = "很长的点评。" * 60
        self.r.draw_all()
        for winner in ('red', 'black', 'draw'):
            gs.game_over = True
            gs.winner = winner
            gs.draw_reason = '测试' if winner == 'draw' else None
            self.r.draw_all()
        gs.game_over = False
        gs.winner = None
        gs.draw_reason = None

    def test_selection_and_last_move_markers_render(self):
        """选中框、可走点、上一步角标都要能画（这些是新增的棋盘叠加层）。"""
        from jieqi.gui_adapter import legal_moves

        moves = legal_moves(self.gs.board, 0)      # 红方
        mv = moves[0]
        self.gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                           (mv & 127) // 9, (mv & 127) % 9)
        self.gs.selected = (self.gs.move_history[-1]['end'])
        self.gs.animations.clear()
        self.r.draw_all()
        targets = self.r._legal_targets()
        self.assertIsInstance(targets, set)

    def test_ai_plies_marks_only_ai_moves(self):
        """走棋记录里的"AI 色条"只能标在 AI 走过的手上。"""
        from jieqi.gui_adapter import legal_moves

        gs = self.gs
        for _ in range(4):
            side = gs.current_player
            moves = legal_moves(gs.board, 0 if side == 'red' else 1)
            if not moves:
                break
            mv = moves[0]
            gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                          (mv & 127) // 9, (mv & 127) % 9)
            gs.current_player = 'black' if side == 'red' else 'red'
        ai = self.r._ai_plies()
        for ply, rec in enumerate(gs.move_history, start=1):
            color = (rec.get('piece') or {}).get('color')
            if color == gs.ai_color:
                self.assertIn(ply, ai, f"第 {ply} 手是 AI 走的，却没被标记")
            else:
                self.assertNotIn(ply, ai, f"第 {ply} 手是玩家走的，却被标成 AI")


if __name__ == "__main__":
    unittest.main()
