# -*- coding: utf-8 -*-
"""对局方式（每方一个引擎档位）的回归测试。

需求原话："支持切换自己的引擎，对方的引擎，大模型三种对局方式，
切换到哪种就完全应用那个引擎的招法。"

这里锁四件事：

1. ``GameState.side_engines`` 是唯一的真相来源，``ai_mode`` / ``ai_color``
   只是**派生**属性（老代码与老测试读它们照样对，但不会再各说各话）；
2. 档位的可用性：外部引擎被完全关闭时**灰掉**，而且 `cycle` 不会停在灰档上
   （"按了没效果"是最糟的体验）；
3. 主循环的按方判定：轮到哪一方就用哪一方的档位（不再依赖"AI 只执一方"）；
4. 每一手都记下"当时那一方用的是什么"，复盘时才说得清"这手是谁下的"。
"""
import os
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

GAME_DIR = Path(__file__).resolve().parents[1] / "Game"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
for _p in (str(GAME_DIR.parent), str(GAME_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pygame  # noqa: E402

from jieqi import engines as _engines  # noqa: E402
from jieqi.gui_adapter import gui_color_to_side, legal_moves  # noqa: E402


def make_gs(plies=0):
    from game_state import GameState

    random.seed(11)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = 'red'
    gs.current_player = 'red'
    rng = random.Random(11)
    for _ in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        mv = rng.choice(moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.current_player = 'black' if side == 0 else 'red'
    return gs


class TestEngineRegistry(unittest.TestCase):
    """``jieqi.engines`` 这层：档位定义、可用性、循环。"""

    def test_three_sources_and_defaults(self):
        self.assertEqual(set(_engines.SOURCES), {'human', 'python', 'llm'})
        self.assertEqual(_engines.DEFAULT_FOR_SIDE['red'], 'human')
        self.assertEqual(_engines.DEFAULT_FOR_SIDE['black'], 'python')

    def test_normalize_aliases_and_garbage(self):
        self.assertEqual(_engines.normalize('HUMAN'), 'human')
        self.assertEqual(_engines.normalize('manual'), 'human')
        self.assertEqual(_engines.normalize('model'), 'llm')
        # 认不出来要**退回**默认，而不是抛异常（配置文件是手写的）
        self.assertEqual(_engines.normalize(''), 'human')
        self.assertEqual(_engines.normalize(None), 'human')
        self.assertEqual(_engines.normalize('nonsense'), 'human')
        self.assertEqual(_engines.normalize('nonsense', fallback='python'),
                         'python')

    def test_removed_external_engine_downgrades_to_python(self):
        """老配置里的 ``abjchess`` 一律折算回本机引擎（不报错、也不静默变人工）。

        外部引擎已**整体移除**，但玩家 `ai_config.json` 里可能还写着它 ——
        老配置不该让游戏起不来，也不该悄悄变成别的语义。
        """
        for legacy in ('abjchess', 'abjchess_then_python', 'abj', 'ABJCHESS'):
            with self.subTest(legacy=legacy):
                self.assertEqual(_engines.normalize(legacy), 'python')

    def test_availability(self):
        # 人工与本机引擎永远可用
        self.assertTrue(_engines.available('human'))
        self.assertTrue(_engines.available('python'))
        self.assertEqual(_engines.reason('human'), '')
        self.assertEqual(_engines.reason('python'), '')
        # 大模型：模块在就可用（配没配 key 由启动页那行说明负责）
        from jieqi import llm as _llm  # noqa: F401
        self.assertTrue(_engines.available('llm'))

    def test_cycle_never_lands_on_a_disabled_source(self):
        """``E``/``T`` 切档位时**不许**停在灰掉的档位上。

        否则玩家按一下"没反应"（其实只是选中了一个不可用的档位），
        这是最难查的一类交互 bug。这里直接把 ``available`` 换成"只有
        human/python 可用"来造出有灰档的情形（比 mock import 更直接，
        不会误伤别处的导入）。
        """
        with mock.patch.object(_engines, 'available',
                               side_effect=lambda s: _engines.normalize(s)
                               in ('human', 'python')):
            seen = set()
            current = 'human'
            for _ in range(8):
                current = _engines.cycle(current)
                self.assertTrue(_engines.available(current),
                                "循环停在了不可用的档位 %r" % current)
                seen.add(current)
            self.assertEqual(seen, {'human', 'python'})
            self.assertEqual(_engines.cycle('human', -1), 'python')

    def test_cycle_visits_all_three_sources(self):
        seen = set()
        current = 'human'
        for _ in range(3):
            current = _engines.cycle(current)
            seen.add(current)
        self.assertEqual(seen, {'python', 'llm', 'human'})

    def test_chips_for_side_marks_unusable(self):
        """chip 的"可用性"要如实反映：人工/本机永远可用，大模型看模块在不在。"""
        chips = {src: ok for src, _label, ok, _why in _engines.chips_for('red')}
        self.assertEqual(set(chips), {'human', 'python', 'llm'})
        self.assertTrue(chips['human'])
        self.assertTrue(chips['python'])
        self.assertTrue(chips['llm'], "大模型模块在，应当可用")


class TestGameStateSideEngines(unittest.TestCase):
    """``GameState`` 上的每方档位与派生属性。"""

    def test_default_is_human_vs_python(self):
        gs = make_gs()
        self.assertEqual(gs.side_engine('red'), 'human')
        self.assertEqual(gs.side_engine('black'), 'python')
        self.assertTrue(gs.ai_mode)
        self.assertEqual(gs.ai_color, 'black')
        self.assertEqual(gs.engine_sides(), ['black'])
        self.assertTrue(gs.is_human_side('red'))
        self.assertFalse(gs.is_human_side('black'))

    def test_both_sides_engine_is_still_ai_mode(self):
        """红黑各配一个引擎（自动对弈）也算"AI 模式"。"""
        gs = make_gs()
        gs.set_side_engine('red', 'llm')
        self.assertTrue(gs.ai_mode)
        self.assertEqual(sorted(gs.engine_sides()), ['black', 'red'])
        # 两方都是引擎时 ai_color 给 'black'（传统"AI 执黑"的显示习惯）
        self.assertEqual(gs.ai_color, 'black')

    def test_both_human_is_not_ai_mode(self):
        gs = make_gs()
        gs.set_side_engine('red', 'human')
        gs.set_side_engine('black', 'human')
        self.assertFalse(gs.ai_mode)
        self.assertEqual(gs.ai_color, '')
        self.assertEqual(gs.engine_sides(), [])

    def test_legacy_abjchess_setting_is_downgraded(self):
        """老配置里的 ``abjchess`` 存不进 ``side_engines``（已移除，折算成本机引擎）。

        关键是"**别让这一方卡住**"：玩家老配置写着外部引擎，那一方也必须有人下棋。
        """
        gs = make_gs()
        changed = gs.set_side_engine('red', 'abjchess')
        self.assertTrue(changed, "从 human 变成 python 也算变了")
        self.assertEqual(gs.side_engine('red'), 'python')

    def test_llm_source_is_kept(self):
        """可用的档位选了它就**真的是它**，不被降级。"""
        gs = make_gs()
        self.assertTrue(gs.set_side_engine('red', 'llm'))
        self.assertEqual(gs.side_engine('red'), 'llm')

    def test_set_side_engine_reports_change(self):
        gs = make_gs()
        self.assertFalse(gs.set_side_engine('red', 'human'), "本来就是人工，不该算变了")
        self.assertTrue(gs.set_side_engine('red', 'python'))
        self.assertFalse(gs.set_side_engine('red', 'python'))
        self.assertFalse(gs.set_side_engine('green', 'python'), "非法阵营")

    def test_legacy_enable_ai_still_works(self):
        """老接口 ``enable_ai('black')`` 的语义不变（AI 执黑、红方人工）。"""
        gs = make_gs()
        gs.set_side_engine('red', 'llm')
        gs.enable_ai('black')
        self.assertEqual(gs.side_engine('black'), 'python')
        self.assertEqual(gs.side_engine('red'), 'human')
        self.assertEqual(gs.ai_color, 'black')

    def test_legacy_direct_assignment_of_ai_color(self):
        gs = make_gs()
        gs.ai_color = 'red'
        self.assertEqual(gs.side_engine('red'), 'python')
        self.assertEqual(gs.side_engine('black'), 'human')
        gs.ai_mode = False
        self.assertFalse(gs.ai_mode)
        self.assertEqual(gs.engine_sides(), [])

    def test_reset_keeps_the_players_choice_by_default(self):
        """``reset()`` 默认**保留**对局方式：换阵营/重开不该把设置吃掉。"""
        gs = make_gs()
        gs.set_side_engine('red', 'llm')
        gs.set_side_engine('black', 'human')
        gs.reset()
        self.assertEqual(gs.side_engine('red'), 'llm')
        self.assertEqual(gs.side_engine('black'), 'human')

    def test_reset_can_go_back_to_defaults(self):
        """``reset(keep_engines=False)`` = 回到默认（红人工 / 黑本机引擎）。"""
        gs = make_gs()
        gs.set_side_engine('red', 'llm')
        gs.set_side_engine('black', 'human')
        gs.reset(keep_engines=False)
        self.assertEqual(gs.side_engine('red'), 'human')
        self.assertEqual(gs.side_engine('black'), 'human',
                         "_init_ai 的语义是'双方都人工'（暂停），不是默认对局方式")


class TestPlayerSide(unittest.TestCase):
    """玩家阵营：启动页上没有选项了，靠配置 + 热键换边。"""

    def test_config_sets_player_side(self):
        gs = make_gs()
        gs.apply_engine_settings({'player_side': 'black'})
        self.assertEqual(gs.player_side, 'black')
        # 换边之后对手的档位要跟着搬过去（默认本机引擎 → 红方本机引擎）
        self.assertEqual(gs.side_engine('black'), 'human')
        self.assertEqual(gs.side_engine('red'), 'python')

    def test_config_player_side_is_optional(self):
        for cfg in ({}, {'player_side': None}, {'player_side': 'nonsense'}):
            with self.subTest(cfg=cfg):
                gs = make_gs()
                gs.apply_engine_settings(cfg)
                self.assertEqual(gs.player_side, 'red')

    def test_config_player_side_combines_with_side_engines(self):
        gs = make_gs()
        gs.apply_engine_settings({'player_side': 'black',
                                  'side_engines': {'red': 'llm'}})
        self.assertEqual(gs.player_side, 'black')
        self.assertEqual(gs.side_engine('red'), 'llm')
        self.assertEqual(gs.side_engine('black'), 'human')


class TestEngineSettings(unittest.TestCase):
    """``ai_config.json`` 里的写法优先级（新写法 + 老写法兼容）。"""

    def test_new_style_side_engines(self):
        gs = make_gs()
        gs.apply_engine_settings({'side_engines': {'red': 'llm',
                                                   'black': 'human'}})
        self.assertEqual(gs.side_engine('red'), 'llm')
        self.assertEqual(gs.side_engine('black'), 'human')

    def test_legacy_move_source_still_works(self):
        """老配置只写 ``ai.move_source`` 时折算成"黑方用它、红方人工"。"""
        gs = make_gs()
        gs.apply_engine_settings({'move_source': 'llm'})
        self.assertEqual(gs.side_engine('black'), 'llm')
        self.assertEqual(gs.side_engine('red'), 'human')

    def test_legacy_python_keeps_the_default(self):
        gs = make_gs()
        gs.apply_engine_settings({'move_source': 'python'})
        self.assertEqual(gs.side_engines,
                         {'red': 'human', 'black': 'python'})

    def test_empty_settings_keep_the_default(self):
        for cfg in ({}, None, {'move_source': None}, {'side_engines': None}):
            with self.subTest(cfg=cfg):
                gs = make_gs()
                gs.apply_engine_settings(cfg)
                self.assertEqual(gs.side_engines,
                                 {'red': 'human', 'black': 'python'})

    def test_legacy_abjchess_setting_is_downgraded_not_stored(self):
        """老配置里配了已移除的 ``abjchess`` → 折算成本机引擎，不会卡住那一方。"""
        gs = make_gs()
        gs.apply_engine_settings({'side_engines': {'black': 'abjchess'}})
        self.assertEqual(gs.side_engine('black'), 'python')

    def test_a_whole_legacy_config_file_is_harmless(self):
        """**整份老配置**（带一堆已移除的外部引擎字段）不该让游戏起不来。

        这是"完全去掉外部引擎"最容易翻车的地方：``Game/ai_config.json``
        是**受保护文件**，重新部署永远不会覆盖玩家那一份 —— 所以线上那份
        一定还留着 ``abjchess_dir`` / ``external_verbose`` / ``external_verify``
        之类的字段。这里把当年真实配置的形状喂进去，断言：
        游戏能起来、档位全落在合法清单里、老字段一律被忽略。
        """
        legacy = {
            "difficulty": "normal",
            "time_limit": 2.0,
            "move_source": "abjchess_then_python",   # 已移除
            "abjchess_dir": r"D:\py\揭棋\_ref\abjchess-0.2c\0.2c_bmi2",
            "external_verbose": True,
            "external_verify": "engine",
            "side_engines": None,
            "player_side": "red",
        }
        gs = make_gs()
        gs.apply_engine_settings(legacy)
        for side in _engines.SIDES:
            self.assertIn(gs.side_engine(side), _engines.SOURCES,
                          "老配置把 %s 方弄成了非法档位" % side)
        # move_source 那一支折算成"黑方用它、红方人工"，而 abjchess 已折算成 python
        self.assertEqual(gs.side_engine('black'), 'python')
        self.assertEqual(gs.side_engine('red'), 'human')
        self.assertEqual(gs.player_side, 'red')

    def test_legacy_settings_dict_is_read_without_error(self):
        """``jieqi.settings.ai_settings()`` 读老配置文件不能抛异常。"""
        import json as _json
        import tempfile

        from jieqi import settings as _settings

        legacy = {"ai": {"move_source": "abjchess", "abjchess_dir": "/nope",
                         "external_verbose": True, "external_verify": "ours",
                         "difficulty": "hard"},
                  "llm": {"enabled": False}}
        path = os.path.join(tempfile.mkdtemp(prefix="jieqi-legacy-"),
                            "ai_config.json")
        with open(path, "w", encoding="utf-8") as handle:
            _json.dump(legacy, handle)
        with mock.patch.dict(os.environ, {"JIEQI_AI_CONFIG": path}, clear=False):
            got = _settings.ai_settings()
        self.assertEqual(got.get("difficulty"), "hard")
        # 已移除的字段不该出现在结果里（读的时候就不认它们）
        self.assertNotIn("abjchess_dir", got)
        self.assertNotIn("external_verify", got)

    def test_configured_llm_is_kept(self):
        """配了可用的档位就保留（不偷偷降级）。"""
        gs = make_gs()
        gs.apply_engine_settings({'side_engines': {'black': 'llm'}})
        self.assertEqual(gs.side_engine('black'), 'llm')

    def test_record_tags_each_move_with_its_engine(self):
        """每一手都要记下"当时这一方用的是什么"（复盘要说清谁下的）。"""
        gs = make_gs()
        gs.set_side_engine('red', 'python')      # 红方也交给引擎
        mv = legal_moves(gs.board, 0)[0]
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        self.assertEqual(gs.move_history[-1].get('engine'), 'python')
        # 换档位之后走的那一手记的是新档位
        gs.current_player = 'red'
        gs.set_side_engine('red', 'human')
        mv = legal_moves(gs.board, 0)[0]
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        self.assertEqual(gs.move_history[-1].get('engine'), 'human')


class TestMainLoopDispatch(unittest.TestCase):
    """主循环的按方分发（``should_auto_move`` / 档位切换）。"""

    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((1, 1))

    def _game(self):
        import main as main_mod

        # ``ChessGame`` 自己建 ResourceManager（按相对路径找 images/，
        # 棋子的 png 在 Game 目录下），所以构造时要切过去，用完切回来。
        cwd = os.getcwd()
        os.chdir(GAME_DIR)
        try:
            game = main_mod.ChessGame()
        finally:
            os.chdir(cwd)
        game.gs.show_side_selection = False
        return game

    def test_auto_move_follows_the_side_engine(self):
        game = self._game()
        gs = game.gs
        gs.set_side_engine('red', 'human')
        gs.set_side_engine('black', 'python')
        gs.current_player = 'black'
        self.assertTrue(game.should_auto_move(), "黑方是本机引擎，该自动走")
        gs.current_player = 'red'
        self.assertFalse(game.should_auto_move(), "红方是人工，不该自动走")
        # 红方也交给引擎 → 轮到红方时也该自动走
        gs.set_side_engine('red', 'python')
        self.assertTrue(game.should_auto_move())

    def test_auto_move_pauses_while_browsing(self):
        game = self._game()
        gs = game.gs
        gs.set_side_engine('black', 'python')
        mv = legal_moves(gs.board, 0)[0]
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        gs.current_player = 'black'
        self.assertFalse(gs.is_browsing)
        gs.goto_ply(0)
        self.assertTrue(gs.is_browsing)
        self.assertFalse(game.should_auto_move(), "浏览历史时不该自动走子")

    def test_switch_side_engine_takes_effect(self):
        game = self._game()
        game.set_side_engine('red', 'llm')
        self.assertEqual(game.gs.side_engine('red'), 'llm')
        game.set_side_engine('black', 'human')
        self.assertEqual(game.gs.side_engine('black'), 'human')

    def test_flip_player_side_swaps_and_keeps_opponent_engine(self):
        """中局换边（键盘 C）：红<->黑，且对手的档位跟着搬过去。"""
        game = self._game()
        game.gs.set_side_engine('black', 'llm')     # 对手（黑）用大模型
        game.flip_player_side()
        self.assertEqual(game.gs.player_side, 'black')
        self.assertEqual(game.gs.side_engine('black'), 'human')
        self.assertEqual(game.gs.side_engine('red'), 'llm',
                         "换边后对手的档位不该丢")
        game.flip_player_side()
        self.assertEqual(game.gs.player_side, 'red')
        self.assertEqual(game.gs.side_engine('black'), 'llm')

    def test_cycle_side_engine_uses_the_hotkey_rule(self):
        game = self._game()
        game.gs.set_side_engine('black', 'human')
        game.cycle_side_engine('black')
        self.assertEqual(game.gs.side_engine('black'), 'python')
        # 循环必须**只落在可用档位**上，而且一圈能回到原处
        seen = set()
        for _ in range(3):
            game.cycle_side_engine('black')
            current = game.gs.side_engine('black')
            self.assertTrue(_engines.available(current),
                            "循环停在了不可用的档位 %r" % current)
            seen.add(current)
        self.assertEqual(seen, {'python', 'llm', 'human'})

    def test_fallback_to_human_when_engine_has_no_move(self):
        """引擎拿不到着法时：把这一方**切回人工**，别每帧重试刷屏。

        用大模型那一档来造"拿不到着法"的情形（它是最可能拿不到的一方：
        没配 key / 调用失败 / 给的着法非法）。
        """
        game = self._game()
        gs = game.gs
        gs.set_side_engine('black', 'llm')
        self.assertEqual(gs.side_engine('black'), 'llm')
        gs.current_player = 'black'
        game._ai_side_started = 'black'
        game.ai_pending_move = None
        game._ai_result_source = 'llm'
        game._ai_result_error = '大模型没有给出可用着法'
        game._ai_think_began = 0.0
        game._consume_ai_move()
        self.assertEqual(gs.side_engine('black'), 'human',
                         "拿不到着法就该交回人工，否则每帧都会重试")
        self.assertFalse(game.should_auto_move())


class TestMixedModesPlayAGame(unittest.TestCase):
    """端到端：**红方本机引擎 + 黑方大模型**（替身）能下完一整局。

    这是"三种对局方式"最容易出问题的那种组合：两方走的是**不同的分岔**，
    而且大模型那一方每步都要过一次合法性校验与兜底。用替身就不联网。
    """

    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((1, 1))

    def _game(self):
        import main as main_mod

        cwd = os.getcwd()
        os.chdir(GAME_DIR)
        try:
            game = main_mod.ChessGame()
        finally:
            os.chdir(cwd)
        game.gs.show_side_selection = False
        game.ai_difficulty = 'easy'
        game.ai_controller = None
        game._ensure_ai_controller()
        controller = game.ai_controller
        controller.settings['time_limit'] = 0.2
        controller.time_limit = 0.2
        controller.engine.options.update(
            {'time_limit': 0.2, 'samples': 1, 'max_depth': 2, 'seed': 7})
        return game

    def test_red_python_vs_black_llm(self):
        from jieqi import llm as llm_mod

        game = self._game()
        gs = game.gs
        gs.set_side_engine('red', 'python')
        gs.set_side_engine('black', 'llm')
        controller = game.ai_controller
        controller.llm_enabled = True
        controller.advisor = object()
        # 大模型：总是推荐"当前局面引擎候选里的第 2 手"（稳定的替身，不联网）
        picks = {"n": 0}

        def fake_ask(snapshot, style, timeout=0.0):
            cands = snapshot.get('engine_candidates') or []
            picks["n"] += 1
            if len(cands) >= 2:
                item = cands[1]
                frm, to = item.get('from'), item.get('to')
                if isinstance(frm, str):
                    from jieqi.notation import name_to_square

                    frm = name_to_square(frm)
                    to = name_to_square(to)
                move = (frm[0], frm[1], to[0], to[1])
            else:
                move = None
            return llm_mod.LLMResult(
                ok=True, text="{}",
                parsed={"move": move, "comment": "替身点评", "confidence": 0.9},
                model='stub')

        controller._ask_llm = fake_ask

        sources = []
        for _ in range(24):
            if gs.game_over or not game.should_auto_move():
                break
            game._process_ai_move()          # 起后台线程（与主循环一致）
            thread = getattr(game, 'ai_thread', None)
            if thread is not None:
                thread.join(timeout=30.0)
            game._consume_ai_move()          # 主线程落子
            if gs.move_history:
                sources.append(gs.move_history[-1].get('engine'))
        self.assertGreaterEqual(len(sources), 8, "一整局没走起来：%s" % sources)
        self.assertIn('python', sources)
        self.assertIn('llm', sources, "黑方大模型一手都没被采用：%s" % sources)
        # 红黑交替，而且每一手都记着"当时用的是什么"
        for i, source in enumerate(sources):
            self.assertIn(source, ('python', 'llm'),
                          "第 %d 手记录了意外的来源 %r" % (i + 1, source))
        self.assertTrue(all(gs.move_history[i].get('engine')
                            for i in range(len(gs.move_history))))
        # 两方的着法都必须是合法的（拿回放出来的终局棋盘验证不现实，
        # 所以直接看每一手都在"当时的合法着法"里——由 move_piece 自己保证，
        # 这里只兜底确认没有崩、没有原地乱走）
        self.assertTrue(any(rec.get('captured') for rec in gs.move_history)
                        or len(gs.move_history) >= 8)


if __name__ == "__main__":
    unittest.main()
