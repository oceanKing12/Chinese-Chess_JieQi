# -*- coding: utf-8 -*-
"""棋谱模块测试：导出/读回/中文记谱/训练样本/失着检测。

锁住的关键约定：

1. **写盘不泄漏暗子真身**——未翻开的子只写摆位身份，``actual`` 绝不落盘
   （否则棋谱就成了"AI 作弊资料"，而这份棋谱是要拿去训练/复盘用的）；
2. 点评与评估挂在**正确的局面格**上（第 i 手取 ``ply_*[i-1]``，即"走这步之前"
   看到的那个局面），错一格会让复盘把两手的评价张冠李戴；
3. 读棋谱时"带不带 .json 后缀、给名字还是给路径"都要能找到；
4. 失着检测把"实际着法"和"引擎最佳"比出来，并按落差排序（引擎用替身，不真搜）。
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import time
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

from jieqi import record as R  # noqa: E402
from jieqi.board import encode  # noqa: E402


def setUpModule():
    pygame.init()
    pygame.display.set_mode((1, 1))


def make_gs(plies: int = 20, seed: int = 7):
    """随机走一盘（优先吃子，凑出翻子与吃子记录）。"""
    from game_state import GameState
    from jieqi.gui_adapter import gui_color_to_side, legal_moves

    random.seed(seed)
    gs = GameState(None)
    gs.show_side_selection = False
    gs.player_side = "red"
    gs.current_player = "red"
    gs.enable_ai("black")
    rng = random.Random(seed)
    for _ in range(plies):
        side = gui_color_to_side(gs.current_player)
        moves = legal_moves(gs.board, side)
        if not moves:
            break
        capturing = [mv for mv in moves
                     if gs.board[(mv & 127) // 9][(mv & 127) % 9]]
        mv = rng.choice(capturing or moves)
        gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                      (mv & 127) // 9, (mv & 127) % 9)
        # 模拟"AI 走完这一手之后给出的点评与评估"。
        # 注意：点评是**按局面**存的（第 k 手之后的局面存在下标 k），而"第 k 手
        # 的点评"只有在下标 k-1 的位置才会被写进棋谱（那才是走这步之前看到的局面）。
        # 所以最后一格（终局之后）在真实对局里本来就是空的，这里也不写，
        # 免得构造出"棋谱里存不下"的格子、把往返测试搞成假失败。
        if gs.ply <= plies - 1:
            gs.set_ply_comment(gs.ply, "第%d手的点评" % gs.ply,
                               "AI 评估：%+d（测试）" % (gs.ply * 10))
        gs.current_player = "black" if side == 0 else "red"
    gs.animations.clear()
    return gs


class TestBuildRecord(unittest.TestCase):
    def setUp(self):
        self.gs = make_gs()

    def test_record_shape(self):
        rec = R.build_record(self.gs, meta={"difficulty": "normal"})
        self.assertEqual(rec["version"], R.RECORD_VERSION)
        self.assertEqual(rec["kind"], "jieqi-game-record")
        self.assertEqual(rec["total_plies"], len(self.gs.move_history))
        # 局面数 = 手数 + 1（含开局）
        self.assertEqual(len(rec["positions"]), len(self.gs.move_history) + 1)
        self.assertEqual(len(rec["moves"]), len(self.gs.move_history))
        self.assertEqual(rec["meta"]["difficulty"], "normal")
        self.assertIn(rec["meta"]["player_side"], ("red", "black"))

    def test_move_sides_alternate_red_first(self):
        rec = R.build_record(self.gs)
        sides = [m["side"] for m in rec["moves"]]
        expect = ["red" if i % 2 == 0 else "black" for i in range(len(sides))]
        self.assertEqual(sides, expect)

    def test_hidden_pieces_never_leak(self):
        rec = R.build_record(self.gs)
        blob = json.dumps(rec, ensure_ascii=False)
        self.assertNotIn('"actual"', blob, "暗子真身被写进了棋谱")
        # 未翻开的格子必须带 hidden=True 与 home
        hidden_cells = 0
        for snap in rec["positions"]:
            for row in snap["board"]:
                for cell in row:
                    if cell and cell.get("hidden"):
                        hidden_cells += 1
                        self.assertNotIn("ident", cell)
                        self.assertTrue(cell.get("home"))
        self.assertGreater(hidden_cells, 0, "整盘棋居然没有暗子格，样本不成立")

    def test_explicit_reveal_does_not_create_ghost_capture(self):
        """只有 captured 存在时才写真身；没吃子不许出现 actual。"""
        rec = R.build_record(self.gs)
        for move in rec["moves"]:
            if "captured" not in move:
                continue
            cap = move["captured"]
            if cap["hidden"]:
                self.assertNotIn("piece", cap, "被吃的暗子不许写兵种")
            else:
                self.assertIn("piece", cap)

    def test_comments_and_evals_attach_to_previous_position(self):
        """第 i 手取 ``ply_*[i-1]``：那是走这一步的人当时看到的局面。"""
        rec = R.build_record(self.gs)
        self.assertIsNone(rec["moves"][0].get("comment"), "第 1 手之前没有点评")
        for move in rec["moves"][1:]:
            index = move["index"]
            expect = "第%d手的点评" % (index - 1)
            self.assertEqual(move.get("comment"), expect,
                             "第 %d 手的点评挂错了局面" % index)
            self.assertIn("评估", move.get("eval", ""))

    def test_reveal_flag_keeps_hidden_truth_only_when_asked(self):
        rec = R.build_record(self.gs, meta={"name": "x"}, reveal_hidden=True)
        blob = json.dumps(rec, ensure_ascii=False)
        self.assertIn('"actual"', blob)
        self.assertTrue(rec["meta"]["reveal_hidden"])


class TestTextAndCodes(unittest.TestCase):
    def setUp(self):
        self.gs = make_gs()

    def test_board_codes_mark_hidden(self):
        codes = R.board_to_codes(self.gs.board)
        self.assertEqual(len(codes), 10)
        joined = " ".join(codes)
        self.assertIn("*", joined, "暗子应当带 * 标记")
        for row in codes:
            for cell in row.split():
                # 空格是 ".."；有子是"归属(+兵种(+暗子标记))"，2~3 个字符
                self.assertIn(len(cell), (2, 3), cell)
                if cell != "..":
                    self.assertIn(cell[0], "RB")

    def test_board_to_text_lists_rows_and_cols(self):
        """棋盘文本用 ``a``~``i`` 列号 + ``0``~``9`` 行号（与引擎同一口径）。"""
        text = R.board_to_text(self.gs.board)
        self.assertIn("a b c d e f g h i", text)
        self.assertIn("行9", text)      # 黑方底线
        self.assertIn("行0", text)      # 红方底线

    def test_record_text_has_moves_and_losses(self):
        rec = R.build_record(self.gs)
        text = R.record_to_text(rec)
        self.assertIn("揭棋棋谱", text)
        self.assertIn("【棋谱】", text)
        self.assertIn("【终局棋盘】", text)
        self.assertIn("【被吃之子】", text)
        self.assertIn("1.", text)
        self.assertIn("点评：", text)

    def test_move_to_chinese_marks_hidden_and_flip(self):
        move = {"index": 1, "side": "red", "hidden": True, "home": "炮",
                "from": "b7", "to": "b5", "revealed_as": "车"}
        text = R.move_to_chinese(move)
        self.assertIn("红暗炮?", text)
        self.assertIn("b7->b5", text)
        self.assertIn("翻开后是 红车", text)
        shown = {"index": 2, "side": "black", "hidden": False, "piece": "馬",
                 "home": "馬", "from": "b0", "to": "c2",
                 "captured": {"side": "red", "hidden": True, "home": "兵"}}
        text2 = R.move_to_chinese(shown)
        self.assertIn("黑馬", text2)
        self.assertIn("吃红暗子(未知)", text2)


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.gs = make_gs()
        self.dir = tempfile.mkdtemp(prefix="jieqi-record-test-")

    def test_save_and_load_by_name_or_path(self):
        rec = R.build_record(self.gs)
        info = R.save_record(rec, directory=self.dir, name="unit-a")
        self.assertTrue(info["ok"], info.get("error"))
        self.assertTrue(os.path.isfile(info["path"]))
        self.assertTrue(os.path.isfile(info["text_path"]))
        # 全路径 / 不带后缀的全路径 / 相对 records 的名字 三种写法都要能读
        self.assertIsNotNone(R.load_record(info["path"]))
        self.assertIsNotNone(R.load_record(
            os.path.join(self.dir, "unit-a")))
        self.assertIsNotNone(R.load_record(
            os.path.join(self.dir, "unit-a.json")))
        self.assertIsNone(R.load_record("no-such-record"))

    def test_index_and_list(self):
        for name in ("unit-a", "unit-b"):
            R.save_record(R.build_record(self.gs), directory=self.dir, name=name)
        items = R.list_records(self.dir)
        self.assertEqual(sorted(item["name"] for item in items),
                         ["unit-a", "unit-b"])
        index_path = os.path.join(self.dir, "index.json")
        self.assertTrue(os.path.isfile(index_path))
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        self.assertEqual(len(index), 2)

    def test_save_is_forgiving(self):
        """坏数据只返回 ok=False，绝不抛异常。"""
        info = R.save_record({"name": "bad", "meta": {"x": object()}},
                             directory=self.dir)
        self.assertFalse(info["ok"])
        self.assertTrue(info["error"])


class TestSaveNeverOverwrites(unittest.TestCase):
    """**棋谱不能被覆盖**（真实反馈："不要每次更新都覆盖之前的棋谱文件"）。

    来源有两个，都要钉住：

    1. ``save_record`` 原本直接 ``open(path, "w")``——同名（尤其是同一秒内的
       两个时间戳）会把已有棋谱悄悄盖掉；
    2. ``import_record`` 原本把**导入文件的名字**写进 ``record_saved``，于是
       "导入一份棋谱 → 接着下 → 终局自动保存"会写回导入的那份文件。
    """

    def setUp(self):
        self.gs = make_gs()
        self.dir = tempfile.mkdtemp(prefix="jieqi-nosave-test-")

    def _jsons(self):
        """棋谱文件（**排除 index.json** —— 那是索引不是棋谱）。"""
        return sorted(n for n in os.listdir(self.dir)
                      if n.endswith(".json") and n != "index.json")

    def _read(self, name):
        with open(os.path.join(self.dir, name), "rb") as handle:
            return handle.read()

    def test_same_name_twice_creates_two_files(self):
        """同一个名字存两次 → 两个文件，第一个**一字节都不能变**。"""
        R.save_record(R.build_record(self.gs), directory=self.dir, name="dup")
        before = self._read("dup.json")
        R.save_record(R.build_record(self.gs), directory=self.dir, name="dup")
        self.assertEqual(len(self._jsons()), 2,
                         "第二次保存覆盖了第一个：%s" % self._jsons())
        self.assertEqual(self._read("dup.json"), before,
                         "原有棋谱的内容被改动了")
        self.assertIn("dup-2.json", self._jsons())

    def test_text_companion_is_paired(self):
        """json 和 txt 是一对：改名时两边必须同步，不能一新一旧。"""
        R.save_record(R.build_record(self.gs), directory=self.dir, name="pair")
        R.save_record(R.build_record(self.gs), directory=self.dir, name="pair")
        texts = sorted(n for n in os.listdir(self.dir) if n.endswith(".txt"))
        self.assertEqual(texts, ["pair-2.txt", "pair.txt"], texts)

    def test_overwrite_flag_still_works(self):
        """显式 ``overwrite=True`` 时要能覆盖（留给知道后果的调用方）。"""
        R.save_record(R.build_record(self.gs), directory=self.dir, name="one")
        R.save_record(R.build_record(self.gs), directory=self.dir, name="one",
                      overwrite=True)
        self.assertEqual(self._jsons(), ["one.json"])

    def test_unique_flag_always_opens_a_new_file(self):
        """``unique=True``（终局自动保存）永远开新文件。"""
        R.save_record(R.build_record(self.gs), directory=self.dir, name="uni")
        R.save_record(R.build_record(self.gs), directory=self.dir, name="uni",
                      unique=True)
        R.save_record(R.build_record(self.gs), directory=self.dir, name="uni",
                      unique=True)
        self.assertEqual(len(self._jsons()), 3, self._jsons())

    def test_result_reports_requested_name(self):
        """返回值要带 ``requested``，调用方才能发现"名字被改了"。"""
        first = R.save_record(R.build_record(self.gs), directory=self.dir,
                              name="req")
        second = R.save_record(R.build_record(self.gs), directory=self.dir,
                               name="req")
        self.assertEqual(first["requested"], "req")
        self.assertEqual(second["requested"], "req")
        self.assertEqual(os.path.basename(second["path"]), "req-2.json")
        self.assertNotEqual(os.path.basename(second["path"]),
                            second["requested"] + ".json")

    def test_save_text_does_not_overwrite(self):
        """单独写 txt 也不能盖掉已有的。"""
        rec = R.build_record(self.gs)
        first = R.save_text(rec, directory=self.dir, name="txt-only")
        second = R.save_text(rec, directory=self.dir, name="txt-only")
        self.assertTrue(first and second)
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.isfile(first) and os.path.isfile(second))

    def test_unique_record_name_skips_taken_pairs(self):
        """``unique_record_name`` 要把 json/txt 当成一个整体来判占位。"""
        R.save_record(R.build_record(self.gs), directory=self.dir, name="u")
        self.assertEqual(R.unique_record_name("u", self.dir), "u-2")
        # 只留 txt 也算占用（否则 json 会覆盖、txt 变成孤儿）
        os.remove(os.path.join(self.dir, "u.json"))
        self.assertEqual(R.unique_record_name("u", self.dir), "u-2")


class TestImportedRecordIsNeverOverwritten(unittest.TestCase):
    """导入棋谱 → 接着下完 → 自动保存，必须**新开一份**，原文件一字不改。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jieqi-import-test-")
        self.source = make_gs()
        R.save_record(R.build_record(self.source), directory=self.dir,
                      name="original")

    def _load_original(self):
        return R.load_record(os.path.join(self.dir, "original.json"))

    def test_import_keeps_name_in_imported_from(self):
        from game_state import GameState

        gs = GameState(None)
        gs.show_side_selection = False
        info = gs.import_record(self._load_original())
        self.assertTrue(info.get("ok"), info.get("error"))
        self.assertEqual(gs.imported_from, "original",
                         "导入的名字应该记在 imported_from")
        self.assertEqual(gs.record_saved, "",
                         "导入的名字**不能**进 record_saved，否则会覆盖它")

    def test_auto_save_after_import_writes_a_new_file(self):
        from game_state import GameState

        gs = GameState(None)
        gs.show_side_selection = False
        gs.import_record(self._load_original())
        before = open(os.path.join(self.dir, "original.json"), "rb").read()
        # 终局自动保存走的就是这条路（unique=True）
        result = gs.export_record(directory=self.dir, unique=True)
        self.assertTrue(result.get("ok"), result.get("error"))
        after = open(os.path.join(self.dir, "original.json"), "rb").read()
        self.assertEqual(before, after, "导入的棋谱被改写了！")
        written = os.path.basename(result.get("path") or "")
        self.assertTrue(written and written != "original.json",
                        "自动保存不该写成导入的那份：%s" % written)
        self.assertTrue(gs.record_saved, "自动保存后应记下自己写出的名字")
        self.assertNotEqual(gs.record_saved, "original")

    def test_manual_export_reuses_its_own_file(self):
        """同一局手动导出两次只更新自己那份，不产生 -2 垃圾。

        关键在 ``record_saved``：第一次导出把它设成自己写出的名字，
        第二次就复用——这正是"编辑同一份"的语义（与"不同对局"区分开）。
        """
        from game_state import GameState

        gs = GameState(None)
        gs.show_side_selection = False
        gs.record_saved = "my-game"
        first = gs.export_record(directory=self.dir)
        self.assertTrue(first.get("ok"), first.get("error"))
        self.assertEqual(gs.record_saved, "my-game")
        second = gs.export_record(directory=self.dir)
        self.assertTrue(second.get("ok"), second.get("error"))
        self.assertEqual(first.get("path"), second.get("path"),
                         "同一局重复导出不该新建文件")
        # 目录里本来就有 setUp 建的 original.json，所以这里断言"没有 my-game-2"
        self.assertNotIn("my-game-2.json", os.listdir(self.dir),
                         "重复导出产生了 -2 垃圾")


class TestTrainingExport(unittest.TestCase):
    def setUp(self):
        self.gs = make_gs()
        self.dir = tempfile.mkdtemp(prefix="jieqi-train-")
        self.rec = R.build_record(self.gs)

    def test_samples_have_labels_without_hidden_truth(self):
        path = os.path.join(self.dir, "train.jsonl")
        info = R.export_training(self.rec, path)
        self.assertTrue(info["ok"], info.get("error"))
        self.assertEqual(info["samples"], self.rec["total_plies"])
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(rows), info["samples"])
        for row in rows:
            self.assertIsInstance(row["move"], str)
            self.assertEqual(len(row["move"]), 4, "着法应当写成 b7b5 这种形式")
            self.assertTrue(row["legal_moves"])
            self.assertIn(row["value"], (1.0, 0.0, -1.0))
            self.assertEqual(len(row["board"]), 10)
            self.assertNotIn("hidden_truth", row)
            self.assertNotIn("actual", json.dumps(row, ensure_ascii=False))
        # 每条样本的走法必须真的在合法着法里（两边都是 ``b7b5`` 写法）
        for row in rows:
            self.assertIn(row["move"], row["legal_moves"],
                          "第 %s 手的着法 %s 不在合法着法里"
                          % (row["ply"], row["move"]))

    def test_hidden_truth_opt_in(self):
        # 真身要先在 build_record 阶段就保留（默认不保留，见 test_hidden_pieces_never_leak）
        full = R.build_record(self.gs, reveal_hidden=True)
        path = os.path.join(self.dir, "train-truth.jsonl")
        info = R.export_training(full, path, include_hidden_truth=True)
        self.assertTrue(info["ok"])
        with open(path, encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        self.assertIn("hidden_truth", first)
        self.assertTrue(first["hidden_truth"])


class _StubStat:
    def __init__(self, move, mean, select=None):
        self.move = move
        self.mean = mean
        self.select_score = mean if select is None else select
        self.pv = [move]
        self.worst = mean
        self.samples = 1


class _StubResult:
    def __init__(self, move, stats, depth=6):
        self.move = move
        self.stats = stats
        self.depth = depth
        self.nodes = 123
        self.samples = 1
        self.score = 0.0
        self.verdict = ""


class TestBlunderScan(unittest.TestCase):
    """失着检测：引擎用替身，只验证"比较与排序"的逻辑。"""

    def setUp(self):
        self.gs = make_gs(plies=6)
        self.rec = R.build_record(self.gs)

    def test_flags_the_worst_move_first(self):
        from jieqi.record import _move_coords

        first = self.rec["moves"][0]
        fr, fc, tr, tc = _move_coords(first)
        real = encode(fr * 9 + fc, tr * 9 + tc)
        other = None
        for move in self.rec["moves"][1:]:
            fr, fc, tr, tc = _move_coords(move)
            cand = encode(fr * 9 + fc, tr * 9 + tc)
            if cand != real:
                other = cand
                break
        self.assertIsNotNone(other, "样本里找不到第二个着法")

        def factory():
            class _Eng:
                def analyse(self, board, **kwargs):
                    return _StubResult(other, [_StubStat(other, 100.0),
                                               _StubStat(real, -200.0)])
            return _Eng()

        out = R.analyse_record(self.rec, engine_factory=factory,
                               threshold=100, top=5)
        self.assertEqual(out["checked"], len(self.rec["moves"]))
        self.assertTrue(out["items"])
        worst = out["items"][0]
        self.assertEqual(worst["ply"], 1)
        self.assertEqual(worst["loss"], 300.0)
        self.assertFalse(worst["is_best"])
        # 排序必须是落差从大到小
        losses = [item["loss"] for item in out["items"]]
        self.assertEqual(losses, sorted(losses, reverse=True))

    def test_no_flags_when_engine_agrees(self):
        def factory():
            class _Eng:
                def analyse(self, board, **kwargs):
                    # 引擎最佳分与实际分一样 -> 落差 0
                    return _StubResult(None, [_StubStat(0, 0.0)])

            return _Eng()

        out = R.analyse_record(self.rec, engine_factory=factory)
        self.assertEqual(out["items"], [])


class TestImportRecord(unittest.TestCase):
    """导入棋谱：重放出来的局面必须与原局**逐格一致**，点评也要挂回原位。

    这是"导入"功能的底线——如果重放出来的棋盘和原局不同，那导入的就不是同一盘棋。
    重放而不是直接套用棋盘快照，是因为棋谱里暗子只存摆位身份（不存真身），
    必须靠 `reviewed_as`（第几手翻出了什么）现场补上才走得通。
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jieqi-import-")
        self.src = make_gs(plies=24, seed=17)
        self.src.game_over = True
        self.src.winner = "red"

    @staticmethod
    def signature(board):
        out = []
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if not piece:
                    out.append("..")
                else:
                    out.append("%s%s%s" % (
                        piece["color"][0],
                        piece["actual"] if piece["revealed"] else "?",
                        "h" if not piece["revealed"] else "v"))
        return "|".join(out)

    def test_round_trip_is_identical(self):
        from game_state import GameState

        record = R.build_record(self.src)
        dst = GameState(None)
        dst.show_side_selection = False
        info = dst.import_record(record)
        self.assertTrue(info["ok"], info.get("error"))
        self.assertEqual(info["warnings"], [], "重放过程中不该有非法着法")
        self.assertEqual(dst.total_plies, self.src.total_plies)
        # 导入之后停在开局（棋谱多半是下完的，不能一进来就弹结算把界面锁住）
        self.assertEqual(dst.ply, 0, "导入后没有停在开局")
        self.assertFalse(dst.game_over, "导入后立刻进入结算状态")
        self.assertIsNone(dst.winner)
        self.assertEqual(self.signature(dst.board),
                         self.signature(self.src.positions[0]["board"]),
                         "导入后棋盘与开局不一致")
        # 点评按局面格恢复：能一路"上一步/下一步"复盘
        self.assertEqual(dst.ply_comments, self.src.ply_comments)
        self.assertEqual(dst.ply_evals, self.src.ply_evals)
        dst.goto_ply(max(0, dst.total_plies - 4))
        self.assertEqual(dst.ply, max(0, dst.total_plies - 4))
        self.assertFalse(dst.game_over, "中途翻历史触发了结算")
        dst.goto_ply(dst.total_plies)
        self.assertEqual(dst.ply, dst.total_plies)
        # 翻到最后一手才按棋谱结果结算，而且棋盘必须与原局终局完全一致
        self.assertEqual(self.signature(dst.board), self.signature(self.src.board))
        self.assertEqual((dst.game_over, dst.winner),
                         (self.src.game_over, self.src.winner))
        # 再退回开局，结算状态要清掉（否则又变成"点哪都重开"）
        dst.goto_ply(0)
        self.assertFalse(dst.game_over, "退回开局后仍处于结算状态")
        self.assertIsNone(dst.winner)

    def test_import_starts_at_opening_and_replays_to_end(self):
        """导入完成的对局：从开局开始，一路「下一步」能走到终局并结算。"""
        from game_state import GameState

        record = R.build_record(self.src)
        dst = GameState(None)
        dst.show_side_selection = False
        dst.import_record(record)
        for ply in range(1, dst.total_plies + 1):
            dst.goto_ply(ply)
            self.assertEqual(dst.ply, ply)
            if ply < dst.total_plies:
                self.assertFalse(dst.game_over,
                                 f"第 {ply} 手就结算了（应当只到最后一手才结算）")
        self.assertTrue(dst.game_over)
        self.assertEqual(dst.winner, self.src.winner)

    def test_import_by_file_name(self):
        from game_state import GameState

        record = R.build_record(self.src)
        R.save_record(record, directory=self.dir, name="import-check")
        dst = GameState(None)
        dst.show_side_selection = False
        info = dst.import_record_file(os.path.join(self.dir, "import-check"))
        self.assertTrue(info["ok"], info.get("error"))
        self.assertEqual(info["plies"], self.src.total_plies)

    def test_bad_input_is_reported_not_raised(self):
        from game_state import GameState

        dst = GameState(None)
        dst.show_side_selection = False
        self.assertFalse(dst.import_record({})["ok"])
        self.assertFalse(dst.import_record({"moves": []})["ok"])
        self.assertFalse(dst.import_record_file("no-such-record-xyz")["ok"])
        self.assertFalse(dst.import_record(None)["ok"])

    def test_illegal_move_stops_replay_with_warning(self):
        """棋谱被改坏（某一步不合法）时要停下并说明，而不是走出非法局面。"""
        from game_state import GameState

        record = R.build_record(self.src)
        record["moves"][3]["to"] = "a1"          # 篡改第 4 手的落点（新记法）
        record["moves"][3]["from"] = "a9"
        dst = GameState(None)
        dst.show_side_selection = False
        info = dst.import_record(record)
        self.assertTrue(info["ok"])
        self.assertTrue(info["warnings"], "非法着法必须给出提示")
        self.assertLess(dst.total_plies, self.src.total_plies)

    def test_paused_after_import_does_not_let_ai_move(self):
        """导入的是别人的对局，不该让 AI 接着往下走。"""
        from game_state import GameState

        record = R.build_record(self.src)
        dst = GameState(None)
        dst.show_side_selection = False
        dst.import_record(record)
        # import 之后 main.py 会调 _init_ai() 清掉人机模式
        dst._init_ai()
        self.assertFalse(dst.ai_mode)
        self.assertEqual(dst.ai_color, "")


class TestReplayPlan(unittest.TestCase):
    def test_plan_has_moves_and_reveals(self):
        gs = make_gs(plies=14, seed=3)
        record = R.build_record(gs)
        moves, reveals, meta = R.replay_plan(record)
        self.assertEqual(len(moves), record["total_plies"])
        self.assertEqual(meta["player_side"], "red")
        # 翻子表里的每一手都必须真的是一手翻子（hidden=True 且带 revealed_as）
        for index, (r, c, piece) in reveals.items():
            self.assertTrue(piece)
            move = record["moves"][index - 1]
            self.assertTrue(move["hidden"])
            self.assertEqual(move["revealed_as"], piece)

    def test_plan_ignores_broken_moves(self):
        record = {"moves": [{"from": [1, 1], "to": [2, 2]},
                            {"from": None, "to": [2, 2]},
                            {"from": [3, 3], "to": [4, 4], "hidden": True,
                             "revealed_as": "车"}]}
        moves, reveals, _meta = R.replay_plan(record)
        self.assertEqual(len(moves), 2)
        # 翻子表按"重放后的手数"编号（坏记录被跳过，编号要跟着走）
        self.assertIn(2, reveals)
        self.assertEqual(reveals[2][2], "车")


class TestAutoSaveConfig(unittest.TestCase):
    """``record.auto_save``：对局结束自动保存棋谱（默认开启）。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jieqi-autosave-")
        from game_state import GameState

        self.gs = GameState(None)
        self.gs.show_side_selection = False
        self.gs.player_side = "red"
        self.gs.enable_ai("black")
        self._play(12, seed=5)

    def _play(self, plies, seed=5):
        from jieqi.gui_adapter import gui_color_to_side, legal_moves

        rng = random.Random(seed)
        for _ in range(plies):
            side = gui_color_to_side(self.gs.current_player)
            moves = legal_moves(self.gs.board, side)
            if not moves:
                break
            mv = rng.choice(moves)
            self.gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                               (mv & 127) // 9, (mv & 127) % 9)
            self.gs.current_player = "black" if side == 0 else "red"

    def _records(self):
        return [f for f in os.listdir(self.dir)
                if f.endswith(".json") and f != "index.json"]

    def test_default_is_on_when_config_is_missing(self):
        with mock.patch("jieqi.settings.load_raw_config", return_value={}):
            self.assertTrue(self.gs.auto_save_record(),
                            "配置里没写 record.auto_save 时必须默认开启")

    def test_reads_the_config_flag(self):
        with mock.patch("jieqi.settings.load_raw_config",
                        return_value={"record": {"auto_save": True}}):
            self.assertTrue(self.gs.auto_save_record())
        with mock.patch("jieqi.settings.load_raw_config",
                        return_value={"record": {"auto_save": False}}):
            self.assertFalse(self.gs.auto_save_record())
        # 类型/结构不对也要退回默认（宁可保存，也别因为配置写错就丢了棋谱）
        for broken in ({"record": None}, {"record": {}}, {"record": "x"},
                       {"record": {"auto_save": None}}):
            with mock.patch("jieqi.settings.load_raw_config",
                            return_value=broken):
                self.assertTrue(self.gs.auto_save_record(), broken)

    def test_auto_save_writes_a_record_on_game_over(self):
        with mock.patch("jieqi.settings.load_raw_config",
                        return_value={"record": {"auto_save": True}}), \
                mock.patch.object(R, "record_dir", return_value=self.dir):
            self.gs.game_over = True
            self.gs.winner = "red"
            self.gs.check_game_state()
        self.assertEqual(len(self._records()), 1,
                         f"没有自动保存棋谱：{os.listdir(self.dir)}")
        self.assertTrue(self.gs.record_saved)
        # 保存成功必须给"看得见"的反馈
        self.assertTrue(self.gs.toast_loud, "自动保存没有放大提示")
        self.assertIn("棋谱", self.gs.toast)
        self.assertGreater(self.gs.toast_until, 0.0)

    def test_auto_save_off_writes_nothing(self):
        with mock.patch("jieqi.settings.load_raw_config",
                        return_value={"record": {"auto_save": False}}), \
                mock.patch.object(R, "record_dir", return_value=self.dir):
            self.gs.game_over = True
            self.gs.winner = "red"
            self.gs.check_game_state()
        self.assertEqual(self._records(), [], "关掉自动保存却还是写了文件")

    def test_auto_save_only_once(self):
        """终局后每帧都会调 check_game_state，不能反复写盘。"""
        with mock.patch("jieqi.settings.load_raw_config",
                        return_value={"record": {"auto_save": True}}), \
                mock.patch.object(R, "record_dir", return_value=self.dir):
            self.gs.game_over = True
            self.gs.winner = "red"
            for _ in range(5):
                self.gs.check_game_state()
        self.assertEqual(len(self._records()), 1)


class TestToastFeedback(unittest.TestCase):
    """导出/导入这类操作必须有**看得见**的反馈（真实反馈："点了不知道成没成功"）。"""

    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        os.chdir(GAME_DIR)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def setUp(self):
        from game_state import GameState

        self.gs = GameState(None)
        self.gs.show_side_selection = False

    def test_set_toast_marks_loud_and_expires(self):
        self.gs.set_toast("棋谱已导出", loud=True, seconds=3.0)
        self.assertTrue(self.gs.toast_loud)
        self.assertGreater(self.gs.toast_until, time.monotonic())
        # 状态类提示不放大
        self.gs.set_toast("AI 思考中…")
        self.assertFalse(self.gs.toast_loud)
        # 空文字不留残留
        self.gs.set_toast("")
        self.assertEqual(self.gs.toast, "")
        self.assertEqual(self.gs.toast_until, 0.0)

    def test_export_record_reports_path_and_failure(self):
        """``GameState.export_record`` 成功给路径、失败给中文原因。"""
        from game_state import GameState

        # 开局的棋谱也能导出（只有一手都没走时才由界面层拒绝）
        empty = GameState(None)
        empty.show_side_selection = False
        empty_dir = tempfile.mkdtemp(prefix="jieqi-export-empty-")
        with mock.patch.object(R, "record_dir", return_value=empty_dir):
            info0 = empty.export_record(directory=empty_dir)
        self.assertTrue(info0["ok"], info0.get("error"))
        self.assertTrue(os.path.isfile(info0["path"]))

        # 走几步再导出，文件名要记进 record_saved
        from jieqi.gui_adapter import gui_color_to_side, legal_moves

        rng = random.Random(3)
        self.gs.player_side = "red"
        self.gs.enable_ai("black")
        for _ in range(6):
            side = gui_color_to_side(self.gs.current_player)
            moves = legal_moves(self.gs.board, side)
            if not moves:
                break
            mv = rng.choice(moves)
            self.gs.move_piece((mv >> 7) // 9, (mv >> 7) % 9,
                               (mv & 127) // 9, (mv & 127) % 9)
            self.gs.current_player = "black" if side == 0 else "red"
        self.dir = tempfile.mkdtemp(prefix="jieqi-export-")
        with mock.patch.object(R, "record_dir", return_value=self.dir):
            info = self.gs.export_record(directory=self.dir)
        self.assertTrue(info["ok"], info.get("error"))
        self.assertTrue(os.path.isfile(info["path"]))
        self.assertTrue(os.path.isfile(info["text_path"]))
        self.assertEqual(self.gs.record_saved,
                         os.path.splitext(os.path.basename(info["path"]))[0])

    def test_renderer_draws_loud_banner_while_fresh(self):
        """放大提示只在有效期内画：横幅要真的出现在棋盘上。"""
        from resource_manager import ResourceManager
        from renderer import Renderer

        rm = ResourceManager()
        rm.load_resources()
        gs = self.gs
        renderer = Renderer(rm, gs)
        gs.set_toast("棋谱已导出到 records\\x.json", loud=True, seconds=5.0)
        renderer.draw_all()
        self.assertTrue(renderer._toast_is_loud())
        # 过期之后不再放大（缩回顶栏那一行）
        gs.toast_until = time.monotonic() - 1.0
        self.assertFalse(renderer._toast_is_loud())
        renderer.draw_all()
        # 非 loud 的提示永远不放大
        gs.set_toast("AI 思考中…")
        self.assertFalse(renderer._toast_is_loud())
        renderer.draw_all()
        # 失败类文案用另一种边框（不抛异常即可，颜色细节由 _draw_toast_banner 决定）
        gs.set_toast("棋谱导出失败：磁盘满了", loud=True)
        self.assertTrue(renderer._toast_is_loud())
        renderer.draw_all()


class TestRecordToolCli(unittest.TestCase):
    """CLI 的 list/show/train 至少要在真实棋谱上跑通。"""

    def setUp(self):
        self.gs = make_gs(plies=8)
        self.dir = tempfile.mkdtemp(prefix="jieqi-cli-")
        R.save_record(R.build_record(self.gs), directory=self.dir, name="cli-test")

    def test_cli_list_and_show(self):
        import record_tool

        self.assertEqual(record_tool.main(["--dir", self.dir, "list"]), 0)
        self.assertEqual(record_tool.main(["--dir", self.dir, "show", "cli-test"]), 0)
        self.assertEqual(record_tool.main(["--dir", self.dir, "show", "nope"]),
                         2)

    def test_cli_train(self):
        import record_tool

        out = os.path.join(self.dir, "train.jsonl")
        self.assertEqual(
            record_tool.main(["--dir", self.dir, "train", "cli-test", "-o", out]), 0)
        self.assertTrue(os.path.isfile(out))
        with open(out, encoding="utf-8") as handle:
            self.assertTrue(handle.readline().strip())


if __name__ == "__main__":
    unittest.main()
