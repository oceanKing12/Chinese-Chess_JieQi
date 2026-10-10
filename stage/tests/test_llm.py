# -*- coding: utf-8 -*-
"""jieqi.llm 单元测试（仅使用标准库 unittest）。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest import mock

STAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ...\stage
GAME_DIR = os.path.join(STAGE_DIR, "Game")
for _path in (GAME_DIR, STAGE_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from jieqi import llm  # noqa: E402
from jieqi.llm import (  # noqa: E402
    LLMAdvisor,
    LLMConfig,
    LLMResult,
    blend_bonus,
    build_messages,
    load_config,
    parse_response,
    render_position_text,
)

FAKE_KEY = "sk-test-abcdefghijklmnop1234"


# ---------------------------------------------------------------------------
# 测试用工具
# ---------------------------------------------------------------------------

FULL_SNAPSHOT = {
    "side_to_move": "red",
    "ai_color": "black",
    "move_number": 17,
    "last_move": {
        "from": "c7",
        "to": "c6",
        "piece": "红兵",
        "captured": "黑卒",
    },
    "board_text": "  ９ ８ ７\n9 車 馬 相\n8 ． ． ．",
    "board": [
        [None, {"side": "red", "hidden": True, "home": "炮", "ident": None}]
        + [None] * 7
        for _ in range(10)
    ],
    "engine_candidates": [
        {
            "from": "c5",
            "to": "e5",
            "score": 18,
            "winrate": 0.54,
            "pv": ["c5e5", "g3e3"],
            "note": "炮二平五",
        },
        {
            "from": "b2",
            "to": "e2",
            "score": 8,
            "winrate": 0.51,
            "pv": [],
            "note": "炮八平五",
        },
    ],
    "engine_depth": 6,
    "engine_nodes": 41000,
    "engine_samples": 8,
    "engine_verdict": "红方稍优",
    "repetition_warning": False,
}


def make_config(**overrides) -> LLMConfig:
    values = {
        "enabled": True,
        "base_url": "https://example.test/v1",
        "api_key": FAKE_KEY,
        "api_key_env": "JIEQI_TEST_KEY_XYZ",
        "model": "test-model",
        "timeout": 5.0,
        "temperature": 0.2,
        "max_tokens": 700,
        "mode": "explain",
        "blend_weight": 0.0,
        "extra_headers": {},
        "warnings": [],
    }
    values.update(overrides)
    return LLMConfig(**values)


class FakeResponse:
    """最小化的 urlopen 返回对象：支持 with 语句和 read()。"""

    def __init__(self, body, status=200):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._body = body
        self.status = status
        self.code = status

    def read(self, amount=None):
        if amount is None:
            return self._body
        return self._body[:amount]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def openai_body(content, model="test-model"):
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
        ensure_ascii=False,
    )


@contextlib.contextmanager
def temp_env(**mapping):
    """临时修改 / 删除环境变量，退出时精确还原（值为 None 表示删除）。"""
    saved = dict(os.environ)
    try:
        for key, value in mapping.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def write_config(payload, directory=None):
    """写一个临时配置文件，返回其路径。"""
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False, dir=directory
    )
    with handle:
        if isinstance(payload, str):
            handle.write(payload)
        else:
            json.dump(payload, handle, ensure_ascii=False)
    return handle.name


# ---------------------------------------------------------------------------
# 1. load_config
# ---------------------------------------------------------------------------


class TestLoadConfig(unittest.TestCase):
    def test_missing_file_returns_defaults_with_warning(self):
        with temp_env(JIEQI_LLM_ENABLED=None, JIEQI_LLM_BASE_URL=None,
                      JIEQI_LLM_MODEL=None, JIEQI_LLM_TIMEOUT=None,
                      JIEQI_LLM_MODE=None, JIEQI_LLM_BLEND_WEIGHT=None,
                      JIEQI_AI_CONFIG=None, OPENAI_API_KEY=None,
                      JIEQI_TEST_KEY_XYZ=None):
            missing = os.path.join(tempfile.gettempdir(), "definitely-missing-jieqi.json")
            if os.path.exists(missing):  # pragma: no cover
                os.remove(missing)
            cfg = load_config(missing)
        self.assertIsInstance(cfg, LLMConfig)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.base_url, "https://api.openai.com/v1")
        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertEqual(cfg.timeout, 20.0)
        self.assertEqual(cfg.mode, "explain")
        self.assertEqual(cfg.blend_weight, 0.0)
        self.assertTrue(any("未找到配置文件" in w for w in cfg.warnings), cfg.warnings)

    def test_malformed_json_returns_defaults_with_warning(self):
        path = write_config('{"llm": {"enabled": true,,}')
        try:
            cfg = load_config(path)
        finally:
            os.remove(path)
        self.assertFalse(cfg.enabled)
        self.assertTrue(any("JSON 解析失败" in w for w in cfg.warnings), cfg.warnings)

    def test_valid_file_is_read_and_ai_section_ignored(self):
        payload = {
            "ai": {"depth": 6, "this": "belongs to another author"},
            "llm": {
                "enabled": True,
                "base_url": "https://api.example.com/v1",
                "api_key": "file-key-000011112222",
                "api_key_env": "SOME_OTHER_VAR",
                "model": "my-model",
                "timeout": 7.5,
                "temperature": 0.6,
                "max_tokens": 321,
                "mode": "review",
                "blend_weight": 0.25,
                "extra_headers": {"X-Test": "1"},
            },
        }
        path = write_config(payload)
        try:
            cfg = load_config(path)
        finally:
            os.remove(path)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.base_url, "https://api.example.com/v1")
        self.assertEqual(cfg.api_key, "file-key-000011112222")
        self.assertEqual(cfg.api_key_env, "SOME_OTHER_VAR")
        self.assertEqual(cfg.model, "my-model")
        self.assertEqual(cfg.timeout, 7.5)
        self.assertEqual(cfg.temperature, 0.6)
        self.assertEqual(cfg.max_tokens, 321)
        self.assertEqual(cfg.mode, "review")
        self.assertEqual(cfg.blend_weight, 0.25)
        self.assertEqual(cfg.extra_headers, {"X-Test": "1"})
        self.assertEqual(cfg.warnings, [])

    def test_missing_llm_section_returns_defaults_with_warning(self):
        path = write_config({"ai": {"depth": 4}})
        try:
            cfg = load_config(path)
        finally:
            os.remove(path)
        self.assertFalse(cfg.enabled)
        self.assertTrue(any("缺少 llm 段" in w for w in cfg.warnings), cfg.warnings)

    def test_env_overrides_file(self):
        payload = {
            "llm": {
                "enabled": False,
                "base_url": "https://file.example/v1",
                "api_key": "file-key-1234567890ab",
                "model": "file-model",
                "timeout": 30.0,
                "mode": "explain",
                "blend_weight": 0.1,
            }
        }
        path = write_config(payload)
        try:
            with temp_env(
                JIEQI_LLM_ENABLED="true",
                JIEQI_LLM_BASE_URL="https://env.example/v1",
                JIEQI_LLM_MODEL="env-model",
                JIEQI_LLM_TIMEOUT="3.5",
                JIEQI_LLM_MODE="suggest",
                JIEQI_LLM_BLEND_WEIGHT="0.5",
            ):
                cfg = load_config(path)
        finally:
            os.remove(path)
        self.assertTrue(cfg.enabled)  # 环境变量 true 覆盖文件 false
        self.assertEqual(cfg.base_url, "https://env.example/v1")
        self.assertEqual(cfg.model, "env-model")
        self.assertEqual(cfg.timeout, 3.5)
        self.assertEqual(cfg.mode, "suggest")
        self.assertEqual(cfg.blend_weight, 0.5)
        self.assertEqual(cfg.api_key, "file-key-1234567890ab")  # 文件显式 key 优先

    def test_api_key_env_fallback_and_precedence(self):
        payload = {"llm": {"enabled": True, "api_key": "", "api_key_env": "JIEQI_TEST_KEY_XYZ"}}
        path = write_config(payload)
        try:
            with temp_env(JIEQI_TEST_KEY_XYZ="env-key-999988887777"):
                cfg = load_config(path)
        finally:
            os.remove(path)
        self.assertEqual(cfg.api_key, "env-key-999988887777")

    def test_api_key_env_var_name_is_configurable(self):
        payload = {
            "llm": {"enabled": True, "api_key_env": "JIEQI_ALT_KEY", "api_key": ""},
        }
        path = write_config(payload)
        try:
            with temp_env(JIEQI_ALT_KEY="alt-key"):
                cfg = load_config(path)
        finally:
            os.remove(path)
        self.assertEqual(cfg.api_key, "alt-key")

    def test_invalid_mode_falls_back_with_warning(self):
        path = write_config({"llm": {"mode": "banana", "enabled": True}})
        try:
            cfg = load_config(path)
        finally:
            os.remove(path)
        self.assertEqual(cfg.mode, "explain")
        self.assertTrue(any("mode" in w for w in cfg.warnings), cfg.warnings)

    def test_invalid_timeout_falls_back(self):
        path = write_config({"llm": {"timeout": -3}})
        try:
            cfg = load_config(path)
        finally:
            os.remove(path)
        self.assertEqual(cfg.timeout, 20.0)
        self.assertTrue(any("timeout" in w for w in cfg.warnings), cfg.warnings)

    def test_default_config_path_env_var_wins(self):
        target = os.path.join(tempfile.gettempdir(), "jieqi-env-config.json")
        with temp_env(JIEQI_AI_CONFIG=target):
            self.assertEqual(llm._default_config_path(), os.path.abspath(target))

    def test_default_config_path_follows_module_location(self):
        with temp_env(JIEQI_AI_CONFIG=None):
            resolved = llm._default_config_path()
        # 规则：<llm.py 所在目录>/../ai_config.json，即 Game 包目录下的 ai_config.json；
        # 该处不可用时回退到 jieqi 包内。
        expected = os.path.normpath(os.path.join(GAME_DIR, "ai_config.json"))
        allowed = {expected, os.path.normpath(os.path.join(GAME_DIR, "jieqi", "ai_config.json"))}
        self.assertIn(resolved, allowed, resolved)
        self.assertTrue(os.path.isabs(resolved))
        self.assertEqual(resolved, expected)


# ---------------------------------------------------------------------------
# 2. redacted()
# ---------------------------------------------------------------------------


class TestRedacted(unittest.TestCase):
    def test_redacted_masks_key_and_keeps_tail(self):
        cfg = make_config(api_key="sk-abcdefghijklmnop1234")
        safe = cfg.redacted()
        self.assertNotEqual(safe.api_key, cfg.api_key)
        self.assertTrue(safe.api_key.endswith("1234"))
        self.assertTrue(safe.api_key.startswith("*"))
        self.assertNotIn("abcdefghijklmnop", safe.api_key)
        self.assertEqual(cfg.api_key, "sk-abcdefghijklmnop1234")  # 原对象未被改动

    def test_redacted_keeps_other_fields(self):
        cfg = make_config(model="m1", timeout=9.0, extra_headers={"A": "b"})
        safe = cfg.redacted()
        self.assertEqual(safe.model, "m1")
        self.assertEqual(safe.timeout, 9.0)
        self.assertEqual(safe.extra_headers, {"A": "b"})
        self.assertEqual(safe.enabled, True)

    def test_redacted_handles_empty_and_short_keys(self):
        self.assertEqual(make_config(api_key="").redacted().api_key, "")
        short = make_config(api_key="abc").redacted().api_key
        self.assertNotIn("abc", short)

    def test_redacted_to_dict_is_json_serialisable(self):
        cfg = make_config(api_key="sk-abcdefghijklmnop1234")
        blob = json.dumps(cfg.redacted().to_dict(redact=True), ensure_ascii=False)
        self.assertIn("model", blob)
        self.assertNotIn("abcdefghijklmnop", blob)


# ---------------------------------------------------------------------------
# 3. render_position_text / build_messages
# ---------------------------------------------------------------------------


class TestRenderPositionText(unittest.TestCase):
    def test_full_snapshot_contains_all_sections(self):
        text = render_position_text(FULL_SNAPSHOT)
        self.assertIn("局面快照", text)
        self.assertIn("轮到：红方", text)
        self.assertIn("AI 执：黑方", text)
        self.assertIn("第 17 手", text)
        self.assertIn("c7", text)
        self.assertIn("红兵", text)
        self.assertIn("黑卒", text)
        self.assertIn("炮二平五", text)
        self.assertIn("c5->e5", text)
        self.assertIn("主变", text)
        self.assertIn("18", text)
        self.assertIn("54%", text)
        self.assertIn("红方稍优", text)
        self.assertIn("深度=6", text)
        self.assertIn("节点=41000", text)

    def test_empty_snapshot_does_not_raise_and_is_sensible(self):
        text = render_position_text({})
        self.assertIn("轮到：未知", text)
        self.assertIn("引擎候选着法", text)
        self.assertIn("未提供", text)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 10)

    def test_none_and_weird_snapshot_do_not_raise(self):
        for value in (None, [], 0, "abc", {"board": "oops"}, {"board": [1, 2, 3]}):
            with self.subTest(value=value):
                self.assertIsInstance(render_position_text(value), str)

    def test_board_render_fallback_when_no_board_text(self):
        snap = {"board": [[None] * 9 for _ in range(10)]}
        text = render_position_text(snap)
        self.assertIn("由快照 board 渲染", text)

    def test_include_board_false_skips_rendered_board(self):
        snap = {"board": [[{"side": "red", "home": "車"}] + [None] * 8]}
        self.assertNotIn("由快照 board 渲染", render_position_text(snap, include_board=False))

    def test_round_move_notation_when_winrate_is_percent(self):
        snap = {
            "engine_candidates": [{"from": [0, 1], "to": [2, 2], "winrate": 62.0, "score": -3}]
        }
        self.assertIn("62%", render_position_text(snap))


class TestBuildMessages(unittest.TestCase):
    def test_full_snapshot_messages(self):
        messages = build_messages(FULL_SNAPSHOT, "explain")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("JSON", messages[0]["content"])
        self.assertIn("move", messages[0]["content"])
        self.assertIn("b3c3", messages[0]["content"])
        self.assertIn("confidence", messages[0]["content"])
        self.assertIn("红方稍优", messages[1]["content"])
        # 消息内容可 JSON 序列化（真实请求需要）
        json.dumps(messages, ensure_ascii=False)

    def test_empty_snapshot_messages(self):
        messages = build_messages({}, "explain")
        self.assertEqual(len(messages), 2)
        self.assertTrue(all(item["content"].strip() for item in messages))
        json.dumps(messages, ensure_ascii=False)

    def test_style_validation_and_fallback(self):
        for style in ("explain", "suggest", "review"):
            with self.subTest(style=style):
                self.assertEqual(len(build_messages(FULL_SNAPSHOT, style)), 2)
        bad = build_messages(FULL_SNAPSHOT, "not-a-style")
        default = build_messages(FULL_SNAPSHOT, "explain")
        self.assertEqual(bad, default)
        self.assertEqual(build_messages(FULL_SNAPSHOT, None), default)

    def test_style_hint_is_reflected(self):
        suggest = build_messages(FULL_SNAPSHOT, "suggest")[1]["content"]
        review = build_messages(FULL_SNAPSHOT, "review")[1]["content"]
        self.assertIn("只给出推荐着法", suggest)
        self.assertIn("复盘", review)


# ---------------------------------------------------------------------------
# 4. parse_response
# ---------------------------------------------------------------------------


class TestParseResponse(unittest.TestCase):
    def test_clean_json(self):
        text = json.dumps(
            {
                "move": "(4,2)->(4,4)",
                "comment": "炮二平五，正中路进攻。",
                "confidence": 0.62,
                "tags": ["进攻", "抢先"],
            },
            ensure_ascii=False,
        )
        parsed = parse_response(text)
        self.assertEqual(parsed["move"], (4, 2, 4, 4))
        self.assertEqual(parsed["comment"], "炮二平五，正中路进攻。")
        self.assertAlmostEqual(parsed["confidence"], 0.62)
        self.assertEqual(parsed["tags"], ["进攻", "抢先"])
        self.assertEqual(parsed["raw"], text)

    def test_json_in_code_fences(self):
        text = (
            "```json\n"
            '{"move": "(0,1)->(2,2)", "comment": "飞象。", "confidence": 0.4, "tags": []}\n'
            "```"
        )
        parsed = parse_response(text)
        self.assertEqual(parsed["move"], (0, 1, 2, 2))
        self.assertEqual(parsed["comment"], "飞象。")
        self.assertEqual(parsed["tags"], [])

    def test_json_with_leading_prose(self):
        text = (
            "好的，我的分析如下：\n"
            '{"move": "(7,1)->(7,4)", "comment": "平炮占中。", "confidence": 0.5, "tags": ["稳健"]}\n'
            "以上就是我的建议。"
        )
        parsed = parse_response(text)
        self.assertEqual(parsed["move"], (7, 1, 7, 4))
        self.assertEqual(parsed["comment"], "平炮占中。")
        self.assertEqual(parsed["tags"], ["稳健"])

    def test_regex_fallback_move_only(self):
        parsed = parse_response("我建议走 (4,2)->(4,4) 这一步。")
        self.assertEqual(parsed["move"], (4, 2, 4, 4))
        self.assertIn("(4,2)->(4,4)", parsed["comment"])
        self.assertEqual(parsed["tags"], [])

    def test_parses_the_project_notation_b3c3(self):
        """模型按**现在的提示词**写的 ``b3c3`` 必须能解析出来。

        ⚠ 这是用户报的"启动了的大模型从不给招法"的真凶：
        提示词早就改成 ``b3c3`` 了，而解析器只认老写法 ``(r,c)->(r,c)``，
        于是 ``{"move": "c6c5"}`` 被解析成 ``None`` →
        ``analyse_llm`` 判定"大模型没有给出着法" → 每步都退回本机引擎。
        """
        # 直接用 JSON 形式（模型实际返回的样子）
        parsed = parse_response('{"move": "c6c5", "comment": "挺兵。",'
                                ' "confidence": 0.55, "tags": ["开局"]}')
        self.assertEqual(parsed["move"], (3, 2, 4, 2))
        self.assertEqual(parsed["comment"], "挺兵。")
        self.assertEqual(parsed["tags"], ["开局"])
        # 散文里出现也要能抓出来
        self.assertEqual(parse_response("我建议走 b9a7 这一步。")["move"],
                         (0, 1, 2, 0))
        # 原地不动不算着法
        self.assertIsNone(parse_response('{"move": "c6c6"}')["move"])
        # 老写法仍然兼容（老棋谱/老提示词的模型回复）
        self.assertEqual(parse_response('{"move": "(4,2)->(4,4)"}')["move"],
                         (4, 2, 4, 4))

    def test_regex_fallback_null_move(self):
        parsed = parse_response('{"move": null, "comment": "无好棋。", "confidence": 0.1}')
        self.assertIsNone(parsed["move"])
        self.assertEqual(parsed["comment"], "无好棋。")

    def test_total_garbage(self):
        for text in ("@@@@", "我觉得这棋很难说。", "}{", "null", "[1,2,3]"):
            with self.subTest(text=text):
                parsed = parse_response(text)
                self.assertIsNone(parsed["move"])
                self.assertIsInstance(parsed["comment"], str)
                self.assertIsInstance(parsed["tags"], list)
                self.assertIsNone(parsed["confidence"])

    def test_empty_string_and_none(self):
        for text in ("", "   ", None, 0):
            with self.subTest(text=text):
                parsed = parse_response(text)
                self.assertEqual(parsed["move"], None)
                self.assertEqual(parsed["comment"], "")
                self.assertEqual(parsed["tags"], [])
                self.assertIsInstance(parsed["raw"], str)

    def test_nested_json_object_uses_outermost(self):
        text = '说明：{"move": "(1,1)->(1,4)", "meta": {"x": 1}, "comment": "平车。"} 完。'
        parsed = parse_response(text)
        self.assertEqual(parsed["move"], (1, 1, 1, 4))
        self.assertEqual(parsed["comment"], "平车。")

    def test_confidence_as_percent_and_string(self):
        parsed = parse_response('{"move": "(0,0)->(0,1)", "comment": "c", "confidence": 80}')
        self.assertAlmostEqual(parsed["confidence"], 0.8)
        parsed = parse_response('{"move": "(0,0)->(0,1)", "comment": "c", "confidence": "0.25"}')
        self.assertAlmostEqual(parsed["confidence"], 0.25)
        parsed = parse_response('{"move": "(0,0)->(0,1)", "comment": "c", "confidence": 5}')
        self.assertLessEqual(parsed["confidence"], 1.0)

    def test_move_as_four_tuple_or_pair_of_pairs(self):
        self.assertEqual(
            parse_response('{"move": [4, 2, 4, 4], "comment": "x"}')["move"], (4, 2, 4, 4)
        )
        self.assertEqual(
            parse_response('{"move": [[4, 2], [4, 4]], "comment": "x"}')["move"], (4, 2, 4, 4)
        )
        self.assertEqual(
            parse_response(
                '{"move": {"from": [4, 2], "to": [4, 4]}, "comment": "x"}'
            )["move"],
            (4, 2, 4, 4),
        )


# ---------------------------------------------------------------------------
# 5. suggest()
# ---------------------------------------------------------------------------


class TestSuggest(unittest.TestCase):
    def test_happy_path(self):
        content = json.dumps(
            {
                "move": "(4,2)->(4,4)",
                "comment": "炮二平五，中路进攻。",
                "confidence": 0.62,
                "tags": ["进攻"],
            },
            ensure_ascii=False,
        )
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse(openai_body(content))

        advisor = LLMAdvisor(make_config(timeout=5.0))
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen) as patched:
            result = advisor.suggest(FULL_SNAPSHOT, "explain")

        self.assertEqual(patched.call_count, 1)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.error, "")
        self.assertEqual(result.move, (4, 2, 4, 4))
        self.assertEqual(result.move_text(), "c5e5")
        self.assertEqual(result.comment, "炮二平五，中路进攻。")
        self.assertAlmostEqual(result.confidence, 0.62)
        self.assertEqual(result.tags, ["进攻"])
        self.assertEqual(result.model, "test-model")
        self.assertEqual(result.usage.get("total_tokens"), 30)
        self.assertGreaterEqual(result.latency, 0.0)

        request = captured["request"]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer %s" % FAKE_KEY)
        self.assertEqual(captured["timeout"], 5.0)
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(len(body["messages"]), 2)
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["max_tokens"], 700)

    def test_extra_headers_are_sent(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["request"] = request
            return FakeResponse(openai_body('{"move": null, "comment": "无"}'))

        cfg = make_config(extra_headers={"X-Custom": "hello", "Authorization": "Bearer override"})
        advisor = LLMAdvisor(cfg)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = advisor.suggest({})
        self.assertTrue(result.ok)
        self.assertEqual(captured["request"].get_header("X-custom"), "hello")
        # extra_headers 中的 Authorization 不被自动 Bearer 覆盖
        self.assertEqual(captured["request"].get_header("Authorization"), "Bearer override")

    def test_base_url_already_ending_with_endpoint(self):
        advisor = LLMAdvisor(make_config(base_url="https://x.test/v1/chat/completions"))
        self.assertEqual(advisor.endpoint(), "https://x.test/v1/chat/completions")
        advisor2 = LLMAdvisor(make_config(base_url="https://x.test/v1/"))
        self.assertEqual(advisor2.endpoint(), "https://x.test/v1/chat/completions")

    def test_http_401_returns_chinese_error(self):
        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}')
            )

        advisor = LLMAdvisor(make_config())
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = advisor.suggest(FULL_SNAPSHOT)
        self.assertFalse(result.ok)
        self.assertTrue(result.error)
        self.assertIn("401", result.error)
        self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in result.error), result.error)
        self.assertIsNone(result.move)

    def test_timeout_returns_chinese_error(self):
        def fake_urlopen(request, timeout=None):
            raise socket.timeout("timed out")

        advisor = LLMAdvisor(make_config(timeout=1.0))
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = advisor.suggest(FULL_SNAPSHOT)
        self.assertFalse(result.ok)
        self.assertIn("超时", result.error)

    def test_url_error_returns_chinese_error(self):
        def fake_urlopen(request, timeout=None):
            raise urllib.error.URLError("域名解析失败")

        advisor = LLMAdvisor(make_config())
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = advisor.suggest(FULL_SNAPSHOT)
        self.assertFalse(result.ok)
        self.assertIn("无法连接", result.error)

    def test_malformed_body_returns_chinese_error(self):
        advisor = LLMAdvisor(make_config())
        with mock.patch(
            "urllib.request.urlopen", return_value=FakeResponse("<html>not json</html>")
        ):
            result = advisor.suggest(FULL_SNAPSHOT)
        self.assertFalse(result.ok)
        self.assertIn("JSON", result.error)
        self.assertIn("not json", result.text)

    def test_empty_body_returns_error(self):
        advisor = LLMAdvisor(make_config())
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(b"")):
            result = advisor.suggest(FULL_SNAPSHOT)
        self.assertFalse(result.ok)
        self.assertIn("空响应体", result.error)

    def test_missing_choices_returns_error(self):
        advisor = LLMAdvisor(make_config())
        with mock.patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(json.dumps({"id": "x", "object": "chat.completion"})),
        ):
            result = advisor.suggest(FULL_SNAPSHOT)
        self.assertFalse(result.ok)
        self.assertIn("choices", result.error)

    def test_api_error_envelope_returns_error(self):
        body = json.dumps({"error": {"message": "配额不足", "type": "quota"}}, ensure_ascii=False)
        advisor = LLMAdvisor(make_config())
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(body)):
            result = advisor.suggest(FULL_SNAPSHOT)
        self.assertFalse(result.ok)
        self.assertIn("配额不足", result.error)

    def test_missing_api_key_never_calls_network(self):
        advisor = LLMAdvisor(make_config(api_key="", enabled=True))
        with mock.patch("urllib.request.urlopen") as patched:
            result = advisor.suggest(FULL_SNAPSHOT)
        self.assertFalse(result.ok)
        self.assertEqual(patched.call_count, 0)
        self.assertIn("API Key", result.error)

    def test_disabled_advisor_is_not_available(self):
        self.assertFalse(LLMAdvisor(make_config(enabled=False)).available())
        self.assertFalse(LLMAdvisor(make_config(api_key="")).available())
        self.assertTrue(LLMAdvisor(make_config()).available())

    def test_history_and_last_result(self):
        advisor = LLMAdvisor(make_config())
        with mock.patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(openai_body('{"move": null, "comment": "c"}')),
        ):
            for _ in range(3):
                advisor.suggest(FULL_SNAPSHOT)
        self.assertEqual(len(advisor.history), 3)
        self.assertIs(advisor.last_result, advisor.history[-1])
        keyless = LLMAdvisor(make_config(api_key=""))
        for _ in range(60):
            keyless.suggest(FULL_SNAPSHOT)  # 缺 key 的失败路径同样会记录历史
        self.assertEqual(len(keyless.history), 50)

    def test_unexpected_exception_is_swallowed(self):
        advisor = LLMAdvisor(make_config())
        with mock.patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            result = advisor.suggest(FULL_SNAPSHOT)  # 不应抛异常
        self.assertFalse(result.ok)
        self.assertIn("boom", result.error)

    def test_result_dataclass_defaults(self):
        result = LLMResult()
        self.assertFalse(result.ok)
        self.assertIsNone(result.move)
        self.assertEqual(result.comment, "")
        self.assertEqual(result.tags, [])
        self.assertIsNone(result.move_text())


# ---------------------------------------------------------------------------
# 6. suggest_async()
# ---------------------------------------------------------------------------


class TestSuggestAsync(unittest.TestCase):
    def test_returns_started_thread_and_calls_callback(self):
        content = json.dumps(
            {"move": "(4,2)->(4,4)", "comment": "好棋。", "confidence": 0.7, "tags": []},
            ensure_ascii=False,
        )
        advisor = LLMAdvisor(make_config())
        done = threading.Event()
        release = threading.Event()
        observed = {}
        box = {}

        def callback(result):
            box["result"] = result
            done.set()

        def slow_urlopen(request, timeout=None):
            release.wait(5.0)  # 把线程卡在网络请求上，便于观察"已启动"状态
            return FakeResponse(openai_body(content))

        with mock.patch("urllib.request.urlopen", side_effect=slow_urlopen):
            thread = advisor.suggest_async(FULL_SNAPSHOT, callback)
            observed["returned_alive"] = thread.is_alive()
            release.set()
            self.assertIsInstance(thread, threading.Thread)
            self.assertTrue(thread.daemon)
            self.assertTrue(done.wait(5.0), "回调未在 5 秒内被调用")
            thread.join(timeout=5.0)
        self.assertTrue(observed["returned_alive"], "返回时线程应当已在运行")

        self.assertFalse(thread.is_alive(), "线程应已结束")
        result = box.get("result")
        self.assertIsNotNone(result)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.move, (4, 2, 4, 4))
        self.assertIs(advisor.last_result, result)

    def test_callback_optional_and_failures_do_not_break_thread(self):
        advisor = LLMAdvisor(make_config(api_key=""))
        with mock.patch("urllib.request.urlopen") as patched:
            thread = advisor.suggest_async(FULL_SNAPSHOT)
            thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(patched.call_count, 0)
        self.assertIsNotNone(advisor.last_result)
        self.assertFalse(advisor.last_result.ok)

    def test_callback_exception_is_ignored(self):
        advisor = LLMAdvisor(make_config(api_key=""))
        called = threading.Event()

        def bad_callback(result):
            called.set()
            raise ValueError("callback 内部错误")

        thread = advisor.suggest_async({}, bad_callback)
        self.assertTrue(called.wait(5.0))
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())


# ---------------------------------------------------------------------------
# 7. blend_bonus
# ---------------------------------------------------------------------------

CANDIDATES = [
    {"from": [4, 2], "to": [4, 4], "score": 18},
    {"from": [7, 1], "to": [7, 4], "score": 8},
]


class TestBlendBonus(unittest.TestCase):
    def test_weight_zero_is_all_zeros(self):
        result = LLMResult(
            ok=True,
            parsed={"move": (4, 2, 4, 4), "confidence": 0.9, "comment": "c", "tags": []},
        )
        table = blend_bonus(result, CANDIDATES, 0.0)
        self.assertEqual(
            table, {((4, 2), (4, 4)): 0.0, ((7, 1), (7, 4)): 0.0}
        )
        self.assertTrue(all(value == 0.0 for value in table.values()))
        # None / 负数权重同样不产生任何影响
        self.assertTrue(
            all(value == 0.0 for value in blend_bonus(result, CANDIDATES, None).values())
        )
        self.assertTrue(
            all(value == 0.0 for value in blend_bonus(result, CANDIDATES, -1.0).values())
        )

    def test_weight_positive_gives_bonus_to_matching_candidate_only(self):
        result = LLMResult(
            ok=True,
            parsed={"move": (4, 2, 4, 4), "confidence": 0.5, "comment": "c", "tags": []},
        )
        table = blend_bonus(result, CANDIDATES, 1.0)
        self.assertGreater(table[((4, 2), (4, 4))], 0.0)
        self.assertEqual(table[((7, 1), (7, 4))], 0.0)
        # 上限：weight=1, confidence=1 → max_bonus
        strong = blend_bonus(
            LLMResult(ok=True, parsed={"move": (4, 2, 4, 4), "confidence": 1.0}),
            CANDIDATES,
            1.0,
        )
        self.assertLessEqual(strong[((4, 2), (4, 4))], llm.BLEND_MAX_BONUS_CP)
        self.assertAlmostEqual(strong[((4, 2), (4, 4))], llm.BLEND_MAX_BONUS_CP)

    def test_weight_scales_monotonically_and_weight_is_capped(self):
        move = {"move": (4, 2, 4, 4), "confidence": 1.0}
        low = blend_bonus(move, CANDIDATES, 0.2)[((4, 2), (4, 4))]
        high = blend_bonus(move, CANDIDATES, 0.8)[((4, 2), (4, 4))]
        self.assertGreater(high, low)
        self.assertLessEqual(
            blend_bonus(move, CANDIDATES, 10.0)[((4, 2), (4, 4))], llm.BLEND_MAX_BONUS_CP
        )

    def test_non_candidate_move_gets_nothing(self):
        move = {"move": (0, 0, 0, 1), "confidence": 1.0}
        table = blend_bonus(move, CANDIDATES, 1.0)
        self.assertEqual(sorted(table.values()), [0.0, 0.0])

    def test_garbage_inputs_do_not_raise(self):
        for result in (None, "", 0, object(), {"move": "???"}, LLMResult(ok=False)):
            with self.subTest(result=result):
                table = blend_bonus(result, CANDIDATES, 0.5)
                self.assertEqual(sorted(table.values()), [0.0, 0.0])
        for candidates in (None, [], "oops", [None, 3, {"from": [1]}], [{}]):
            with self.subTest(candidates=candidates):
                self.assertIsInstance(blend_bonus({"move": (1, 1, 1, 2)}, candidates, 0.5), dict)

    def test_accepts_move_tuple_and_move_string(self):
        self.assertGreater(
            blend_bonus((4, 2, 4, 4), CANDIDATES, 1.0)[((4, 2), (4, 4))], 0.0
        )
        self.assertGreater(
            blend_bonus("(4,2)->(4,4)", CANDIDATES, 1.0)[((4, 2), (4, 4))], 0.0
        )

    def test_confidence_defaults_when_missing(self):
        table = blend_bonus({"move": (4, 2, 4, 4)}, CANDIDATES, 1.0)
        self.assertGreater(table[((4, 2), (4, 4))], 0.0)


# ---------------------------------------------------------------------------
# 8. CLI --dry-run --demo
# ---------------------------------------------------------------------------


class TestCli(unittest.TestCase):
    CLI = os.path.join(GAME_DIR, "llm_advisor.py")

    def _isolated_config(self):
        """一份「未启用」的临时配置。

        必须隔离：否则这些测试会读到 Game/ai_config.json 里用户真实的
        enabled/api_key —— 既会让「不可用应退出 2」的断言失效，
        更糟的是会真的发出联网请求、花掉用户的额度。
        """
        if getattr(self, "_cfg_path", None) is None:
            fd, path = tempfile.mkstemp(suffix="-jieqi-ai-config.json")
            os.close(fd)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"llm": {"enabled": False, "api_key": ""}}, fh)
            self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
            self._cfg_path = path
        return self._cfg_path

    def run_cli(self, *args, cwd=None, isolated=True):
        env = dict(os.environ)
        for name in (
            "JIEQI_AI_CONFIG",
            "JIEQI_LLM_BASE_URL",
            "JIEQI_LLM_MODEL",
            "JIEQI_LLM_ENABLED",
            "JIEQI_LLM_MODE",
            "JIEQI_LLM_TIMEOUT",
            "JIEQI_LLM_BLEND_WEIGHT",
            "OPENAI_API_KEY",
        ):
            env.pop(name, None)
        # 强制子进程以 UTF-8 输出，避免 Windows 代码页把中文变成乱码。
        env["PYTHONIOENCODING"] = "utf-8"
        if isolated and "--config" not in args:
            env["JIEQI_AI_CONFIG"] = self._isolated_config()
        return subprocess.run(
            [sys.executable, self.CLI, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd or STAGE_DIR,
            env=env,
            timeout=120,
        )

    def test_demo_dry_run_exit_zero_and_contains_endpoint(self):
        proc = self.run_cli("--demo", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("chat/completions", proc.stdout)
        self.assertIn("局面快照", proc.stdout)
        self.assertIn("(4,2)->(4,4)", proc.stdout)
        # 配置以脱敏形式打印
        self.assertIn("api_key", proc.stdout)
        self.assertIn("炮二平五", proc.stdout)

    def test_demo_json_dry_run_is_valid_json_with_payload(self):
        proc = self.run_cli("--demo", "--json", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertTrue(data["dry_run"])
        payload = data["payload"]
        self.assertIn("messages", payload)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIn("model", data["config"])
        self.assertIn("api_key", data["config"])
        self.assertIn("chat/completions", data["endpoint"])
        # JSON 输出不应被人类可读文本污染
        self.assertNotIn("=== 局面文本 ===", proc.stdout)

    def test_runs_from_another_cwd(self):
        proc = self.run_cli("--demo", "--json", "--dry-run", cwd=tempfile.gettempdir())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["dry_run"])

    def test_style_flag_is_respected(self):
        proc = self.run_cli("--demo", "--json", "--dry-run", "--style", "review")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["style"], "review")

    def test_missing_config_file_exits_2(self):
        proc = self.run_cli("--config", os.path.join(tempfile.gettempdir(), "nope-jieqi.json"))
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)

    def test_not_available_exits_2_without_network(self):
        proc = self.run_cli("--demo")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("API Key", proc.stderr + proc.stdout)
        # 硬性保证：这条测试永远不许真的联网成功
        self.assertNotIn("状态：成功", proc.stdout + proc.stderr)

    def test_demo_snapshot_is_json_serialisable(self):
        script = (
            "import json, sys;"
            "sys.path.insert(0, r'%s');"
            "import llm_advisor;"
            "print(json.dumps(llm_advisor.DEMO_SNAPSHOT, ensure_ascii=False))" % GAME_DIR
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        snapshot = json.loads(proc.stdout)
        self.assertIn("engine_candidates", snapshot)
        self.assertEqual(len(snapshot["board"]), 10)
        self.assertTrue(all(len(row) == 9 for row in snapshot["board"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
