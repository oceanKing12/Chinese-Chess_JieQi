# -*- coding: utf-8 -*-
"""发布相关的回归测试：**"别人 clone 下来能不能跑起来"**。

为什么单独一个文件：这三条都不是"功能 bug"，而是"只在我这台机器上能跑"型
问题 —— 开发时工作目录永远是 ``Game/``，所以相对路径一直没暴露；一旦发布到
GitHub，别人在仓库根目录敲 ``python Game/main.py`` 就会崩。这类问题必须在
测试里钉住，因为本地永远复现不出来。
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

GAME = Path(__file__).resolve().parents[1] / "Game"


class TestResourcePathsAreIndependentOfCwd(unittest.TestCase):
    """图片/资源路径必须按"文件自身位置"算，不能依赖当前工作目录。"""

    def test_image_dir_is_absolute_and_exists(self):
        sys.path.insert(0, str(GAME))
        try:
            import resource_manager as rm
        finally:
            sys.path.pop(0)
        self.assertTrue(os.path.isabs(rm.IMAGE_DIR),
                        "IMAGE_DIR 应当是绝对路径：%r" % rm.IMAGE_DIR)
        self.assertTrue(os.path.isdir(rm.IMAGE_DIR),
                        "IMAGE_DIR 不存在：%r" % rm.IMAGE_DIR)
        # 16 张棋子图（红黑各 7 种 + 2 张暗子）
        pngs = sorted(p.name for p in Path(rm.IMAGE_DIR).glob("*.png"))
        self.assertGreaterEqual(len(pngs), 16, pngs)

    def test_load_images_from_a_foreign_cwd(self):
        """把工作目录换到别处，``_load_piece_images`` 仍然要能读到图。

        这正是"在仓库根目录跑 ``Game/main.py``"的情形。
        """
        code = (
            "import sys, os\n"
            "sys.path.insert(0, %r)\n"
            "os.environ.setdefault('SDL_VIDEODRIVER','dummy')\n"
            "os.environ.setdefault('SDL_AUDIODRIVER','dummy')\n"
            "import pygame; pygame.init(); pygame.display.set_mode((1,1))\n"
            "from resource_manager import ResourceManager\n"
            "rm = ResourceManager()\n"
            "rm.load_resources()\n"
            "print('PIECES', len(rm.piece_images))\n" % str(GAME)
        )
        # 故意换到一个**完全无关**的目录（不能是 Game/，也不能是它的父目录）
        foreign = Path(os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(foreign),
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         "在外部目录加载资源失败：\n%s\n%s"
                         % (proc.stdout[-1500:], proc.stderr[-1500:]))
        self.assertIn("PIECES 16", proc.stdout,
                      "棋子图数量不对：%s" % proc.stdout[-400:])

    def test_record_dir_is_inside_the_game_package(self):
        """棋谱默认目录要落在 ``Game/records``，与工作目录无关。"""
        sys.path.insert(0, str(GAME))
        try:
            from jieqi import record as R
        finally:
            sys.path.pop(0)
        got = Path(R.record_dir()).resolve()
        want = (GAME / "records").resolve()
        self.assertEqual(got, want,
                         "棋谱目录跑偏了：%s（应为 %s）" % (got, want))


class TestNoSecretsInTree(unittest.TestCase):
    """仓库里不能有真凭据（测试用的假 key 不算）。"""

    def test_no_real_looking_secrets(self):
        import json
        import re

        bad = []
        # 1) ai_config.json 里的 api_key 必须是空的
        cfg = GAME / "ai_config.json"
        if cfg.is_file():
            data = json.loads(cfg.read_text(encoding="utf-8"))
            key = str((data.get("llm") or {}).get("api_key") or "")
            if key:
                bad.append("ai_config.json 里有 api_key（%d 字符）" % len(key))

        # 2) 源码里除测试/示例外不该出现 sk- 开头的真 key
        rx = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")
        for path in list(GAME.rglob("*.py")) + list(GAME.rglob("*.json")):
            if "test" in path.name.lower():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for m in rx.finditer(text):
                value = m.group(0)
                # 全是占位符/明显假 key 的放过
                if re.fullmatch(r"sk-(test|fake|dummy|xxx|x+)[A-Za-z0-9_\-]*",
                                value, re.I):
                    continue
                bad.append("%s: %s..." % (path.name, value[:12]))
        self.assertEqual(bad, [], "发现可能的真凭据：\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
