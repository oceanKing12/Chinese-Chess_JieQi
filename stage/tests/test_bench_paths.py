# -*- coding: utf-8 -*-
"""``_bench`` 工具的路径探测回归测试。

为什么值得单测：这些脚本在**开发树**（``D:\\py\\揭棋``）里一直好用，被
``deploy.py`` 复制到**部署树**（``D:\\py\\Chinese Chess``）之后，
``Path(__file__).resolve().parents[1] / "stage"`` 指向一个根本不存在的目录，
于是 ``from jieqi import record`` 直接 ``ModuleNotFoundError``——用户实际
拿到的就是"评估工具在游戏目录用不了"。修法是把路径探测抽到
``_bench/_paths.py``，两种布局都认。

这里锁三件事：

1. 探测结果落在**含有 ``Game/jieqi`` 的那棵树**上（不是凭名字猜的）；
2. ``STAGE`` / ``GAME`` 两个都要能直接 ``import jieqi``；
3. ``_bench`` 里**没有脚本再写死开发树路径**（防止以后新脚本又抄一遍老写法）。
"""
import ast
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

def _find_bench() -> Path:
    """找到 ``_bench`` 目录（两种树的深度不一样）。

    开发树：``D:\\py\\揭棋\\stage\\tests`` → 往上两层是 ``D:\\py\\揭棋``；
    部署树：``D:\\py\\Chinese Chess\\tests`` → 往上一层就够了。
    所以**不能写死 parents[2]**，要按"哪一层同时有 ``_bench`` 和 ``Game/jieqi``"
    去认（这正是被测的 ``_paths.py`` 用的同一套思路——顺手也验证了它成立）。
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "_bench").is_dir() and (parent / "Game" / "jieqi").is_dir():
            return parent / "_bench"
    for parent in Path(__file__).resolve().parents:
        if (parent / "_bench").is_dir():
            return parent / "_bench"
    return Path(__file__).resolve().parents[2] / "_bench"


BENCH = _find_bench()
GAME_DIR = Path(__file__).resolve().parents[1] / "Game"


class TestBenchPaths(unittest.TestCase):
    def test_paths_module_finds_this_tree(self):
        """``_paths`` 必须认出**当前这棵树**的 ``Game``（含 jieqi 包）。"""
        sys.path.insert(0, str(BENCH))
        try:
            import _paths
        finally:
            sys.path.pop(0)
        self.assertTrue((_paths.GAME / "jieqi" / "__init__.py").is_file(),
                        f"_paths.GAME 不含 jieqi 包：{_paths.GAME}")
        self.assertEqual(_paths.GAME.resolve(), GAME_DIR.resolve(),
                         "_paths 认到了另一棵树的 Game 目录")
        self.assertEqual(_paths.STAGE.resolve(), GAME_DIR.parent.resolve())
        self.assertTrue(_paths.JIEQI_OK)

    def test_jieqi_importable_after_add_to_path(self):
        """按文档写 ``import _paths`` 之后，``from jieqi import ...`` 必须成功。

        跑在**子进程**里：本测试进程已经把别的路径塞进 ``sys.path`` 了，
        在进程内测等于没测。
        """
        code = (
            "import sys\n"
            "sys.path.insert(0, r'%s')\n"
            "import _paths\n"
            "_paths.add_to_path()\n"
            "from jieqi import record\n"
            "from jieqi.consts import RED\n"
            "import game_state\n"
            "print(_paths.GAME)\n" % BENCH)
        env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
                   PYTHONIOENCODING="utf-8")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, encoding="utf-8", env=env, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         f"子进程导入失败：\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("Game", proc.stdout)

    def test_no_script_hardcodes_the_dev_tree(self):
        """不许再有脚本写死 ``parents[1] / "stage"`` 或某棵树的绝对路径。

        ``apply_reveal_verdict.py`` 例外：它刻意留了一条"另一棵树"的兜底路径
        （要同时清两份配置文件）。
        """
        allowed = {"_paths.py", "migrate_paths.py", "legacy_AI_original.py",
                   "apply_reveal_verdict.py"}
        bad = []
        for path in sorted(BENCH.glob("*.py")):
            if path.name in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in ('parents[1] / "stage"',
                           "揭棋\\stage\\Game", "揭棋/stage/Game",
                           'Path(r"D:\\py\\Chinese Chess\\Game")'):
                if needle in text:
                    bad.append(f"{path.name}：{needle}")
        self.assertEqual(bad, [], "这些脚本还写死了开发树路径：\n" + "\n".join(bad))

    def test_every_script_compiles(self):
        """语法体检：批量改写路径之后，每个脚本至少要是合法 Python。"""
        bad = []
        for path in sorted(BENCH.glob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                bad.append(f"{path.name}: {exc}")
        self.assertEqual(bad, [], "语法错误：\n" + "\n".join(bad))

    def test_patched_scripts_call_add_to_path(self):
        """被改写的脚本必须真的调用 ``_paths.add_to_path()``（不能只 import）。"""
        missing = []
        for path in sorted(BENCH.glob("*.py")):
            if path.name in {"_paths.py", "migrate_paths.py",
                             "legacy_AI_original.py"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "import _paths" in text and "add_to_path()" not in text:
                missing.append(path.name)
        self.assertEqual(missing, [], "import 了 _paths 却没调用 add_to_path：" +
                         ", ".join(missing))

    def test_every_script_imports(self):
        """每个脚本都要能**真的 import 进去**（模块级不能缺名字/缺模块）。

        这一条抓的是 ``NameError: name 'GAME' is not defined`` 这类问题：
        把写死路径换成 ``_paths`` 之后，脚本后面还在用 ``GAME``，
        语法检查（``ast.parse``）完全看不出来，只有真的执行模块级代码才会炸。

        ``SystemExit`` 单独放行：有几个诊断脚本的干活代码写在模块级（没有
        ``if __name__ == "__main__"``），import 会把它们整段跑完再 ``sys.exit(0)``
        ——那不是错误，只是"这个文件不适合当模块导入"。
        """
        scripts = sorted(p for p in BENCH.glob("*.py")
                         if p.name not in {"legacy_AI_original.py"})
        self.assertTrue(scripts, "没找到任何 _bench 脚本")
        code = (
            "import importlib, sys\n"
            "sys.path.insert(0, r'%s')\n"
            "bad = []\n"
            "for name in %r:\n"
            "    try:\n"
            "        importlib.import_module(name)\n"
            "    except SystemExit:\n"
            "        pass\n"
            "    except BaseException as exc:\n"
            "        bad.append('%%s: %%r' %% (name, exc))\n"
            "sys.stderr.write('@@BAD@@' + '\\n'.join(bad) + '@@END@@')\n"
            % (BENCH, [p.stem for p in scripts]))
        env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
                   PYTHONIOENCODING="utf-8")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, encoding="utf-8", env=env, timeout=600)
        # 有些脚本在 import 时就会打印自己的诊断结果（老脚本没有 main 保护），
        # 所以不能拿 stdout 的最后一行当结果——用显式的标记把结论夹出来。
        marker = re.search(r"@@BAD@@(.*?)@@END@@", proc.stderr or "", re.S)
        self.assertIsNotNone(marker, "子进程没给出结论：\n" + (proc.stderr or "")[-2000:])
        bad = [item for item in marker.group(1).splitlines() if item.strip()]
        self.assertEqual(bad, [], "这些脚本 import 就炸：\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
