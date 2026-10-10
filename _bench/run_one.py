# -*- coding: utf-8 -*-
"""按名字跑单个测试并打印完整回溯（PowerShell 里传中文比较麻烦，所以做成脚本）。

用法::

    python _bench/run_one.py test_llm_ownership TestAdvisorValidation.test_retry_with_hint_when_not_repairable
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

#: 路径探测：开发树（揭棋）与部署树（Chinese Chess）都能跑。
import _paths
_paths.add_to_path()
GAME = _paths.GAME
STAGE = _paths.STAGE
for path in (str(STAGE), str(GAME), str(STAGE / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.stdout.reconfigure(encoding="utf-8")


def main(argv) -> int:
    module_name = argv[0] if argv else "test_llm_ownership"
    names = argv[1:] or [None]
    module = __import__(module_name)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in names:
        if name:
            suite.addTests(loader.loadTestsFromName(name, module))
        else:
            suite.addTests(loader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
