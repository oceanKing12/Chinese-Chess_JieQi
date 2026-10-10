"""把工作区里开发好的揭棋 AI 部署到 D:\\py\\Chinese Chess。

只做「备份 + 覆盖 / 新增」，不删除任何原有文件；
唯一的删除动作是清理两个明确列出的 __pycache__ 缓存目录（有断言保护）。
重复运行结果一致（幂等）。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SRC = Path(r"D:\py\揭棋\stage")
DST = Path(r"D:\py\Chinese Chess")
GAME_SRC = SRC / "Game"
GAME_DST = DST / "Game"
BACKUP = DST / "_backup_before_ai_optimize"

BACKUP_FILES = [
    "AI.py", "main.py", "game_state.py", "config.py", "input_handler.py",
    "renderer.py", "resource_manager.py", "position_tables.py", "aiplayer.py",
    "tree.html", "decision_tree.json",
]

GAME_FILES = [
    "AI.py", "AI优化说明.md", "自对弈训练准备.md",
    "ai_config.json",
    "ai_config.llm.example.json", "llm_advisor.py", "record_tool.py",
    "main.py", "game_state.py", "config.py", "input_handler.py",
    "renderer.py", "start_screen.py", "resource_manager.py",
]

#: ``Game`` 下**所有** ``*.py`` 都要一起部署（引擎主流程 + 启动页 + 渲染…）。
#: 用 glob 而不是只靠手写清单：新加一个模块忘了写进清单 = 部署出去少个文件，
#: 报错现场是"ImportError: No module named xxx"这种看不出所以然的样子。
GAME_PY_GLOB = "*.py"

TOOL_FILES = ["selfplay.py", "gui_smoke.py", "ui_preview.py",
              "legality_audit.py", "debug_legal_filter.py",
              # 棋谱目录护栏：任何清理都先归档、不直接删（详见文件头说明）
              "record_safety.py"]

#: 工作区里用于人工检查大模型提示词 / 归属质检 / 引擎体检的脚本（部署到 ``_bench``）。
BENCH_FILES = ["_paths.py", "migrate_paths.py",
               "preview_llm_prompt.py", "check_llm_ownership.py", "run_one.py",
               "bench_engine_depth.py", "ab_selfplay.py", "diff_eval.py",
               "smoke_record.py", "replay_log.py", "check_ui_buttons.py",
               "gen_training_data.py", "deploy_selftest.py",
               "diagnose_game.py", "check_mate_blindness.py",
               "check_hidden_piece_play.py", "ab_hidden_value.py",
               "check_replay_match.py",
               "bench_eval_sections.py", "check_eval_accuracy.py",
               "tune_reveal.py", "bench_attack_map.py",
               # 老配置兼容性体检：ai_config.json 是受保护文件，线上那份里
               # 一定还留着已移除功能的字段，这个脚本专门验证"照样能起来"
               "check_legacy_config.py"]

#: 这些文件一旦在目标目录存在就**不覆盖**——它们保存着用户自己的设置
#: （尤其是 llm.api_key）。否则每次重新部署都会把用户的配置冲掉。
PROTECTED = {"ai_config.json"}

#: 期望装好的文件清单（部署完逐个校验）
EXPECTED = (
    [f"Game/jieqi/{n}" for n in (
        "__init__.py", "consts.py", "board.py", "belief.py", "evaluate.py",
        "search.py", "engine.py", "gui_adapter.py", "llm.py", "settings.py")]
    + [f"Game/{n}" for n in GAME_FILES]
    + [f"Game/{p.name}" for p in sorted((GAME_SRC).glob(GAME_PY_GLOB))]
    + [f"tests/{p.name}" for p in sorted((SRC / "tests").glob("test_*.py"))]
    + [f"tools/{n}" for n in TOOL_FILES]
    + [f"_bench/{n}" for n in BENCH_FILES]
)

#: 允许删除的缓存目录（绝对路径白名单，避免误删）
CACHE_DIRS = [GAME_DST / "__pycache__", GAME_DST / "jieqi" / "__pycache__"]


def fail(msg: str) -> None:
    print(f"错误：{msg}")
    sys.exit(1)


def main() -> int:
    if not GAME_SRC.is_dir():
        fail(f"源目录不存在：{GAME_SRC}")
    if not GAME_DST.is_dir():
        fail(f"目标目录不存在：{GAME_DST}")

    # ------------------------------------------------------------ 1) 备份
    # 只做一次性备份：如果备份里已经有了，就绝不覆盖，
    # 这样重复部署也不会把「原始版本」冲掉。
    BACKUP.mkdir(parents=True, exist_ok=True)
    n_backup = 0
    for name in BACKUP_FILES:
        p = GAME_DST / name
        target = BACKUP / name
        if p.is_file() and not target.exists():
            shutil.copy2(p, target)
            n_backup += 1
    print(f"1. 备份原始文件 {n_backup} 个（已存在的备份不覆盖） -> {BACKUP}")

    # -------------------------------------------------------- 2) 引擎包
    pkg_dst = GAME_DST / "jieqi"
    pkg_dst.mkdir(parents=True, exist_ok=True)
    engine = sorted((GAME_SRC / "jieqi").glob("*.py"))
    if not engine:
        fail(f"引擎源码为空：{GAME_SRC / 'jieqi'}")
    for f in engine:
        shutil.copy2(f, pkg_dst / f.name)
    print(f"2. 安装引擎模块 {len(engine)} 个 -> {pkg_dst}")

    # ------------------------------------------------- 3) Game 下的文件
    skipped = []
    installed_names = set()
    for name in GAME_FILES:
        s = GAME_SRC / name
        if not s.is_file():
            fail(f"缺少源文件：{s}")
        target = GAME_DST / name
        if name in PROTECTED and target.exists():
            skipped.append(name)
            continue
        shutil.copy2(s, target)
        installed_names.add(name)
    # 3b) Game 下其余的 *.py 全部同步（新增模块不会因为忘了写清单而漏掉）
    for s in sorted(GAME_SRC.glob(GAME_PY_GLOB)):
        if s.name in installed_names or s.name in skipped:
            continue
        shutil.copy2(s, GAME_DST / s.name)
    print(f"3. 安装 Game 文件 {len(GAME_FILES) - len(skipped)} 个"
          + (f"（跳过已存在的 {', '.join(skipped)}，保留你自己的配置）" if skipped else ""))
    print(f"4. 说明文档：{GAME_DST / 'AI优化说明.md'}")

    # ----------------------------------------------------- 4) tests/tools
    tests_dst = DST / "tests"
    tools_dst = DST / "tools"
    tests_dst.mkdir(parents=True, exist_ok=True)
    tools_dst.mkdir(parents=True, exist_ok=True)
    tests = sorted((SRC / "tests").glob("test_*.py"))
    for f in tests:
        shutil.copy2(f, tests_dst / f.name)
    for name in TOOL_FILES:
        shutil.copy2(SRC / "tools" / name, tools_dst / name)
    print(f"5. 安装测试 {len(tests)} 个 -> {tests_dst}，工具 {len(TOOL_FILES)} 个 -> {tools_dst}")

    # ------------------------------------------- 5b) _bench（人工检查用脚本）
    bench_src = SRC.parent / "_bench"
    bench_dst = DST / "_bench"
    bench_dst.mkdir(parents=True, exist_ok=True)
    installed = 0
    for name in BENCH_FILES:
        source = bench_src / name
        if source.is_file():
            shutil.copy2(source, bench_dst / name)
            installed += 1
    print(f"5b. 安装提示词/质检自检脚本 {installed} 个 -> {bench_dst}")

    # ------------------------------------------------------------ 5) 校验
    print("\n=== 部署结果 ===")
    missing = []
    for rel in EXPECTED:
        p = DST / rel
        if p.is_file():
            print(f"  OK   {rel:<40} {p.stat().st_size:>8} B")
        else:
            missing.append(rel)
            print(f"  MISS {rel}")
    if missing:
        fail("部署不完整，缺少：" + ", ".join(missing))

    # ------------------------------------------- 6) 清理过期字节码缓存
    for cache in CACHE_DIRS:
        # 白名单 + 名称断言，双重确认只删缓存目录
        if cache.name != "__pycache__":
            fail(f"拒绝删除非缓存目录：{cache}")
        if cache.parent not in (GAME_DST, pkg_dst):
            fail(f"拒绝删除目标之外的目录：{cache}")
        if cache.is_dir():
            shutil.rmtree(cache)
            print(f"6. 清理过期字节码缓存：{cache}")

    print("\n部署完成。")
    print(f"回滚方法：把 {BACKUP} 里的文件复制回 {GAME_DST} 即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
