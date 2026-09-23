# 测试

只有一个测试目录 `tests/`，里面两套，分别针对**两棵同名但已分叉的模块树**：

| 子目录 | 被测代码 | 谁在跑 | 数量 |
| --- | --- | --- | --- |
| `tests/root/` | 仓库根目录那棵（`geometry/ filtering/ tracking/ classification/ pipeline/`）—— Truck / VRU / 入口编排 | 本进程直接跑 | 136 |
| `tests/main_chain/` | `main_chain/` 那棵 —— 合并车链里 **Car 的 step2~step5 + region/** | 由 `tests/root/test_main_chain_suite.py` 用**独立子进程**拉起 | 149 |

```bash
pytest                  # 全部（137 = 136 原地 + 1 个转发用例，子进程里再跑 149）
pytest tests/root       # 只跑根目录那棵
pytest tests/main_chain # 只跑 Car 共享阶段那棵（调试用，最常跑）
```

## 为什么要分子进程

两棵树里的 `geometry / filtering / tracking / classification / pipeline / region`
**同名但内容已分叉**（例如 `geometry/static_yaw.py`：根目录那份没有逐观测静止门控，
`main_chain/` 那份有 —— Car 走的是后者）。一个 Python 进程里同一个包名只能解析到一棵树。
而且：

* `main_chain` 的用例里有字符串打桩（`mock.patch("region.retrack._direction_assignments")`），
  运行期按 `sys.modules` 解析目标模块；
* 根目录那棵有函数体内的延迟 import（如 `geometry/truck_trailer_rules.py:78`）。

两者在同一个进程里会互相踩（现象是「单独跑都绿、一起跑挂几个」）。所以：
`tests/root/` 原地跑（`tests/conftest.py` 把仓库根放进 `sys.path`），
`tests/main_chain/` 交给子进程（`tests/main_chain/conftest.py` 把 `main_chain` 放进 `sys.path`）。

## pytest.ini 里的两处特殊配置

1. `--import-mode=importlib`：两个子目录里有同名测试文件（`test_step2_stages.py`、
   `test_step3_roof_fit.py` 等），默认的 prepend 模式会因为模块重名直接报错。
2. `-p no:launch_testing -p no:ament_*`：**openpcdet 环境里装了 ROS 的 pytest 插件**，
   它们会在收集阶段把每个 `.py` 当 launch test 去 import，直接干扰本仓库的测试。
   换环境跑测试时如果又出现奇怪的收集错误，先看这里。

## 依赖

```bash
/home/moga/miniconda3/envs/openpcdet/bin/python -m pytest -q
```

根目录 `tests/root/test_bevfusion_*.py` 会用 `importlib` 直接加载 `bevfusion/scripts/*.py`
与 `bevfusion/configs/*.py`（不走包导入），所以它们只校验文件存在与环境变量，不需要 mmdet3d 环境。
