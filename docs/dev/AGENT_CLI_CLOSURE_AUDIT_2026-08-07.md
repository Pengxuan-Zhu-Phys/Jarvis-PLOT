# Jarvis-PLOT Agent CLI 收口核验

Status: review
Date: 2026-08-07
Reviewer: Claude
Scope: **只核验收口声明是否在代码里成立**，不提新设计、不做新方案。
基线：HEAD `7853638`，工作区干净，`432 passed / 1 skipped`。

---

## 0. 结论

**收口成立。9 条 ✅ 里 8 条完全核实，1 条（⑧）属实但措辞需要更精确。**
另外核验中发现 **1 个新的实现缺陷**（渲染期观测重复），不影响闭环成立，属打磨项。

一句话：**这次收口最值得记的不是"做完了什么"，而是 `70fc9d1 → 12a7a06` 这一对提交——
先把 `--deep` 做出来，跑通了，再因为违反纪律把它撤掉，而把同一提交里合规的执行期钩子留下。
纪律赢了已经能工作的代码，这是收口质量最硬的证据。**

---

## 1. 逐条核验

| # | 声明 | 判定 | 证据 |
|:--:|---|:--:|---|
| ① | 发布向 P0/P1 修复 | ✅ | 见 §2 |
| ② | 检查 vs 执行纪律 + 文档 | ✅ | 纪律进 `man workflow.anti_patterns`，agent 可机读 |
| ③ | 劝退物删除（context / deep） | ✅ | `verbs/context.py` 已删；`VERBS` 10 个无 context；`--deep` 全代码库无残留 |
| ④ | 缺失功能报错 → 指 CLI | ✅ | `jplot context` → stderr 一行 + exit 2，stdout 0 字节 |
| ⑤ | `_digest_axes` 画完清掉 | ✅ | 实跑：expand 后 1 处 → 渲染后 0 处，`agent_output` 本体保留 |
| ⑥ | `--report` 临时 JP-VIZ | ✅ | 写在 YAML 旁 `*.render-report.json`，`ephemeral: true` + 删除提示 |
| ⑦ | decades 写清 | ✅ | 输出带 `decades_basis: min_max` + `decades_note` 主动劝退误用 |
| ⑧ | 轻 transform 共用 | ✅ *措辞* | 见 §3 |
| ⑨ | flowchart 定位 HEP scan | ✅ | help 与 `args.json` 均为 "Jarvis-HEP project-scan flowchart only … not general plot YAML" |

### 工程状态

- 工作区 **clean**（核验当时）；remote 名为 **`jplot`**（不是 `origin`）。
- **更正（对核验报告自身）**：`git log jplot/main..HEAD` 在本机复算为 **10**（`f5bc859..7853638`），
  即 **仍 ahead 10，并非已与 `jplot/main` 同步**。核验正文 §6.1 写「0 / 已 push」**不成立**，
  以 `git branch -vv` / `jplot/main..HEAD` 为准再点一次 push。
- 432 passed / 1 skipped（核验基线）；收口后若修观测重复会再增测试。

---

## 2. ① 发布向 P0/P1 的实际落点

| 初评项 | 落点 | 核验方式 | 结果 |
|---|---|---|---|
| **P0.1** digest `top_density` 被退化 cell 支配 | `agent_digest.py:359` `degenerate = (cnt < 3) or (width <= 0) or (height <= 0)`；`:364` 退化 cell `area = 0.0`；`:438` 注释 *"Never rank degenerate cells by density"* | 读码 + 2 个测试文件含 `degenerate` | ✅ 原 `1e-30` 绝对下限已去除，改为显式 flag + 排名剔除 |
| **P0.2** `type:` 上 JP-VIZ 失明 | 双侧闭合：检查期 pre-transform 代理；执行期 `Figure/layer_runtime.py:744` post-transform 观测 | 实跑坏 `xlim` 的 `posterior_2d` | ✅ `doctor` 报 `JP-VIZ-002`（`basis: pre-transform`）；`jplot <yaml> --report` 报 post-mesh `JP-VIZ-002` |
| **P1.1** suggest log 判据 | `data_access.py:177` `decades = log10(q_hi/q_lo)`；`:178-179` `skew_ratio = median/mean`，`use_log = positive and decades >= 2.0 and skew_ratio < 0.5` | 实跑 `Uniform(0,5)` | ✅ 现给 `yscale: linear`（此前误判 log） |
| **P1.2** transform 契约漂移 | `transform_contracts.py:224` 已补 `pregrid`（含 `enable` 子键说明）；`tests/test_transform_contracts.py` 已建 | grep + 测试存在 | ✅ |
| **P1.3** agent 通道 fork 流水线 | 新 `data_access.py`（852 行）；`verbs/data.py` 1073 → **361 行**；`grep "from .verbs"` 在 runtime 层 **零命中** | grep | ✅ 反向依赖已消除；`verbs/data.py` / `dryrun_runtime.py` 直接读文件调用数均为 **0** |
| **P2.1** 裸 `jplot cap` 吐 JSON | — | 实跑 | ✅ 现为 Rich `╭─ cap ─` 卡片 |

**残留（初评已标、本次未主张关闭）**：`column_probe.py` 仍有 2 处独立读取。
这是 validate 的"只读表头"路径，与 `data_access` 的定位不同，**保留是合理的**，不算 fork。

---

## 3. ⑧ 的措辞需要更精确

**声明**："轻 transform 共用" ✅
**实际**：属实，但含义比字面窄，建议在台账里写准，免得后人误以为"只剩一条派发链"。

- `Figure/preprocessor_runtime.py:89` 新增 `apply_light_transforms()`，docstring 写明
  *"Shared by dryrun/doctor (check phase) so light logic is not forked from the render runtime helpers."*
- `dryrun_runtime.py:586` 的 `_apply_simple_transforms` 已退化成 **薄包装**，直接调用上面那个函数。
  **检查期的那份 copy 确实没了。**
- 但渲染期的 `apply_transforms_impl`（同文件 `:479` 起）**仍有自己的 `if "filter" / elif "profile" / …` 链**。

即：**共用的是原语与检查期实现（原来的 fork 已消除），不是"渲染与检查合一条派发链"。**
按当前纪律（检查不跑 heavy、渲染跑全量）这两条链的职责本就不同，**不构成缺口**；
只是台账若写成"轻 transform 已统一"，一年后会被误读。建议措辞：
**"检查期轻 transform 已收归 `preprocessor_runtime.apply_light_transforms`，不再自带副本。"**

---

## 4. 核验中新发现的缺陷（1 项，打磨级）

### 渲染期每个用色层被观测两次，JP-VIZ 诊断重复发出

**现象**　2 层的 `posterior_2d`，`--report` 的 `layers` 有 **4 条**
（`_density, _hpd, _density, _hpd`），每条 `JP-VIZ-002` 因此重复出现，
`path` / `message` / `context` 完全一致。

**根因**（已定位到行）
```
figure.py:734   _prescan_colorbar_ranges → runtime_load_layer_runtime_data  → 观测 (1)
figure.py:743   _prescan 结尾 release_layer_runtime_data(consume_sources=False)
                                                        └─ data_loaded = False
figure.py:1006  渲染循环   → runtime_load_layer_runtime_data（守卫失效，重新加载）→ 观测 (2)
```
`layer_runtime.py:711` 的 `if layer_info.get("data_loaded"): return` 守卫本来能挡住第二次，
但 prescan 的 release 把它置回了 `False`。
不用色的层（`figure.py:728` `layer_uses_color` 为假时 `continue`）只被观测一次——
**所以重复量取决于图里有多少层用色**。

**复现**
```bash
jplot bad_type.yaml --report
python3 -c "import json;d=json.load(open('bad_type.render-report.json'));print([l['layer'] for l in d['figures'][0]['layers']])"
# → ['_density', '_hpd', '_density', '_hpd']
```

**影响**　agent 读 `--report` 会看到重复告警，可能把"两条 JP-VIZ-002"理解成两个不同问题；
`layers` 数组也不能直接当层清单用。**不影响判断正确性，属打磨项**，与既有可选后续（P2.3/P2.4）同级。

---

## 5. 测试锁覆盖情况

收口行为大多有测试钉住，且有几处锁得很到位：

| 行为 | 锁 | 备注 |
|---|---|---|
| `--deep` 不得存在 | `test_doctor_partial.py:69` `test_doctor_rejects_unknown_deep_flag`，docstring 写明 *"check phase never re-runs heavy transforms"* | **纪律本身被写成测试**，这是最好的形式 |
| envelope 不含 `deep` 字段 | 同文件 `:48` `:64` `assert "deep" not in env["data"]` | 防止字段悄悄回潮 |
| digest 退化 cell | 2 个测试文件含 `degenerate` | |
| `partial_renderable` | 1 个测试文件 | |
| `_digest_axes` 清除 | 17 个文件涉及（含间接） | |
| `pregrid` 契约 | `test_transform_contracts.py` | |
| 未知动词 | 1 个测试文件 | |

**未锁的一项**：本次新发现的观测重复（§4）——没有测试断言"每层只观测一次"。

---

## 6. 对收口声明本身的更正

1. **"轻 transform 共用"** —— 属实但易误读，见 §3 建议措辞
   （应写「检查期收归 `apply_light_transforms`」，勿写「渲染与检查已合一条链」）。
2. **核验报告自称「jplot/main..HEAD = 0 / 已同步」** —— **不成立**。
   复算为 **ahead 10**；remote 名是 `jplot` 没错，但 **push 债仍在**。

收口技术结论（纪律、删 deep/context、双侧 JP-VIZ、P0/P1 落点）仍成立；
只有「是否已发布到远端」这一条工程状态要改。

---

## 7. 核验命令与关键输出

```bash
git status --short                       # clean（核验当时）
git log jplot/main..HEAD --oneline | wc -l   # 复算: 10（ahead；非 0）
git branch -vv                           # main ... [jplot/main: ahead 10]
pytest tests/ -q                         # 432 passed, 1 skipped

# ③ 劝退物
ls jarvisplot/verbs/context.py           # 不存在
python -c "from jarvisplot.verbs import VERBS; print(sorted(VERBS))"
                                         # 10 个，无 context
grep -rn -- "--deep" jarvisplot/         # 无命中

# ④ 缺失功能报错
jplot context                            # exit=2；stdout 0 字节；stderr 一行指向 -h/man/cap

# ⑤ _digest_axes 生命周期
jplot config expand p.yaml --write       # grep -c _digest_axes → 1
jplot p.yaml                             # grep -c _digest_axes → 0，agent_output 本体保留

# ⑥ + P0.2 双侧
jplot doctor bad_type.yaml --json        # JP-VIZ-002 (basis: pre-transform)，coverage=partial
jplot bad_type.yaml --report             # post-mesh JP-VIZ-002；报告落在 YAML 旁
                                         # ⚠️ layers 4 条 / 2 层（§4）

# ⑦ decades
jplot data describe eggbox.csv --json    # decades_basis: min_max + decades_note 劝退误用

# P1.1 回归
jplot suggest --data eggbox.csv --kind posterior_2d --x x --y y --weight "exp(LogL)" --json
                                         # Uniform(0,5) → yscale: linear

# P1.3 依赖方向
grep -rn "from .verbs" jarvisplot/*.py jarvisplot/Figure/*.py   # 零命中
wc -l jarvisplot/verbs/data.py                                  # 361（原 1073）

# ② 纪律可机读
jplot man workflow --json                # anti_patterns 含
                                         #  "Expecting doctor/dryrun to re-run profile or density (no --deep)"
                                         #  "Treating doctor partial_renderable / coverage partial as failure"
```

---

## 8. 与"不在计划里"的三项对照

核验确认这三项**在代码里也确实不存在**，不是只在文档上宣告：

| 已否决项 | 代码状态 |
|---|---|
| 检查阶段 heavy / `--deep` | `--deep` 零残留；`apply_light_transforms` 显式 `heavy_skipped` 并标 `"skipped in check phase (heavy step)"` |
| 恢复 `context` | 文件已删，`VERBS` 无此键，`args.json` 无此条目，未知动词走统一报错 |
| 更多 `type:` 宏 | `cap types` 仍为 2 个（`posterior_2d` / `profile_2d`），未偷偷扩张 |

---

## 附：本次核验读过的文件

`jarvisplot/agent_digest.py`、`data_access.py`、`dryrun_runtime.py`、`render_health.py`、
`core.py`、`transform_contracts.py`、`verbs/__init__.py`、`verbs/data.py`、
`Figure/figure.py`（`_prescan_colorbar_ranges` / 渲染循环）、`Figure/layer_runtime.py`、
`Figure/preprocessor_runtime.py`、`cards/args.json`、`manual_cards/workflow.yaml`、
`tests/test_doctor_partial.py`、`tests/test_transform_contracts.py`；
git 历史 `1d2c52a..7853638`（12 个提交）。
