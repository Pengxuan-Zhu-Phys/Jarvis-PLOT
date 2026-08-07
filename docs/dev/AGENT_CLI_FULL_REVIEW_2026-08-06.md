# Jarvis-PLOT Agent CLI 全通量评审

Status: review
Date: 2026-08-06
Updated: 2026-08-07（产品纪律回写：检查阶段禁止 heavy 重跑；删 --deep 计划）
Reviewer: Claude（初评）+ 产品决策修订
Scope: `jplot` 全动词面 + `jarvisplot/schema/**` + `manual_cards/**` + agent_output 端到端 +
`docs/dev/*` 设计文档一致性。初评以 HEAD `1d2c52a` 为准；§0 与 §8 以当前产品纪律为准。

---

## 0. 产品纪律（检查 vs 执行）——优先于下文任何「修裁判」提案

> **Agent 写 YAML → 检查阶段只过结构/语法/列/轻步骤 → 通过后 `jplot <yaml>` 执行一次。**  
> **检查阶段（`validate` / `dryrun` / `doctor`）禁止再跑 profile / density / interp 等重步骤。**  
> **看图内容靠执行产物（PNG + `agent_output` digest），不靠 doctor 重算 mesh。**

因此下列方案**一律否决，不得再写进路线图或实现**：

| 否决 | 原因 |
|---|---|
| `jplot dryrun --deep` / `doctor --deep` / `doctor --full` | 检查阶段重跑 heavy = 第二条执行流水线 |
| 让 doctor 默认 full mesh / 覆盖 post-mesh 全套 JP-VIZ | 同上；且贵、与「写通就执行」冲突 |
| 为「type: 上 JP-VIZ 全覆盖」而在 dryrun 里接真 transform 派发 | 同上 |

**允许且已落地的轻量检查**（不重跑 mesh）：

- schema / 列 / 轻 transform 行数账本  
- `coverage: partial` + `partial_renderable`（含重步骤时**明示判不全**，不是失败）  
- pre-transform lim 代理（`JP-VIZ-002` + `basis: pre-transform`）——只用输入坐标，不跑 density/profile mesh  
- 执行期：`jplot <yaml>` 上的观测钩子 / digest（**唯一** heavy 入口）

下文 §5 保留初评现象与证据；**修复建议以本节为准**，冲突处一律作废。

---

## 0.1 执行摘要（初评 + 修订）

**初评总分：7.5 / 10。** 主线闭环已通；P0/P1 中与纪律兼容的项已在后续 commit 落地（见 §8）。

> 修订后的结论：**CLI = 知识库 + 轻检查 + 一次执行。**  
> doctor 在 `type:` 上 `partial` **是设计结果**（不假装看过 mesh），不是要靠 --deep 补全的洞。  
> agent 看图靠 **`jplot <yaml>` 写出的 lossy digest**，不靠 dryrun 重算。

### 三个最大优点（仍成立）

1. **产品原则可验证。** DR-08 bare-path render；doctor 只 plan `agent_output`、`jplot <yaml>` 才写。  
2. **假成功已消灭。** 坏列：`validate` → `doctor failed` → render exit=1。  
3. **`man` 契约化。** transform 死胡同已填；cap/man 可机器消费。

### 初评三大风险 → 修订状态

| 初评 | 修订 |
|---|---|
| P0 digest `top_density` 退化 cell | **已修**（`degenerate` + 过滤 ranking） |
| P0 type: 上 JP-VIZ「失明」 | **接受为纪律**：检查不做 heavy；补 pre-transform lim 代理；全图判断在执行/digest |
| P1 transform 契约漂移 | **已修**（pregrid + 一致性测试） |

### 是否建议发布

内部可用。对外：消化 §8 已修项即可；**不要**再把「doctor full mesh」当发布前置。

---

## 1. 系统理解（评审者的重构表述）

三层，职责分得比大多数同类工具清楚：

```
┌── 人 / coding agent ───────────────────────────────────────────────┐
│                     用编辑器写 YAML（唯一作业单）                    │
└───────────┬──────────────────────────────────────┬─────────────────┘
            │ 我该写什么？                          │ 我写对了吗？
            ▼                                      ▼
    ┌─── 知识库（纯读）───┐              ┌─── 裁判（读+判）───┐
    │ data  描述数据      │              │ validate  形状     │
    │ cap   合法字符串    │              │ dryrun    数据+行数 │
    │ man   契约与陷阱    │              │ doctor    两者合一  │
    │ explain 码/糖展开   │              └─────────┬──────────┘
    └────────┬───────────┘                        │
             │ 起草                                │
             ▼                                     │
    ┌─ template / suggest / config ─┐              │
    │ 写-验-回滚，ruamel 保注释      │              │
    └────────┬──────────────────────┘              │
             ▼                                     ▼
      ═══════════ jplot <yaml>（唯一执行面，DR-08）═══════════
             ├──→ PNG / PDF                （给人看）
             └──→ <name>.agent.json        （给 agent 看，YAML agent_output 声明）
```

**四条设计承诺，我理解为：**

| 承诺 | 机制 |
|---|---|
| CLI 不打字 | 所有写操作只经 `config set/rm/expand`，且 validate-before-write + 默认 dry-run |
| 唯一可复现入口 | 没有 `run` 动词；`RESERVED_NON_VERBS` 显式拦截并解释理由 |
| 双图模式可逆 | `type:` 是 `layers` 的宏；`config expand` 展开且 `agent_output._digest_axes` 存住展开前的轴语义 |
| agent 看图靠数值孪生 | 不是"强制真画图截屏"，而是 YAML 声明 `agent_output` → 真流水线跑完写 lossy digest |

**数据流的关键判断**：`doctor` 与 `jplot <yaml>` 走的**不是同一条数据路径**（见 §4、§5-P1.3）。
这是整套架构里唯一一处承诺与实现分叉的地方，也是两个 P0 的共同根因。

---

## 2. 维度评分表

| 维度 | 分 | 一句话 | 证据路径 |
|---|:--:|---|---|
| **A** 产品与架构一致性 | **8** | DR-08 与双图模式贯彻到位、可逆；`context` 是唯一边界模糊处 | `verbs/__init__.py:45` `RESERVED_NON_VERBS`；`jplot config expand --json` |
| **B** Coding-agent 工作流 | **7** | 闭环端到端跑通、假成功已消灭；但裁判在默认路径失明 | 本文 §3 全流程；`dryrun_runtime.py:477` |
| **C** 知识库质量 | **8** | `man` 契约化程度高，transform 死胡同已填；但已发生漂移且有不可发现键 | `jplot man transform.make_interp_2d --json`；`profile_runtime.py:626` |
| **D** agent_output 设计与实现 | **7** | schema 真接通、provenance 扎实；`top_density` 会误导 | `agent_digest.py:354`、`schema/core/common.json:95` |
| **E** 人机双界面 | **8** | Rich 卡片几何统一、覆盖 usage error；`jplot cap` 一处泄漏 | `jplot cap` 裸调用吐 JSON |
| **F** 可靠性 / 测试 / 可维护性 | **7** | 415 测试锁住关键契约；transform 轴无守卫、`verbs/data.py` 成 monolith | `tests/test_method_contracts.py` 有，transform 无 |
| **G** 安全与误用 | **8** | 写盘纪律好、`lossy: true` 醒目；digest density 可被误读 | `config expand` 默认 `wrote: false` |
| **H** 与 Jarvis 生态对齐 | **8** | Rich 几何与 HEP V2 一致；Operas 函数表是策划过的，不吹牛 | `jplot cap funcs --json` |

---

## 3. 工作流走查

对着产品原则规定的路径逐步实跑，数据为 20k 行 EggBox 型 CSV（`x,y,LogL,mass`）。

| # | 步骤 | 预期 | 实际 | 缺口 |
|---|---|---|---|---|
| 1 | `data describe --json` | 列 + 分位 + role_hint | ✅ `LogL → role_hint: log_likelihood`，`x/y/mass → parameter` | — |
| 2 | `cap types --json` | 可选 type 清单 | ✅ 2 个，各带 `explain` / `man` 指针 | 只有 2 个 type（`scatter_2d`/`posterior_1d` 不存在） |
| 3 | `suggest --kind posterior_2d` | 可跑 YAML + 决策理由 | ✅ `yaml_text` + `decisions[]` 每条带 `reason` | ⚠️ **`yscale: log` 判错**（§5 P1.1） |
| 4 | `validate` | 0 诊断 | ✅ `ok=true`, 0 diags | — |
| 5 | `config set agent_output` | schema 接受 | ✅ `ok=true`, 0 diags，ruamel 写入 | — |
| 6 | `doctor` | 状态可解释 | ✅ `ok=null` / `status=partial_renderable` / `coverage=partial` / `renderable=true` / `heavy_skipped` | ⚠️ **JP-VIZ 不可达**（§5 P0.2） |
| 7 | `doctor` exports | planned 不写盘 | ✅ `status: planned`，磁盘无文件 | — |
| 8 | `jplot plot.yaml` | PNG + digest | ✅ 两者都出，exit=0 | — |
| 9 | 读 `.agent.json` | 能定位主峰 | ⚠️ **`top_density` 头名是 count=1 的退化 cell**（§5 P0.1） | 致命 |
| 10 | `config expand --write` | 无缝降级 | ✅ `type` 移除、`layers: [_density,_hpd]` 生成、注释保留、`agent_output` 存活、重渲染仍出 digest | — |

**对照组（手写 layers，无重步骤）**：故意写坏 `xlim` 与 `vmax`，`doctor` 立刻给出
`JP-VIZ-002 ~90% of data extent is outside axes lim` 与
`JP-VIZ-004 colorbar vmax=0 is below data max=5.49`。
**说明 JP-VIZ 规则本身是好的，问题纯粹在覆盖面。**

---

## 4. 设计 vs 实现差距矩阵

| 设计承诺 | 出处 | 实现状态 | 证据 | 严重度 |
|---|---|---|---|:--:|
| agent 通道是 skin，绝不 fork 流水线 | `docs/specs/AGENT_DATA_API.md:24-26` | ❌ **已 fork** | 三条读数据实现：`data_loader.py`（lazy/polars/剪枝/cache）、`verbs/data.py:665`（eager `pd.read_csv`）、`column_probe.py`（只读头） | **P1** |
| 观测点取送进 matplotlib 的那份数组 | `V2_DEV_LEDGER.md` E1 | ❌ **未接线** | `Figure/` 对 `render_health` 零引用；观测全来自 `dryrun_runtime` 的第二条链 | **P0** |
| transform 契约与 runtime 同源 | `transform_contracts.py` docstring | ❌ **已漂移** | `profile`：契约有 `bins`/`seed`（runtime grep 不到），runtime 有 `pregrid`/`pregrid_bin`（契约/schema/docs/cards 全无） | **P1** |
| `agent_output` 是 YAML 执行面一等公民 | `docs/dev/AGENT_OUTPUT_YAML_DESIGN.md` | ✅ **真接通** | `config set` 无 JP-SCH-001；`doctor` planned；render 写盘；expand 保留 | — |
| digest 是 lossy、不可当 raw data | 同上 | ✅ 字段醒目 | `lossy: true` 顶层；`provenance.source_hash`；`algorithm.max_cells/actual_cells` | — |
| 人类默认 Rich 卡片 / agent 走 `--json` | 产品原则 6 | ⚠️ **一处泄漏** | `jplot cap`（裸）直接吐 `cap.all` JSON；`data`/`config`/`man`/`template` 均正常 | **P2** |
| `type` 与 `layers` 语义可逆 | 产品原则 4 | ✅ 成立 | `config expand --write` 后重渲染产出等价 PNG + digest | — |
| 无假成功 | 产品原则 2 | ✅ 成立 | 坏列 → validate false / doctor failed / render exit=1 | — |
| `doctor` 只 plan | 产品原则 5 | ✅ 成立 | exports `status: planned`，磁盘无文件 | — |

---

## 5. 问题清单（可执行）

### P0

#### P0.1 · digest `top_density` 被退化 cell 支配，agent 会上报错误众数

**现象**　`highlights.top_density[0].density = 2.17e+62`，对应 cell `count=1`、bbox 宽高均为 0。
次名 `3.87e+06`——相差 56 个数量级。top-5 的 `count` 依次为 `1, 2, 3, 2, 3`。

**根因**　[`agent_digest.py:354`](jarvisplot/agent_digest.py:354)
```python
area = max(bbox[1] - bbox[0], 1e-30) * max(bbox[3] - bbox[2], 1e-30)
```
退化 cell 面积 = `1e-60`，`density = mass/area` 随即爆炸。
[`agent_digest.py:419`](jarvisplot/agent_digest.py:419) 按 density 降序取前 10，只滤 `count>0`。

**复现**
```bash
jplot suggest --data eggbox.csv --kind posterior_2d --x x --y y --weight "exp(LogL)" --json \
  | python3 -c "import json,sys;open('p.yaml','w').write(json.load(sys.stdin)['data']['yaml_text'])"
jplot config set p.yaml "Figures[posterior_2d].agent_output" '{"method":"voronoi","max_cells":512,"path":"auto","seed":0}' --write --json
jplot p.yaml
python3 -c "import json;d=json.load(open('plots/posterior_2d.agent.json'));print(d['highlights']['top_density'][:3])"
```

**影响对象**　**agent（严重）**。`top_density` 是非多模态模型判断"峰在哪"的唯一凭据。
人类不受影响（看 PNG）。这不是失败，是**自信的错误**。

**修复建议**
1. 面积下限改为与数据尺度挂钩（如 `median(cell_area) * 1e-3`，或 `axes_extent / max_cells`），
   不用绝对 `1e-30`；
2. 复用已有 `flags` 机制（现有 `empty` / `tail`）新增 `degenerate`（bbox 任一边为 0）与
   `undersampled`（`count < k`，建议 k=5）；
3. `top_density` 排除带这两个 flag 的 cell，并在 `algorithm` 里记录 `excluded_cells` 计数。

**建议文件**　`jarvisplot/agent_digest.py`；新增 `tests/test_agent_digest.py::test_degenerate_cells_excluded_from_top_density`。

---

#### P0.2 · `type:` 路径上 doctor 不做 post-mesh JP-VIZ（设计，非缺陷）

**现象（初评）**　`type: posterior_2d` / `profile_2d` 展开后含重步骤 → doctor
`coverage: partial` / `partial_renderable`；跳过 mesh 时拿不到 post-transform 坐标上的全套 JP-VIZ。

**产品结论（2026-08-07）**　这是**正确行为**，不是要修掉的洞：

- 检查阶段**不得**重跑 profile / density / interp（已否决 `--deep` / doctor full mesh）。  
- `ok: null` + `partial_renderable` + `renderable: true` 的语义是：  
  **「结构/列过了；图内容要等 `jplot <yaml>`」**——agent **不准**据此改一份好 YAML。  
- 轻量兜底（已做）：重步骤跳过时用**输入** x/y 对 `frame.lim` 做 `JP-VIZ-002`，  
  `context.basis: pre-transform`（不声称 post-mesh）。  
- 图质量：只看 **执行** 产物（PNG / `.agent.json`），可选渲染期观测日志。

**已否决的「修复」**（勿复活）

1. ~~`dryrun --deep` / `doctor --full` 真跑重步骤~~  
2. ~~让 doctor 在 type: 上覆盖 full post-mesh JP-VIZ~~  
3. ~~检查阶段接 `apply_transforms_impl` 与 render 并跑 heavy~~  

**可选后续（不与纪律冲突）**

- 执行期 report / envelope 暴露渲染路径上的 JP-VIZ（**一次执行内**，不是 doctor 再跑一遍）。  
- man anti_patterns 写死：`partial` ≠ 图一定对 / 一定错。

---

### P1

#### P1.1 · `suggest` 的 log/linear 判据与取值来自两套统计量

**现象**　`y ~ Uniform(0,5)`（20k 样本）被判 `yscale: log`，理由
`"all positive; spans 4.54 decades"`。

**根因**　[`verbs/data.py:437`](jarvisplot/verbs/data.py:437)
```python
decades = float(np.log10(vmax / vmin)) if positive and vmin > 0 else 0.0
```
`decades` 用 **min/max**（样本里最 outlier 敏感的两个量），而同一函数的 `lim` 用
**q0.5%–q99.5%**。均匀分布的样本最小值 ≈ 1.4e-4 → 4.54 decades → 误判 log。

**连带损害**　误判后 `_nice_log_bound(q005)` 给出 `ylim[0]=0.02`，在 log 轴上把
q0.5% 以下全部裁掉——`suggest` 自己制造了一个 P0.2 抓不到的问题。

**影响对象**　agent（`suggest` 是 NL→YAML 的第一跳）与人。

**修复建议**　`decades` 换成分位数跨度不够（均匀分布仍得 2.3 decades）。
建议改用**分布形状判据**：`log` 当且仅当
`median / mean < 0.5`（对数正态 / 幂律成立，均匀 / 正态不成立）**且** 分位跨度 ≥ 2 decades。
对 `Uniform(0,5)`：`median/mean = 1.0` → linear ✅；对 `LogUniform(0.01,100)`：`≈0.046` → log ✅。

**建议文件**　`jarvisplot/verbs/data.py::suggest_axes`；
新增 `tests/test_suggest.py::test_uniform_positive_column_stays_linear`。

---

#### P1.2 · `transform_contracts` ↔ runtime 已漂移，且无守卫

**现象**
```
contract_for('profile') 声明: bin bins coordinates empty_value fill_empty grid_points method objective seed
profile_runtime.py 实际读:    bin coordinates empty_value fill_empty grid_points method objective
                              + pregrid, pregrid_bin, enable, name, scale, lim, fillna
```
`pregrid` / `pregrid_bin`（[`profile_runtime.py:626`](jarvisplot/Figure/profile_runtime.py:626)）
是用户可写的 profile 键，但 **`transform_contracts` / `schema/` / `docs/` / `manual_cards/` /
全部 24 份语料里一处都没有**——agent 无从发现。反向的 `bins` / `seed` 则是契约声明了但 runtime grep 不到。

**为什么这条特别刺眼**　methods 那条派发轴**做对了**：`METHOD_DISPATCH ↔ schema/methods/*.json`
有一致性测试（`tests/test_method_contracts.py`、`tests/test_capabilities.py`）。
同一个项目里两条派发轴，一条有 CI 守卫一条没有。而 `cap transforms` 与 `man transforms`
都以 `transform_contracts` 为真相源。

**影响对象**　agent（会得到自信但不完整的键表——正是本项目存在的理由所要消灭的）。

**修复建议**
1. 加一致性测试：对每个 transform，断言 `contract_for(name)` 的键集合 ⊇ runtime 模块中
   `.get("<key>")` 的正则抽取结果（差集为空，或列在显式 `_INTERNAL_KEYS` 白名单里）；
2. 补 `pregrid` / `pregrid_bin` 到契约与 `manual_cards/`；
3. 核实 `bins` / `seed` 是否为幽灵字段，是则删。

**建议文件**　`tests/test_transform_contracts.py`（新建）、`jarvisplot/transform_contracts.py`。

---

#### P1.3 · agent 通道 fork 了数据流水线，且依赖方向倒置

**现象**　三条独立的读数据实现：

| 路径 | 位置 | 特性 |
|---|---|---|
| 渲染 | `data_loader.py` | lazy、polars、列剪枝、`.cache/` 指纹 |
| agent | [`verbs/data.py:665`](jarvisplot/verbs/data.py:665) | **eager `pd.read_csv(path)` 整表进内存** |
| 校验 | `column_probe.py` | 只读表头 |

且 [`dryrun_runtime.py:204`](jarvisplot/dryrun_runtime.py:204)
`from .verbs.data import _detect_type, _load_dataframe`——**runtime 模块 import CLI 动词模块的私有函数**，
依赖方向倒置。

**违反的明文纪律**　`docs/specs/AGENT_DATA_API.md:24-26`：
> Both entries must converge on the same loader / transform / cache engine …
> The agent channel is a *skin*, never a fork of the pipeline.

**影响对象**　agent（`jplot data describe` 对大 HDF5 会整表读入，绕开专门为此建的
`MEMORY_OPTIMIZATION_GUIDE.md` 全套机器）+ 维护者（loader bug 要修两遍）。

**修复建议**　把 `describe_file` / `head_file` / `eval_on_file` / `_load_dataframe` / `_detect_type`
提到 `jarvisplot/data_access.py`（runtime 层），走 `data_loader` 的 lazy 路径；
`verbs/data.py` 只留 parser + 4 个 handler；`dryrun_runtime` 从新模块取。
**一次改动同时消解 P1.3、P2.2，并为 P0.2 的根治方案提供正确地基。**

---

### P2

| # | 现象 | 复现 | 影响 | 修复 |
|---|---|---|---|---|
| **P2.1** | 裸 `jplot cap` 在无 `--json` 时向 stdout 吐 35KB `cap.all` JSON；`data`/`config`/`man`/`template` 均给人类输出 | `jplot cap \| head -c 60` | 人（违反产品原则 6） | `verbs/cap.py`：无子命令时渲染 Rich 子命令卡片；`cap all` 保留为显式子命令 |
| **P2.2** | `verbs/data.py` 1073 行 = parser + 4 handler + 3 loader + HDF5 树 + cache 层 | `wc -l` | 维护者 | 见 P1.3，同一次重构解决 |
| **P2.3** | `man --json` 的 section 同时出现 `body: null` 与 `items: [...]`，消费方要查两个键 | `jplot man agent-output --json` | agent（轻微） | `man_render_agent.py:88`：无 body 时不写该键 |
| **P2.4** | 14 份 `manual_cards/*.yaml` 散文无真实性检查，而 `docs/` 有 `test_docs_status_labels.py` | — | agent（散文过期即成幻觉源） | 加测试：card 中出现的 `jplot …` 命令必须可解析；引用的 topic id 必须存在 |
| **P2.5** | `_digest_axes` 是下划线私有键，但 `config expand --write` 后会**留在用户 YAML 里** | `config expand --write` 后看 `agent_output` | 人（困惑） | schema 已有条目（`common.json:95`），补一行 `description` 说明它由 expand 自动写入、勿手改 |

---

## 6. 过度设计 / 欠设计（奥卡姆视角）

### 该删 / 该藏

- **`jplot context`（478 行）**。它返回 16 个顶层键，全部是 `data describe` + `cap` + `template`
  + `suggest` 的聚合。`jplot -h` 已经标了 "(advanced)"，`manual_cards/agent-output.yaml` 的
  `anti_patterns` 里甚至写着 *"Do not use jplot context as the primary path for data shape."*
  ——**产品自己在劝退它**。它是第二个聚合面，承担与 `man` 相同的漂移风险却没有 `man` 的契约纪律。
  **建议：从 `jplot -h` 移除，保留命令但标 unstable，或直接折进 `man workflow`。**
- **`decades` 字段**（`data describe` 输出）。既然判据要换（P1.1），这个用 min/max 算的量
  留在输出里只会诱导下游复用同一个错误启发式。**建议改为 `decades_q`（分位跨度）+ `shape_hint`。**

### 该合并

- **三条读数据路径 → 一条**（P1.3）。
- **两条 transform 派发链 → 一条**（P0.2 根治方案）。

### 该补

- **transform 一致性测试**（P1.2）——methods 有，transforms 没有，补齐即可。
- **`type:` 的第三个成员**。现在只有 `posterior_2d` / `profile_2d`。产品原则 4 说 `type` 是人类默认，
  但 `scatter_2d` 这种最基础的用法**没有 type**，人只能掉回手写 layers。
  这与"type 是默认、layers 是逃生口"的定位相反。**优先补 `scatter_2d` 与 `hist_1d`。**
- **digest 的 `flags` 词汇表**（`degenerate` / `undersampled`），见 P0.1。

---

## 7. 对 coding agent 的明确操作手册

基于本次实跑，可直接贴进 agent 的 system prompt。

### 只准（5 条）

1. **只准从 `jplot data describe <file> --json` 取列名。** 任何列名不得出自记忆或推断。
2. **只准从 `jplot cap <section> --json` 取 PLOT 词汇。** method / type / style token / axes 名
   （`ax` / `axc` 从 style card 来）/ cmap / transform / 表达式函数，全部如此。
3. **只准用 `jplot man <topic> --json` 查字段契约。** 写 transform 前先
   `jplot man transform.<name> --json`。
4. **只准用 `jplot config set/rm/expand` 改 YAML**（带 `--write` 才落盘，默认给 diff）；
   它保注释、先验后写，**不可能留下坏 YAML**。
5. **只准把 `jplot <yaml>` 当执行入口**，并用它产出的 `<name>.agent.json` 判断图的内容。

### 不准（5 条）

1. **不准把 `doctor` 的 `ok: null` / `status: partial_renderable` 当成失败。**
   它表示「含重步骤、检查阶段**故意**不重跑 mesh」，`renderable: true` 才是结论。
   **不要据此改一份正确的 YAML。** 图内容看 `jplot <yaml>` 产物。
2. **不准指望 `doctor` 写 `.agent.json`。** 它只 `status: planned`；只有 `jplot <yaml>` 写。
3. **不准把 digest 的 `cells` 当原始样本。** 顶层 `lossy: true` 是硬约束。
4. **不准用 `jplot context` 作为拿数据形状的主路径**（产品文档明文反对）。
5. **不准假设存在 `jplot run`。** 渲染是裸路径 `jplot <file>`（DR-08）。
6. **不准在检查阶段重跑 heavy**（无 `doctor --deep` / 等价物）。执行一次即可。

---

## 8. 路线图（修订：与纪律对齐）

### 已落地（勿再当 backlog）

| 项 | 状态 |
|---|---|
| P0.1 digest degenerate / top_density | done |
| P0.2 pre-transform lim 代理 | done |
| P1.1 suggest median/mean + 分位 decades | done |
| P1.2 transform 契约 + pregrid + CI | done |
| P1.3 `data_access` 读路径收敛 | done |
| P2.1 裸 `jplot cap` 索引卡 | done |
| 删除 `--deep` 及检查期 heavy 重跑 | done |

### 仍可做（不与纪律冲突）

| 序 | 事项 | 工作量 | 说明 |
|:--:|---|:--:|---|
| 1 | P2.3 man JSON 无 `body: null` 时省略键 | XS | 消费方更简单 |
| 2 | P2.5 `_digest_axes` schema 说明 | XS | expand 自动写、勿手改 |
| 3 | P2.4 manual_cards 真实性 CI | S | 命令/topic 可解析 |
| 4 | man anti_patterns：partial ≠ 图对/错 | XS | 锁 agent 纪律 |
| 5 | 执行期 JP-VIZ → report/envelope（可选） | M | **仅** `jplot <yaml>` 内，不是 doctor 再跑 |
| 6 | `context` 降级 unstable | S | 减聚合面 |
| 7 | 更多 `type:` 宏 | M | 产品默认路径 |

### 明确不做（已从路线图删除）

- ~~dryrun/doctor `--deep` / `--full`~~  
- ~~检查阶段 full post-mesh JP-VIZ~~  
- ~~为 type: 在 dryrun 接真 heavy 派发以「补全裁判」~~  
- ~~「两条 transform 链合并」若意味着检查也跑 heavy~~（维护性合并若只影响 render 内部，另议）

---

## 附录 A · 尖锐问题的直接回答

**1. 不读源码的 coding agent 能否只靠 CLI 完成三类图？**

- `posterior_2d` / `profile_2d`：**能。** `describe → suggest → validate → config set → doctor →
  jplot <yaml>` 全程实跑通过，无需读源码。
- 手写 `layers` + `transform`：**基本能，但会卡在两处**——
  (a) `profile` 的 `pregrid` / `pregrid_bin` 在任何 CLI 出口里都查不到（P1.2）；
  (b) 图对不对没有裁判可问，只能靠 `coverage: full` 的运气。

**2. `agent_output` 是一等公民还是文档超前？**

**一等公民，已接线。** 四个证据：schema 接受（`config set` 无 JP-SCH-001）、`doctor` 会 plan、
`jplot <yaml>` 真写盘、`config expand` 后经 `_digest_axes` 存活并仍能产出 digest。
这是本次评审里设计与实现吻合度最高的一块。

**3. `ok: null` + `partial_renderable` 是否足以防止误改正确 YAML？**

**防「误判失败」够了——这正是产品要的。** 它表示检查阶段**故意**不重跑 mesh，不是配置坏了。
「图对不对」不在 doctor 的职责里；在 `jplot <yaml>` + digest。  
~~用 --deep 补全 doctor~~ **已否决。**

**4. 是否应删除或隐藏 `context`？**

**隐藏，不删除。** 从 `jplot -h` 移除、标 unstable。产品 `anti_patterns` 已劝退；可择机并入 `man workflow`。

**5. `transform_contracts` / schema / runtime 三源会漂移吗？**

初评时已漂移；**P1.2 已加一致性测试与 pregrid 契约**。继续靠 CI 守。

**6. 与「强制真画图拿信息」相比？**

**更优：执行一次拿 digest。** 检查不重跑；不把风险押在 dryrun 假装看过 mesh。

**7. 发布前置还剩什么？**

与纪律冲突的 P0 已按兼容方式处理（digest + pre-transform lim；不 deep）。  
剩余多为 P2 体验项（man JSON 形状、card 真实性、文案纪律）。

---

## 附录 B · 本次运行过的命令与关键输出

```bash
# 知识库
jplot -h                                         # ✅ Rich 卡片，5 组：Discover/Draft&edit/Judge/Render/Flowchart
jplot man --json                                 # ✅ 14 topic，带 priority/role/summary
jplot man agent-output --json                    # ✅ sections + anti_patterns（body:null + items 并存，P2.3）
jplot man transform.make_interp_2d --json        # ✅ 完整字段契约，无 delegated 死胡同
jplot cap types --json                           # ✅ 2 个 type，各带 explain/man 指针
jplot cap funcs --json                           # ✅ 策划过的函数表（exp/log/Gauss/pdg.*/cmb.*），无噪音
jplot cap                                        # ❌ 裸调用吐 35KB cap.all JSON（P2.1）

# 起草 → 校验 → 执行（20k 行 EggBox 型 CSV）
jplot data describe eggbox.csv --json            # ✅ role_hint: LogL→log_likelihood
jplot suggest --data eggbox.csv --kind posterior_2d --x x --y y --weight "exp(LogL)" --json
                                                 # ⚠️ yscale:log 误判（P1.1）
jplot validate plot.yaml --json                  # ✅ ok=true, 0 diags
jplot config paths plot.yaml --json              # ✅ 具名地址 Figures[posterior_2d] 等
jplot config set plot.yaml "Figures[posterior_2d].agent_output" '{...}' --write --json
                                                 # ✅ ok=true，schema 接通
jplot doctor plot.yaml --json                    # ⚠️ partial_renderable / coverage=partial / heavy_skipped
jplot doctor plot.yaml --json  (exports)         # ✅ status=planned，磁盘无文件
jplot plot.yaml                                  # ✅ exit=0，产出 PNG + .agent.json
jplot config expand plot_exp.yaml --write --json # ✅ ruamel 保注释，agent_output + _digest_axes 存活

# 对照：手写 layers（无重步骤）+ 故意坏 xlim/vmax
jplot doctor handwritten.yaml --json             # ✅ JP-VIZ-002 (90% outside lim) + JP-VIZ-004 (vmax 饱和)

# 假成功检查（坏列 exp(LogLL)）
jplot validate bad.yaml --json                   # ✅ ok=false, JP-COL-001
jplot doctor  bad.yaml --json                    # ✅ status=failed
jplot bad.yaml                                   # ✅ exit=1

# 测试
pytest tests/ -q                                                    # 415 passed, 1 skipped
pytest tests/test_man_cli.py tests/test_agent_digest.py \
       tests/test_doctor_partial.py tests/test_cli_help.py \
       tests/test_config_expand.py tests/test_suggest.py \
       tests/test_capabilities.py -q                                # 64 passed
```

## 附录 C · 阅读过的关键文件

**设计 / 台账**　`docs/dev/AGENT_OUTPUT_YAML_DESIGN.md`(392)、`docs/dev/MAN_CLI.md`(444)、
`docs/dev/YAML_HUMAN_AI_OCCAM_REVIEW.md`(278)、`docs/roadmap/V2_DEV_LEDGER.md`(780)、
`docs/roadmap/V2_AGENT_CLI_SURFACE.md`(485)、`docs/roadmap/V2_YAML_AGENT_ERGONOMICS.md`(339)、
`docs/specs/AGENT_DATA_API.md`

**CLI / 路由**　`verbs/__init__.py`(162)、`cli.py`(311)、`cli_help.py`(309)、`client.py`、`cards/args.json`

**Agent I/O**　`agent_io.py`(139)、`diagnostics.py`(285)、`diagnostic_guidance.py`(371)

**知识库**　`capabilities.py`(360)、`transform_contracts.py`(497)、`man_methods.py`(350)、
`man_render_agent.py`、`man_render_human.py`、`manual_cards/*.yaml`(14 份)

**起草 / 编辑**　`verbs/suggest.py`(530)、`verbs/data.py`(1073)、`verbs/config_cmd.py`(540)、
`verbs/context.py`(478)、`templates_catalog.py`(315)

**裁判**　`validation.py`(532)、`dryrun_runtime.py`(632)、`render_health.py`(465)、
`schema/`(34 份 JSON)、`column_demand.py`(518)、`column_probe.py`

**Digest / 渲染**　`agent_digest.py`(731)、`Figure/figure_types.py`、`Figure/profile_runtime.py`、
`Figure/preprocessor_runtime.py`、`core.py`

**测试**　44 个文件 / 416 用例；重点读了 `test_method_contracts.py`、`test_capabilities.py`、
`test_doctor_partial.py`、`test_agent_digest.py`、`test_config_expand.py`
