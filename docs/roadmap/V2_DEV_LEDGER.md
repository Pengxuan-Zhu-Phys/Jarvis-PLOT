# Jarvis-PLOT V2 开发台账

Status: active ledger
Created: 2026-08-06
Owner doc set: `V2_YAML_AGENT_ERGONOMICS.md`（YAML 结构侧）+ `V2_AGENT_CLI_SURFACE.md`（Agent CLI 侧）

---

## 0. 这份台账怎么用

两份 brainstorm 回答的是"该做什么"和"为什么"。**这份台账只回答"下一行代码写在哪个文件里、
做完怎么算数"**。任何施工都从这里取任务，不从 brainstorm 里取。

- 一个任务 = 一次可独立提交、可独立回滚、可独立验收的改动。
- 任务不写工时，只写**规模档**（S/M/L/XL）和**依赖**。规模档的含义见 §1。
- 验收标准必须是**可执行的**（一条命令、一个测试文件名），不能是"更好用"。
- 未拍板的问题不进任务，进 §6 决策记录（DR）。**依赖某个 DR 的任务标 `⊘`，不许提前开工。**
- 每次施工在 §7 施工日志追加一行，不改历史行。

**与既有 backlog 的关系**：`IMPLEMENTATION_ROADMAP.md` §3.1 的 `JP-A1…A5` 仍是 frozen 规格。
本台账的 A/C/F 轨是它的**超集与细化**；解冻决定见 `DR-01`。在 DR-01 拍板前，A 轨可以按
"内部实现，不对外承诺 API"推进，但不得写进 `docs/specs/AGENT_DATA_API.md`。

---

## 1. 图例

| 状态 | 含义 |
|---|---|
| `☐` | 未开工 |
| `◐` | 进行中（施工日志里必须有对应行） |
| `☑` | 完成且验收通过 |
| `⊘` | 阻塞（依赖未完成任务或未拍板 DR，备注里必须写明阻塞源） |
| `—` | 决定不做（理由进 §8） |

| 规模 | 含义 |
|---|---|
| S | 单文件、无新概念、≤150 行 |
| M | 2–4 文件、引入一个新模块或新数据文件 |
| L | 跨层（CLI + runtime + schema），需要新测试套 |
| XL | 破坏兼容，需要迁移工具和文档重排 |

**破坏兼容**列：`否` = 现存 YAML/CLI 调用零改动；`是` = 需要 `jplot migrate`（见 G7）。

---

## 2. 里程碑

| 里程碑 | 主题 | 出口条件（可执行） | 任务 |
|---|---|---|---|
| **M0** | 地基 | `jplot validate x.yaml --json` 输出合法 envelope，退出码 0/1/2 正确，且**全过程不 import matplotlib** | A1–A4, B7 |
| **M1** | 反幻觉 | agent 能通过两条命令拿到两个白名单：`jplot data describe --json` 给列名，`jplot cap all --json` 给 PLOT 全部合法字符串；且写错的键会报 did-you-mean 而不是静默通过 | B1–B4, B6, C1–C5, D1–D5, H4 |
| **M2** | 可自纠 | 一轮 `jplot doctor` 报全部问题；机械错误可 `--fix` 自动修；空图/裁掉/饱和这类"画错了"能被 `JP-VIZ-*` 检出 | B5, B8, C6, E1–E4, H2 |
| **M3** | NL→YAML | `jplot suggest` 能从数据直接产出可跑 YAML；`jplot config set` 能保留注释地改 YAML 且**永不落盘坏文件** | F1–F6, E5 |
| **M4** | V2 break | R2–R7 全部落地，`jplot migrate` 能把 1.x YAML 无损转过来 | G1–G7, D6, H1, H3 |

M0+M1 **不破坏任何兼容**，可以在 1.5.x 发布。M4 才是 2.0。

---

## 3. 依赖图

```
  A1 诊断模型+envelope
   ├─→ A2 动词化 CLI ──→ A3 退出码/流分离 ──→ A4 --json 约定冻结
   │        │
   │        ├─────────────────────→ C1 describe ─→ C2 role_hint ─→ C3 缓存
   │        │                          │             └─→ C6 suggest-axes ─┐
   │        │                          └─→ C4 head                        │
   │        ├─→ C5 eval                                                   │
   │        ├─→ D1 cap methods/transforms/types ─┐                        │
   │        ├─→ D2 cap styles ────────────────────┼─→ D5 cap all ─→ D6 cap mcp
   │        ├─→ D3 cap cmaps/funcs ───────────────┤                       │
   │        └─→ D4 cap cli ────────────────────────┘                      │
   │                                                                      │
   └─→ B1 schema 骨架+manifest ─→ B2 zone lint ─→ B3 R1 闭合词汇表        │
                                    │                 │                   │
                                    │                 └─→ B4 JP-* 码+guidance
                                    │                        └─→ B8 --fix │
                                    └─→ B5 method 坐标合约                │
                                                                          │
   B7 validate 零 matplotlib（守卫测试，随 A2 一起立）                     │
   B6 列存在性校验（复用 core_runtime.plan_dataset_required_columns）      │
                                                                          │
   E1 渲染观测点 ─→ E2 JP-VIZ 规则 ─→ E3 行数账本                          │
              └──→ E4 数值孪生 ──→ E5 ASCII 缩略图                        │
                                                                          │
   F1 template ←── D5                                                     │
   F2 suggest  ←──────────────────────────────────────────────────────────┘
   F3 地址语法+layer name 必填 ─→ F4 config get/set ─→ F5 add/rm/move ─→ F6 fmt/diff
                                                                          
   G1 R2 transform ─┐
   G2 R3 columns   ─┼─→ G7 migrate ─→ H3 文档重排
   G3 R4 单一写法  ─┤
   G4 R5 颜色所有权─┤
   G5 R6 normalizer─┘
   G6 R7 type 主接口 ←── F1
```

**关键路径**：`A1 → A2 → B1 → B3 → B4`。这条链走通，实测里最严重的"静默接受"就消失了。

---

## 4. 任务明细

### 轨 A — 地基

| ID | 标题 | 里程碑 | 依赖 | 规模 | 破坏 | 状态 |
|---|---|---|---|---|---|---|
| A1 | 诊断模型 + JSON envelope | M0 | — | M | 否 | ☑ |
| A2 | 动词化 CLI 骨架（渲染=裸路径，无 `run` 动词；见 DR-08） | M0 | A1 | L | 否 | ☑ |
| A3 | 退出码与 stdout/stderr 分离 | M0 | A2 | S | 否 | ☑ |
| A4 | `--json` 全局约定 + `api_version` 冻结测试 | M0 | A3 | S | 否 | ☑ |

**A1 — 诊断模型 + JSON envelope**
- 产物：新 `jarvisplot/diagnostics.py`（`Diagnostic` dataclass：`code/level/path/message/suggestion/example/fix`；
  `DiagnosticBag` 收集器）+ 新 `jarvisplot/agent_io.py`（`envelope(kind, ok, data, diagnostics, error) -> dict`）。
- 验收：`python -c "from jarvisplot.agent_io import envelope; print(envelope('validate', True, {}, [], None))"`
  输出含 `api_version/kind/ok/data/diagnostics/error` 六个键，且 `json.dumps` 无异常。
- 测试：`tests/test_agent_envelope.py` — envelope 键集合冻结、diagnostics 序列化、`level` 只允许 `error|warning|info`。
- 备注：字段名沿用 `docs/specs/AGENT_DATA_API.md` §2，**新增 `diagnostics` 顶层数组**（原规格只有 `error`
  单数）。这是对 frozen 规格的第一处偏离，记入 DR-01。

**A2 — 动词化 CLI 骨架**
- 产物：`jarvisplot/cli.py` 从"扁平 argparse + `cards/args.json`"改为"subparser 树 + `cards/args.json` 扩展成
  `{commands: [...]}`"。**`cards/args.json` 必须仍是唯一真源**（D4 依赖这一点）。
  `jarvisplot/core.py:56` 现有的 `_is_flowchart_command()` 特例被 subparser 吸收。
- 验收：`jplot old_config.yaml` 行为逐字节不变（**裸路径 = 渲染，没有 `run` 动词**，见 DR-08）；
  `jplot validate x.yaml`、`jplot data describe f.h5`、`jplot cap all` 三条能被解析（可先 stub）。
  `jplot run x.yaml` / `Jarvis2 plot run x.yaml` **不得**被当成合法渲染入口。
- 测试：扩展 `tests/test_cli_help.py`；新增 `tests/test_cli_verbs.py` — 裸路径路由、未知动词的
  did-you-mean、`jplot flowchart` 仍走原路径、**拒绝 `run` 伪动词**。
- 备注：**这是整条关键路径上最容易做砸的一步**。动词只承载非渲染意图；渲染永远是「给文件」。
  `jplot <file>` 与 `jplot <verb>` 的歧义靠「首 token 是否在 VERBS」消解（动词表优先），
  文件名若撞动词则需用路径前缀（如 `./validate`）——规则写进 docstring 并测。

**A3 — 退出码与 stdout/stderr 分离**
- 产物：`core.py:232` 的 loguru sink 从 `sys.stdout` 改 `sys.stderr`；`--json` 模式下 stdout 只允许一个
  JSON 对象。退出码：`0` 成功 / `1` 配置或数据错误 / `2` 用法错误（沿用 AGENT_DATA_API）。
- 验收：`jplot validate broken.yaml --json 1>out.json 2>err.log; python -m json.tool out.json` 成功解析。
- 测试：`tests/test_cli_verbs.py::test_json_stdout_is_pure`。

**A4 — `--json` 全局约定 + `api_version` 冻结测试**
- 产物：`--json` 提升为全局 flag；`api_version` 常量单点定义。
- 验收：所有已实现动词加 `--json` 都返回同构 envelope。
- 测试：参数化测试遍历动词表，断言 envelope 形状一致。

---

### 轨 B — Schema 与诊断

| ID | 标题 | 里程碑 | 依赖 | 规模 | 破坏 | 状态 |
|---|---|---|---|---|---|---|
| B1 | schema 目录骨架 + `manifest.json` 派发索引 | M1 | A1 | M | 否 | ☑ |
| B2 | `x-jarvis-zone` 标注 + catalog 自检 lint | M1 | B1 | M | 否 | ☑ |
| B3 | **R1 闭合根/figure/layer/transform 词汇表 + did-you-mean** | M1 | B2 | L | 否 | ☑ |
| B4 | `JP-*` 错误码表 + suggestion/example 知识库 | M1 | B3 | M | 否 | ☐ |
| B5 | method 坐标合约进 schema | M2 | B4, D1 | L | 否 | ☐ |
| B6 | 列存在性校验 | M1 | A1 | M | 否 | ☑ |
| B7 | `validate` 零 matplotlib 守卫 | M0 | A2 | S | 否 | ☑ |
| B8 | `--fix` 结构化修复 | M2 | B4 | M | 否 | ☐ |

**B1 — schema 目录骨架 + manifest**
- 产物：`jarvisplot/schema/manifest.json`（**data-only**，只列文件名与派发键，绝不含逻辑）+
  `schema/core/{root,dataset,figure,layer,frame}.json` + `jarvisplot/schema_catalog.py`
  （通用加载器，本地 `referencing.Registry`，**永不联网取 schema**）。
- 验收：`python -c "from jarvisplot.schema_catalog import load_catalog; load_catalog()"` 无异常，
  且每个文件都过 `Draft202012Validator.check_schema()`。
- 测试：`tests/test_schema_catalog.py` — manifest 里每个文件存在、每个文件是合法 Draft 2020-12、
  没有 `$ref` 指向 http(s)。
- 备注：对照实现 `Jarvis-HEP-v2/jarvishep2/task_schema.py`。**扩展纪律**：加一个 method/transform =
  加一个 schema 文件 + manifest 里加一行 + 加 fixture，**`schema_catalog.py` 永不改动**。
  PLOT 有 5 条派发轴（`Figures[].type` / `layers[].method` / transform 类型 / `DataSet[].type` / style token），
  是 HEP2 的 2.5 倍，所以这条纪律比在 HEP2 更值钱。

**B2 — `x-jarvis-zone` 标注 + 自检 lint**
- 产物：每个 object schema 必须带 `x-jarvis-zone: closed | delegated`。
  **不引入 HEP2 的 `open` zone**（理由见 YAML doc §6：V2 允许 break，不给自己开后门）。
- 验收：`jplot validate --self-check` 报 0 错误；故意删掉一个 zone 标记后 CI 失败。
- 测试：`tests/test_schema_catalog.py::test_every_object_declares_zone`。
- 备注：`layers[].style` 必须是 `delegated`（下游是 matplotlib kwargs，PLOT 不该穷举）；
  但 `cmap/vmin/vmax` 在绑定 colorbar 时与 `frame.axc` 冲突 → 由 B4 的 `JP-OWN-002` 单独管。

**B3 — R1 闭合根/figure/layer 词汇表 + did-you-mean** ⭐
- 产物：根词汇表闭合为 `version/project/DataSet/Figures/Functions/output`；figure 层闭合为
  `name/enable/type/style/frame/layers/debug/…`；layer 层闭合为
  `name/data/share_data/axes/method/coordinates/style/colorbar`。未知键 → 错误 +
  `difflib.get_close_matches` 建议。
- **验收（已按 §9.1 扩写）**：不是消灭 case A/B 两例，而是消灭**"静默接受"这一整类**：
  1. case A（`Layers:` 大写 + `outputs:` 拼错，今天完全静默、退出码 0、写空 PNG）
  2. case B（`styles:` 拼错 → 报无关的 `'axes'` KeyError）
  3. **transform 步骤名写错**（`- fitler:`，今天整步静默跳过，图照画）← 新增，见 §9.1
  4. **figure 层 `legend:`**（今天从不读，本仓库 13 个 figure 中招）← 新增，走 JP-OWN-001 警告
- 因此 transform 词汇表必须**在这一步**进 schema（`core/transform.json`），不能等 G1/R2。
- 测试：`tests/test_schema_catalog.py` — 四类回归 fixture 各一条 + 全语料零假阳性断言。
- 备注：**这一条单独就消灭了实测里最严重的一整类失败，且对合法 YAML 零影响。**优先级最高。

**B4 — `JP-*` 错误码表 + guidance**
- 产物：`jarvisplot/diagnostic_guidance.py`（对照 `Jarvis-HEP-v2/jarvishep2/diagnostic_guidance.py` 的
  `_GUIDANCE_BY_PREFIX` + `guidance_for(code, path, message) -> (suggestion, example)`）。
  码段划分：`JP-SCH-*` schema / `JP-MTH-*` method / `JP-TRF-*` transform / `JP-COL-*` 列 /
  `JP-OWN-*` 所有权冲突 / `JP-EXP-*` 表达式 / `JP-VIZ-*` 渲染体检（E2 用）。
- 验收：每条诊断都带非空 `suggestion`；`--json` 与人类输出的 `suggestion`/`example` 逐字一致。
- 测试：`tests/test_diagnostic_guidance.py` — 遍历码表断言无空 suggestion；人机一致性断言。
- 备注：HEP2 的 `guidance_for` 有一条重要教训写在注释里（D21.14：参数级码必须**赢过**列表级前缀匹配），
  抄的时候把这个顺序一起抄过来。

**B5 — method 坐标合约进 schema**
- 产物：`schema/methods/<name>.json`，每个描述 `coordinates.required`（如 `pcolormesh` 需 `x/y/z` 且
  `z` 为 2D）、`axes_types`、`x-jarvis-example`。数据源是 `Figure/method_registry.py:METHOD_DISPATCH`
  （~30 个键，已是权威名单）。
- 验收：`jplot validate` 对缺 `z` 的 `pcolormesh` 报
  `JP-MTH-*: method=pcolormesh requires coordinates.z`，路径精确到 `$.Figures[0].layers[1].coordinates`。
- 测试：`tests/test_method_contracts.py` + **CI 一致性检查**：`METHOD_DISPATCH` 的键集合 ==
  `schema/methods/` 的文件名集合（见 DR-05：手写 schema + CI 校验，不生成）。
- 备注：`MethodSpec` 现在只有 `key/mpl_method/axes_types`，**没有坐标合约**——这正是文档查不到、
  agent 只能试错的那部分知识。

**B6 — 列存在性校验**
- 产物（**已按 §9.4 改写，原"复用 15 行"的估计是错的**）：
  1. 把 `core_runtime.py` 里的纯表达式分析器（312 行）抽成无依赖的 `column_demand.py`，两边共用；
  2. 新写 `plan_source_demand()`——**不能复用** `plan_dataset_required_columns()`，
     它把列需求向所有 dataset 做并集（对剪枝安全、对存在性判断是假阳性源）；
  3. `column_probe.py` 只读 header/元数据（csv / parquet / hdf5），一行数据都不读；
  4. 报 `JP-COL-001` + `did_you_mean` + `available_columns`，**路径指向出错的表达式本身**，
     不是 DataSet 条目——这是这条诊断有没有用的分水岭。
- 验收：实测 **case D**（`coordinates.x.expr: aa`，今天报 `name 'aa' is not defined`，无 YAML 路径、
  无可用列表）变成带路径 + 可用列清单的诊断。
- 测试：`tests/test_column_existence.py`。
- 备注：仍然是性价比很高的一条，但**不是"机器已经在那了"**。必踩的坑：列名里带点
  （真实语料有 `pVa.E`）会被标识符正则拆成 `pVa` + `E`，检查必须接受真实列名的标识符片段，
  否则一批假阳性。**精度优先于覆盖：一条假的"列不存在"会让 agent 学会忽略这条诊断。**

**B7 — `validate` 零 matplotlib 守卫**
- 产物：确保 validate 路径不 import matplotlib（今天 `core.init()` 无条件走 `load_cmaps` →
  `plot()`，全部错误只在渲染期暴露）。
- **验收是两条不变量，不是一条**（见 §9.5——列检查要读文件头，和"零 I/O"不可兼得）：
  1. `matplotlib` / `scipy` / `shapely` **永不加载**，无论什么 flag；
  2. `--no-columns` 时连 `pandas` / `h5py` / `polars` / `pyarrow` 都不加载（纯形状判决）。
- 测试：`tests/test_validate_no_matplotlib.py` — 子进程跑 validate，断言 `matplotlib` 不在
  `sys.modules`。
- 备注：这是"一轮报全"的物理前提。一个 10 图 YAML 今天要 10 轮才收敛。

**B8 — `--fix` 结构化修复**
- 产物：`Diagnostic.fix = {op, path, from, to}` + `confidence: certain | heuristic`；
  `jplot validate --fix --diff`（默认）/ `--fix --write`。只修 100% 机械的三类：拼写、别名归一、
  已知重命名。
- 验收：case A 的 `Layers:`→`layers:`、`outputs:`→`output:` 能被 `--fix --write` 一次修好。
- 测试：`tests/test_fix_application.py`。
- 备注：ruff 的模式。`heuristic` 的修复默认不写，需 `--fix-unsafe`。

---

### 轨 C — 数据自省（闭环①：看得见数据）

| ID | 标题 | 里程碑 | 依赖 | 规模 | 破坏 | 状态 |
|---|---|---|---|---|---|---|
| C1 | `jplot data describe --json` | M1 | A2 | M | 否 | ☑ |
| C2 | `role_hint` 列角色推断 | M1 | C1 | S | 否 | ☑ |
| C3 | describe 结果缓存 | M1 | C1 | S | 否 | ☐ |
| C4 | `jplot data head` | M1 | C1 | S | 否 | ☐ |
| C5 | `jplot data eval` 表达式沙盒 | M1 | A2 | M | 否 | ☐ |
| C6 | `jplot data suggest-axes` | M2 | C1 | M | 否 | ☐ |

**C1 — `data describe --json`**
- 产物：包 `jarvisplot/data_loader_summary.py:dataframe_summary_rows()`（已产出
  `name/dtype/nonnull%/min|uniq/max`）为 JSON；HDF5 走已有 `print_hdf5_tree_ascii()` 的结构化版本。
- 验收：`jplot data describe samples.hdf5 --json` 返回 `{rows, columns:[{name,dtype,...}], tree}`。
- 测试：`tests/test_data_describe.py`。
- 备注：**这是"列名唯一合法来源"纪律的物理出口**（`Jarvis-Agent/Docs/PLOT_TOOLS.md` §3.1 已确立纪律，
  但 PLOT 侧一直没有出口）。现成机器已覆盖一半。

**C2 — `role_hint` 列角色推断**
- 产物：按列名 + 数值特征推断 `log_likelihood | weight | chi2 | parameter | flag`。
- 验收：含 `LogL` 列的文件，describe 输出该列 `role_hint: "log_likelihood"`。
- 测试：`tests/test_data_describe.py::test_role_hint`。
- 备注：**这是 PLOT 有信息优势而 agent 没有的地方**——知道某列是 log-likelihood，agent 才知道权重
  该写 `exp(LogL)` 而不是 `LogL`。

**C3 — describe 缓存**
- 产物：用已有 `cache_store.ProjectCache.source_fingerprint(path, extra)` +
  `get_summary`/`put_summary` 缓存 describe 结果。
- 验收：同一文件第二次 describe 不重读数据（用计时或 mock 断言）。
- 测试：`tests/test_data_describe.py::test_cache_hit`。
- 备注：agent 一个会话里会调几十次 describe。

**C4 — `jplot data head`**
- 产物：前 N 行真实样本（默认 5），JSON。
- 验收：`jplot data head f.h5 -n 5 --json` 返回 5 行。
- 测试：同 C1 测试文件。
- 备注：**5 行真实样本对 LLM 的推理价值高于一页统计量**——量级、符号、是否有 NaN 一眼可见。

**C5 — `jplot data eval` 表达式沙盒**
- 产物：复用 `jarvisplot/utils/expression.py`。成功返回 `dtype/finite_count/range/samples`；
  失败返回 `available_columns` + `did_you_mean` + `available_functions`。
- 验收：`jplot data eval "exp(LogL)" --data f.h5 --json` 成功；`eval "exp(LogLL)"` 返回带
  did-you-mean 的 `JP-EXP-*`。
- 测试：`tests/test_data_eval.py`。
- 备注：让 agent 在**写进 YAML 之前**验证表达式，消灭一整类错误。

**C6 — `jplot data suggest-axes`**
- 产物：每列给 `scale`（全正且跨 ≥2 数量级 → log）、`lim`（q0.5–q99.5 向外取整）、`reason`。
- 验收：正值跨 3 个量级的列返回 `scale: log` 且 `reason` 非空。
- 测试：`tests/test_suggest_axes.py`。
- 备注：F2 `suggest` 的数值来源。

---

### 轨 D — 能力自省（闭环②：看得见能力）

| ID | 标题 | 里程碑 | 依赖 | 规模 | 破坏 | 状态 |
|---|---|---|---|---|---|---|
| D1 | `cap methods \| transforms \| types` | M1 | A2 | M | 否 | ☑ |
| D2 | `cap styles`（含 `axes` 字段） | M1 | A2 | M | 否 | ☑ |
| D3 | `cap cmaps \| funcs` | M1 | A2 | S | 否 | ☑ |
| D4 | `cap cli`（从 `cards/args.json` 生成） | M1 | A2 | S | 否 | ☑ |
| D5 | `cap all` + digest hash | M1 | D1–D4 | S | 否 | ☑ |
| D6 | `cap mcp` 生成 MCP tool schema | M4 | D5 | M | 否 | ⊘ DR-03 |

> **D1–D5 当前状态（☑）**：采集器 + `jplot cap [section] --json` 动词 + `tests/test_capabilities.py`
> 已落地。styles 带 `axes`/`usable`（两张 1x1 Ternary 报 `usable=false`）；transforms 含
> `to_parquet`；digest 稳定。`_funcs()` 仍会 import numpy（cap 唯一不免费的段）。

**D1 — `cap methods | transforms | types`**
- 产物：从 `Figure/method_registry.py:METHOD_DISPATCH`（权威名单）+ transform 注册表 +
  `Figure/figure_types.py` 导出 JSON。method 条目带 `coordinates.required`（与 B5 同源）。
- 验收：`jplot cap methods --json | jq '.data.methods | length'` == `len(METHOD_DISPATCH)`。
- 测试：`tests/test_capabilities.py::test_methods_match_registry`。
- 备注：`method_registry.resolve()`（`method_registry.py:138`）今天抛
  `KeyError("Unknown method key: 'scattr'")`——**正确但无 did-you-mean、无可用清单**。
  cap 落地后，该错误路径应改为引用 `jplot cap methods`。

**D2 — `cap styles`（含 `axes` 字段）** ⭐
- 产物：遍历 `jarvisplot/cards/` 导出每张 style card 的 `name/size/axes/…`。
  **`axes: ["ax", "axc"]` 是这条里最关键的字段。**
- 验收：`jplot cap styles --json` 中 `a4paper/2x1/rect_cmap` 的 `axes` 含 `axc`。
- 测试：`tests/test_capabilities.py::test_style_cards_expose_axes`。
- 备注：**`axc` 从哪来是当前最隐蔽的陷阱**——agent 在 layer 里写 `axes: axc`，但没有任何出口告诉它
  这个名字来自 style card。同时注意 `rectcmap`（YAML 里）vs `rect_cmap`（文件名）的别名，
  cap 必须两个都报（归一化见 G5）。

**D3 — `cap cmaps | funcs`**
- 产物：全部注册 colormap 名（含 `_r` 变体）+ 表达式可用函数签名。
- 验收：`jplot cap cmaps --json` 含 `jarvis_rainbow2` 及其 `_r`。
- 测试：`tests/test_capabilities.py`。

**D4 — `cap cli`**
- 产物：直接 dump `cards/args.json`（A2 扩展后的 commands 树）。
- 验收：输出与 `cards/args.json` 语义等价。
- 测试：`tests/test_capabilities.py::test_cap_cli_matches_spec`。
- 备注：**几乎零成本**——`cards/args.json` 本来就是数据文件，CLI 天然自描述。

**D5 — `cap all` + digest hash**
- 产物：一次性合并输出 + 内容 hash（agent 可缓存并判断是否变更）。
- 验收：两次调用 hash 相同；改一张 card 后 hash 变化。
- 测试：`tests/test_capabilities.py::test_digest_stability`。
- 备注：**当前最大的臆造来源就是这里**（`jarvis_rainbow2_r`、`a4paper_2x1`、`axc` 全靠猜），
  但恰恰最容易做——基本只是把现成数据 dump 成 JSON。

**D6 — `cap mcp`** ⊘ 阻塞于 DR-03
- 产物：从 cap 数据生成 MCP tool schema，供 `Jarvis-Agent` 的 5 个 `plot_*` 工具使用。
- 收益：Jarvis-Agent 的 tool schema 从手写变成生成，跨仓漂移消失。
- 代价：PLOT 要承认 MCP 这个下游形态。

---

### 轨 E — 渲染读回（闭环④）

| ID | 标题 | 里程碑 | 依赖 | 规模 | 破坏 | 状态 |
|---|---|---|---|---|---|---|
| E1 | 渲染期观测点采集 | M2 | A1 | M | 否 | ☐ |
| E2 | `JP-VIZ-001…009` 体检规则 | M2 | E1, B4 | L | 否 | ☐ |
| E3 | 行数账本（row ledger） | M2 | E1 | S | 否 | ☐ |
| E4 | 数值孪生 `--with-data` | M2 | E1 | M | 否 | ☐ |
| E5 | ASCII 缩略图原型 | M3 | E1 | S | 否 | ⊘ DR-04 |

**E1 — 渲染期观测点采集**
- 产物：`jarvisplot/Figure/render_observations.py`——在 adapter 调用 matplotlib 之前，记录每层的
  `n_points / finite_ratio / data_bbox / axes_lim / zorder / nan_ratio / cmap_range`。
- 验收：渲染任一 example 后能拿到每层观测记录。
- 测试：`tests/test_render_observations.py`。
- 备注：**渲染时这些数据全在手上，采集成本几乎为零。**所有权纪律：观测点必须取**送进 matplotlib 的
  那份数组**，不能重算（重算 = 第二条流水线 = 必然漂移）。

**E2 — `JP-VIZ-001…009` 体检规则** ⭐
- 产物：把"图画错了"从不可解的多模态问题降成**可枚举的确定性检查**：

  | 码 | 症状 |
  |---|---|
  | `JP-VIZ-001` | 该 axes 上零个可见元素（空图） |
  | `JP-VIZ-002` | N% 的点落在 `lim` 外被裁掉（阈值 >50% 报 warning，>90% 报 error） |
  | `JP-VIZ-003` | transform 后行数归零 |
  | `JP-VIZ-004` | colorbar `vmax` 远小于数据 max → 大面积饱和成同一色 |
  | `JP-VIZ-005` | `scale: log` 但该轴有非正值，被 matplotlib 静默丢弃 |
  | `JP-VIZ-006` | 某层被更高 zorder 的层完全遮挡（画了等于没画） |
  | `JP-VIZ-007` | 插值网格 NaN 占比过高（凸包外） |
  | `JP-VIZ-008` | 全部数据点集中在 <1% 的 axes 面积内 |
  | `JP-VIZ-009` | legend 引用了不存在或不可见的 handle |

- 验收：为每个码造一个必然触发的 fixture YAML，`jplot doctor` 能报出对应码。
- 测试：`tests/test_render_health.py`，9 个 fixture。
- 备注：**这是整套方案里我最看好的一条。**现有规格的 `--with-data` 只给数值孪生——**数值 ≠ 判断**，
  agent 拿到一张数组表仍然不知道图对不对。`JP-VIZ-004` 直接告诉它"去调 vmax"。

**E3 — 行数账本**
- 产物：每步 transform 前后行数，`filter "LogL > -100": 204,132 → 0 ⚠ JP-VIZ-003`。
- 验收：`jplot dryrun` 输出账本。
- 测试：`tests/test_row_ledger.py`。
- 备注：空图的头号成因就是某步 filter 把数据滤光了。

**E4 — 数值孪生 `--with-data`**
- 产物：每层 parquet sidecar（复用已有 `to_parquet`）。
- 验收：sidecar 行数与 E1 观测的 `n_points` 一致。
- 测试：`tests/test_numeric_twin.py`。
- 备注：对应 frozen 规格的 JP-A5。**所有权纪律同 E1**。

**E5 — ASCII 缩略图原型** ⊘ 阻塞于 DR-04
- 产物：`--thumb-ascii 32x16`，把 2D 密度场降成字符图。
- 验收：在 EggBox（多峰后验）上人工判读，看是否能分辨峰数与位置。
- 备注：**先做原型验证再决定是否进主线。**几十个 token 换非多模态模型的"低保真视觉"，
  对"结构性"问题（几个峰、峰在哪、是不是全空）可能真的有用。

---

### 轨 F — 起草与修改（横切）

| ID | 标题 | 里程碑 | 依赖 | 规模 | 破坏 | 状态 |
|---|---|---|---|---|---|---|
| F1 | `template list \| show` type 化模板 + slot schema | M3 | D5 | M | 否 | ☐ |
| F2 | `suggest` 数据感知合成 + `decisions[]` | M3 | C1, C6, F1 | L | 否 | ☐ |
| F3 | 地址语法 + `layers[].name` 必填/自动命名 | M3 | B3 | M | **是** | ☐ |
| F4 | `config get / set`（round-trip 保留注释） | M3 | F3 | L | 否 | ⊘ DR-02 |
| F5 | `config add-layer / rm / move` | M3 | F4 | M | 否 | ⊘ DR-02 |
| F6 | `config fmt` + `diff --semantic` | M3 | F4, G5 | M | 否 | ⊘ DR-02 |

**F1 — `template`**
- 产物：吐 type 化模板（`posterior_2d` 等）+ 每个槽的 slot schema（`source_hint` 合约）。
- 验收：`jplot template show posterior_2d` 的输出填上真列名后能直接 `jplot <yaml>` 渲染。
- 测试：`tests/test_template_catalog.py`。
- 备注：对应 frozen 规格 JP-A4。

**F2 — `suggest` 数据感知合成** ⭐
- 产物：`jplot suggest --data samples.hdf5 --kind posterior_2d --x m_A --y tanb --weight "exp(LogL)"`。
  **分工纪律：agent 出意图结构，PLOT 出数值。**PLOT 自动定 `scale`/`lim`/`bins`，
  **每个决定带 `reason`**。
- 验收：对 EggBox 数据能一步产出可 `jplot <yaml>` 渲染通过的 YAML；`decisions[]` 每项 `reason` 非空。
- 测试：`tests/test_suggest.py` — 产出必须过 `jplot validate`。
- 备注：**这是 NL→YAML 的第一跳。**`reason` 的作用是让 agent 能把理由复述给用户，
  而不是端出一份来历不明的 YAML。

**F3 — 地址语法 + `layers[].name` 必填**
- 产物：地址语法 `Figures[EggBox].layers[_density].style.cmap`（按名字定位，不按下标）。
  `layers[].name` 变必填，或由 normalizer 自动命名（`_layer0`…）。
- 验收：调换 layers 顺序后同一地址仍指向同一层。
- 测试：`tests/test_config_addressing.py`。
- 备注：**跨文档依赖**——`V2_AGENT_CLI_SURFACE.md` §7.2 提出这条约束，但它属于 YAML 结构，
  必须回写进 `V2_YAML_AGENT_ERGONOMICS.md` 的 R1 词汇表小节。**这条回写尚未做，见 H4 备注。**
  `$.Figures[0].layers[2]` 这种下标寻址一旦顺序变就全错，所以必须按名字。

**F4 — `config get / set`** ⊘ 阻塞于 DR-02
- 产物：引入 `ruamel.yaml` round-trip（保留注释、保留键序）。三条写操作纪律：
  1. **保留注释**——agent 改一个 `vmax` 不该把人写的注释全冲掉（这是 agent 编辑配置最常见的破坏行为）；
  2. **写-验-回滚**——落盘前内部 validate，不通过不写。**结果是 agent 永远不可能在磁盘上留下坏 YAML**；
  3. **`--diff` 默认**，`--write` 显式。
- 验收：对带注释的 YAML 执行 `config set` 后，`git diff` 只有目标那一行。
- 测试：`tests/test_config_edit.py` — 注释保留、坏值回滚、diff 最小性。

**F5 / F6** — 依赖 F4，细节同上。`diff --semantic` 在 `type:` 展开后比较，避免糖层差异淹没真实差异。

---

### 轨 G — YAML 结构重构（V2 break）

| ID | 标题 | 里程碑 | 依赖 | 规模 | 破坏 | 状态 |
|---|---|---|---|---|---|---|
| G1 | R2 transform 改判别式 `{type: …}` | M4 | B1 | L | **是** | ☐ |
| G2 | R3 `coordinates` / `columns` 分裂 | M4 | B3 | L | **是** | ⊘ DR-06 |
| G3 | R4 One Obvious Way（消灭多形态输入） | M4 | G5 | M | **是** | ☐ |
| G4 | R5 颜色所有权唯一化 | M4 | B4 | S | **是** | ☐ |
| G5 | R6 别名归一化收进单个 normalizer | M4 | B3 | M | 否 | ☐ |
| G6 | R7 `type:` 升格为主接口 + **先补齐缺失的 type** | M4 | F1 | XL | 否 | ⊘ DR-07 |
| G7 | `jplot migrate` 1.x → 2.0 | M4 | G1–G5 | L | 否 | ☐ |

**G1 — R2 transform 判别式**
- 从 `- filter: "LogL > -100"` 改为 `- {type: filter, expr: "LogL > -100"}`。
- 三条硬理由（**不是审美**）：① 单键 dict 只能写成 11 分支 `oneOf`，错误退化成
  `is not valid under any of the given schemas`，对 agent 毫无用处；② agent 常把子键缩进写错一级，
  YAML 仍合法但语义全变；③ 无法表达每步共有的元字段（`enabled`/`name`/`comment`）。
- 备注：`preprocessor_runtime` 已部分支持 `type:` 写法（`YAML_DESIGN.md` §10.7 提到旧的
  `type: make_density_core` 继续生效），**这条路已经趟过一半**。

**G2 — R3 `coordinates` / `columns` 分裂** ⊘ 阻塞于 DR-06
- 当前 YAML **最让 agent 困惑的一处**：三个形状相同、语义不同的 dict。
  `layers[].coordinates.x` 里的 `name/lim/scale/label` 全部**静默忽略**——agent 理所当然地写 `lim`
  （完全合理的推断），得到一张范围不对的图，**没有任何提示**。
- 倾向方案 (a)：layer 用 `columns: {x: …, y: …}`（纯标量映射），transform / figure-type 用
  `coordinates`（带 `lim`/`scale`/`name`）。**形状不同才是最强的防混淆手段。**

**G5 — R6 normalizer**
- `lim` vs `limits`、`bin` vs `bins` 现在散落在 runtime 各处的 `.get(a, x.get(b))` 里。
  收进单个 normalizer，**之后 runtime 只见规范名**。
- ~~`rectcmap` vs `rect_cmap`~~ —— **不是别名问题，见 §9.3**：token 是 `rectcmap`，
  `rect_cmap.json` 只是 `style_preference.json` 映射过去的文件名，两者不在同一层。
- 真正要处理的：**`combine: seperate` 的拼写错在 runtime 代码里**，改名必须留别名。
- 两个副作用都是好的：schema 只描述规范名；诊断只报规范名。
- 备注：**G5 不破坏兼容**（别名仍接受，只是集中处理），可以先于 G3 落地。

**G3 — R4 理由**：不是洁癖。**面对 3 种合法写法，agent 会挑训练数据里最常见的那种，
而不是这个项目里最常见的那种。**单一写法把这类错误直接归零。

**G6 — R7** ⊘ 阻塞于 DR-07
- `posterior_2d` 等 4 个 type **已实现**（`Figure/figure_types.py`），但文档里被定位成"纯语法糖"，
  排在 2230 行文档的第十一~十三章。**从头读文档的 agent 会先学会那套 90 行的手写 layer 栈。**
- 对照数据：`YAML_DESIGN.md` §11.9 自己给的例子是 **92 行 → 11 行**。
- 待补 type：`hist_1d` / `line_2d` / `corner` / `ternary_2d` / `dynesty_diagnostics`。

---

### 轨 H — 文档与语料

| ID | 标题 | 里程碑 | 依赖 | 规模 | 破坏 | 状态 |
|---|---|---|---|---|---|---|
| H1 | `examples list \| show` + CI 验证语料 | M4 | A2 | M | 否 | ☐ |
| H2 | `explain <code \| type \| yaml>` | M2 | B4 | M | 否 | ☐ |
| H3 | `YAML_DESIGN.md` 重排（type 前置） | M4 | G6 | L | 否 | ⊘ DR-07 |
| H4 | `AGENT_PLAYBOOK.md`（给 agent 读的一页纸） | M1 | D5, C1 | S | 否 | ☐ |

**H1 — `examples`**：CI 验证过的真实 YAML 语料。**few-shot 比 schema 更有效**——但前提是语料必须
在 CI 里真的跑得过，否则就是新的幻觉源。

**H2 — `explain`**：rustc `--explain` 模式。`jplot explain JP-VIZ-004` 打印成因 + 修法 + 最小示例；
`jplot explain posterior_2d --expand config.yaml` 打印糖展开后的低层 YAML（让 agent 能平滑降级到手写）。
**省上下文 token**：agent 不必把 2230 行文档塞进上下文。

**H4 — `AGENT_PLAYBOOK.md`**：一页纸，内容只有两条纪律 + 对应命令：
1. 列名只能来自 `jplot data describe`（已有纪律，见 `Jarvis-Agent/Docs/PLOT_TOOLS.md` §3.1）；
2. **PLOT 词汇表里的任何字符串**（method / figure type / style card / cmap / transform / 轴名 /
   表达式函数）**只能来自 `jplot cap`**（新纪律）。
- 备注：写这份文档时**顺手把 F3 的"`layers[].name` 必填"回写进 `V2_YAML_AGENT_ERGONOMICS.md`**，
  这是目前唯一悬空的跨文档依赖。

---

## 5. 反哺 Jarvis-Agent 的接口清单

`Jarvis-Agent/Docs/PLOT_TOOLS.md`（milestone M4.6）已定义 5 个消费工具。本台账与之的对应：

| Jarvis-Agent 工具 | 依赖本台账任务 | 现状 |
|---|---|---|
| `plot_describe` | C1, C2, C3, C4 | PLOT 侧无出口 |
| `plot_analyze` | JP-A3（frozen）+ E4 | frozen |
| `plot_template` | F1 | frozen |
| `plot_validate` | A1, B3, B4, B6 | PLOT 侧无实现 |
| `plot_render` | E1, E2, E4 | 部分（渲染有，读回无） |
| **（缺）`plot_capabilities`** | D5 | **两边规格都没有** |
| **（缺）`plot_edit`** | F4, F5 | **两边规格都没有** |

后两行是这次 brainstorm 新识别出的缺口。`plot_edit` 直接对应用户提的"**或者修改 YAML**"。

---

## 6. 决策记录（DR）

| ID | 问题 | 倾向 | 阻塞的任务 | 状态 |
|---|---|---|---|---|
| DR-01 | **`AGENT_DATA_API` 的 frozen 要不要解？** 本台账一切都建在它的 envelope 之上 | 至少解冻 JP-A1（envelope + validate），否则每一条都无处安放 | 全部 A 轨的对外承诺 | **待拍板** |
| DR-02 | `config set` 由 PLOT 自己做，还是 agent 用文本工具改、PLOT 只提供 validate？ | **PLOT 自己做**——"agent 不可能留下坏 YAML"这个保证值 ruamel 依赖 + 一套地址语法的成本 | F4, F5, F6 | **待拍板** |
| DR-03 | `cap mcp` 生成 MCP tool schema 值不值得？ | 做——跨仓 schema 漂移消失；代价是 PLOT 承认 MCP 这个下游形态 | D6 | **待拍板** |
| DR-04 | ASCII 缩略图是玩具还是真需求？ | 低成本，**先做原型在 EggBox 上验一次**再决定 | E5 | **待拍板** |
| DR-05 | schema 手写还是从 registry 生成？ | **手写 + CI 一致性检查**——与 HEP2 同构，维护心智一致；坐标合约和 example 本来就得手写 | B5 | **待拍板** |
| DR-06 | R3 用 (a) `columns:` 分裂 / (b) 报错指向 frame / (c) 让 layer `lim` 生效？ | **(a)**——形状不同是最强防混淆；(c) 明确反对（会造出第二个 frame 所有权，更糟） | G2 | **待拍板** |
| DR-07 | `type:` 是糖还是主接口？ | **主接口**——但这意味着它成为兼容性承诺的对象，不能再随便改 | G6, H3 | **待拍板** |
| DR-08 | 渲染要不要做成 `jplot run`，以便和 `Jarvis2 run` 对称？ | **不要。** `Jarvis2 plot ≡ jplot` 全 argv 透传；`Jarvis2 plot run scene.yaml` 会让 `run` 在 scan/plot 两侧语义打架。**规范：`jplot <yaml>` / `Jarvis2 plot <yaml>` = 渲染；动词只承载 validate/data/cap/…** 渲染附加行为用 flag（`--report` 等）。禁止广告或实现 `jplot run`。 | A2, E1, help/args.json, 所有文档示例 | **已拍板（2026-08-06）** |

**DR-01 是总闸。**在它拍板前，A 轨按"内部实现"推进，不写进 `docs/specs/`。
**DR-08 已拍板**：任何新文档/help/示例不得再写 `jplot run`；渲染附加能力挂在裸路径 flag 上。

---

## 7. 施工日志

格式：`YYYY-MM-DD | 任务ID | 动作 | 结果`。**追加，不改历史行。**

> **2026-08-06 状态**：下面标 ☑ / ◐ 的条目是一轮探路施工的成果，**决定保留**。
> 全部在工作树里，**尚未 commit**；全套 **289 passed**（1 skipped），24 份真实 YAML 语料零假阳性。
> ☑ = 有产物 + 有测试 + 验收过；◐ = 有产物但未接动词或未测（逐条见任务明细里的状态块）。
> 副产物见 §9：brainstorm 里若干条推测被证伪，计划已按实测改写。

```
2026-08-06 | DR-08 | 拍板 | 渲染=裸路径 `jplot <yaml>` / `Jarvis2 plot <yaml>`；禁止 `jplot run`。
                        理由：Jarvis2 plot 透传 jplot，plot run 与 scan run 语义打架。
                        落地：去掉 LEGACY_ALIAS run；RESERVED_NON_VERBS 拒绝 jplot run（exit 2）
2026-08-06 | D1-5| 完成 | verbs/cap.py 注册；tests/test_capabilities.py；args.json 列 cap；
                        styles 报 usable/axes；to_parquet 进 transform schema
2026-08-06 | C1-2| 完成 | verbs/data.py describe --json；role_hint（LogL/weight/…）；
                        tests/test_data_describe.py
2026-08-06 | —   | 修  | expr_names.py 与 build_eval_globals 同源忽略表（ln/log10/Min 不再假阳性）
2026-08-06 | M1  | 进度 | 反幻觉两条白名单 CLI 出口已通：`jplot data describe` + `jplot cap all`
                        仍缺 C3–C5 / B4 / 语料上的 did-you-mean 全覆盖等
2026-08-06 | —   | 建账 | 从两份 V2 brainstorm 落成 46 条任务 / 5 个里程碑 / 7 条待拍板 DR
2026-08-06 | A1  | 完成 | 新增 diagnostics.py（Diagnostic/Fix/DiagnosticBag/did_you_mean/join_path）
                        + agent_io.py（envelope/emit/exit_code_for）；23 条测试
2026-08-06 | A2  | 完成 | 新增 verbs/ 包（前置路由，不动 legacy 解析器）+ validation.py
                        + verbs/validate.py；args.json 加 commands 表，root help 从数据生成
2026-08-06 | A2  | 修正 | JP-REF-001 对 share_data 发布的名字误报（bin/ 语料上 45 → 1）；
                        剩下那 1 条是 HinoLLP.yaml 里真实的笔误 dfmuTBv1 → dfmuTB01
2026-08-06 | A3  | 完成 | core.py loguru sink 从 stdout 改 stderr；`jplot <file>` stdout 现为 0 字节
2026-08-06 | A4  | 完成 | envelope 键集合冻结测试 + 遍历 VERBS 强制 --json 约定的参数化测试
2026-08-06 | B7  | 完成 | 子进程守卫：validate 后 sys.modules 无 matplotlib/pandas/polars/h5py/scipy/shapely
2026-08-06 | —   | 副作用 | docs status 词汇表加 brainstorm / active ledger，并约束其只能出现在 docs/roadmap/
2026-08-06 | M0  | 出口 | 全套 233 passed；`jplot validate x.yaml --json` 一轮报全、stdout 纯 JSON
2026-08-06 | B1  | 完成 | jarvisplot/schema/{manifest,plot-config,core/*}.json + schema_catalog.py
                        （本地 Registry，$ref 永不出网；manifest 纯数据）
2026-08-06 | B2  | 完成 | x-jarvis-zone 自检 lint；只保留 closed/delegated，不抄 HEP2 的 open zone。
                        lint 第一次跑就抓到自己两处疏漏（columnmap 无 zone、if 断言被误判为面定义）
2026-08-06 | B3  | 完成 | 闭合词汇表接进 validate；schema_diagnostics.py 把 jsonschema 错误翻译成
                        带 did-you-mean + example + fix 的诊断。**case A/B/C 全部消灭**
2026-08-06 | B3  | 重构 | schema 独占"形状"，validation.py 只留文件存在性/重名/跨引用/静默忽略。
                        删掉 JP-SCH-010/011、JP-FIG-001/002、JP-LAY-001/002、JP-DAT-001/002/003
                        —— 一个问题只出一条诊断
2026-08-06 | B3  | 语料 | 全 24 份真实 YAML 零假阳性。发现三处词汇表缺口（columns.rename 的
                        source/target、combine: seperate、坐标字面量数组）并修正
2026-08-06 | —   | 发现 | 本仓库自己的配置里 13 个 figure 写了 figure 层 `legend:`，runtime 从不读
                        （只读 frame.<ax>.legend）—— 这些图例一直没渲染。新增 JP-OWN-001 报出来
2026-08-06 | —   | 发现 | bin/HinoLLP.yaml 的 source: dfmuTBv1 是笔误（应为 dfmuTB01）；
                        bin/EggBox_Dynesty_06.yaml 的 method: grid_profile 不存在。两条都是真 bug
2026-08-06 | —   | 打包 | pyproject package-data 加 schema/**/*.json，否则 wheel 里没有 schema
2026-08-06 | B6  | 重构 | core_runtime 的纯表达式分析器（312 行）抽成 column_demand.py，两边共用。
                        **不能直接复用 plan_dataset_required_columns**：它把列需求向所有 dataset
                        做并集（对剪枝无害、对存在性判断致命），所以新写 plan_source_demand 保精度
2026-08-06 | B6  | 完成 | column_probe.py（只读 header/元数据，csv/parquet/hdf5）+ JP-COL-001。
                        **case D 消灭**，且诊断指向出错的表达式本身而非 DataSet 条目
2026-08-06 | B6  | 修正 | 列名含点（pVa.E）被符号提取拆成 pVa + E → 8 条假阳性。
                        ColumnProbe.resolves() 接受真实列名的标识符片段，归零
2026-08-06 | B7  | 补强 | 守卫拆成两档：matplotlib/scipy/shapely 永不加载；
                        --no-columns 连 pandas/h5py/polars/pyarrow 都不加载
2026-08-06 | M1  | 进度 | 269 passed。四例实测全部消灭（A/B/C/D）。语料零假阳性
2026-08-06 | B3  | 补做 | 发现 transform 的 if/elif 链无 else（写错步骤名整步静默跳过），
                        补 core/transform.json 闭合词汇表。`- fitler:` 现在报 did-you-mean
2026-08-06 | D1-5| 部分 | capabilities.py 七个采集器跑通并核对过真实数据；动词与测试未做（◐）
2026-08-06 | —   | 发现 | 12 张 style card 有 2 张顶层是 Figure 而非 Frame，
                        figure.py:279 直接 bundle["Frame"] → 一用就 KeyError。见 §9.7
2026-08-06 | —   | 收口 | 271 passed。按"代码保留、标注状态"停止施工，转回纯计划
```

---

## 9. 实测修正（原 brainstorm 的推测被证伪的部分）

建账后做了一轮探路施工，把两份 brainstorm 里靠推测得出的结论拿真代码和真语料验了一遍。
**下面每一条都推翻或修改了原文，计划按这里为准。**

### 9.1 静默失败点比原来数的多

原诊断表列了 4 例。实际上"静默接受"是**一整类**，至少还有两处，都在生产配置里活着：

| 位置 | 表现 | 波及 |
|---|---|---|
| **transform 步骤名写错** | `Figure/preprocessor_runtime.py::apply_transforms_impl` 的 if/elif 链**没有 else 分支**。写 `- fitler:` 整步被跳过，图照画，数据没过滤，无任何提示 | 未知，但这是最危险的一处——**图看起来是对的** |
| **figure 层 `legend:`** | runtime 只读 `frame.<ax>.legend`，figure 层的 `legend` 从不读 | **本仓库自己的配置里 13 个 figure**，图例一直没渲染出来 |

**对计划的影响**：B3 的验收标准要从"消灭 case A/B"扩成"消灭静默接受这一类"，
并且 transform 词汇表必须单独进 schema（`core/transform.json`），不能等 G1/R2。

### 9.2 figure type 只有 2 个，不是 4 个

`Figure/figure_types.py::expand_figure_type` 只派发 `posterior_2d` 和 `profile_2d`。
brainstorm §R7 说的 `scatter_2d` / `posterior_1d` **不存在**。

**对计划的影响**：G6（R7 升格为主接口）不是"文档重排"，是**先要把糖补齐**。
规模从 L 上调，且它挡在"agent 的默认路径从 90 行变 10 行"前面。

### 9.3 `rectcmap` / `rect_cmap` 不是别名 bug

YAML 里的 token 就是 `rectcmap`；`rect_cmap.json` 只是 `cards/style_preference.json`
映射过去的**文件名**。两者本来就不在同一层。

**对计划的影响**：G5（R6 别名归一化）的例子要换掉。真正的别名问题只剩 `lim`/`limits`、`bin`/`bins`。
另外发现一个真的：**`combine: seperate` 的拼写错在 runtime 代码里**
（`layer_runtime.py:648`），所以 G3/R4 不能直接改名，必须留别名。

### 9.4 B6 不能"复用 15 行"

brainstorm §3.3a 说列存在性校验可以复用 `plan_dataset_required_columns()`，"~15 行"。
**不行。**那个函数把 `global_layer_cols` 并进**每一个** dataset 的需求集——
对列剪枝是安全的过近似，对存在性判断是致命的假阳性源。

**对计划的影响**：B6 规模从 S 上调到 M，且要先把纯表达式分析器从 `core_runtime.py` 抽出来
（312 行，两个消费者共用），再写一个**保精度**的 `plan_source_demand`。

另一个必须处理的坑：列名里带点（真实语料里有 `pVa.E`）会被标识符正则拆成 `pVa` + `E`，
任何基于符号提取的列检查都必须接受"真实列名的标识符片段"，否则一批假阳性。

### 9.5 `validate` 的成本不是一档，是两档

"零 I/O 校验"和"列存在性校验"不可兼得——后者要读文件头，就要 pandas/h5py。

**对计划的影响**：B7 的守卫拆成两条不变量：
- **matplotlib / scipy / shapely 永不加载**（这是"一轮报全"的物理前提，无条件成立）
- **`--no-columns` 时连数据库都不加载**（纯形状判决）

### 9.6 schema 和手写检查会重复报同一件事

一开始 schema 归 schema、手写检查归手写检查，结果 `layers: {}` 同时触发两条诊断。

**对计划的影响**：所有权必须写进计划，不能等施工时再想：

| 归 schema catalog | 归 validation.py |
|---|---|
| 词汇表、类型、枚举、required | 文件存在性、名字唯一性、跨块引用、静默忽略的键 |

### 9.7 12 张 style card 里有 2 张是坏的

`cards/style_preference.json` 广告了 12 个 `[bundle, token]` 组合。其中
`a4paper_1x1/Ternary` 和 `gambit_1x1/Ternary` 的卡文件顶层键是 `Figure`，
而其余 13 个文件是 `Frame`。`Figure/figure.py:279` 直接 `bundle["Frame"]`，
所以这两张卡**一用就 KeyError**，不是降级而是崩。

仓库里没有配置在用它们，所以一直没暴露。但 agent 只要读 `style_preference.json`
就会理所当然地挑一个。

**对计划的影响**：D2（`cap styles`）不能只列卡，必须带 `usable` 字段和不可用原因——
"广告了但用不了"和"不存在"对 agent 是两种完全不同的信息。
另外这属于 §8"明确不做的事"之外的一笔债：要么修卡，要么从 `style_preference.json` 摘掉。

### 9.9 顺带查出两个真 bug（本仓库自己的配置）

- `bin/HinoLLP.yaml` `Figures[2]`：`source: dfmuTBv1` 不存在，最近的是 `dfmuTB01`
- `bin/EggBox_Dynesty_06.yaml` `Figures[3]`：`method: grid_profile` 不在 `METHOD_DISPATCH` 里

两条今天都只会在渲染期炸。**这本身就是 M1 价值的证据**：语料上零假阳性，捞出的全是真问题。

### 9.10 对 DR 的影响

- **DR-05（schema 手写还是生成）**：倾向"手写 + CI 校验"得到验证。手写 schema 的
  `description` 字段承载了大量"runtime 到底读不读这个键"的知识，registry 里没有这些信息，生成不出来。
- **DR-01（解冻）更紧迫**：探路发现 envelope 需要一个 frozen 规格里没有的 `diagnostics` 顶层数组
  （原规格只有单数 `error`，无法表达"这个配置有 9 个问题"，而一轮报全正是整条 A 轨的意义）。

---

## 8. 明确不做的事

- **HEP2 的 ASCII gate（`JV2-ENC-001`）**。HEP2 禁非 ASCII 是因为 `Scan.name` 会变成目录名、
  tar 成员、HDF5 属性。PLOT 的 label 是给 matplotlib 的，**中文/希腊字母标签是合法需求**。
  只对"会变成文件名的字段"（`project.name` / `Figures[].name` / `output.dir`）做路径安全检查。
- **HEP2 的 `open` zone + 警告**。那是 HEP2 的历史包袱（RLTPMCMC 的 `Control`/`Reward`/`PPO`）。
  PLOT V2 既然允许 break，就只留 `closed` 和 `delegated`，不给自己开后门。
- **第二条渲染流水线**。所有观测点（E1/E4）必须取送进 matplotlib 的那份数组。
  沿用 `IMPLEMENTATION_ROADMAP.md` §4 的既有纪律：*"keep Agent Data API as a thin skin over
  existing loaders/transforms/cache, never a pipeline fork."*
- **给 `layers[].style` 写穷举 schema**。它是 `delegated` zone，下游是 matplotlib kwargs。
  PLOT 只管所有权冲突（`JP-OWN-002`），不管 kwargs 合法性。
