# Jarvis-PLOT V2 — YAML 结构与 Agent 工效学 Brainstorm

Status: brainstorm
Date: 2026-08-06
对照物: Jarvis-HEP v2 (`Jarvis-Workshop/Jarvis-HEP-v2`)
相关文档: `docs/design/YAML_DESIGN.md`、`docs/roadmap/soft-cooking-wilkinson.md`、`docs/specs/AGENT_DATA_API.md`

---

## 0. 结论先行

1. **Jarvis-PLOT 当前最危险的失败模式不是"报错难懂"，而是"静默接受"。** 拼错的 key 被 `.get()` 忽略，
   图照画、日志报 success、输出是一张空图。Agent 拿到 exit 0 就认为成功了。
2. **Jarvis-HEP v2 已经把这个问题系统性解决了**（schema catalog + 所有权分区 + 稳定错误码 + suggestion/example
   + did-you-mean）。这套机制是**领域无关的**，可以整体搬到 PLOT，不需要重新发明。
3. **但 PLOT 的派发面比 HEP2 大得多**（5 条派发轴 vs 2 条），所以 manifest 分文件 schema 的收益比 HEP2 **更高**，
   不是更低。
4. **PLOT 有一类 HEP2 没有的问题：配置依赖用户数据。** Agent 不知道用户 HDF5 里的列名，只能猜。
   这类错误 schema 治不了，只能靠 **introspection（`describe` / `capabilities`）+ 早期列存在性校验**。
   好消息：`core_runtime.plan_dataset_required_columns()` 已经在渲染前算出了"每个 DataSet 需要哪些列"的
   完整需求表，**只差和实际列名做一次比对并报告**。
5. **`type:` 封装层（posterior_2d / profile_2d / …）应该从"语法糖"升格为"主接口"。**
   Agent 写 8 行 type 化 YAML 的正确率，远高于写 90 行手工 layer 栈。现在它被埋在 2230 行设计文档的第十一章。

---

## 1. 现状诊断（实测，非推测）

用 4 个"agent 最常犯的错"跑真实 `jplot`，结果：

| # | Agent 的错误 | 现在的行为 | 严重度 |
|---|---|---|---|
| A | `Layers:` 写成大写、`output:` 写成 `outputs:` | **完全静默**。日志 `Succefully loading figure -> f1` + `successfully draw f1`，产出一张空图，exit 0 | 🔴 致命 |
| B | `style:` 写成 `styles:` | `Failed to configure figure 'f1': 'axes'` —— 一个来自完全无关位置的 `KeyError` | 🔴 误导 |
| C | `method: scattr` | `Figure f2 failed: "Unknown method key: 'scattr'"` —— 正确，但**在读完全部数据之后**才报，且没有 did-you-mean、没有合法值列表 | 🟡 迟到 |
| D | `coordinates.x.expr: aa`（列不存在） | `Figure f3 failed: name 'aa' is not defined` —— 没有 YAML 路径、没有"该 DataSet 可用列是 a/b/LogL" | 🟡 信息不足 |

**三类失败模式：**

- **静默接受（A）** — 未知 key 一律被忽略。这是所有失败模式里最坏的一种：agent 没有任何信号可以自我纠正。
- **错位报错（B）** — 报错点距离真实原因十万八千里，agent 会去修 `axes`，越修越错。
- **迟到报错（C/D）** — 全部错误都在**渲染时**、**逐图**暴露。一个 10 图的 YAML 要跑 10 次才能收敛。

**根因只有一条：YAML 层没有 owner。** runtime 直接消费裸 `dict`，用 `.get(key, default)` 穿过所有拼写错误。
没有 schema、没有闭合词汇表、没有 validate gate。

对比：HEP2 的一张 card 在 **YAML parse 之后、任何计算之前**就被 gate 住了。

---

## 2. 从 Jarvis-HEP v2 直接可搬的机制

HEP2 的 `jarvishep2/schema/` + `task_schema.py` + `contracts/` + `diagnostic_guidance.py` 是一套完整方案。
逐条映射：

| HEP2 机制 | 位置 | PLOT 对应物 | 说明 |
|---|---|---|---|
| **manifest.json 中央派发表** | `schema/manifest.json` | `jarvisplot/schema/manifest.json` | 只加载表内文件，绝不联网取 schema。新增一个 method/transform = 加一个 schema 文件 + 加一行 manifest + 加 fixture，**Python loader 不动** |
| **分文件 JSON Schema (2020-12)** | `schema/core/*`, `schema/sampling/methods/*` | `schema/core/*`, `schema/methods/*`, `schema/transforms/*` | 一个派发值一个文件，schema 就是那个特性的文档所在地 |
| **两层验证** | JSON Schema → Python contracts | 同构 | Schema 管结构/类型/词汇表；Python 管跨字段、数值关系、文件存在、表达式解析。**能用 schema 表达的绝不写进 Python，反之亦然** |
| **所有权分区 `x-jarvis-zone`** | `closed` / `delegated` / `open` | 同名照搬 | PLOT 尤其需要（见 §3.4）。catalog 加载时自检：object schema 缺 zone = CI 失败 |
| **稳定错误码 + suggestion + example** | `JV2-*` | `JP-*` | 每条诊断都带"下一步该怎么改"，外加最小可用 YAML 片段 |
| **`x-jarvis-example` 就近示例** | schema 文件内嵌 | 同 | 文档和接口不会漂移 |
| **did-you-mean** | `difflib.get_close_matches` | 同 | `styles` → `style`、`scattr` → `scatter`、`bins` vs `bin` |
| **`validate --json` 机器可读** | `Jarvis2 validate --json` | `jplot validate --json` | 与人类可读报告同源，同一份 `suggestion`/`example` 字段 |
| **多错误汇总表** | Code / YAML path / Problem | 同 | Agent 一次拿到全部错误，一轮就能改完，而不是 fail-fix-fail-fix |

> `docs/specs/AGENT_DATA_API.md` 里 **JP-A1（`--validate --json`）已经写好规格，但被标为 frozen**。
> `jsonschema` 已经在 `pyproject.toml` 依赖里。这是整个仓库里投入产出比最高的一项，建议**优先解冻**。

---

## 3. PLOT 特有的问题（HEP2 没有的）

### 3.1 PLOT 的派发面是 HEP2 的 2.5 倍

HEP2 只有两条派发轴：`Sampling.Method`、calculator I/O 的 `type`。

PLOT 有 **五条**：

```
Figures[].type            → figure type schema   (posterior_2d / profile_2d / scatter_2d / posterior_1d)
Figures[].layers[].method → method schema        (~30 个：scatter / pcolormesh / contour / voronoi / jpfield / …)
transform[] 的 key        → transform schema     (~11 个：filter / profile / make_interp_2d / posterior_density / …)
DataSet[].type            → loader schema        (csv / hdf5 / parquet)
Figures[].style tokens    → style card catalog   (a4paper_2x1 × rect|rectcmap|ternary|… ≈ 20 张卡)
```

派发轴越多，"用一个大 schema 文件 + 一堆 `oneOf`"越不可维护，manifest 分文件的收益就越大。
**HEP2 的 manifest 模式在 PLOT 身上比在 HEP2 身上更值。**

### 3.2 method 的坐标合约 —— schema 能查、文档查不了的东西

`scatter` 要 x,y；`pcolormesh` 要 x,y,z；`fill_between` 要 x,y1,y2；`quiver` 要 x,y,u,v。
这套契约现在**只存在于 `docs/design/YAML_DESIGN.md` 的第四节表格里**，运行时完全不检查——错了就等 matplotlib
在 adapter 深处炸掉。

把它编码进 `schema/methods/<name>.json`：

```json
{
  "$id": "https://jarvis-plot.org/schema/v2/methods/pcolormesh.json",
  "x-jarvis-zone": "closed",
  "type": "object",
  "properties": {
    "coordinates": {
      "type": "object",
      "x-jarvis-zone": "closed",
      "required": ["x", "y", "z"],
      "properties": {
        "x": {"$ref": ".../core/coord.json"},
        "y": {"$ref": ".../core/coord.json"},
        "z": {"$ref": ".../core/coord.json"}
      },
      "additionalProperties": false
    },
    "style": {"type": "object", "x-jarvis-zone": "delegated"}
  },
  "x-jarvis-example": "method: pcolormesh\ncoordinates:\n  x: {expr: x}\n  y: {expr: y}\n  z: {expr: density}"
}
```

顺带一个免费收益：`jarvisplot/Figure/method_registry.py` 已经是唯一权威的 method 名单
（`METHOD_DISPATCH`，30 项）。schema 目录可以**从它生成**，或者反过来加一条 CI 检查确保两边一致——
不会出现"注册了但没 schema"或"schema 有但不能用"。

### 3.3 Agent 不知道用户的数据长什么样

这是 PLOT 和 HEP2 最本质的差异。写 HEP2 task card，agent 主要写路径和数字；写 PLOT YAML，agent 必须
知道**用户 HDF5 里的列名**。它不知道，于是它猜——case D 就是这么来的。

Schema 永远治不了这个（schema 不可能知道 `m_A` 存不存在）。两条正交的解法：

**(a) 早期列存在性校验（低成本、高收益）**

`core_runtime.plan_dataset_required_columns()` 已经在渲染前遍历完整个配置，构造出
`demand: Dict[dataset_name, Set[column]]`（用于列裁剪）。**只需要再加一步：把 demand 和 DataSet 的实际
列名求差集并报告。** 大约 15 行，就能把 case D 变成：

```text
[error] JP-COL-001  $.Figures[0].layers[0].coordinates.x.expr
        Column 'aa' is not available in DataSet 'd1'.
        available: a, b, LogL
        suggestion: Did you mean 'a'? Or add it with a transform `add_column` step.
```

**(b) introspection 动词**

```bash
jplot describe samples.hdf5 --json      # 列名 / dtype / 行数 / 分位数 / HDF5 树
jplot capabilities --json               # methods+坐标合约 / transforms+参数 / style cards / colormaps / figure types
```

`capabilities` 是 HEP2 没有、PLOT 特别需要的：PLOT 的词汇表规模是
30 methods × 11 transforms × 20 style cards × N colormaps。Agent 在会话开头拉一次，
之后就不需要靠"记忆里的 matplotlib 知识"去赌 `jarvis_rainbow2_r` 到底叫什么。
（`data_loader_summary.py` 已经能做 describe 的一大半。）

### 3.4 `style:` 是天然的 delegated zone

PLOT 不可能穷举 matplotlib 的所有 kwargs，所以 `layers[].style` **必须**是 `delegated` —— 未知键放行。
这正是 HEP2 zone 概念的价值：不是"要么全查要么不查"，而是**声明谁负责**。

但 delegated 不等于放弃。有一小撮 style 键和 `frame` 抢所有权，应该显式报冲突：

```text
[warn] JP-OWN-002  $.Figures[0].layers[1].style.cmap
       Colour mapping is owned by frame.axc.color for a layer bound to a colorbar.
       This layer sets style.cmap and frame.axc.color.cmap; the frame value wins.
       suggestion: Remove style.cmap, or detach this layer from the colorbar.
```

（`vmin`/`vmax` 同理。这是 `YAML_DESIGN.md` §六 自己标注的"⚠️ 复杂度"之一。）

---

## 4. YAML 结构本身的重构提案

V2 允许 break（见 `soft-cooking-wilkinson.md` 的 User Decisions），所以下面都按"可以破坏兼容"来提。

### R1 — 闭合根词汇表【最高优先，且不破坏兼容】

根只有 `version` / `project` / `DataSet` / `Figures` / `Functions` / `output`。多一个键就是错误 + did-you-mean。
同样闭合 figure 层（`name`/`enable`/`type`/`style`/`frame`/`layers`/`debug`/…）和 layer 层
（`name`/`data`/`share_data`/`axes`/`method`/`coordinates`/`style`/`colorbar`）。

**这一条单独就消灭了 case A 和 case B**，且对合法 YAML 零影响，1.x 就可以上。

### R2 — transform 从"单键 dict"改成"判别式 dict"

现在：

```yaml
transform:
  - filter: "LogL > -100"
  - profile: {method: bridson, bin: 100, ...}
```

问题不是审美，是三条硬伤：
1. JSON Schema 只能写成 11 分支的 `oneOf` + `minProperties/maxProperties`，错误信息退化成
   `is not valid under any of the given schemas` —— 对 agent 毫无用处。
2. Agent 经常把 `- profile:` 的子键的缩进写错一级，YAML 依然合法，语义完全不同。
3. 无法表达"每一步共有的元字段"（如 `enabled`、`name`、`comment`）。

V2：

```yaml
transform:
  - {type: filter, expr: "LogL > -100"}
  - {type: profile, method: bridson, bins: 100, x: {...}, y: {...}, z: {...}}
```

`type` 作为 discriminator，manifest 精确派发到 `schema/transforms/profile.json`，错误信息可以精确到
`$.Figures[0].layers[0].data[0].transform[1]: type=profile requires 'z'`。
（`preprocessor_runtime` 里其实已经部分支持 `type:` 写法——见 YAML_DESIGN §10.7 "旧的 `type: make_density_core` 写法继续生效"，
说明这条路已经趟过一半。）

### R3 — 两个同形不同义的 `coordinates` 命名空间

这是当前 YAML **最让 agent 困惑的一处**：

| 位置 | `expr` | `name` | `lim` | `scale` | `label` |
|---|---|---|---|---|---|
| `layers[].coordinates.x` | ✅ 唯一有效 | ❌ 静默忽略 | ❌ 静默忽略 | ❌ 静默忽略 | ❌ 静默忽略 |
| `transform[].coordinates.x` | ✅ | ✅ 输出列名 | ✅ | ✅ | ❌ |
| `type:` 封装的 `x` | ✅ | ✅ | ✅ | ✅ | ✅ |

三种形状一样、语义不同的 dict。Agent 会理所当然地在 layer 里写 `lim`（完全合理的推断），
然后得到一张范围不对的图，而且**没有任何提示**。

V2 三选一，我倾向 (a)：

- **(a) 分裂类型名**：layer 用 `columns: {x: xcol, y: ycol, z: zcol}`（纯列/表达式映射，标量），
  transform 和 figure-type 用 `coordinates`（带 `lim`/`scale`/`name`）。
  形状不同 → agent 不会混。schema 直接把 layer 的 `lim` 判为 unknown key + suggestion "轴范围写在 frame.ax.xlim"。
- (b) 保持一个 `CoordSpec` 类型，但 layer 里出现 `lim`/`scale` 时报 `JP-OWN-*` 错误并指向 frame。
- (c) 让 layer 的 `lim` 真的生效（提升为 frame 的 lim）—— 反对：会造出第二个 frame 所有权，更糟。

### R4 — One Obvious Way（消灭多形态输入）

| 键 | 现在接受 | V2 只留 |
|---|---|---|
| `keep_columns` / `drop_columns` | `str` \| `list` \| `dict` | `list` |
| `grid` | `int` \| `[nx,ny]` \| `{nx,ny}` | `int` \| `[nx,ny]` |
| `bins` / `bin` | 两个名字都行 | `bins` |
| `lim` / `limits` | 两个名字都行 | `lim` |
| `hpd` / `credible_region` | `false` \| `{}` \| `{enabled: false}` | `{enabled: bool, ...}` |
| `source` | `str` \| `list` | 两者都留（语义确实不同：单源 vs concat） |

理由不是洁癖：**面对 3 种合法写法，agent 会挑训练数据里最常见的那种，而不是这个项目里最常见的那种。**
单一写法直接把这类错误归零，代价只是一次迁移。

### R5 — 颜色所有权唯一化

`frame.axc.color.{cmap,scale,vmin,vmax}` 是唯一 owner；绑定 colorbar 的 layer 里再写 `style.cmap` 就是
`JP-OWN-002`（见 §3.4）。不绑 colorbar 的 layer 可以自由用 `style.cmap`（因为没有第二个 owner）。

### R6 — 别名归一化收进一个 normalizer

`rectcmap` vs 文件名 `rect_cmap`、`lim` vs `limits`、`bin` vs `bins` —— 这些别名现在散落在 runtime 各处的
`.get(a, x.get(b))` 里。V2：**在 normalizer 里一次做完，之后 runtime 只见规范名**。
两个副作用都是好的：schema 只需描述规范名；诊断只需要报规范名。

### R7 — `type:` 封装层升格为主接口

现在 `posterior_2d` / `profile_2d` / `scatter_2d` / `posterior_1d` 已经**实现了**（`Figure/figure_types.py`），
但在文档里被定位成"纯粹的语法糖"，排在 2230 行文档的第十一~十三章。
一个从头读文档的 agent，会先学会那套 90 行的手写 layer 栈。

V2 反过来：

- **`type:` 是默认写法**（文档第二章就是它），手写 `layers:` 是 escape hatch（"当你需要 type 覆盖不到的组合时"）。
- `jplot template posterior_2d` 直接吐 type 化模板。
- 补齐缺失的 type：`hist_1d`、`line_2d`、`corner`、`ternary_2d`、`dynesty_diagnostics`。
- 对每个 type 提供 `--explain`：`jplot explain posterior_2d --expand config.yaml` 打印展开后的低层 YAML，
  让 agent（和人）能看懂糖背后是什么，需要时平滑降级到手写。

对照数据：`YAML_DESIGN.md` §11.9 自己给的例子是 **92 行 → 11 行**。11 行写对的概率比 92 行高一个量级。

---

## 5. CLI 动词化 + agent 自省面

HEP2 是动词式 CLI（`run` / `check` / `validate` / `monitor` / `plot` / `project …` / `convert` / `ps` / `kill`）。
PLOT 现在是 `jplot <file>` + 一个 `jplot flowchart` 特例 + 5 个 flag。

向 HEP2 对齐时有一条 **PLOT 特有约束（DR-08）**：`Jarvis2 plot ≡ jplot` 全 argv 透传，
因此 **渲染不能做成 `run` 动词**——否则 `Jarvis2 plot run scene.yaml` 会和
`Jarvis2 run task.yaml`（跑 scan）语义打架。渲染保持裸路径；动词只承载非渲染意图。

| 形式 | 作用 | 状态 |
|---|---|---|
| `jplot <yaml>` / `Jarvis2 plot <yaml>` | **渲染**（默认动作，不是动词） | 已有；**规范写法** |
| `jplot validate <yaml> [--json]` | **不渲染**、不建 matplotlib figure，全量收集诊断 | ⭐ 已实现（M0） |
| `jplot data describe <data> [--json]` | 数据集摘要：列/dtype/行数/范围/HDF5 树 | 半有（`data_loader_summary.py`） |
| `jplot cap … [--json]` | methods / transforms / style cards / colormaps / figure types 全清单 | 采集器有，动词未接 |
| `jplot template [<type>]` | 吐一份可跑的 type 化模板 + slot schema | 规格已有，frozen |
| `jplot explain <yaml> [--expand]` | 展开 `type:` 糖，打印等价低层 YAML | 新增 |
| `jplot flowchart <scene>` | 现有 | 已有 |
| ~~`jplot run <yaml>`~~ | — | **禁止**（见 DR-08） |

**关键要求：`validate` 必须能在完全不 import matplotlib、不渲染的前提下跑完整个配置解析 + 列需求规划。**
今天所有错误都在渲染期暴露，一个 10 图 YAML 要 10 轮才能收敛；validate 应该一轮报全。

---

## 6. 不建议从 HEP2 抄的

- **ASCII gate（`JV2-ENC-001`）**。HEP2 禁非 ASCII 是因为 `Scan.name` 会变成目录名、tar 成员、HDF5 属性。
  PLOT 的 label 是给 matplotlib 的，**中文/希腊字母标签是合法需求**。
  只应该对"会变成文件名的字段"（`project.name`、`Figures[].name`、`output.dir`）做路径安全性检查，
  不要全局禁非 ASCII。
- **HEP2 的 `open` zone + 警告**。HEP2 有历史包袱（RLTPMCMC 的 `Control`/`Reward`/`PPO`）。
  PLOT V2 既然允许 break，就应该只有 `closed` 和 `delegated` 两种 zone，不要一开始就给自己开后门。

---

## 7. 分期建议

| 阶段 | 内容 | 破坏兼容？ | 收益 |
|---|---|---|---|
| **P0（可以立刻做，1.5.x）** | R1 闭合根/figure/layer 词汇表 + did-you-mean + 列存在性校验（§3.3a）+ `jplot validate` | ❌ 否 | **消灭 case A/B/D**，即"静默接受"和"错位报错"两大类归零 |
| **P1** | schema catalog（manifest + 分文件） + `JP-*` 错误码 + suggestion/example + `capabilities` / `describe` 动词 | ❌ 否 | Agent 可自省、错误可自纠；method 坐标合约进入机器可查范围 |
| **P2（V2 break）** | R2 transform 判别式 + R3 coordinates 分裂 + R4 One Obvious Way + R5 颜色所有权 + R6 归一化 | ✅ 是 | 消灭剩下的结构性歧义 |
| **P3** | R7 `type:` 升为主接口 + 补齐 figure types + `explain` + 文档重排 | ⚠️ 文档层 | 把 agent 的默认路径从 90 行改成 10 行 |

P0 是**不需要等 V2、不需要等任何决策**就能做的，而且它单独解决了实测里最严重的两类问题。

---

## 8. 待拍板的三个问题

1. **`type:` 到底是糖还是主接口？** 这决定 P3 的文档重排规模，也决定要不要为 figure type 单独设计
   稳定 schema（如果是主接口，它就成了兼容性承诺的对象，不能再随便改）。
2. **R3 用哪个方案？** 我倾向 (a) `columns:` / `coordinates:` 分裂——形状不同才是最强的防混淆手段，
   但它会改掉所有现存 YAML 的 layer 块。若嫌太狠，(b) 报错方案零迁移成本，只是防不住"agent 觉得写了就该生效"。
3. **schema 是手写还是从 registry 生成？** `method_registry.METHOD_DISPATCH` 已经是权威名单。
   生成派：不会漂移，但坐标合约和 example 还是得手写元数据。手写派：像 HEP2 那样 schema 即文档，
   但要靠 CI 检查两边一致。我倾向"手写 schema + CI 一致性检查"，和 HEP2 保持同构，维护心智一致。
