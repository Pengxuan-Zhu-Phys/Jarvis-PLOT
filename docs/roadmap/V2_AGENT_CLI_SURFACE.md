# Jarvis-PLOT V2 — Agent CLI 面 Brainstorm

Status: brainstorm
Date: 2026-08-06
姊妹文档: `docs/roadmap/V2_YAML_AGENT_ERGONOMICS.md`（YAML 结构侧）
既有契约: `docs/specs/AGENT_DATA_API.md`（6 动词，frozen）、`Jarvis-Agent/Docs/PLOT_TOOLS.md`（5 工具，M4.6）

---

## 0. 目标一句话

> **用户说人话 → Agent 拿到能跑的 YAML；用户再说人话 → Agent 精确改那份 YAML。**
> 全程 agent 不需要读 PLOT 的文档，也不允许臆造任何一个字符串。

要做到这点，PLOT 必须把自己**完全自描述化**：数据长什么样、我会画什么、你写的对不对、
画出来是什么样 —— 四件事全部有 CLI 出口，且全部 `--json`。

---

## 1. 组织原则：四个闭环 + 一条横切

现有的 6 动词规格（AGENT_DATA_API）覆盖了其中约 40%。缺口集中在**能力自省**、**渲染读回**、
**增量修改**三处 —— 而后两者恰恰是"改 YAML"和"确认画对了"的必经之路。

```
                    ┌──────────────── 用户自然语言 ────────────────┐
                    ↓                                              ↓
  ① 看得见数据                                          ④ 画完能读回
  jplot data describe/head/eval/suggest-axes    ←──┐    jplot <yaml> --report
  「列名的唯一合法来源」                            │    「图健康体检 + 数值孪生」
                    ↓                              │                ↑
  ② 看得见能力                                     │                │
  jplot cap methods/types/styles/cmaps/funcs       │                │
  「字符串的唯一合法来源」                          │                │
                    ↓                              │                │
              起草 / 修改 YAML  ──────────────────→ ③ 写完能验 ─────┘
              jplot suggest                        jplot validate / dryrun
              jplot config set/add/rm              「诊断可机器应用 --fix」
                    ↑                                    │
                    └──────── 横切：稳定寻址 + 注释保留 + 写-验-回滚 ────┘
```

**两条反幻觉纪律**（承 `PLOT_TOOLS.md` §3.1 已确立的原则，扩大适用范围）：

1. **列名只能来自 `jplot data describe`。**（已有纪律）
2. **PLOT 词汇表里的任何字符串**（method / figure type / style card / cmap / transform / 轴名 /
   表达式函数）**只能来自 `jplot cap`。** ← 这条现在没有出口，所以 agent 只能靠记忆赌
   `jarvis_rainbow2_r`、`a4paper_2x1` 、`axc` 这些名字。

---

## 2. 动词树

### 2.0 固定契约：渲染不是动词（DR-08）

`Jarvis2 plot` **完整透传** `jplot` 的 argv（`Jarvis2 plot ≡ jplot`）。因此 PLOT 的
渲染入口**不能**叫 `run`：

| 写法 | 是否允许 | 原因 |
|---|---|---|
| `jplot scene.yaml` | **规范** | 裸路径 = 渲染 |
| `Jarvis2 plot scene.yaml` | **规范** | 透传后同上 |
| `jplot validate scene.yaml` | **规范** | 非渲染动词，语义清晰 |
| `Jarvis2 plot validate scene.yaml` | **规范** | 透传后同上 |
| `jplot run scene.yaml` | **禁止** | 会变成 `Jarvis2 plot run …`，`run` 在 HEP 侧是「跑 scan」 |
| `Jarvis2 plot run scene.yaml` | **禁止** | 同上；与 `Jarvis2 run task.yaml` 肌肉记忆冲突 |

**规则一句话：动词只承载非渲染意图；渲染永远是「给文件」。**
渲染附加行为用 **flag**（`--report` / `--thumb-ascii` / `--rebuild-cache`），
不发明 `jplot run` 子命令。

### 2.1 命令树

```
# 渲染（默认动作 —— 不是动词）
jplot <yaml>                        渲染图
jplot <yaml> --report               渲染 + 体检报告（规划中）
jplot <yaml> --rebuild-cache        渲染前重建 cache

# 非渲染动词（可出现在 Jarvis2 plot <verb> … 下，语义仍清晰）
jplot validate <yaml>               形状/schema/列（检查阶段，不渲染）
jplot dryrun <yaml>                 读数据 + 轻 transform 账本；跳过 profile/density/interp
jplot doctor <yaml>                 validate + dryrun 合一；不重跑 heavy（执行用 jplot <file>）
jplot explain <code|type|yaml>      错误码知识库 / type 糖展开

jplot data  describe <file>         列/dtype/行数/分位数/HDF5 树
            head <file>             前 N 行真实样本
            eval "<expr>"           表达式沙盒
            suggest-axes <file>     每列的 scale/lim 建议

jplot cap   all                     一次性能力清单（带 digest hash）
            methods | transforms | types | styles | cmaps | funcs | cli

jplot suggest --data … --kind …     数据感知的配置合成（NL→YAML 的第一跳）
jplot template list | show <kind>   模板 + slot schema

jplot config get  <yaml> <addr>     结构化读
            set  <yaml> <addr> <v>  结构化写（保留注释）
            add-layer / rm / move
            diff <a> <b> --semantic
            fmt  <yaml>             规范化 + 别名归一

jplot examples list | show <name>   CI 验证过的真实 YAML 语料
jplot flowchart <scene>             现有
```

**全局约定**：所有**动词**接受 `--json`；stdout 只有一个 JSON 对象，人类日志走 stderr；
统一 envelope `{api_version, kind, ok, data, diagnostics, error}`（沿用 AGENT_DATA_API §2）。
所有 introspection 动词**纯读、可缓存、可并发**——agent 会疯狂调用它们。
渲染路径的机器可读出口用 flag（例如规划中的 `--report --json`），**kind 写 `render` 或
具体 report 名，绝不叫 `run`**。

---

## 3. 闭环 ① 看得见数据

### 3.1 `jplot data describe` —— 把已有的统计表做成一等公民

现有 `data_loader_summary.dataframe_summary_rows()` 已经产出 `(name, dtype, nonnull%, min, max)`。
只需要：接 `--json`、加分位数、加 HDF5 树、**按 `ProjectCache.source_fingerprint()` 缓存**
（10GB HDF5 不该被反复扫）。

```json
{"kind": "data.describe", "ok": true, "data": {
  "path": "outputs/scan/DATABASE/samples.hdf5", "type": "hdf5",
  "rows": 204132, "groups": ["Parameters", "Loglikelihood"],
  "columns": [
    {"name": "Parameters/m_A", "dtype": "float64", "nonnull": 1.0,
     "min": 91.3, "max": 1987.4, "q": {"01": 120.2, "25": 380.1, "50": 742.0, "75": 1310.4, "99": 1904.8},
     "n_unique": 204132, "positive": true, "decades": 1.34,
     "role_hint": "parameter"},
    {"name": "LogL", "dtype": "float64", "min": -312.1, "max": -12.0,
     "role_hint": "log_likelihood"}
  ],
  "tree": "…ASCII HDF5 tree…"
}}
```

**新增的关键字段是 `role_hint`**：按列名与数值特征识别 `log_likelihood` / `weight` / `chi2` /
`parameter` / `flag`。Agent 拿到"这列是 log-likelihood"，就知道权重该写 `exp(LogL)` 而不是 `LogL`，
不需要猜。这是 PLOT 有信息优势、agent 没有的地方。

### 3.2 `jplot data head` —— 5 行真实样本抵得上一页统计

```bash
jplot data head samples.hdf5 --n 5 --cols "m_A,tanb,LogL" --json
```

模型对具体数值的推理能力远强于对统计量的推理。看到 `m_A = 91.3, 88.7, 1204.5` 它立刻知道量级；
看到统计表 `min 91.3 max 1987.4` 它还得算。**成本几乎为零，收益很大。**

### 3.3 `jplot data eval` —— 表达式沙盒（消灭一整类错误）

```bash
jplot data eval "exp(LogL)" --data samples.hdf5 --json
jplot data eval "np.log10(m_A)" --data samples.hdf5 --json
```

```json
{"ok": true, "data": {"expr": "exp(LogL)", "dtype": "float64",
  "n": 204132, "n_finite": 204132, "n_nan": 0, "n_nonpositive": 0,
  "min": 1.1e-136, "max": 6.1e-6, "sample": [3.2e-9, 1.1e-8, …],
  "symbols_used": ["LogL"], "symbols_unresolved": []}}
```

失败时直接给出可用列表：

```json
{"ok": false, "error": {"code": "JP-EXPR-002",
  "message": "Symbol 'aa' is not a column of this dataset and not a known function.",
  "available_columns": ["a","b","LogL"], "did_you_mean": ["a"],
  "available_functions": ["exp","log","sqrt","Gauss","Normal","LogGauss","Heaviside","np.*"]}}
```

Agent 在把表达式写进 YAML **之前**就能验证它。这一条同时解决了实测里的 case D 和
"表达式函数名猜错"（`ln` 还是 `log`？`Max` 还是 `max`？—— 现在只能翻 YAML_DESIGN §8）。

### 3.4 `jplot data suggest-axes` —— 让 PLOT 算 agent 猜不准的东西

```json
{"data": {"axes": [
  {"col": "m_A", "scale": "log", "lim": [88, 2000],
   "reason": "all positive; spans 1.34 decades; lim from q0.5–q99.5 rounded outward"},
  {"col": "tanb", "scale": "linear", "lim": [0, 60],
   "reason": "spans 0.4 decades; lim from full range rounded to nice numbers"}
]}}
```

`lim` / `scale` 是 agent 最常写错的两个字段（写错了图不报错，只是难看或空白）。
**这类决策应该由看得见数据的一方做，而不是由猜的一方做。**

---

## 4. 闭环 ② 看得见能力

### 4.1 `jplot cap all --json` —— 字符串的唯一合法来源

一次性吐出 PLOT 的全部词汇表。带 `digest`（内容 hash），agent 缓存后凭 digest 判断是否需要重拉。

```json
{"kind": "cap.all", "data": {
  "digest": "sha1:9f3c…", "package_version": "2.0.0", "api_version": 1,
  "figure_types": [
    {"name": "posterior_2d", "required": ["data","x","y","weight"],
     "optional": ["density","colorbar","hpd","extra_layers","style_card"],
     "produces_layers": ["pcolormesh","contour"], "summary": "二维后验概率密度热力图 + HPD 等高线"}
  ],
  "methods": [
    {"name": "pcolormesh", "coordinates": {"required": ["x","y","z"]},
     "axes_types": ["rect"], "supports_colorbar": true,
     "style_keys_common": ["edgecolor","linewidth","shading"]},
    {"name": "scatter", "coordinates": {"required": ["x","y"], "optional": ["c"]},
     "axes_types": ["rect","tri"], "supports_colorbar": true,
     "style_keys_common": ["marker","s","alpha","color","cmap","zorder"]}
  ],
  "transforms": [
    {"name": "filter", "params": {"expr": {"type": "string", "required": true}}},
    {"name": "posterior_density", "params": {"method": {"enum": ["voronoi","adaptive","kde","grid"]},
      "bins": {"type": "integer", "default": 64}, "grid": {"type": "integer", "default": 256}},
     "outputs": ["x","y","density"]}
  ],
  "style_cards": [
    {"tokens": ["a4paper_2x1","rectcmap"], "axes": ["ax","axc"],
     "figsize_cm": [8.6, 6.4], "has_colorbar": true, "family": "a4paper", "variant": "rect_cmap"}
  ],
  "colormaps": ["viridis","jarvis_rainbow2_r","…"],
  "expression_functions": ["exp","log","ln","sqrt","Gauss","Normal","LogGauss","Heaviside","…"],
  "cli": { /* 从 cards/args.json 生成 —— CLI 自描述 */ }
}}
```

几个要点：

- **`methods[].coordinates`**：现在这套契约只活在 `YAML_DESIGN.md` 的表格里，运行时不检查、
  agent 也读不到。`Figure/method_registry.METHOD_DISPATCH` 已经是唯一权威名单，
  只差把坐标合约挂上去。
- **`style_cards[].axes`**：**这是当前最隐蔽的一个坑**。`axes: axc` 里的 `axc` 从哪来？
  只能从 style card JSON 里来，agent 无从知道。把每张卡的轴名列出来，`axes:` 就不用猜了。
- **`cli` 段从 `cards/args.json` 生成** —— PLOT 的 CLI spec 本来就是数据文件，天然自描述。
  顺手可以再吐一个 `jplot cap mcp --json` 直接生成 MCP tool schema，
  **Jarvis-Agent 那 5 个 tool 的 schema 就不用手写了，单一真相源。**

### 4.2 `jplot cap styles --preview` —— 让布局可见

`Figures[].debug: true` 已经实现了 design-reference overlay（画出每个 axes 的 rect 和 cm 尺寸）。
把它变成动词：

```bash
jplot cap styles a4paper_2x1:rectcmap --preview /tmp/card.png --json
```

JSON 给 agent（轴名、几何、是否有 colorbar），PNG 给人。**同一份信息，两个受众。**

### 4.3 `jplot examples` —— few-shot 比 schema 更有效

```bash
jplot examples list --json          # 目录：名字 / 用到的 type / method / 一句话
jplot examples show posterior_2d_basic --json
```

模型学 YAML 靠 few-shot 的效率远高于读 schema。硬性要求：
**每个 example 都在 CI 里真跑一遍**，agent 永远拿不到过期示例。
（对照 HEP2 的 `Jarvis2 project list/fetch` —— 同一个思路，PLOT 缺这块。）

---

## 5. 闭环 ③ 写完能验

### 5.1 三档成本

| 动词 | I/O | matplotlib | 用途 |
|---|---|---|---|
| `validate` | 无（列名对照缓存的 describe） | 不 import | 每次编辑后都跑，秒级 |
| `dryrun` | 读数据、只跑轻 transform（跳过 heavy） | 不 import | 确认列/filter 等不把表滤空；mesh 只在 render |
| `run` | 全部 | 是 | 真出图 |

**Agent 应该挑能回答问题的最便宜那档。** 今天只有第三档，所以一个 10 图的 YAML 要跑 10 轮才收敛。

### 5.2 诊断要可机器应用 —— `--fix`

HEP2 的 `suggestion` + `example` 是给人看的文本。再往前一步：**结构化修复**。

```json
{"level": "error", "code": "JP-KEY-001", "path": "$.Figures[0].styles",
 "message": "Unknown key 'styles' in figure object.",
 "suggestion": "Rename 'styles' to 'style'.",
 "fix": {"op": "rename_key", "path": "$.Figures[0]", "from": "styles", "to": "style"},
 "confidence": "certain"}
```

```bash
jplot validate config.yaml --fix --diff      # 打印 diff，不落盘
jplot validate config.yaml --fix --write     # 只应用 confidence=certain 的
```

拼写、重命名、别名归一、字段迁移 —— 这几类是 100% 机械的，agent 不该花 token 去改。
这是 ruff `--fix` 的模式。剩下的（`confidence: heuristic`）留给 agent 决策。

### 5.3 行数账本 —— 空图的头号成因

`dryrun` 输出每个 transform 步骤的进出行数：

```text
DataSet d1                                204,132 rows × 46 cols
Figure EggBox_posterior / layer _density
  ← source d1                             204,132
  → filter  "LogL > -100"                 204,132 →       0   ⚠ JP-VIZ-003
  → posterior_density (voronoi, bins=120)        0 →       0
```

Agent 一眼看到"filter 把数据滤光了"，而不是拿到一张空白 PNG 和 exit 0。

---

## 6. 闭环 ④ 画完能读回 —— 最深的一环

> Agent 永远读不了渲染出的图（`PLOT_TOOLS.md` 开篇即言）。
> 现有规格给了 `--with-data`（数值孪生 sidecar），但**数值 ≠ 判断**：
> agent 拿到一张表，仍然不知道"这图画对了没有"。

### 6.1 渲染体检 `JP-VIZ-*`（执行期，不是 doctor 重跑）

**纪律**：检查阶段（`validate` / `dryrun` / `doctor`）**禁止**为体检再跑 profile /
density / interp。全套 post-mesh JP-VIZ 只发生在 **`jplot <yaml>` 执行路径**
（或规划中的 `--report`，仍是同一次 render 内采集）。

`doctor` / `dryrun` 只做轻步骤账本 + 可负担的代理（例如重步骤跳过时的
`JP-VIZ-002` pre-transform lim）。`type:` 上 `coverage: partial` **是设计结果**。

| 码 | 检查 | 为什么 agent 看不见 |
|---|---|---|
| `JP-VIZ-001` | 图上零个可见元素 | 空图和正常图的 exit code 一样 |
| `JP-VIZ-002` | 94% 的点落在 xlim/ylim 外被裁掉 | 图会显示，但几乎是空的 |
| `JP-VIZ-003` | 某 transform 后 0 行 | 见 §5.3 |
| `JP-VIZ-004` | colorbar `vmax=0.8` 但数据 max=12.3 → 97% 面积饱和成同一色 | 图看起来"有东西"，其实全是一个颜色 |
| `JP-VIZ-005` | `xscale: log` 但 x 有 1,204 个非正值（被 matplotlib 静默丢弃） | 完全无提示 |
| `JP-VIZ-006` | layer `_hpd` (zorder 1) 被 layer `_density` (zorder 10) 完全遮挡 | 画了等于没画 |
| `JP-VIZ-007` | 插值网格 87% 是 NaN（凸包外） | 大片空白 |
| `JP-VIZ-008` | 所有数据点落在 <1% 的轴面积内（lim 量级错了） | 一个点 |
| `JP-VIZ-009` | 图例引用了不存在的 label / 没有任何 handle | |

**价值**：把「看不见图」变成可枚举码。**采集时机**必须是 render 已有的数组，
不是 doctor 再算一遍（重算 = 第二条流水线 = 必然漂移，且违背检查/执行分离）。

### 6.2 低保真视觉：ASCII 缩略图

```bash
jplot config.yaml --thumb-ascii 32x16
# 等价：Jarvis2 plot config.yaml --thumb-ascii 32x16
```

```text
figure EggBox_posterior / ax    density ∈ [0, 0.79]
  ░░░░▒▒▓▓██▓▓▒▒░░░░░░░░▒▒▓▓██▓▓▒▒
  ░░▒▒▓▓████▓▓▒▒░░░░░░▒▒▓▓████▓▓▒▒
  …
```

把 2D 场降采样成 32×16 字符图直接进上下文。**几十个 token，但让非多模态模型"看到"结构**：
几个峰、峰在哪、是不是全空、是不是全饱和。对 EggBox 这种多峰后验尤其有用。
（散点图同理可做二维直方 ASCII。）

### 6.3 数值孪生的所有权

`--with-data` 导出的必须是**送进 matplotlib 的那一份 prepared 数据**，不是重新算一遍。
这样"人看到的"和"agent 读到的"来自同一份数组 —— 图文不一致从机制上不可能。
（`PLOT_TOOLS.md` §3.5 已经这么要求，这里只是强调实现纪律。）

---

## 7. 横切：起草与修改 YAML

这是 `AGENT_DATA_API` 和 `PLOT_TOOLS` **都没有覆盖**的一块，但用户的需求里明确包含"修改 YAML"。

### 7.1 起草：`jplot suggest` —— 数据感知的配置合成

`template` 给的是空模板，槽还是 agent 自己填（包括它最不会填的 lim/scale/bins）。
`suggest` 把分工摆正：**agent 出意图结构，PLOT 出数值。**

```bash
jplot suggest --data samples.hdf5 --kind posterior_2d \
      --x "Parameters/m_A" --y "Parameters/tanb" --weight "exp(LogL)" --json
```

PLOT 自动决定并**解释每一个决定**：

```json
{"data": {
 "yaml": "…可直接跑的完整 YAML…",
 "decisions": [
   {"field": "x.scale", "value": "log", "reason": "m_A all positive, spans 1.34 decades"},
   {"field": "x.lim", "value": [88, 2000], "reason": "q0.5–q99.5 rounded outward"},
   {"field": "density.bins", "value": 120, "reason": "204k rows → ~14 samples per support cell"},
   {"field": "style_card", "value": ["a4paper_2x1","rectcmap"], "reason": "posterior_2d default (needs colorbar)"}
 ]}}
```

`decisions` 字段让 agent 能把理由复述给用户（"我用了对数坐标，因为 m_A 跨了 1.3 个数量级"），
而不是端出一份来历不明的 YAML。**这直接服务于"用户说人话 → 拿到 YAML"。**

### 7.2 修改：稳定寻址是前提

`$.Figures[0].layers[2]` 这种下标地址一旦列表顺序变了就全错。**V2 应该把"可寻址"当成
YAML 结构的硬需求**（这一条要回写到 YAML 侧文档）：

- `layers[]` 的 `name` 从可选变必填（或加载时自动分配稳定名 `L0`/`L1`）
- 地址语法按名字定位：`Figures[EggBox_posterior].layers[_density].style.cmap`
- 同名即冲突，validate 报错

```bash
jplot config get config.yaml 'Figures[EggBox].layers[_density].style' --json
jplot config set config.yaml 'Figures[EggBox].colorbar.vmax' 1.2 --diff
jplot config add-layer config.yaml --figure EggBox --from-template scatter --after _density
jplot config rm  config.yaml 'Figures[EggBox].layers[_hpd]'
```

### 7.3 三条写操作纪律

1. **保留注释与格式**（ruamel.yaml round-trip）。人写的 YAML 有注释，agent 改一个 `vmax`
   不应该把整个文件重排、注释全丢。这是 agent 编辑配置最常见的破坏性行为。
2. **写-验-回滚**：每次写操作内部先跑 `validate`，不通过就不落盘并返回诊断。
   **Agent 永远不可能在磁盘上留下一个坏 YAML。**
3. **`--diff` 默认**：先给 diff，`--write` 才落盘。可审计、可撤销。

### 7.4 `jplot config diff --semantic`

展开 `type:` 糖之后再比较，告诉 agent"这两份配置实际只差 `colorbar.vmax`"。
用于"改之前 vs 改之后到底动了什么"的自查，也用于把手写 layer 版本和 type 版本对齐。

---

## 8. 错误码即知识库：`jplot explain`

```bash
jplot explain JP-VIZ-004
jplot explain posterior_2d
jplot explain config.yaml --expand      # type 糖 → 等价低层 YAML
```

rustc `--explain` 的模式，对 agent 特别合适：**诊断行只带一行摘要 + 码，需要细节时再拉。**
省上下文，且保证解释和实现同源（解释文本就存在 schema 的 `x-jarvis-example` 旁边）。

`--expand` 那一支是 type 封装层的安全网：agent 需要 type 覆盖不到的效果时，
可以先展开成低层 YAML 再手工改，而不是从零手写 90 行。

---

## 9. 三条工程纪律

1. **一个引擎，两张皮。** Agent 通道必须复用同一条 loader / transform / cache 管线。
   一旦分叉，`--with-data` 的数值孪生保证立刻失效。（契约 §1 已写死，这里重申。）
2. **Token 预算。** 每个动词两档输出：默认 `digest`（≤1K token）+ artifact 路径；`--full` 才吐全量。
   `cap all` 这种大响应必须支持分段拉取和 digest 缓存。
3. **确定性 & 可缓存。** 所有 introspection 动词纯读、幂等、可并发，按
   `ProjectCache.source_fingerprint()` 缓存。Agent 一个会话里会调几十次 `describe`。

---

## 10. 落地优先级（按"每行代码解锁多少 agent 能力"排）

| 序 | 项 | 现有机器 | 解锁的能力 |
|---|---|---|---|
| 1 | JSON envelope + `validate` + `--version-json` | 规格已有（JP-A1），`jsonschema` 已在依赖 | 写完能验；一轮报全部错误 |
| 2 | `data describe --json`（+ role_hint + 缓存） | `data_loader_summary.py` 已有一半 | 列名反幻觉（Agent 侧纪律的前提） |
| 3 | `cap all --json` | `method_registry` / `cards/` / `cmaps` 都是现成数据 | **字符串反幻觉** —— 当前最大的臆造来源 |
| 4 | `data eval` | `utils/expression.py` 现成 | 表达式先验后写 |
| 5 | 渲染体检 `JP-VIZ-*` | 渲染期数据都在手上 | **agent 不用眼睛也能判断图对不对** |
| 6 | `suggest`（数据感知合成 + decisions） | 依赖 2 + type 展开器 | **NL → YAML 的第一跳** |
| 7 | `config get/set/add/rm`（round-trip + 写-验-回滚） | 需要引入 ruamel.yaml | **NL → 改 YAML** |
| 8 | `--fix` 结构化修复 | 依赖 1 的诊断结构 | 机械错误零 token 消耗 |
| 9 | `examples` + `explain` + ASCII 缩略图 | examples 是现成 YAML | 少花上下文、few-shot、低保真视觉 |

1–4 都是**纯加法、不破坏任何现有行为**，且都有现成机器可复用。
5–7 是这套方案里真正的新东西。

---

## 11. 待拍板

1. **`AGENT_DATA_API` 的 frozen 状态要不要解？** 这里提的一切都建立在它的 envelope 之上。
   建议至少解冻 JP-A1（envelope + validate），否则每一条都无处安放。
2. **`jplot config set` 要不要 PLOT 自己做？** 另一条路是 agent 直接用文本编辑工具改 YAML，
   PLOT 只提供 `validate`。取舍：PLOT 自己做能保证注释保留 + 写-验-回滚 + 稳定寻址，
   但要引入 ruamel.yaml 依赖并维护一套地址语法。我倾向 PLOT 做——
   **"agent 不可能留下坏 YAML"这个保证值这个成本。**
3. **`cap mcp` 生成 MCP tool schema 值不值得做？** 如果做，Jarvis-Agent 的 5 个 tool schema
   从手写变成生成，跨仓漂移问题消失；代价是 PLOT 要承认 MCP 这个下游形态。
4. **ASCII 缩略图（§6.2）是玩具还是真需求？** 我觉得对多峰后验这类"结构性"问题真的有用，
   但需要在 EggBox 上试一次才知道。低成本，值得先做个原型验证。
