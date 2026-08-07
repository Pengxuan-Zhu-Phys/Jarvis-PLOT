# Agent Output via YAML（设计）

Status: partial  
Date: 2026-08-06  
Implementation: A1–A2 landed including **schema wiring** (`figure.agent_output` + `output.agent_output` in JSON Schema so `config set` / validate accept the key). A3+ polish remaining.  
Audience: Jarvis-PLOT maintainers + coding-agent workflow  
Related: DR-08 bare-path render; CLI = knowledge; YAML = execution  

---

## 0. 一句话

> **CLI 只当知识库；一切可复现的数据处理与 agent 摘要输出，必须写进 YAML，由 `jplot <yaml>`（及 `doctor` 的计划视图）执行。**  
> Agent 压缩点云（默认 Voronoi cell 摘要，**有损、可解释、带 provenance**）是 figure 级声明式导出，不是新 CLI 动词。

这取代「再堆 `data digest` / 强化 `context`」的方向。

---

## 1. 产品原则（已拍板）

| 层 | 职责 | 不做 |
|---|---|---|
| **CLI** | 查合法值、手册、校验、计划 | 不承载 x/y/weight/bin 等数据处理参数面 |
| **YAML** | 声明数据、图、transform、输出 | 唯一可复现任务描述 |
| **`jplot <file>`** | 执行管线；可选写 PNG/PDF **与** agent JSON | — |
| **`jplot doctor`** | 结构/列/ledger + **planned exports** | **不**生成 agent JSON 文件 |

工作流：

```text
jplot data / cap / man / explain / template / suggest   # 知识与起草
→ 编辑 plot.yaml（含 agent_output）
→ jplot doctor plot.yaml --json                       # 计划 + 门禁
→ jplot plot.yaml                                     # 真跑：图 + agent JSON
```

**强制真画图不是默认。**  
默认：agent 摘要与图共用**同一数据管线**，但 `agent_output` 可在 render 路径生成；`doctor` 只计划。  
若 agent 无多模态，只消费 JSON digest 即可；需要像素时再看 PNG。

---

## 2. 非目标

- 不新增 `jplot data digest` / `jplot agent-digest` 等带满参数的 CLI。  
- **`jplot context` 已删除**。数据形状用 `data describe`；词汇用 `cap`/`man`；点云摘要用 YAML `agent_output` + `jplot <file>`。  
- 默认 **不**输出完整 Voronoi polygon 顶点（JSON 会再次膨胀）。  
- Digest **不能**替代 `validate` / `doctor` / 最终 render 的科学正确性判断。  
- 不在 doctor 中跑重型 density/profile 只为写 digest（与现有 heavy-skip / partial 语义一致）。

---

## 3. YAML 形状

### 3.1 Figure 级（推荐）

```yaml
Figures:
  - name: profile
    type: profile_2d          # 或手写 layers
    data: samples
    x: {expr: m_A, lim: [0.1, 5000], scale: log}
    y: {expr: tanb, lim: [1, 60]}
    z: {expr: LogL}
    style: [a4paper_2x1, rectcmap]

    agent_output:             # 可省略 = 不写 agent JSON
      enable: true            # 默认 true if block present
      format: json            # v1 仅 json
      method: voronoi         # 压缩实现；用户不必懂细节
      path: auto              # auto | 相对/绝对路径
      # 预算：不是保证规则网格
      max_cells: 1024         # 优先
      # 兼容别名（二选一或同时出现时 max_cells 优先）:
      # xbin: 32
      # ybin: 32              # → max_cells = xbin * ybin
      seed: 0
      geometry: none          # none | polygon（v1 可只实现 none）
      weight: auto            # auto | 表达式；auto 时从 type 宏 / 层推断
      include:                # 可选全局摘要开关
        - quantiles
        - top_cells
        - tails
        - categorical
```

### 3.2 根级默认（可选，figure 覆盖）

```yaml
output:
  dir: ./plots
  formats: [png, pdf]
  agent_output:                 # 应用到所有 enable 的 figure
    format: json
    method: voronoi
    max_cells: 1024
    path: auto
```

合并规则：`Figures[].agent_output` 深合并覆盖 `output.agent_output`；`enable: false` 关闭该 figure。

### 3.3 路径解析

| `path` | 行为 |
|---|---|
| `auto` / 省略 | `{output.dir}/{figure.name}.agent.json` |
| 相对路径 | 相对 `output.dir`（若仅文件名）或 YAML/workdir（若含 `/`）— **实现时写死一种并测** |
| 绝对路径 | 原样 |

与图文件并列示例：

```text
plots/profile.png
plots/profile.pdf
plots/profile.agent.json
```

### 3.4 字段字典（v1）

| 键 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enable` | bool | true if block | 总开关 |
| `format` | enum | `json` | v1 仅 json |
| `method` | enum | `voronoi` | 压缩算法 id |
| `path` | string | `auto` | 输出路径 |
| `max_cells` | int | 1024 | **上限**，非保证网格点数 |
| `xbin`/`ybin` | int | — | 若无 max_cells：`max_cells = xbin*ybin` |
| `seed` | int | 0 | 可复现 |
| `geometry` | enum | `none` | `polygon` 为 v1.1 |
| `weight` | string/object | auto | 样本权重 |
| `include` | list | 见下 | 全局块开关 |

`include` 默认：`[quantiles, top_cells, tails, nan_stats, provenance]`。

---

## 4. Agent JSON 载荷（有损 digest）

### 4.1 顶层

```json
{
  "schema_version": 1,
  "kind": "agent_digest",
  "figure": "profile",
  "lossy": true,
  "algorithm": {
    "method": "voronoi",
    "version": "1.0.0",
    "seed": 0,
    "max_cells": 1024,
    "actual_cells": 987
  },
  "provenance": {
    "source_rows": 300000,
    "finite_rows": 299812,
    "nan_rows": 120,
    "inf_rows": 0,
    "source_hash": "sha256:…",
    "yaml_path": "/abs/plot.yaml",
    "figure_fingerprint": "…",
    "jarvisplot_version": "1.4.x",
    "created_at": "ISO-8601"
  },
  "axes": {
    "x": {"expr": "m_A", "scale": "log", "lim": [0.1, 5000]},
    "y": {"expr": "tanb", "scale": "linear", "lim": [1, 60]},
    "z": {"expr": "LogL"},
    "weight": {"expr": "exp(LogL)"}
  },
  "global": {
    "quantiles": {
      "x": {"q05": …, "q50": …, "q95": …},
      "y": {…},
      "z": {…},
      "weight": {…}
    },
    "weighted_centroid": […, …],
    "ess": 12345.6,
    "categorical": {
      "group": [{"value": "A", "count": 1000}, …]
    }
  },
  "highlights": {
    "top_density": [{"cell_index": 12, "density": 42.7}, …],
    "top_mass": […],
    "tails": [{"cell_index": 3, "flags": ["tail"]}, …]
  },
  "cells": [ /* 见下 */ ]
}
```

### 4.2 Cell 记录（默认，无 polygon）

```json
{
  "i": 0,
  "site": [1.2, 3.4],
  "bbox": [1.0, 1.5, 3.0, 3.8],
  "count": 184,
  "mass": 0.023,
  "weight_mean": 0.71,
  "z_mean": -12.3,
  "z_min": -15.1,
  "z_max": -9.8,
  "density": 42.7,
  "flags": ["tail"]
}
```

| 字段 | 含义 |
|---|---|
| `site` | 生成元 / 代表点（数据空间，已考虑 log 轴时用**投影前**的物理坐标写文档，实现二选一并写清） |
| `bbox` | `[xmin, xmax, ymin, ymax]` 轴对齐包围盒（**不**输出 polygon 顶点） |
| `count` | 落入该 cell 的有限样本数 |
| `mass` | 权重和 / 总权重（若无 weight 则 count/N） |
| `density` | 有定义时：mass / area（area 用 bbox 或 mesh 面积，需在 algorithm 注明） |
| `flags` | `empty` / `tail` / `outlier` / `nan_adjacent` … |

`geometry: polygon` 时另加 `vertices: [[x,y],…]`（可选，默认关）。

### 4.3 关键不变量

1. **`lossy: true` 恒为 true**（除非 future 全量 dump 模式）。  
2. **`actual_cells ≤ max_cells`**。  
3. **`sum(mass) ≈ 1`**（有 weight 时）或与有限样本一致。  
4. **同一 seed + 同 fingerprint → 同 cells 摘要**（测试锁）。  
5. Digest 使用的 x/y/z/weight **与 figure 声明同源**（type 宏展开后或第一数据层），禁止 CLI 旁路。

---

## 5. `doctor` 行为

不写文件。在 envelope `data` 中增加：

```json
{
  "exports": [
    {
      "figure": "profile",
      "format": "json",
      "method": "voronoi",
      "max_cells": 1024,
      "path": "/abs/plots/profile.agent.json",
      "status": "planned",
      "notes": []
    }
  ]
}
```

校验（失败 → diagnostic，可 error 或 warning）：

| 条件 | 码（建议） | 级别 |
|---|---|---|
| `max_cells < 4` 或过大（如 > 50_000） | JP-AGT-001 | error / warning |
| `method` 未知 | JP-AGT-002 | error |
| `path` 父目录不可写（能探测时） | JP-AGT-003 | warning |
| figure 无可用 2D 坐标 | JP-AGT-004 | error |
| heavy 未展开/未跑 | 既有 partial | info：digest **仅 render 生成** |

`status` 枚举：`planned` | `skipped` | `invalid`。

---

## 6. Render 行为（`jplot <yaml>`）

在 figure 数据就绪后、或与 density/profile 同源 mesh 构建阶段：

1. 解析 `agent_output`（合并根默认）。  
2. 取该 figure 的点列 `(x,y[,z,w])`（有限行）。  
3. 按 `method` + `max_cells` + `seed` 压缩。  
4. 写 JSON（原子写：temp + rename）。  
5. 日志：`Agent digest written -> path (cells=N/max=M)`。  
6. 失败：记 error，**默认不拖垮整图**（可配置 `agent_output.strict: true` 则 render exit 1）。

实现优先复用：

- Bridson / density mesh / Voronoi 现有 runtime（`posterior_mesh`、profile bridson、density_cell）。  
- **禁止**再实现第二套网格算法当默认。

---

## 7. CLI 面（保持小）

| 动词 | 变化 |
|---|---|
| `man` / `cap` / `data` / `explain` / `template` | 知识库；文档说明 `agent_output` |
| `doctor` | + `exports[]` planned |
| `jplot <file>` | 执行写 digest |
| **`context` CLI** | **已删除**（2026-08） |
| 不新增 | `data digest` |

`jplot man agent-output`（短卡）：如何写 `agent_output`、JSON 字段、与 doctor/render 分工。

---

## 8. 与「强制出图」的关系

| 模式 | 何时 |
|---|---|
| 只 JSON digest | figure `agent_output` + render；PNG 仍按 `output.formats` 正常出（可 `formats: []` 若未来支持只导出 agent——**v1 不要求**） |
| 图 + JSON | 默认：有 formats 就出图，有 agent_output 就出 JSON |
| 只检查计划 | `doctor` |

不把「必须看图」设为 agent 默认；JSON 是有损 **reference**，图是人类/多模态附件。

---

## 9. 实现分期

| 阶段 | 内容 | 规模 |
|---|---|---|
| **A0** | 本设计入库；workflow/man 去推 `context`；`agent_output` schema 草案 | S |
| **A1** | 解析 + doctor `exports` planned + 校验码 | S–M |
| **A2** | Render 写 v1 digest（voronoi/bridson 摘要，无 polygon） | M |
| **A3** | quantiles / top_cells / tails / ESS / categorical / 稳定 seed 测试 | M |
| **A4** | 可选 `geometry: polygon`；`strict`；根级 defaults 精修 | S |
| **A5** | ~~废弃 context~~ **done：命令已删除** | — |

建议 **A1+A2** 为一版可交付；A3 锁科学摘要质量。

---

## 10. 测试验收

1. 无 `agent_output`：行为与今日一致，无 JSON。  
2. 有 `agent_output`：`doctor --json` 含 `exports[0].status=planned` 与解析后 path。  
3. `jplot plot.yaml`：产生 `.agent.json`；`actual_cells ≤ max_cells`；`lossy=true`。  
4. 改 `seed` 或数据 → fingerprint/cells 变化（或稳定时一致）。  
5. 缺列 figure：doctor invalid export 或 render 跳过 digest 且有码。  
6. 同一 YAML 两次 render：digest 字节级或字段级稳定（浮点容差可配置）。  
7. **无** `jplot data digest` 动词。

---

## 11. 开放默认（实现时写死）

| # | 问题 | 默认 |
|---|---|---|
| Q1 | log 轴 site 坐标 | **物理坐标**（反投影后），lim/scale 写在 `axes` |
| Q2 | density 面积 | 优先 mesh cell area；否则 bbox 面积并 `flags: approx_area` |
| Q3 | type 宏 vs layers 取点 | 展开后主数据层 / density 输入点（与 type 同源） |
| Q4 | digest 失败是否 fail render | 默认否；`strict: true` 才是 |
| Q5 | `context` 去留 | **删除**（done） |

---

## 12. 示例：agent 最小闭环

```yaml
# plot.yaml
DataSet:
  - {name: samples, path: ./samples.csv, type: csv}
Figures:
  - name: posterior
    type: posterior_2d
    data: samples
    x: {expr: m_A}
    y: {expr: tanb}
    weight: {expr: exp(LogL)}
    style: [a4paper_2x1, rectcmap]
    agent_output:
      method: voronoi
      max_cells: 1024
      path: auto
output:
  dir: ./plots
  formats: [png]
```

```bash
jplot man agent-output --json          # 知识
jplot doctor plot.yaml --json          # exports: planned
jplot plot.yaml                        # png + posterior.agent.json
```

---

## 13. 总结

| 要 | 不要 |
|---|---|
| YAML `agent_output` 声明 digest | 复杂 `data digest` CLI |
| doctor 只计划 exports | doctor 写大 JSON |
| render 写有损、带 provenance 的 cell 摘要 | 默认 polygon 顶点 dump |
| CLI = 知识库 | CLI = 第二套数据处理面 |
| digest 辅助 agent | digest 替代 validate/render |

**一句话：**  
Agent 想要的「压缩数据形状」是 **figure 的导出产物**，与 PNG 并列，由 YAML 声明、`jplot <yaml>` 执行；CLI 继续教人怎么写，不负责再做一遍管线。
