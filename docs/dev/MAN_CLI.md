# `jplot man` — 结构化手册 CLI（开发文档）

Status: design / ready to implement  
Date: 2026-08-06  
Audience: Jarvis-PLOT maintainers  
Precedent: **Jarvis-Portal** `jportal man` / `jportal man <format>`  
Related: coding-agent workflow (data/cap/validate/dryrun); DR-08 bare-path render  

---

## 0. 一句话

> **`jplot man` 是给人和 coding agent 共用的「可调用说明书」。**  
> 默认终端输出是人类可读的 Rich 手册；`--json`（或等价 agent 旗标）给出结构化、可长文的机器载荷。  
> **写 YAML 不靠 man / 不靠 config set**——agent 用编辑器写文件；man 只回答「怎么写、写什么、错了看哪」。

这与当前产品定位一致：CLI = **传感器 + 裁判 + 说明书**，不是打字机。

---

## 1. 对照 Jarvis-Portal（已有事实）

### 1.1 Portal 做对了什么

| 点 | Portal 实现 | 移植到 PLOT |
|---|---|---|
| 动词形状 | `jportal man` / `jportal man <topic>` | `jplot man` / `jplot man <topic>` |
| 内容与渲染分离 | `manual_cards/<topic>.yaml` 纯数据；`cli.py` 只负责 Rich | 同样：`jarvisplot/manual_cards/*.yaml` + 独立渲染器 |
| 人类版式 | 固定宽 Rich Panel / Rule / Syntax，与全族 CLI help 同几何 | 复用 PLOT 已有 `cli.py` help geometry（Portal `CLI_HELP_STYLE` 同源） |
| 主题发现 | 无额外 `--formats`；`man` 索引即目录 | 禁止再发明第二套 list；索引页即 topic 表 |
| 边界纪律 | man 不承载业务执行；业务只 `jportal file` | man 不渲染、不改 YAML；执行仍是 `jplot <file>` / validate / dryrun |

### 1.2 Portal 没有、PLOT 需要补的

| 缺口 | 原因 |
|---|---|
| **双受众输出** | Portal man **只有人类 Rich**；coding agent 只能吞 TTY 样式文本。PLOT 必须提供 **干净的结构化 `--json`**。 |
| **主题不止「格式」** | Portal 主题 = 文件格式（json/csv/…）。PLOT 主题 = **工作流 + YAML 面 + 诊断 + 命令**。 |
| **与 live cap/data 交叉引用** | 手册可以静态，但 agent 版应声明「权威 live 源」：`jplot cap …` / `jplot data describe`。静态文案不得与 live 白名单打架。 |
| **长文 agent body** | 人类页要短、分卡；agent 可以吃长 markdown / 完整示例 / 反例表。 |

### 1.3 Portal card 形状（摘要）

`src/jarvis_portal/manual_cards/json.yaml` 一类卡片是 **声明式 mapping**：

- `title` / `summary` / `lexer`
- `yaml:` 大块示例（input/output 段）
- `input_before` / `input_after` / `output_slice` / `observables`
- `input_notes` / `output_notes`

渲染器按固定卡片序列画 Panel，**不把 prose 散落在 Python 字符串里**。

PLOT 沿用「**card 是数据，renderer 无文案**」。

---

## 2. 产品契约

### 2.1 CLI 形状

```text
jplot man                         # 人类：索引（主题目录 + 最短工作流）
jplot man <topic>                 # 人类：该主题 Rich 手册
jplot man --json                  # agent：索引结构化
jplot man <topic> --json          # agent：该主题完整结构化 + 长文
jplot man -h                      # man 子命令 help（沿用 Rich help 几何）
```

兼容别名（可选，实现期二选一写死）：

- `--agent` ≡ `--json` 且 `audience=agent`（若未来 human 也要 json 摘要，再拆 `--format`）
- 第一版：**仅 `--json` = agent 完整载荷**；无 flag = human Rich。

全局约定与其它 agent 动词一致：

| 流 | 规则 |
|---|---|
| stdout | human：Rich 文本；agent：一个 JSON envelope |
| stderr | 仅用法错误 / 加载失败的人话 |
| exit | 0 ok；1 topic 缺失或卡损坏；2 用法错误 |
| envelope | `{api_version, kind: "man"\|"man.<topic>", ok, data, diagnostics, error}` |

### 2.2 人类默认输出（Human）

- 几何：与 `jplot -h` / Portal man **同款**（`terminal_width = max(80, columns)`，Panel `dim` 边、`bold magenta` 标题、主列 24）。
- 索引页：  
  - Overview（coding agent + human 各一句话）  
  - **Agent workflow**（4–6 条必跑命令，短）  
  - **Topics** 表（topic id + 一行摘要）  
  - Usage 表  
- 主题页：固定卡片序列（见 §4），**短、可扫、带 YAML 高亮**。  
- 不打印完整 `cap all` 转储；指向 `jplot cap …`。

### 2.3 Agent 输出（`--json`）

原则：**比人类版更自由、更长、更可组合**——coding agent 上下文窗口大，省 token 不如省来回猜。

`data` 建议形状（实现可加字段，勿删下列键）：

```json
{
  "topic": "workflow",
  "audience": "agent",
  "title": "…",
  "summary": "一句话",
  "role": "how-to | reference | trap | catalog",
  "priority": 10,
  "see_also": ["validate", "cap", "data.describe"],
  "related_cli": [
    {"argv": ["jplot", "data", "describe", "<file>", "--json"], "why": "列名唯一合法来源"}
  ],
  "live_sources": [
    {"verb": "cap.methods", "truth": "METHOD_DISPATCH + coordinates.required"}
  ],
  "sections": [
    {
      "id": "steps",
      "title": "Recommended agent loop",
      "kind": "ordered_steps",
      "items": [
        {"step": 1, "do": "…", "cli": "jplot data describe … --json", "write_yaml": false}
      ]
    },
    {
      "id": "yaml_shape",
      "title": "Minimal type-first figure",
      "kind": "yaml",
      "body": "Figures:\n  - name: …\n    type: posterior_2d\n    …"
    },
    {
      "id": "anti_patterns",
      "title": "Do not",
      "kind": "list",
      "items": ["Do not invent cmap names; use jplot cap cmaps", "…"]
    }
  ],
  "body_markdown": "可选：整页长文 markdown，人类版不展示或只展示摘要",
  "examples": [ { "title": "…", "yaml": "…", "notes": ["…"] } ],
  "diagnostics": [ { "code": "JP-VIZ-003", "when": "…", "fix_hint": "…" } ],
  "schema_ids": ["https://jarvis-plot.org/schema/v2/core/figure.json"],
  "card_version": 1
}
```

硬性规则：

1. **`write_yaml: false` 出现在 workflow 步骤里**——明确「CLI 不负责写 YAML」。  
2. 凡涉及合法字符串 / 列名，**必须**有 `live_sources` 或 `related_cli` 指向 cap/data，禁止只在 prose 里列死表当唯一真相。  
3. `body_markdown` 可以很长；`sections` 保持可解析。  
4. 索引 `--json` 的 `data.topics[]` 至少含 `id, title, summary, priority, role`。

### 2.4 非目标

- 不在 man 里执行 validate/dryrun/render（可 **链接** 命令，不代跑）。  
- 不提供 `man --edit` / 写回卡片。  
- 不把整份 `YAML_DESIGN.md` 原样塞进一个 topic（拆主题；长文放 `body_markdown`）。  
- 不实现第二套 `jplot docs` 动词——**就叫 `man`**，与 Portal/HEP 肌肉记忆一致。

---

## 3. 主题目录（v1 必做）

主题 id 用 **小写 + 连字符**，与 Portal format 名同风格。

| id | 角色 | 人类一句话 | agent 额外内容 |
|---|---|---|---|
| *(index)* | catalog | 目录 + 最短闭环 | topics[] + workflow 摘要 |
| `workflow` | how-to | coding agent 推荐命令顺序 | 完整有序 steps、write_yaml=false |
| `yaml-root` | reference | 根键 DataSet/Figures/output | 闭合词汇、反例、schema_id |
| `type-posterior-2d` | how-to | type 宏主接口 | slots、最小 YAML、展开后看什么 |
| `type-profile-2d` | how-to | profile 宏 | 同上 |
| `layer-method` | reference | method + coordinates 合约 | 指向 cap methods；常见缺 z |
| `style-axes` | trap | style card 与 ax/axc | usable 坏卡、axc 只在 *cmap |
| `data-columns` | how-to | 列名从哪来 | describe/head/eval/suggest-axes |
| `validate` | how-to | validate / --fix / doctor | exit 码、diagnostics 形状 |
| `dryrun-viz` | how-to | 行数账本与 JP-VIZ | 各 VIZ 码 when/fix_hint |
| `diagnostics` | catalog | JP-* 码段地图 | 链 explain + guidance 表摘要 |
| `cli-map` | catalog | 全部 agent 动词 | 与 cap cli 一致，勿手写漂移 |

v1 **至少**：index + `workflow` + `yaml-root` + `type-posterior-2d` + `data-columns` + `validate` + `cli-map`。  
其余可随后迭代加卡，**加卡不改 renderer**。

---

## 4. 卡片数据模型（package data）

### 4.1 布局

```text
jarvisplot/manual_cards/
  manifest.yaml          # topic 顺序、priority、别名
  workflow.yaml
  yaml-root.yaml
  type-posterior-2d.yaml
  …
```

`pyproject.toml` `package-data` 增加 `manual_cards/**/*`。

### 4.2 单卡 schema（逻辑）

```yaml
# jarvisplot/manual_cards/workflow.yaml
id: workflow
title: Coding-agent workflow
summary: Discover columns and vocabulary, write YAML yourself, then validate/dryrun.
role: how-to          # how-to | reference | trap | catalog
priority: 10          # 索引排序，小者优先
see_also: [data-columns, validate, dryrun-viz, cli-map]
related_cli:
  - argv: [jplot, data, describe, "<file>", "--json"]
    why: column-name whitelist
  - argv: [jplot, cap, all, "--json"]
    why: string whitelist
  - argv: [jplot, validate, "<yaml>", "--json"]
    why: shape + columns gate
  - argv: [jplot, dryrun, "<yaml>", "--json"]
    why: row ledger + JP-VIZ
live_sources:
  - verb: data.describe
    truth: real file columns only
  - verb: cap.all
    truth: methods/styles/cmaps/funcs/transforms
human:
  # 只放人类页需要的短块；renderer 按固定顺序画
  panels:
    - kind: overview
      body: |
        CLI discovers and judges. You write the YAML in your editor.
    - kind: steps
      title: Loop
      items:
        - "1. jplot data describe …"
        - "2. jplot cap all …"
        - "3. edit plot.yaml"
        - "4. jplot doctor plot.yaml --json"
    - kind: yaml
      title: Minimal type-first sketch
      lexer: yaml
      body: |
        Figures:
          - name: posterior
            type: posterior_2d
            …
    - kind: notes
      items:
        - Never invent method or cmap strings.
        - Bare path renders: jplot plot.yaml (no jplot run).
agent:
  # 人类页可忽略；--json 时并入 data
  body_markdown: |
    ## Full agent playbook
    ...
  sections: []          # 若空，由 loader 从 human.panels 投影 + body_markdown
  examples: []
  diagnostics: []
  anti_patterns: []
```

### 4.3 manifest

```yaml
# jarvisplot/manual_cards/manifest.yaml
schema_version: 1
default_topic: null          # null => index
aliases:
  playbook: workflow
  agent: workflow
  posterior: type-posterior-2d
  viz: dryrun-viz
topics:
  - workflow
  - yaml-root
  - type-posterior-2d
  - type-profile-2d
  - layer-method
  - style-axes
  - data-columns
  - validate
  - dryrun-viz
  - diagnostics
  - cli-map
```

---

## 5. 模块划分（实现地图）

| 模块 | 职责 |
|---|---|
| `jarvisplot/manual_cards/*.yaml` | 唯一文案与示例真相（声明式） |
| `jarvisplot/man_catalog.py` | 加载 manifest/卡、别名解析、lint（缺 id/title 失败） |
| `jarvisplot/man_render_human.py` | Rich 索引 + 主题页（几何对齐 `cli.py` help） |
| `jarvisplot/man_render_agent.py` | 组装 envelope `data`（长文 OK） |
| `jarvisplot/verbs/man.py` | argparse：`man [topic] [--json]` |
| `tests/test_man_cli.py` | 索引/主题/--json 键集合/未知 topic exit 1/无 matplotlib |

**禁止**：在 `verbs/man.py` 里堆长字符串手册正文。

### 5.1 与现有能力的关系

| 已有 | man 用法 |
|---|---|
| `cap` | man 讲「何时调用」；枚举本身 **永远 live 拉 cap** |
| `data describe` | man 讲「列名纪律」；不复制列 |
| `template` / `suggest` | man 的 workflow 可引用；模板正文仍由 template 动词出 |
| `explain JP-*` | `diagnostics` topic 做地图；单码细节可 `related_cli: explain` |
| `docs/*.md` | 人读设计；**man 是运行时、可调用的子集**，不替代仓库文档 |

### 5.2 可选增强（非 v1）

- `jplot man workflow --json | jq .data.body_markdown` 已足够 agent。  
- 从 `diagnostic_guidance.KNOWN_CODES` **生成** `diagnostics` 卡的附录（CI 锁一致性）。  
- `man style-axes` 调 `capabilities.section("styles")` 注入 live `usable` 表（card 只写解释，表 live 拼）。

---

## 6. 人类页卡片序列（渲染契约）

对齐 Portal 的「固定顺序、少分支」：

**Index**

1. Panel `Jarvis-PLOT manual` — summary  
2. Panel `Agent loop (write YAML yourself)` — 短步骤  
3. Panel `Topics` — 表  
4. Panel `Usage` — `man` / `man TOPIC` / `man TOPIC --json`  

**Topic**

1. Panel title + summary  
2. Section `What to run` — related_cli  
3. Section `YAML` — 一个主示例（Syntax yaml）  
4. Section `Traps` / `Notes` — bullet  
5. Section `See also` — topic ids + CLI  

Agent 的 `body_markdown` / 多 examples **不**全部打进 human 页。

---

## 7. Coding agent 使用约定（写进 `workflow` 卡）

强制叙事（与用户产品判断一致）：

1. **Discover** — `data describe` + `cap all`（或 man 提示的子集）。  
2. **Author** — 在工作区直接写/改 `*.yaml`（**不**要求 `config set`）。  
3. **Judge** — `validate --json` → `dryrun --json` 或 `doctor --json`。  
4. **Render** — `jplot <file>` 仅在需要出图时。  
5. **Learn errors** — `explain <JP-*>` 或 `man diagnostics`。

`config set` / `suggest --write` 在 man 里标成 **optional tools**，不是主路径。

---

## 8. 验收标准（可执行）

1. `jplot man` 退出 0，stdout 含 Topics 与 workflow 要点；无 JSON。  
2. `jplot man workflow --json | python -m json.tool` 成功；  
   `data.related_cli` 非空；存在明确「自行写 YAML」的步骤。  
3. `jplot man no-such-topic --json` → `ok=false`，exit 1，`error` 含 did-you-mean 主题。  
4. 子进程：`jplot man workflow` 后 `matplotlib` 不在 `sys.modules`（与 validate 同纪律）。  
5. `tests/test_man_cli.py`：manifest 每个 topic 可加载；human 渲染非空；agent 键集合冻结。  
6. 加一张新卡 = 新 yaml + manifest 一行；**不改** `man_render_*.py` 也能过测。

---

## 9. 实现分期

| 阶段 | 内容 | 规模 |
|---|---|---|
| **M-man-0** | 本设计入库；台账 H 轨挂 `man` 任务 | S |
| **M-man-1** | catalog loader + `verbs/man.py` + 索引 + `workflow`/`cli-map`/`validate` 三卡 | M |
| **M-man-2** | human Rich 渲染完整；其余 v1 topic 卡 | M |
| **M-man-3** | agent 长文 `body_markdown` + live 注入 styles usable / methods 摘要（可选） | S–M |
| **M-man-4** | 与 `explain`/guidance 交叉 CI；文档 README 链到 `jplot man` | S |

建议：**M-man-1 即可让 coding agent 靠 CLI 自学主路径**；不必等全部 topic 写完。

---

## 10. 与 Portal 的差异清单（实现时勿「抄歪」）

| | Portal | PLOT |
|---|---|---|
| 主题粒度 | 文件格式 | 工作流 + YAML 面 + 诊断 |
| Agent JSON | 无 | **一等公民** |
| 长文 | 无 | `body_markdown` 允许长 |
| Live 数据 | 注册表 formats | cap / 可选 styles usable |
| 写配置 | N/A | 明确 **不**通过 man 写 |
| 渲染依赖 | 无 matplotlib | man 路径 **禁止** matplotlib |

---

## 11. 开放问题（实现前可默认）

| # | 问题 | 默认 |
|---|---|---|
| Q1 | human 是否也提供 `--json` 短摘要？ | 否；仅 agent 完整 JSON |
| Q2 | topic 别名是否稳定 API？ | 是；写在 manifest.aliases |
| Q3 | 是否把 `suggest`/`template` 升成 workflow 必选？ | 否；optional |
| Q4 | man 是否显示完整 JP-* 表？ | 摘要 + `jplot explain`；全表可放 agent body_markdown |

---

## 12. 文档与台账挂钩

- 实现任务建议记入 `V2_DEV_LEDGER` 轨 H：  
  - **H4** 从「静态 AGENT_PLAYBOOK.md」改为 **`jplot man`（可调用 playbook）**；静态 md 可作卡片源的人读镜像，但 **不以 md 为 agent 主入口**。  
- README / 根 help：Usage 增加 `jplot man [topic] [--json]`。  
- 不修改 frozen `AGENT_DATA_API.md` 直至 DR-01；man 的 envelope 沿用现有内部 `agent_io` 形状。

---

## 13. 最小示例（实现后形态预览）

```bash
# Human
jplot man
jplot man workflow

# Coding agent
jplot man --json
jplot man workflow --json
jplot man type-posterior-2d --json
jplot man style-axes --json
```

Agent 侧伪流程：

```text
man workflow --json     → 步骤与纪律
data describe --json    → 列
cap methods/styles --json → 词
# 自己写 plot.yaml
validate / doctor --json
man diagnostics --json  → 若需码段地图
explain JP-… --json
```

---

**End of design.** 实现时以本文件为验收依据；行为变更先改本文件再改代码。
