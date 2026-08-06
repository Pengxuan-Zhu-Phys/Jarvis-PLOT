# YAML 设计审查：Human + AI 友好 × 奥卡姆剃刀

Status: review  
Date: 2026-08-06  
原则（今日对齐）：

1. **写 YAML 不经 CLI**——人与 coding agent 都用编辑器写文件。  
2. **CLI = 查事实 + 验结果 +（即将）`man` 说明书**。  
3. **为人友好 ⇒ 默认不破坏现有 YAML 习惯**。  
4. **奥卡姆：能删则删、能合并则合并；删不掉的要有「唯一职责」**。  

对照：`docs/design/YAML_DESIGN.md`、`docs/roadmap/V2_YAML_AGENT_ERGONOMICS.md`、当前 schema/validate/cap/data/dryrun 实现，以及 `bin/` + `Example/` 语料抽样。

---

## 0. 总判

| 维度 | 判断 |
|---|---|
| **管道骨架** `DataSet → Figures → layers → method/coordinates/style` | **保留**。人读得懂，语料大量使用，AI 也容易套。 |
| **Style card 两级 token** | **保留**。把几何与默认美学从每张图里抽走，是人侧最大的减负。 |
| **表达式 `expr:`** | **保留**。科学绘图的刚需；AI 用 `data eval` / `cap funcs` 自证。 |
| **今日已上的 validate/cap/data/dryrun** | **不改 YAML，却大幅改善双方**——这是正确的第一刀。 |
| **真正该剃的** | 不是骨架，而是 **「同形不同义」**、**「多写法同一事」**、**「写了等于没写」**、**「两条等价长路径」**。 |

**一句话：**  
YAML 不必为 AI 推倒重来；要剃的是**认知税**和**静默语义**，不是 `DataSet`/`Figures` 这层名词。

语料信号（`bin/` + `Example/`，约 21 文件）：

- **0** 张 figure 用 `type:` 宏（文档推 type 为主接口，语料仍是手写 layers）。  
- layer 键高度收敛：`name/axes/method/coordinates/style/data`（+ 部分 `share_data`）。  
- method 高度偏斜：`voronoi`/`fill` 主导，不是 30 个 method 均匀使用。  
- coordinates 值：dict 与 list 混用；layer 上仍有少量 `lim/scale/name/label`（**运行时忽略**）。

---

## 1. 已经「变友好」且不改 YAML 的部分（继续压）

这些不改文件形状，却是 human+AI 最大杠杆——**优先于任何 break 变更**：

| 机制 | 人 | AI |
|---|---|---|
| 闭合词汇 + did-you-mean | 拼错立刻知道 | 一轮收敛 |
| `cap` 白名单 | 少翻文档 | 不猜字符串 |
| `data describe` | 看列 | 列名唯一来源 |
| method 坐标合约 | 缺 z 早报 | 同左 |
| `dryrun` / JP-VIZ | 少盯空白图 | 无多模态也能判 |
| `man`（设计中） | 可扫手册 | 可调用说明书 |

**审查建议：** 把「文档写清楚」升级为 **`jplot man` + live cap**，而不是再写第三份 YAML 长文。

---

## 2. 奥卡姆清单：值得优化的 YAML 点

分级：

- **P0 可静默减负**——不 break 语料，或仅 warn  
- **P1 可选糖**——加法兼容，旧写法仍收  
- **P2 V2 break**——只在大版本、有 migrate 时做  

### 2.1 同形不同义：`coordinates` 三张皮 — **最高认知税**

| 位置 | 人以为 | 实际 |
|---|---|---|
| `layers[].coordinates.x` | 可写 lim/scale/name | **几乎只要 expr**；lim/scale/name/label **静默忽略**（JP-OWN-001 已报一部分） |
| transform `profile` / density 的 coordinates | 同结构 | **要 name/lim/scale** |
| figure `type:` 的 x/y | 又一种 | 宏字段，展开后进 frame/transform |

语料：layer 上已出现被忽略的 `lim/scale/name/label`（少但说明人会写）。

| 刀法 | 级别 | 说明 |
|---|---|---|
| **继续加强 JP-OWN-001**（layer 上 lim/scale 明确「请写到 frame」） | **P0** | 不改形状 |
| 文档 / `man` 画一张「三处 coordinates 对照表」 | **P0** | 人+AI |
| Layer 允许 `coordinates: {x: m_A, y: tanb}` 纯字符串糖 | **P1** | 旧 dict 保留；AI 更不易写错 |
| V2：layer 改 `columns:`，transform 独享 `coordinates:` | **P2** | 形状不同=最强防混；需 migrate |

**为人友好默认：** 先 P0+P1，**不急着 columns 分裂**。

---

### 2.2 多写法同一事 — **纯复杂度，无科学收益**

| 现象 | 建议 | 级别 |
|---|---|---|
| `keep_columns` / `drop_columns`：string \| list \| dict | 文档与 man **只教 list**；其余标 legacy，validate **info** 可提示「可写成 list」 | P0 文档 / 软提示 |
| `make_interp_2d.grid`：`500` / `[nx,ny]` / `{nx,ny}` | man **只教一种**（推荐 `{nx, ny}` 或标量）；实现保留 | P0 |
| `style` vs `style_card`（type 宏） | **归一文档名：只教 `style`**；`style_card` 作别名 | P0 |
| `combine: seperate` 拼写 | 实现认错词；**对外只教 `concat`**，`seperate` 不出现在 man/模板 | P0 |
| `posterior_density` vs `make_density_core`+`make_interp_2d` 长链 | **人侧主推一条**：`type: posterior_2d` 或单一 `posterior_density`；长链标 advanced | P0 叙事 |
| `jpfield` / `jpcontour*` vs transform 插值再 contour | man 写清 **推荐路径一张表**；勿在无迁移时删 method | P0 |

奥卡姆：**实现允许多入口，界面只宣传一个入口。**

---

### 2.3 写了等于没写 — **对双方都是背叛**

| 项 | 现状 | 建议 |
|---|---|---|
| figure 级 `legend:` | 不读，只读 `frame.<ax>.legend` | **已有 JP-OWN-001**；man 强调；长期可考虑 validate **error**（小 break） |
| layer `coordinates.*.lim/scale/name` | 静默忽略 | 同上，P0 warn 即可 |
| 未知 transform 键 | 曾整步跳过 | **schema 已堵**；保持 |
| 未知 layer/root 键 | 曾静默 | **schema 已堵**；保持 |

**不要为「兼容坏配置」继续静默。** 静默是最反奥卡姆的：表面省事，系统变不可信。

---

### 2.4 `type:` 宏 vs 手写 layers — **叙事与现实脱节**

- 设计：`type:` 应是主接口。  
- 语料：`bin/`/`Example/` **0** 使用。  
- 人：会 copy 旧长 YAML。  
- AI：`template`/`suggest` 已走 type 路径，但 dryrun **不展开 type**。

| 建议 | 级别 |
|---|---|
| man / suggest **默认 type-first**；手写 layers 标 advanced | P0 |
| dryrun/doctor **展开 type 再查账本** | **P0 实现**（不改 YAML，修工具） |
| 新 example 全部 type-first，旧 example 不删 | P0 语料 |
| 强迫 layers 消失 | **不要**（人侧破坏大） |

---

### 2.5 Style / 颜色所有权 — **该瘦的是规则，不是 card 系统**

| 项 | 建议 |
|---|---|
| `frame.axc.color.cmap` vs `layer.style.cmap` | **一条规则写死并 man 化**：colorbar 归属 `frame.axc`；layer style 只影响 artist。JP-VIZ-004 已帮 AI。 |
| 坏 1x1 Ternary card（`Figure` 顶层） | **修卡或 delist**（P0 工程）；`usable:false` 已够 AI |
| `style: [family, variant]` | **保留**；人友好 |

---

### 2.6 Transform 单键步骤 `- filter:` — **AI 痛、人熟**

| 方案 | 级别 |
|---|---|
| 保持单键；靠闭合 schema + did-you-mean | **现状，P0 够用** |
| V2 判别式 `{type: filter, expr: …}` | P2（G1）；人迁移成本高 |

**为人友好：不先做判别式 break。**

---

### 2.7 表达式与列

| 项 | 建议 |
|---|---|
| `expr: col` vs `expr: "col"` | 保持；`data eval` 已覆盖 |
| 列名带点 `pVa.E` | 实现已处理；man 一句即可 |
| 函数名 `ln`/`log`/`log10` | `cap funcs` + expr_names；无需改 YAML |

---

### 2.8 根级与杂项

| 项 | 建议 |
|---|---|
| `project` / `version` / `Functions` | 保留；`Functions` 少见则 man 标 optional |
| `transform` 挂在 DataSet 与 layer.data | **两处都合理**（全局预滤 vs 层局部）；man 说清归属即可，勿强行只留一处 |
| `share_data` | 保留；是少层重复的关键抽象 |
| `debug: true` | 保留；人侧 layout 调试刚需 |

---

## 3. 「别动」清单（奥卡姆下的保留理由）

| 设计 | 为何不剃 |
|---|---|
| DataSet + Figures 分治 | 数据与画布分离，人脑模型清晰 |
| layer 五元组 data/method/coordinates/style/axes | 管线可读 |
| style card 外置 | 减重复、统一刊物风格 |
| method 字符串派发 | 与 matplotlib 心智贴近 |
| `share_data` | 密度层复用，删了更啰嗦 |
| 表达式语言 | 科学场景不可删 |

---

## 4. 对人 vs 对 AI：YAML 是否要两套？

**不要两套方言。**

| | Human | Coding agent |
|---|---|---|
| 写 | 编辑器 | 编辑器 / 生成后落盘 |
| 查词/列 | man 短文 + 偶尔 cap | **强制** data/cap JSON |
| 验 | validate / doctor | 同左，--json |
| 糖 | type:、字符串 coordinates（P1） | 同糖；更依赖 template 示例 |

AI 需要的额外东西应在 **CLI/man 载荷**，不在 YAML 第二语法。

---

## 5. 推荐路线图（按剃刀排序）

### 现在就做（不改语料语义）

1. **实现 `jplot man`**（`docs/dev/MAN_CLI.md`）— 把今日原则写进可调用说明书。  
2. **dryrun 展开 `type:`** — 补工具洞，不改 YAML。  
3. **OWN 类诊断保持严格** — 消灭「写了没效果」。  
4. **文档/man 只宣传一条后验路径**（`type: posterior_2d` 或 `posterior_density`）。  
5. **修/下架坏 style 卡**；examples 逐步 type-first。

### 可加法兼容（P1）

6. Layer `coordinates` **字符串简写** `x: m_A`。  
7. validate 对 multi-form 输入给 **info「推荐写法」**（不 fail）。  

### 大版本再谈（P2，非今日）

8. coordinates / columns 分裂。  
9. transform 判别式 union。  
10. 颜色所有权硬 fail 旧写法。  
11. 消灭 `seperate` 等历史别名。  

---

## 6. 与「代码审查」交叉的结论

| 代码面 | 评价 |
|---|---|
| schema closed + zone | 正确方向；继续用 **文档与 live 双通道** 而非再扩 figure 属性堆 |
| figure.json 为 type 宏开的一堆可选键 | 务实，但 **增加了「根 figure 属性噪声」**；man 应分「手写 figure」vs「type figure」两套最小面 |
| method_contracts 27 文件 | 对 AI 好；对人类 **man + cap methods 即可**，不必读 27 个 json |
| config set | 对人/AI **均非写 YAML 主路径**；保留作可选工具即可 |
| suggest/template | 对 AI 友好；人可当示例生成器 |

**过度设计风险：** 为 type 宏在 figure 上挂齐所有 keys，长期会让「最小 figure 长什么样」变糊——应用 **man 分区** 化解，而不是再加第三种 figure 形态。

---

## 7. 判决表（给决策用）

| 议题 | 改 YAML？ | 优先级 |
|---|---|---|
| 静默键 / 错键 | 否（已 gate） | 维持 |
| coordinates 三义 | 暂否；P1 字符串糖；P2 分裂 | 高（叙事+warn） |
| 多形态 list/string/dict | 否；只教一种 | 中 |
| type 为主接口 | 否；工具+示例+man | 高 |
| 后验两条路径 | 否；只推荐一条 | 中 |
| transform 判别式 | V2 | 低（今日） |
| style card | 否 | — |
| 坏卡 / usable | 修数据 | 高 |
| man CLI | 否 | **最高产品优先级** |

---

## 8. 结语

奥卡姆用在 Jarvis-PLOT 上不是「YAML 越短越好」，而是：

> **每个出现在用户眼前的概念，只对应一件运行时真事；  
> 每件真事，对外只教一种写法；  
> 查不全的，用 CLI 查，不靠记忆。**

今日栈已经把「静默失败」从致命降到可修。下一步最大的人+AI 收益是 **说明书（man）+ type 路径工具闭环 + 消灭假配置**，而不是立刻 break 大家手里的 `voronoi`/`fill` 长卷。

**End of review.**
