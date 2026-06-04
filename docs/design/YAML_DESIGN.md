# Jarvis-PLOT YAML Design Document

> 目标：为 YAML 重封装工作提供完整参考。梳理当前 YAML 所有顶层结构、Transform 链、画图 Method、坐标/表达式系统、Style 系统。标注出"过度复杂 / 可简化"的地方。

---

## 一、YAML 顶层结构

```yaml
version: "0.3"                  # schema 版本

project:
  name: "My Project"            # 项目名
  workdir: /path/to/workdir     # 工作目录（默认 YAML 所在目录）

DataSet:                        # 数据源列表
  - name: df1
    path: data.csv
    type: csv
    transform: [...]            # 可选：数据集级别的 transform 链

Figures:                        # 图列表
  - name: fig1
    enable: true
    style: [a4paper_2x1, rectcmap]
    frame: { ... }
    layers: [ ... ]

Functions: []                   # 自定义函数（通常为空）

output:
  dir: ./plots                  # 输出目录
  dpi: 400
  formats: [png, pdf]
```

### 各部分关系图

```
DataSet (加载 → 可选 transform → DataFrame)
    ↓
Figures[].layers[].data[].source  引用 DataSet.name
    ↓
layers[].data[].transform         层级 transform 链
    ↓
share_data 缓存（跨 layer 复用）
    ↓
method + coordinates + style      渲染到 axes
```

---

## 二、DataSet 配置

### 2.1 基本字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 唯一标识符，在 layers 中通过 `source:` 引用 |
| `path` | string | 是 | 文件路径（相对于 workdir 或绝对路径） |
| `type` | string | 是 | `csv` / `hdf5` / `parquet` |
| `dataset` | string | 否 | HDF5 group 名称 |
| `columns` | dict | 否 | HDF5 列配置（见下） |
| `transform` | list | 否 | 数据集级别 transform 链 |

### 2.2 HDF5 列配置

```yaml
columns:
  isvalid_policy: clean          # clean（过滤无效行）| raw（保留全部）
  rename:
    - source: "Parameters/m_A"
      target: m_A
  load_whitelist:                # 只加载指定列
    - "Parameters/m_A"
    - "Loglikelihood"
```

### 2.3 多数据源合并

```yaml
layers:
  - data:
    - source: [df_samples_0, df_samples_1]   # 数组语法 → concat
      transform: [...]
```

**⚠️ 复杂度标注：** HDF5 的 `columns.rename` + `load_whitelist` + `isvalid_policy` 组合已经很灵活，但仅限 HDF5。CSV/Parquet 没有等价配置，不对称。

---

## 三、Transform 链（完整列表）

Transform 在两个地方都可以使用：DataSet 级别 和 Layer 级别。按顺序执行，每步的输出是下一步的输入。

### 3.1 总览

| YAML 键 | 功能 | 输入→输出 | 适用级别 |
|---------|------|----------|---------|
| `filter` | 按条件过滤行 | N行 → M行（M≤N） | 两级 |
| `sortby` | 按列/表达式排序 | 不变行数 | 两级 |
| `add_column` | 添加计算列 | 增加列 | 两级 |
| `keep_columns` | 保留指定列 | 减少列 | 两级 |
| `drop_columns` | 删除指定列 | 减少列 | 两级 |
| `profile` | 二维数据归约/分箱 | 原始采样 → 网格值 | 两级 |
| `make_density_core` | 后验密度支撑重建 | 原始加权采样 → (x,y,mass) 三列 | 两级 |
| `make_interp_2d` | 二维插值到规则网格 | 离散点 → 网格 (x,y,z) 三列 | 两级 |
| `posterior_density` | 后验密度一体化管线 | 原始加权采样 → 规则网格密度 (x,y,density) 三列 | 两级 |
| `to_csv` | 导出当前 DataFrame 到 CSV | 不改变数据 | 两级 |
| `to_parquet` | 导出到 Parquet | 不改变数据 | 仅 DataSet |

### 3.2 各 Transform 详细规格

---

#### `filter`

```yaml
- filter: "LogL > -100"
- filter: "(x >= 0) & (x <= 5)"
- filter: true                   # 保留全部
```

参数：布尔表达式字符串。支持 `&&`/`||`/`&`/`|`、列名直接引用、NumPy 函数。

---

#### `sortby`

```yaml
- sortby: LogL                   # 按列升序
- sortby: "np.abs(x)"           # 按表达式排序
```

参数：列名或表达式字符串。始终升序。

---

#### `add_column`

```yaml
- add_column:
    name: weight                 # 新列名
    expr: "exp(LogL)"            # 表达式
```

**⚠️ 复杂度标注：** 必须用 dict 嵌套。可简化为 `- add: {weight: "exp(LogL)"}` 形式。

---

#### `keep_columns` / `drop_columns`

```yaml
- keep_columns: [x, y, weight]          # 列表
- keep_columns: important_col           # 单个字符串
- keep_columns: {columns: [a, b]}       # dict 形式
- drop_columns: [temp, debug]
```

**⚠️ 复杂度标注：** 三种输入形式（string/list/dict）增加了理解成本。

---

#### `profile`（数据归约）

```yaml
- profile:
    method: bridson              # grid | bridson
    bin: 100                     # 分箱数
    objective: max               # max | min | mean
    grid_points: rect            # 网格几何
    coordinates:
      x:
        expr: xx
        name: x
        scale: log               # linear | log
        lim: [0.1, 5]
      y:
        expr: yy
        name: y
        lim: [0, 5]
      z:
        expr: LogL
        name: z0
```

功能：将大量采样点归约到二维网格/支撑点上，每个 bin 取 objective 值。

**⚠️ 复杂度标注：** coordinates 每个轴需要同时声明 `expr`（输入来源）、`name`（输出列名）、`scale`、`lim` 四个字段。`expr` 和 `name` 常常重复。

---

#### `make_density_core`（后验密度重建核心）

```yaml
- make_density_core:
    method: bridson              # grid | bridson | kde
    coordinates:
      x: {expr: xx, name: x, lim: [0, 5], scale: linear}
      y: {expr: yy, name: y, lim: [0, 5]}
      weight: {expr: "exp(LogL)", name: weight}
    normalize: true              # 归一化总质量为 1

    # method=grid 时
    grid: {bins: 128}            # 或 {nx: 128, ny: 128}

    # method=bridson 时
    bridson: {bin: 32, seed: 42, k: 30}

    # method=kde 时
    bw_method: "0.5 * scott"     # scott | silverman | float | "factor*base"
    grid: {bins: 128}

    # 可选：自适应 refinement
    refinement:
      enabled: true
      iterations: 2
      alpha: 0.30
      eta: 0.50
      anisotropic: true
      split_enabled: true
      merge_enabled: true
      split_quantile: 0.75
      merge_quantile: 0.20
      max_splits: 2
      max_generators: 64
      seed: 7

    # 调试导出
    _mesh_debug: {to_csv: ./mesh_debug.csv}
```

**输出：** 仅三列（由 coordinates 中的 name 决定），默认 `[x, y, weight]`。

**⚠️ 复杂度标注：**
- coordinates 结构与 profile 相同，冗余
- `refinement` 有 15+ 参数，绝大多数用户不需要调
- `_mesh_debug` 是内部调试用，不应暴露给普通用户
- `grid` 和 `bridson` 分别配置 bins，语义不统一

---

#### `make_interp_2d`（二维插值）

```yaml
- make_interp_2d:
    method: natural_neighbor     # natural_neighbor | triangulation | griddata
    as_density: true             # 是否把 z 当作保守质量 → 除以面积得密度
    normalize: true              # 是否归一化积分为 1
    coordinates:
      x: {expr: x, name: x, lim: [0, 5], scale: linear}
      y: {expr: y, name: y, lim: [0, 5]}
      z: {expr: weight, name: density}
    grid: 500                    # int → 正方形 | [nx, ny] | {nx:, ny:}
    nan_policy: strict           # strict（hull 外为 NaN）

    # method=triangulation 时
    triangulation: {kind: linear}    # linear | cubic

    # method=griddata 时
    griddata: {kind: linear}         # nearest | linear | cubic
```

**输出：** 三列规则网格 (x, y, z)。

**⚠️ 复杂度标注：**
- coordinates 结构又一次重复
- `grid` 支持三种格式（int / list / dict），增加学习成本
- `as_density` + `normalize` 的组合语义不直观

---

#### `to_csv` / `to_parquet`

```yaml
- to_csv: ./output.csv
- to_parquet: ./output.parquet
```

不改变数据流。

---

### 3.3 常见 Transform 管线模式

**模式 A：直接绘制采样散点**
```yaml
transform:
  - sortby: LogL
```

**模式 B：后验密度 → 插值 → 渲染**
```yaml
transform:
  - make_density_core:
      method: bridson
      coordinates: { ... }
      normalize: true
  - make_interp_2d:
      method: natural_neighbor
      as_density: true
      coordinates: { ... }
      grid: 500
```

**模式 C：剖面 → 渲染**
```yaml
transform:
  - profile:
      bin: 100
      objective: max
      coordinates: { ... }
```

---

## 四、画图 Method（完整列表）

在 `layers[].method` 字段中指定。

### 4.1 基础二维绘图

| method | 所需坐标 | 可选坐标 | 说明 |
|--------|---------|---------|------|
| `scatter` | x, y | c (颜色) | 散点图 |
| `plot` | x, y | — | 折线图 |
| `errorbar` | x, y | xerr, yerr | 误差线 |
| `step` | x, y | — | 阶梯图 |
| `bar` | x, height | width, bottom | 竖条形图 |
| `barh` | y, width | height, left | 横条形图 |
| `hist` | x | bins, density | 直方图 |
| `fill` | x, y | — | 填充多边形 |
| `fill_between` | x, y1, y2 | — | 区间填充 |
| `fill_betweenx` | y, x1, x2 | — | 纵向区间填充 |
| `quiver` | x, y, u, v | — | 矢量场 |

### 4.2 网格/图像绘图

| method | 所需坐标 | 说明 |
|--------|---------|------|
| `pcolormesh` | x, y, z | 伪彩色网格（最常用的密度渲染方式） |
| `pcolor` | x, y, z | 伪彩色（较慢，少用） |
| `contour` | x, y, z | 等高线 |
| `contourf` | x, y, z | 填充等高线 |
| `imshow` | image | 图像显示 |

### 4.3 JarvisPLOT 自定义散点插值方法

| method | 所需坐标 | 说明 |
|--------|---------|------|
| `jpcontour` | x, y, z | 散点数据 → 内部插值 → 等高线 |
| `jpcontourf` | x, y, z | 散点数据 → 内部插值 → 填充等高线 |
| `jpfield` | x, y, z | 散点数据 → 内部插值 → pcolormesh |

这三个方法在 adapter 内部自动做插值，不需要用户先跑 `make_interp_2d`。

**⚠️ 复杂度标注：**
- `jpcontourf` 和 `make_density_core → make_interp_2d → contourf` 管线实现了类似目标，但方式完全不同
- `jpfield` 的插值参数通过 `style.interp` 传递，不通过 `coordinates`，不一致

### 4.4 三角化方法

| method | 所需坐标 | 说明 |
|--------|---------|------|
| `tripcolor` | x, y, z | 三角剖分伪彩色 |
| `tripcolor_axes` | x, y, z | 同上，明确在 axes 坐标系 |
| `tricontour` | x, y, z | 三角剖分等高线 |
| `tricontourf` | x, y, z | 三角剖分填充等高线 |
| `triplot` | x, y | 显示三角剖分网格 |

### 4.5 Voronoi 方法

| method | 所需坐标 | 说明 |
|--------|---------|------|
| `voronoi` | x, y [, z] | Voronoi 图（有 z 则着色） |
| `voronoif` | x, y | Voronoi 边界层 hatched 填充 |

### 4.6 特殊方法

| method | 说明 |
|--------|------|
| `dynesty_runplot` | Dynesty 嵌套采样诊断图 |

---

## 五、Layer 配置结构

```yaml
layers:
  - name: layer_name               # 可选标识
    data:                           # 数据源
      - source: dataset_name        # DataSet name 或 share_data name
        transform: [...]            # 可选 transform 链
    share_data: cache_name          # 可选：缓存数据供其他 layer 使用
    axes: ax                        # 目标轴（ax / ax0 / ax1 等）
    method: scatter                 # 绘图方法（见第四节）
    coordinates:                    # 列映射
      x: {expr: col_x}
      y: {expr: col_y}
      c: {expr: col_color}         # 仅对支持颜色的 method
    style:                          # 样式参数
      marker: .
      s: 2
      cmap: viridis
      zorder: 1
    colorbar: axc                   # 可选：关联哪个 colorbar 轴
```

### 5.1 Coordinates 坐标映射

两种写法：
```yaml
# 简写
coordinates:
  x: {expr: column_name}

# 完整
coordinates:
  x:
    expr: "np.log10(mass)"     # 表达式
    name: x                    # （仅 transform 使用，layer 不需要）
    lim: [0, 10]               # （仅 transform 使用）
    scale: log                 # （仅 transform 使用）
```

**⚠️ 复杂度标注：** Layer 的 coordinates 只需要 `expr`。但因为和 transform 共用了同一个 coordinate dict 结构，所以带了很多 layer 不需要的字段（`name`, `lim`, `scale`）。

### 5.2 HPD 等高线（特殊 style）

```yaml
style:
  contour_mode: posterior_hpd
  masses: [0.6827, 0.9545]         # 1σ, 2σ
  labels: ["1σ / 68%", "2σ / 95%"]
  colors: [black, white]
  linestyles: [solid, solid]
  linewidths: [0.2, 0.2]
```

这是 `contour` method 的特殊模式，自动计算 HPD 等高线级别。

---

## 六、Frame / Axes 配置

```yaml
frame:
  ax:                              # 主绘图轴
    labels:
      x: "$x$ label"              # LaTeX 支持
      y: "$y$ label"
    xlim: [0, 5]
    ylim: [0, 5]
    xscale: linear                 # linear | log
    yscale: linear
    ticks:
      major: {direction: in, length: 4}
      minor: {direction: in, length: 2}

  axc:                             # Colorbar 轴
    label:
      xlabel: density              # 水平 colorbar
      ylabel: density              # 垂直 colorbar
    color:
      cmap: jarvis_rainbow2_r
      scale: linear                # linear | log
      vmin: 0
      vmax: 0.8
```

多轴使用 `ax0`, `ax1` 等命名。

**⚠️ 复杂度标注：**
- `axc.color` 和 layer 的 `style.cmap`/`style.vmin`/`style.vmax` 有重叠和冲突
- `axc.label` 的 `xlabel` 对应水平 colorbar、`ylabel` 对应垂直 colorbar，不直观

---

## 七、Style 系统

### 7.1 引用方式

```yaml
style:
  - a4paper_2x1              # 纸张/布局 family
  - rectcmap                 # 变体：rect / rectcmap / ternary 等
```

### 7.2 可用 Style Card 列表

| Family | 变体 | 说明 |
|--------|------|------|
| `a4paper_2x1` | `rect`, `rectcmap`, `ternary`, `ternary_cmap`, `rect5x1`, `dynesty_runplot` | A4 纸 2 列 1 行 |
| `a4paper_4x1` | `rect`, `rectcmap`, `ternary`, `ternary_cmap` | A4 纸 4 列 1 行 |
| `a4paper_1x1` | `ternary` | A4 纸单图 |
| `gambit_2x1` | `rectcmap`, `ternary`, `ternary_cmap` | GAMBIT 协作组风格 |
| `gambit_1x1` | `ternary` | GAMBIT 单图 |

每个 card 是 JSON 文件，包含 `Frame`（布局/字体/刻度）和 `Style`（method 默认参数）两大块。

---

## 八、表达式系统

在 `expr:`, `filter:`, `add_column.expr` 等处可用的函数：

### 8.1 标准数学

| 函数 | 说明 |
|------|------|
| `exp`, `log` / `ln`, `sqrt`, `abs` | 基本函数 |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` | 三角 |
| `sinh`, `cosh`, `tanh` 等 | 双曲 |
| `Min`, `Max` | 最值 |
| `pi`, `E`, `Inf` | 常数 |

### 8.2 特殊函数

| 函数 | 说明 |
|------|------|
| `Gauss(x, mean, sigma)` | 非归一化高斯 |
| `Normal(x, mean, sigma)` | 归一化高斯 |
| `LogGauss(x, mean, sigma)` | 对数空间高斯 |
| `Heaviside(x)` | 阶跃函数 |

### 8.3 命名空间

所有 NumPy 函数通过 `np.xxx` 访问，DataFrame 的列名作为变量直接可用。

---

## 九、复杂度分析与简化建议（重封装参考）

### 高复杂度

1. **coordinates 结构在多处重复且语义不同**
   - Transform 中的 coordinates 需要 `expr` + `name` + `lim` + `scale`
   - Layer 中的 coordinates 只需要 `expr`
   - 可简化：Layer 的 coordinates 用 `{x: col_name}` 简写，transform 的 coordinates 用完整 dict

2. **`make_density_core` 参数过多**
   - `method` x `coordinates` x `grid`/`bridson` x `normalize` x `refinement`(15+ 子参数) x `_mesh_debug`
   - 大多数用户只需要：`method`, `bins`, `coordinates`, `normalize`
   - **已设计解决方案：** 见第十节 `posterior_density` 新 transform

3. **`make_interp_2d` 的 grid 参数三种格式**
   - `grid: 500` / `grid: [500, 300]` / `grid: {nx: 500, ny: 300}`
   - 建议统一为一种

4. **style 和 frame 的 color 配置有冲突**
   - `frame.axc.color.cmap` vs `layer.style.cmap`
   - `frame.axc.color.vmin/vmax` vs `layer.style.vmin/vmax`

### 中等复杂度

5. **`keep_columns` / `drop_columns` 三种输入格式**
   - string / list / dict — 建议只保留 list

6. **`jpcontourf` vs `make_density_core -> make_interp_2d -> contourf` 两条路径**
   - 功能重叠，用户不知道选哪个
   - 建议：文档明确推荐路径，或合并为一个高层 method

7. **`profile` 和 `make_density_core` 的 coordinates 结构完全一致**
   - 但 profile 多了 `objective`、`grid_points`，density_core 多了 `weight`
   - 可以共享 coordinate 定义模板

### 合理复杂度

8. **DataSet 基本配置** — 简洁明了
9. **Layer 基本结构** — source -> transform -> method -> coordinates -> style 管线清晰
10. **Style card 系统** — 两级引用（family + variant）足够灵活
11. **HPD contour** — 通过 `contour_mode: posterior_hpd` 特殊 style 实现，干净
12. **`to_csv` / `to_parquet`** — 简洁的旁路导出

---

## 十、重封装设计：`make_density_core` 简化 + `posterior_density` 合并管线

### 10.1 Before vs After 对比

**Before（现状 — 22 行，coordinates 写两遍，lim 写四遍）：**
```yaml
- make_density_core:
    method: bridson
    coordinates:
      x: {expr: xx, name: x, lim: [0, 5], scale: linear}
      y: {expr: yy, name: y, lim: [0, 5], scale: linear}
      weight: {expr: "exp(LogL)", name: weight}
    bridson:
      bins: 200
    normalize: true
- make_interp_2d:
    method: natural_neighbor
    as_density: true
    normalize: true
    coordinates:
      x: {expr: x, name: x, lim: [0, 5], scale: linear}
      y: {expr: y, name: y, lim: [0, 5], scale: linear}
      z: {expr: weight, name: density}
    grid: 500
    nan_policy: strict
```

**After — 简化的 make_density_core（4 行核心）：**
```yaml
- make_density_core:
    x: {expr: xx, lim: [0, 5]}
    y: {expr: yy, lim: [0, 5]}
    weight: {expr: "exp(LogL)"}
    bins: 120
```

**After — 合并管线 posterior_density（5 行全搞定）：**
```yaml
- posterior_density:
    x: {expr: xx, lim: [0, 5]}
    y: {expr: yy, lim: [0, 5]}
    weight: {expr: "exp(LogL)"}
    bins: 120
    grid: 300
```

---

### 10.2 `make_density_core` 新接口规格

#### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `{expr: ..., lim: [lo, hi]}` | x 坐标。`lim` 可省略（自动推断） |
| `y` | `{expr: ..., lim: [lo, hi]}` | y 坐标。`lim` 可省略（自动推断） |
| `weight` | `{expr: ...}` | 权重表达式 |

#### 常用可选参数（顶层）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bins` | `64` | 分辨率（bridson 的 radius = 1/bins；grid/kde 的网格数） |
| `method` | `bridson` | `bridson` / `grid` / `kde` |
| `normalize` | `true` | 输出质量归一化 |
| `seed` | `null` | 随机种子（可复现） |

#### 高级可选参数（极少需要）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `scale` | `linear` | 轴缩放（或在 x/y dict 中写 `scale: log`） |
| `output` | `{x: "x", y: "y", weight: "weight"}` | 重命名输出列 |
| `bw_method` | `scott` | KDE 专用：带宽方法 |
| `refinement` | `null` | 自适应网格精修（高级用户） |
| `diagnostics` | `true` | 是否输出诊断日志 |

#### 坐标 dict 完整字段

```yaml
x:
  expr: "column_or_expression"    # 必需
  lim: [lo, hi]                   # 可选，不提供则自动推断
  scale: linear                   # 可选，默认 linear
  name: output_col_name           # 可选，默认等于 key（x/y/weight）
```

#### 设计原则

1. **`coordinates:` 嵌套层取消** — x/y/weight 直接放在 cfg 顶层
2. **`bridson: {bins: ...}` 嵌套取消** — `bins` 直接放顶层，对所有 method 生效
3. **`name` 默认等于 key** — 不再需要写 `name: x`
4. **`scale: linear` 不用写** — 只有 log 才需要显式声明
5. **`normalize: true` 不用写** — 默认就是 true
6. **`method: bridson` 不用写** — bridson 是默认
7. **完全向后兼容** — 旧的 `coordinates: {...}` 写法继续生效

---

### 10.3 `make_interp_2d` 对应简化

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `x` | `{expr: "x"}` | x 列（默认来自上游的 "x" 列） |
| `y` | `{expr: "y"}` | y 列（默认来自上游的 "y" 列） |
| `z` | `{expr: "weight"}` | z 列（默认来自上游的 "weight" 列） |
| `xlim` / `ylim` | 自动推断 | 或在 x/y dict 中写 lim |
| `grid` | `256` | 插值输出网格大小（int 或 [nx, ny]） |
| `method` | `natural_neighbor` | 插值方法 |
| `as_density` | `true` | 把 z 当保守质量除以面积 |
| `normalize` | `true` | 归一化积分为 1 |
| `output_z` | `density` | 输出 z 列名 |
| `nan_policy` | `strict` | hull 外为 NaN |

**典型简写：**
```yaml
- make_interp_2d:
    z: {expr: weight}
    xlim: [0, 5]
    ylim: [0, 5]
    grid: 300
```
x/y 不写时默认读 "x"/"y" 列。as_density、normalize、nan_policy 都用默认值。

---

### 10.4 `posterior_density` 合并管线

新增 transform 类型，内部按 method 自动选择执行路径。

#### 通用接口

```yaml
- posterior_density:
    method: voronoi                    # voronoi | adaptive | kde | grid
    x: {expr: xx, lim: [0, 5]}
    y: {expr: yy, lim: [0, 5]}
    weight: {expr: "exp(LogL)"}
    bins: 120                          # 支撑点 / 分箱 分辨率
    grid: 300                          # 输出网格（仅 voronoi/adaptive 需要）
```

**输出：** DataFrame 三列 `(x, y, density)`，可直接接 `pcolormesh`。

#### 通用顶层参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `method` | `voronoi` | `voronoi` / `adaptive` / `kde` / `grid` |
| `x` | — | **必需**，`{expr: ..., lim: [...]}` |
| `y` | — | **必需**，`{expr: ..., lim: [...]}` |
| `weight` | — | **必需**，`{expr: ...}` |
| `bins` | `64` | 支撑点密度 / 网格分箱数 |
| `grid` | `256` | 插值输出网格大小（仅 voronoi/adaptive） |
| `normalize` | `true` | 归一化 |
| `seed` | `null` | 随机种子 |
| `output` | `density` | 输出 z 列名 |

---

#### method: voronoi（默认）

Bridson 准均匀采样 -> Voronoi 最近邻归属 -> 质量聚合 -> natural_neighbor 插值到规则网格。

**最简写法：**
```yaml
- posterior_density:
    x: {expr: xx, lim: [0, 5]}
    y: {expr: yy, lim: [0, 5]}
    weight: {expr: "exp(LogL)"}
    bins: 120
    grid: 300
```

**可调参数：**

| voronoi 子参数 | 默认值 | 说明 |
|---------------|--------|------|
| `k` | `30` | Bridson disk sampling 每轮候选数 |

```yaml
voronoi:
  k: 30
```

**内部执行链：**
`bridson_points(bins) -> assign_nearest_ownership -> aggregate_masses -> normalize -> natural_neighbor_interp(grid) -> as_density -> normalize`

---

#### method: adaptive

在 voronoi 基础上增加自适应网格精修：按 stress 指标 split 高应力区域、merge 过近生成点、reseed 空生成点。适合多峰 / 高曲率后验。

**典型写法：**
```yaml
- posterior_density:
    method: adaptive
    x: {expr: xx, lim: [0, 5]}
    y: {expr: yy, lim: [0, 5]}
    weight: {expr: "exp(LogL)"}
    bins: 120
    grid: 300
```

**常用可调参数（全部有默认值）：**

| adaptive 子参数 | 默认值 | 说明 |
|----------------|--------|------|
| `iterations` | `2` | 精修迭代轮数 |
| `split` | `true` | 允许 split |
| `merge` | `true` | 允许 merge |
| `max_generators` | `null` | 生成点数上限（null=不限） |
| `seed` | `null` | 精修随机种子 |

```yaml
adaptive:
  iterations: 2
  split: true
  merge: true
  max_generators: 64
  seed: 7
```

**完整高级参数（几乎不需要调）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | `0.30` | centroid drift 步长 (0.05-1.0) |
| `eta` | `0.50` | 各向异性混合权重 (0.0-1.0) |
| `anisotropic` | `true` | 使用 Mahalanobis 各向异性归属 |
| `split_quantile` | `0.80` | 应力分位数阈值，超过触发 split |
| `merge_quantile` | `0.20` | 应力分位数阈值，低于触发 merge |
| `split_offset` | `0.45` | split 新点偏移量 |
| `min_separation` | `auto` | 最小生成点间距 |
| `merge_distance` | `auto` | merge 距离阈值 |
| `max_splits` | `null` | 每轮最大 split 数 |
| `max_merges` | `null` | 每轮最大 merge 数 |
| `min_generators` | `4` | 生成点数下限 |
| `empty_reseed` | `true` | reseed 无样本的生成点 |
| `drift_clip` | `2.0` | centroid drift 裁剪 |
| `anisotropy_cap` | `30.0` | 各向异性比上限 |
| `record_history` | `false` | 记录每轮生成点快照 |

**内部执行链：**
`bridson_points(bins) -> [iterate: ownership -> moments -> split/merge/reseed -> centroid_drift] -> normalize -> natural_neighbor_interp(grid) -> as_density -> normalize`

---

#### method: kde

高斯核密度估计。直接输出规则网格，**不经过插值步骤**。

**典型写法：**
```yaml
- posterior_density:
    method: kde
    x: {expr: xx, lim: [0, 5]}
    y: {expr: yy, lim: [0, 5]}
    weight: {expr: "exp(LogL)"}
    bins: 128
```

注意：`kde` 不需要 `grid` 参数，`bins` 直接决定输出网格分辨率。

**可调参数：**

| kde 子参数 | 默认值 | 说明 |
|-----------|--------|------|
| `bw_method` | `scott` | `scott` / `silverman` / `float` / `"0.5 * scott"` |

```yaml
kde:
  bw_method: scott
```

**内部执行链：**
`scipy.gaussian_kde(bw_method) -> evaluate_on_grid(bins) -> normalize`

---

#### method: grid

最简单的直方图分箱。直接输出规则网格，**不经过插值步骤**。速度最快但分辨率受限于样本数。

**典型写法：**
```yaml
- posterior_density:
    method: grid
    x: {expr: xx, lim: [0, 5]}
    y: {expr: yy, lim: [0, 5]}
    weight: {expr: "exp(LogL)"}
    bins: 150
```

无额外子参数。`bins` 直接决定输出网格。

**内部执行链：**
`histogram2d(bins) -> normalize`

---

#### method 选择指南

| 场景 | 推荐 method | 原因 |
|------|------------|------|
| 常规后验密度 | `voronoi` | 质量保守、无平滑偏差 |
| 多峰 / 高曲率后验 | `adaptive` | 自动加密高应力区 |
| 快速预览 | `grid` | 最快，一步到位 |
| 需要平滑处理 | `kde` | KDE 天然平滑 |
| 对比 KDE 带宽敏感性 | `kde` + `bw_method` | 调带宽观察变化 |

---

### 10.5 内部 config 映射逻辑

```
posterior_density config
    |
    +-- method == "voronoi" ?
    |     -> density_cell(method="bridson", refinement=None)
    |     -> make_interp_2d(grid=grid, as_density=True)
    |
    +-- method == "adaptive" ?
    |     -> density_cell(method="bridson", refinement={...from adaptive dict...})
    |     -> make_interp_2d(grid=grid, as_density=True)
    |
    +-- method == "kde" ?
    |     -> density_cell(method="kde", bw_method=kde.bw_method)
    |     -> mass / cell_area -> density (no interp_2d)
    |
    +-- method == "grid" ?
          -> density_cell(method="grid")
          -> mass / cell_area -> density (no interp_2d)
```

---

### 10.6 向后兼容保证

- 旧的 `coordinates: {x: {expr: ..., name: ..., lim: ..., scale: ...}}` 写法 **不做任何改动**
- 旧的 `make_density_core` + `make_interp_2d` 分步写法继续生效
- 新的 `posterior_density` 是纯粹的 **语法糖**，解析后走完全相同的代码路径

### 10.7 实现状态

已落地：

- `make_density_core` 支持 compact 顶层 `x` / `y` / `weight` 配置。
- `make_density_core` 默认 `method: bridson`，默认 `bins: 64`；顶层 `bins` 同时映射到 `grid` / `kde` 网格和 `bridson.bin`。
- `make_density_core.output` 支持 `{x, y, weight}` 输出列重命名，旧 `coordinates.*.name` 仍然有效。
- `make_interp_2d` 已支持顶层 `xlim` / `ylim` 和 `output_z`，用于配合简写管线。
- 新增 `posterior_density` transform，DataSet 级别和 Layer/runtime 级别都可用。
- `posterior_density.method` 支持 `voronoi`、`adaptive`、`kde`、`grid`：
  - `voronoi` / `adaptive` 内部执行 `make_density_core(method=bridson)` 后接 `make_interp_2d(as_density=true, normalize=true)`。
  - `grid` / `kde` 内部执行 `make_density_core(method=grid|kde)`，再用支撑单元面积把 mass 转成 density。
- Polars pushdown 遇到 `posterior_density` 会回退到 pandas transform path。

仍保持兼容：

- 旧的 `method: grid`、`method: kde`、`bridson: {bin: ...}`、`grid: {bins: ...}` 写法继续生效。
- 旧的 `type: make_density_core` / `type: make_interp_2d` 写法继续生效。
- `make_interp_2d` 的通用默认语义未全局改成 density 模式；`posterior_density` 会显式传入 `as_density: true`、`normalize: true`。

---

## 十一、封装层（Encapsulated Figure Types）

### 11.1 设计目标

在底层 method + transform 之上定义 **高层 figure type**。用户通过极少参数（x, y, weight 等）定义一张完整的图，系统自动生成 style、frame、transforms、layers 全套配置。允许用户通过 `extra_layers` 在默认渲染之上叠加自定义 layer。

**与低层 YAML 的关系：**
- 封装层是纯粹的 **语法糖**，展开后得到标准的 layers + frame + style 配置
- 无 `type:` 字段的 figure 走现有手动 layer 路径，完全不受影响
- 用户可混用：同一个 YAML 中既有封装层 figure 又有手动 figure

---

### 11.2 通用结构

```yaml
Figures:
  - name: my_figure
    type: posterior_2d              # 封装类型（见下方各类型定义）
    data: my_samples                # 数据源名称（或列表 → concat）
    x: {expr: xx, lim: [0, 5], label: "$x$"}
    y: {expr: yy, lim: [0, 5], label: "$y$"}
    # ... type-specific 参数 ...

    # 可选：覆盖自动选择的 style card
    style_card: [a4paper_2x1, rectcmap]

    # 可选：在自动生成的 layer 之后追加自定义 layer
    extra_layers:
      - method: scatter
        coordinates: {x: {expr: xx}, y: {expr: yy}}
        style: {s: 1, alpha: 0.3, color: gray, zorder: 30}
```

#### 通用坐标 dict（扩展版）

封装层的坐标 dict 在原有基础上增加 `label` 字段，用于自动设置轴标签：

```yaml
x:
  expr: "column_or_expression"    # 必需
  lim: [lo, hi]                   # 可选，不提供则自动推断
  scale: linear                   # 可选，默认 linear
  label: "$m_A$ [GeV]"            # 可选，默认用 expr 的值
```

#### 通用参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|:----:|------|
| `type` | string | **是** | 封装类型名 |
| `data` | string 或 list | **是** | DataSet name（列表 → concat） |
| `name` | string | 否 | figure 名称（默认自动生成） |
| `enable` | bool | 否 | 是否启用（默认 true） |
| `style_card` | list | 否 | 覆盖自动选择的 style card |
| `extra_layers` | list | 否 | 追加自定义 layer（标准 layer 格式） |

---

### 11.3 type: posterior_2d

二维后验概率密度图。详细设计见 **第十二章**。

---

### 11.4 type: scatter_2d

**用途：** 二维散点图 — 可选颜色维度 + colorbar。

**最简写法（无颜色，5 行）：**
```yaml
- name: scatter_xy
  type: scatter_2d
  data: my_samples
  x: {expr: xx, label: "$x$"}
  y: {expr: yy, label: "$y$"}
```

**带颜色维度（6 行）：**
```yaml
- name: scatter_colored
  type: scatter_2d
  data: my_samples
  x: {expr: xx, lim: [0, 5], label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  color: {expr: LogL, label: "$\\log\\mathcal{L}$"}
```

**完整写法：**
```yaml
- name: scatter_colored
  type: scatter_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0, 5], scale: log, label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  color: {expr: LogL, label: "$\\log\\mathcal{L}$"}

  sortby: LogL                     # 排序表达式（控制 z-order 渲染顺序）

  scatter:                         # 散点样式
    marker: "."
    s: 2
    alpha: 1.0

  colorbar:                        # colorbar 设置（仅有 color 时生效）
    cmap: viridis
    vmin: auto
    vmax: auto
```

**自动生成规则：**

| 自动生成项 | 有 color | 无 color |
|-----------|---------|---------|
| style_card | `[a4paper_2x1, rectcmap]` | `[a4paper_2x1, rect]` |
| transform | `sortby` (如配置) | `sortby` (如配置) |
| Layer 1 | scatter with c=color | scatter without c |
| colorbar | 有 | 无 |

---

### 11.5 type: profile_2d

二维剖面图。用户做三个正交选择：归约方法（bridson/grid）、渲染方式（NN 插值图/cell 图）、可信区间（1σ/2σ）。详细设计见 **第十三章**。

**最简写法（6 行，默认 bridson + NN 插值）：**
```yaml
- name: XY_profLogL
  type: profile_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0.1, 5], scale: log, label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  z: {expr: LogL, label: "$\\log\\mathcal{L}$"}
```

**加上 credible region（9 行）：**
```yaml
- name: XY_profLogL
  type: profile_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0.1, 5], scale: log, label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  z: {expr: LogL, label: "$\\log\\mathcal{L}$"}
  colorbar: {cmap: jarvis_rainbow2_r, vmin: -50, vmax: 0}
  credible_region: {sigma: [1, 2]}
```

---

### 11.6 type: posterior_1d

**用途：** 一维边缘化后验分布 — 加权直方图或 KDE 曲线。

**最简写法（5 行）：**
```yaml
- name: marginal_x
  type: posterior_1d
  data: my_samples
  x: {expr: xx, lim: [0, 5], label: "$x$"}
  weight: {expr: "exp(LogL)"}
```

**完整写法：**
```yaml
- name: marginal_x
  type: posterior_1d
  data: my_samples
  x: {expr: xx, lim: [0, 5], label: "$x$"}
  weight: {expr: "exp(LogL)"}

  histogram:
    bins: 50
    density: true                  # 归一化为概率密度
    histtype: stepfilled
    color: steelblue
    alpha: 0.6
    edgecolor: black

  # 可选：叠加 KDE 平滑曲线
  kde:
    bw_method: scott
    color: red
    linewidth: 1.5
```

**自动生成规则：**

| 自动生成项 | 默认值 |
|-----------|--------|
| style_card | `[a4paper_2x1, rect]` |
| frame.ax.labels.y | `"posterior PDF"` |
| Layer 1 | hist（加权直方图） |
| Layer 2 | plot（KDE 曲线，仅当 kde 配置存在时） |

---

### 11.7 extra_layers 机制

`extra_layers` 中的每个 layer 使用 **标准 layer 格式**（与当前低层 YAML 完全相同），附加在自动生成的 layer 之后。

```yaml
extra_layers:
  - method: scatter                 # 标准 layer 格式
    data: other_source              # 可选：不同数据源（不写则用 figure 级 data）
    coordinates:
      x: {expr: xx}
      y: {expr: yy}
      c: {expr: chi2}
    style:
      s: 1
      alpha: 0.3
      cmap: coolwarm
      zorder: 30

  - method: plot                    # 第二个额外 layer
    data: reference_curve
    coordinates:
      x: {expr: x_ref}
      y: {expr: y_ref}
    style:
      color: red
      linestyle: "--"
      linewidth: 1.5
      zorder: 40
```

**规则：**

1. extra_layers 的 layer 格式与当前 `layers[]` 完全一致，无需学新语法
2. `data` 不写时，继承 figure 级的 `data` 源
3. 渲染顺序：auto-generated layers → extra_layers（按列表顺序）
4. extra_layers 可以使用 auto-generated layers 的 `share_data` 缓存

---

### 11.8 展开流程（内部实现）

```
Figure YAML with type: xxx
    |
    +-- has "type" field?
    |     |
    |     yes → FigureTypeExpander.expand(config)
    |     |       |
    |     |       +-- resolve type → get expansion template
    |     |       +-- build frame config from x/y/label/lim/scale
    |     |       +-- build auto-generated layers from type-specific params
    |     |       +-- append extra_layers
    |     |       +-- auto-select style_card (if not overridden)
    |     |       +-- return expanded standard figure config
    |     |
    |     no → pass through (standard layer-based figure)
    |
    +-- Figure.from_dict(expanded_config)
        +-- ... standard rendering pipeline ...
```

**关键设计：展开发生在 Figure 解析之前，展开后的配置与手写的低层 YAML 完全等价。Figure 类本身无需任何改动。**

---

### 11.9 典型用例对比

#### 场景 A：EggBox 后验密度（adaptive refinement）

**Before（92 行）：**
```yaml
- name: EggBox_posterior_pdf_adaptive
  enable: true
  style: [a4paper_4x1, rectcmap]
  frame:
    ax:
      labels: {x: "$x$", y: "$y$"}
      xlim: [0, 5]
      ylim: [0, 5]
      xscale: linear
      yscale: linear
    axc:
      label: {xlabel: "adaptive posterior PDF density"}
      color: {cmap: jarvis_rainbow2_r, scale: linear, vmin: 0.0, vmax: 0.8}
  layers:
    # ... 70+ lines of transforms, layers, coordinates, styles ...
```

**After（11 行）：**
```yaml
- name: EggBox_posterior_pdf_adaptive
  type: posterior_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0, 5], label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  weight: {expr: "exp(LogL)"}
  density:
    method: adaptive
    bins: 120
    grid: 300
  colorbar: {label: "adaptive posterior PDF density", cmap: jarvis_rainbow2_r, vmax: 0.8}
```

#### 场景 B：简单散点图

**Before（40 行）：**
```yaml
- name: Scatter_XY
  style: [a4paper_2x1, rectcmap]
  frame:
    ax:
      labels: {x: "$x$", y: "$y$"}
      xlim: [0.1, 5]
      ylim: [0, 5]
      xscale: log
    axc:
      label: {ylabel: "posterior PDF density"}
      color: {cmap: jarvis_rainbow2_r, vmin: 0, vmax: 0.8}
  layers:
    # ... 20+ lines ...
```

**After（8 行）：**
```yaml
- name: Scatter_XY
  type: scatter_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0.1, 5], scale: log, label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  color: {expr: LogL, label: "posterior PDF density"}
  sortby: LogL
  colorbar: {cmap: jarvis_rainbow2_r, vmin: 0, vmax: 0.8}
```

#### 场景 C：后验密度 + 采样点叠加

**After（12 行）：**
```yaml
- name: posterior_with_samples
  type: posterior_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0, 5], label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  weight: {expr: "exp(LogL)"}
  density: {method: voronoi, bins: 120, grid: 300}
  colorbar: {cmap: jarvis_rainbow2_r, vmax: 0.8}
  extra_layers:
    - method: scatter
      coordinates: {x: {expr: xx}, y: {expr: yy}}
      style: {s: 0.5, color: gray, alpha: 0.2, zorder: 5}
```

---

## 十二、posterior_2d 封装详细设计

### 12.1 概述

`type: posterior_2d` 是最核心的封装类型，将加权后验采样数据渲染为二维概率密度热力图。

**自动生成的完整 layer 栈：**

```
Layer 1: pcolormesh (密度热力图)
    ← transform: posterior_density
    → share_data 缓存

Layer 2: contour (HPD 等高线)
    ← source: share_data 缓存

+ extra_layers (用户自定义)
```

---

### 12.2 通用接口

```yaml
- name: figure_name
  type: posterior_2d
  data: source_name                    # 必需
  x: {expr: ..., lim: [...]}          # 必需
  y: {expr: ..., lim: [...]}          # 必需
  weight: {expr: ...}                  # 必需
  density: { ... }                     # 可选：密度重建参数
  colorbar: { ... }                    # 可选：colorbar 设置
  hpd: { ... }                         # 可选：HPD 等高线设置
  extra_layers: [ ... ]                # 可选：追加自定义 layer
```

#### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | string 或 list | DataSet name（列表 → concat） |
| `x` | coord dict | x 坐标 `{expr:, lim:, scale:, label:}` |
| `y` | coord dict | y 坐标 `{expr:, lim:, scale:, label:}` |
| `weight` | coord dict | 权重 `{expr:}` |

#### 可选顶层参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `style_card` | `[a4paper_2x1, rectcmap]` | 覆盖自动选择的 style card |
| `density` | `{}` | 密度重建参数（见 12.3） |
| `colorbar` | `{}` | colorbar 设置（见 12.4） |
| `hpd` | `{}` | HPD 等高线设置（见 12.5） |
| `extra_layers` | `[]` | 追加自定义 layer |

---

### 12.3 density 参数：按 method 分档

`density` 字段控制后验密度重建算法。对应底层 `posterior_density` transform。

```yaml
density:
  method: voronoi              # voronoi | adaptive | kde | grid
  bins: 120                    # 分辨率
  grid: 300                    # 插值网格（仅 voronoi/adaptive）
  seed: null                   # 随机种子
  # + method 子参数 dict
```

---

#### method: voronoi（默认）

Bridson 准均匀采样 → Voronoi 质量聚合 → natural_neighbor 插值。质量保守，无平滑偏差。

```yaml
density:
  method: voronoi
  bins: 120                    # Bridson radius = 1/bins
  grid: 300                    # 插值输出 grid x grid
  seed: null                   # Bridson 随机种子
  voronoi:                     # 极少需要改
    k: 30                      # Bridson 每轮候选点数
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bins` | `64` | Bridson 分辨率（越大支撑点越密） |
| `grid` | `256` | 插值输出网格边长 |
| `seed` | `null` | 随机种子 |
| `voronoi.k` | `30` | Bridson 候选数 |

**展开的 transform 链：**
```yaml
- posterior_density:
    method: voronoi
    x: {expr: ..., lim: [...]}
    y: {expr: ..., lim: [...]}
    weight: {expr: ...}
    bins: 120
    grid: 300
```

**展开的 Layer 1（pcolormesh）：**
```yaml
- name: _density
  data:
    - source: {data}
      transform: [posterior_density: {...}]
  share_data: _pd_cache
  axes: ax
  method: pcolormesh
  coordinates: {x: {expr: x}, y: {expr: y}, z: {expr: density}}
  style: {edgecolor: none, linewidth: 0}
  colorbar: axc
```

---

#### method: adaptive

在 voronoi 基础上增加自适应网格精修。按 stress 指标自动 split 高密度梯度区域、merge 过近生成点、reseed 空生成点。**适合多峰或高曲率后验**。

```yaml
density:
  method: adaptive
  bins: 120
  grid: 300
  adaptive:                    # 常用参数（全部有默认值）
    iterations: 2
    split: true
    merge: true
    max_generators: 64
    seed: 7
```

| adaptive 子参数 | 默认值 | 说明 |
|----------------|--------|------|
| `iterations` | `2` | 精修迭代轮数 |
| `split` | `true` | 允许 split 高应力生成点 |
| `merge` | `true` | 允许 merge 过近生成点 |
| `max_generators` | `null` | 生成点数上限 |
| `seed` | `null` | 精修随机种子 |

**完整高级参数（几乎不需要调）：**

| 参数 | 默认 | 参数 | 默认 |
|------|------|------|------|
| `alpha` | `0.30` | `split_quantile` | `0.80` |
| `eta` | `0.50` | `merge_quantile` | `0.20` |
| `anisotropic` | `true` | `split_offset` | `0.45` |
| `min_separation` | `auto` | `merge_distance` | `auto` |
| `max_splits` | `null` | `max_merges` | `null` |
| `min_generators` | `4` | `empty_reseed` | `true` |
| `drift_clip` | `2.0` | `anisotropy_cap` | `30.0` |

**展开的 transform 链：** 同 voronoi，但 `posterior_density` 内部传入 `refinement` config。

---

#### method: kde

高斯核密度估计。直接输出规则网格，**不经过 Voronoi 或插值步骤**。

```yaml
density:
  method: kde
  bins: 128                    # 输出网格 = bins x bins
  kde:
    bw_method: scott           # scott | silverman | float | "0.5 * scott"
```

| kde 子参数 | 默认值 | 说明 |
|-----------|--------|------|
| `bw_method` | `scott` | 带宽选择方法 |

注意：`kde` 不使用 `grid` 参数，`bins` 同时决定 KDE 评估网格和输出网格。

**展开的 transform 链：**
```yaml
- posterior_density:
    method: kde
    x: {expr: ..., lim: [...]}
    y: {expr: ..., lim: [...]}
    weight: {expr: ...}
    bins: 128
    kde: {bw_method: scott}
```

---

#### method: grid

最简单的直方图分箱。速度最快但分辨率受限于样本密度。

```yaml
density:
  method: grid
  bins: 150                    # 输出网格 = bins x bins
```

无额外子参数。

---

#### method 选择指南

| 场景 | 推荐 method | 原因 |
|------|------------|------|
| 常规后验密度 | `voronoi`（默认） | 质量保守、无平滑偏差 |
| 多峰 / 窄脊 / 高曲率 | `adaptive` | 自动加密高梯度区 |
| 快速预览 | `grid` | 最快，一步到位 |
| 需要平滑 / 对比带宽 | `kde` | KDE 天然平滑 |

---

### 12.4 colorbar 参数

```yaml
colorbar:
  label: "posterior density"       # colorbar 标签
  cmap: jarvis_rainbow2_r          # colormap 名称
  scale: linear                    # linear | log
  vmin: 0.0                        # 颜色下限
  vmax: 0.8                        # 颜色上限（auto = 数据最大值）
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `label` | `"density"` | colorbar 轴标签 |
| `cmap` | `jarvis_rainbow2_r` | colormap |
| `scale` | `linear` | 颜色比例尺 |
| `vmin` | `0.0` | 颜色下限 |
| `vmax` | `auto` | 颜色上限 |

**展开到 frame.axc：**
```yaml
frame:
  axc:
    label: {xlabel: "posterior density"}    # 水平 colorbar 用 xlabel
    color:
      cmap: jarvis_rainbow2_r
      scale: linear
      vmin: 0.0
      vmax: 0.8
```

---

### 12.5 hpd 参数

HPD (Highest Posterior Density) 等高线叠加在密度热力图上。

```yaml
hpd:
  masses: [0.6827, 0.9545]            # 概率质量（1sigma, 2sigma）
  labels: ["$1\\sigma$", "$2\\sigma$"]
  colors: [black, white]
  linestyles: [solid, solid]
  linewidths: [0.2, 0.2]
  zorder: 20
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `masses` | `[0.6827, 0.9545]` | HPD 概率质量列表 |
| `labels` | `["$1\\sigma$", "$2\\sigma$"]` | 等高线标签 |
| `colors` | `[black, white]` | 等高线颜色 |
| `linestyles` | `[solid, solid]` | 线型 |
| `linewidths` | `[0.2, 0.2]` | 线宽 |
| `zorder` | `20` | 渲染层级 |

设置 `hpd: false` 可完全禁用 HPD 等高线层。

**展开的 Layer 2（contour）：**
```yaml
- name: _hpd
  data:
    - source: _pd_cache            # 复用 Layer 1 的 share_data
  axes: ax
  method: contour
  coordinates: {x: {expr: x}, y: {expr: y}, z: {expr: density}}
  style:
    contour_mode: posterior_hpd
    masses: [0.6827, 0.9545]
    labels: ["$1\\sigma$", "$2\\sigma$"]
    colors: [black, white]
    linestyles: [solid, solid]
    linewidths: [0.2, 0.2]
    zorder: 20
```

---

### 12.6 自动生成规则总表

| 自动生成项 | 来源 | 默认值 |
|-----------|------|--------|
| style_card | `style_card` 或自动 | `[a4paper_2x1, rectcmap]` |
| frame.ax.xlim | `x.lim` | 自动推断 |
| frame.ax.ylim | `y.lim` | 自动推断 |
| frame.ax.xscale | `x.scale` | `linear` |
| frame.ax.yscale | `y.scale` | `linear` |
| frame.ax.labels.x | `x.label` | `x.expr` |
| frame.ax.labels.y | `y.label` | `y.expr` |
| frame.axc | `colorbar` | 见 12.4 |
| Layer 1: pcolormesh | `density` | voronoi, bins=64, grid=256 |
| Layer 2: contour | `hpd` | 1sigma + 2sigma |
| Layer 3+: extra | `extra_layers` | 无 |

---

### 12.7 完整展开示例

**封装写法（11 行）：**
```yaml
- name: EggBox_posterior
  type: posterior_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0, 5], label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  weight: {expr: "exp(LogL)"}
  density:
    method: adaptive
    bins: 120
    grid: 300
  colorbar: {label: "adaptive posterior PDF density", cmap: jarvis_rainbow2_r, vmax: 0.8}
```

**等价展开（完整低层 YAML）：**
```yaml
- name: EggBox_posterior
  style: [a4paper_2x1, rectcmap]
  frame:
    ax:
      labels: {x: "$x$", y: "$y$"}
      xlim: [0, 5]
      ylim: [0, 5]
      xscale: linear
      yscale: linear
    axc:
      label: {xlabel: "adaptive posterior PDF density"}
      color: {cmap: jarvis_rainbow2_r, scale: linear, vmin: 0.0, vmax: 0.8}
  layers:
    - name: _density
      data:
        - source: [df_samples_0, df_samples_1]
          transform:
            - posterior_density:
                method: adaptive
                x: {expr: xx, lim: [0, 5]}
                y: {expr: yy, lim: [0, 5]}
                weight: {expr: "exp(LogL)"}
                bins: 120
                grid: 300
      share_data: _pd_cache
      axes: ax
      method: pcolormesh
      coordinates: {x: {expr: x}, y: {expr: y}, z: {expr: density}}
      style: {edgecolor: none, linewidth: 0}
      colorbar: axc
    - name: _hpd
      data:
        - source: _pd_cache
      axes: ax
      method: contour
      coordinates: {x: {expr: x}, y: {expr: y}, z: {expr: density}}
      style:
        contour_mode: posterior_hpd
        masses: [0.6827, 0.9545]
        labels: ["$1\\sigma$", "$2\\sigma$"]
        colors: [black, white]
        linewidths: [0.2, 0.2]
        zorder: 20
```

---

### 12.8 实现状态

已落地：

- `type: posterior_2d` figure 会在 Figure 解析前展开为标准 `style` / `frame` / `layers` 配置。
- `density.method` 支持四种方法：`voronoi`、`adaptive`、`kde`、`grid`。
- 自动生成的密度层使用底层 `posterior_density` transform，并通过 `share_data` 缓存供 HPD 层复用。
- 默认生成 `pcolormesh` 密度热力图和 `contour_mode: posterior_hpd` HPD 等高线层。
- `hpd: false` 或 `hpd: {enabled: false}` 可禁用 HPD 层。
- `colorbar` 会展开到 `frame.axc.label` 和 `frame.axc.color`，支持 `label`、`cmap`、`scale`、`vmin`、`vmax`。
- `density.output` 可重命名密度列，自动同步 pcolormesh / HPD contour 的 `z` 表达式。
- `extra_layers` 支持标准 layer 格式；未显式写 `data` 时继承 figure 级 `data`。
- 展开发生在列需求规划之前，因此 HDF5/Parquet column pruning 能看到展开后的 transform 和 layer 坐标需求。

实现边界：

- `posterior_2d` 是语法糖，展开后继续使用现有 `posterior_density`、`pcolormesh`、`contour`、colorbar 和 HPD runtime。
- `profile_2d` 的落地状态见第十三章。

---

## 十三、profile_2d 封装详细设计

### 13.1 概述

`type: profile_2d` 将大量采样数据归约为二维标量场，渲染为热力图。

用户只需做 **三个正交选择**：

```
① 归约方法（method）     bridson（默认） | grid
② 渲染方式（interp）     true → NN 插值图（默认） | false → cell 图
③ 可信区间（credible_region）  是否叠加 1σ/2σ 等高线
```

**组合矩阵：**

| method | interp | 底层管线 | 渲染方式 |
|--------|--------|---------|---------|
| `bridson` | `true` | bridson profile → Natural Neighbor → pcolormesh | **推荐默认**，C¹ 连续平滑，支持 contour |
| `bridson` | `false` | bridson profile → voronoi | cell 着色，展示 Voronoi 胞元结构 |
| `grid` | `true` | grid profile → Natural Neighbor → pcolormesh | 先网格分箱再平滑插值 |
| `grid` | `false` | grid profile → pcolormesh | 最快，直接网格分箱渲染 |

**与 `posterior_2d` 的区别：**

| | posterior_2d | profile_2d |
|-|-------------|------------|
| 输入 | x, y, weight（概率权重） | x, y, z（标量场） |
| 本质 | 概率密度重建 | 数据归约 + 可视化 |
| 归约 | 质量聚合（保守） | objective (max/min/mean) |
| 典型 z | `exp(LogL)` → density | `LogL` / `chi2` / 物理量 |
| 等高线 | HPD 概率等高线 | profile likelihood credible region |

---

### 13.2 接口

```yaml
- name: figure_name
  type: profile_2d
  data: source_name                    # 必需
  x: {expr: ..., lim: [...]}          # 必需
  y: {expr: ..., lim: [...]}          # 必需
  z: {expr: ..., label: ...}          # 必需：被归约的标量场

  # 选择 ①：归约方法
  method: bridson                      # bridson (默认) | grid
  bins: 100                            # 分辨率
  objective: max                       # max | min | mean

  # 选择 ②：渲染方式
  interp: true                         # true → NN 插值图 | false → cell 图
  grid: 500                            # NN 插值输出网格（仅 interp: true）

  # 选择 ③：可信区间
  credible_region: { ... }             # 可选：1σ/2σ 等高线

  # 其他
  colorbar: { ... }                    # 可选：colorbar 设置
  extra_layers: [ ... ]                # 可选：追加自定义 layer
```

#### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | string 或 list | DataSet name |
| `x` | coord dict | x 坐标 `{expr:, lim:, scale:, label:}` |
| `y` | coord dict | y 坐标 `{expr:, lim:, scale:, label:}` |
| `z` | coord dict | 被归约的标量 `{expr:, label:}` |

#### 选择 ① 归约参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `method` | `bridson` | `bridson`：Poisson-disk 准均匀采样归约；`grid`：规则网格分箱 |
| `bins` | `100` | bridson: radius ∝ 1/bins；grid: 网格边长 |
| `objective` | `max` | 归约方式：`max` / `min` / `mean` |
| `seed` | `null` | 随机种子（仅 bridson） |

#### 选择 ② 渲染参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `interp` | `true` | `true`：Natural Neighbor 插值 → pcolormesh；`false`：cell 直接渲染 |
| `grid` | `500` | 插值输出网格边长（仅 `interp: true`） |

#### 选择 ③ 可信区间参数

见 13.5 节。

#### 其他可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `style_card` | `[a4paper_2x1, rectcmap]` | 覆盖 style card |
| `colorbar` | `{}` | colorbar 设置（见 13.4） |
| `extra_layers` | `[]` | 追加自定义 layer |

---

### 13.3 四种组合的内部展开

#### 13.3.1 bridson + interp: true（默认，推荐）

最佳效果：Bridson 准均匀归约 → Natural Neighbor C¹ 插值 → pcolormesh。

```
Layer 0: profile(bridson) → share_data: _profile_points    (不可见)
Layer 1: make_interp_2d(natural_neighbor) → pcolormesh      (热力图)
         → share_data: _profile_grid
Layer 2: credible_region contour                            (可选)
Layer 3+: extra_layers
```

**展开的 transform 链：**

Step 1 — Bridson profile 归约：
```yaml
- profile:
    method: bridson
    bin: {bins}
    objective: {objective}
    coordinates:
      x: {expr: ..., name: x, lim: [...], scale: ...}
      y: {expr: ..., name: y, lim: [...], scale: ...}
      z: {expr: ..., name: z}
```

Step 2 — Natural Neighbor 插值：
```yaml
- make_interp_2d:
    method: natural_neighbor
    coordinates:
      x: {expr: x, name: x, lim: [...], scale: ...}
      y: {expr: y, name: y, lim: [...], scale: ...}
      z: {expr: z, name: z}
    grid: {grid}
    nan_policy: strict
```

**与 `posterior_density` 的关键区别：** profile 的插值 **不做** `as_density` 和 `normalize`。z 是原始标量值（如 LogL），不需要转换为密度。

---

#### 13.3.2 bridson + interp: false

Bridson 归约 → Voronoi 胞元着色。适合查看支撑点分布和 Voronoi 结构。

```
Layer 0: profile(bridson) → share_data: _profile_points    (不可见)
Layer 1: voronoi                                            (cell 着色)
Layer 2: credible_region contour                            (可选，需隐式 interp）
Layer 3+: extra_layers
```

**注意：** 如果同时请求了 `credible_region`，展开时会自动追加一个隐藏的 `make_interp_2d` 步骤（仅用于生成 contour 所需的规则网格），不影响主渲染层。

---

#### 13.3.3 grid + interp: true

Grid 分箱 → Natural Neighbor 平滑 → pcolormesh。先做规则网格归约再平滑，消除锯齿。

```
Layer 0: profile(grid) → share_data: _profile_points       (不可见)
Layer 1: make_interp_2d(natural_neighbor) → pcolormesh      (平滑热力图)
         → share_data: _profile_grid
Layer 2: credible_region contour                            (可选)
Layer 3+: extra_layers
```

---

#### 13.3.4 grid + interp: false

最简单最快。Grid 分箱 → 直接 pcolormesh。输出就是规则网格，无需插值。

```
Layer 0: profile(grid) → share_data: _profile_grid          (不可见)
Layer 1: pcolormesh                                          (直接渲染)
Layer 2: credible_region contour                             (可选)
Layer 3+: extra_layers
```

---

### 13.4 colorbar 参数

```yaml
colorbar:
  label: "$\\log\\mathcal{L}$"
  cmap: jarvis_rainbow2_r
  scale: linear
  vmin: -50
  vmax: 0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `label` | `z.label` 或 `z.expr` | colorbar 轴标签 |
| `cmap` | `jarvis_rainbow2_r` | colormap |
| `scale` | `linear` | 颜色比例尺 |
| `vmin` | `auto` | 颜色下限 |
| `vmax` | `auto` | 颜色上限 |

`vmin`/`vmax` 默认 auto（从数据推断），因为 profile 的 z 值范围因场景而异。

---

### 13.5 credible_region 参数（可选）

Profile likelihood 可信区间等高线。类似 `posterior_2d` 的 HPD 等高线，但物理含义不同：HPD 基于概率质量积分，credible_region 基于 profile likelihood ratio（Δχ² 阈值）。

#### 两种指定方式

**方式 A：sigma 级别（推荐，自动计算阈值）**

```yaml
credible_region:
  sigma: [1, 2]
  colors: [black, white]
  linewidths: [0.2, 0.2]
```

系统自动从数据中找到 `z_max`，然后根据 2 DOF 的 χ² 分位数计算 contour 级别：

| sigma | 置信度 | Δ(−2 ln L) | contour level（z 为 ln L 时） |
|-------|--------|-----------|-------------------------------|
| 1 | 68.27% | 2.30 | z_max − 1.15 |
| 2 | 95.45% | 6.18 | z_max − 3.09 |
| 3 | 99.73% | 11.83 | z_max − 5.915 |

**方式 B：直接指定 levels（自由指定）**

```yaml
credible_region:
  levels: [-10, -5]
  colors: [black, white]
  linewidths: [0.3, 0.3]
```

当 z 不是 log-likelihood（如 chi2、某个物理量）时，直接给出等高线数值。

#### 完整参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sigma` | — | sigma 级别列表（与 `levels` 二选一） |
| `levels` | — | 等高线数值列表（与 `sigma` 二选一） |
| `ndof` | `2` | χ² 自由度（仅 sigma 模式，2D profile 固定为 2） |
| `colors` | `[black, white]` | 等高线颜色 |
| `linestyles` | `[solid, solid]` | 线型 |
| `linewidths` | `[0.2, 0.2]` | 线宽 |
| `labels` | `["$1\\sigma$", "$2\\sigma$"]` | 等高线标签（自动匹配 sigma） |
| `zorder` | `20` | 渲染层级 |

设置 `credible_region: false` 可显式禁用。

#### sigma 模式内部逻辑

```python
from scipy.stats import chi2

z_max = grid_data["z"].max()
for s in sigma:
    cl = 1.0 - 2.0 * (1.0 - scipy.stats.norm.cdf(s))  # sigma → CL
    delta = chi2.ppf(cl, df=ndof)                       # Δ(−2lnL)
    level = z_max - delta / 2.0                         # z 阈值
```

#### 展开的 contour Layer

```yaml
- name: _credible_region
  data:
    - source: _profile_grid         # 复用插值网格 share_data
  axes: ax
  method: contour
  coordinates: {x: {expr: x}, y: {expr: y}, z: {expr: z}}
  style:
    levels: [computed_level_1, computed_level_2]    # 自动计算或直接 levels
    colors: [black, white]
    linestyles: [solid, solid]
    linewidths: [0.2, 0.2]
    zorder: 20
```

**注意：** `credible_region` 需要规则网格数据（contour 要求）。当 `interp: false` + `method: bridson` 时，展开会自动插入一个隐藏的 `make_interp_2d` 层（仅供 credible_region contour 使用），不影响 voronoi 主渲染。

---

### 13.6 自动生成规则总表

| 自动生成项 | 来源 | 默认值 |
|-----------|------|--------|
| style_card | `style_card` | `[a4paper_2x1, rectcmap]` |
| frame.ax.xlim | `x.lim` | 自动推断 |
| frame.ax.ylim | `y.lim` | 自动推断 |
| frame.ax.xscale | `x.scale` | `linear` |
| frame.ax.yscale | `y.scale` | `linear` |
| frame.ax.labels.x | `x.label` | `x.expr` |
| frame.ax.labels.y | `y.label` | `y.expr` |
| frame.axc | `colorbar` | cmap=jarvis_rainbow2_r, vmin/vmax=auto |
| transform | `method`, `bins`, `objective` | profile(bridson, 100, max) |
| interp | `interp`, `grid` | make_interp_2d(natural_neighbor, 500)，仅 interp=true |
| 渲染 Layer | `interp` | true→pcolormesh, false→voronoi(bridson)/pcolormesh(grid) |
| credible_region Layer | `credible_region` | 仅当配置存在时 |
| extra_layers | `extra_layers` | 无 |

---

### 13.7 完整展开示例

#### 示例 A：最简写法 — 全默认（6 行）

```yaml
- name: XY_profLogL
  type: profile_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0.1, 5], scale: log, label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  z: {expr: LogL, label: "$\\log\\mathcal{L}$"}
```

默认：method=bridson, interp=true, bins=100, objective=max, grid=500, 无 credible_region。

---

#### 示例 B：bridson + NN 插值 + 2σ credible region（9 行）

```yaml
- name: XY_profLogL
  type: profile_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0.1, 5], scale: log, label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  z: {expr: LogL, label: "$\\log\\mathcal{L}$"}
  bins: 100
  colorbar: {cmap: jarvis_rainbow2_r, vmin: -50, vmax: 0}
  credible_region: {sigma: [1, 2]}
```

**等价展开（对标 XY_nnprofL + credible region）：**

```yaml
- name: XY_profLogL
  style: [a4paper_2x1, rectcmap]
  frame:
    ax:
      labels: {x: "$x$", y: "$y$"}
      xlim: [0.1, 5]
      ylim: [0, 5]
      xscale: log
      yscale: linear
    axc:
      label: {xlabel: "$\\log\\mathcal{L}$"}
      color: {cmap: jarvis_rainbow2_r, scale: linear, vmin: -50, vmax: 0}
  layers:
    - name: _profile_bridson
      data:
        - source: [df_samples_0, df_samples_1]
          transform:
            - profile:
                method: bridson
                bin: 100
                objective: max
                coordinates:
                  x: {expr: xx, name: x, scale: log, lim: [0.1, 5]}
                  y: {expr: yy, name: y, scale: linear, lim: [0, 5]}
                  z: {expr: LogL, name: z}
      share_data: _profile_points
      axes: ax
      method: scatter
      coordinates: {x: {expr: x}, y: {expr: y}}
      style: {marker: ".", s: 0, color: "none"}
    - name: _profile_interp
      data:
        - source: _profile_points
          transform:
            - make_interp_2d:
                method: natural_neighbor
                coordinates:
                  x: {expr: x, name: x, lim: [0.1, 5], scale: log}
                  y: {expr: y, name: y, lim: [0, 5], scale: linear}
                  z: {expr: z, name: z}
                grid: 500
                nan_policy: strict
      share_data: _profile_grid
      axes: ax
      method: pcolormesh
      coordinates: {x: {expr: x}, y: {expr: y}, z: {expr: z}}
      style: {edgecolor: none, linewidth: 0}
      colorbar: axc
    - name: _credible_region
      data:
        - source: _profile_grid
      axes: ax
      method: contour
      coordinates: {x: {expr: x}, y: {expr: y}, z: {expr: z}}
      style:
        levels: [z_max - 1.15, z_max - 3.09]    # 运行时计算
        colors: [black, white]
        linewidths: [0.2, 0.2]
        labels: ["$1\\sigma$", "$2\\sigma$"]
        zorder: 20
```

---

#### 示例 C：bridson + cell 图（无插值）+ credible region（9 行）

```yaml
- name: XY_voronoi_profLogL
  type: profile_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0.1, 5], scale: log, label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  z: {expr: LogL, label: "$\\log\\mathcal{L}$"}
  interp: false
  colorbar: {cmap: jarvis_rainbow2_r, vmin: -50, vmax: 0}
  credible_region: {sigma: [1, 2], colors: [black, white]}
```

**等价展开（对标 XY_profL + credible region 叠加）：**

```yaml
  layers:
    - name: _profile_bridson
      data:
        - source: [df_samples_0, df_samples_1]
          transform:
            - profile: { method: bridson, bin: 100, objective: max, ... }
      share_data: _profile_points
      axes: ax
      method: scatter
      coordinates: {x: {expr: x}, y: {expr: y}}
      style: {marker: ".", s: 0, color: "none"}
    - name: _voronoi
      data:
        - source: _profile_points
      axes: ax
      method: voronoi
      coordinates: {x: {expr: x}, y: {expr: y}, z: {expr: z}}
      style: {}
      colorbar: axc
    - name: _hidden_interp                          # 隐藏层：仅为 credible_region 提供网格
      data:
        - source: _profile_points
          transform:
            - make_interp_2d: { method: natural_neighbor, ..., grid: 500 }
      share_data: _profile_grid
      axes: ax
      method: scatter                                # 不可见
      coordinates: {x: {expr: x}, y: {expr: y}}
      style: {marker: ".", s: 0, color: "none"}
    - name: _credible_region
      data:
        - source: _profile_grid
      axes: ax
      method: contour
      coordinates: {x: {expr: x}, y: {expr: y}, z: {expr: z}}
      style:
        levels: [z_max - 1.15, z_max - 3.09]
        colors: [black, white]
        linewidths: [0.2, 0.2]
        zorder: 20
```

---

#### 示例 D：grid + 直接渲染（最快，7 行）

```yaml
- name: XY_grid_profLogL
  type: profile_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0.1, 5], scale: log, label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  z: {expr: LogL, label: "$\\log\\mathcal{L}$"}
  method: grid
  interp: false
```

等价于 EggBox 的 `XY_gridprofL`：直接 grid 分箱 → pcolormesh。

---

#### 示例 E：extra_layers 叠加采样散点（11 行）

```yaml
- name: XY_profLogL_scatter
  type: profile_2d
  data: [df_samples_0, df_samples_1]
  x: {expr: xx, lim: [0.1, 5], scale: log, label: "$x$"}
  y: {expr: yy, lim: [0, 5], label: "$y$"}
  z: {expr: LogL, label: "$\\log\\mathcal{L}$"}
  colorbar: {cmap: jarvis_rainbow2_r, vmin: -50, vmax: 0}
  credible_region: {sigma: [1, 2]}
  extra_layers:
    - method: scatter
      coordinates: {x: {expr: xx}, y: {expr: yy}}
      style: {marker: ".", s: 1, color: none, zorder: 10}
```

---

### 13.8 实现注意

1. **interp=true 展开 3 层**：profile 归约（不可见 scatter → share_data）、NN 插值 + pcolormesh（渲染 → share_data）、credible_region contour（可选）
2. **interp=false 展开**：profile 归约（不可见 scatter → share_data）、voronoi/pcolormesh（渲染）；如果有 credible_region，追加隐藏 interp 层 + contour 层
3. **make_interp_2d 用于 profile**：`as_density: false`、`normalize: false`（z 是原始标量）
4. **credible_region 的 sigma 模式**：运行时从网格数据计算 z_max，然后用 χ² 分位数生成 levels；展开阶段只写入 `contour_mode: profile_likelihood` 与 `sigma`
5. **share_data 命名**：`_profile_points`（归约输出）、`_profile_grid`（插值输出）
6. **grid + interp=false**：profile(grid) 输出已经是规则网格，直接 pcolormesh 渲染，credible_region contour 可直接使用同一份 share_data

实现边界：

- `profile_2d` 是语法糖，展开后使用现有 `profile`、`make_interp_2d`、`pcolormesh`、`voronoi`、`contour` runtime
- credible_region 的 sigma→levels 转换已通过 contour runtime 的 `contour_mode: profile_likelihood` 模式支持
- Figure 类无需任何修改

### 13.9 实现状态

已落地：

- `type: profile_2d` figure 会在 Figure 解析前展开为标准 `style` / `frame` / `layers` 配置。
- 顶层 `method` 支持 `bridson` / `grid`，默认 `bridson`。
- 顶层 `interp` 支持 `true` / `false`，默认 `true`；`interp: true` 使用 `make_interp_2d(method=natural_neighbor)` 生成规则网格。
- 四种组合均已展开：`bridson+interp`、`bridson+cell`、`grid+interp`、`grid+direct`。
- `credible_region.levels` 直接展开为 contour levels；`credible_region.sigma` 展开为 `contour_mode: profile_likelihood`，运行时按 z_max 计算 profile likelihood levels。
- `interp: false` 且 `method: bridson` 时，如果请求 `credible_region`，会自动追加隐藏插值层，仅供 contour 使用。
- `make_interp_2d` 在 profile 管线中显式使用 `as_density: false`、`normalize: false`。
- `colorbar` 默认 label 使用 `z.label` 或 `z.expr`，默认 `vmin/vmax` 为 auto。
- `extra_layers` 支持标准 layer 格式；未显式写 `data` 时继承 figure 级 `data`。
