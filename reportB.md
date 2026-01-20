## 1. 报告摘要

研究目标：分析某省市万级别小区在 2021 年春节前后的流量与用户分布规律。

核心发现：简述识别出的主要小区画像（如办公区、住宅区）及其在节假日的行为特征。

实践意义：基于分析结论，为运营商提供基站扩容或节能的决策建议。

## 数据集概述与预处理

### 2.1 数据集介绍

数据规模：包含万级别小区的时序指标与属性数据 。

关键指标：业务流量（FLOW_SUM）、用户数（USER_COUNT）及小区 ID 。

静态属性：小区经纬度（LATITUDE/LONGITUDE）、覆盖类型（TYPE）和覆盖场景（SCENE）
。

### 2.2 预处理流程 

数据清洗：剔除流量或用户数为负的异常记录，处理异常值，处理缺失值，处理重复数据。

多表关联：利用 CELL_ID 将时序数据与 attributes
属性表进行左连接，补全场景标签 。

特征构造：根据 DATETIME_KEY 构造"小时"、"星期"、"是否节假日"等关键特征。

## 3. 描述性统计分析

说明：本节必须使用 `all_final_data_with_attributes.csv`，该数据已包含属性字段与派生指标（日期、小时、星期、是否节假日、flow_per_user、PAR、ActivityScore 等）。在剔除负值、保留缺失为 NaN 的前提下完成统计与可视化。

### 3.1 全局概览

全量有效记录数为 93,850,560 条（FLOW_SUM/USER_COUNT 均为非负），核心统计量如下：

|指标|均值|标准差|最小值|最大值|有效记录数|
|-|-|-|-|-|-|
|业务流量（FLOW_SUM, MB）|956.57|1493.66|0.00|329743.91|93,850,560|
|用户数（USER_COUNT）|269.12|393.58|0.00|15925.00|93,850,560|
|人均流量（flow_per_user, MB/人）|6.30|18.07|0.00|7050.71|93,488,836|
|PAR|4.78|3.66|1.27|104.91|93,850,560|
|ActivityScore|0.0099|0.0138|0.0000|0.5419|93,850,560|

![业务流量与用户数分布](report_assets/section3/fig01_flow_user_hist.png)

![业务流量与用户数箱线图](report_assets/section3/fig02_flow_user_box.png)

![业务流量-用户数密度](report_assets/section3/fig03_flow_user_scatter.png)

![全网日总流量走势](report_assets/section3/fig04_daily_total_flow.png)

![全网日总用户数走势](report_assets/section3/fig05_daily_total_user.png)

### 3.2 时间维度与节假日差异

从小时、星期与节假日维度观察网络潮汐规律（节假日按 2021 年法定假日表重新标注），并用热力图揭示“小时-星期”交互特征。

![小时维度平均流量](report_assets/section3/fig06_hourly_mean_flow.png)

![小时维度平均用户数](report_assets/section3/fig07_hourly_mean_user.png)

![星期维度平均流量](report_assets/section3/fig08_weekday_mean_flow.png)

![星期维度平均用户数](report_assets/section3/fig09_weekday_mean_user.png)

![节假日与工作日的小时流量对比](report_assets/section3/fig25_hourly_holiday_flow.png)

![节假日与工作日的小时用户数对比](report_assets/section3/fig26_hourly_holiday_user.png)

![小时-星期维度流量热力图](report_assets/section3/fig15_hour_weekday_heatmap_flow.png)

![小时-星期维度用户数热力图](report_assets/section3/fig16_hour_weekday_heatmap_user.png)

### 3.3 场景与类型差异

按场景（SCENE）与类型（TYPE）分析小区平均水平和总量结构，识别核心价值场景。

![不同场景的小区平均流量分布](report_assets/section3/fig10_scene_flow_box.png)

![不同场景的小区平均用户数分布](report_assets/section3/fig11_scene_user_box.png)

![场景总流量 TOP10](report_assets/section3/fig12_scene_flow_share.png)

![不同 TYPE 的平均流量](report_assets/section3/fig13_type_flow_bar.png)

![不同 TYPE 的平均用户数](report_assets/section3/fig14_type_user_bar.png)

### 3.4 TOP 分析

从小区总流量与人均流量两个维度进行排名，识别核心贡献与高价值小区。

TOP10 小区总流量（单位：百万 MB）：

|排名|小区ID|场景|类型|总流量（百万MB）|
|-|-|-|-|-|
|1|621|2|1|24.89|
|2|39033|2|1|22.82|
|3|620|2|1|21.36|
|4|59748|18|1|20.81|
|5|29982|15|0|18.93|
|6|29983|15|0|17.18|
|7|25630|16|1|16.87|
|8|31945|0|0|16.83|
|9|2821|2|1|15.55|
|10|53661|2|1|14.68|

TOP10 人均流量（单位：MB/人）：

|排名|小区ID|场景|类型|人均流量（MB/人）|
|-|-|-|-|-|
|1|16334|5|0|74.29|
|2|54676|5|0|55.25|
|3|16305|5|0|53.40|
|4|16311|5|0|51.92|
|5|54688|5|0|50.29|
|6|54681|5|0|50.19|
|7|16308|5|0|49.84|
|8|54687|5|0|49.74|
|9|16310|5|0|49.26|
|10|54674|5|0|47.65|

![TOP10 小区总流量](report_assets/section3/fig17_top10_flow.png)

![TOP10 人均流量](report_assets/section3/fig18_top10_flow_per_user.png)

### 3.5 异常分析

定义“静默小区”为“有用户无流量”的时间占比 ≥ 50% 的小区；定义“高负荷小区”为全量小区总流量位于前 1% 的小区（阈值 6,093,868.32 MB）。识别结果如下：

- 静默小区：6 个（占比极低，属于边缘异常）。
- 高负荷小区：652 个（核心承载区，需重点关注扩容与保障）。

![静默比例分布](report_assets/section3/fig19_silent_ratio_hist.png)

![静默小区数量（按场景）](report_assets/section3/fig20_silent_scene.png)

![高负荷小区数量（按场景）](report_assets/section3/fig21_highload_scene.png)

### 3.6 指标扩展与空间特征

基于 flow_per_user、PAR、ActivityScore 与经纬度特征补充多维分析。

![人均流量分布](report_assets/section3/fig24_flow_per_user_hist.png)

![PAR 分布](report_assets/section3/fig22_par_hist.png)

![活跃度评分分布](report_assets/section3/fig21_activityscore_hist.png)

![经纬度泡泡图（大小=流量，颜色=人均流量）](report_assets/section3/fig27_geo_bubble_flow.png)

### 3.7 流程与派生指标说明

本节分析基于 `all_final_data_with_attributes.csv`，节假日字段按 2021 年法定假日表重新计算，在剔除负值、保留缺失的前提下完成统计与可视化，输出 20+ 张中文图表与小区聚合结果。派生指标口径如下：

- 小区平均业务流量：`flow_mean = flow_sum / flow_count`
- 小区平均用户数：`user_mean = user_sum / user_count`
- 人均流量：`flow_per_user = flow_sum / user_sum`
- 流量变异系数：`flow_cv = flow_std / flow_mean`
- 静默比例：`silent_ratio = silent_count / record_count`
- 峰均比：`peak_ratio = flow_max / flow_mean`
- 活跃度均值：`activity_mean = activity_sum / activity_count`
- PAR 均值：`par_mean = par_sum / par_count`

对应的产出文件：`section3_descriptive.py`、`report_assets/section3/section3_stats.json`、`report_assets/section3/section3_cell_agg.csv` 与 `report_assets/section3/fig01_flow_user_hist.png` 至 `report_assets/section3/fig27_geo_bubble_flow.png`。

## 4. 多维度对比与趋势分析

4.1 日内潮汐规律分析

小时分布：绘制全天 24 小时的流量走势图，分析不同 TYPE
小区的早晚高峰特征。

4.2 场景、类型差异对比

场景画像：对比不同 SCENE（如场景 2 与场景
6）的流量密度，分析哪些场景是运营商的核心价值区 。

Type画像

4.3 节假日专项分析（核心点）

春节效应：对比 2 月 11 日至 2 月 17 日与平日（3 月）的流量变化 。

人口迁徙：识别因过年导致的"办公区骤降"与"住宅区上升"现象，量化各场景受节假日影响的程度。

**例：分组柱状图** (Grouped Bar
Chart)：对比"春节期间"与"工作日"在不同场景下的平均流量
。目的：量化迁徙效应（例如：商业区流量下降 60%，住宅区上升 40%） 1212。

**雷达图** (Radar
Chart)：针对某一特定小区，对比其在除夕、初一、普通周日的多个指标
13。指标建议：平均流量、峰值用户数、人均流量 \$Flow / User\$、波动率等

**地理热力图** (Geographic Heatmap)：在地图上根据经纬度展示流量密度 。

建议：制作两张对比图，分别展示"除夕夜 20:00"与"工作日
20:00"的全城热力分布 。

**气泡图** (Bubble
Chart)：横轴经度，纵轴纬度，气泡大小代表流量大小，颜色代表 SCENE 类型 。

## 5. 高阶分析与挖掘

### 5.1 小区画像聚类（多特征剖面）

选取全量小区，构建“24 小时流量占比 + 24 小时用户占比 + 关键指标（flow_mean、user_mean、flow_per_user、activity_mean）”的组合特征，并进行 K-means 聚类。综合轮廓系数、CH 与 DB 指标，最优 K=2，聚类规模分别为 11819 与 53355。两类在峰值时段上存在明显差异：Cluster 0 峰值集中在 20 点，Cluster 1 峰值集中在 12 点，体现“夜间活跃型”与“午间主峰型”的对照。

![聚类能量曲线](report_assets/section5/fig01_elbow_kmeans.png)

![轮廓系数曲线](report_assets/section5/fig02_silhouette_scores.png)

![CH 指标曲线](report_assets/section5/fig02b_ch_scores.png)

![DB 指标曲线](report_assets/section5/fig02c_db_scores.png)

![聚类平均日内曲线](report_assets/section5/fig03_cluster_profiles.png)

![聚类规模](report_assets/section5/fig04_cluster_sizes.png)

![聚类与场景关系热力图](report_assets/section5/fig05_cluster_scene_heatmap.png)

![聚类与 TYPE 分布](report_assets/section5/fig06_cluster_type_bar.png)

![聚类空间分布](report_assets/section5/fig07_cluster_geo_scatter.png)

![聚类综合雷达图](report_assets/section5/fig08_cluster_radar.png)

![聚类时间热力图](report_assets/section5/fig09_cluster_hour_heatmap.png)

![聚类 PCA 可视化](report_assets/section5/fig12_cluster_pca.png)

![小区峰值时段分布](report_assets/section5/fig11_peak_hour_distribution.png)

### 5.2 趋势预测（小时-星期基线模型）

以流量贡献最高的小区（CELL_ID=621）为样本，采用“小时-星期”均值作为基线预测方法，使用最近 7 天作为测试集。预测结果表明：流量 MAE=1732.52，MAPE=1.219；用户数 MAE=107.70，MAPE=0.242。该基线模型能刻画整体周期，但在高波动时段仍有残差。

![流量预测对比](report_assets/section5/fig13_actual_vs_pred_flow.png)

![用户数预测对比](report_assets/section5/fig14_actual_vs_pred_user.png)

![流量残差分布](report_assets/section5/fig15_residual_hist_flow.png)

![流量残差走势](report_assets/section5/fig16_residual_ts_flow.png)

![流量真实-预测散点](report_assets/section5/fig17_actual_vs_pred_scatter_flow.png)

![用户数残差分布](report_assets/section5/fig18_residual_hist_user.png)

![用户数真实-预测散点](report_assets/section5/fig19_actual_vs_pred_scatter_user.png)

### 5.3 指标相关性洞察

基于小区层面的多指标相关性，流量、用户数、人均流量与活跃度之间存在明显相关结构，可用于后续的特征筛选与异常解释。

![指标相关性热力图](report_assets/section5/fig10_correlation_heatmap.png)

### 5.4 高负荷/静默小区画像

结合第 3 部分的高负荷阈值与静默比例阈值，对高负荷与静默小区进行场景、类型分布对比，并输出 top‑N 个体画像。

![高负荷小区场景分布](report_assets/section5/fig20_highload_scene_bar.png)

![高负荷小区 TYPE 分布](report_assets/section5/fig21_highload_type_bar.png)

![静默小区场景分布](report_assets/section5/fig20_silent_scene_bar.png)

![静默小区 TYPE 分布](report_assets/section5/fig21_silent_type_bar.png)

![TOP20 高负荷小区总流量](report_assets/section5/fig22_highload_top20.png)

![TOP20 静默小区比例](report_assets/section5/fig23_silent_top20.png)

### 5.5 结论摘要

- 聚类显示明显的“午间主峰型”与“夜间活跃型”小区结构，对运维策略可进行差异化配置。
- 预测基线能抓住周期性，但高波动时段误差偏大，后续可引入节假日与场景变量优化。
- 相关性热力图说明流量、用户、活跃度可形成稳定的联合指标体系。

### 5.6 代码与产出

- 代码文件：`section5_advanced.py`
- 统计汇总：`report_assets/section5/section5_stats.json`
- 图表输出（27 张）：`report_assets/section5/fig01_elbow_kmeans.png` 至 `report_assets/section5/fig23_silent_top20.png`
- 个体画像输出：`report_assets/section5/highload_top20.csv`、`report_assets/section5/silent_top20.csv`

## 6. 商业价值建议

网络扩容建议：针对高负荷、高增速的小区，提出增加基站部署的方案。

节能降耗策略：针对深夜流量极低、场景稳定的区域，建议实施动态基站关停以节省能耗。

精准营销参考：识别高价值活跃场景，为运营商推送针对性套餐提供数据支持。

## 7. 结论与反思

### 7.1 技术难点与解决方案

- 数据体量大（近亿条记录），单机内存难以一次性加载。解决方案：采用分块读取、分组聚合、按需抽样绘图，并将中间结果落盘。
- 多维指标复杂（时间、场景、类型、节假日），易出现口径不一致。解决方案：统一以 DATETIME_KEY 生成日期/小时/星期/节假日特征，并固定清洗规则（负值置为缺失）。
- 图表数量多且维度跨度大。解决方案：将绘图流程模块化，分别输出场景对比、节假日对比、日历热力图与趋势对比图。

### 7.2 关键结论与图表支撑

- 典型 24 小时流量曲线显示，不同场景存在明显的昼夜差异，适用于差异化运维策略。

![24小时典型流量曲线（场景2 vs 6）](report_assets/section7/fig01_diurnal_scene_2_6.png)

- 双轴趋势图显示流量与用户数大体同步，但在节假日前后出现“高流量/低用户”或“高用户/低流量”的异常波动。

![双轴趋势图（流量 vs 用户数）](report_assets/section7/fig02_dual_axis_daily.png)

- 日历热力图能直观看到春节期间（2 月 12 日附近）的整体流量波动特征。

![日历热力图（日总流量）](report_assets/section7/fig03_calendar_heatmap_flow.png)

- 人均流量与日流量变化率曲线反映出节假日与工作日之间的结构性差异。

![日级人均流量趋势](report_assets/section7/fig04_daily_flow_per_user.png)

![日流量变化率](report_assets/section7/fig07_daily_flow_change.png)

- 节假日与非节假日对比表明，节假日整体流量与用户数水平出现显著变化。

![节假日与非节假日对比](report_assets/section7/fig06_holiday_vs_nonholiday.png)

- 日级流量与用户数散点图说明两者整体相关，但存在离群点，提示潜在的异常运营场景。

![日级流量-用户数散点](report_assets/section7/fig05_daily_flow_user_scatter.png)

### 7.3 未来运营商大数据分析展望

- 从“离线报表”走向“实时洞察”：结合流式计算，建立分钟级网络运行指标监控。
- 从“静态规则”走向“智能预测”：引入时空深度模型与节假日/天气等外部特征，提高预测与告警能力。
- 从“单点优化”走向“全网协同”：基于场景画像与聚类结果，实现基站能耗、扩容与营销的联动优化。

### 7.4 代码与产出

- 代码文件：`section7_conclusion.py`
- 图表输出：`report_assets/section7/fig01_diurnal_scene_2_6.png` 至 `report_assets/section7/fig07_daily_flow_change.png`
