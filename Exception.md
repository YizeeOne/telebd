2. 识别突发高负荷小区（流量突增）

这些小区的流量突然增加，可能由于外部事件或内部问题导致流量激增。

定义：FLOW_DIFF > threshold（例如，流量增幅超过标准差的 2 倍）

分析步骤：

计算流量增幅：通过计算每个时段与前一时段的流量差异，找出流量突增的时段。

统计流量异常：找到流量突然增加的小区，按 CELL_ID 和 DATETIME_KEY 排序。

对比分析：分析这些小区的 USER_COUNT、TYPE、SCENE、dayofweek 等特征，看看是否有共性。

可能原因：特殊事件（如活动、促销、临时广告）或异常流量（如DDoS攻击）。

指标结果（全量数据）：
- 阈值（max_diff 均值 + 2σ）：15581.03
- 突发高负荷小区：1464 / 65174（2.25%）
- 高发时段：周四、14 时
- 场景 Top3：SCENE=5（31.3%）、SCENE=2（25.6%）、SCENE=6（10.2%）
- 类型 Top3：TYPE=1（55.9%）、TYPE=0（43.0%）、TYPE=2（1.2%）
- Top5 样本：
  - CELL_ID=6350，max_diff=328779.88，2021-03-27 10:00，SCENE=6，TYPE=1
  - CELL_ID=34313，max_diff=273619.56，2021-03-12 12:00，SCENE=2，TYPE=1
  - CELL_ID=39683，max_diff=216781.25，2021-03-21 12:00，SCENE=15，TYPE=0
  - CELL_ID=64590，max_diff=199963.73，2021-03-13 17:00，SCENE=5，TYPE=1
  - CELL_ID=57275，max_diff=188305.16，2021-03-26 11:00，SCENE=2，TYPE=1

结论要点：突发高负荷集中在部分场景与类型，小区级联排查可优先关注上述高发场景与时段。

产出文件：`report_assets/exception/sudden_spike_top20.csv`

3. 长期高负荷小区（高流量持续）

长期高负荷小区通常是网络资源紧张的区域，可能需要优化或扩容。

定义：FLOW_SUM 长期高于某个阈值，如 P90(FLOW_SUM) 或某个固定值。

分析步骤：

统计高流量时段：找出在较长时间内流量持续偏高的小区，生成流量分布图，展示这些小区的流量情况。

对比分析：对比这些小区的 USER_COUNT 和 FLOW_SUM，找出流量的季节性和趋势性特征。

长期监控：分析一段时间内（如周、月）的流量变化，确定哪些小区持续处于高负荷状态。

可能原因：小区周围的用户密度大、重要商业区域、高密度居民区等。

指标结果（全量数据）：
- 阈值（flow_mean P90）：2101.40
- 长期高负荷小区：6518 / 65174（10.00%）
- 稳定高负荷（flow_cv ≤ 中位数）：5874（占高负荷小区 90.12%）
- 场景 Top3：SCENE=2（41.1%）、SCENE=5（12.7%）、SCENE=6（12.5%）
- 类型 Top3：TYPE=1（64.8%）、TYPE=0（32.8%）、TYPE=2（2.4%）
- Top5 样本（按 flow_mean）：
  - CELL_ID=621，flow_mean=17286.17，user_mean=1632.24，SCENE=2，TYPE=1
  - CELL_ID=39033，flow_mean=15847.06，user_mean=2678.05，SCENE=2，TYPE=1
  - CELL_ID=620，flow_mean=14833.86，user_mean=870.74，SCENE=2，TYPE=1
  - CELL_ID=59748，flow_mean=14452.30，user_mean=3027.06，SCENE=18，TYPE=1
  - CELL_ID=29982，flow_mean=13145.19，user_mean=1813.79，SCENE=15，TYPE=0

结论要点：高负荷小区以稳定型为主，适合做长期扩容与分层调度策略。

产出文件：`report_assets/exception/long_highload_top20.csv`

4. 识别用户异常行为

某些用户的行为可能显得不符合常规，可能是机器行为、攻击行为或者其他异常情况。

定义：USER_COUNT 非常大，且 FLOW_SUM 高于正常范围。

分析步骤：

异常用户：识别活跃度极高的用户（例如，流量非常高但没有显著增加用户数）。

行为模式：对比正常用户与异常用户的行为模式，查看是否存在时间上的偏差（如深夜高频活动、非常短时间内大量下载）。

对比分析：比较异常用户与正常用户的流量、用户数和使用时间的差异。

可能原因：用户设备异常、机器人流量、恶意攻击等。

指标结果（全量数据）：
- user_mean P99=1445.96，flow_mean P99=4231.85，user_mean P10=35.18，flow_per_user P10=1.48
- 高用户低人均流量：261
- 高流量低用户：0
- 高用户高流量：148
- Top3 高用户低人均流量样本：
  - CELL_ID=38985，user_mean=3659.71，flow_mean=4315.56，flow_per_user=1.18，SCENE=1，TYPE=1
  - CELL_ID=25913，user_mean=3593.50，flow_mean=3712.05，flow_per_user=1.03，SCENE=2，TYPE=1
  - CELL_ID=40621，user_mean=3482.66，flow_mean=3989.13，flow_per_user=1.15，SCENE=15，TYPE=1
- Top3 高用户高流量样本：
  - CELL_ID=621，user_mean=1632.24，flow_mean=17286.17，flow_per_user=10.59，SCENE=2，TYPE=1
  - CELL_ID=39033，user_mean=2678.05，flow_mean=15847.06，flow_per_user=5.92，SCENE=2，TYPE=1
  - CELL_ID=59748，user_mean=3027.06，flow_mean=14452.30，flow_per_user=4.77，SCENE=18，TYPE=1

结论要点：异常主要体现为“高用户但人均流量偏低”的集中型场景，需结合业务类型排查是否为群体性并发或采集口径问题。

产出文件：
- `report_assets/exception/abnormal_high_user_low_fpu_top20.csv`
- `report_assets/exception/abnormal_high_flow_low_user_top20.csv`
- `report_assets/exception/abnormal_high_user_high_flow_top20.csv`

5. 时间序列分析（变化趋势）

通过对比不同时间段的流量和用户数，发现是否存在周期性的变化或突发的流量激增。

定义：分析时间序列中的异常波动。

分析步骤：

季节性对比：分析工作日与周末、节假日与普通日的流量差异，找出是否有大范围的季节性波动。

趋势分析：计算移动平均线，查看流量的趋势变化，是否有突然的波动。

对比分析：比较节假日（如春节）与普通日期的流量差异，确定哪些场景受节假日影响较大。

指标结果（全量数据）：
- 工作日 vs 周末：flow 1.500e9 vs 1.485e9（-1.0%），user 4.229e8 vs 4.156e8（-1.7%）
- 节假日 vs 普通日：flow 1.253e9 vs 1.545e9（-18.9%），user 3.409e8 vs 4.370e8（-22.0%）
- 日流量最大/最小：2021-04-09=1.797e9，2021-02-15=1.074e9，差异 67.4%
- 日流量/用户相关系数：0.911
- 日变化 |Δ| P90：9.48e7

结论要点：节假日的整体流量与用户显著下降，日流量与用户数高度同步，可作为容量规划与异常预警的基线参考。

6. 异常值检测方法（统计与模型）

采用统计学或机器学习的方法，基于过去的正常数据，检测出异常。

统计方法：

标准差法：flow_mean > 均值 + 2σ，阈值=2788.47，异常小区 3037（4.66%）。

分位数法：P95=2715.87（异常 3259，占 5.00%），P99=4231.85（异常 652，占 1.00%）。

机器学习方法：

孤立森林（Isolation Forest）：contamination=1%，异常 652（1.00%）。

K-means 聚类：k=3，距离阈值=5.79，异常 652（1.00%）。

结论要点：统计阈值法能覆盖更多高值样本，模型方法更强调“结构性异常”；可结合两类结果做分级告警。

产出文件：
- `report_assets/exception/anomaly_isoforest_top20.csv`
- `report_assets/exception/anomaly_kmeans_top20.csv`
- `report_assets/exception/exception_stats.json`
