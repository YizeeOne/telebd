# 指标与阈值说明
- 覆盖率1：满足(RSRP >= -105dBm 且 SINR >= -3)的采样点占比
- 下行速率达标率：应用层下行速率 >= 2Mbps 的占比
- 上行速率达标率：应用层上行速率 >= 0.5Mbps 的占比
- 单位：吞吐率为 Mbps，测试总里程为 KM，总时长为 h
- delta_rsrp：x1_avg_rsrp - x2_avg_rsrp（空间不均匀）
- delta_cov2：x1_cov2 - x2_cov2（空间不均匀）
- effective_dl/ul：平均吞吐率 * (1 - 掉线率)
- dl/ul_tail_drag：前10%峰值 - 平均吞吐率（尾部拖累）
- dl/ul_drops_per_hour：掉线次数 / 测试总时长
- dl/ul_drops_per_km：掉线次数 / 测试总里程
- availability：RedCap 驻留比 * 5G覆盖率（优先覆盖率3/4）
- stay_eff：驻留时长 / 测试总里程
- mobility_penalty：KPI_DT - KPI_CQT（移动惩罚）

## 综合评分权重
- 覆盖与无线质量(coverage) 权重 0.25，方向 higher
- 体验与容量(experience) 权重 0.3，方向 higher
- 稳定性(stability) 权重 0.2，方向 lower
- 空间不均匀(spatial) 权重 0.15，方向 lower
- 体验波动(volatility) 权重 0.1，方向 lower

- 显著性检验可用：是
