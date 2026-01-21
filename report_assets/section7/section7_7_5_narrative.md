## 7.5 周四流量偏弱专项分析

从非节假日工作日的对比来看，周四日均用户数为 4.279e8，低于周一-周三均值 4.419e8（-3.2%），说明周四整体活跃度偏弱。图 fig08_weekday_flow_bar 与 fig09_weekday_user_bar 进一步显示，周四在工作日序列中处于低位，用户侧的低位更明显，这为“周四偏弱”提供了量化支撑。

分时段维度上，图 fig10_weekday_hour_flow 与 fig11_weekday_hour_user 显示周四在 10:00-17:00 以及晚高峰时段都弱于周一-周三基线：10:00-17:00 平均流量 1119.5 vs 1195.7（-6.4%），平均用户 307.9 vs 351.0（-12.3%）；18:00-22:00 平均流量 1272.2 vs 1316.8（-3.4%）。用户下降幅度大于流量下降幅度，体现出“用户规模偏低但人均流量略高”的结构性差异。

晚高峰持续性方面，图 fig12_weekday_peak_duration 显示周四晚高峰持续时长为 4 小时，而周一-周三为 5 小时，周四高峰更短、峰值更容易回落。综合这些证据可以判断：周四在工作日中属于“偏弱日”，其主要表现是日间活跃与晚高峰强度略弱，且高峰持续性不足。这种结构与资源调度的匹配度较低，容易出现“峰值不强但保障成本不低”的问题，需要在运营和保障上做更精细化的调整（例如周四日间拉活、晚高峰适度压缩保障窗口）。

![工作日均流量（非节假日）](report_assets/section7/fig08_weekday_flow_bar.png)
![工作日均用户数（非节假日）](report_assets/section7/fig09_weekday_user_bar.png)
![工作日时段流量对比（非节假日）](report_assets/section7/fig10_weekday_hour_flow.png)
![工作日时段用户数对比（非节假日）](report_assets/section7/fig11_weekday_hour_user.png)
![晚高峰强度持续时长（>=90%峰值）](report_assets/section7/fig12_weekday_peak_duration.png)
