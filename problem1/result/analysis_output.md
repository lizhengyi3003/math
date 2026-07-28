# data_analysis.py 完整终端输出

```
Font: QQXHN
============================================================
正在加载 data_modeling.xlsx (男胎建模数据) ...
  行数: 605, 列数: 32, 受试者数: 251
  人均测量次数: 2.4 次
  健康/不健康: {'是': np.int64(583), '否': np.int64(22)}

加载女胎建模数据 ...
  女胎: 358 行, 139 人
  女胎 健康/不健康: {'是': np.int64(358)}
  合并后: 963 行 (男=605, 女=358)

============================================================
阶段1：探索性数据分析
  Y染色体浓度: 均值=0.0772, 标准差=0.0326, 最小值=0.0100, 最大值=0.2342, 偏度=0.76
  孕周数值: 均值=16.4973, 标准差=3.9501, 最小值=11.0000, 最大值=29.0000, 偏度=0.77
  孕妇BMI: 均值=32.2656, 标准差=2.8433, 最小值=26.6200, 最大值=46.8800, 偏度=1.03
  年龄: 均值=29.1008, 标准差=3.6915, 最小值=21.0000, 最大值=43.0000, 偏度=0.61
  OK hist_distributions.png
  OK boxplot_by_group.png
  OK gc_quality_control.png
  OK hist_height.png

============================================================
阶段2：相关性分析
  OK corr_heatmap.png

Y染色体浓度 与其他变量的相关性:
  孕周数值                : r=+0.1618, p=0.0001 ***
  孕妇BMI               : r=-0.1165, p=0.0041 **
  年龄                  : r=-0.0922, p=0.0233 *
  身高                  : r=-0.1514, p=0.0002 ***
  X染色体浓度              : r=+0.4858, p=0.0000 ***
  GC含量                : r=-0.0039, p=0.9243 ns
    （注：***/**/* 表示显著性水平，非错误）

偏相关分析（控制 年龄+IVF）:
  Y ~ 孕周数值: r_partial=+0.1644, p=0.0000
  Y ~ 孕妇BMI: r_partial=-0.1162, p=0.0042
  OK scatter_relationships.png

============================================================
阶段3：个体轨迹
  OK spaghetti_plots.png

============================================================
阶段4：线性混合模型（升级版）
  建模样本: 605 行, 251 人
  新增特征: 身高_z, 孕周数值_z², 孕妇BMI_z²

  M0：空模型（计算 ICC）
    ICC = 0.5626  (个体间变异占比 56.3%)

--------------------------------------------------
候选模型比较 (reml=False, ML估计)
--------------------------------------------------
  M_Base      AIC= -2700.6  BIC= -2682.9  -2LL(ML)= -2708.6  params=3
  M_BMI       AIC= -2703.3  BIC= -2681.3  -2LL(ML)= -2713.3  params=4
  M_Quad      AIC= -2714.0  BIC= -2687.6  -2LL(ML)= -2726.0  params=5
  M_Height    AIC= -2716.8  BIC= -2686.0  -2LL(ML)= -2730.8  params=6
  M_Full      AIC= -2716.0  BIC= -2676.4  -2LL(ML)= -2734.0  params=8
  M_All       AIC= -2714.5  BIC= -2670.5  -2LL(ML)= -2734.5  params=9

模型选择（基于 AIC 最小原则）:
  ★ 最优模型: M_Height  (AIC=-2716.8)

嵌套模型 LRT 检验 (ML):
  M_Base vs M_BMI: χ²=4.72, df=1, p=0.0299 *
  M_BMI vs M_Quad: χ²=12.74, df=1, p=0.0004 ***
  M_Quad vs M_Height: χ²=4.77, df=1, p=0.0289 *
  M_Height vs M_Full: χ²=3.22, df=2, p=0.1997 ns
  M_Height vs M_All: χ²=3.74, df=3, p=0.2904 ns

最终模型 M_Height (reml=True):
            Coef. Std.Err.       z  P>|z|  [0.025  0.975]
Intercept   0.075    0.002  36.364  0.000   0.071   0.079
孕周数值_z      0.010    0.001   8.814  0.000   0.008   0.012
孕妇BMI_z    -0.004    0.002  -2.097  0.036  -0.007  -0.000
孕周数值_z2     0.003    0.001   3.576  0.000   0.001   0.005
身高_z       -0.004    0.002  -2.181  0.029  -0.008  -0.000
Group Var   0.001    0.006                               

Wald 联合检验（所有固定效应是否联合为 0）:
  统计量 = 218.63, p = 0.000000

  Marginal R² = 0.3909, Conditional R² = 0.8280

最终模型 M_Height 系数汇总:
  Intercept           : β=+0.074724, 95%CI=[+0.070696, +0.078751]
  孕周数值_z              : β=+0.009663, 95%CI=[+0.007514, +0.011811]
  孕妇BMI_z             : β=-0.003726, 95%CI=[-0.007208, -0.000244]
  孕周数值_z2             : β=+0.002917, 95%CI=[+0.001318, +0.004516]
  身高_z                : β=-0.004177, 95%CI=[-0.007930, -0.000424]

★ 拐点分析（孕周二次项）:
    β1(孕周_z) = 0.009663, β2(孕周_z²) = 0.002917
    标准化空间拐点: t_z_min = -1.66
    原始单位拐点:   t_min = 9.96 周
    解释：Y染色体浓度在约 10 周达到最低点，此后开始快速上升。
    这解释了临床 NIPT 检测在 10 周前容易失败的生物学原因——
    早期胎儿 DNA 释放缓慢，浓度处于曲线底部区域。
  OK model_forest_plot.png

残差诊断:
  OK model_residuals_diagnostics.png
  OK model_qq_random_effects.png

============================================================
阶段5：健康 vs 不健康 对比
  健康（n=583）：均值=0.077703, 标准差=0.032651, 中位数=0.076264
  不健康（n=22）：均值=0.063410, 标准差=0.028730, 中位数=0.053296
  Mann-Whitney U = 8114.0, p = 0.0346
  OK coef_comparison_health.png

============================================================
阶段6：非线性验证与二次曲线可视化
二次项 LRT (ML, 原始单位): χ²=12.54, p=0.0004
  线性模型 AIC=-2700.6, 二次模型 AIC=-2711.1
  原始单位拐点: t_min = 10.27 周
  OK scatter_nonlinear_and_bloodtime.png

============================================================
全部完成！
  result/age_abnormality_rate.png
  result/analysis_output.md
  result/boxplot_by_group.png
  result/coef_comparison_health.png
  result/corr_heatmap.png
  result/gc_quality_control.png
  result/hist_distributions.png
  result/hist_height.png
  result/model_forest_plot.png
  result/model_qq_random_effects.png
  result/model_residuals_diagnostics.png
  result/model_residuals_fitted.png
  result/scatter_nonlinear_and_bloodtime.png
  result/scatter_relationships.png
  result/spaghetti_plots.png
```
