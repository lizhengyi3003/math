# data_analysis.py 完整终端输出

```
Font: QQXHN
============================================================
正在加载 data_modeling.xlsx ...
  行数: 605, 列数: 32, 受试者数: 251
  人均测量次数: 2.4 次
  健康/不健康: {'是': np.int64(583), '否': np.int64(22)}

============================================================
阶段1：探索性数据分析
  Y染色体浓度: 均值=0.0772, 标准差=0.0326, 最小值=0.0100, 最大值=0.2342, 偏度=0.76
  孕周数值: 均值=16.4973, 标准差=3.9501, 最小值=11.0000, 最大值=29.0000, 偏度=0.77
  孕妇BMI: 均值=32.2656, 标准差=2.8433, 最小值=26.6200, 最大值=46.8800, 偏度=1.03
  年龄: 均值=29.1008, 标准差=3.6915, 最小值=21.0000, 最大值=43.0000, 偏度=0.61
  OK hist_distributions.png
  OK boxplot_by_group.png

============================================================
阶段2：相关性分析
  OK corr_heatmap.png

Y染色体浓度 与其他变量的相关性:
  孕周数值                : r=+0.1618, p=0.0001 ***
  孕妇BMI               : r=-0.1165, p=0.0041 **
  年龄                  : r=-0.0922, p=0.0233 *
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
阶段4：线性混合模型

  M0：空模型（ICC）
    ICC = 0.5626

  M1：+ 孕周 + BMI
            Coef. Std.Err.       z  P>|z|  [0.025  0.975]
Intercept   0.078    0.002  41.012  0.000   0.074   0.081
孕周数值_z      0.012    0.001  12.965  0.000   0.010   0.014
孕妇BMI_z    -0.004    0.002  -2.181  0.029  -0.007  -0.000
Group Var   0.001    0.006                               

  M2：+ 年龄 + IVF
            Coef. Std.Err.       z  P>|z|  [0.025  0.975]
Intercept   0.078    0.002  40.756  0.000   0.074   0.081
孕周数值_z      0.012    0.001  12.947  0.000   0.010   0.014
孕妇BMI_z    -0.004    0.002  -2.102  0.036  -0.007  -0.000
年龄_z       -0.003    0.002  -1.521  0.128  -0.007   0.001
IVF_试管     -0.024    0.022  -1.092  0.275  -0.066   0.019
IVF_人授      0.013    0.022   0.598  0.550  -0.030   0.056
Group Var   0.001    0.006                               

  M3：+ 孕周 × BMI 交互项
                 Coef. Std.Err.       z  P>|z|  [0.025  0.975]
Intercept        0.077    0.002  40.608  0.000   0.074   0.081
孕周数值_z           0.012    0.001  13.023  0.000   0.010   0.014
孕妇BMI_z         -0.004    0.002  -2.392  0.017  -0.008  -0.001
孕周数值_z:孕妇BMI_z   0.002    0.001   2.224  0.026   0.000   0.003
年龄_z            -0.003    0.002  -1.486  0.137  -0.007   0.001
IVF_试管          -0.023    0.022  -1.075  0.282  -0.066   0.019
IVF_人授           0.012    0.022   0.532  0.594  -0.031   0.054
Group Var        0.001    0.006                               

模型比较:
  Model           AIC        BIC       -2LL
  M0              nan        nan    -2566.6
  M1              nan        nan    -2679.5
  M2              nan        nan    -2661.4
  M3              nan        nan    -2653.9
  LRT M0 vs M1: χ²=112.85, df=2, p=0.0000
  LRT M1 vs M2: χ²=-18.07, df=3, p=1.0000

R²（Nakagawa 近似）:
  M1: Marginal R2=0.3820, Conditional R2=0.8214
  M2: Marginal R2=0.3840, Conditional R2=0.8215
  OK model_forest_plot.png
  OK model_residuals_fitted.png
  OK model_qq_random_effects.png

============================================================
阶段5：健康 vs 不健康 对比
  健康（n=583）：均值=0.077703, 标准差=0.032651, 中位数=0.076264
  不健康（n=22）：均值=0.063410, 标准差=0.028730, 中位数=0.053296
  Mann-Whitney U = 8114.0, p = 0.0346
  OK coef_comparison_health.png

============================================================
阶段6：非线性检验
二次项 LRT: χ²=-5.38, p=1.0000
  OK scatter_nonlinear_and_bloodtime.png

============================================================
全部完成！
  result/analysis_output.md
  result/boxplot_by_group.png
  result/coef_comparison_health.png
  result/corr_heatmap.png
  result/hist_distributions.png
  result/model_forest_plot.png
  result/model_qq_random_effects.png
  result/model_residuals_fitted.png
  result/scatter_nonlinear_and_bloodtime.png
  result/scatter_relationships.png
  result/spaghetti_plots.png
```
