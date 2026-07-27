"""
问题二核心分析：基于LMM模型求解各BMI分组的最优NIPT检测时点
================================================================
1. 硬编码问题一 M1 模型系数与误差参数
2. 反解"使 Y 浓度 ≥ 0.04（95%置信）"的最早孕周
3. 双轨分组（K-Means + 经验）的时点对比
4. 敏感性分析（首次BMI vs 均值BMI，不同置信水平）
5. 可视化
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings("ignore")

# ================================================================
# 路径与字体
# ================================================================
INPUT_FILE = r"../data_modeling_p2.xlsx"
RESULT_DIR = r"result"
os.makedirs(RESULT_DIR, exist_ok=True)

sns.set_style("whitegrid")
sns.set_palette("Set2")

_FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "qiaoqiaoxihuanni.ttf")
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    _font_name = fm.FontProperties(fname=_FONT_PATH).get_name()
    plt.rcParams["font.family"] = _font_name
    plt.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def save_fig(name):
    path = os.path.join(RESULT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  OK {name}")


# ================================================================
# 0. 模型常量（来自问题一 M1 LMM）
# ================================================================
INTERCEPT   = 0.078          # 截距
BETA_WEEK_Z = 0.012          # 标准化孕周系数
BETA_BMI_Z  = -0.004         # 标准化BMI系数
WEEK_MEAN   = 16.4973        # 孕周均值
WEEK_STD    = 3.9501         # 孕周标准差
BMI_MEAN    = 32.2656        # BMI均值
BMI_STD     = 2.8433         # BMI标准差
SIGMA_RESID = 0.017471       # 残差标准差 (r.scale 开方)
TARGET_Y    = 0.04           # 目标浓度阈值

# 置信水平 → Z值
Z_VALUES = {
    0.80: 0.842,
    0.90: 1.282,
    0.95: 1.645,
    0.99: 2.326,
}
PRIMARY_CONF = 0.95          # 主分析使用 95% 置信
z_primary = Z_VALUES[PRIMARY_CONF]

print("=" * 60)
print("问题二 · 最优 NIPT 检测时点求解")
print("=" * 60)
print(f"\n模型常数:")
print(f"  Y = {INTERCEPT} + {BETA_WEEK_Z}×(孕周−{WEEK_MEAN})/{WEEK_STD} + ({BETA_BMI_Z})×(BMI−{BMI_MEAN})/{BMI_STD}")
print(f"  残差标准差 σ_ε = {SIGMA_RESID:.6f}")
print(f"  目标浓度 Y ≥ {TARGET_Y}")
print(f"  安全余量 (z_{PRIMARY_CONF}) = {z_primary:.3f} × {SIGMA_RESID:.6f} = {z_primary*SIGMA_RESID:.6f}")

# ================================================================
# 1. 核心函数
# ================================================================

def predict_y(week, bmi):
    """给定孕周和BMI，返回预测的Y染色体浓度（原始单位）"""
    week_z = (week - WEEK_MEAN) / WEEK_STD
    bmi_z  = (bmi - BMI_MEAN) / BMI_STD
    return INTERCEPT + BETA_WEEK_Z * week_z + BETA_BMI_Z * bmi_z

def solve_optimal_week(bmi, confidence=PRIMARY_CONF):
    """
    反解：使预测 Y 浓度 + 安全余量 ≥ 0.04 的最早孕周
    safety_margin = z_confidence × σ_ε
    求解: INTERCEPT + BETA_WEEK_Z × (week − WEEK_MEAN)/WEEK_STD + BETA_BMI_Z × (bmi−BMI_MEAN)/BMI_STD ≥ 0.04 + z×σ
    """
    z = Z_VALUES.get(confidence, 1.645)
    target_with_margin = TARGET_Y + z * SIGMA_RESID
    bmi_z = (bmi - BMI_MEAN) / BMI_STD
    # 从方程反解 week
    week = ((target_with_margin - INTERCEPT - BETA_BMI_Z * bmi_z) / BETA_WEEK_Z) * WEEK_STD + WEEK_MEAN
    return week

def solve_optimal_week_no_margin(bmi):
    """不考虑误差（safety_margin=0）的最早达标孕周"""
    bmi_z = (bmi - BMI_MEAN) / BMI_STD
    week = ((TARGET_Y - INTERCEPT - BETA_BMI_Z * bmi_z) / BETA_WEEK_Z) * WEEK_STD + WEEK_MEAN
    return week


# ================================================================
# 2. 加载数据
# ================================================================
print(f"\n读取: {INPUT_FILE}")
df = pd.read_excel(INPUT_FILE)
# 恢复 category 类型
for col in ["BMI分组_A", "BMI分组_B"]:
    df[col] = df[col].astype("category")
print(f"  总记录: {df.shape[0]} 行, 孕妇数: {df['孕妇代码'].nunique()} 人")

# ================================================================
# 3. 分组最优时点计算
# ================================================================
print("\n" + "=" * 60)
print("分组最优时点计算")

# 生成每人的首次 BMI 汇总
person_info = df.groupby("孕妇代码").agg(
    首次BMI=("BMI首次", "first"),
    均值BMI=("BMI均值", "first"),
    BMI分组_A=("BMI分组_A", "first"),
    BMI分组_B=("BMI分组_B", "first"),
).reset_index()

# 对每个人计算最优时点
person_info["最优时点_无误差"] = person_info["首次BMI"].apply(solve_optimal_week_no_margin)
person_info["最优时点_95误差"] = person_info["首次BMI"].apply(lambda b: solve_optimal_week(b, 0.95))
person_info["最优时点_均值BMI_95"] = person_info["均值BMI"].apply(lambda b: solve_optimal_week(b, 0.95))

# 分组汇总（方案B - 经验分组，论文主表）
print("\n  方案B（经验分组）—— 论文核心表格:")
print(f"  {'BMI分组':<12s} {'人数':>5s} {'组均BMI':>8s} {'无误差(周)':>10s} {'95%误差(周)':>10s} {'推迟量(周)':>10s}")
print(f"  {'-'*60}")

results_b = []
for grp_name in df["BMI分组_B"].cat.categories:
    sub = person_info[person_info["BMI分组_B"] == grp_name]
    if len(sub) == 0:
        continue
    mean_bmi = sub["首次BMI"].mean()
    week_no = solve_optimal_week_no_margin(mean_bmi)
    week_95 = solve_optimal_week(mean_bmi, 0.95)
    delay = week_95 - week_no
    results_b.append({
        "分组": grp_name, "人数": len(sub), "组均BMI": mean_bmi,
        "无误差周": week_no, "95误差周": week_95, "推迟量": delay
    })
    print(f"  {grp_name:<12s} {len(sub):>5d} {mean_bmi:>8.2f} {week_no:>10.2f} {week_95:>10.2f} {delay:>10.2f}")

# 方案A（K-Means分组）
print(f"\n  方案A（K-Means）—— 对比参考:")
results_a = []
for grp_name in df["BMI分组_A"].cat.categories:
    sub = person_info[person_info["BMI分组_A"] == grp_name]
    if len(sub) == 0:
        continue
    mean_bmi = sub["首次BMI"].mean()
    week_no = solve_optimal_week_no_margin(mean_bmi)
    week_95 = solve_optimal_week(mean_bmi, 0.95)
    delay = week_95 - week_no
    results_a.append({
        "分组": grp_name, "人数": len(sub), "组均BMI": mean_bmi,
        "无误差周": week_no, "95误差周": week_95, "推迟量": delay
    })
    print(f"  {grp_name:<12s} {len(sub):>5d} {mean_bmi:>8.2f} {week_no:>10.2f} {week_95:>10.2f} {delay:>10.2f}")

# ================================================================
# 4. 敏感性分析
# ================================================================
print("\n" + "=" * 60)
print("敏感性分析")

# 4a. 首次BMI vs 均值BMI 分组对比
print("\n  首次BMI vs 均值BMI 分组时点差异:")
for grp_name in df["BMI分组_B"].cat.categories:
    sub = person_info[person_info["BMI分组_B"] == grp_name]
    if len(sub) == 0:
        continue
    w_first = solve_optimal_week(sub["首次BMI"].mean(), 0.95)
    w_mean  = solve_optimal_week(sub["均值BMI"].mean(), 0.95)
    diff = w_mean - w_first
    print(f"    {grp_name}: 首次BMI→{w_first:.1f}周, 均值BMI→{w_mean:.1f}周, 差异={diff:+.2f}周")

# 4b. 不同置信水平
print(f"\n  不同置信水平下的最优时点（方案B）:")
conf_levels = [0.80, 0.90, 0.95, 0.99]
sensitivity_data = {}
for grp_name in df["BMI分组_B"].cat.categories:
    sub = person_info[person_info["BMI分组_B"] == grp_name]
    if len(sub) == 0:
        continue
    mean_bmi = sub["首次BMI"].mean()
    row = {}
    for conf in conf_levels:
        w = solve_optimal_week(mean_bmi, conf)
        row[conf] = w
    sensitivity_data[grp_name] = row
    weeks_str = "  ".join([f"{conf:.0%}:{row[conf]:.1f}周" for conf in conf_levels])
    print(f"    {grp_name} (BMI={mean_bmi:.1f}): {weeks_str}")


# ================================================================
# 5. 可视化
# ================================================================

# 5a. 核心图：BMI 连续曲线（有/无误差 + 分组色带）
print("\n" + "=" * 60)
print("生成图表")

fig, ax = plt.subplots(figsize=(12, 6.5))

bmi_range = np.linspace(26, 48, 300)
y_no_margin = [solve_optimal_week_no_margin(b) for b in bmi_range]
y_with_margin = [solve_optimal_week(b, PRIMARY_CONF) for b in bmi_range]

# 背景色带（方案B分组）
colors_band = ["#e8f5e9", "#fff3e0", "#fce4ec", "#e3f2fd", "#f3e5f5"]
band_labels = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "≥40"]
for i, (lo, hi) in enumerate([(20,28),(28,32),(32,36),(36,40),(40,60)]):
    ax.axvspan(max(lo, 26), min(hi, 48), alpha=0.3, color=colors_band[i % len(colors_band)],
               label=f"BMI {band_labels[i]}" if i < 5 else "")

# 核心曲线
ax.plot(bmi_range, y_with_margin, color="#C44E52", lw=3, label=f"考虑 95% 误差（安全余量={z_primary*SIGMA_RESID:.4f}）")
ax.plot(bmi_range, y_no_margin, color="#4C72B0", lw=2.5, ls="--", label="不考虑误差（均值预测）")

# 标注关键阈值
ax.axhline(12, color="gray", ls=":", lw=1, alpha=0.7)
ax.axhline(27, color="gray", ls=":", lw=1, alpha=0.7)
ax.text(47, 12.2, "12周（早期NIPT下限）", fontsize=9, ha="right", color="gray")
ax.text(47, 27.2, "27周（临床检测上限）", fontsize=9, ha="right", color="gray")

# 标注各组最优时点
for r in results_b:
    ax.scatter(r["组均BMI"], r["95误差周"], color="#C44E52", s=100, zorder=5, edgecolors="white", lw=1.5)
    ax.annotate(f"{r['95误差周']:.1f}周", (r["组均BMI"], r["95误差周"]),
                textcoords="offset points", xytext=(8, -12), fontsize=9, color="#C44E52", fontweight="bold")

ax.set_xlabel("孕妇 BMI", fontsize=13)
ax.set_ylabel("最早达标孕周（周）", fontsize=13)
ax.set_title("BMI 与最优 NIPT 检测时点的关系\n（实线=考虑检测误差，虚线=不考虑误差）", fontsize=14)
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(26, 48)
ax.set_ylim(10, 30)
save_fig("bmi_vs_optimal_week.png")

# 5b. 分组柱状图：方案A vs 方案B vs 均值BMI
fig, ax = plt.subplots(figsize=(12, 6))

groups_display = [r["分组"] for r in results_b]
x = np.arange(len(groups_display))
width = 0.25

weeks_a = [r["95误差周"] for r in results_a] if len(results_a) == len(results_b) else [np.nan]*len(groups_display)
weeks_b = [r["95误差周"] for r in results_b]
weeks_mean = [solve_optimal_week(person_info[person_info["BMI分组_B"]==g]["均值BMI"].mean(), 0.95)
              if len(person_info[person_info["BMI分组_B"]==g]) > 0 else np.nan
              for g in df["BMI分组_B"].cat.categories]

bars1 = ax.bar(x - width, weeks_a, width, label="方案A（K-Means首次BMI）", color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x, weeks_b, width, label="方案B（经验首次BMI）", color="#55A868", edgecolor="white")
bars3 = ax.bar(x + width, weeks_mean, width, label="方案B（均值BMI）", color="#C44E52", edgecolor="white")

# 数值标注
for bar in bars1:
    h = bar.get_height()
    if not np.isnan(h):
        ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f"{h:.1f}", ha="center", fontsize=8, color="#4C72B0")
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{bar.get_height():.1f}", ha="center", fontsize=8, color="#55A868")
for bar in bars3:
    h = bar.get_height()
    if not np.isnan(h):
        ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f"{h:.1f}", ha="center", fontsize=8, color="#C44E52")

ax.set_xticks(x)
ax.set_xticklabels(groups_display)
ax.set_ylabel("最早达标孕周（周，95%置信）", fontsize=12)
ax.set_title("不同 BMI 分组方案的最优 NIPT 时点对比", fontsize=14)
ax.legend(fontsize=9)
ax.axhline(12, color="gray", ls=":", alpha=0.5)
save_fig("group_timing_comparison.png")

# 5c. 误差敏感度热力图
fig, ax = plt.subplots(figsize=(10, 5))
heatmap_data = pd.DataFrame(sensitivity_data).T
heatmap_data.columns = [f"{c:.0%}" for c in heatmap_data.columns]
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
            linewidths=0.5, cbar_kws={"label": "最早达标孕周（周）"})
ax.set_title("不同置信水平下的最优 NIPT 时点（方案B）", fontsize=14)
ax.set_xlabel("置信水平", fontsize=12)
ax.set_ylabel("BMI 分组", fontsize=12)
save_fig("error_sensitivity.png")

# 5d. 误差推迟量柱状图
fig, ax = plt.subplots(figsize=(10, 5))
delays = [r["推迟量"] for r in results_b]
bars = ax.bar(groups_display, delays, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CC6677"],
               edgecolor="white")
for bar, d in zip(bars, delays):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"+{d:.1f}周",
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("检测推迟量（周）", fontsize=12)
ax.set_xlabel("BMI 分组", fontsize=12)
ax.set_title("各组因检测误差导致的 NIPT 时点推迟量", fontsize=14)
save_fig("timing_delay_by_group.png")

# ================================================================
# 6. 汇总总结
# ================================================================
print("\n" + "=" * 60)
print("全部完成！")
print(f"\n核心结论:")
print(f"  安全余量: z_{PRIMARY_CONF} × σ_ε = {z_primary:.3f} × {SIGMA_RESID:.6f} = {z_primary*SIGMA_RESID:.6f}")
print(f"  即：需要预测 Y 浓度 ≥ {TARGET_Y + z_primary*SIGMA_RESID:.6f} 才能 95% 确信真实浓度 ≥ {TARGET_Y}")
for r in results_b:
    print(f"  {r['分组']}: 建议 {r['95误差周']:.1f} 周后进行 NIPT（误差推迟 +{r['推迟量']:.1f} 周）")

print(f"\n图表文件:")
for c in sorted(os.listdir(RESULT_DIR)):
    print(f"  result/{c}")
