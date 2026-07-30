"""
data_modeling.xlsx 统计分析
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, gaussian_kde
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
import os
warnings.filterwarnings("ignore")

# ================================================================
# 输出目录与日志文件
# ================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(_SCRIPT_DIR, "..", "data_modeling.xlsx")
RESULT_DIR = os.path.join(_SCRIPT_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

_OUTPUT_MD = os.path.join(RESULT_DIR, "analysis_output.md")
_log_file = open(_OUTPUT_MD, "w", encoding="utf-8")
_log_file.write("# data_analysis.py 完整终端输出\n\n```\n")

def tee_print(*args, **kwargs):
    """同时输出到终端和 md 日志文件"""
    import builtins
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        # Windows GBK 终端无法编码部分 Unicode，回退为 ASCII-safe 输出
        safe_args = []
        for a in args:
            if isinstance(a, str):
                a = a.encode("gbk", errors="replace").decode("gbk")
            safe_args.append(a)
        builtins.print(*safe_args, **kwargs)
    builtins.print(*args, file=_log_file, **kwargs)

# ── 步骤1：seaborn 样式（不触碰字体）──
sns.set_style("whitegrid")
sns.set_palette("Set2")

# ── 步骤2：中文字体（必须在 seaborn 之后设置，防止被覆盖）──
import matplotlib.font_manager as fm

_FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "qiaoqiaoxihuanni.ttf")  # 字体相对路径
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    _font_name = fm.FontProperties(fname=_FONT_PATH).get_name()
    plt.rcParams["font.family"] = _font_name
    plt.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
    plt.rcParams["font.serif"] = [_font_name]
    tee_print(f"Font: {_font_name}")
else:
    tee_print(f"WARNING: Font not found at {_FONT_PATH}")

plt.rcParams["axes.unicode_minus"] = False

def save_fig(name):
    path = os.path.join(RESULT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    tee_print(f"  OK {name}")


# ================================================================
# 加载数据
# ================================================================
tee_print("=" * 60)
tee_print("正在加载 data_modeling.xlsx (男胎建模数据) ...")
df = pd.read_excel(INPUT_FILE, sheet_name="男胎建模数据")
n_rows = len(df)
n_subjects = df['孕妇代码'].nunique()
tee_print(f"  行数: {n_rows}, 列数: {len(df.columns)}, 受试者数: {n_subjects}")
tee_print(f"  人均测量次数: {n_rows / n_subjects:.1f} 次")
tee_print(f"  健康/不健康: {dict(df['胎儿是否健康'].value_counts())}")

# ── 加载女胎数据（用于 EDA 全局对比）──
tee_print("\n加载女胎建模数据 ...")
df_female = pd.read_excel(INPUT_FILE, sheet_name="女胎建模数据")
n_female = len(df_female)
n_f_subjects = df_female['孕妇代码'].nunique()
tee_print(f"  女胎: {n_female} 行, {n_f_subjects} 人")
tee_print(f"  女胎 健康/不健康: {dict(df_female['胎儿是否健康'].value_counts())}")

# 合并男女胎数据（全局视角）
df_male_labeled = df.copy()
df_male_labeled["性别"] = "男"
df_female_labeled = df_female.copy()
df_female_labeled["性别"] = "女"
common_cols = [c for c in df_male_labeled.columns if c in df_female_labeled.columns]
df_all = pd.concat([df_male_labeled[common_cols], df_female_labeled[common_cols]], ignore_index=True)
tee_print(f"  合并后: {len(df_all)} 行 (男={n_rows}, 女={n_female})")

# ================================================================
# 阶段1：探索性数据分析（EDA）
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段1：探索性数据分析")

for col in ["Y染色体浓度", "孕周数值", "孕妇BMI", "年龄", "身高"]:
    s = df[col].dropna()
    tee_print(f"  {col}: 均值={s.mean():.4f}, 标准差={s.std():.4f}, "
          f"最小值={s.min():.4f}, 最大值={s.max():.4f}, 偏度={s.skew():.2f}")

# 四合一分布图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (col, title, color) in zip(axes.flat, [
    ("Y染色体浓度", "Y染色体浓度分布", "#4C72B0"),
    ("孕周数值", "孕周数值分布", "#55A868"),
    ("孕妇BMI", "孕妇BMI分布", "#C44E52"),
    ("年龄", "年龄分布", "#8172B2"),
]):
    s = df[col].dropna()
    ax.hist(s, bins=30, color=color, edgecolor="white", alpha=0.85, density=True)
    kde = gaussian_kde(s)
    xr = np.linspace(s.min(), s.max(), 300)
    ax.plot(xr, kde(xr), color="black", lw=2)
    ax.axvline(s.mean(), color="red", ls="--", lw=1.2, label=f"均值={s.mean():.3f}")
    ax.axvline(s.median(), color="orange", ls="--", lw=1.2, label=f"中位数={s.median():.3f}")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8)
    # 裁剪 x 轴：对右偏数据裁剪长尾，聚焦主体分布
    p99 = s.quantile(0.99)
    p01 = s.quantile(0.01)
    margin = (p99 - p01) * 0.1
    ax.set_xlim(max(s.min(), p01 - margin), min(s.max(), p99 + margin * 2))
save_fig("hist_distributions.png")

# 箱线图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="IVF妊娠", y="Y染色体浓度", palette="Set2", ax=axes[0])
axes[0].set_title("Y染色体浓度 按 IVF妊娠 分组", fontsize=13)
sns.boxplot(data=df, x="胎儿是否健康", y="Y染色体浓度", palette="Set2", ax=axes[1])
axes[1].set_title("Y染色体浓度 按 胎儿是否健康 分组", fontsize=13)
save_fig("boxplot_by_group.png")

# ── GC含量质量控制直方图 ──
fig, ax = plt.subplots(figsize=(8, 5))
gc_vals = df["GC含量"].dropna()
ax.hist(gc_vals, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(0.40, color="red", ls="--", lw=2, label="下界 40%")
ax.axvline(0.60, color="red", ls="--", lw=2, label="上界 60%")
ax.axvline(gc_vals.mean(), color="orange", ls="-", lw=1.5, label=f"均值={gc_vals.mean():.4f}")
ax.set_title("GC含量质量控制直方图（男胎）", fontsize=14)
ax.set_xlabel("GC含量"); ax.set_ylabel("频数")
ax.legend(fontsize=10, loc="upper right")
# 裁剪 x 轴：聚焦数据密集区，但保留 40%/60% 参考线可见
p99_gc = gc_vals.quantile(0.99)
p01_gc = gc_vals.quantile(0.01)
xlo = max(0.395, p01_gc - 0.002)
xhi = min(0.605, p99_gc + 0.005)
ax.set_xlim(xlo, xhi)
save_fig("gc_quality_control.png")

# ── 年龄-胎儿异常率柱状图（仅男胎）──
fig, ax = plt.subplots(figsize=(10, 5.5))
age_bins = [0, 25, 30, 35, 100]
age_labels = ["<25", "25-30", "30-35", ">35"]
colors = ["#55A868", "#F5C242", "#E8923F", "#C44E52"]

d = df_male_labeled[["年龄", "胎儿是否健康"]].dropna().copy()
d["年龄组"] = pd.cut(d["年龄"], bins=age_bins, labels=age_labels)
grp = d.groupby("年龄组")["胎儿是否健康"].agg(["count", lambda x: (x == "否").sum()])
grp.columns = ["总数", "异常数"]
grp["异常率"] = grp["异常数"] / grp["总数"] * 100

x = np.arange(len(age_labels))
rates = [grp.loc[lbl, "异常率"] if lbl in grp.index else 0 for lbl in age_labels]
counts = [f"{int(grp.loc[lbl, '异常数'])}/{int(grp.loc[lbl, '总数'])}" if lbl in grp.index else "—" for lbl in age_labels]
bars = ax.bar(x, rates, color=colors, edgecolor="white", width=0.6)
for bar, rate, cnt, c in zip(bars, rates, counts, colors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{rate:.1f}%\n({cnt})", ha="center", fontsize=10, color=c, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(age_labels, fontsize=11)
ax.set_xlabel("年龄组", fontsize=12); ax.set_ylabel("异常率 (%)", fontsize=12)
ax.set_title("男胎：年龄-胎儿异常率", fontsize=14)
ax.set_ylim(0, max(max(rates) * 1.6, 8))
# 标注样本总量
ax.text(0.02, 0.96, f"男胎共 605 例，异常 22 例\n（注：女胎 358 例全部健康，未展示）",
        transform=ax.transAxes, ha="left", va="top", fontsize=9, color="gray",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
save_fig("age_abnormality_rate.png")

# ── 男女胎特征对比箱线图 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 使用男女胎共有的列（女胎无 Y染色体浓度）
feat_pairs = [("孕妇BMI", "孕妇BMI 男女胎对比"), ("孕周数值", "孕周数值 男女胎对比")]
for ax, (col, title) in zip(axes, feat_pairs):
    d_plot = df_all[[col, "性别"]].dropna()
    if d_plot["性别"].nunique() >= 2 and d_plot[col].notna().sum() > 0:
        sns.boxplot(data=d_plot, x="性别", y=col, palette={"男": "#4C72B0", "女": "#C44E52"}, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("")
    else:
        ax.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=13)
save_fig("male_vs_female_comparison.png")

# ── 身高分布直方图 ──
fig, ax = plt.subplots(figsize=(8, 5))
s_height = df["身高"].dropna()
ax.hist(s_height, bins=25, color="#8172B2", edgecolor="white", alpha=0.85, density=True)
kde_h = gaussian_kde(s_height)
xr_h = np.linspace(s_height.min(), s_height.max(), 300)
ax.plot(xr_h, kde_h(xr_h), color="black", lw=2)
ax.axvline(s_height.mean(), color="red", ls="--", lw=1.5, label=f"均值={s_height.mean():.2f}")
ax.axvline(s_height.median(), color="orange", ls="--", lw=1.5, label=f"中位数={s_height.median():.2f}")
ax.set_title("身高分布（男胎）", fontsize=14)
ax.set_xlabel("身高（cm）"); ax.legend()
# 裁剪长尾
p99_h = s_height.quantile(0.99); p01_h = s_height.quantile(0.01)
m_h = (p99_h - p01_h) * 0.1
ax.set_xlim(max(s_height.min(), p01_h - m_h), min(s_height.max(), p99_h + m_h * 2))
save_fig("hist_height.png")

# ================================================================
# 阶段2：相关性分析
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段2：相关性分析")

corr_vars = ["Y染色体浓度", "孕周数值", "孕妇BMI", "年龄", "身高", "X染色体浓度", "GC含量"]
corr_data = df[corr_vars].dropna()

pearson_mat = np.zeros((len(corr_vars), len(corr_vars)))
pearson_p = np.zeros_like(pearson_mat)
spearman_mat = np.zeros_like(pearson_mat)
spearman_p = np.zeros_like(pearson_mat)

for i, vi in enumerate(corr_vars):
    for j, vj in enumerate(corr_vars):
        if i <= j:
            r_p, p_p = pearsonr(corr_data[vi], corr_data[vj])
            r_s, p_s = spearmanr(corr_data[vi], corr_data[vj])
            pearson_mat[i, j] = pearson_mat[j, i] = r_p
            pearson_p[i, j] = pearson_p[j, i] = p_p
            spearman_mat[i, j] = spearman_mat[j, i] = r_s
            spearman_p[i, j] = spearman_p[j, i] = p_s

# 热力图（Pearson + Spearman）
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, mat, pmat, title in [
    (axes[0], pearson_mat, pearson_p, "Pearson 相关系数"),
    (axes[1], spearman_mat, spearman_p, "Spearman 秩相关系数"),
]:
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    annot = np.empty_like(mat, dtype=object)
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            star = "***" if pmat[i, j] < 0.001 else ("**" if pmat[i, j] < 0.01 else ("*" if pmat[i, j] < 0.05 else ""))
            annot[i, j] = f"{mat[i, j]:.3f}{star}"
    sns.heatmap(mat, annot=annot, fmt="", mask=mask,
                xticklabels=corr_vars, yticklabels=corr_vars,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
save_fig("corr_heatmap.png")

tee_print("\nY染色体浓度 与其他变量的相关性:")
for v in corr_vars:
    if v != "Y染色体浓度":
        r, p = pearsonr(corr_data["Y染色体浓度"], corr_data[v])
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        tee_print(f"  {v:20s}: r={r:+.4f}, p={p:.4f} {stars}")
        if v == "GC含量":
            tee_print(f"    （注：***/**/* 表示显著性水平，非错误）")

# 偏相关分析
tee_print("\n偏相关分析（控制 年龄+IVF）:")
control = pd.DataFrame({
    "age": df["年龄"].values,
    "ivf1": (df["IVF妊娠"] == "IVF（试管婴儿）").astype(float).values,
    "ivf2": (df["IVF妊娠"] == "IUI（人工授精）").astype(float).values,
})
for target in ["孕周数值", "孕妇BMI"]:
    valid = df[[target, "Y染色体浓度"]].notna().all(axis=1) & control.notna().all(axis=1)
    y_target = df.loc[valid, target].astype(float).values
    y_y = df.loc[valid, "Y染色体浓度"].astype(float).values
    Xc = sm.add_constant(control.loc[valid].astype(float).values)
    resid_target = sm.OLS(y_target, Xc).fit().resid
    resid_y = sm.OLS(y_y, Xc).fit().resid
    r_partial, p_partial = pearsonr(resid_target, resid_y)
    tee_print(f"  Y ~ {target}: r_partial={r_partial:+.4f}, p={p_partial:.4f}")

# 散点图 + LOESS 平滑
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (xc, yc, title) in zip(axes, [
    ("孕周数值", "Y染色体浓度", "Y染色体浓度与孕周数值"),
    ("孕妇BMI", "Y染色体浓度", "Y染色体浓度与BMI"),
    ("孕周数值", "孕妇BMI", "孕周数值与BMI"),
]):
    d = df[[xc, yc]].dropna()
    x, y = d[xc].values, d[yc].values
    ax.scatter(x, y, alpha=0.4, s=25, color="#4C72B0")
    try:
        s = lowess(y, x, frac=0.3, return_sorted=True)
        ax.plot(s[:, 0], s[:, 1], color="red", lw=2.5, label="LOESS 平滑趋势")
    except:
        pass
    r, p = pearsonr(x, y)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xc); ax.set_ylabel(yc)
    ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.4f}", transform=ax.transAxes,
            fontsize=11, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(fontsize=9)
save_fig("scatter_relationships.png")

# ================================================================
# 阶段3：个体轨迹
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段3：个体轨迹")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
subjects = df["孕妇代码"].unique()
np.random.seed(42)
sample = np.random.choice(subjects, min(80, len(subjects)), replace=False)

for subj in sample:
    d = df[df["孕妇代码"] == subj].sort_values("孕周数值")
    axes[0].plot(d["孕周数值"], d["Y染色体浓度"], alpha=0.3, lw=0.8, color="gray")
sm_lo = lowess(df["Y染色体浓度"], df["孕周数值"], frac=0.3, return_sorted=True)
axes[0].plot(sm_lo[:, 0], sm_lo[:, 1], color="red", lw=3, label="LOESS 平滑趋势")
axes[0].set_title("个体轨迹（Y染色体浓度 随孕周变化）", fontsize=13)
axes[0].set_xlabel("孕周（周）"); axes[0].set_ylabel("Y染色体浓度")
axes[0].legend()

for label, color, health_val in [("健康", "#55A868", "是"), ("不健康", "#C44E52", "否")]:
    sub = df[df["胎儿是否健康"] == health_val]
    for sj in sub["孕妇代码"].unique()[:40]:
        d = sub[sub["孕妇代码"] == sj].sort_values("孕周数值")
        axes[1].plot(d["孕周数值"], d["Y染色体浓度"], alpha=0.25, lw=0.7, color=color)
    if len(sub) > 10:
        s = lowess(sub["Y染色体浓度"], sub["孕周数值"], frac=0.4, return_sorted=True)
        axes[1].plot(s[:, 0], s[:, 1], color=color, lw=3, label=label)
axes[1].set_title("个体轨迹（按健康状态分面）", fontsize=13)
axes[1].set_xlabel("孕周（周）"); axes[1].legend()
save_fig("spaghetti_plots.png")

# ================================================================
# 阶段4：线性混合模型（LMM）—— 特征工程 + 嵌套模型选择
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段4：线性混合模型")

# ── 特征工程：引入身高和孕周二次项 ──
df_m = df[["Y染色体浓度", "孕周数值", "孕妇BMI", "年龄", "身高", "IVF妊娠", "孕妇代码"]].dropna().copy()
df_m["IVF_试管"] = (df_m["IVF妊娠"] == "IVF（试管婴儿）").astype(int)
df_m["IVF_人授"] = (df_m["IVF妊娠"] == "IUI（人工授精）").astype(int)

# Z-score 标准化
for c in ["孕周数值", "孕妇BMI", "年龄", "身高"]:
    df_m[f"{c}_z"] = (df_m[c] - df_m[c].mean()) / df_m[c].std()
# 二次项（标准化后的平方，以及原始单位平方 ↑ 用于拐点计算）
df_m["孕周数值_z2"] = df_m["孕周数值_z"] ** 2
df_m["孕周数值2"] = df_m["孕周数值"] ** 2
df_m["孕妇BMI_z2"] = df_m["孕妇BMI_z"] ** 2

tee_print(f"  建模样本: {len(df_m)} 行, {df_m['孕妇代码'].nunique()} 人")
tee_print(f"  新增特征: 身高_z, 孕周数值_z², 孕妇BMI_z²")

# ── M0：空模型（ICC）──
tee_print("\n  M0：空模型（计算 ICC）")
r0 = smf.mixedlm("Y染色体浓度 ~ 1", df_m, groups=df_m["孕妇代码"]).fit(reml=True)
vr = r0.cov_re.values[0, 0] if hasattr(r0.cov_re, 'values') else r0.cov_re.iloc[0, 0]
icc = vr / (vr + r0.scale)
tee_print(f"    ICC = {icc:.4f}  (个体间变异占比 {icc*100:.1f}%)")

# ================================================================
# ── 候选模型集（全部 reml=False，ML 估计，用于模型选择）──
# ================================================================
tee_print("\n" + "-" * 50)
tee_print("候选模型比较 (reml=False, ML估计)")
tee_print("-" * 50)

candidate_formulas = {
    "M_Base":   "Y染色体浓度 ~ 孕周数值_z",
    "M_BMI":    "Y染色体浓度 ~ 孕周数值_z + 孕妇BMI_z",
    "M_Quad":   "Y染色体浓度 ~ 孕周数值_z + 孕妇BMI_z + 孕周数值_z2",
    "M_Height": "Y染色体浓度 ~ 孕周数值_z + 孕妇BMI_z + 孕周数值_z2 + 身高_z",
    "M_Full":   "Y染色体浓度 ~ 孕周数值_z + 孕妇BMI_z + 孕周数值_z2 + 身高_z + 孕妇BMI_z2 + 孕周数值_z:孕妇BMI_z",
    "M_All":    "Y染色体浓度 ~ 孕周数值_z + 孕妇BMI_z + 孕周数值_z2 + 身高_z + 年龄_z + IVF_试管 + IVF_人授",
}

candidate_models = {}
for name, formula in candidate_formulas.items():
    try:
        m = smf.mixedlm(formula, df_m, groups=df_m["孕妇代码"]).fit(reml=False)
        candidate_models[name] = m
        tee_print(f"  {name:10s}  AIC={m.aic:8.1f}  BIC={m.bic:8.1f}  -2LL(ML)={-2*m.llf:8.1f}  params={len(m.params)}")
    except Exception as e:
        tee_print(f"  {name:10s}  未收敛: {e}")

# ── 模型选择决策 ──
tee_print(f"\n模型选择（基于 AIC 最小原则）:")
best_name = min(candidate_models, key=lambda n: candidate_models[n].aic)
tee_print(f"  ★ 最优模型: {best_name}  (AIC={candidate_models[best_name].aic:.1f})")

# ── LRT 嵌套比较 ──
tee_print(f"\n嵌套模型 LRT 检验 (ML):")
lrt_pairs = [
    ("M_Base", "M_BMI", 1),
    ("M_BMI", "M_Quad", 1),
    ("M_Quad", "M_Height", 1),
    ("M_Height", "M_Full", 2),
    ("M_Height", "M_All", 3),
]
for a, b, dfd in lrt_pairs:
    if a in candidate_models and b in candidate_models:
        lrt_stat = -2 * (candidate_models[a].llf - candidate_models[b].llf)
        p_val = stats.chi2.sf(lrt_stat, df=dfd)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        tee_print(f"  {a} vs {b}: χ²={lrt_stat:.2f}, df={dfd}, p={p_val:.4f} {sig}")

# ================================================================
# ── 最终模型（reml=True 获得无偏估计）──
# ================================================================
tee_print(f"\n最终模型 {best_name} (reml=True):")
final_formula = candidate_formulas[best_name]
final_best = smf.mixedlm(final_formula, df_m, groups=df_m["孕妇代码"]).fit(reml=True)
tee_print(final_best.summary().tables[1])

# ── Wald 联合检验 ──
n_params = len(final_best.params)
wald_mat = np.eye(n_params)[1:]  # 排除截距，检验所有斜率是否联合=0
try:
    wald_result = final_best.wald_test(wald_mat)
    tee_print(f"\nWald 联合检验（所有固定效应是否联合为 0）:")
    tee_print(f"  统计量 = {wald_result.statistic[0][0]:.2f}, p = {wald_result.pvalue:.6f}")
except Exception as e:
    tee_print(f"  Wald 检验未执行: {e}")

# ── R²（Nakagawa）──
vf = np.var(final_best.fittedvalues - final_best.fittedvalues.mean())
vr2 = final_best.cov_re.values[0, 0] if hasattr(final_best.cov_re, 'values') else final_best.cov_re.iloc[0, 0]
ve = final_best.scale
mr2 = vf / (vf + vr2 + ve)
cr2 = (vf + vr2) / (vf + vr2 + ve)
tee_print(f"\n  Marginal R² = {mr2:.4f}, Conditional R² = {cr2:.4f}")

# ── 输出最终模型系数（用于论文）──
tee_print(f"\n最终模型 {best_name} 系数汇总:")
for name, val in final_best.params.items():
    if name != "Group Var":
        ci = final_best.conf_int()
        lo, hi = ci.loc[name, 0], ci.loc[name, 1]
        tee_print(f"  {name:20s}: β={val:+.6f}, 95%CI=[{lo:+.6f}, {hi:+.6f}]")

# ================================================================
# ── 计算拐点（若二次项存在）──
# ================================================================
if "孕周数值_z2" in final_best.params.index:
    beta1 = final_best.params["孕周数值_z"]  # 标准化一次项系数
    beta2 = final_best.params["孕周数值_z2"]  # 标准化二次项系数
    # 还原到原始单位：对 Z-score 标准化变量，拐点计算
    # t_z_min = -beta1 / (2*beta2) （标准化空间中的拐点）
    mean_week = df_m["孕周数值"].mean()
    std_week = df_m["孕周数值"].std()
    t_z_min = -beta1 / (2 * beta2)
    t_min_weeks = t_z_min * std_week + mean_week  # 还原到原始孕周
    tee_print(f"\n★ 拐点分析（孕周二次项）:")
    tee_print(f"    β1(孕周_z) = {beta1:.6f}, β2(孕周_z²) = {beta2:.6f}")
    tee_print(f"    标准化空间拐点: t_z_min = {t_z_min:.2f}")
    tee_print(f"    原始单位拐点:   t_min = {t_min_weeks:.2f} 周")
    if 8 <= t_min_weeks <= 15:
        tee_print(f"    解释：Y染色体浓度在约 {t_min_weeks:.0f} 周达到最低点，此后开始快速上升。")
        tee_print(f"    这解释了临床 NIPT 检测在 10 周前容易失败的生物学原因——")
        tee_print(f"    早期胎儿 DNA 释放缓慢，浓度处于曲线底部区域。")

# ── 森林图（最终模型）──
fig, ax = plt.subplots(figsize=(12, 6))
coef = final_best.params.to_frame("coef")
ci = final_best.conf_int()
coef["low"] = ci.iloc[:, 0]; coef["high"] = ci.iloc[:, 1]
coef = coef.drop("Group Var", errors="ignore")
names = {
    "Intercept": "截距", "孕周数值_z": "孕周（标准化）",
    "孕妇BMI_z": "BMI（标准化）", "年龄_z": "年龄（标准化）",
    "身高_z": "身高（标准化）", "孕周数值_z2": "孕周²（标准化）",
    "孕妇BMI_z2": "BMI²（标准化）",
    "孕周数值_z:孕妇BMI_z": "孕周×BMI 交互",
    "IVF_试管": "IVF-试管", "IVF_人授": "IUI-人授",
}
coef["label"] = [names.get(i, i) for i in coef.index]
yp = range(len(coef))
ax.errorbar(coef["coef"], yp, xerr=[coef["coef"] - coef["low"], coef["high"] - coef["coef"]],
            fmt="o", capsize=5, color="#4C72B0", ms=10, lw=2.5)
ax.axvline(0, color="gray", ls="--")
ax.set_yticks(yp); ax.set_yticklabels(coef["label"]); ax.invert_yaxis()
ax.set_xlabel("系数估计值（95% 置信区间）", fontsize=12)
ax.set_title(f"LMM 固定效应（最终模型 {best_name}）", fontsize=15)
save_fig("model_forest_plot.png")

# ── 残差诊断 ──
tee_print(f"\n残差诊断:")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fitted = final_best.fittedvalues; resid = final_best.resid
std_resid = resid / np.sqrt(resid.var())
abs_resid = np.abs(resid)

# (1) 原始残差 vs 拟合值 + LOESS
axes[0, 0].scatter(fitted, resid, alpha=0.5, s=20, color="#4C72B0")
axes[0, 0].axhline(0, color="red", ls="--")
try:
    s_rf = lowess(resid, fitted, frac=0.4, return_sorted=True)
    axes[0, 0].plot(s_rf[:, 0], s_rf[:, 1], color="red", lw=2.5, label="LOESS")
except: pass
axes[0, 0].set_xlabel("拟合值"); axes[0, 0].set_ylabel("残差")
axes[0, 0].set_title("原始残差 vs 拟合值"); axes[0, 0].legend(fontsize=8)

# (2) 标准化残差 vs 拟合值
axes[0, 1].scatter(fitted, std_resid, alpha=0.5, s=20, color="#55A868")
axes[0, 1].axhline(0, color="red", ls="--")
axes[0, 1].axhline(2, color="orange", ls=":", label="±2 SD")
axes[0, 1].axhline(-2, color="orange", ls=":")
try:
    s_sr = lowess(std_resid, fitted, frac=0.4, return_sorted=True)
    axes[0, 1].plot(s_sr[:, 0], s_sr[:, 1], color="red", lw=2.5, label="LOESS")
except: pass
axes[0, 1].set_xlabel("拟合值"); axes[0, 1].set_ylabel("标准化残差")
axes[0, 1].set_title("标准化残差 vs 拟合值"); axes[0, 1].legend(fontsize=8)

# (3) 残差绝对值 vs 拟合值（检验异方差）
axes[1, 0].scatter(fitted, abs_resid, alpha=0.5, s=20, color="#C44E52")
try:
    s_ar = lowess(abs_resid, fitted, frac=0.4, return_sorted=True)
    axes[1, 0].plot(s_ar[:, 0], s_ar[:, 1], color="red", lw=2.5, label="LOESS")
except: pass
axes[1, 0].set_xlabel("拟合值"); axes[1, 0].set_ylabel("|残差|")
axes[1, 0].set_title("残差绝对值 vs 拟合值（异方差诊断）"); axes[1, 0].legend(fontsize=8)

# (4) 残差 Q-Q 图
sm.qqplot(resid, stats.norm, fit=True, line="45", ax=axes[1, 1],
          markerfacecolor="#8172B2", markersize=5)
axes[1, 1].set_title("残差 Q-Q 图")
save_fig("model_residuals_diagnostics.png")

# ── 随机效应 Q-Q 图 ──
fig, ax = plt.subplots(figsize=(7, 6))
re_vals = np.array(list(final_best.random_effects.values()))
sm.qqplot(re_vals, stats.norm, fit=True, line="45", ax=ax,
          markerfacecolor="#55A868", markersize=6)
ax.set_title("随机截距 Q-Q 图", fontsize=14)
ax.set_xlabel("理论分位数"); ax.set_ylabel("样本分位数")
save_fig("model_qq_random_effects.png")

# ================================================================
# 阶段5：健康 vs 不健康 对比
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段5：健康 vs 不健康 对比")

df_h = df[df["胎儿是否健康"] == "是"]
df_u = df[df["胎儿是否健康"] == "否"]

for lb, d in [("健康", df_h), ("不健康", df_u)]:
    s = d["Y染色体浓度"].dropna()
    tee_print(f"  {lb}（n={len(s)}）：均值={s.mean():.6f}, 标准差={s.std():.6f}, 中位数={s.median():.6f}")

u_stat, u_p = mannwhitneyu(df_h["Y染色体浓度"].dropna(), df_u["Y染色体浓度"].dropna())
tee_print(f"  Mann-Whitney U = {u_stat:.1f}, p = {u_p:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
sns.violinplot(data=df, x="胎儿是否健康", y="Y染色体浓度",
               palette={"是": "#55A868", "否": "#C44E52"}, inner="quartile", ax=axes[0])
axes[0].set_title(f"Y染色体浓度 按健康状态分组\nMann-Whitney U 检验 p={u_p:.4f}", fontsize=13)
sns.boxplot(data=df, x="IVF妊娠", y="Y染色体浓度", hue="胎儿是否健康",
            palette={"是": "#55A868", "否": "#C44E52"}, ax=axes[1])
axes[1].set_title("Y染色体浓度 按 IVF×健康 分组", fontsize=13)
axes[1].legend(title="是否健康")
save_fig("coef_comparison_health.png")

# ================================================================
# 阶段6：非线性验证与可视化
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段6：非线性验证与二次曲线可视化")

# ── 用原始单位跑二次 LMM（方便可视化）──
d_q = df[["Y染色体浓度", "孕周数值", "孕妇代码"]].dropna()
d_q["孕周2"] = d_q["孕周数值"] ** 2
m_lin = smf.mixedlm("Y染色体浓度 ~ 孕周数值", d_q, groups=d_q["孕妇代码"]).fit(reml=False)
m_quad = smf.mixedlm("Y染色体浓度 ~ 孕周数值 + 孕周2", d_q, groups=d_q["孕妇代码"]).fit(reml=False)
lrt_q = -2 * (m_lin.llf - m_quad.llf)
p_q = stats.chi2.sf(lrt_q, 1)
tee_print(f"二次项 LRT (ML, 原始单位): χ²={lrt_q:.2f}, p={p_q:.4f}")
tee_print(f"  线性模型 AIC={m_lin.aic:.1f}, 二次模型 AIC={m_quad.aic:.1f}")

# ── 二次曲线可视化 + 拐点标注 ──
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
xs = np.linspace(d_q["孕周数值"].min(), d_q["孕周数值"].max(), 300)

# 左图：线性 vs 二次拟合
axes[0].scatter(d_q["孕周数值"], d_q["Y染色体浓度"], alpha=0.3, s=20, color="#4C72B0")
b0, b1 = m_lin.fe_params[["Intercept", "孕周数值"]]
axes[0].plot(xs, b0 + b1 * xs, color="green", lw=2.5, label=f"线性 (AIC={m_lin.aic:.0f})")
b0q, b1q, b2q = m_quad.fe_params[["Intercept", "孕周数值", "孕周2"]]
axes[0].plot(xs, b0q + b1q * xs + b2q * xs**2, color="red", ls="--", lw=2.5,
             label=f"二次 (AIC={m_quad.aic:.0f})")
# 标注拐点
if abs(b2q) > 1e-12:
    t_min_raw = -b1q / (2 * b2q)
    if 8 <= t_min_raw <= 20:
        y_min = b0q + b1q * t_min_raw + b2q * t_min_raw**2
        axes[0].axvline(t_min_raw, color="purple", ls=":", lw=2, alpha=0.8)
        axes[0].annotate(f"拐点 ≈ {t_min_raw:.1f}周", xy=(t_min_raw, y_min),
                         xytext=(t_min_raw + 2.5, y_min + 0.02),
                         arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
                         fontsize=11, color="purple", fontweight="bold")
        tee_print(f"  原始单位拐点: t_min = {t_min_raw:.2f} 周")
axes[0].set_title(f"线性 vs 二次拟合（LRT p={p_q:.4f}）", fontsize=13)
axes[0].legend(); axes[0].set_xlabel("孕周（周）"); axes[0].set_ylabel("Y染色体浓度")

# 右图：抽血次数分组箱线图
sns.boxplot(data=df, x="检测抽血次数", y="Y染色体浓度", palette="Set3", ax=axes[1])
axes[1].set_title("Y染色体浓度 按抽血次数分组", fontsize=13)
axes[1].set_xlabel("抽血次数"); axes[1].set_ylabel("Y染色体浓度")
save_fig("scatter_nonlinear_and_bloodtime.png")

# ================================================================
tee_print("\n" + "=" * 60)
tee_print("全部完成！")
for c in sorted(os.listdir(RESULT_DIR)):
    tee_print(f"  result/{c}")

_log_file.write("```\n")
_log_file.close()
