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
INPUT_FILE = r"../data_modeling.xlsx"
RESULT_DIR = r"result"
os.makedirs(RESULT_DIR, exist_ok=True)

_OUTPUT_MD = os.path.join(RESULT_DIR, "analysis_output.md")
_log_file = open(_OUTPUT_MD, "w", encoding="utf-8")
_log_file.write("# data_analysis.py 完整终端输出\n\n```\n")

def tee_print(*args, **kwargs):
    """同时输出到终端和 md 日志文件"""
    import builtins
    builtins.print(*args, **kwargs)
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
tee_print("正在加载 data_modeling.xlsx ...")
df = pd.read_excel(INPUT_FILE)
n_rows = len(df)
n_subjects = df['孕妇代码'].nunique()
tee_print(f"  行数: {n_rows}, 列数: {len(df.columns)}, 受试者数: {n_subjects}")
tee_print(f"  人均测量次数: {n_rows / n_subjects:.1f} 次")
tee_print(f"  健康/不健康: {dict(df['胎儿是否健康'].value_counts())}")

# ================================================================
# 阶段1：探索性数据分析（EDA）
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段1：探索性数据分析")

for col in ["Y染色体浓度", "孕周数值", "孕妇BMI", "年龄"]:
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
save_fig("hist_distributions.png")

# 箱线图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="IVF妊娠", y="Y染色体浓度", palette="Set2", ax=axes[0])
axes[0].set_title("Y染色体浓度 按 IVF妊娠 分组", fontsize=13)
sns.boxplot(data=df, x="胎儿是否健康", y="Y染色体浓度", palette="Set2", ax=axes[1])
axes[1].set_title("Y染色体浓度 按 胎儿是否健康 分组", fontsize=13)
save_fig("boxplot_by_group.png")

# ================================================================
# 阶段2：相关性分析
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段2：相关性分析")

corr_vars = ["Y染色体浓度", "孕周数值", "孕妇BMI", "年龄", "X染色体浓度", "GC含量"]
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
# 阶段4：线性混合模型（LMM）
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段4：线性混合模型")

df_m = df[["Y染色体浓度", "孕周数值", "孕妇BMI", "年龄", "IVF妊娠", "孕妇代码"]].dropna().copy()
df_m["IVF_试管"] = (df_m["IVF妊娠"] == "IVF（试管婴儿）").astype(int)
df_m["IVF_人授"] = (df_m["IVF妊娠"] == "IUI（人工授精）").astype(int)
for c in ["孕周数值", "孕妇BMI", "年龄"]:
    df_m[f"{c}_z"] = (df_m[c] - df_m[c].mean()) / df_m[c].std()

models, results = {}, {}

tee_print("\n  M0：空模型（ICC）")
r0 = smf.mixedlm("Y染色体浓度 ~ 1", df_m, groups=df_m["孕妇代码"]).fit(reml=True)
models["M0"] = "Null"; results["M0"] = r0
vr = r0.cov_re.values[0, 0] if hasattr(r0.cov_re, 'values') else r0.cov_re.iloc[0, 0]
icc = vr / (vr + r0.scale)
tee_print(f"    ICC = {icc:.4f}")

tee_print("\n  M1：+ 孕周 + BMI")
r1 = smf.mixedlm("Y染色体浓度 ~ 孕周数值_z + 孕妇BMI_z", df_m, groups=df_m["孕妇代码"]).fit(reml=True)
models["M1"] = "+Week+BMI"; results["M1"] = r1
tee_print(r1.summary().tables[1])

tee_print("\n  M2：+ 年龄 + IVF")
r2 = smf.mixedlm("Y染色体浓度 ~ 孕周数值_z + 孕妇BMI_z + 年龄_z + IVF_试管 + IVF_人授",
                  df_m, groups=df_m["孕妇代码"]).fit(reml=True)
models["M2"] = "+Age+IVF"; results["M2"] = r2
tee_print(r2.summary().tables[1])

tee_print("\n  M3：+ 孕周 × BMI 交互项")
try:
    r3 = smf.mixedlm("Y染色体浓度 ~ 孕周数值_z * 孕妇BMI_z + 年龄_z + IVF_试管 + IVF_人授",
                      df_m, groups=df_m["孕妇代码"]).fit(reml=True)
    models["M3"] = "+Interaction"; results["M3"] = r3
    tee_print(r3.summary().tables[1])
except Exception as e:
    tee_print(f"    M3 未收敛: {e}")

tee_print(f"\n模型比较:")
tee_print(f"  {'Model':<8s} {'AIC':>10s} {'BIC':>10s} {'-2LL':>10s}")
for n, r in results.items():
    tee_print(f"  {n:<8s} {r.aic:>10.1f} {r.bic:>10.1f} {-2*r.llf:>10.1f}")

for a, b, dfd in [("M0", "M1", 2), ("M1", "M2", 3)]:
    if results.get(a) and results.get(b):
        lrt = -2 * (results[a].llf - results[b].llf)
        p = stats.chi2.sf(lrt, df=dfd)
        tee_print(f"  LRT {a} vs {b}: χ²={lrt:.2f}, df={dfd}, p={p:.4f}")

tee_print(f"\nR²（Nakagawa 近似）:")
for n in ["M1", "M2"]:
    if n in results:
        r = results[n]
        vf = np.var(r.fittedvalues - r.fittedvalues.mean())
        vr2 = r.cov_re.values[0, 0] if hasattr(r.cov_re, 'values') else r.cov_re.iloc[0, 0]
        ve = r.scale
        mr2 = vf / (vf + vr2 + ve)
        cr2 = (vf + vr2) / (vf + vr2 + ve)
        tee_print(f"  {n}: Marginal R2={mr2:.4f}, Conditional R2={cr2:.4f}")

# 森林图
best = results.get("M2") or results.get("M1")
if best is not None:
    fig, ax = plt.subplots(figsize=(10, 5))
    coef = best.params.to_frame("coef")
    ci = best.conf_int()
    coef["low"] = ci.iloc[:, 0]; coef["high"] = ci.iloc[:, 1]
    coef = coef.drop("Group Var", errors="ignore")
    names = {"Intercept": "截距", "孕周数值_z": "孕周（标准化）",
             "孕妇BMI_z": "BMI（标准化）", "年龄_z": "年龄（标准化）",
             "IVF_试管": "IVF-试管", "IVF_人授": "IUI-人授"}
    coef["label"] = [names.get(i, i) for i in coef.index]
    yp = range(len(coef))
    ax.errorbar(coef["coef"], yp, xerr=[coef["coef"] - coef["low"], coef["high"] - coef["coef"]],
                fmt="o", capsize=4, color="#4C72B0", ms=8, lw=2)
    ax.axvline(0, color="gray", ls="--")
    ax.set_yticks(yp); ax.set_yticklabels(coef["label"]); ax.invert_yaxis()
    ax.set_xlabel("系数估计值（95% 置信区间）")
    ax.set_title("LMM 固定效应（模型 M2）", fontsize=14)
save_fig("model_forest_plot.png")

# 残差诊断
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fitted = best.fittedvalues; resid = best.resid
axes[0].scatter(fitted, resid, alpha=0.5, s=20, color="#4C72B0")
axes[0].axhline(0, color="red", ls="--")
axes[0].set_xlabel("拟合值"); axes[0].set_ylabel("残差")
axes[0].set_title("残差与拟合值")
sm.qqplot(resid, stats.norm, fit=True, line="45", ax=axes[1],
          markerfacecolor="#4C72B0", markersize=4)
axes[1].set_title("残差 Q-Q 图")
axes[1].set_xlabel("理论分位数"); axes[1].set_ylabel("样本分位数")
axes[2].hist(resid, bins=30, color="#55A868", edgecolor="white", alpha=0.8, density=True)
xr = np.linspace(resid.min(), resid.max(), 200)
axes[2].plot(xr, stats.norm.pdf(xr, resid.mean(), resid.std()), color="red", lw=2)
axes[2].set_title("残差分布")
save_fig("model_residuals_fitted.png")

# 随机效应 Q-Q 图
fig, ax = plt.subplots(figsize=(7, 6))
re_vals = np.array(list(best.random_effects.values()))
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
# 阶段6：补充分析
# ================================================================
tee_print("\n" + "=" * 60)
tee_print("阶段6：非线性检验")

d_q = df[["Y染色体浓度", "孕周数值", "孕妇代码"]].dropna()
d_q["孕周2"] = d_q["孕周数值"] ** 2
# 注意：LRT 比较不同固定效应结构时，必须用 ML（reml=False），不能用 REML
m_lin = smf.mixedlm("Y染色体浓度 ~ 孕周数值", d_q, groups=d_q["孕妇代码"]).fit(reml=False)
m_quad = smf.mixedlm("Y染色体浓度 ~ 孕周数值 + 孕周2", d_q, groups=d_q["孕妇代码"]).fit(reml=False)
lrt_q = -2 * (m_lin.llf - m_quad.llf)
p_q = stats.chi2.sf(lrt_q, 1)
tee_print(f"二次项 LRT: χ²={lrt_q:.2f}, p={p_q:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
xs = np.linspace(d_q["孕周数值"].min(), d_q["孕周数值"].max(), 200)
axes[0].scatter(d_q["孕周数值"], d_q["Y染色体浓度"], alpha=0.3, s=20, color="#4C72B0")
b0, b1 = m_lin.fe_params[["Intercept", "孕周数值"]]
axes[0].plot(xs, b0 + b1 * xs, color="green", lw=2.5, label="线性拟合")
b0q, b1q, b2q = m_quad.fe_params[["Intercept", "孕周数值", "孕周2"]]
axes[0].plot(xs, b0q + b1q * xs + b2q * xs**2, color="red", ls="--", lw=2.5, label="二次拟合")
axes[0].set_title(f"线性与二次拟合（似然比检验 p={p_q:.4f}）", fontsize=13)
axes[0].legend(); axes[0].set_xlabel("孕周（周）"); axes[0].set_ylabel("Y染色体浓度")

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
