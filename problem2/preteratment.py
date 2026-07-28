"""
问题二数据准备：BMI波动分析 → K-Means聚类 → 双轨分组 → 组间差异检验
==============================================================
1. 分析每人BMI的变异系数，论证首次BMI的代表性
2. K-Means 聚类（全部605行），肘部法确定最优k
3. 双轨分组：数据驱动(K-Means边界) + 经验参考([20,28),[28,32),...)
4. 每人按"首次检测BMI"分配进组（敏感性：均值BMI备选）
5. Kruskal-Wallis 检验各组达标周差异
6. 输出 data_modeling_p2.xlsx
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy.cluster.vq import kmeans2
from scipy import stats
import warnings
import os
warnings.filterwarnings("ignore")

# ================================================================
# 路径与字体
# ================================================================
INPUT_FILE = r"../data_modeling.xlsx"
OUTPUT_FILE = r"../data_modeling_p2.xlsx"
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
# 1. 加载数据
# ================================================================
print("=" * 60)
print("问题二 · 数据准备")
print("=" * 60)

print(f"\n读取: {INPUT_FILE}")
df = pd.read_excel(INPUT_FILE)
print(f"  总记录: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"  孕妇数: {df['孕妇代码'].nunique()} 人")

# 提取核心列
cols_keep = ["孕妇代码", "Y染色体浓度", "孕周数值", "孕妇BMI", "胎儿是否健康", "检测抽血次数"]
df = df[cols_keep].copy()
print(f"  使用列: {cols_keep}")

# ================================================================
# 2. BMI 波动分析（每人 BMI 变异系数）
# ================================================================
print("\n" + "=" * 60)
print("BMI 波动分析")

person_bmi = df.groupby("孕妇代码").agg(
    BMI均值=("孕妇BMI", "mean"),
    BMI标准差=("孕妇BMI", "std"),
    BMI首次=("孕妇BMI", "first"),
    BMI末次=("孕妇BMI", "last"),
    测量次数=("孕妇BMI", "count"),
    首次孕周=("孕周数值", "min"),
).reset_index()

person_bmi["BMI变异系数"] = person_bmi["BMI标准差"] / person_bmi["BMI均值"]
person_bmi["BMI变化量"] = person_bmi["BMI末次"] - person_bmi["BMI首次"]

# 统计
cv = person_bmi["BMI变异系数"].dropna()
cv_under_5pct = (cv < 0.05).mean()
cv_under_3pct = (cv < 0.03).mean()
print(f"  BMI 变异系数: 均值={cv.mean():.4f}, 中位数={cv.median():.4f}")
print(f"  CV < 3%: {cv_under_3pct:.1%}, CV < 5%: {cv_under_5pct:.1%}")
print(f"  BMI 变化量(末次-首次): 均值={person_bmi['BMI变化量'].mean():.2f}")

# 画 CV 分布直方图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(cv, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(cv.median(), color="red", ls="--", lw=2, label=f"中位数={cv.median():.4f}")
axes[0].axvline(0.05, color="orange", ls=":", lw=2, label="5% 阈值")
axes[0].set_title(f"每人 BMI 变异系数分布\n（{cv_under_5pct:.0%} 的孕妇 CV < 5%）", fontsize=13)
axes[0].set_xlabel("BMI 变异系数 (CV)")
axes[0].set_ylabel("人数")
axes[0].legend()

axes[1].scatter(person_bmi["BMI首次"], person_bmi["BMI变化量"],
                alpha=0.5, s=30, c="#55A868", edgecolors="white")
axes[1].axhline(0, color="gray", ls="--")
axes[1].set_title("BMI 变化量 vs 首次 BMI", fontsize=13)
axes[1].set_xlabel("首次检测 BMI")
axes[1].set_ylabel("BMI 变化量（末次 − 首次）")
save_fig("bmi_cv_distribution.png")

# ================================================================
# 3. K-Means 聚类（全部 605 行）
# ================================================================
print("\n" + "=" * 60)
print("K-Means 聚类（全部 605 行 BMI 数据）")

bmi_data = df[["孕妇BMI"]].values.astype(np.float64)

k_range = range(2, 8)
sse_list = []
all_results = {}

np.random.seed(42)
for k in k_range:
    centroids, labels = kmeans2(bmi_data, k, minit="points", missing="warn")
    sse = sum(((bmi_data[labels == i] - centroids[i]) ** 2).sum() for i in range(k))
    sse_list.append(sse)
    all_results[k] = {"centroids": centroids.flatten(), "labels": labels, "sse": sse}
    print(f"  k={k}: SSE={sse:.3f}, 聚类中心={np.sort(centroids.flatten()).round(2)}")

# 肘部图
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_range), sse_list, "o-", color="#4C72B0", lw=2, ms=8, markerfacecolor="white")
ax.set_xlabel("聚类数 k", fontsize=12)
ax.set_ylabel("簇内误差平方和 (SSE)", fontsize=12)
ax.set_title("K-Means 肘部图 — BMI 聚类最优 k 选择", fontsize=14)
ax.set_xticks(list(k_range))
# 找出肘部点：最大曲率法
deltas = np.diff(sse_list)
deltas2 = np.diff(deltas)
if len(deltas2) > 0:
    elbow_idx = np.argmax(np.abs(deltas2)) + 2  # +2 because diff twice
    ax.annotate(f"建议 k={k_range[elbow_idx]}", 
                (k_range[elbow_idx], sse_list[elbow_idx]),
                xytext=(k_range[elbow_idx]+0.5, sse_list[elbow_idx]+sse_list[0]*0.05),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=12, color="red")
save_fig("kmeans_elbow.png")

# 确定最优 k（最大曲率）
optimal_k = k_range[elbow_idx] if len(deltas2) > 0 else 5
print(f"\n  肘部法最优 k = {optimal_k}")
opt_centroids = np.sort(all_results[optimal_k]["centroids"])
print(f"  聚类中心: {opt_centroids.round(2)}")

# 生成 K-Means 分组边界（相邻聚类中心的中点）
kmeans_boundaries = []
for i in range(len(opt_centroids) - 1):
    boundary = (opt_centroids[i] + opt_centroids[i+1]) / 2
    kmeans_boundaries.append(round(boundary, 1))
print(f"  分组边界: {kmeans_boundaries}")

# ================================================================
# 4. 双轨分组方案
# ================================================================
print("\n" + "=" * 60)
print("双轨分组方案")

# 方案A：K-Means 数据驱动边界
def build_bins_a(boundaries):
    """根据聚类边界生成分组区间"""
    bins = [20.0] + boundaries + [float("inf")]
    labels = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i+1]
        if hi == float("inf"):
            labels.append(f"≥{lo:.0f}")
        else:
            labels.append(f"[{lo:.0f},{hi:.0f})")
    return bins, labels

bins_a, labels_a = build_bins_a(kmeans_boundaries)
print(f"  方案A（K-Means, k={optimal_k}）: {labels_a}")

# 方案B：经验参考分组
bins_b = [20, 28, 32, 36, 40, float("inf")]
labels_b = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "≥40"]
print(f"  方案B（经验分组）: {labels_b}")

# ================================================================
# 5. 分组分配（每人用首次检测 BMI）
# ================================================================
print("\n" + "=" * 60)
print("分组分配（首次检测 BMI）")

# 合并首次 BMI 到主表
df = df.merge(person_bmi[["孕妇代码", "BMI首次", "BMI均值", "BMI变异系数", "测量次数"]],
              on="孕妇代码", how="left")

# 方案A 分组
df["BMI分组_A"] = pd.cut(df["BMI首次"], bins=bins_a, labels=labels_a, right=False)
# 方案B 分组
df["BMI分组_B"] = pd.cut(df["BMI首次"], bins=bins_b, labels=labels_b, right=False)

# 各分组统计
for scheme, col in [("方案A(K-Means)", "BMI分组_A"), ("方案B(经验)", "BMI分组_B")]:
    print(f"\n  {scheme}:")
    grp = df.groupby(col, observed=False)
    for name, g in grp:
        n_person = g["孕妇代码"].nunique()
        print(f"    {name}: {len(g)} 条记录, {n_person} 人, 首次BMI均值={g['BMI首次'].mean():.2f}")

# ================================================================
# 6. Kruskal-Wallis 预检验（各组达标周差异）
# ================================================================
print("\n" + "=" * 60)
print("Kruskal-Wallis 检验（各组达标周差异）")

# 用 M1 模型常数快速估算每人达标周（不考虑误差）
INTERCEPT = 0.078
BETA_WEEK_Z = 0.012
BETA_BMI_Z = -0.004
WEEK_MEAN = 16.4973
WEEK_STD = 3.9501
BMI_MEAN = 32.2656
BMI_STD = 2.8433
TARGET = 0.04

def estimate_week(bmi):
    """不考虑误差时，反解达标孕周"""
    bmi_z = (bmi - BMI_MEAN) / BMI_STD
    week = ((TARGET - INTERCEPT + BETA_BMI_Z * bmi_z) / BETA_WEEK_Z) * WEEK_STD + WEEK_MEAN
    return week

# 每人用首次 BMI 估算
person_bmi["估算达标周"] = person_bmi["BMI首次"].apply(estimate_week)

# 方案A
groups_a = []
for name in labels_a:
    w = person_bmi[person_bmi["BMI首次"].apply(lambda x: pd.cut([x], bins=bins_a, labels=labels_a, right=False)[0]) == name]["估算达标周"].dropna().values
    groups_a.append(w)

h_a, p_a = stats.kruskal(*groups_a)
print(f"  方案A: H={h_a:.2f}, p={p_a:.6f} {'***' if p_a < 0.001 else ('**' if p_a < 0.01 else ('*' if p_a < 0.05 else 'ns'))}")

# 方案B
groups_b = []
for name in labels_b:
    w = person_bmi[person_bmi["BMI首次"].apply(lambda x: pd.cut([x], bins=bins_b, labels=labels_b, right=False)[0]) == name]["估算达标周"].dropna().values
    groups_b.append(w)

h_b, p_b = stats.kruskal(*groups_b)
print(f"  方案B: H={h_b:.2f}, p={p_b:.6f} {'***' if p_b < 0.001 else ('**' if p_b < 0.01 else ('*' if p_b < 0.05 else 'ns'))}")

# ================================================================
# 7. 输出
# ================================================================
print("\n" + "=" * 60)
print(f"写入: {OUTPUT_FILE}")
df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ 完成！最终: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n列清单 ({len(df.columns)} 列):")
for i, col in enumerate(df.columns, 1):
    print(f"  [{i:2d}] {col}  ({df[col].dtype})")
