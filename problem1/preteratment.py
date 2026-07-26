"""
第二阶段数据处理：GC过滤 → 去重合并 → 构造建模特征
=====================================================
1. GC含量列过滤 (40%-60%，仅主GC列)
2. 仅保留男胎数据
3. 合并同一孕妇+同次采血的重复记录（数值列取均值）
4. 构造孕周数值特征（"13w+6" → 13.857）
"""
import pandas as pd
import numpy as np
import re

INPUT_FILE = r"../data_processed.xlsx"
OUTPUT_FILE = r"../data_modeling.xlsx"
SHEET_NAME = "男胎检测数据"

# ============================================================
# 工具函数
# ============================================================

def parse_week_to_float(val):
    """检测孕周字符串 → 浮点数周数（如 '13w+6' → 13.8571...）"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    m = re.match(r"^(\d{1,2})w\+(\d{1,2})$", s)
    if m:
        weeks = int(m.group(1))
        days = int(m.group(2))
        return round(weeks + days / 7.0, 6)
    try:
        return float(s)
    except ValueError:
        return np.nan


# ============================================================
# 主流程
# ============================================================

print("=" * 60)
print("第二阶段数据处理")
print("=" * 60)

# --- 读取 ---
print(f"\n读取: {INPUT_FILE} → [{SHEET_NAME}]")
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
print(f"  原始: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============================================================
# 1. GC含量过滤（仅主 GC含量 列，40%-60%）
# ============================================================
before = len(df)
mask = (df["GC含量"] >= 0.40) & (df["GC含量"] <= 0.60)
df = df[mask].copy()
print(f"\n✅ GC含量过滤 (40%-60%): 删除 {before - len(df)} 行, 保留 {len(df)} 行")

# ============================================================
# 2. 合并重复记录（同孕妇 + 同检测抽血次数 → 数值列取均值）
# ============================================================
group_keys = ["孕妇代码", "检测抽血次数"]

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
non_num_cols = [c for c in df.columns if c not in num_cols]

agg_dict = {}
for col in num_cols:
    if col not in group_keys:
        agg_dict[col] = "mean"
for col in non_num_cols:
    if col not in group_keys:
        agg_dict[col] = "first"

before_merge = len(df)
dup_mask = df.duplicated(subset=group_keys, keep=False)
dup_groups = df[dup_mask].groupby(group_keys).ngroups if dup_mask.sum() > 0 else 0
print(f"\n🔍 按 {group_keys} 分组聚合")
print(f"   重复记录: {dup_mask.sum()} 条 (共 {dup_groups} 组)")

df = df.groupby(group_keys, as_index=False).agg(agg_dict)
print(f"   ✅ 合并: {before_merge} → {len(df)} 行 (减少 {before_merge - len(df)} 行)")

# ============================================================
# 3. 构造孕周数值特征
# ============================================================
print(f"\n📐 构造孕周数值特征...")
df["孕周数值"] = df["检测孕周"].apply(parse_week_to_float)
samples = list(zip(df["检测孕周"].head(6), df["孕周数值"].head(6)))
for gw, gv in samples:
    print(f"   {gw} → {gv}")

# ============================================================
# 输出
# ============================================================
print(f"\n{'=' * 60}")
print(f"写入: {OUTPUT_FILE}")
df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ 完成！最终: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n列清单 ({len(df.columns)} 列):")
for i, col in enumerate(df.columns, 1):
    print(f"  [{i:2d}] {col}  ({df[col].dtype})")
