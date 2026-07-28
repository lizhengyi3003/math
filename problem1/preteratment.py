"""
第二阶段数据处理：GC过滤 → 去重合并 → 构造建模特征
=====================================================
1. GC含量列过滤 (40%-60%，仅主GC列)
2. 合并同一孕妇+同次采血的重复记录（数值列取均值）
3. 构造孕周数值特征（"13w+6" → 13.857）
4. ★ 同时处理男胎和女胎数据，输出两个 Sheet
"""
import pandas as pd
import numpy as np
import re

INPUT_FILE = r"../data_processed.xlsx"
OUTPUT_FILE = r"../data_modeling.xlsx"

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


def process_sheet(df, sheet_label):
    """对单个 Sheet 执行 GC过滤 → 去重 → 孕周数值化"""
    # --- GC含量过滤 ---
    before = len(df)
    mask = (df["GC含量"] >= 0.40) & (df["GC含量"] <= 0.60)
    df = df[mask].copy()
    print(f"  GC过滤 (40%-60%): 删除 {before - len(df)} 行, 保留 {len(df)} 行")

    # --- 合并重复记录 ---
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
    print(f"  重复记录: {dup_mask.sum()} 条 (共 {dup_groups} 组)")

    df = df.groupby(group_keys, as_index=False).agg(agg_dict)
    print(f"  合并: {before_merge} → {len(df)} 行 (减少 {before_merge - len(df)} 行)")

    # --- 构造孕周数值 ---
    df["孕周数值"] = df["检测孕周"].apply(parse_week_to_float)
    samples = list(zip(df["检测孕周"].head(3), df["孕周数值"].head(3)))
    for gw, gv in samples:
        print(f"  孕周样例: {gw} → {gv}")

    return df


# ============================================================
# 主流程
# ============================================================

print("=" * 60)
print("第二阶段数据处理（男胎 + 女胎）")
print("=" * 60)

all_processed = {}

for sheet_name in ["男胎检测数据", "女胎检测数据"]:
    print(f"\n{'─' * 50}")
    print(f"处理: [{sheet_name}]")

    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
    print(f"  原始: {df.shape[0]} 行 × {df.shape[1]} 列")

    df_processed = process_sheet(df, sheet_name)

    out_sheet = sheet_name.replace("检测数据", "建模数据")
    all_processed[out_sheet] = df_processed

# ============================================================
# 输出
# ============================================================
print(f"\n{'=' * 60}")
print(f"写入: {OUTPUT_FILE}")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    for sheet_name, df in all_processed.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  [{sheet_name}] {df.shape[0]} 行 × {df.shape[1]} 列")

print(f"\n✅ 完成！")
print(f"\n汇总:")
for sheet_name, df in all_processed.items():
    n_subj = df["孕妇代码"].nunique()
    print(f"  {sheet_name}: {df.shape[0]} 行, {n_subj} 人, {df.shape[1]} 列")
