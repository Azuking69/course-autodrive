import pandas as pd

# 入力CSV
INPUT_CSV = "datacollector/dataset/data_labels_clean.csv"
OUTPUT_CSV = "datacollector/dataset/data_labels_balanced.csv"

# 目標比率（合計 = 1.0）
target_ratio = {
    1: 0.26,
    40: 0.26,
    90: 0.26,
    120: 0.14,
    150: 0.08
}

df = pd.read_csv(INPUT_CSV)

# servo_angle を整数に統一（超重要）
df["servo_angle"] = df["servo_angle"].astype(int)

total_count = len(df)

result = []

for angle, ratio in target_ratio.items():
    target_count = int(total_count * ratio)
    sub = df[df["servo_angle"] == angle]

    if len(sub) == 0:
        print(f"[WARN] angle {angle} has no data")
        continue

    if len(sub) > target_count:
        # 多すぎ → 減らす
        sampled = sub.sample(target_count, random_state=0)
    else:
        # 少なすぎ → 増やす
        sampled = sub.sample(target_count, replace=True, random_state=0)

    result.append(sampled)

df_final = pd.concat(result, ignore_index=True)

# シャッフル
df_final = df_final.sample(frac=1, random_state=0).reset_index(drop=True)

print("=== Final distribution ===")
print(df_final["servo_angle"].value_counts(normalize=True) * 100)

df_final.to_csv(OUTPUT_CSV, index=False)
