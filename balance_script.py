import os
import shutil
import pandas as pd

# ===== 設定 =====
# プロジェクトのルートからの相対パスで指定する想定
CSV_PATH = "datacollector/dataset/data_labels_updated.csv"   # 元のCSV
NEW_CSV_PATH = "datacollector/dataset/data_labels_balanced.csv"  # 出力CSV

# 画像ファイルが置いてあるフォルダ
# CSVの image_path が相対パスなら、その起点になるフォルダ
DATASET_ROOT = "datacollector/dataset"

# "up" なら少ないクラスを増やす（オーバーサンプリング）
# "down" なら多いクラスを減らす（アンダーサンプリング）
BALANCE_MODE = "up"   # "up" か "down" に変更

# True にすると画像ファイルも複製して増やす
# False なら CSV の行だけ増やして、image_path は元のファイルを指したまま
COPY_IMAGES = False

# 乱数シード（再現性のため）
RANDOM_STATE = 42

# 角度を表す列名
ANGLE_COL = "servo_angle"
# 画像パスを表す列名
IMG_COL = "image_path"


def duplicate_rows_with_image_copy(group, need, dataset_root, img_col):
    """行を複製しつつ画像ファイルもコピーする"""
    sampled = group.sample(n=need, replace=True, random_state=RANDOM_STATE)
    new_rows = []

    for _, row in sampled.iterrows():
        orig_rel = row[img_col]              # 相対パス想定
        orig_abs = os.path.join(dataset_root, orig_rel)

        base, ext = os.path.splitext(orig_rel)
        idx = 1
        while True:
            new_rel = f"{base}_dup{idx}{ext}"
            new_abs = os.path.join(dataset_root, new_rel)
            if not os.path.exists(new_abs):
                break
            idx += 1

        # 画像ファイルをコピー
        shutil.copy2(orig_abs, new_abs)

        # CSV行を複製してパスだけ差し替える
        new_row = row.copy()
        new_row[img_col] = new_rel
        new_rows.append(new_row)

    return pd.DataFrame(new_rows)


def main():
    df = pd.read_csv(CSV_PATH)

    # 各角度の枚数を確認
    counts = df[ANGLE_COL].value_counts().sort_index()
    print("before:")
    print(counts)

    if BALANCE_MODE == "up":
        target = counts.max()
        print(f"\nターゲット枚数（各角度）：{target}（最大に合わせて増やす）")
    elif BALANCE_MODE == "down":
        target = counts.min()
        print(f"\nターゲット枚数（各角度）：{target}（最小に合わせて減らす）")
    else:
        raise ValueError("BALANCE_MODE は 'up' か 'down' を指定してください")

    new_parts = []

    # 角度ごとに処理
    for angle, group in df.groupby(ANGLE_COL):
        n = len(group)
        if n == target:
            print(f"angle={angle}: {n} 枚 → そのまま")
            new_parts.append(group)
            continue

        if BALANCE_MODE == "up":
            need = target - n
            print(f"angle={angle}: {n} 枚 → {need} 枚追加して {target} 枚に")
            if need > 0:
                if COPY_IMAGES:
                    added = duplicate_rows_with_image_copy(group, need, DATASET_ROOT, IMG_COL)
                else:
                    added = group.sample(n=need, replace=True, random_state=RANDOM_STATE)
                new_group = pd.concat([group, added], ignore_index=True)
            else:
                new_group = group

        else:  # "down"
            take = target
            print(f"angle={angle}: {n} 枚 → {take} 枚に間引き")
            new_group = group.sample(n=take, replace=False, random_state=RANDOM_STATE)

        new_parts.append(new_group)

    # 結合してシャッフル
    new_df = pd.concat(new_parts, ignore_index=True)
    new_df = new_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # 保存
    os.makedirs(os.path.dirname(NEW_CSV_PATH), exist_ok=True)
    new_df.to_csv(NEW_CSV_PATH, index=False, encoding="utf-8")

    print(f"\n保存完了: {NEW_CSV_PATH}")
    print("after:")
    print(new_df[ANGLE_COL].value_counts().sort_index())


if __name__ == "__main__":
    main()
