import os
import cv2
import pandas as pd

CSV_PATH = "datacollector/dataset/data_labels_updated.csv"   # ← 今使ってるCSV
DATASET_ROOT = "datacollector/dataset"
NEW_CSV_PATH = "datacollector/dataset/data_labels_clean.csv"

IMG_COL = "image_path"

def is_valid_image(rel_path: str) -> bool:
    abs_path = os.path.join(DATASET_ROOT, rel_path)

    # ① ファイルが存在しない場合
    if not os.path.exists(abs_path):
        return False

    # ② サイズが0バイトの場合（破損ファイル）
    if os.path.getsize(abs_path) == 0:
        return False

    # ③ OpenCV で読み込めない場合（読み取り不能）
    img = cv2.imread(abs_path)
    if img is None:
        return False

    return True

def main():
    df = pd.read_csv(CSV_PATH)
    print("総レコード数:", len(df))

    valid_mask = df[IMG_COL].apply(is_valid_image)

    ok_df = df[valid_mask]
    bad_df = df[~valid_mask]

    print("正常レコード数:", len(ok_df))
    print("壊れている or 欠損レコード数:", len(bad_df))

    if len(bad_df) > 0:
        print("\n--- 壊れている画像名の例 ---")
        print(bad_df[IMG_COL].head())

    ok_df.to_csv(NEW_CSV_PATH, index=False, encoding="utf-8")
    print("\n保存完了:", os.path.abspath(NEW_CSV_PATH))

if __name__ == "__main__":
    main()
