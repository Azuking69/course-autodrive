import os
import csv
import random
import shutil

# 1. 경로 설정
base_dir = r"C:\Users\USER\Desktop\dataset"      # 이미지들이 들어있는 폴더
csv_path = r"C:\Users\USER\Desktop\dataset\data_labels.csv"  # 기존 CSV 파일 경로
output_base_dir = os.path.join(base_dir, "balanced")        # 새로 만들 폴더 묶음
output_csv_path = os.path.join(output_base_dir, "balanced_dataset.csv")

# 2. 설정값
angles = [30, 60, 90, 120, 150]
target_count = 114

# 각 각도별로 CSV 행을 모아둘 딕셔너리
rows_by_angle = {a: [] for a in angles}

# 3. 기존 CSV 읽어서 각도별로 묶기
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            angle = int(row["servo_angle"])
        except (KeyError, ValueError):
            continue

        if angle in rows_by_angle:
            rows_by_angle[angle].append(row)

# 새 CSV에 쓸 행들을 모아둘 리스트
selected_rows = []

# 4. 각 각도별로 114개씩 샘플링해서 이미지 복사 + 새 CSV용 행 만들기
for angle in angles:
    rows = rows_by_angle[angle]
    print(f"{angle}도: 총 {len(rows)}개")

    if len(rows) < target_count:
        print(f"  → {angle}도는 {len(rows)}개라서 114개로 맞출 수 없음")
        continue

    sampled = random.sample(rows, target_count)

    # balanced_각도 폴더 만들기
    out_dir = os.path.join(output_base_dir, f"balanced_{angle}")
    os.makedirs(out_dir, exist_ok=True)

    for row in sampled:
        # 원본 이미지 경로 만들기
        # image_path에 파일 이름만 들어 있다고 가정
        filename = os.path.basename(row["image_path"])
        src_path = os.path.join(base_dir, filename)

        if not os.path.isfile(src_path):
            print(f"  [경고] 파일 없음: {src_path}")
            continue

        # 새 위치로 복사
        dst_filename = filename
        dst_path = os.path.join(out_dir, dst_filename)
        shutil.copy(src_path, dst_path)

        # 새 CSV에 쓸 행 만들기 (image_path를 balanced 폴더 기준으로 수정)
        new_row = row.copy()
        new_row["image_path"] = os.path.join(f"balanced_{angle}", dst_filename)
        selected_rows.append(new_row)

    print(f"  → {angle}도 114장 복사 완료: {out_dir}")

# 5. 새 CSV 저장
if selected_rows:
    os.makedirs(output_base_dir, exist_ok=True)

    fieldnames = ["timestamp", "image_path", "servo_angle", "dc_motor_speed"]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"새 CSV 생성 완료: {output_csv_path}")
else:
    print("선택된 데이터가 없어서 CSV를 만들지 못했음.")
