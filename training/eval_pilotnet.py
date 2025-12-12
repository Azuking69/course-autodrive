# training/eval_pilotnet.py
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.RCDataset import RCDataset
from preprocessor.RCPreprocessor import RCPreprocessor
from training.model import PilotNet


def evaluate_one_model(model_path: str):
    # 1. 設定（必要ならここだけ書き換え）
    csv_filename = "data_labels_balanced_clean.csv"
    dataset_root = "datacollector/dataset"
    batch_size = 128
    split_ratio = 0.8   # train_pilotnet.py と同じ

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] model_path = {model_path}")

    # 2. Dataset / DataLoader（test だけ作る）
    preproc = RCPreprocessor(
        out_size=(200, 66),
        crop_top_ratio=0.4,
        crop_bottom_ratio=1.0
    )

    test_dataset = RCDataset(
        csv_filename=csv_filename,
        root=dataset_root,
        preprocessor=preproc,
        augmentor=None,
        split="test",
        split_ratio=split_ratio,
    )

    num_classes = len(test_dataset.angles)
    print(f"[INFO] num_classes = {num_classes}")
    print(f"[INFO] test_samples = {len(test_dataset)}")

    pin_memory = (device.type == "cuda")
    num_workers = 12 if device.type == "cuda" else 4

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # 3. モデル読み込み
    model = PilotNet(num_classes=num_classes, input_shape=(3, 66, 200)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4. 推論して予測と正解を全部集める
    all_preds = []
    all_labels = []

    start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    elapsed = time.time() - start
    print(f"[INFO] inference time = {elapsed:.2f}s")

    # 5. Confusion Matrix
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        conf_mat[t, p] += 1

    # 6. 指標計算（Accuracy / Precision / Recall / F1）
    total = conf_mat.sum()
    correct = np.trace(conf_mat)
    accuracy = correct / total if total > 0 else 0.0

    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    macro_precision = float(np.mean(precision_list))
    macro_recall = float(np.mean(recall_list))
    macro_f1 = float(np.mean(f1_list))

    print("\n=== Evaluation Result ===")
    print(f"Accuracy      : {accuracy*100:.2f}%")
    print(f"Macro Precision: {macro_precision*100:.2f}%")
    print(f"Macro Recall   : {macro_recall*100:.2f}%")
    print(f"Macro F1       : {macro_f1*100:.2f}%")
    print("\nConfusion Matrix (row=true, col=pred):")
    print(conf_mat)


if __name__ == "__main__":
    # ここを評価したいモデルに合わせて書き換える
    MODEL_PATH = "models/pilotnet_steering_20251211_183119.pth"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    evaluate_one_model(MODEL_PATH)
