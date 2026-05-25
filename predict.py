# predict.py
# 使用训练好的 YOLO26 SeaShips 六分类模型做目标检测预测。
# 建议放在 YOLO26 项目根目录执行：python predict.py

from __future__ import annotations

import re
from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# 六分类 SeaShips 验证集；也可以改成任意图片文件夹、单张图片或视频路径。
SOURCE = ROOT / "my_dataset" / "sea_ships_6cls" / "images" / "val"

# 必须和 train.py 的 PROJECT_DIR 一致，避免误用之前单类别 ship 的 best.pt。
PROJECT_DIR = ROOT / "runs" / "yolo26_seaships_6cls"
RUN_PREFIX = "predict"

LOCAL_YOLO26_WEIGHTS = ROOT / "yolo26n.pt"
FALLBACK_WEIGHTS = LOCAL_YOLO26_WEIGHTS if LOCAL_YOLO26_WEIGHTS.exists() else "yolo26n.pt"

# 六分类模型预测时不要过滤类别。
PREDICT_CLASSES = None


def select_device():
    """有 CUDA 就用第 0 张 GPU，否则用 CPU。."""
    return 0 if torch.cuda.is_available() else "cpu"


def next_run_name(project_dir: Path, prefix: str) -> str:
    """生成 predict1、predict2、predict3 这种不覆盖旧结果的目录名。."""
    project_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    nums = []

    for p in project_dir.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            nums.append(int(m.group(1)))

    return f"{prefix}{max(nums, default=0) + 1}"


def find_latest_trained_weight(project_dir: Path) -> Path | None:
    """从 train、train1、train2... 中寻找最近一次训练得到的 best.pt。."""
    if not project_dir.exists():
        return None

    candidates = []
    for train_dir in project_dir.iterdir():
        if not train_dir.is_dir():
            continue
        if not re.match(r"^train\d*$", train_dir.name):
            continue
        best_pt = train_dir / "weights" / "best.pt"
        if best_pt.exists():
            candidates.append(best_pt)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"找不到预测来源：{SOURCE}")

    trained_weights = find_latest_trained_weight(PROJECT_DIR)
    weights = trained_weights if trained_weights is not None else FALLBACK_WEIGHTS
    device = select_device()
    run_name = next_run_name(PROJECT_DIR, RUN_PREFIX)
    save_dir = PROJECT_DIR / run_name

    print("========== YOLO26 SeaShips 六分类预测配置 ==========")
    print(f"项目根目录：{ROOT}")
    print(f"使用权重：{weights}")
    print(f"预测来源：{SOURCE}")
    print(f"预测设备：{device}")
    print(f"类别过滤：{PREDICT_CLASSES}")
    print(f"结果目录：{save_dir}")
    print("==================================================")

    model = YOLO(str(weights))
    print("模型类别表：")
    for k, v in model.names.items():
        print(f"  {k}: {v}")

    if len(model.names) == 1:
        print("\n[警告] 当前权重只有 1 个类别。你可能加载到了之前的单类别 ship 模型。")
        print("请确认使用的是 runs/yolo26_seaships_6cls/train*/weights/best.pt。\n")

    predict_kwargs = dict(
        task="detect",
        source=str(SOURCE),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        max_det=300,
        device=device,
        end2end=True,
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=False,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=False,
        show=False,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        line_width=None,
        stream=False,
        visualize=False,
        augment=False,
        agnostic_nms=False,
    )

    if PREDICT_CLASSES is not None:
        predict_kwargs["classes"] = PREDICT_CLASSES

    results = model.predict(**predict_kwargs)
    actual_save_dir = Path(results[0].save_dir) if results else save_dir

    print("\n预测完成。逐图检测结果：")
    for result in results:
        image_name = Path(result.path).name
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        print(f"\n{image_name}: 检测到 {count} 个目标")

        if count == 0:
            continue

        for cls_id, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
            cls_id = int(cls_id.item())
            conf = float(conf.item())
            class_name = result.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [round(float(v), 2) for v in xyxy.tolist()]
            print(f"  - {class_name}: conf={conf:.3f}, box=({x1}, {y1}, {x2}, {y2})")

    print(f"\n可视化结果保存到：{actual_save_dir}")
    print(f"标签 txt 保存到：{actual_save_dir / 'labels'}")

    return results


if __name__ == "__main__":
    main()
