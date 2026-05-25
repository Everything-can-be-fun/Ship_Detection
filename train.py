# train.py
# 使用 YOLO26n 训练 SeaShips 六分类 detect 目标检测模型。
# 建议放在 YOLO26 项目根目录执行：python train.py

import re
from collections import Counter
from pathlib import Path

import torch
import yaml

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# 六分类 SeaShips 数据集配置。
# 注意：这个 yaml 应该对应 my_dataset/sea_ships_6cls，而不是之前合并成 0 ship 的 sea_ships。
DATA_YAML = ROOT / "my_dataset" / "sea_ships.yaml"
EXPECTED_DATASET_DIR = ROOT / "my_dataset" / "sea_ships_6cls"

# 明确使用 YOLO26n，而不是 YOLOv8n。
LOCAL_YOLO26_WEIGHTS = ROOT / "yolo26n.pt"
MODEL_WEIGHTS = LOCAL_YOLO26_WEIGHTS if LOCAL_YOLO26_WEIGHTS.exists() else "yolo26n.pt"

# 六分类结果单独保存，避免误用之前单类别 ship 的 best.pt。
PROJECT_DIR = ROOT / "runs" / "yolo26_seaships_6cls"
RUN_PREFIX = "train"

# 训练参数集中放这里，方便以后改。
EPOCHS = 100
IMGSZ = 640
BATCH = 16
WORKERS = 8


EXPECTED_NAMES = {
    0: "bulk cargo carrier",
    1: "container ship",
    2: "fishing boat",
    3: "general cargo ship",
    4: "ore carrier",
    5: "passenger ship",
}


def select_device():
    """有 CUDA 就用第 0 张 GPU，否则用 CPU。."""
    return 0 if torch.cuda.is_available() else "cpu"


def next_run_name(project_dir: Path, prefix: str) -> str:
    """生成 train1、train2、train3 这种不覆盖旧结果的目录名。."""
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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"数据集 YAML 内容异常：{path}")
    return data


def normalize_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("data.yaml 的 names 必须是 dict 或 list")


def resolve_dataset_dir(data: dict) -> Path:
    """优先使用期望的 sea_ships_6cls 目录；不存在时再按 YAML path 解析。."""
    if EXPECTED_DATASET_DIR.exists():
        return EXPECTED_DATASET_DIR

    raw_path = data.get("path")
    if raw_path is None:
        raise ValueError("data.yaml 缺少 path 字段")

    dataset_dir = Path(raw_path)
    if not dataset_dir.is_absolute():
        # 先按项目根目录解析，再按 YAML 所在目录解析。
        candidate1 = ROOT / dataset_dir
        candidate2 = DATA_YAML.parent / dataset_dir
        if candidate1.exists():
            dataset_dir = candidate1
        else:
            dataset_dir = candidate2

    return dataset_dir.resolve()


def make_runtime_yaml(dataset_dir: Path, names: dict[int, str]) -> Path:
    """生成运行时 YAML，避免服务器/本地绝对路径不一致。."""
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    runtime_yaml = PROJECT_DIR / "_runtime_sea_ships_6cls.yaml"

    lines = [
        f"path: {dataset_dir}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for k in sorted(names):
        lines.append(f"  {k}: {names[k]}")

    runtime_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return runtime_yaml


def validate_dataset(dataset_dir: Path, names: dict[int, str]) -> None:
    """训练前检查：目录、图片/标签数量、类别编号是否匹配。."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"找不到数据集目录：{dataset_dir}")

    expected_names = EXPECTED_NAMES
    if names != expected_names:
        print("\n[警告] 当前 YAML 类别表不是推荐的 SeaShips 六分类类别表。")
        print("当前 names:", names)
        print("推荐 names:", expected_names)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_counter = Counter()
    bad_lines = []

    print("\n========== 数据集检查 ==========")
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if not img_dir.exists():
            raise FileNotFoundError(f"找不到图片目录：{img_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"找不到标签目录：{label_dir}")

        images = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]
        labels = list(label_dir.glob("*.txt"))
        print(f"{split}: images={len(images)}, labels={len(labels)}")

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        if missing_labels:
            print(f"[警告] {split} 有图片但没有标签：{len(missing_labels)}，示例：{missing_labels[:5]}")
        if missing_images:
            print(f"[警告] {split} 有标签但没有图片：{len(missing_images)}，示例：{missing_images[:5]}")

        for txt in labels:
            for line_no, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad_lines.append((txt, line_no, "列数不是5", line))
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    xywh = [float(v) for v in parts[1:]]
                except ValueError:
                    bad_lines.append((txt, line_no, "无法解析数字", line))
                    continue
                class_counter[cls_id] += 1
                if cls_id not in names:
                    bad_lines.append((txt, line_no, f"类别 {cls_id} 不在 names 中", line))
                if not all(0.0 <= v <= 1.0 for v in xywh):
                    bad_lines.append((txt, line_no, "xywh 坐标不在 0~1", line))
                if xywh[2] <= 0 or xywh[3] <= 0:
                    bad_lines.append((txt, line_no, "宽高小于等于0", line))

    print("类别框数量：")
    for cls_id in sorted(class_counter):
        print(f"  {cls_id}: {names.get(cls_id, 'UNKNOWN')} -> {class_counter[cls_id]}")
    print("================================\n")

    if bad_lines:
        print("标签异常示例：")
        for item in bad_lines[:20]:
            print(item)
        raise RuntimeError(f"标签检查失败，异常行数：{len(bad_lines)}")

    if len(names) > 1 and set(class_counter) == {0}:
        raise RuntimeError(
            "你现在的 YAML 是多类别，但所有标签仍然只有类别 0。\n"
            "这说明你还在使用之前合并成单类 ship 的 YOLO 标签，必须从 VOC XML 重新转换六分类标签。"
        )

    if len(names) == 1:
        raise RuntimeError("当前 data.yaml 只有 1 个类别。SeaShips 六分类训练不应该使用 names: {0: ship}。")


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"找不到数据集配置文件：{DATA_YAML}")

    yaml_data = load_yaml(DATA_YAML)
    names = normalize_names(yaml_data.get("names"))
    dataset_dir = resolve_dataset_dir(yaml_data)
    validate_dataset(dataset_dir, names)
    runtime_yaml = make_runtime_yaml(dataset_dir, names)

    device = select_device()
    run_name = next_run_name(PROJECT_DIR, RUN_PREFIX)
    save_dir = PROJECT_DIR / run_name

    print("========== YOLO26 SeaShips 六分类训练配置 ==========")
    print(f"项目根目录：{ROOT}")
    print(f"使用模型：{MODEL_WEIGHTS}")
    print(f"数据集目录：{dataset_dir}")
    print(f"运行时 YAML：{runtime_yaml}")
    print(f"训练设备：{device}")
    print(f"结果目录：{save_dir}")
    print("类别表：")
    for k in sorted(names):
        print(f"  {k}: {names[k]}")
    print("==================================================")

    model = YOLO(str(MODEL_WEIGHTS))

    results = model.train(
        task="detect",
        data=str(runtime_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=WORKERS,
        patience=100,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=False,
        save=True,
        save_period=10,  # 防止最后一轮崩掉导致没有中间权重
        plots=True,
        end2end=True,
        single_cls=False,  # 明确禁止把多类别合并成单类别
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        amp=torch.cuda.is_available(),
        seed=0,
        deterministic=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        val=True,
        split="val",
        iou=0.7,
        max_det=300,
        verbose=True,
    )

    actual_save_dir = Path(getattr(results, "save_dir", save_dir))
    best_weight = actual_save_dir / "weights" / "best.pt"
    last_weight = actual_save_dir / "weights" / "last.pt"

    print("\n训练完成。")
    print(f"本次训练目录：{actual_save_dir}")
    print(f"best.pt：{best_weight}")
    print(f"last.pt：{last_weight}")
    print("后续预测建议直接运行：python predict.py")

    return results


if __name__ == "__main__":
    main()
