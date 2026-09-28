# voc_to_yolo_seaships_6cls.py
# 从 SeaShips VOC XML 重新生成六分类 YOLO 标签。
# 用法示例：
#   python voc_to_yolo_seaships_6cls.py --voc-root my_dataset/SeaShips_VOC --out my_dataset/sea_ships_6cls
# 如果你的 VOC 原始目录就是 my_dataset/Annotations、my_dataset/ImageSets、my_dataset/JPEGImages，则：
#   python voc_to_yolo_seaships_6cls.py --voc-root my_dataset --out my_dataset/sea_ships_6cls

import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


CLASSES = [
    "bulk cargo carrier",
    "container ship",
    "fishing boat",
    "general cargo ship",
    "ore carrier",
    "passenger ship",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SeaShips VOC XML to YOLO six-class format.")
    parser.add_argument(
        "--voc-root", type=Path, required=True, help="VOC root containing Annotations, ImageSets, JPEGImages"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output YOLO dataset directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    return parser.parse_args()


def read_ids(split_dir: Path, split: str):
    split_file = split_dir / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"找不到划分文件：{split_file}")
    ids = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(line.split()[0])
    return ids


def find_image(img_dir: Path, image_id: str):
    for suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = img_dir / f"{image_id}{suffix}"
        if p.exists():
            return p
    return None


def image_is_decodable(path: Path) -> bool:
    if path.stat().st_size == 0:
        return False
    if cv2 is None or np is None:
        return True
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return False
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return im is not None and im.size > 0


def voc_box_to_yolo(img_w, img_h, xmin, ymin, xmax, ymax):
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def convert_xml(xml_file: Path, label_file: Path):
    root = ET.parse(xml_file).getroot()
    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    lines = []
    for obj in root.findall("object"):
        cls_name = obj.findtext("name").strip()
        if cls_name not in CLASS_TO_ID:
            raise ValueError(f"未知类别 {cls_name!r} in {xml_file}")
        cls_id = CLASS_TO_ID[cls_name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        xmin = max(0, min(xmin, img_w))
        xmax = max(0, min(xmax, img_w))
        ymin = max(0, min(ymin, img_h))
        ymax = max(0, min(ymax, img_h))
        if xmax <= xmin or ymax <= ymin:
            continue

        x, y, w, h = voc_box_to_yolo(img_w, img_h, xmin, ymin, xmax, ymax)
        if not all(0 <= v <= 1 for v in [x, y, w, h]):
            continue
        if w <= 0 or h <= 0:
            continue

        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lines


def main():
    args = parse_args()
    voc_root = args.voc_root.resolve()
    out_dir = args.out.resolve()

    ann_dir = voc_root / "Annotations"
    img_dir = voc_root / "JPEGImages"
    split_dir = voc_root / "ImageSets" / "Main"

    for p in [ann_dir, img_dir, split_dir]:
        if not p.exists():
            raise FileNotFoundError(f"找不到 VOC 目录：{p}")

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    elif out_dir.exists():
        raise FileExistsError(f"输出目录已存在：{out_dir}。如需覆盖，请加 --overwrite")

    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_counter = Counter()
    skipped = []

    for split in ["train", "val", "test"]:
        ids = read_ids(split_dir, split)
        print(f"开始转换 {split}: {len(ids)} 张")
        for image_id in ids:
            xml_file = ann_dir / f"{image_id}.xml"
            img_file = find_image(img_dir, image_id)
            if not xml_file.exists():
                skipped.append((split, image_id, "缺少 XML"))
                continue
            if img_file is None:
                skipped.append((split, image_id, "缺少图片"))
                continue
            if not image_is_decodable(img_file):
                skipped.append((split, image_id, f"图片无法解码：{img_file.name}"))
                continue

            dst_img = out_dir / "images" / split / img_file.name
            dst_label = out_dir / "labels" / split / f"{image_id}.txt"
            shutil.copy2(img_file, dst_img)
            lines = convert_xml(xml_file, dst_label)
            for line in lines:
                class_counter[int(line.split()[0])] += 1

    yaml_path = out_dir.parent / "sea_ships.yaml"
    yaml_lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for i, name in enumerate(CLASSES):
        yaml_lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print("\n转换完成。")
    print(f"输出目录：{out_dir}")
    print(f"YAML：{yaml_path}")
    print("类别框数量：")
    for i, name in enumerate(CLASSES):
        print(f"  {i}: {name} -> {class_counter[i]}")

    if skipped:
        print("\n跳过文件：")
        for item in skipped[:50]:
            print(" ", item)
        print(f"共跳过：{len(skipped)}")


if __name__ == "__main__":
    main()
