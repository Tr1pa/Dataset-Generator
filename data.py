import os
import sys
import random
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "augmented_dataset")
OUT_DIR = os.path.join(BASE_DIR, "dataset_yolo")

SPLIT = 0.8  # 80% train, 20% val


def main():
    print("=" * 55)
    print("  ПОДГОТОВКА ДАТАСЕТА ДЛЯ YOLO")
    print("=" * 55)

    src_images = os.path.join(SRC_DIR, "images")
    src_labels = os.path.join(SRC_DIR, "labels")

    if not os.path.isdir(src_images):
        sys.exit(f"Нет папки {src_images}\nСначала запустите augment.py")

    # Собираем пары image + label
    all_imgs = sorted([
        f for f in os.listdir(src_images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    pairs = []
    missing_labels = 0
    for img_name in all_imgs:
        stem = os.path.splitext(img_name)[0]
        lbl_name = f"{stem}.txt"
        lbl_path = os.path.join(src_labels, lbl_name)
        if os.path.exists(lbl_path):
            pairs.append((img_name, lbl_name))
        else:
            missing_labels += 1

    print(f"\n  Найдено пар (image + label): {len(pairs)}")
    if missing_labels:
        print(f"  Без лейблов (пропущены): {missing_labels}")

    # Перемешиваем и делим
    random.seed(42)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * SPLIT)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"  Train: {len(train_pairs)}")
    print(f"  Val:   {len(val_pairs)}")

    # Создаём структуру
    dirs = {
        "train_img": os.path.join(OUT_DIR, "train", "images"),
        "train_lbl": os.path.join(OUT_DIR, "train", "labels"),
        "val_img":   os.path.join(OUT_DIR, "val", "images"),
        "val_lbl":   os.path.join(OUT_DIR, "val", "labels"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Копируем файлы
    print(f"\n  Копирование...")

    for img_name, lbl_name in train_pairs:
        shutil.copy2(os.path.join(src_images, img_name),
                     os.path.join(dirs["train_img"], img_name))
        shutil.copy2(os.path.join(src_labels, lbl_name),
                     os.path.join(dirs["train_lbl"], lbl_name))

    for img_name, lbl_name in val_pairs:
        shutil.copy2(os.path.join(src_images, img_name),
                     os.path.join(dirs["val_img"], img_name))
        shutil.copy2(os.path.join(src_labels, lbl_name),
                     os.path.join(dirs["val_lbl"], lbl_name))

    # data.yaml
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    yaml_content = f"""path: {OUT_DIR}
train: train/images
val: val/images

names:
  0: damaged_seat
  1: damaged_floor
  2: damaged_metal
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # Статистика по классам
    print(f"\n  Статистика по классам:")
    class_counts = {"train": {0: 0, 1: 0, 2: 0}, "val": {0: 0, 1: 0, 2: 0}}
    class_names = {0: "damaged_seat", 1: "damaged_floor", 2: "damaged_metal"}

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        lbl_dir = dirs[f"{split_name}_lbl"]
        for _, lbl_name in split_pairs:
            with open(os.path.join(lbl_dir, lbl_name)) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[split_name][cls_id] += 1

    print(f"  {'Класс':20s} {'Train':>8s} {'Val':>8s}")
    print(f"  {'─'*18}  {'─'*8} {'─'*8}")
    for cls_id in sorted(class_names):
        name = class_names[cls_id]
        tr = class_counts["train"][cls_id]
        va = class_counts["val"][cls_id]
        print(f"  {name:20s} {tr:8d} {va:8d}")

    # Итог
    n_train = len(os.listdir(dirs["train_img"]))
    n_val = len(os.listdir(dirs["val_img"]))

    print(f"\n{'='*55}")
    print(f"  ГОТОВО!")
    print(f"")
    print(f"  {OUT_DIR}/")
    print(f"    train/images/  {n_train}")
    print(f"    train/labels/  {n_train}")
    print(f"    val/images/    {n_val}")
    print(f"    val/labels/    {n_val}")
    print(f"    data.yaml")
    print(f"")
    print(f"  СЛЕДУЮЩИЙ ШАГ — обучение:")
    print(f"")
    print(f"  pip install ultralytics")
    print(f"")
    print(f"  yolo detect train \\")
    print(f"    model=yolov8n.pt \\")
    print(f"    data={yaml_path} \\")
    print(f"    epochs=50 \\")
    print(f"    imgsz=640 \\")
    print(f"    batch=16 \\")
    print(f"    name=metro_damage")
    print(f"")
    print(f"  Или из Python:")
    print(f"")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolov8n.pt')")
    print(f"  model.train(")
    print(f"      data=r'{yaml_path}',")
    print(f"      epochs=50,")
    print(f"      imgsz=640,")
    print(f"      batch=16,")
    print(f"      name='metro_damage',")
    print(f"  )")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()