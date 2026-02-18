import os
import sys
import random
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "generated_dataset")
OUT_DIR = os.path.join(BASE_DIR, "augmented_dataset")


def flip_h(img):
    return ImageOps.mirror(img)

def flip_v(img):
    return ImageOps.flip(img)

def rotate(img):
    angle = random.choice([-12, -8, -5, 5, 8, 12])
    rot = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0,0,0))
    w, h = img.size
    rw, rh = rot.size
    return rot.crop(((rw-w)//2, (rh-h)//2, (rw+w)//2, (rh+h)//2))

def crop(img):
    w, h = img.size
    s = random.uniform(0.80, 0.92)
    cw, ch = int(w*s), int(h*s)
    x = random.randint(0, w-cw)
    y = random.randint(0, h-ch)
    return img.crop((x, y, x+cw, y+ch)).resize((w, h), Image.LANCZOS)

def brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.65, 1.35))

def contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))

def saturation(img):
    return ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))

def sharpness(img):
    return ImageEnhance.Sharpness(img).enhance(random.uniform(1.3, 2.0))

def noise(img):
    a = np.array(img).astype(np.float32)
    a += np.random.normal(0, random.uniform(5, 18), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def blur(img):
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))

def warm(img):
    a = np.array(img).astype(np.float32)
    a[:,:,0] = np.clip(a[:,:,0] + random.uniform(8, 20), 0, 255)
    a[:,:,2] = np.clip(a[:,:,2] - random.uniform(5, 15), 0, 255)
    return Image.fromarray(a.astype(np.uint8))

def cold(img):
    a = np.array(img).astype(np.float32)
    a[:,:,0] = np.clip(a[:,:,0] - random.uniform(5, 15), 0, 255)
    a[:,:,2] = np.clip(a[:,:,2] + random.uniform(8, 20), 0, 255)
    return Image.fromarray(a.astype(np.uint8))

def jpeg_compress(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=random.randint(25, 55))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


AUGMENTATIONS = [
    ("fliph",           [flip_h]),
    ("bright_cont",     [brightness, contrast]),
    ("fliph_warm",      [flip_h, warm]),
    ("crop_noise",      [crop, noise]),
    ("rot_bright",      [rotate, brightness]),
    ("sat_blur_jpeg",   [saturation, blur, jpeg_compress]),
    ("cold_sharp",      [cold, sharpness]),
]


def apply_chain(img, funcs):
    for f in funcs:
        img = f(img)
    return img


def transform_label(line, aug_name):
    parts = line.strip().split()
    if len(parts) != 5:
        return line
    cls, xc, yc, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    if "fliph" in aug_name:
        xc = 1.0 - xc
    if "flipv" in aug_name:
        yc = 1.0 - yc
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def main():
    print("=" * 55)
    print("  АУГМЕНТАЦИЯ ДАТАСЕТА")
    print("=" * 55)

    if not os.path.isdir(SRC_DIR):
        sys.exit(f"Нет папки {SRC_DIR}")

    src_labels = os.path.join(SRC_DIR, "labels")
    class_dirs = sorted([
        d for d in os.listdir(SRC_DIR)
        if os.path.isdir(os.path.join(SRC_DIR, d)) and d.startswith("damaged_")
    ])

    if not class_dirs:
        sys.exit("Нет папок damaged_*")

    # Выходные папки
    out_images = os.path.join(OUT_DIR, "images")
    out_labels = os.path.join(OUT_DIR, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # classes.txt
    src_cls = os.path.join(SRC_DIR, "classes.txt")
    if os.path.exists(src_cls):
        with open(src_cls) as f:
            txt = f.read()
        with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
            f.write(txt)

    # Собираем все картинки
    all_images = []
    for cls_dir in class_dirs:
        full = os.path.join(SRC_DIR, cls_dir)
        for fname in sorted(os.listdir(full)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append((cls_dir, fname))

    total_src = len(all_images)
    total_out = total_src * (1 + len(AUGMENTATIONS))
    print(f"\n  Исходных: {total_src}")
    for d in class_dirs:
        c = sum(1 for x in all_images if x[0] == d)
        print(f"    {d}: {c}")
    print(f"  Аугментаций: {len(AUGMENTATIONS)} на каждую")
    print(f"  Итого: {total_out}")
    print(f"\n{'─'*55}\n")

    done = 0

    for cls_dir, fname in all_images:
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(SRC_DIR, cls_dir, fname)
        lbl_path = os.path.join(src_labels, f"{stem}.txt")

        img = Image.open(img_path).convert("RGB")

        label_lines = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                label_lines = [l.strip() for l in f if l.strip()]

        # Оригинал
        img.save(os.path.join(out_images, f"{stem}.jpg"), "JPEG", quality=95)
        with open(os.path.join(out_labels, f"{stem}.txt"), "w") as f:
            f.write("\n".join(label_lines) + "\n")
        done += 1

        # Аугментации
        for aug_name, aug_funcs in AUGMENTATIONS:
            out_stem = f"{stem}_{aug_name}"
            try:
                aug_img = apply_chain(img, aug_funcs)
                aug_img.save(os.path.join(out_images, f"{out_stem}.jpg"),
                             "JPEG", quality=90)
                aug_labels = [transform_label(l, aug_name) for l in label_lines]
                with open(os.path.join(out_labels, f"{out_stem}.txt"), "w") as f:
                    f.write("\n".join(aug_labels) + "\n")
                done += 1
            except Exception as e:
                print(f"  ✗ {out_stem}: {e}")

        if done % 100 == 0:
            print(f"  {done}/{total_out} ...")

    # Итог
    n_imgs = len([f for f in os.listdir(out_images) if f.endswith(".jpg")])
    n_lbls = len([f for f in os.listdir(out_labels) if f.endswith(".txt")])

    print(f"\n{'='*55}")
    print(f"  ГОТОВО!")
    print(f"  {total_src} → {n_imgs} картинок (×{n_imgs/total_src:.1f})")
    print(f"  Лейблов: {n_lbls}")
    print(f"")
    print(f"  {OUT_DIR}/")
    print(f"    images/  {n_imgs}")
    print(f"    labels/  {n_lbls}")
    print(f"    classes.txt")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()