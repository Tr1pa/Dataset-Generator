import os
import io
import json
import base64
import time
import random
import requests
from PIL import Image
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "generated_dataset")
PROMPTS_FILE = os.path.join(BASE_DIR, "prompts.json")

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-5-image-mini"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

TOTAL = 220


# ──────────────────────────────────────────────────────────
#  Загрузка и разбор prompts.json
# ──────────────────────────────────────────────────────────
def load_prompts():
    """
    Ожидаемый формат:
    {
      "scenes": {
        "old": "Interior of an old Soviet-style ...",
        "mid": "Interior of a Russian subway car ...",
        "new": "Interior of a modern Moscow metro ..."
      },
      "defects": {
        "damaged_seat":  ["FOCUS ON SEAT DAMAGE: ...", ...],
        "damaged_floor": ["FOCUS ON FLOOR DAMAGE: ...", ...],
        "damaged_metal": ["FOCUS ON METAL DAMAGE: ...", ...]
      }
    }

    Возвращает:
        scenes  — list строк-описаний сцен
        classes — {0: {"name": "damaged_seat",  "prompts": [...]},
                   1: {"name": "damaged_floor", "prompts": [...]},
                   ...}
    """
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # ── Сцены ─────────────────────────────────────────────
    scenes_dict = raw.get("scenes", {})
    if not scenes_dict:
        raise ValueError("prompts.json: раздел 'scenes' пуст или отсутствует")
    scenes = list(scenes_dict.values())

    # ── Дефекты → CLASSES ─────────────────────────────────
    defects_dict = raw.get("defects", {})
    if not defects_dict:
        raise ValueError("prompts.json: раздел 'defects' пуст или отсутствует")

    classes = {}
    for idx, (name, prompts) in enumerate(defects_dict.items()):
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"prompts.json: '{name}' должен быть непустым списком")
        classes[idx] = {"name": name, "prompts": prompts}

    return scenes, classes


def get_balance():
    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
        )
        if r.status_code == 200:
            d = r.json().get("data", {})
            used = d.get("usage", 0)
            limit = d.get("limit", None)
            return (limit - used if limit else None), used
    except Exception:
        pass
    return None, None


def build_prompt(scene: str, defect_prompt: str) -> str:
    """
    Склеивает описание сцены + описание дефекта в один промпт.

    Пример результата:
        "Interior of an old Soviet-style Moscow metro train carriage ...
         FOCUS ON SEAT DAMAGE: Brown faux-leather severely deteriorated ..."
    """
    return f"{scene}\n\n{defect_prompt}"


def generate_image(prompt: str, retries: int = 3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/subway-damage",
    }

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                ENDPOINT,
                headers=headers,
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=180,
            )

            if r.status_code == 200:
                body = r.json()
                cost = body.get("usage", {}).get("cost", 0)

                # Извлечение base64-картинки из ответа
                images = (
                    body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("images", [])
                )
                for item in images:
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        b64 = url.split(",", 1)[1]
                        img = Image.open(
                            io.BytesIO(base64.b64decode(b64))
                        ).convert("RGB")
                        return img, cost
                # Модель ответила, но без картинки
                print("[нет картинки]", end="", flush=True)
                return None, cost

            elif r.status_code == 429:
                wait = 20 * attempt
                print(f"[429 ждём {wait}с]", end="", flush=True)
                time.sleep(wait)
                continue

            elif r.status_code == 402:
                return "NO_MONEY", 0

            else:
                print(f"[HTTP {r.status_code}]", end="", flush=True)
                return None, 0

        except requests.exceptions.Timeout:
            print(f"[таймаут #{attempt}]", end="", flush=True)
            if attempt < retries:
                time.sleep(5)
                continue
        except Exception as e:
            print(f"[ошибка: {e}]", end="", flush=True)

    return None, 0


def main():
    # ── Загрузка ──────────────────────────────────────────
    scenes, CLASSES = load_prompts()

    print("=" * 60)
    print("  ДАТАСЕТ ПОВРЕЖДЕНИЙ — МОСКОВСКОЕ МЕТРО")
    print(f"  Модель:  {MODEL}")
    print(f"  Цель:    {TOTAL} изображений")
    print(f"  Классов: {len(CLASSES)}")
    print(f"  Сцен:    {len(scenes)}")
    print("=" * 60)

    # ── Баланс ────────────────────────────────────────────
    remaining, used = get_balance()
    actual = TOTAL
    if remaining is not None:
        max_imgs = int(remaining / 0.042)
        actual = min(TOTAL, max_imgs)
        print(f"\n  Остаток:  ${remaining:.2f} (~{max_imgs} картинок)")
        if actual < TOTAL:
            print(f"  ⚠ Хватит только на {actual} из {TOTAL}")
    else:
        print("\n  Баланс: не удалось определить, продолжаем...")

    # ── Папки ─────────────────────────────────────────────
    for info in CLASSES.values():
        os.makedirs(os.path.join(OUT_DIR, info["name"]), exist_ok=True)
    labels_dir = os.path.join(OUT_DIR, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    with open(os.path.join(OUT_DIR, "classes.txt"), "w", encoding="utf-8") as f:
        for cls_id in sorted(CLASSES.keys()):
            f.write(CLASSES[cls_id]["name"] + "\n")

    # ── Расписание ────────────────────────────────────────
    # Каждому изображению назначаем:
    #   • случайную сцену   (old / mid / new)
    #   • случайный промпт дефекта из класса
    per_class = actual // len(CLASSES)
    remainder = actual % len(CLASSES)
    schedule = []

    for cls_id, info in CLASSES.items():
        count = per_class + (1 if cls_id < remainder else 0)
        for _ in range(count):
            scene = random.choice(scenes)
            defect = random.choice(info["prompts"])
            full_prompt = build_prompt(scene, defect)
            schedule.append((cls_id, info["name"], full_prompt))

    random.shuffle(schedule)

    # ── Статистика по сценам ──────────────────────────────
    print(f"\n  Распределение по классам:")
    for cls_id in sorted(CLASSES.keys()):
        info = CLASSES[cls_id]
        c = sum(1 for s in schedule if s[0] == cls_id)
        print(f"    [{cls_id}] {info['name']:20s} — {c} шт")

    est_cost = len(schedule) * 0.042
    est_time = len(schedule) * 55 / 60
    print(f"\n  Всего:    {len(schedule)} изображений")
    print(f"  Оценка:   ~${est_cost:.2f}, ~{est_time:.0f} мин")
    print(f"\n{'─' * 60}\n")

    # ── Генерация ─────────────────────────────────────────
    t0 = time.time()
    ok = errors = 0
    total_spent = 0.0
    counters = {c: 0 for c in CLASSES}

    for i, (cls_id, cls_name, prompt) in enumerate(schedule, 1):
        counters[cls_id] += 1
        idx = counters[cls_id]
        fname = f"{cls_name}_{idx:04d}"

        print(f"  [{i:3d}/{len(schedule)}] {fname:30s} ", end="", flush=True)

        img, cost = generate_image(prompt)

        # Деньги кончились
        if img == "NO_MONEY":
            print("💰 БАЛАНС КОНЧИЛСЯ!")
            break

        total_spent += cost if isinstance(cost, (int, float)) else 0

        if img is not None and img != "NO_MONEY":
            # Сохраняем изображение
            img.save(
                os.path.join(OUT_DIR, cls_name, f"{fname}.jpg"),
                "JPEG",
                quality=95,
            )
            # YOLO-метка: класс + bbox на всё изображение
            with open(os.path.join(labels_dir, f"{fname}.txt"), "w") as lf:
                lf.write(f"{cls_id} 0.500000 0.500000 1.000000 1.000000\n")

            ok += 1
            elapsed = time.time() - t0
            eta = (elapsed / i) * (len(schedule) - i)
            print(
                f"✅ ${cost:.3f}  "
                f"[{elapsed / 60:.0f}м / ~{eta / 60:.0f}м]  "
                f"итого ${total_spent:.2f}"
            )
        else:
            errors += 1
            print("❌")
            if errors > 15:
                print("\n  ⛔ Слишком много ошибок подряд, остановка.")
                break

        # Пауза между запросами
        time.sleep(3)

    # ── Итоги ─────────────────────────────────────────────
    t = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  ГОТОВО")
    print(f"  Успешно:  {ok} картинок")
    print(f"  Ошибок:   {errors}")
    print(f"  Потрачено: ${total_spent:.2f}")
    print(f"  Время:    {int(t // 60)}м {int(t % 60)}с")
    print(f"{'─' * 60}")
    for cls_id in sorted(CLASSES.keys()):
        info = CLASSES[cls_id]
        d = os.path.join(OUT_DIR, info["name"])
        c = len([f for f in os.listdir(d) if f.endswith(".jpg")])
        print(f"    {info['name']:20s}  {c:4d} файлов")
    print(f"{'─' * 60}")
    print(f"  📁 {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()