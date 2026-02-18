import os
import io
import base64
import time
import random
import requests
from PIL import Image
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "generated_dataset")

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-5-image-mini"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

TOTAL = 220

# ── Базовые сцены (3 типа вагонов) ────────────────────────────────────────

SCENE_OLD = (
    "Interior of an old Soviet-style Moscow metro train carriage, "
    "one-point perspective looking down the narrow aisle. "
    "Vanishing point at the far end door. Eye-level shot. "
    "Rows of brown faux-leather padded bench seats along cream-colored walls. "
    "Wood-grain texture plastic wall panels. Grey-brown spotted linoleum floor. "
    "Chrome steel handrails and vertical poles. "
    "Dim warm yellowish longitudinal ceiling lights. "
    "Slightly narrow tunnel-like feel. "
    "No people. No text overlay. No watermark. Photorealistic, 8k."
)

SCENE_MID = (
    "Interior of a Russian subway car from the 2000s era, "
    "wide-angle one-point perspective down the center aisle. "
    "Rows of blue vinyl bench seats along light grey walls. "
    "Grey spotted linoleum floor. Chrome grab poles and overhead rails. "
    "Bright neutral fluorescent ceiling lights. "
    "Typical Moscow metro atmosphere. "
    "No people. No text overlay. No watermark. Photorealistic, 8k."
)

SCENE_NEW = (
    "Interior of a modern Moscow metro train car, "
    "symmetrical one-point perspective. "
    "Blue anti-vandal fabric bench seats along white plastic walls. "
    "Dark grey rubber flooring. Stainless steel poles and handrails. "
    "Bright cold LED strip lighting on ceiling. "
    "Clean modern design but showing signs of heavy public use. "
    "No people. No text overlay. No watermark. Photorealistic, 8k."
)

SCENES = [SCENE_OLD, SCENE_MID, SCENE_NEW]

# ── Классы и промпты ──────────────────────────────────────────────────────

CLASSES = {
    0: {
        "name": "damaged_seat",
        "prompts": [
            # Старый вагон — дерматин порван
            f"{SCENE_OLD} "
            "FOCUS ON SEAT DAMAGE: The brown faux-leather seats are severely "
            "deteriorated. Leather surface cracked in a web-like pattern from age. "
            "Multiple deep knife cuts and slashes exposing bright yellow foam "
            "padding underneath. Foam crumbling and dirty. Leather peeling off "
            "at seat edges. Dark greasy stains on headrest areas from years of use. "
            "Depressive gloomy atmosphere.",

            # Старый вагон — засаленность
            f"{SCENE_OLD} "
            "FOCUS ON SEAT GRIME: The brown leatherette bench seats are extremely "
            "dirty and worn. Surface darkened and greasy from millions of passengers. "
            "Original brown color barely visible under layers of grime. "
            "Leather cracking and peeling at corners. Some seats have carved "
            "initials and scratched graffiti. Sticky unknown residue on surfaces. "
            "Yellow foam visible through worn-out spots.",

            # Средний вагон — синий винил
            f"{SCENE_MID} "
            "FOCUS ON SEAT DAMAGE: The blue vinyl bench seats show heavy wear. "
            "Vinyl torn in several places showing yellow padding inside. "
            "Dark stains from spilled drinks absorbed into seat crevices. "
            "Surface scratched and scuffed to pale blue. Black marker graffiti "
            "tags scrawled across seat backs. Chewing gum stuck to undersides. "
            "Cigarette burn marks as small melted circles.",

            # Новый вагон — ткань
            f"{SCENE_NEW} "
            "FOCUS ON SEAT WEAR: The blue anti-vandal fabric seats are stained "
            "and discolored. Dark greasy patches on sitting areas and headrests "
            "from body oils. Fabric pilling and worn thin in high-contact zones. "
            "Some spots with mysterious dark wet stains. "
            "Dried food crumbs in seat crevices. Fabric fraying at edges. "
            "Despite modern interior, seats look heavily used and neglected.",

            # Микс — порезы и вандализм
            f"{SCENE_MID} "
            "FOCUS ON SEAT VANDALISM: Blue seats are vandalized. "
            "Long knife slash across one seat cushion with foam bulging out. "
            "Multiple sticker residues — sticky grey rectangles where ads were "
            "peeled off. Pen and marker scribbles. Scratched obscenities. "
            "One seat has a large burn hole. Surrounding seats discolored "
            "and grimy. Harsh fluorescent light emphasizing every defect.",

            # Старый — комбинированный
            f"{SCENE_OLD} "
            "FOCUS ON SEAT DETERIORATION: Brown faux-leather seats in terrible "
            "condition. Massive rip exposing dirty yellow foam on center seat. "
            "Adjacent seats covered in scratches, pen markings, and dark grime. "
            "Leather texture completely worn off on sitting surfaces — smooth "
            "and shiny from friction. Armrest edges chipped and cracked. "
            "Dim yellowish light casting shadows on the damage.",
        ],
    },
    1: {
        "name": "damaged_floor",
        "prompts": [
            # Зимняя грязь — соль и реагенты
            f"{SCENE_OLD} "
            "FOCUS ON FLOOR DAMAGE: The grey-brown linoleum floor is covered "
            "in dried white salt stains and chemical reagent residue from winter "
            "boots. Grey dried sludge trails from entrance doors along the aisle. "
            "Dark wet muddy footprints overlapping. Whitish crystalline deposits "
            "along seat edges where snow melted and dried. "
            "Floor looks like a winter mess. Gloomy atmosphere.",

            # Износ линолеума
            f"{SCENE_MID} "
            "FOCUS ON FLOOR WEAR: The grey spotted linoleum is heavily worn. "
            "Center aisle strip worn completely smooth and pale from millions "
            "of footsteps. Original speckled pattern barely visible. "
            "Black rubber scuff marks from shoes crisscrossing everywhere. "
            "Linoleum edges curling up near doors exposing dark adhesive. "
            "Deep scratch lines from dragged luggage.",

            # Пятна и разливы
            f"{SCENE_NEW} "
            "FOCUS ON FLOOR STAINS: The dark grey rubber floor has multiple "
            "visible stains. Large dried brown coffee spill near center pole. "
            "Sticky dark soda puddle near seat edge. Whitish dried liquid "
            "splash marks. Ground-in black dirt giving uneven coloring. "
            "Scuff marks from heavy boots. Despite modern car, floor is filthy.",

            # Грязь и мусор
            f"{SCENE_OLD} "
            "FOCUS ON FLOOR FILTH: Floor of old Moscow metro car is disgusting. "
            "Mixture of dried winter mud, salt residue, and tracked-in dirt. "
            "Small pieces of litter — receipt paper, sunflower seed shells, "
            "a crushed paper cup near the door. Dark sticky patches. "
            "Linoleum cracked near threshold with metal edge strip coming loose. "
            "Years of accumulated grime in floor corners.",

            # Трещины и отслоение
            f"{SCENE_MID} "
            "FOCUS ON FLOOR DAMAGE: Linoleum floor with structural damage. "
            "Long crack running along high-traffic zone with edges lifting. "
            "Patch of linoleum completely missing near door revealing dark "
            "metal subfloor. Surrounding area covered in black grime and "
            "scuff marks. White salt stains from winter boots everywhere. "
            "Floor surface bubbling and separating from substrate.",

            # Слякоть
            f"{SCENE_OLD} "
            "FOCUS ON WINTER FLOOR: Moscow metro floor during winter season. "
            "Wet grey slush puddles near door entrances. Muddy brown water "
            "footprint trails down the aisle. White dried salt crystallization "
            "patterns where previous puddles evaporated. Dark wet patches. "
            "Gritty sand particles scattered on linoleum surface. "
            "Damp depressive atmosphere, dim warm lighting.",
        ],
    },
    2: {
        "name": "damaged_metal",
        "prompts": [
            # Ржавчина на поручнях
            f"{SCENE_OLD} "
            "FOCUS ON METAL DAMAGE: The chrome steel handrails and grab poles "
            "are heavily corroded. Chrome plating worn off at hand height "
            "exposing dull grey steel with orange rust spots. "
            "Rust streaks running down from overhead rail mounting brackets. "
            "Pitting and rough texture on pole surfaces. "
            "Base connections have green verdigris oxidation.",

            # Тусклые отпечатки
            f"{SCENE_MID} "
            "FOCUS ON METAL WEAR: Chrome grab poles lost their shine. "
            "Covered in dark fingerprint smudges and greasy handprints. "
            "Dull matte finish instead of mirror chrome. "
            "Small dents from impacts. Horizontal scratch lines from rings. "
            "Surface feels rough and tarnished. Some rust spots at joints "
            "where moisture collects. Overall neglected appearance.",

            # Краска на стенных панелях
            f"{SCENE_OLD} "
            "FOCUS ON WALL AND METAL DAMAGE: Wood-grain plastic wall panels "
            "heavily scratched and scuffed at back-height from passengers. "
            "Panel edges separating from wall showing dark gap. "
            "Painted metal surfaces near ceiling with peeling cream paint "
            "revealing rusty metal underneath. Rust drip stains on wall. "
            "Screw heads corroded with orange rust halos.",

            # Современный — царапины
            f"{SCENE_NEW} "
            "FOCUS ON METAL DAMAGE: Stainless steel poles and handrails in "
            "modern car showing abuse marks. Deep scratches carved into "
            "pole surfaces. Circular scratch patterns from spinning on poles. "
            "Dents from kicked surfaces. Small rust spots forming at weld "
            "points. Hand-grip zones darkened and roughened from heavy use. "
            "Contrast between clean car design and damaged metal.",

            # Двери и рамы
            f"{SCENE_MID} "
            "FOCUS ON DOOR METAL DAMAGE: Metal door frames and thresholds "
            "showing heavy wear. Chrome strip along door edge deeply "
            "scratched and dented from thousands of closings. "
            "Paint chipped off door frame corners exposing bare metal. "
            "Threshold plate worn shiny smooth with edges bent upward. "
            "Rust forming along floor-level metal joints. "
            "Rubber door seals cracked and discolored.",

            # Комбо ржавчина + грязь
            f"{SCENE_OLD} "
            "FOCUS ON METAL CORROSION: Every metal surface shows neglect. "
            "Overhead rails have thick orange-brown rust at every bracket. "
            "Vertical poles pitted and rough, chrome completely gone at "
            "waist height. Door handles tarnished dark. "
            "Metal ventilation grille clogged with dust and corroded. "
            "Dim yellowish light making rust look even more orange. "
            "Soviet-era decay atmosphere.",
        ],
    },
}


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


def generate_image(prompt, retries=2):
    h = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/subway-damage",
    }

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                ENDPOINT, headers=h,
                json={"model": MODEL,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=180,
            )

            if r.status_code == 200:
                body = r.json()
                cost = body.get("usage", {}).get("cost", 0)
                images = (body.get("choices", [{}])[0]
                          .get("message", {}).get("images", []))
                for item in images:
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        b64 = url.split(",", 1)[1]
                        img = Image.open(
                            io.BytesIO(base64.b64decode(b64))
                        ).convert("RGB")
                        return img, cost
                return None, cost

            elif r.status_code == 429:
                wait = 20 * attempt
                print(f"[лимит {wait}с]", end="", flush=True)
                time.sleep(wait)
                continue
            elif r.status_code == 402:
                return "NO_MONEY", 0
            else:
                return None, 0

        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(5)
                continue
        except Exception:
            pass
    return None, 0


def main():
    print("=" * 60)
    print("  ДАТАСЕТ ПОВРЕЖДЕНИЙ — МОСКОВСКОЕ МЕТРО")
    print(f"  Модель: {MODEL}")
    print(f"  Цель: {TOTAL} изображений")
    print("=" * 60)

    remaining, used = get_balance()
    actual = TOTAL
    if remaining is not None:
        max_imgs = int(remaining / 0.042)
        actual = min(TOTAL, max_imgs)
        print(f"\n  Остаток: ${remaining:.2f} (~{max_imgs} картинок)")

    # Папки
    for info in CLASSES.values():
        os.makedirs(os.path.join(OUT_DIR, info["name"]), exist_ok=True)
    labels_dir = os.path.join(OUT_DIR, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
        for cls_id in sorted(CLASSES.keys()):
            f.write(CLASSES[cls_id]["name"] + "\n")

    # Расписание
    per_class = actual // len(CLASSES)
    schedule = []
    for cls_id, info in CLASSES.items():
        count = per_class + (1 if cls_id < actual % len(CLASSES) else 0)
        for i in range(count):
            prompt = random.choice(info["prompts"])
            schedule.append((cls_id, info["name"], prompt))
    random.shuffle(schedule)

    print(f"\n  Распределение:")
    for cls_id, info in CLASSES.items():
        c = sum(1 for s in schedule if s[0] == cls_id)
        print(f"    {info['name']}: {c}")
    print(f"  ~${len(schedule)*0.042:.2f}, ~{len(schedule)*55/60:.0f} мин")
    print(f"\n{'─'*60}\n")

    t0 = time.time()
    ok = errors = 0
    total_spent = 0.0
    counters = {c: 0 for c in CLASSES}

    for i, (cls_id, cls_name, prompt) in enumerate(schedule, 1):
        counters[cls_id] += 1
        idx = counters[cls_id]
        fname = f"{cls_name}_{idx:04d}"

        print(f"  [{i}/{len(schedule)}] {fname} ", end="", flush=True)

        img, cost = generate_image(prompt)

        if img == "NO_MONEY":
            print("БАЛАНС КОНЧИЛСЯ!")
            break

        total_spent += cost if isinstance(cost, (int, float)) else 0

        if img is not None and img != "NO_MONEY":
            img.save(os.path.join(OUT_DIR, cls_name, f"{fname}.jpg"),
                     "JPEG", quality=95)
            with open(os.path.join(labels_dir, f"{fname}.txt"), "w") as f:
                f.write(f"{cls_id} 0.500000 0.500000 1.000000 1.000000\n")

            ok += 1
            elapsed = time.time() - t0
            eta = (elapsed / i) * (len(schedule) - i)
            print(f"✓ ${cost:.3f} [{elapsed/60:.0f}м/~{eta/60:.0f}м] "
                  f"итого ${total_spent:.2f}")
        else:
            errors += 1
            print("✗")
            if errors > 15:
                print("\n  Много ошибок!")
                break

        time.sleep(3)

    t = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ГОТОВО: {ok} картинок, ${total_spent:.2f}")
    print(f"  Время: {int(t//60)}м {int(t%60)}с")
    for info in CLASSES.values():
        d = os.path.join(OUT_DIR, info["name"])
        c = len([f for f in os.listdir(d) if f.endswith(".jpg")])
        print(f"    {info['name']}/  {c}")
    print(f"  {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()