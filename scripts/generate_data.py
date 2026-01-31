from PIL import Image, ImageDraw, ImageFont
import os
import glob
import random
import shutil

IMG_SIZE = 256
FONT_SIZE = 200
TRAIN_SPLIT = 0.9
BACKGROUND_COLOR = 255

FONT_PATH = "song.ttf"
CHAR_LIST_PATH = "Chinese-common.txt"

HANDWRITING_DIR = "./handwrittings"

A_ALL = "dataset/allA"
A_TRAIN = "dataset/trainA"
A_TEST = "dataset/testA"

B_ALL = "dataset/allB"
B_RESIZED = "dataset/allB_resized"
B_TRAIN = "dataset/trainB"
B_TEST = "dataset/testB"

chars = []
seen = set()

with open(CHAR_LIST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        for ch in line.strip():
            if ch not in seen:
                chars.append(ch)
                seen.add(ch)
            if len(chars) >= 1000:
                break

print(f"Loaded {len(chars)} characters")


for d in [A_ALL, A_TRAIN, A_TEST]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)


def center_font(img):
    bounds = img.getbbox()
    if bounds:
        img = img.crop(bounds)
        img = img.resize((int(IMG_SIZE * 0.95), int(IMG_SIZE * 0.95)), Image.LANCZOS)

    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), BACKGROUND_COLOR)
    canvas.paste(img, ((IMG_SIZE - img.width) // 2 + 25,
                        (IMG_SIZE - img.height) // 2 - 25))
    return canvas


for idx, ch in enumerate(chars):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, font=font, fill=0)
    img = center_font(img)
    img.save(os.path.join(A_ALL, f"{idx}_{ch}.png"))


os.makedirs(B_ALL, exist_ok=True)

for root, _, files in os.walk(HANDWRITING_DIR):
    for filename in files:
        if not filename.endswith(".gif"):
            continue

        ch = root[-1]
        if ch not in chars:
            continue

        idx = chars.index(ch)
        img = Image.open(os.path.join(root, filename)).convert("L")
        img.save(os.path.join(B_ALL, f"{idx}_{ch}.png"))


os.makedirs(B_RESIZED, exist_ok=True)


def center_handwritten(img):
    bounds = img.getbbox()
    if bounds:
        img = img.crop(bounds)
        img = img.resize((int(IMG_SIZE * 0.80), int(IMG_SIZE * 0.80)), Image.LANCZOS)

    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), BACKGROUND_COLOR)
    canvas.paste(img, ((IMG_SIZE - img.width) // 2 + 10,
                        (IMG_SIZE - img.height) // 2 - 5))
    return canvas


for path in glob.glob(os.path.join(B_ALL, "*.png")):
    img = Image.open(path)
    img = center_handwritten(img)
    img.save(os.path.join(B_RESIZED, os.path.basename(path)))


def split_dataset(src, train, test):
    for d in [train, test]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    files = os.listdir(src)
    random.shuffle(files)
    split = int(len(files) * TRAIN_SPLIT)

    for f in files[:split]:
        shutil.copy(os.path.join(src, f), train)
    for f in files[split:]:
        shutil.copy(os.path.join(src, f), test)


split_dataset(A_ALL, A_TRAIN, A_TEST)
split_dataset(B_RESIZED, B_TRAIN, B_TEST)

shutil.rmtree(A_ALL)
shutil.rmtree(B_ALL)
shutil.rmtree(B_RESIZED)

print("Dataset ready.")
print("TrainA:", len(os.listdir(A_TRAIN)))
print("TrainB:", len(os.listdir(B_TRAIN)))
