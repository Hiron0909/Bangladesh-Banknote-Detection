import random, shutil
from pathlib import Path
import cv2
import numpy as np

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATASET = BASE_DIR / 'Dataset'      # folders: 2,5,10...
BACKGROUND_DIR = BASE_DIR / 'Background'
OUTPUT_DIR = BASE_DIR / 'yolo_dataset'
IMAGE_SIZE = 640
TRAIN_RATIO = 0.85
SAMPLES_PER_CLASS = 800
SEED = 42
SCALE_RANGE = (0.30, 0.65)
ROTATE_RANGE = (-15, 15)
# ==========================================

random.seed(SEED)
np.random.seed(SEED)

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
classes = sorted([p.name for p in SOURCE_DATASET.iterdir() if p.is_dir()], key=lambda x: int(x))
class_map = {name: i for i, name in enumerate(classes)}


def files_in(folder):
    return [p for p in folder.rglob('*') if p.suffix.lower() in EXTS]


def prepare_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def remove_bg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    return mask


def rotate(img, mask, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    img2 = cv2.warpAffine(img, M, (nw, nh), borderValue=(255,255,255))
    mask2 = cv2.warpAffine(mask, M, (nw, nh), borderValue=0)
    return img2, mask2


def paste(bg, fg, mask, x, y):
    h, w = fg.shape[:2]
    roi = bg[y:y+h, x:x+w]
    alpha = (mask / 255.0)[..., None]
    bg[y:y+h, x:x+w] = (fg * alpha + roi * (1 - alpha)).astype(np.uint8)
    return bg


def save_yaml():
    txt = 'path: yolo_dataset\ntrain: images/train\nval: images/val\n\nnames:\n'
    for k, v in class_map.items():
        txt += f'  {v}: {k}\n'
    (OUTPUT_DIR / 'data.yaml').write_text(txt, encoding='utf-8')


def main():
    prepare_output()
    backgrounds = files_in(BACKGROUND_DIR)
    if not backgrounds:
        print('No backgrounds found')
        return

    for cls in classes:
        note_files = files_in(SOURCE_DATASET / cls)
        cid = class_map[cls]
        for i in range(SAMPLES_PER_CLASS):
            note = cv2.imread(str(random.choice(note_files)))
            bg = cv2.imread(str(random.choice(backgrounds)))
            if note is None or bg is None:
                continue

            bg = cv2.resize(bg, (IMAGE_SIZE, IMAGE_SIZE))
            scale = random.uniform(*SCALE_RANGE)
            new_w = int(IMAGE_SIZE * scale)
            new_h = int(note.shape[0] * new_w / note.shape[1])
            note = cv2.resize(note, (new_w, new_h))

            mask = remove_bg(note)
            note, mask = rotate(note, mask, random.uniform(*ROTATE_RANGE))
            h, w = note.shape[:2]
            if h >= IMAGE_SIZE or w >= IMAGE_SIZE:
                continue

            x = random.randint(0, IMAGE_SIZE - w)
            y = random.randint(0, IMAGE_SIZE - h)
            img = paste(bg.copy(), note, mask, x, y)

            split = 'train' if random.random() < TRAIN_RATIO else 'val'
            name = f'{cls}_{i:05d}'
            cv2.imwrite(str(OUTPUT_DIR / f'images/{split}/{name}.jpg'), img)

            xc = (x + w/2) / IMAGE_SIZE
            yc = (y + h/2) / IMAGE_SIZE
            bw = w / IMAGE_SIZE
            bh = h / IMAGE_SIZE
            label = f'{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n'
            (OUTPUT_DIR / f'labels/{split}/{name}.txt').write_text(label)

    save_yaml()
    print('Dataset created successfully!')
    print(class_map)

if __name__ == '__main__':
    main()