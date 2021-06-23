import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

values = ["jack", "queen", "king", "ace"]
suits = ["hearts", "clubs", "diamonds", "spades"]


def render_svg(in_path, out_path, height):
    print("Rendering {} to {}", in_path, out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "inkscape",
            in_path,
            "-h",
            str(height),
            "--export-filename=" + out_path,
            "-b",
            '"#FFFFFF"',
        ]
    )


def get_images(img_h):
    images = {}
    for value in values:
        for suit in suits:
            path = Path("cards/rendered/{2}/{0}_{1}.png".format(value, suit, img_h))
            if not path.is_file():
                render_svg(
                    "cards/{}_{}.svg".format(value, suit),
                    "cards/rendered/{2}/{0}_{1}.png".format(value, suit, img_h),
                    img_h,
                )
            images[(value, suit)] = Image.open(path)
    return images


def render_all(games, img_h, path):
    imgs = get_images(img_h)
    path.mkdir(parents=True, exist_ok=True)
    for i, cards in enumerate(games):
        images = [apply_transformations(imgs[card]) for card in cards]
        for j, img in enumerate(images):
            img.save(path / "{}_{}.png".format(i, j))


def apply_transformations(image, max_rot=10, hue=13, sat=13, val=13):
    image = image.convert("HSV")
    data = np.array(image, dtype="int16")

    for i, v in enumerate([hue, sat, val]):
        data[:, :, i] += np.random.randint(-v, v, dtype="int16")
        data[:, :, 0] = data[:, :, 0] % 255
        data = data.clip(0, 255)
    image = Image.fromarray(data.astype("uint8"), mode="HSV")
    image = image.convert("RGB")

    w, h = image.width, image.height
    theta = random.uniform(-max_rot, max_rot)
    image = image.rotate(
        theta, resample=Image.BILINEAR, expand=True, fillcolor="#FFFFFF"
    )
    image = image.resize((w, h))

    return image


def read_file(path):
    games = []
    with open(path) as f:
        for line in f:
            cards = line.strip().split(",")[:4]
            cards = [tuple(c.split(" of ")) for c in cards]
            games.append(cards)
    return games


IMG_H = 150
for dataset in ["fair", "unfair"]:  # , 'more_jacks', 'more_spades','ace_spades']:
    render_all(
        read_file("labels/{}.csv".format(dataset)),
        IMG_H,
        Path("images/{}/".format(dataset)),
    )

for dataset in [
    "fair_test",
    "unfair_test",
]:  # , 'more_jacks_test', 'more_spades_test','ace_spades_test']:
    render_all(
        read_file("labels/{}.csv".format(dataset)),
        IMG_H,
        Path("images/{}/".format(dataset)),
    )
