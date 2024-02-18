from sys import argv
from math import cos, radians

import os
from PIL import Image
import numpy
from time import time, sleep

FILE = "tmp.jpg"
STEP = 10


def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = numpy.matrix(matrix, dtype=float)
    B = numpy.array(source_coords).reshape(8)
    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


def make_left_transform(img, angle, W=1024, H=1024):
    cosA = cos(radians(angle))
    img = img.resize((W, H))
    w, h = img.width, img.height
    ww = w * cosA
    coeffs = find_coeffs(
        [(0, 0), (W, 0), (W, H), (0, H)],
        [(0, 0), (ww, 0), (ww, H), (0, H)])

    return img.transform((W, H), Image.PERSPECTIVE, coeffs,
                  Image.BICUBIC).crop(((0, 0, ww, H)))


def make_right_transform(img, angle, W=1024, H=1024):
    cosA = cos(radians(90 - angle - 0.1))
    img = img.resize((W, H))
    w, h = img.width, img.height
    ww = w * cosA
    coeffs = find_coeffs(
        [(0, 0), (W, 0), (W, H), (0, H)],
        [(W - ww, 0), (W, 0), (W, H), (W - ww, H)])

    return img.transform((W, H), Image.PERSPECTIVE, coeffs,
                  Image.BICUBIC).crop((W - ww, 0, W, H))


def merge_slide(img1, img2, angle):
    l = make_left_transform(img1, angle, img1.width, img1.height)
    r = make_right_transform(img2, angle, img2.width, img2.height)
    return merge_images(l, r)


def merge_images(img1, img2):
    res = Image.new(img1.mode, (img1.width + img2.width, img1.height), (255, 255, 255))
    res.paste(img1, (0, 0))
    res.paste(img2, (img1.width, 0))
    return res


def slice_image(img, parts):
    step = img.width / parts
    slices = []
    for j in range(parts):
        slice_ = img.crop((j * step, 0, (j + 1) * step, img.height))
        slices.append(slice_)
    return slices


def made_everything(img1, img2):
    STEP = 5
    parts = 32
    total = []
    for i in range(STEP + 1):
        im1_sl = slice_image(img1, parts)
        im2_sl = slice_image(img2, parts)
        pairs = list(zip(im1_sl, im2_sl))

        res_pairs = []
        for im1, im2 in pairs:
            r = merge_slide(im1, im2, (90 / STEP) * i)
            res_pairs.append(r)

        res = res_pairs[0]
        for j in range(1, len(res_pairs)):
            res = merge_images(res, res_pairs[j])
        total.append(res.resize(img1.size))
    return total


def image_show(img):
    img.save(FILE)
    os.system("fbi -d /dev/fb0 -T 5 tmp.jpg")


def trifects(img1: str, img2: str, img3: str, rotation_delay=0.01, show_delay=1):
    images = [Image.open(img1).resize((W, H)), Image.open(img2).resize((W, H)), Image.open(img3).resize((W, H))]
    image = 0
    while True:
        image_show(images[image])
        sleep(show_delay)
        for changes in made_everything(images[image], images[(image + 1) % len(images)]):
            image_show(changes)
            sleep(rotation_delay)

        image = (image + 1) % len(images)


W, H = 1024, 1024

trifects(argv[1], argv[2], argv[3])
