import os
import math
import random
from PIL import Image
import numpy as np

im = Image.open("approaching_storm.jpg")
pixels = np.array(im, dtype=np.float)
size = im.size

# Gamma校正
for x in np.nditer(pixels, op_flags=['readwrite']):
    x[...] = pow(x / 255, 2)

math.one_over_pi = 1 / math.pi


def LongLat2UV(dir):
    v = math.acos(dir[2]) * math.one_over_pi
    u = (math.atan2(dir[1], dir[0]) * math.one_over_pi * 0.5) % 1
    return (u, v)


def UV2LongLat(uv):
    theta = math.pi * uv[1]
    phi = 2 * math.pi * uv[0]

    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    z = costheta
    x = sintheta * math.cos(phi)
    y = sintheta * math.sin(phi)
    return np.array((x, y, z))


def sample(dir):
    uv = LongLat2UV(dir)

    x = uv[0] * size[0] - 0.5
    x0 = math.floor(x) % size[0]
    x1 = (x0 + 1) % size[0]
    y = uv[1] * size[1] - 0.5
    y0 = max(math.floor(y), 0)
    y1 = min(y0 + 1, size[1] - 1)

    p00 = pixels[y0, x0]
    p01 = pixels[y1, x0]
    p10 = pixels[y0, x1]
    p11 = pixels[y1, x1]
    kx = x % 1
    ky = y % 1

    return p00 * (1 - kx) * (1 - ky) + p01 * (1 - kx) * ky + p10 * kx * (1 - ky) + p11 * kx * ky


def random_sequenc(n):
    one_over_n = 1 / n
    sequence = []
    for i in range(n):
        for j in range(n):
            sequence.append((one_over_n * (i + random.random()),
                             one_over_n * (j + random.random())))
    return sequence


def normalize(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    inv_length = 1 / (max(length, 0.0001))
    return np.array((v[0] * inv_length, v[1] * inv_length, v[2] * inv_length))


# 指定dir作为正z轴, 得到切空间变换矩阵
def transform_matrix(dir):
    v3 = dir
    v1 = np.array((1, 0, 0))
    dot = np.inner(v1, v3)
    # if abs(dot - 1) < 0.01:
    #     v2 = np.array((0, 1, 0))
    #     v2 = normalize(v2 - v2 * np.inner(v2, v3))
    #     v1 = np.cross(v2, v3)
    # else:
    v1 = normalize(v1 - v1 * dot)
    v2 = np.cross(v3, v1)
    return np.array((v1, v2, v3))


# 余弦权重在半球面上取一点
def sample_consine_hemisphere(eta):
    r = math.sqrt(eta[0])
    theta = 2 * math.pi * eta[1]
    z = math.sqrt(1 - eta[0])
    cos_theta = math.cos(theta)
    x = r * cos_theta
    y = r * math.sin(theta)
    # 返回随机的点
    return np.array((x, y, z))


# 蒙特卡洛估计
def monte_carlo(direction, etas):
    s = np.zeros(3, dtype=np.float)
    transform = transform_matrix(direction)
    for eta in etas:
        d = sample_consine_hemisphere(eta)
        rotate_d = d @ transform
        s += sample(rotate_d)
    return s / len(etas)


def sample_uniform_angle(eta):
    theta = math.pi * 0.5 * eta[0]
    phi = math.pi * 2 * eta[1]
    z = math.cos(theta)
    sin_theta = math.sin(theta)
    x = sin_theta * math.cos(phi)
    y = sin_theta * math.sin(phi)
    return np.array((x, y, z)), math.cos(theta), sin_theta


# 黎曼和估计
def riemann_sum(direction, etas):
    s = np.zeros(3, dtype=np.float)
    transform = transform_matrix(direction)
    for eta in etas:
        d, cos_theta, sin_theta = sample_uniform_angle(eta)
        rotate_d = d @ transform
        sam = sample(rotate_d)
        s += sam * cos_theta * sin_theta
    return s / len(etas) * math.pi


pixels2 = np.empty([size[1], size[0], 3], dtype=np.float)
for x in range(size[0]):
    for y in range(size[1]):
        random_etas = random_sequenc(4)
        uv = ((x + 0.5) / size[0], (y + 0.5) / size[1])
        pixels2[y, x] = monte_carlo(UV2LongLat(uv), random_etas)

# Gamma
for x in np.nditer(pixels2, op_flags=['readwrite']):
    x[...] = min(math.sqrt(x) * 255, 255)

im2 = Image.fromarray(pixels2.astype('uint8'), 'RGB')
im2.save('output_monte_carlo.jpg')

pixels3 = np.empty([size[1], size[0], 3], dtype=np.float)
for x in range(size[0]):
    for y in range(size[1]):
        random_etas = random_sequenc(4)
        uv = ((x + 0.5) / size[0], (y + 0.5) / size[1])
        pixels3[y, x] = riemann_sum(UV2LongLat(uv), random_etas)

# Gamma
for x in np.nditer(pixels3, op_flags=['readwrite']):
    x[...] = min(math.sqrt(x) * 255, 255)

im3 = Image.fromarray(pixels3.astype('uint8'), 'RGB')
im3.save('output_riemann_sum.jpg')

