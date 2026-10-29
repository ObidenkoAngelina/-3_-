import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

plt.rcParams['toolbar'] = 'None' 

W = H = 300 # размер картинки 300×300 пикселей
F = 1.3 * W # фокусное расстояние камеры (для проекции)

# ---------- параметры модели освещённости ----------
Ia, Ka = 0.2, 0.5    # фоновый свет и его коэффициент
Il = 8.0             # яркость источника
k = 5.0              # затухание 
kd, ks, n = 0.7, 0.6, 30   # диффуз, зеркальный, степень Фонга
LIGHT = np.array([4.0, -4.0, 5.0])   # где стоит лампа

# ---------- камера ----------
P1 = np.array([5.0, -5.0, 1.0])
P2 = np.array([-5.0, -4.0, 4.0])
FRAMES = 50 # сколько кадров в анимации

# ---------- 1. Фигура вращения ----------
M, N = 40, 48                    # M — точек вдоль образующей, N — по окружности
t = np.linspace(0, 1, M)         # параметр вдоль образующей (0..1)

# r(t) — радиус на высоте t: гофры + сужение к полюсам
r = (0.8 + 0.3 * np.sin(4 * np.pi * t)) * np.sqrt(np.sin(np.pi * t))
h = 3 * t - 1.5                  # высота: от -1.5 до +1.5

phi = np.linspace(0, 2 * np.pi, N, endpoint=False)  # углы поворота вокруг оси OZ: 48 точек, равномерно от 0 до 2π
V = np.zeros((M * N, 3))         # массив вершин (x, y, z)

for i in range(M): # Вращение образующей вокруг OZ: x=r·cosφ, y=r·sinφ, z=h
    for j in range(N):
        V[i * N + j] = [r[i] * np.cos(phi[j]),
                        r[i] * np.sin(phi[j]),
                        h[i]]

# ---------- 2. Триангуляция ----------
T = []
for i in range(M - 1):
    for j in range(N):
        a = i * N + j
        b = i * N + (j + 1) % N
        c = (i + 1) * N + (j + 1) % N
        d = (i + 1) * N + j
        T.append((a, b, c))
        T.append((a, c, d))
T = np.array(T)

A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
FN = np.cross(B - A, C - A) # нормаль каждой грани. (B-A) × (C-A)
len_fn = np.linalg.norm(FN, axis=1) # и нормируем (делаем длину = 1)
FN[len_fn > 1e-12] /= len_fn[len_fn > 1e-12, None]

# Каждая вершина собирает нормали соседних граней и берёт их среднее
VN = np.zeros_like(V) 
for f in range(len(T)):
    for v in T[f]:
        VN[v] += FN[f] # для каждой вершины складываем нормали всех граней, где она встречается
len_vn = np.linalg.norm(VN, axis=1) # и нормируем (делаем длину = 1)
VN[len_vn > 1e-12] /= len_vn[len_vn > 1e-12, None] # делим на длину → длина 1

# ---------- 3. Модель освещённости ----------
def light(P, Nrm, cam):
    Nrm = Nrm / np.maximum(np.linalg.norm(Nrm, axis=-1, keepdims=True), 1e-12)
    L = LIGHT - P
    L = L / np.linalg.norm(L, axis=-1, keepdims=True)
    S = cam - P
    d = np.linalg.norm(S, axis=-1)
    S = S / d[..., None]
    NL = np.maximum(np.sum(Nrm * L, axis=-1), 0)
    R = 2 * NL[..., None] * Nrm - L
    RS = np.maximum(np.sum(R * S, axis=-1), 0)
    I = Ia * Ka + Il / (k + d) * (kd * NL + ks * (NL > 0) * RS ** n)
    return np.clip(I, 0, 1)

# ---------- Проекция и растеризация ----------
def render(cam, phong):
    f = -cam / np.linalg.norm(cam) # куда смотрит камера (в начало координат)
    right = np.cross(f, [0, 0, 1]); right /= np.linalg.norm(right) # правый вектор
    up = np.cross(right, f) # верхний вектор

    Q = V - cam
    xc, yc, zc = Q @ right, Q @ up, Q @ f # координаты вершин в системе камеры
    sx = W / 2 + F * xc / zc # проекция на экран X
    sy = H / 2 - F * yc / zc # проекция на экран Y (ось вниз)

    if not phong:
        centers = (A + B + C) / 3
        flat = light(centers, FN, cam) # плоская: яркость по центру грани

    img = np.zeros((H, W))
    zbuf = np.full((H, W), np.inf) # z-буфер: глубины пикселей

    for q in range(len(T)):
        a, b, c = T[q]
        if min(zc[a], zc[b], zc[c]) < 0.1: continue # если треугольник за камерой — пропускаем.
        x0 = max(int(min(sx[a], sx[b], sx[c])), 0)
        x1 = min(int(max(sx[a], sx[b], sx[c])) + 1, W - 1)
        y0 = max(int(min(sy[a], sy[b], sy[c])), 0)
        y1 = min(int(max(sy[a], sy[b], sy[c])) + 1, H - 1)
        if x0 > x1 or y0 > y1: continue
        area = (sx[b]-sx[a])*(sy[c]-sy[a]) - (sx[c]-sx[a])*(sy[b]-sy[a]) # площадь всего треугольника 
        if abs(area) < 1e-9: continue

        xs, ys = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
        #площадь маленького треугольника PBC, делённая на площадь всего. То есть доля A.
        w0 = ((sx[b]-xs)*(sy[c]-ys) - (sx[c]-xs)*(sy[b]-ys)) / area
        #площадь треугольника PCA, делённая на площадь всего. Доля B.
        w1 = ((sx[c]-xs)*(sy[a]-ys) - (sx[a]-xs)*(sy[c]-ys)) / area
        w2 = 1 - w0 - w1 # доля C (чтобы сумма была 1)
        depth = w0*zc[a] + w1*zc[b] + w2*zc[c] # Смешиваем глубины вершин
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (depth < zbuf[ys, xs])
        if not mask.any(): continue
        zbuf[ys[mask], xs[mask]] = depth[mask]

        if not phong:
            img[ys[mask], xs[mask]] = flat[q]
        else:
            a0, a1, a2 = w0[mask, None], w1[mask, None], w2[mask, None]
            Pp = a0*V[a] + a1*V[b] + a2*V[c] # позиция точки (нужна для вектора S)
            Np = a0*VN[a] + a1*VN[b] + a2*VN[c]
            img[ys[mask], xs[mask]] = light(Pp, Np, cam) # считаем яркость В КАЖДОЙ ТОЧКЕ
    return img

def camera(s):
    return P1 + s * (P2 - P1)

# ---------- ОДНО ОКНО ----------
fig = plt.figure(figsize=(13, 5))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0.08)

ax_flat  = fig.add_subplot(gs[0, 0])
ax_phong = fig.add_subplot(gs[0, 1])
ax_anim  = fig.add_subplot(gs[0, 2])

# статичные кадры (камера в P1)
ax_flat.imshow(render(camera(0), False), cmap='gray', vmin=0, vmax=1)
ax_flat.set_title('Простая закраска')

ax_phong.imshow(render(camera(0), True), cmap='gray', vmin=0, vmax=1)
ax_phong.set_title('Закраска Фонга')

for a_ in (ax_flat, ax_phong):
    a_.axis('off')

# предрасчёт кадров анимации
frames = [render(camera(s), True) for s in np.linspace(0, 1, FRAMES)]

im = ax_anim.imshow(frames[0], cmap='gray', vmin=0, vmax=1)
ax_anim.set_title('Движение камеры')
ax_anim.axis('off')

def update(i):
    im.set_data(frames[i])
    return [im]

anim = FuncAnimation(fig, update, frames=FRAMES, interval=100, repeat=True)

plt.tight_layout()
plt.show()