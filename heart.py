import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl

# hi there
# plt.style.use('seaborn')

# Создаем фигуру
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.axis('off')

# Основное сердце
theta = np.linspace(0, 2*np.pi, 1000)
x_heart = 16 * np.sin(theta)**3
y_heart = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
heart = plt.plot(x_heart, y_heart, color='crimson', linewidth=7, zorder=3)

# Заливка сердца градиентом
x_fill = np.linspace(-16, 16, 400)
y_fill = np.linspace(-15, 15, 400)
X, Y = np.meshgrid(x_fill, y_fill)
F = (X**2 + Y**2 - 1)**3 - X**2 * Y**3
ax.contourf(X, Y, F, levels=[-100, 0], colors=['darkred'], alpha=0.7, zorder=2)

# Вензели вокруг сердца
def draw_ornament(x, y, scale=1.0, angle=0, color='gold'):
    verts = [
        (0., 0.),
        (0.2, 0.3),
        (0.5, 0.4),
        (0.8, 0.3),
        (1.0, 0.0),
        (0.8, -0.3),
        (0.5, -0.4),
        (0.2, -0.3),
        (0., 0.),
    ]
    
    verts = np.array(verts)
    verts *= scale
    
    # Поворот
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    verts = np.dot(verts, rot)
    
    # Смещение
    verts[:, 0] += x
    verts[:, 1] += y
    
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) - 1)
    
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color, edgecolor='darkgoldenrod', lw=1.5, zorder=4)
    ax.add_patch(patch)

# Рисуем вензели по кругу
n_ornaments = 24
for i in range(n_ornaments):
    angle = 360 / n_ornaments * i
    rad = np.radians(angle)
    distance = 18 + 2 * np.sin(5 * rad)  # Небольшая вариация расстояния для изящества
    
    x = distance * np.cos(rad)
    y = distance * np.sin(rad)
    
    draw_ornament(x, y, scale=1.5 + 0.5 * np.sin(3 * rad), angle=angle+45, color='gold')

# Декоративные элементы внутри сердца
def draw_swirl(x, y, scale=1.0, angle=0, color='gold'):
    t = np.linspace(0, 2*np.pi, 100)
    r = 0.5 * (1 + 0.2 * np.sin(5*t))
    x_swirl = r * np.cos(t) * scale
    y_swirl = r * np.sin(t) * scale
    
    # Поворот
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    coords = np.column_stack([x_swirl, y_swirl])
    coords = np.dot(coords, rot)
    
    x_swirl = coords[:, 0] + x
    y_swirl = coords[:, 1] + y
    
    ax.plot(x_swirl, y_swirl, color=color, linewidth=2, zorder=4)

# Рисуем завитки внутри сердца
for i in range(8):
    angle = 360 / 8 * i
    rad = np.radians(angle)
    distance = 8 + 2 * np.sin(3 * rad)
    
    x = distance * np.cos(rad)
    y = distance * np.sin(rad)
    
    draw_swirl(x, y, scale=1.5, angle=angle*2, color='gold')

# Добавляем текст (инициалы)
ax.text(0, -3, "A & L", fontsize=24, fontfamily='script', color='gold', 
        ha='center', va='center', zorder=5)

# Добавляем свечение
ax.scatter([0], [0], s=2000, color='red', alpha=0.3, zorder=1)

plt.tight_layout()
plt.show()
