import numpy as np

from data_generation.shape import Shape
from data_generation.utils import (
    sample_contact_many,
    sample_position_inside_1,
    sample_positions_bb,
    sample_random_colors,
)


# ---------- Generador de figuras ----------
def recenter_scene(xy, size, margin=0.05, auto_scale=True):
    """
    Reubica y opcionalmente escala las coordenadas `xy` para que queden centradas
    en el canvas [0,1]x[0,1], respetando un margen definido.
    """
    # Bounding box total de la escena
    bb_min = (xy - size / 2).min(axis=0)[0]
    bb_max = (xy + size / 2).max(axis=0)[0]
    bb_center = (bb_min + bb_max) / 2
    bb_extent = (bb_max - bb_min)

    # Recentrar al nuevo centro aleatorio
    new_center = np.random.rand(2) * (1 - 2 * margin) + margin
    offset = new_center - bb_center
    xy_new = xy + offset[None, None, :]

    if auto_scale:
        # Si alguna dimensión ocupa más del 1 - 2*margin, reducimos la escala
        scale_factors = (1 - 2 * margin) / bb_extent
        scale = min(scale_factors.min(), 1.0)
        xy_centered = xy_new - new_center[None, None, :]
        xy_scaled = xy_centered * scale + new_center[None, None, :]
        return xy_scaled
    else:
        return xy_new



def create_shape(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int | None = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True
):
    if shape_mode == 'normal':
        return Shape(radius=radius, hole_radius=hole_radius)

    if shape_mode == 'rigid':
        s = Shape(randomize=False)
        s.rigid_transform(type='polygon', points=n_sides, rotate=1)
        return s

    if shape_mode == 'smooth':
        s = Shape(radius=radius, hole_radius=hole_radius)
        s.smooth(fourier_terms=fourier_terms)
        return s

    if shape_mode == 'symm':
        s = Shape(radius=radius, hole_radius=hole_radius)
        s.symmetrize(rotate=symm_rotate)
        return s

    raise ValueError(f"shape_mode '{shape_mode}' no reconocido")

# ---------- Decorador de figuras ----------

def decorate_shapes(shapes, max_size=0.4, min_size=None):
    """
    Adorna N figuras con tamaño, posición (sin solapamiento), rotación, flip y color.
    Devuelve: (xy, size, shape_wrapped, colors)
    """
    n = len(shapes)
    min_size = min_size or max_size / 2

    size_vals = np.random.rand(n) * (max_size - min_size) + min_size
    size = size_vals[:, None]  # shape (n, 1)

    # Posiciones sin superposición usando bounding boxes
    size_batch = size[None, ...]  # shape (1, n, 1)
    xy_vals = sample_positions_bb(size_batch)[0]  # shape (n, 2)
    xy = xy_vals[:, None, :]  # shape (n, 1, 2)

    for s in shapes:
        s.rotate(np.random.rand() * 2 * np.pi)
        if np.random.rand() < 0.5:
            s.flip()

    colors = sample_random_colors(n)
    colors = [colors[i:i+1] for i in range(n)]
    shape_wrapped = [[s] for s in shapes]

    return xy, size, shape_wrapped, colors

# ---------- Tarea SVRT 1: mismo tipo vs distinto tipo ----------

def task_svrt_1(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2
):
    """
    SVRT #1 – Devuelve (sample_pos, sample_neg).
    sample_pos: dos figuras del mismo tipo (clase 1)
    sample_neg: dos figuras de distinto tipo (clase 0)
    """
    shape1 = create_shape(shape_mode, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    n_sides_diff = np.random.choice([k for k in range(poly_min_sides, poly_max_sides) if k != n_sides])
    shape3 = create_shape(shape_mode, radius, hole_radius, n_sides_diff, fourier_terms, symm_rotate)

    sample_pos = decorate_shapes([shape1.clone(), shape1.clone()], max_size=max_size, min_size=min_size)
    sample_neg = decorate_shapes([shape2, shape3], max_size=max_size, min_size=min_size)

    return sample_neg, sample_pos

def task_svrt_2(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
):
    """
    SVRT #2 – Devuelve (sample_neg, sample_pos).
    sample_pos: una figura completamente dentro de otra (clase 1)
    sample_neg: dos figuras separadas (clase 0)
    """

    # Elegir n_sides diferentes si estamos usando 'rigid'
    if shape_mode == 'rigid':
        outer_sides = n_sides
        inner_sides = np.random.choice([k for k in range(poly_min_sides, poly_max_sides) if k != n_sides])
    else:
        outer_sides = inner_sides = n_sides

    # Positivo: inner dentro de outer con posición aleatoria
    outer = create_shape(shape_mode, radius, hole_radius, outer_sides, fourier_terms)
    inner = create_shape(shape_mode, radius, hole_radius, inner_sides, fourier_terms)

    size_outer = max_size * 0.9
    size_inner = size_outer * 0.3

    # posición global del centro del outer
    xy_outer = np.random.rand(2) * (1 - size_outer) + size_outer / 2

    # obtener posición relativa del inner dentro del outer
    xy_inner_rel = sample_position_inside_1(outer, inner, scale=size_inner / size_outer)
    if len(xy_inner_rel) == 0:
        xy_inner = xy_outer + 0.01  # fallback
    else:
        xy_inner = xy_inner_rel[0] * size_outer + xy_outer

    xy_pos = np.stack([xy_outer, xy_inner])[:, None, :]
    size_pos = np.array([[size_outer], [size_inner]])
    shapes_pos = [[outer], [inner]]
    color_pos = sample_random_colors(2)
    color_pos = [color_pos[i:i+1] for i in range(2)]
    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)

    # Negativo: dos figuras separadas (también con distintos n_sides si es 'rigid')
    if shape_mode == 'rigid':
        n_a = n_sides
        n_b = np.random.choice([k for k in range(poly_min_sides, poly_max_sides) if k != n_sides])
        shape_a = create_shape(shape_mode, radius, hole_radius, n_a, fourier_terms)
        shape_b = create_shape(shape_mode, radius, hole_radius, n_b, fourier_terms)
    else:
        shape_a = create_shape(shape_mode, radius, hole_radius, n_sides, fourier_terms)
        shape_b = create_shape(shape_mode, radius, hole_radius, n_sides, fourier_terms)

    sample_neg = decorate_shapes([shape_a, shape_b], max_size=max_size, min_size=min_size)

    return sample_neg, sample_pos

def task_svrt_3(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    max_size: float = 0.3,
    min_size: float | None = 0.2,
    shrink_factor: float = 0.5,
    min_group_dist: float = 0.4,
):
    """
    SVRT #3 – Clase 1: 3 figuras en contacto, 1 separada.
               Clase 0: dos pares en contacto, sin tocarse entre sí.
    """

    def normalize_scene(xy, size, margin=0.05):
        bb_min = (xy - size / 2).min(axis=0)[0]
        bb_max = (xy + size / 2).max(axis=0)[0]
        scale = (1 - 2 * margin) / (bb_max - bb_min).max()
        offset = (0.5 - (bb_min + bb_max) / 2 * scale)
        return xy * scale + offset, size * scale

    # --------- Tamaños ---------
    size_vals = np.random.rand(4) * (max_size - min_size) + min_size
    size_vals *= shrink_factor
    size = size_vals[:, None]

    # --------- Positivo ---------
    shapes = [create_shape(shape_mode, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(4)]
    contact3, solo = shapes[:3], shapes[3]

    xy_contact3, _ = sample_contact_many(contact3, size_vals[:3])
    xy_lone = sample_positions_bb(size[None, 3:], n_sample_min=1)[0, 0]

    xy_pos = np.concatenate([xy_contact3, xy_lone[None]], axis=0)[:, None, :]
    xy_pos, size = normalize_scene(xy_pos, size)
    shapes_pos = [[s] for s in contact3 + [solo]]
    colors_pos = [c.flatten() for c in sample_random_colors(4)]
    sample_pos = (xy_pos, size, shapes_pos, colors_pos)

    # --------- Negativo: dos pares alejados ---------
    shapes1 = [create_shape(shape_mode, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(2)]
    shapes2 = [create_shape(shape_mode, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(2)]

    xy1, _ = sample_contact_many(shapes1, size_vals[:2])
    xy2, _ = sample_contact_many(shapes2, size_vals[2:])

    # Calcular centros
    center1 = xy1.mean(axis=0)
    center2 = xy2.mean(axis=0)

    # Vector de dirección aleatorio
    direction = np.random.rand(2) - 0.5
    direction /= np.linalg.norm(direction)

    # Calcular desplazamiento necesario
    current_dist = np.linalg.norm(center2 - center1)
    move = (min_group_dist - current_dist) * direction if current_dist < min_group_dist else np.zeros(2)

    xy2_shifted = xy2 + move

    # Combinar escena
    xy_neg = np.concatenate([xy1, xy2_shifted], axis=0)[:, None, :]
    xy_neg, size = normalize_scene(xy_neg, size)
    shapes_neg = [[s] for s in shapes1 + shapes2]
    colors_neg = [c.flatten() for c in sample_random_colors(4)]
    sample_neg = (xy_neg, size, shapes_neg, colors_neg)

    return sample_neg, sample_pos




# ---------- Registro de tareas ----------

TASKS_SVRT = [
    ["task_svrt_1", task_svrt_1, "The images contain objects of the same shape."],
    ["task_svrt_2", task_svrt_2, "One object is inside the other."],
    ["task_svrt_3", task_svrt_3, "The images contain objects in contact."],
]
