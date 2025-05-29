import numpy as np

from data_generation.shape import Shape
from data_generation.utils import (
    sample_contact_many,
    sample_position_inside_1,
    sample_positions_bb,
    sample_positions_align,
    sample_random_colors,
)


# ---------- Generador de figuras ----------

def create_shape(
    shape_mode: str = 'normal',
    rigid_type: str = 'polygon',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int | None = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True
):
    if shape_mode == 'normal':
        return Shape(radius=radius, hole_radius=hole_radius)

    if shape_mode == 'rigid':
        s = Shape(radius=radius, hole_radius=hole_radius)
        s.rigid_transform(type=rigid_type, points=n_sides, rotate=1)
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

def decorate_shapes(shapes, max_size=0.4, min_size=None, size=False, rotate=False, color=False, flip=False, align=False):
    """
    Adorna N figuras con tamaño, posición (sin solapamiento), rotación, flip y color.
    Devuelve: (xy, size, shape_wrapped, colors)
    """
    n = len(shapes)

    if size:
        min_size = min_size or max_size / 2
        size_vals = np.random.rand(n) * (max_size - min_size) + min_size
        size = size_vals[:, None]  # shape (n, 1)
    else:
        size = np.full((n, 1), fill_value=max_size/2)

    # Posiciones sin superposición usando bounding boxes
    size_batch = size[None, ...]  # shape (1, n, 1)
    if align:
        xy_vals = sample_positions_align(size_batch)[0]  # shape (n, 2)

    else:
        xy_vals = sample_positions_bb(size_batch)[0]  # shape (n, 2)

    xy = xy_vals[:, None, :]  # shape (n, 1, 2)

    for s in shapes:
        if rotate:
            s.rotate(np.random.rand() * 2 * np.pi)
        if flip and np.random.rand() > 0.5:
            s.flip()

    if color:
        colors = sample_random_colors(n)
        colors = [colors[i:i+1] for i in range(n)]
    else:
        colors = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(n)]

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
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #1 – Devuelve (sample_pos, sample_neg).
    sample_pos: dos figuras del mismo tipo (clase 1)
    sample_neg: dos figuras de distinto tipo (clase 0)
    """
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    n_sides_diff = np.random.choice([k for k in range(poly_min_sides, poly_max_sides) if k != n_sides])
    shape3 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides_diff, fourier_terms, symm_rotate)

    sample_pos = decorate_shapes([shape1.clone(), shape1.clone()], max_size=max_size, min_size=min_size, color=color)
    sample_neg = decorate_shapes([shape2, shape3], max_size=max_size, min_size=min_size, color=color)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 2 ----------

def task_svrt_2(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #2 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 3 ----------

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
    color: bool = False,
    rigid_type: str = 'polygon'
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
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(4)]
    contact3, solo = shapes[:3], shapes[3]

    xy_contact3, _ = sample_contact_many(contact3, size_vals[:3])
    xy_lone = sample_positions_bb(size[None, 3:], n_sample_min=1)[0, 0]

    xy_pos = np.concatenate([xy_contact3, xy_lone[None]], axis=0)[:, None, :]
    xy_pos, size = normalize_scene(xy_pos, size)
    shapes_pos = [[s] for s in contact3 + [solo]]
    if color:
        colors_pos = [c.flatten() for c in sample_random_colors(4)]
    else:
        colors_pos = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(4)]
    sample_pos = (xy_pos, size, shapes_pos, colors_pos)

    # --------- Negativo: dos pares alejados ---------
    shapes1 = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(2)]
    shapes2 = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(2)]

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
    if color:
        colors_neg = [c.flatten() for c in sample_random_colors(4)]
    else:
        colors_neg = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(4)]
    sample_neg = (xy_neg, size, shapes_neg, colors_neg)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 4 ----------

def task_svrt_4(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #4 – Devuelve (sample_neg, sample_pos).
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
    outer = create_shape(shape_mode, rigid_type, radius, hole_radius, outer_sides, fourier_terms)
    inner = create_shape(shape_mode, rigid_type, radius, hole_radius, inner_sides, fourier_terms)

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
    if color:
        color_pos = sample_random_colors(2)
        color_pos = [color_pos[i:i+1] for i in range(2)]
    else:
        color_pos = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(2)]
    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)

    # Negativo: dos figuras separadas (también con distintos n_sides si es 'rigid')
    if shape_mode == 'rigid':
        n_a = n_sides
        n_b = np.random.choice([k for k in range(poly_min_sides, poly_max_sides) if k != n_sides])
        shape_a = create_shape(shape_mode, rigid_type, radius, hole_radius, n_a, fourier_terms)
        shape_b = create_shape(shape_mode, rigid_type, radius, hole_radius, n_b, fourier_terms)
    else:
        shape_a = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)
        shape_b = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)

    sample_neg = decorate_shapes([shape_a, shape_b], max_size=max_size, min_size=min_size, color=color)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 5 ----------

def task_svrt_5(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    max_size: float = 0.3,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #5 – Clase 1: dos pares de figuras idénticas (hasta traslación)
            - Clase 0: cuatro figuras diferentes
    """

    # Clase 1: dos pares de figuras idénticas (hasta traslación)
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes = [shape1.clone(), shape1.clone(), shape2.clone(), shape2.clone()]

    sample_pos = decorate_shapes(shapes, max_size=max_size, min_size=min_size, color=color)

    # Clase 0: cuatro figuras diferentes
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(4)]
    sample_neg = decorate_shapes(shapes, max_size=max_size, min_size=min_size, color=color)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 6 ----------

def task_svrt_6(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #6 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 7 ----------

def task_svrt_7(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    max_size: float = 0.3,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #7 – Clase 1: tres pares de figuras idénticas (hasta traslación)
            - Clase 0: dos tríos de figuras identicas (hasta traslación)
    """
    # Clase 1: tres pares de figuras idénticas (hasta traslación)
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape3 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes = [shape1.clone(), shape1.clone(), shape2.clone(), shape2.clone(), shape3.clone(), shape3.clone()]
    sample_pos = decorate_shapes(shapes, max_size=max_size, min_size=min_size, color=color)
    
    # Clase 0: dos tríos de figuras idénticas (hasta traslación)
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes = [
        shape1.clone(), shape1.clone(), shape1.clone(),
        shape2.clone(), shape2.clone(), shape2.clone()
    ]
    sample_neg = decorate_shapes(shapes, max_size=max_size, min_size=min_size, color=color)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 8 ----------

def task_svrt_8(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #8 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 9 ----------

def task_svrt_9(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #9 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 10 ----------

def task_svrt_10(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #10 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 11 ----------
# Falta hacer que los objetos no se toquen en caso negativo.
def task_svrt_11(
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
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #11 – Devuelve (sample_neg, sample_pos).
    2 objetos distinto tamaño, clase 1 en contacto.

    """

    def normalize_scene(xy, size, margin=0.05):
        # xy: (n, 1, 2), size: (n, 1)
        bb_min = (xy - size[..., None] / 2).min(axis=(0, 1))
        bb_max = (xy + size[..., None] / 2).max(axis=(0, 1))
        scale = (1 - 2 * margin) / (bb_max - bb_min).max()
        offset = 0.5 - ((bb_min + bb_max) / 2) * scale
        return xy * scale + offset, size * scale

    # --------- Tamaños ---------
    size_vals = np.random.rand(2) * (max_size - min_size) + min_size
    size_vals *= shrink_factor
    size = size_vals[:, None]

    # --------- Positivo ---------
    shape11 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape12 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)

    xy_contact, _ = sample_contact_many([shape11, shape12], size_vals)

    xy_pos, size = normalize_scene(xy_contact, size)
    shapes_pos = [shape11, shape12]
    if color:
        colors_pos = [c.flatten() for c in sample_random_colors(2)]
    else:
        colors_pos = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(2)]
    sample_pos = (xy_pos, size, shapes_pos, colors_pos)

    # --------- Negativo ---------
    shape21 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape22 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)

    sample_neg = decorate_shapes([shape21, shape22], max_size=max_size, min_size=min_size, color=color)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 12 ----------

def task_svrt_12(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #13 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 13 ----------

def task_svrt_13(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #13 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 14 ----------

def task_svrt_14(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #14 – Devuelve sample_neg, sample_pos
    """
    def normalize_scene(xy, size, margin=0.05):
        # xy: (n, 1, 2), size: (n, 1)
        bb_min = (xy - size[..., None] / 2).min(axis=(0, 1))
        bb_max = (xy + size[..., None] / 2).max(axis=(0, 1))
        scale = (1 - 2 * margin) / (bb_max - bb_min).max()
        offset = 0.5 - ((bb_min + bb_max) / 2) * scale
        return xy * scale + offset, size * scale

    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)

    # --------- Positivo ---------
    xy_pos, size_pos, shapes_pos, color_pos = decorate_shapes(
        [shape1.clone(), shape1.clone(), shape1.clone()],
        max_size=max_size, min_size=min_size, color=color, align=True
    )
    xy_pos, size_pos = normalize_scene(xy_pos, size_pos)
    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)
    # --------- Negativo ---------
    sample_neg = decorate_shapes([shape1.clone(), shape1.clone(), shape1.clone()], max_size=max_size, min_size=min_size, color=color)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 15 ----------

def task_svrt_15(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #15 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 16 ----------

def task_svrt_16(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #16 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 17 ----------

def task_svrt_17(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #17 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 18 ----------

def task_svrt_18(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #18 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 19 ----------

def task_svrt_19(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #19 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 20 ----------

def task_svrt_20(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #20 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 21 ----------

def task_svrt_21(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #21 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 22 ----------

def task_svrt_22(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #22 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SVRT 23 ----------

def task_svrt_23(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #23 – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Registro de tareas ----------
# Tareas SVRT
TASKS_SVRT = [
    ["task_svrt_1", task_svrt_1],
    ["task_svrt_2", task_svrt_2],
    ["task_svrt_3", task_svrt_3],
    ["task_svrt_4", task_svrt_4],
    ["task_svrt_5", task_svrt_5],
    ["task_svrt_6", task_svrt_6],
    ["task_svrt_7", task_svrt_7],
    ["task_svrt_8", task_svrt_8],
    ["task_svrt_9", task_svrt_9],
    ["task_svrt_10", task_svrt_10],
    ["task_svrt_11", task_svrt_11],
    ["task_svrt_12", task_svrt_12],
    ["task_svrt_13", task_svrt_13],
    ["task_svrt_14", task_svrt_14],
    ["task_svrt_15", task_svrt_15],
    ["task_svrt_16", task_svrt_16],
    ["task_svrt_17", task_svrt_17],
    ["task_svrt_18", task_svrt_18],
    ["task_svrt_19", task_svrt_19],
    ["task_svrt_20", task_svrt_20],
    ["task_svrt_21", task_svrt_21],
    ["task_svrt_22", task_svrt_22],
    ["task_svrt_23", task_svrt_23],
]
