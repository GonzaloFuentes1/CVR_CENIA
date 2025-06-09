import numpy as np
from matplotlib.path import Path
from shapely import Polygon

from data_generation.shape import Shape
from data_generation.utils import (
    sample_contact_many,
    sample_position_inside_1,
    sample_positions_bb,
    sample_positions_align,
    sample_random_colors,
    sample_positions_symmetric_pairs,
    sample_positions_circle
)

from data_generation.utils import (check_square, sample_contact_many,
                                   sample_position_inside_1,
                                   sample_positions_align, sample_positions_bb,
                                   sample_positions_equidist,
                                   sample_positions_square,
                                   sample_random_colors,
                                   sample_positions_symmetric_pairs,
                                   sample_positions_circle)

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

def decorate_shapes(shapes, max_size=0.4, min_size=None, size=None, rotate=False, color=False, flip=False, align=False, mirror=False, circle=False, middle=0):
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
    if middle == 1:
        size[1] = max_size
    if middle == 2:
        k = np.random.rand()
        if k >= 0.5:
            size[0] = max_size
        else:
            size[2] = max_size
    # Posiciones sin superposición usando bounding boxes
    size_batch = size[None, ...]  # shape (1, n, 1)
    if align:
        sizes = size_batch[0]
        if sizes.sum() >= 1:
            size_batch *= (1 / (sizes.sum()*0.8))  # Normalizar para que sumen 1
        xy_vals = sample_positions_align(size_batch)[0]  # shape (n, 2)

    elif mirror:
        xy_vals = sample_positions_symmetric_pairs(size_batch[0])

    elif circle:
        xy_vals = sample_positions_circle(size_batch[0])  # shape (n, 2)

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
    symm_rotate: bool = False,
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
    max_size: float = 0.9,
    min_size: float = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon',
    max_tries: int = 30
):
    """
    SVRT #2 – Devuelve (sample_neg, sample_pos).
    sample_pos: inner centrado dentro del outer (clase 1)
    sample_neg: inner desplazado hacia el borde (clase 0)
    """
    if min_size is None:
        raise ValueError("min_size debe estar definido.")

    size_outer = max_size
    size_inner = min_size * size_outer

    # === Crear shapes ===
    outer = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)
    inner = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)
    outer.scale(size_outer)
    inner.scale(size_inner)

    # === Clase Positiva: centrado ===
    for _ in range(max_tries):
        xy_outer = np.random.rand(2) * (1 - size_outer) + size_outer / 2
        # Centro del inner coincide con centro del outer
        xy_inner = xy_outer
        contour_inner = inner.get_contour() + xy_inner
        if ((0 <= contour_inner).all() and (contour_inner <= 1).all()):
            break
    else:
        raise RuntimeError("No se pudo generar clase positiva centrada.")

    xy_pos = np.stack([xy_outer, xy_inner])[:, None, :]
    size_pos = np.array([[1.0], [1.0]])
    shapes_pos = [[outer], [inner]]
    color_pos = sample_random_colors(2) if color else [np.zeros((1, 3), dtype=np.float32)] * 2
    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)

    # === Clase Negativa: desplazado cerca del borde ===
    for _ in range(max_tries):
        xy_outer = np.random.rand(2) * (1 - size_outer) + size_outer / 2
        # Samplear posición alejada del centro
        rel_offset = np.random.rand(2) - 0.5
        rel_offset /= np.linalg.norm(rel_offset) + 1e-6
        offset = rel_offset * (size_outer / 2 - size_inner / 1.1)
        xy_inner = xy_outer + offset
        contour_inner = inner.get_contour() + xy_inner
        if ((0 <= contour_inner).all() and (contour_inner <= 1).all()):
            break
    else:
        raise RuntimeError("No se pudo generar clase negativa desplazada.")

    xy_neg = np.stack([xy_outer, xy_inner])[:, None, :]
    size_neg = np.array([[1.0], [1.0]])
    shapes_neg = [[outer.clone()], [inner.clone()]]
    color_neg = sample_random_colors(2) if color else [np.zeros((1, 3), dtype=np.float32)] * 2
    sample_neg = (xy_neg, size_neg, shapes_neg, color_neg)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 3 ----------

def task_svrt_3(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = False,
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
    size = size_vals[:, None]  # (4, 1)

    # --------- Positivo: 3 en contacto + 1 separada ---------
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(4)]
    contact3, solo = shapes[:3], shapes[3]

    xy_contact3, _ = sample_contact_many(contact3, size_vals[:3])
    xy_lone = sample_positions_bb(size[None, 3:], n_sample_min=1)[0, 0]

    xy_pos = np.concatenate([xy_contact3, xy_lone[None]], axis=0)[:, None, :]
    xy_pos, size_pos = normalize_scene(xy_pos, size.copy())
    shapes_pos = [[s] for s in contact3 + [solo]]
    colors_pos = sample_random_colors(4) if color else [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3)] * 4
    sample_pos = (xy_pos, size_pos, shapes_pos, colors_pos)

    # --------- Negativo: 2 pares en contacto, separados ---------
    shapes1 = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(2)]
    shapes2 = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(2)]

    xy1_local, bb1 = sample_contact_many(shapes1, size_vals[:2])
    center1 = xy1_local.mean(axis=0)
    xy1_local -= center1
    shapes1 = [[s] for s in shapes1]

    xy2_local, bb2 = sample_contact_many(shapes2, size_vals[2:])
    center2 = xy2_local.mean(axis=0)
    xy2_local -= center2
    shapes2 = [[s] for s in shapes2]

    bbox1 = np.max(xy1_local + size_vals[:2, None] / 2, axis=0) - np.min(xy1_local - size_vals[:2, None] / 2, axis=0)
    bbox2 = np.max(xy2_local + size_vals[2:, None] / 2, axis=0) - np.min(xy2_local - size_vals[2:, None] / 2, axis=0)
    bbox_sizes = np.stack([bbox1.max(), bbox2.max()])[:, None][None, :, :]  # shape (1, 2, 1)

    group_centers = sample_positions_bb(bbox_sizes, n_sample_min=1)[0]  # shape (2, 2)

    xy1 = xy1_local + group_centers[0]
    xy2 = xy2_local + group_centers[1]
    xy_neg = np.concatenate([xy1, xy2], axis=0)[:, None, :]
    shapes_neg = shapes1 + shapes2
    colors_neg = sample_random_colors(4) if color else [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3)] * 4
    sample_neg = (xy_neg, size.copy(), shapes_neg, colors_neg)

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
    max_size: float = 0.9,
    min_size: float = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon',
    max_tries: int = 1000,
):
    """
    SVRT #4 – Devuelve (sample_neg, sample_pos).
    sample_pos: figura chica completamente dentro de la grande (clase 1)
    sample_neg: mismas figuras separadas, sin inclusión ni solapamiento (clase 0)
    """
    if min_size is None:
        raise ValueError("min_size debe estar definido.")

    border_margin = 0.01
    size_outer = max_size
    size_inner = min_size * size_outer

    # ==== CLASE POSITIVA ====
    if shape_mode == 'rigid':
        outer_sides = n_sides
        inner_sides = np.random.choice([
            k for k in range(poly_min_sides, poly_max_sides) if k != n_sides
        ])
    else:
        outer_sides = inner_sides = n_sides

    outer = create_shape(shape_mode, rigid_type, radius, hole_radius, outer_sides, fourier_terms)
    inner = create_shape(shape_mode, rigid_type, radius, hole_radius, inner_sides, fourier_terms)
    outer.scale(size_outer)
    inner.scale(size_inner)

    for _ in range(max_tries):
        xy_outer = np.random.rand(2) * (1 - size_outer - 2 * border_margin) + size_outer / 2 + border_margin
        xy_inner_rel = sample_position_inside_1(outer, inner, scale=1 - (size_inner / size_outer))
        if len(xy_inner_rel) == 0:
            continue
        xy_inner = xy_inner_rel[0] + xy_outer

        contour_outer = outer.get_contour() + xy_outer
        contour_inner = inner.get_contour() + xy_inner

        path_outer = Path(contour_outer)
        if ((0 <= contour_inner).all() and (contour_inner <= 1).all()
            and path_outer.contains_points(contour_inner).all()):
            break
    else:
        raise RuntimeError("No se pudo generar clase positiva con inclusión real.")

    xy_pos = np.stack([xy_outer, xy_inner])[:, None, :]
    size_pos = np.array([[1.0], [1.0]])
    shapes_pos = [[outer], [inner]]

    if color:
        color_pos = sample_random_colors(2)
        color_pos = [color_pos[i:i+1] for i in range(2)]
    else:
        color_pos = [np.zeros((1, 3), dtype=np.float32) for _ in range(2)]

    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)

    # ==== CLASE NEGATIVA ====
    shape_a = outer
    shape_b = inner

    for _ in range(max_tries):
        xy_a = np.random.rand(2) * (1 - size_outer - 2 * border_margin) + size_outer / 2 + border_margin
        xy_b = np.random.rand(2) * (1 - size_inner - 2 * border_margin) + size_inner / 2 + border_margin

        contour_a = shape_a.get_contour() + xy_a
        contour_b = shape_b.get_contour() + xy_b

        path_a = Path(contour_a)
        path_b = Path(contour_b)

        a_in_b = path_b.contains_points(contour_a).any()
        b_in_a = path_a.contains_points(contour_b).any()

        if not (a_in_b or b_in_a):
            break
    else:
        raise RuntimeError("No se pudo generar clase negativa sin inclusión.")

    xy_neg = np.stack([xy_a, xy_b])[:, None, :]
    size_neg = np.array([[1.0], [1.0]])
    shapes_neg = [[shape_a], [shape_b]]

    if color:
        color_neg = sample_random_colors(2)
        color_neg = [color_neg[i:i+1] for i in range(2)]
    else:
        color_neg = [np.zeros((1, 3), dtype=np.float32) for _ in range(2)]

    sample_neg = (xy_neg, size_neg, shapes_neg, color_neg)

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
    SVRT #6 – Clase 1: dos pares de figuras idénticas, distancias entre figuras idénticas son iguales en ambos pares
            - Clase 0: dos pares de figuras idénticas
    """

    # Clase 0:
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes = [shape1.clone(), shape1.clone(), shape2.clone(), shape2.clone()]

    equal_dist_flag = True

    while equal_dist_flag:    
        sample_neg = decorate_shapes(shapes, max_size=max_size * 2 * 0.33, min_size=min_size, color=color) 
        xy = sample_neg[0][:, 0, :]  # shape (4, 2)
        # Comprobar si las distancias entre figuras idénticas son iguales
        dist1 = np.linalg.norm(xy[0] - xy[1])  
        dist2 = np.linalg.norm(xy[2] - xy[3])
        if np.abs(dist1 - dist2) > 0.01:
            equal_dist_flag = False


    # Clase 1:
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes = [shape1.clone(), shape1.clone(), shape2.clone(), shape2.clone()]

    size = np.full((4, 1), fill_value=max_size * 0.33)
    xy = sample_positions_equidist(size)
    xy = xy[:, None, :]  # shape (4, 1, 2)
    if color:
        colors = sample_random_colors(4)
        colors = [colors[i:i+1] for i in range(4)]
    else:
        colors = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(4)]
    shapes_wrapped = [[s] for s in shapes]
    sample_pos = (xy, size, shapes_wrapped, colors)

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
    SVRT #8 – Devuelve sample_neg, sample_pos
    Clase 0 (sample_neg): Si la figura grande contiene la pequeña, son diferentes. Si no se contienen, son iguales (hasta escalamiento y traslación).
    Clase 1 (sample_pos): La figura grande contiene a la pequeña, que es igual a la grande (hasta escalamiento y traslación).
    """


    # Clase 1: Figura grande con figura idéntica escalada dentro
    outer = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    inner = outer.clone()

    size_outer = max_size * 0.9
    size_inner = size_outer * 0.3

    # posición global del centro del outer
    xy_outer = np.random.rand(2) * (1 - size_outer) + size_outer / 2


    contour_outer = outer.get_contour() * size_outer + xy_outer
    contour_inner = inner.get_contour() * size_inner

    max_attempts = 10

    done_flag = False
    for _ in range(max_attempts):
        xy_inner_rel = sample_position_inside_1(outer, inner, scale=size_inner / size_outer)
        if len(xy_inner_rel) > 0:
            for pos in xy_inner_rel:
                xy_inner = pos * size_outer + xy_outer
                contour_inner_temp = contour_inner + xy_inner
                if Polygon(contour_outer).contains_properly(Polygon(contour_inner_temp)):
                    done_flag = True
                    break
        if done_flag:
            break
        else:
            outer = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
            inner = outer.clone()
    if not done_flag:
        raise RuntimeError("No se pudo encontrar una posición válida para la clase positiva.")

    xy_pos = np.stack([xy_outer, xy_inner])[:, None, :]
    size_pos = np.array([[size_outer], [size_inner]])
    shapes_pos = [[outer], [inner]]

    if color:
        color_pos = sample_random_colors(2)
        color_pos = [color_pos[i:i+1] for i in range(2)]
    else:
        color_pos = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(2)]

    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)

    # Clase 0:
    # Coinflip para determinar subclase

    if np.random.rand() > 0.5:
        # Subclase 0: Figura grande con figura pequeña dentro, pero diferentes
        outer = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        inner = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)

        size_outer = max_size * 0.9
        size_inner = size_outer * 0.3

        # posición global del centro del outer
        xy_outer = np.random.rand(2) * (1 - size_outer) + size_outer / 2

        contour_outer = outer.get_contour()*size_outer + xy_outer
        contour_inner = inner.get_contour()*size_inner

        max_attempts = 10
        done_flag = False
        for _ in range(max_attempts):
            xy_inner_rel = sample_position_inside_1(outer, inner, scale=size_inner / size_outer)
            if len(xy_inner_rel) > 0:
                for pos in xy_inner_rel:
                    xy_inner = pos * size_outer + xy_outer
                    contour_inner_temp = contour_inner + xy_inner
                    if Polygon(contour_outer).contains_properly(Polygon(contour_inner_temp)):
                        done_flag = True
                        break
            if done_flag:
                break
            else:
                outer = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
                inner = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        if not done_flag:
            raise RuntimeError("No se pudo encontrar una posición válida para la clase negativa.")

        xy_neg = np.stack([xy_outer, xy_inner])[:, None, :]
        size_neg = np.array([[size_outer], [size_inner]])
        shapes_neg = [[outer], [inner]]
        if color:
            color_neg = sample_random_colors(2)
            color_neg = [color_neg[i:i+1] for i in range(2)]
        else:
            color_neg = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(2)]
        sample_neg = (xy_neg, size_neg, shapes_neg, color_neg)
    else:
        # Subclase 1: Figuras idénticas hasta traslación y escalamiento, sin contenerse
        shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape2 = shape1.clone()

        sample_neg = decorate_shapes([shape1, shape2], max_size=max_size, min_size=min_size, color=color, size = True)

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
    SVRT #9 3 figuras identicas alineadas con una mayor que las otras 2.
            – Clase 1: La figura de mayor tamaño se encuentra entre las de menor tamaño
            - Clase 0: La figura de mayor tamaño no se encuentra entra las de menor tamaño
    """
    def normalize_scene(xy, size, margin=0.05):
        # xy: (n, 1, 2), size: (n, 1)
        bb_min = (xy - size[..., None] / 2).min(axis=(0, 1))
        bb_max = (xy + size[..., None] / 2).max(axis=(0, 1))
        scale = (1 - 2 * margin) / (bb_max - bb_min).max()
        offset = 0.5 - ((bb_min + bb_max) / 2) * scale
        return xy * scale + offset, size * scale

    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)

    # --------- Positivo ---------
    xy_pos, size_pos, shapes_pos, color_pos = decorate_shapes(
        [shape1.clone(), shape1.clone(), shape1.clone()],
        max_size=max_size, min_size=min_size, color=color, align=True, middle=1
    )
    xy_pos, size_pos = normalize_scene(xy_pos, size_pos)
    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)
    # --------- Negativo ---------
    xy_neg, size_neg, shapes_neg, color_neg = decorate_shapes(
        [shape2.clone(), shape2.clone(), shape2.clone()],
        max_size=max_size, min_size=min_size, color=color, align=True, middle=2
    )
    xy_neg, size_neg = normalize_scene(xy_neg, size_neg)
    sample_neg = (xy_neg, size_neg, shapes_neg, color_neg)

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
    SVRT #10 – Devuelve sample_neg, sample_pos
    Clase 0 (sample_neg): Cuatro figuras idénticas hasta traslación
    Clase 1 (sample_pos): Cuatro figuras idénticas hasta traslación, sus centros forman un cuadrado
    """

    size = np.full((4, 1), fill_value=max_size / 3)

    shape = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes = [shape.clone() for _ in range(4)]

    # Sample neg: cuatro figuras idénticas hasta traslación
    square_flag = True
    while square_flag:
        sample_neg = decorate_shapes(shapes, max_size=max_size * 2/3, min_size=min_size, color=color)
        xy_neg = sample_neg[0][:, 0, :]  # shape (4, 2)
        # Comprobar si los centros forman un cuadrado
        square_flag = check_square(xy_neg)

    # Sample pos: cuatro figuras idénticas hasta traslación, sus centros forman un cuadrado
    shape = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    xy = sample_positions_square(size)

    if color:
        colors_pos = sample_random_colors(4)
        colors_pos = [colors_pos[i:i+1] for i in range(4)]
    else:
        colors_pos = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(4)]
    
    shapes_pos = [[shape.clone()] for _ in range(4)]
    xy_pos = xy[:, None, :]  # shape (4, 1, 2)
    sample_pos = (xy_pos, size, shapes_pos, colors_pos)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 11 ----------
# Ver tema de los sizes, como hacer para tener figuras de varios tamaños de manera mas pronunciada, estandarizarlo?
def task_svrt_11(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    max_size: float = 0.4,
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

    sample_neg = decorate_shapes([shape21, shape22], max_size=max_size, min_size=min_size, color=color, size=True)

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
    SVRT #12 – Clase 1: dos figuras pequeñas, una grande. Las figuras pequeñas están más cerca entre sí que de la figura grande.
            - Clase 0: dos figuras pequeñas, una grande. Alguna de las figuras pequeñas está más cerca de la figura grande que de la otra figura pequeña.
    """

    # Clase 1: dos figuras pequeñas, una grande. Las figuras pequeñas están más cerca entre sí que de la figura grande.
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(3)]
    size = np.array([[max_size * 0.6], [max_size * 0.2], [max_size * 0.2]])

    dist_flag = True
    while dist_flag:
        xy = sample_positions_bb(size[None, ...])[0]
        # Comprobar distancias
        dist1 = np.linalg.norm(xy[0] - xy[1])
        dist2 = np.linalg.norm(xy[0] - xy[2])
        dist3 = np.linalg.norm(xy[1] - xy[2])
        if dist1 - 1e-2 > dist3 and dist2 - 1e-2 > dist3:
            dist_flag = False

    xy = xy[:, None, :]  # shape (3, 1, 2)
    if color:
        colors = sample_random_colors(3)
        colors = [colors[i:i+1] for i in range(3)]
    else:
        colors = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(3)]
    
    shapes_wrapped = [[s] for s in shapes]
    sample_pos = (xy, size, shapes_wrapped, colors)

    # Clase 0: dos figuras pequeñas, una grande. Alguna de las figuras pequeñas está más cerca de la figura grande que de la otra figura pequeña.

    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(3)]
    size = np.array([[max_size * 0.6], [max_size * 0.2], [max_size * 0.2]])

    dist_flag = True
    while dist_flag:
        xy = sample_positions_bb(size[None, ...])[0]
        # Comprobar distancias
        dist1 = np.linalg.norm(xy[0] - xy[1])
        dist2 = np.linalg.norm(xy[0] - xy[2])
        dist3 = np.linalg.norm(xy[1] - xy[2])
        if dist1 + 1e-3 < dist3 or dist2 + 1e-3 < dist3:
            dist_flag = False

    xy = xy[:, None, :]  # shape (3, 1, 2)
    if color:
        colors = sample_random_colors(3)
        colors = [colors[i:i+1] for i in range(3)]
    else:
        colors = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(3)]
    shapes_wrapped = [[s] for s in shapes]
    sample_neg = (xy, size, shapes_wrapped, colors)

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
    delta_min: float = 0.01,
    delta_max: float = 0.3,
    color: bool = False,
    rigid_type: str = 'polygon',
    max_tries: int = 30,
):
    def center_scene_without_scaling(xy, size):
        bb_min = (xy - size[..., None] / 2).min(axis=(0, 1))
        bb_max = (xy + size[..., None] / 2).max(axis=(0, 1))
        bb_center = (bb_min + bb_max) / 2
        canvas_center = np.array([0.5, 0.5])
        delta = canvas_center - bb_center
        return xy + delta

    if min_size is None:
        raise ValueError("min_size debe estar definido.")

    # === 1. Crear una figura grande y una figura chica ===
    shape_big = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape_big.scale(max_size)

    shape_small = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape_small.scale(min_size)

    # === 2. Ubicar las dos figuras grandes sin solaparse ===
    for _ in range(max_tries):
        xy_big, _, _, _ = decorate_shapes(
            [shape_big, shape_big],
            max_size=max_size,
            min_size=min_size,
            size=False,
            rotate=False,
            flip=False,
            color=False,
            align=True
        )
        dist = np.linalg.norm(xy_big[0, 0] - xy_big[1, 0])
        if dist > max_size:
            break
    else:
        raise RuntimeError("No se pudo ubicar figuras grandes sin solapamiento.")

    pos_big1 = xy_big[0, 0]
    pos_big2 = xy_big[1, 0]

    # === 3. Clase POSITIVA ===
    for _ in range(max_tries):
        delta = np.random.uniform(delta_min, delta_max, size=2)
        pos_small1 = pos_big1 + delta
        pos_small2 = pos_big2 + delta

        dist_11 = np.linalg.norm(pos_small1 - pos_big1)
        dist_22 = np.linalg.norm(pos_small2 - pos_big2)
        dist_12 = np.linalg.norm(pos_small1 - pos_small2)

        if dist_11 > (max_size + min_size) / 2 and dist_22 > (max_size + min_size) / 2 and dist_12 > min_size:
            break
    else:
        raise RuntimeError("No se pudo ubicar las figuras chicas sin solapamiento (positiva).")

    xy_pos = np.stack([pos_big1, pos_small1, pos_big2, pos_small2])[:, None, :]
    xy_pos = center_scene_without_scaling(xy_pos, np.ones((4, 1)))
    size_pos = np.ones((4, 1))
    shapes_pos = [[shape_big], [shape_small], [shape_big], [shape_small]]

    if color:
        color_pos = sample_random_colors(4)
        color_pos = [color_pos[i:i+1] for i in range(4)]
    else:
        color_pos = [np.zeros((1, 3), dtype=np.float32) for _ in range(4)]

    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)

    # === 4. Clase NEGATIVA: delta + ruido angular ===
    for _ in range(max_tries):
        delta_base = np.random.uniform(delta_min, delta_max, size=2)
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, delta_max)
        noise = r * np.array([np.cos(angle), np.sin(angle)])

        delta1 = delta_base
        delta2 = delta_base + noise

        pos_small1 = pos_big1 + delta1
        pos_small2 = pos_big2 + delta2

        d11 = np.linalg.norm(pos_small1 - pos_big1)
        d22 = np.linalg.norm(pos_small2 - pos_big2)
        d12 = np.linalg.norm(pos_small1 - pos_small2)

        if d11 > (max_size + min_size) / 2 and d22 > (max_size + min_size) / 2 and d12 > min_size:
            break
    else:
        raise RuntimeError("No se pudo ubicar composiciones negativas válidas.")

    xy_neg = np.stack([pos_big1, pos_small1, pos_big2, pos_small2])[:, None, :]
    xy_neg = center_scene_without_scaling(xy_neg, np.ones((4, 1)))
    size_neg = np.ones((4, 1))
    shapes_neg = [[shape_big], [shape_small], [shape_big], [shape_small]]

    if color:
        color_neg = sample_random_colors(4)
        color_neg = [color_neg[i:i+1] for i in range(4)]
    else:
        color_neg = [np.zeros((1, 3), dtype=np.float32) for _ in range(4)]

    sample_neg = (xy_neg, size_neg, shapes_neg, color_neg)

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
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)

    # --------- Positivo ---------
    xy_pos, size_pos, shapes_pos, color_pos = decorate_shapes(
        [shape1.clone(), shape1.clone(), shape1.clone()],
        max_size=max_size, min_size=min_size, color=color, align=True
    )
    xy_pos, size_pos = normalize_scene(xy_pos, size_pos)
    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)
    # --------- Negativo ---------
    xy_neg, size_neg, shapes_neg, color_neg = decorate_shapes(
        [shape2.clone(), shape2.clone(), shape2.clone()],
        max_size=max_size, min_size=min_size, color=color, align=False,
    )
    xy_neg, size_neg = normalize_scene(xy_neg, size_neg)
    sample_neg = (xy_neg, size_neg, shapes_neg, color_neg)

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
    SVRT #15 – Devuelve sample_neg, sample_pos
    Clase 0 (sample_neg): Cuatro figuras distintas, sus centros forman un cuadrado
    Clase 1 (sample_pos): Cuatro figuras idénticas hasta traslación, sus centros forman un cuadrado
    """

    size = np.full((4, 1), fill_value=max_size / 3)

    # Sample neg: cuatro figuras distintas, sus centros forman un cuadrado
    
    shapes = [
        create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(4)
    ]
    xy = sample_positions_square(size)
    if color:
        colors_neg = sample_random_colors(4)
        colors_neg = [colors_neg[i:i+1] for i in range(4)]
    else:
        colors_neg = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(4)]
    shapes_neg = [[s] for s in shapes]
    xy_neg = xy[:, None, :]  # shape (4, 1, 2)
    sample_neg = (xy_neg, size, shapes_neg, colors_neg)

    # Sample pos: cuatro figuras idénticas hasta traslación, sus centros forman un cuadrado

    shape = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes = [[shape.clone()] for _ in range(4)]
    xy_pos = sample_positions_square(size)
    if color:
        colors_pos = sample_random_colors(4)
        colors_pos = [colors_pos[i:i+1] for i in range(4)]
    else:
        colors_pos = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(4)]
    xy_pos = xy_pos[:, None, :]  # shape (4, 1, 2)
    sample_pos = (xy_pos, size, shapes, colors_pos)

    return sample_neg, sample_pos


# ---------- Tarea SVRT 16 ----------

def task_svrt_16(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    max_size: float = 0.2,
    min_size: float | None = 0.13,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #16 – Devuelve:
    - Clase 0: seis figuras idénticas en posiciones simétricas respecto al eje vertical (no reflejadas).
    - Clase 1: mismas posiciones, pero las tres figuras de la derecha son el reflejo especular de las de la izquierda.
    """
    n_pairs = 3
    base_shape = create_shape(
        shape_mode, rigid_type, radius, hole_radius,
        n_sides, fourier_terms, symm_rotate
    )

    # --- Clase 1 ---
    shapes_pos = []
    for i in range(2 * n_pairs):
        s = base_shape.clone()
        # Si está a la derecha (pares impares, i=1,3,5): flip respecto al eje vertical
        if i % 2 == 1:
            s.flip()
        shapes_pos.append(s)
    sample_pos = decorate_shapes(
        shapes_pos,
        max_size=max_size,
        min_size=min_size,
        color=color,
        mirror=True
    )
    
    # --- Clase 0 ---
    shapes_neg = [base_shape.clone() for _ in range(2 * n_pairs)]
    sample_neg = decorate_shapes(
        shapes_neg,
        max_size=max_size,
        min_size=min_size,
        color=color,
        mirror=True
    )

    return sample_neg, sample_pos



# ---------- Tarea SVRT 17 ----------

def task_svrt_17(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    max_size: float = 0.13,
    min_size: float | None = 0.09,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SVRT #17 – Devuelve:
    - Clase 1: 4 figuras, 3 idénticas y 1 diferente, todas del mismo tamaño. Odd se ubica aleatoriamente en círculo de radio 0.3.
    - Clase 0: 4 figuras, 3 idénticas y 1 diferente, mismas figuras y tamaños. Posiciones aleatorias.
    """

    # No sé si forzar el tamaño de las figuras, pero lo dejo como parámetro
    min_size = 0.05
    max_size = 0.3

    base_shape = create_shape(
        shape_mode, rigid_type, radius, hole_radius,
        n_sides, fourier_terms, symm_rotate
    )
    odd_shape = create_shape(
        shape_mode, rigid_type, radius, hole_radius,
        n_sides, fourier_terms, symm_rotate
    )

    # --- Clase 1 ---
    shapes_pos = [base_shape.clone() for _ in range(3)] + [odd_shape.clone()]
    sample_pos = decorate_shapes(
        shapes_pos,
        max_size=max_size,
        min_size=min_size,
        color=color,
        circle=True
    )

    # --- Clase 0 ---
    shapes_neg = [base_shape.clone() for _ in range(3) ] + [odd_shape.clone()]    
    sample_neg = decorate_shapes(
        shapes_neg,
        max_size=max_size,
        min_size=min_size,
        color=color
    )

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
    SVRT #18 – Devuelve:
    - Clase 0: seis figuras idénticas en posiciones simétricas respecto al eje vertical.
    - Clase 1: seis figuras idénticas posicionadas aleatoriamente.
    """
    # --- Clase 1 ---
    shape = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes_pos = [shape.clone() for _ in range(6)]
    sample_pos = decorate_shapes(shapes_pos, max_size=max_size, min_size=min_size, color=color, mirror=True)
    
    # --- Clase 0 ---
    # Opcional: pueden ser iguales o diferentes, pero lo clave es que NO usen mirror=True
    shapes_neg = [shape.clone() for _ in range(6)]
    sample_neg = decorate_shapes(shapes_neg, max_size=max_size, min_size=min_size, color=color, mirror=False)
    
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
    SVRT #19 – Devuelve:
    - Clase 1: dos figuras iguales, solo que una está escalada.
    - Clase 0: dos figuras diferentes.
    """

    # --- Clase 1 ---
    shape_pos_1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    scale_factor = np.random.uniform(0.3, 1.5) # Valores arbitrarios
    shape_pos_2 = shape_pos_1.clone()
    shape_pos_2.scale(scale_factor)
    shapes_pos = [shape_pos_1, shape_pos_2]
    sample_pos = decorate_shapes(
        shapes_pos,
        max_size=max_size,
        min_size=min_size,
        color=color
    )

    # --- Clase 0 ---
    shape_neg_1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape_neg_2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes_neg = [shape_neg_1, shape_neg_2]    
    sample_neg = decorate_shapes(
        shapes_neg,
        max_size=max_size,
        min_size=min_size,
        color=color
    )

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
    SVRT #20 – Devuelve sample_neg, sample_pos
    Clase 0 (sample_neg): Dos figuras
    Clase 1 (sample_pos): Dos figuras, una es reflexión de la otra con respecto a la bisectriz perpendicular a la línea que une sus centros
    """

    # Clase 0:
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)

    sample_neg = decorate_shapes([shape1, shape2], max_size=max_size, min_size=min_size, color=color, size=True)

    # Clase 1:
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)
    shape2 = shape1.clone()

    size = np.array([[max_size * 0.5], [max_size * 0.5]])
    size_aux = size * np.sqrt(2)
    xy = sample_positions_bb((size_aux[None, ...]))[0]

    # Calcular el ángulo de rotación para que shape2 sea la reflexión de shape1
    angle = np.arctan2(xy[1, 1] - xy[0, 1], xy[1, 0] - xy[0, 0])
    shape2.flip()
    shape1.rotate(angle)
    shape2.rotate(angle)
    xy = xy[:, None, :]  # shape (2, 1, 2)
    if color:
        colors = sample_random_colors(2)
        colors = [colors[i:i+1] for i in range(2)]
    else:
        colors = [np.array([0, 0, 0], dtype=np.float32).reshape(1, 3) for _ in range(2)]
    shapes = [[shape1], [shape2]]
    sample_pos = (xy, size, shapes, colors)

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
    SVRT #10 – Devuelve sample_neg, sample_pos
    Clase 0 (sample_neg): Dos figuras
    Clase 1 (sample_pos): Dos figuras idénticas hasta rotación, traslación, y escalamiento
    """

    # Clase 0:
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)
    sample_neg = decorate_shapes([shape1, shape2], max_size=max_size, min_size=min_size, color=color, size=True)

    # Clase 1:
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)
    shape2 = shape1.clone()
    sample_pos = decorate_shapes([shape1, shape2], max_size=max_size, min_size=min_size, color=color, size=True, rotate=True)

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
    SVRT #22 3 figuras alineadas
            – Clase 1: Todas las figuras son iguales
            - Clase 0: Las figuras no son iguales
    """
    def normalize_scene(xy, size, margin=0.05):
        # xy: (n, 1, 2), size: (n, 1)
        bb_min = (xy - size[..., None] / 2).min(axis=(0, 1))
        bb_max = (xy + size[..., None] / 2).max(axis=(0, 1))
        scale = (1 - 2 * margin) / (bb_max - bb_min).max()
        offset = 0.5 - ((bb_min + bb_max) / 2) * scale
        return xy * scale + offset, size * scale

    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape3 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape4 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)

    # --------- Positivo ---------
    xy_pos, size_pos, shapes_pos, color_pos = decorate_shapes(
        [shape1.clone(), shape1.clone(), shape1.clone()],
        max_size=max_size, min_size=min_size, color=color, align=True
    )
    xy_pos, size_pos = normalize_scene(xy_pos, size_pos)
    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)
    # --------- Negativo ---------
    xy_neg, size_neg, shapes_neg, color_neg = decorate_shapes(
        [shape2.clone(), shape3.clone(), shape4.clone()],
        max_size=max_size, min_size=min_size, color=color, align=True,
    )
    xy_neg, size_neg = normalize_scene(xy_neg, size_neg)
    sample_neg = (xy_neg, size_neg, shapes_neg, color_neg)
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


# ---------- Tarea MTS ----------

def task_MTS(
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
    MTS – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SD ----------

def task_SD(
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
    SD – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea SOSD ----------

def task_SOSD(
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
    SOSD – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Tarea RMTS ----------

def task_RMTS(
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
    RMTS – Devuelve...
    """
    sample_pos = False
    sample_neg = False
    return sample_neg, sample_pos


# ---------- Registro de tareas ----------
# Tareas SVRT y tareas cenia
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
    ["task_MTS", task_MTS],
    ["task_SD", task_SD],
    ["task_SOSD", task_SOSD],
    ["task_RMTS", task_RMTS]
]
