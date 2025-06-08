import numpy as np

from data_generation.shape import Shape
from data_generation.utils import (
    sample_contact_many,
    sample_position_inside_1,
    sample_positions_align,
    sample_positions_bb,
    sample_random_colors,
    sample_positions_square,
    check_square,
    sample_positions_equidist
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

def decorate_shapes(shapes, max_size=0.4, min_size=None, size=False, rotate=False, color=False, flip=False, align=False, middle=0):
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
    max_tries: int = 30,
):
    """
    SVRT #4 – Devuelve (sample_neg, sample_pos).
    sample_pos: figura chica completamente dentro de la grande (clase 1)
    sample_neg: figuras separadas, una grande y otra chica, sin solapamiento (clase 0)
    """
    if min_size is None:
        raise ValueError("min_size debe estar definido.")

    size_outer = max_size
    size_inner = min_size*size_outer

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
        xy_outer = np.random.rand(2) * (1 - size_outer) + size_outer / 2
        xy_inner_rel = sample_position_inside_1(outer, inner, scale=1- (size_inner / size_outer))
        if len(xy_inner_rel) == 0:
            continue
        xy_inner = xy_inner_rel[0] + xy_outer
        contour_inner = inner.get_contour() + xy_inner
        if ((0 <= contour_inner).all() and (contour_inner <= 1).all()):
            break
    else:
        raise RuntimeError("No se pudo encontrar una posición válida para la clase positiva.")

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
    if shape_mode == 'rigid':
        n_a = n_sides
        n_b = np.random.choice([
            k for k in range(poly_min_sides, poly_max_sides) if k != n_sides
        ])
    else:
        n_a = n_b = n_sides

    shape_a = create_shape(shape_mode, rigid_type, radius, hole_radius, n_a, fourier_terms)
    shape_b = create_shape(shape_mode, rigid_type, radius, hole_radius, n_b, fourier_terms)
    shape_a.scale(size_outer)
    shape_b.scale(size_inner)

    for _ in range(max_tries):
        xy_a = np.random.rand(2) * (1 - size_outer) + size_outer / 2
        xy_b = np.random.rand(2) * (1 - size_inner) + size_inner / 2
        contour_a = shape_a.get_contour() + xy_a
        contour_b = shape_b.get_contour() + xy_b
        dist = np.min(np.linalg.norm(contour_a[:, None, :] - contour_b[None, :, :], axis=-1))
        if dist > 0.01:
            break
    else:
        raise RuntimeError("No se pudo generar clase negativa sin superposición.")

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

    max_attempts = 10

    done_flag = False
    for _ in range(max_attempts):
        xy_inner_rel = sample_position_inside_1(outer, inner, scale=size_inner / size_outer)
        if len(xy_inner_rel) > 0:
            xy_inner = xy_inner_rel[0] * size_outer + xy_outer
            done_flag = True
            break
        else:
            outer = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
            inner = outer.clone()
    if not done_flag:
        xy_inner = xy_outer 
        print("Fallo al encontrar posición válida para inner dentro de outer.")

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

        max_attempts = 10
        done_flag = False
        for _ in range(max_attempts):
            xy_inner_rel = sample_position_inside_1(outer, inner, scale=size_inner / size_outer)
            if len(xy_inner_rel) > 0:
                xy_inner = xy_inner_rel[0] * size_outer + xy_outer
                done_flag = True
                break
            else:
                outer = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
                inner = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        if not done_flag:
            xy_inner = xy_outer
            print("Fallo al encontrar posición válida para inner dentro de outer.")

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
