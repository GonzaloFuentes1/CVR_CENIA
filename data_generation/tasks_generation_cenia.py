import numpy as np
from matplotlib.path import Path
from shapely import Polygon, box

from data_generation.shape import Shape

from data_generation.utils import (check_square, sample_contact_many,
                                   sample_position_inside_1,
                                   sample_positions_align, sample_positions_bb,
                                   sample_positions_equidist,
                                   sample_positions_square,
                                   sample_random_colors,
                                   sample_positions_symmetric_pairs,
                                   sample_positions_circle,
                                   sample_position_inside_many,
                                   sample_position_outside_1
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

def decorate_shapes(
    shapes,
    max_size=0.4,
    min_size=None,
    size=None,         # Booleano legacy, True para aleatorio, False para fijo
    sizes=None,        # Array explícito de tamaños, preferido para tareas modernas
    rotate=False,
    color=False,
    flip=False,
    align=False,
    mirror=False,
    circle=False,
    inside=False,
    middle=0
):
    """
    Adorna N figuras con tamaño, posición (sin solapamiento), rotación, flip y color.
    Devuelve: (xy, size, shape_wrapped, colors)
    """
    n = len(shapes)

    # --- Selección de tamaños robusta ---
    if sizes is not None:
        size = np.array(sizes)
        if size.ndim == 1:
            size = size[:, None]
    else:
        min_size = min_size or max_size / 2
        if size is True:
            # Compatibilidad: tamaño aleatorio por figura
            size_vals = np.random.rand(n) * (max_size - min_size) + min_size
            size = size_vals[:, None]
        else:
            # Todas las figuras mismo tamaño
            size = np.full((n, 1), fill_value=max_size/2)

    # Lógica especial para middle (usada en algunas tareas legacy)
    if middle == 1 and n > 1:
        size[1] = max_size
    elif middle == 2 and n > 2:
        k = np.random.rand()
        if k >= 0.5:
            size[0] = max_size
        else:
            size[2] = max_size

    # --- Posicionamiento ---
    size_batch = size[None, ...]  # shape (1, n, 1)

    if align:
        sizes_sum = size_batch[0].sum()
        if sizes_sum >= 1:
            size_batch *= (1 / (sizes_sum * 0.8))  # Normalizar para que sumen menos de 1
        xy_vals = sample_positions_align(size_batch)[0]
    elif mirror:
        xy_vals = sample_positions_symmetric_pairs(size_batch[0])

    elif circle:
        xy_vals = sample_positions_circle(size_batch[0])
    else:
        xy_vals = sample_positions_bb(size_batch)[0]

    xy = xy_vals[:, None, :]  # shape (n, 1, 2)

    # --- Transformaciones geométricas ---
    for s in shapes:
        if rotate:
            s.rotate(np.random.rand() * 2 * np.pi)
        if flip and np.random.rand() > 0.5:
            s.flip()

    # --- Colores ---
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
    shape_mode: str = "normal",
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    rigid_type: str = "polygon",
    max_size: float = 0.9,
    min_size: float = 0.2,
    color: bool = False,
    rotate: bool = False,
    flip: bool = False,
    min_center_dist: float = 0.05,
    edge_gap: float = 0.01,
    max_global_tries: int = 10000
):
    """
    sample_pos : inner centrado y estrictamente contenido   (clase 1)
    sample_neg : inner desplazada y pegada al borde interior (clase 0)
    """

    for _ in range(max_global_tries):

        # Crear figuras base
        outer = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)
        inner = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms)

        outer.scale(max_size)
        inner.scale(min_size* max_size)

        _, _, _, col = decorate_shapes(
            [outer, inner],
            sizes=[[1.0], [1.0]],
            rotate=rotate,
            flip=flip,
            color=color,
        )
        colors_pair = col if color else [np.zeros((1, 3), dtype=np.float32)] * 2
        size_pair = np.array([[1.0], [1.0]])

        # --- Intentar ubicar outer centrado dentro del canvas ---
        for _outer_try in range(30):
            xy_outer = np.random.uniform(0.1, 0.9, size=2)
            cont_out = outer.get_contour() + xy_outer
            if (cont_out >= 0).all() and (cont_out <= 1).all():
                break
        else:
            continue  # no se pudo ubicar outer dentro del canvas

        # --- sample_pos ---
        xy_inner_pos = xy_outer
        cont_in_pos = inner.get_contour() + xy_inner_pos
        if not ((cont_in_pos >= 0).all() and (cont_in_pos <= 1).all()):
            continue
        if not Polygon(cont_out).contains_properly(Polygon(cont_in_pos)):
            continue

        # --- sample_neg ---
        r_out_min = np.min(np.linalg.norm(outer.get_contour(), axis=1))
        r_in_max = np.max(np.linalg.norm(inner.get_contour(), axis=1))
        available = r_out_min - r_in_max

        d_low = max(min_center_dist, available - edge_gap)
        d_high = available - 1e-6
        if d_high <= d_low:
            continue

        # Proyección radial única
        u = np.random.randn(2)
        u /= np.linalg.norm(u) + 1e-9
        d = np.random.uniform(d_low, d_high)
        xy_inner_neg = xy_outer + u * d
        cont_in_neg = inner.get_contour() + xy_inner_neg

        if not ((cont_in_neg >= 0).all() and (cont_in_neg <= 1).all()):
            continue
        if not Polygon(cont_out).contains_properly(Polygon(cont_in_neg)):
            continue

        # Empaquetado
        xy_pos = np.stack([xy_outer, xy_inner_pos])[:, None, :]
        xy_neg = np.stack([xy_outer, xy_inner_neg])[:, None, :]

        shapes_pos = [[outer], [inner]]
        shapes_neg = [[outer.clone()], [inner.clone()]]

        sample_pos = (xy_pos, size_pair, shapes_pos, colors_pair)
        sample_neg = (xy_neg, size_pair, shapes_neg, colors_pair)
        return sample_neg, sample_pos

    raise RuntimeError("task_svrt_2: no se pudo generar datos válidos")


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
    max_tries: int = 10000,
):
    """
    SVRT #4 – Devuelve (sample_neg, sample_pos).
    sample_pos: figura chica completamente dentro de la grande (clase 1)
    sample_neg: mismas figuras separadas, sin inclusión ni solapamiento (clase 0)
    """
    if min_size is None:
        raise ValueError("min_size debe estar definido.")

    border_margin = 0.001
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

    max_attempts = 1000

    done_flag = False
    for _ in range(max_attempts):
        xy_inner_rel = sample_position_inside_1(outer, inner, scale=size_inner / size_outer)
        contour_outer = outer.get_contour() * size_outer + xy_outer
        contour_inner = inner.get_contour() * size_inner
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

        max_attempts = 1000
        done_flag = False
        for _ in range(max_attempts):
            xy_inner_rel = sample_position_inside_1(outer, inner, scale=size_inner / size_outer)
            contour_outer = outer.get_contour()*size_outer + xy_outer
            contour_inner = inner.get_contour()*size_inner
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
    shape_mode: str = "normal",
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = True,
    max_size: float = 0.25,
    min_size: float = 0.12,
    color: bool = False,
    rotate: bool = False,
    flip: bool = False,
    rigid_type: str = "polygon",
    max_tries: int = 300,
):
    canvas_box = box(0, 0, 1, 1)

    def to_polygons(shapes, xy, size):
        polys = []
        for i, shape in enumerate(shapes):
            scaled = shape.clone()
            scaled.scale(size[i][0])
            coords = scaled.get_contour() + xy[i, 0]
            poly = Polygon(coords)
            polys.append(poly)
        return polys

    def valid_scene(polys):
        for poly in polys:
            if not poly.within(canvas_box):
                return False
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i].intersects(polys[j]):
                    return False
        return True
    for _ in range(max_tries):
        big = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        small = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)

        bigs = [big.clone(), big.clone()]
        smalls = [small.clone(), small.clone()]
        all_shapes = bigs + smalls

        sizes = np.array([[max_size]] * 2 + [[min_size]] * 2)
        xy = sample_positions_bb(sizes[None, ...])[0]

        delta = xy[2] - xy[0]
        xy[3] = xy[1] + delta
        xy_pos = xy[:, None, :]

        polys = to_polygons(all_shapes, xy_pos, sizes)
        if valid_scene(polys):
            shapes_wrapped_pos = [[s] for s in all_shapes]
            colors_pos = sample_random_colors(4) if color else [np.zeros((1, 3), dtype=np.float32)] * 4
            sample_pos = (xy_pos, sizes, shapes_wrapped_pos, colors_pos)
            break
    else:
        raise RuntimeError("task_svrt_13: no se pudo generar ejemplo positivo válido")

    for _ in range(max_tries):
        big = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        small = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)

        bigs = [big.clone(), big.clone()]
        smalls = [small.clone(), small.clone()]
        all_shapes = bigs + smalls

        sizes = np.array([[max_size]] * 2 + [[min_size]] * 2)

        # Posiciones totalmente aleatorias con separación mínima implícita por sample_positions_bb
        xy = sample_positions_bb(sizes[None, ...])[0]
        xy_neg = xy[:, None, :]

        polys = to_polygons(all_shapes, xy_neg, sizes)
        if not valid_scene(polys):
            continue

        v1 = xy[2] - xy[0]
        v2 = xy[3] - xy[1]
        if np.linalg.norm(v1 - v2) < 0.05:
            continue

        shapes_wrapped_neg = [[s] for s in all_shapes]
        colors_neg = sample_random_colors(4) if color else [np.zeros((1, 3), dtype=np.float32)] * 4
        sample_neg = (xy_neg, sizes, shapes_wrapped_neg, colors_neg)
        break
    else:
        raise RuntimeError("task_svrt_13: no se pudo generar ejemplo negativo válido")

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

    max_size = 0.2
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
    max_size = 0.2

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
    max_size = 0.2
    # --- Clase 1 ---
    shape = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shapes_pos = [shape.clone() for _ in range(6)]
    sample_pos = decorate_shapes(shapes_pos, max_size=max_size, min_size=min_size, color=color, mirror=True)
    
    # --- Clase 0 ---
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
    scale_factor = np.random.uniform(0.2, 1.5)
    size1 = np.random.uniform(min_size, max_size)
    size2 = size1 * scale_factor
    sizes = np.array([[size1], [size2]])   
    shape_pos_2 = shape_pos_1.clone()
    shapes_pos = [shape_pos_1, shape_pos_2]
    sample_pos = decorate_shapes(
        shapes_pos,
        max_size=max_size,
        min_size=min_size,
        color=color,
        sizes=sizes
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
    max_size: float = 0.8,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon',
    max_tries: int = 50,
):
    """
    SVRT #23 Devuelve:
     - Clase 1: una pequeña dentro y otra fuera de la grande.
     - Clase 0: las dos pequeñas ambas dentro o ambas fuera.
    """
    # Genera sample_neg
    for _ in range(max_tries):
        try:
            max_size = 0.8
            size_outer = max_size
            size_inner = min_size * 0.4

            # ——— Clase 0 ———
            outer0   = create_shape(shape_mode, rigid_type, radius, hole_radius,
                                    n_sides, fourier_terms, symm_rotate)
            inner0_1 = create_shape(shape_mode, rigid_type, radius, hole_radius,
                                    n_sides, fourier_terms, symm_rotate)
            inner0_2 = create_shape(shape_mode, rigid_type, radius, hole_radius,
                                    n_sides, fourier_terms, symm_rotate)
            sizes0 = np.array([[size_outer], [size_inner], [size_inner]], dtype=np.float32)
            shapes0 = [[outer0], [inner0_1], [inner0_2]]

            if np.random.rand() < 0.5:
                # Caso 0a: las dos pequeñas dentro de la grande
                rels = sample_position_inside_many(
                    outer0, [inner0_1, inner0_2], [size_inner/size_outer]*2
                )
                if len(rels) == 0:
                    raise RuntimeError("No posiciones válidas clase 0 dentro")
                raw_rel = rels[np.random.randint(len(rels))]

                c = outer0.get_contour()
                cmin, cmax = c.min(0), c.max(0)
                rel_norm = (raw_rel - cmin) / (cmax - cmin)
                center = np.array([0.5, 0.5], dtype=np.float32)
                offset = (rel_norm - 0.5) * size_outer
                xy0 = np.stack([
                    center,
                    center + offset[0],
                    center + offset[1]
                ])[:, None, :]

                if color:
                    cols0 = sample_random_colors(3)
                    colors0 = [cols0[i:i+1] for i in range(3)]
                else:
                    colors0 = [np.zeros((1,3), dtype=np.float32) for _ in range(3)]

                sample_neg = (xy0, sizes0, shapes0, colors0)

            else:
                # Caso 0b: las dos pequeñas fuera
                sample_neg = decorate_shapes(
                    [outer0, inner0_1, inner0_2],
                    sizes=sizes0,
                    max_size=max_size,
                    min_size=min_size,
                    color=color
                )

            break

        except RuntimeError:
            continue

    # Genera sample_pos
    for _ in range(max_tries):
        try:
            size_outer = max_size
            size_inner = min_size * size_outer * 0.6

            # ——— Clase 1 ———
            outer1 = create_shape(shape_mode, rigid_type, radius, hole_radius,
                                  n_sides, fourier_terms, symm_rotate)
            inner1 = create_shape(shape_mode, rigid_type, radius, hole_radius,
                                  n_sides, fourier_terms, symm_rotate)
            inner2 = create_shape(shape_mode, rigid_type, radius, hole_radius,
                                  n_sides, fourier_terms, symm_rotate)
            sizes1 = np.array([[size_outer], [size_inner], [size_inner]], dtype=np.float32)

            # 1) inner1 dentro
            cand_in = sample_position_inside_1(outer1, inner1, size_inner/size_outer)
            if cand_in.shape[0] == 0:
                raise RuntimeError("No pos válida inner1 dentro")
            raw_in = cand_in[np.random.randint(len(cand_in))]
            c = outer1.get_contour()
            cmin, cmax = c.min(0), c.max(0)
            pos_norm = (raw_in - cmin) / (cmax - cmin)
            center = np.array([0.5, 0.5], dtype=np.float32)
            xy1 = center + (pos_norm - center) * size_outer

            # 2) inner2 fuera
            cand_out = sample_position_outside_1(outer1, inner2, size_inner/size_outer)
            if cand_out.shape[0] == 0:
                raise RuntimeError("No pos válida inner2 fuera")
            raw_out = cand_out[np.random.randint(len(cand_out))]
            pos_norm2 = (raw_out - cmin) / (cmax - cmin)
            xy2 = center + (pos_norm2 - center) * size_outer

            xy_pos = np.stack([center, xy1, xy2])[:, None, :]

            if color:
                cols1 = sample_random_colors(3)
                colors1 = [cols1[i:i+1] for i in range(3)]
            else:
                colors1 = [np.zeros((1,3), dtype=np.float32) for _ in range(3)]

            sample_pos = (xy_pos, sizes1, [[outer1], [inner1], [inner2]], colors1)

            break

        except RuntimeError:
            continue

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
    # max_size <= 0.5
    max_size: float = 0.4,
    min_size: float | None = 0.2,
    color: bool = False,
    rigid_type: str = 'polygon'
):
    """
    MTS – Devuelve...
    """
    # Centro de la figura superior es (0.5, 0.75)
    # Centro de la figura inferior izquierda es (0.25, 0.75)
    # Centro de la figura inferior derecha es (0.75, 0.75)
    # Considerando esto, debemos tener max_size <= 0.5. (Se divide por 2 en el decorate shapes por lo que max_size <= 1)
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape3 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape4 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    if max_size > 1: 
        raise ValueError("max_size debe ser <= 1 para que las figuras encajen en el espacio definido.")
    # --------- Categoria 1 --------- 
    xy_pos, size_pos, shapes_pos, color_pos = decorate_shapes(
        [shape1.clone(), shape2.clone(), shape1.clone()],
        max_size=max_size, min_size=min_size, color=color
    )
    # Ajustamos las posiciones
    xy_pos[0] = np.array([[0.5, 0.25]], dtype=np.float32)  # Figura superior
    xy_pos[1] = np.array([[0.25, 0.75]], dtype=np.float32)  # Figura inferior izquierda
    xy_pos[2] = np.array([[0.75, 0.75]], dtype=np.float32)  # Figura inferior derecha

    sample_pos = (xy_pos, size_pos, shapes_pos, color_pos)
    # --------- Categoria 0 ---------
    xy_neg, size_neg, shapes_neg, color_neg = decorate_shapes(
        [shape3.clone(), shape3.clone(), shape4.clone()],
        max_size=max_size, min_size=min_size, color=color, align=True,
    )
    # Ajustamos las posiciones
    xy_neg[0] = np.array([[0.5, 0.25]], dtype=np.float32)  # Figura superior
    xy_neg[1] = np.array([[0.25, 0.75]], dtype=np.float32)  # Figura inferior izquierda
    xy_neg[2] = np.array([[0.75, 0.75]], dtype=np.float32)  # Figura inferior derecha

    sample_neg = (xy_neg, size_neg, shapes_neg, color_neg)
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

    return task_svrt_1(shape_mode, radius, hole_radius, n_sides, fourier_terms, symm_rotate,
                       poly_min_sides, poly_max_sides, max_size, min_size, color, rigid_type)


# ---------- Tarea SOSD ----------

def task_SOSD(
    shape_mode: str = 'normal',
    radius: float = 0.5,
    hole_radius: float = 0.05,
    n_sides: int = 5,
    fourier_terms: int = 20,
    symm_rotate: bool = False,
    poly_min_sides: int = 3,
    poly_max_sides: int = 10,
    max_size: float = 0.3,
    min_size: float | None = 0.15,
    color: bool = False,
    rotate: bool = False,
    flip: bool = False,
    rigid_type: str = 'polygon'
):
    """
    SOSD – Second-order same-different:
    sample_pos: A==B y C==D  ó  A!=B y C!=D  → clase 1
    sample_neg: A==B y C!=D  ó  A!=B y C==D  → clase 0
    """

    def apply_decorator_with_quadrants(shapes):
        # Aplica el decorador completo (tamaño, color, rotación, flip)
        xy, sizes, shapes_wrapped, colors = decorate_shapes(
            shapes,
            sizes=[[max_size]] * 4,
            rotate=rotate,
            flip=flip,
            color=color,
            align=False
        )
        # Reemplaza posiciones por cuadrantes fijos
        fixed_xy = np.array([
            [0.25, 0.75],  # A
            [0.75, 0.75],  # B
            [0.25, 0.25],  # C
            [0.75, 0.25],  # D
        ])[:, None, :]
        return fixed_xy, sizes, shapes_wrapped, colors

    # --- sample_pos ---
    if np.random.rand() < 0.5:
        # A == B, C == D → usar solo 2 shapes
        shape_ab = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_cd = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shapes_pos = [shape_ab.clone(), shape_ab.clone(), shape_cd.clone(), shape_cd.clone()]
    else:
        # A != B, C != D → 4 shapes distintas
        shapes_pos = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
                      for _ in range(4)]

    sample_pos = apply_decorator_with_quadrants(shapes_pos)

    # --- sample_neg ---
    if np.random.rand() < 0.5:
        # A == B, C != D
        shape_ab = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_c = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_d = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shapes_neg = [shape_ab.clone(), shape_ab.clone(), shape_c, shape_d]
    else:
        # A != B, C == D
        shape_a = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_b = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_cd = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shapes_neg = [shape_a, shape_b, shape_cd.clone(), shape_cd.clone()]

    sample_neg = apply_decorator_with_quadrants(shapes_neg)

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
    RMTS 
        AB
    CD       EF
    Clase 0: A == B y C == D y E != F o A != B y C != D y E == F
    Clase 1: A == B y C != D y E == F o A != B y C == D y E != F
    """
    max_size *= 0.4 # Ajuste para que figuras quepan sin superponerse, y permitir separar los pares

    fixed_xy = np.array([
        [0.4, 0.25],  # A
        [0.6, 0.25],  # B
        [0.15, 0.75],  # C
        [0.35, 0.75],  # D
        [0.65, 0.75],  # E
        [0.85, 0.75],  # F
    ])[:, None, :]  # shape (6, 1, 2)

    # Categoría 0:
    if np.random.rand() < 0.5:
        # A == B y C == D y E != F
        shape_ab = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_cd = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_e = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_f = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shapes_neg = [shape_ab, shape_ab.clone(), shape_cd, shape_cd.clone(), shape_e, shape_f]
        xy, size, shapes_neg, colors_neg = decorate_shapes(
            shapes_neg,
            max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 6
        )
        sample_neg = (fixed_xy, size, shapes_neg, colors_neg)
    else:
        # A != B y C != D y E == F
        shape_a = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_b = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_c = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_d = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_ef = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shapes_neg = [shape_a, shape_b, shape_c, shape_d, shape_ef, shape_ef.clone()]
        xy, size, shapes_neg, colors_neg = decorate_shapes(
            shapes_neg,
            max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 6
        )
        sample_neg = (fixed_xy, size, shapes_neg, colors_neg)

    # Categoría 1:
    if np.random.rand() < 0.5:
        # A == B y C != D y E == F
        shape_ab = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_c = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_d = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_ef = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shapes_pos = [shape_ab, shape_ab.clone(), shape_c, shape_d, shape_ef, shape_ef.clone()]
        xy, size, shapes_pos, colors_pos = decorate_shapes(
            shapes_pos,
            max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 6
        )
        sample_pos = (fixed_xy, size, shapes_pos, colors_pos)
    else:
        # A != B y C == D y E != F
        shape_a = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_b = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_cd = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_e = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shape_f = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
        shapes_pos = [shape_a, shape_b, shape_cd, shape_cd.clone(), shape_e, shape_f]
        xy, size, shapes_pos, colors_pos = decorate_shapes(
            shapes_pos,
            max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 6
        )
        sample_pos = (fixed_xy, size, shapes_pos, colors_pos)

    return sample_neg, sample_pos

# ---------- Tareas basadas en simetría ----------

# ---------- Tarea de clasificación de simetría ----------

def task_sym_classification(
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
    Clase 0: Figura es asimétrica
    Clase 1: Figura es simétrica
    """

    # Clase 0: Figura asimétrica
    shape_neg = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    sample_neg = decorate_shapes(
        [shape_neg],
        max_size=max_size,
        min_size=min_size,
        color=color,
        sizes=[[max_size]]
    )

    # Clase 1: Figura simétrica
    shape_pos = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    # Aplicar simetría
    shape_pos.symmetrize(rotate = True)
    sample_pos = decorate_shapes(
        [shape_pos],
        max_size=max_size,
        min_size=min_size,
        color=color,
        sizes=[[max_size]]
    )

    return sample_neg, sample_pos

# ---------- Tarea MTS basada en simetría ----------

def task_sym_MTS(
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
    Tarea de clasificación de simetría MTS.
    En clase 0, figura de la izquierda tiene mismo estatus de simetría que figura de arriba, mientras que figura de la derecha tiene el estatus opuesto.
    En clase 1, figura de la derecha tiene el mismo estatus de simetría que figura de arriba, mientras que figura de la izquierda tiene el estatus opuesto.
    """

    fixed_xy = np.array([
        [0.5, 0.25],  # A
        [0.25, 0.75],  # B
        [0.75, 0.75]   # C
    ])[:, None, :]  # shape (3, 1, 2)

    # Clase 0
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(3)]

    if np.random.rand() < 0.5:
        # Comparten simetría
        shapes[0].symmetrize(rotate=True)  # A simétrica
        shapes[1].symmetrize(rotate=True)
    else:
        # A y B comparten ser asimétricas, C es simétrica
        shapes[2].symmetrize(rotate=True)  # C simétrica

    xy, size, shapes_neg, colors_neg = decorate_shapes(
        shapes,
        max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 3
    )
    sample_neg = (fixed_xy, size, shapes_neg, colors_neg)

    # Clase 1
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(3)]
    if np.random.rand() < 0.5:
        # A y C comparten simetría, B es asimétrica
        shapes[0].symmetrize(rotate=True)  # A simétrica
        shapes[2].symmetrize(rotate=True)  # C simétrica
    else:
        # A y C son asimétricas, B es simétrica
        shapes[1].symmetrize(rotate=True)
    xy, size, shapes_pos, colors_pos = decorate_shapes(
        shapes,
        max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 3
    )
    sample_pos = (fixed_xy, size, shapes_pos, colors_pos)

    return sample_neg, sample_pos

# ---------- Tarea SD basada en simetría ----------

def task_sym_SD(
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
    Tarea de clasificación de simetría SD.
    Clase 0: Una figura simétrica y una asimétrica.
    Clase 1: Ambas figuras son simétricas o ambas son asimétricas.
    """

    # Clase 0: Una figura simétrica y una asimétrica
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2.symmetrize(rotate=True)  # Hacemos que shape2 sea simétrica
    sample_neg = decorate_shapes(
        [shape1, shape2],
        max_size=max_size,
        min_size=min_size,
        color=color,
        sizes=[[max_size], [max_size]]
    )
    # Clase 1: Ambas figuras son simétricas o ambas son asimétricas
    shape1 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    shape2 = create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate)
    if np.random.rand() < 0.5:
        # Ambas figuras simétricas
        shape1.symmetrize(rotate=True)
        shape2.symmetrize(rotate=True)
    sample_pos = decorate_shapes(
        [shape1, shape2],
        max_size=max_size,
        min_size=min_size,
        color=color,
        sizes=[[max_size], [max_size]]
    )

    return sample_neg, sample_pos

# ---------- Tarea SOSD basada en simetría ----------

def task_sym_SOSD(
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
    Tarea de clasificación de simetría SOSD.
    Clase 1: Ambos pares de figuras tienen misma relación de simetría
    Es decir, si en un par ambas son simétricas, en el otro par debe ocurrir lo mismo.
    Si en un par una es simétrica y la otra no, en el otro par debe ocurrir lo mismo.
    Si en un par ambas son asimétricas, en el otro par debe ocurrir lo mismo. 
    -------Duda: Consideramos este último caso?---------

    Clase 0: Un par de figuras tiene una relación de simetría diferente al otro par.
    """

    fixed_xy = np.array([
        [0.25, 0.25],  # A
        [0.75, 0.25],  # B
        [0.25, 0.75],  # C
        [0.75, 0.75]   # D
    ])[:, None, :]  # shape (4, 1, 2)

    # Clase 0
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(4)]

    if np.random.rand() < 0.5:
        # Relación de arriba será 1 sim y 1 asim
        if np.random.rand() < 0.5:
            shapes[0].symmetrize(rotate=True)  # A simétrica
        else:
            shapes[1].symmetrize(rotate=True)
        # Relación de abajo puede ser sim sim o asim asim
        if np.random.rand() < 0.5:
            shapes[2].symmetrize(rotate=True)
            shapes[3].symmetrize(rotate=True)
        else:
            # Ambas asimétricas
            pass
    else:
        # Relación de arriba será sim sim o asim asim
        if np.random.rand() < 0.5:
            shapes[0].symmetrize(rotate=True)
            shapes[1].symmetrize(rotate=True)
        else:
            # Ambas asimétricas
            pass  
        # Relación de abajo será 1 sim y 1 asim
        if np.random.rand() < 0.5:
            shapes[2].symmetrize(rotate=True)
        else:
            shapes[3].symmetrize(rotate=True)
    xy, size, shapes_neg, colors_neg = decorate_shapes(
        shapes,
        max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 4
    )
    sample_neg = (fixed_xy, size, shapes_neg, colors_neg)

    # Clase 1
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(4)]
    if np.random.rand() < 0.5:
        # Relación de arriba será 1 sim y 1 asim
        if np.random.rand() < 0.5:
            shapes[0].symmetrize(rotate=True)  # A simétrica
        else:
            shapes[1].symmetrize(rotate=True)
        # Relación de abajo será igual a la de arriba (1 y 1)
        if np.random.rand() < 0.5:
            shapes[2].symmetrize(rotate=True)
        else:
            shapes[3].symmetrize(rotate=True)
    else:
        # Relación de arriba será sim sim o asim asim
        if np.random.rand() < 0.5:
            shapes[0].symmetrize(rotate=True)
            shapes[1].symmetrize(rotate=True)
            # Relación de abajo será igual a la de arriba
            shapes[2].symmetrize(rotate=True)
            shapes[3].symmetrize(rotate=True)
        else:
            # Ambas asimétricas
            # Relación de abajo será igual a la de arriba
            pass  

    xy, size, shapes_pos, colors_pos = decorate_shapes(
        shapes,
        max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 4
    )
    sample_pos = (fixed_xy, size, shapes_pos, colors_pos)

    return sample_neg, sample_pos

# ---------- Tarea RMTS basada en simetría ----------

def task_sym_RMTS(
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
    Tarea de clasificación de simetría RMTS.
    Clase 0: par de la izquierda tiene misma relación de simetría que el par de arriba, mientras que el par de la derecha tiene una relación diferente.
    Clase 1: par de la izquierda tiene relación de simetría diferente al par de arriba, mientras que el par de la derecha tiene la misma relación que el par de arriba.
    """
    max_size *= 0.4 # Ajuste para que figuras quepan sin superponerse, y permitir separar los pares

    fixed_xy = np.array([
        [0.4, 0.25],  # A
        [0.6, 0.25],  # B
        [0.15, 0.75],  # C
        [0.35, 0.75],  # D
        [0.65, 0.75],  # E
        [0.85, 0.75],  # F
    ])[:, None, :]  # shape (6, 1, 2)

    # Clase 0
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(6)]
    if np.random.rand() < 0.5:
        # arriba relación 1 sim y 1 asim
        if np.random.rand() < 0.5:
            shapes[0].symmetrize(rotate=True)
        else:
            shapes[1].symmetrize(rotate=True)
        # izquierda relación 1 sim y 1 asim
        if np.random.rand() < 0.5:
            shapes[2].symmetrize(rotate=True)
        else:
            shapes[3].symmetrize(rotate=True)
        # derecha ambos sim o ambos asim
        if np.random.rand() < 0.5:
            shapes[4].symmetrize(rotate=True)
            shapes[5].symmetrize(rotate=True)
        else:
            # Ambas asimétricas
            pass
    else:
        # arriba relación sim sim o asim asim
        if np.random.rand() < 0.5:
            shapes[0].symmetrize(rotate=True)
            shapes[1].symmetrize(rotate=True)
            # izquierda misma relación
            shapes[2].symmetrize(rotate=True)
            shapes[3].symmetrize(rotate=True)
        else:
            # Ambas asimétricas
            # izquierda misma relación
            pass
        # derecha relación 1 sim y 1 asim
        if np.random.rand() < 0.5:
            shapes[4].symmetrize(rotate=True)
        else:
            shapes[5].symmetrize(rotate=True)
    xy, size, shapes_neg, colors_neg = decorate_shapes(
        shapes,
        max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 6
    )
    sample_neg = (fixed_xy, size, shapes_neg, colors_neg)

    # Clase 1
    shapes = [create_shape(shape_mode, rigid_type, radius, hole_radius, n_sides, fourier_terms, symm_rotate) for _ in range(6)]

    if np.random.rand() < 0.5:
        # arriba relación 1 sim y 1 asim
        if np.random.rand() < 0.5:
            shapes[0].symmetrize(rotate=True)
        else:
            shapes[1].symmetrize(rotate=True)
        # izquierda relación sim sim o asim asim
        if np.random.rand() < 0.5:
            shapes[2].symmetrize(rotate=True)
            shapes[3].symmetrize(rotate=True)
        else:
            # Ambas asimétricas
            pass
        # derecha relación 1 sim y 1 asim
        if np.random.rand() < 0.5:
            shapes[4].symmetrize(rotate=True)
        else:
            shapes[5].symmetrize(rotate=True)
    else:
        # arriba relación sim sim o asim asim
        if np.random.rand() < 0.5:
            shapes[0].symmetrize(rotate=True)
            shapes[1].symmetrize(rotate=True)
            # derecha misma relación
            shapes[4].symmetrize(rotate=True)
            shapes[5].symmetrize(rotate=True)
        else:
            # Ambas asimétricas
            # derecha misma relación
            pass
        # izquierda relación 1 sim y 1 asim
        if np.random.rand() < 0.5:
            shapes[2].symmetrize(rotate=True)
        else:
            shapes[3].symmetrize(rotate=True)
    xy, size, shapes_pos, colors_pos = decorate_shapes(
        shapes,
        max_size=max_size, min_size=min_size, color=color, sizes=[[max_size]] * 6
    )
    sample_pos = (fixed_xy, size, shapes_pos, colors_pos)

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
    ["task_RMTS", task_RMTS],
    ["task_sym_classification", task_sym_classification],
    ["task_sym_MTS", task_sym_MTS],
    ["task_sym_SD", task_sym_SD],
    ["task_sym_SOSD", task_sym_SOSD],
    ["task_sym_RMTS", task_sym_RMTS]
]
