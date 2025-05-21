import numpy as np

from data_generation.shape import Shape
from data_generation.utils import sample_positions_bb, sample_random_colors

# ---------- Generador de figuras ----------

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

    return sample_pos, sample_neg

# ---------- Registro de tareas ----------

TASKS_SVRT = [
    ["task_svrt_1", task_svrt_1, "The images contain objects of the same shape."],
]
