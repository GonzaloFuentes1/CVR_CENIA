import argparse
import logging
import os

import numpy as np
from PIL import Image

from data_generation.tasks_generation_cenia import TASKS_SVRT
from data_generation.utils import render_scene_safe

# Mapeo de índice entero → nombre de la tarea
TASKS_IDX = {
    1: "task_svrt_1",
    # puedes ir agregando más como:
    # 2: "task_svrt_2",
    # 3: "task_symmetry_rule",
}

def generate_dataset(task_name, task_fn, data_path, image_size, seed,
                     train_size, val_size, test_size):
    task_path = os.path.join(data_path, task_name)
    print(f"Generando dataset para {task_name} en {task_path}")
    os.makedirs(os.path.join(task_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(task_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(task_path, 'test'), exist_ok=True)

    splits = {
        'train': (seed, train_size),
        'val': (seed + 1, val_size),
        'test': (seed + 2, test_size),
    }

    for split, (split_seed, n_samples) in splits.items():
        np.random.seed(split_seed)
        for i in range(n_samples):
            sample_pos, sample_neg = task_fn()
            for label, (xy, size, shape, color) in enumerate([sample_pos, sample_neg]):
                image = render_scene_safe(xy, size, shape, color, image_size=image_size)
                img = Image.fromarray(image).convert('RGB')
                save_path = os.path.join(task_path, split, f'class_{label}_{i:05d}.png')
                img.save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='../cvrt_data/')
    parser.add_argument('--task_idx', type=int, required=True)
    parser.add_argument('--train_size', type=int, default=4)
    parser.add_argument('--val_size', type=int, default=4)
    parser.add_argument('--test_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)

    # Argumentos para generación de figuras
    parser.add_argument('--shape_mode', type=str, default='normal',
                        choices=['normal', 'rigid', 'smooth', 'symm'])
    parser.add_argument('--radius', type=float, default=0.5)
    parser.add_argument('--hole_radius', type=float, default=0.05)
    parser.add_argument('--n_sides', type=int, default=5)
    parser.add_argument('--fourier_terms', type=int, default=20)
    parser.add_argument('--max_size', type=float, default=0.4)
    parser.add_argument('--min_size', type=float, default=0.2)

    args = parser.parse_args()
    logging.info(f'JOB PID {os.getpid()}')

    # Obtener nombre y función desde TASKS_SVRT
    try:
        task_name = TASKS_IDX[args.task_idx]
    except KeyError:
        raise ValueError(f"Tarea con índice {args.task_idx} no registrada en TASKS_IDX.")

    try:
        task_fn_base = next(fn for name, fn, _ in TASKS_SVRT if name == task_name)
    except StopIteration:
        raise ValueError(f"No se encontró la función para {task_name} en TASKS_SVRT.")

    # Generar función con hiperparámetros inyectados
    def task_fn():
        return task_fn_base(
            shape_mode=args.shape_mode,
            radius=args.radius,
            hole_radius=args.hole_radius,
            n_sides=args.n_sides,
            fourier_terms=args.fourier_terms,
            max_size=args.max_size,
            min_size=args.min_size
        )

    generate_dataset(task_name, task_fn, args.data_dir, args.image_size,
                     args.seed, args.train_size, args.val_size, args.test_size)
