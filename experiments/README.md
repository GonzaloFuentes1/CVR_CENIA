# Experimentos CVR_CENIA

Este directorio contiene los scripts y configuraciones necesarios para entrenar y evaluar modelos en las tareas principales del benchmark (MTS, SD, SOSD, RMTS) usando ResNet50 y CLIP-ViT.

## 1. Generación de la base de datos

Antes de entrenar cualquier modelo, debes generar los datasets para cada tarea. Ejecuta los siguientes comandos desde la raíz del proyecto:

```sh
python generate_dataset_parallel.py --data_dir "experiments/images" --task_idx 27 --train_size 98000 --val_size 14000 --test_size 56000 --num_workers 30

python generate_dataset_parallel.py --data_dir "experiments/images" --task_idx 26 --train_size 49000 --val_size 7000 --test_size 28000 --num_workers 30

python generate_dataset_parallel.py --data_dir "experiments/images" --task_idx 25 --train_size 14000 --val_size 2800 --test_size 5600 --num_workers 30

python generate_dataset_parallel.py --data_dir "experiments/images" --task_idx 24 --train_size 14000 --val_size 2800 --test_size 5600 --num_workers 30
```

Esto generará las carpetas necesarias en `experiments/images` para cada tarea.

## 2. Configuración de los experimentos

Antes de ejecutar los entrenamientos, revisa y ajusta las rutas de entrada y salida en los archivos de ejecución masiva:

- `ClipVIT/run_all_clipvit.py`
- `Resnet50/run_all_resnets.py`

Asegúrate de que las variables de rutas (`BASE_INPUT`, `BASE_OUTPUT`, `BASE_MODELS`) estén correctamente configuradas según tu entorno.

> **Nota:** El entrenamiento se realizó utilizando 4 GPUs NVIDIA A100 en los clusters de CENIA. Es fundamental configurar correctamente las rutas, la asignación de GPUs y las variables de entorno (`cuda_devices`) para aprovechar los recursos del cluster y evitar errores de ejecución.

## 3. Ejecución de los experimentos

Para entrenar todos los modelos de cada tarea, ejecuta:

```sh
python experiments/Resnet50/run_all_resnets.py
python experiments/ClipVIT/run_all_clipvit.py.py
```

Cada script se encargará de preparar los datos, entrenar el modelo y guardar los resultados y métricas en la carpeta `models/` correspondiente.

## 4. Notas adicionales

- Si deseas entrenar solo una tarea específica, modifica la lista `tasks` en los scripts `run_all_*`.
- Puedes ajustar parámetros como batch size, learning rate y epochs según la memoria y tiempo disponible.
- Para usar múltiples GPUs, configura la variable `cuda_devices` en los scripts de entrenamiento y asegúrate de que la configuración sea compatible con el entorno del cluster.

## 5. Resultados

Los modelos entrenados y las métricas se guardarán en:

- `experiments/Resnet50/models/`
- `experiments/ClipVIT/models/`

---

Si tienes dudas sobre la generación de datasets o la ejecución de los experimentos, revisa los scripts y la documentación
