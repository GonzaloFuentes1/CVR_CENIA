import os
import subprocess

def run_prepare_data(original_dir, output_dir):
    print(f"🔧 Reorganizando dataset desde {original_dir}")
    subprocess.run([
        "python", "experiments/ClipVIT/prepare_data.py",
        "--original_dir", original_dir,
        "--output_dir", output_dir,
        "--splits", "train,val,test"
    ], check=True)
    print(f"✅ Dataset procesado en: {output_dir}\n")


def run_training(data_dir, output_dir, cuda_devices, batch_size, lr, epochs):
    print(f"🚀 Entrenando modelo CLIP-ViT con datos en: {data_dir}")
    command = [
        "python", "experiments/ClipVIT/ClipVIT_training.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--cuda_devices", cuda_devices,
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--epochs", str(epochs)
    ]
    subprocess.run(command, check=True)
    print(f"✅ Entrenamiento completado y modelo guardado en: {output_dir}\n")


if __name__ == "__main__":
    BASE_INPUT = "/workspace1/gonzalo.fuentes/CVR_CENIA/experiments/images"
    BASE_OUTPUT = "/workspace1/gonzalo.fuentes/CVR_CENIA/experiments/ClipVIT/images_processed"
    BASE_MODELS = "/workspace1/gonzalo.fuentes/CVR_CENIA/experiments/ClipVIT/models"

    tasks = [
        {
            "name": "task_MTS",
            "batch_size": 128,
            "lr": 1e-6,
            "epochs": 100
        },
        {
            "name": "task_SD",
            "batch_size": 128,
            "lr": 1e-5,
            "epochs": 100
        },
        {
            "name": "task_SOSD",
            "batch_size": 128,
            "lr": 1e-6,
            "epochs": 100
        },
        {
            "name": "task_RMTS",
            "batch_size": 128,
            "lr": 1e-6,
            "epochs": 100
        }
    ]

    cuda_devices = "4,5,6,7"

    for task in tasks:
        name = task["name"]
        print(f"\n========================= 🔁 Procesando {name} =========================")

        original_dir = os.path.join(BASE_INPUT, name)
        processed_dir = os.path.join(BASE_OUTPUT, name)
        model_output_dir = os.path.join(BASE_MODELS, f"clipvit-{name}")

        run_prepare_data(original_dir, processed_dir)
        run_training(
            data_dir=processed_dir,
            output_dir=model_output_dir,
            cuda_devices=cuda_devices,
            batch_size=task["batch_size"],
            lr=task["lr"],
            epochs=task["epochs"]
        )
