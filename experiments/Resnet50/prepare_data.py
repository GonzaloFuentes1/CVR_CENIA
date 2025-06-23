import os
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image

def reorganize_dataset(original_dir, output_dir, split_names):
    for split in split_names:
        split_path = os.path.join(original_dir, split)
        if not os.path.exists(split_path):
            print(f"❌ Carpeta no encontrada: {split_path}")
            continue

        output_split_dir = os.path.join(output_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)

        images = glob(os.path.join(split_path, "*.png"))

        for path in tqdm(images, desc=f"Procesando {split}"):
            filename = os.path.basename(path)
            if not filename.startswith("class_"):
                continue
            try:
                class_id = filename.split("_")[1]
                class_dir = os.path.join(output_split_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)

                # Convertir a JPG
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    new_name = os.path.splitext(filename)[0] + ".jpg"
                    img.save(os.path.join(class_dir, new_name), format="JPEG", quality=95)
            except Exception as e:
                print(f"⚠️ Error con archivo {filename}: {e}")

    print(f"\n✅ Dataset reestructurado en: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reestructura dataset a formato tipo ImageFolder con JPGs.")
    parser.add_argument("--original_dir", type=str, required=True, help="Ruta al dataset original con PNGs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Ruta destino para el dataset procesado.")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Nombres de los splits separados por coma.")

    args = parser.parse_args()
    split_names = args.splits.split(",")

    reorganize_dataset(args.original_dir, args.output_dir, split_names)
