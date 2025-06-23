import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Ruta base con task_MTS/train, validation y test')
    parser.add_argument('--output_dir', type=str, default='./resnet50-mts-imagefolder')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cuda_devices', type=str, default="", help='IDs de CUDA visibles (ej: "4,5,6,7")')
    args = parser.parse_args()

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # ───── Configuración ───────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")
    print(f"🧠 GPUs visibles: {torch.cuda.device_count()}")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")

    # ───── Transformaciones ────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ───── Datasets y Loaders ──────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"📁 Clases detectadas: {train_dataset.classes}")
    num_classes = len(train_dataset.classes)

    # ───── Modelo ──────────────────────────────────────────────────────────────
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Usando {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ───── Entrenamiento ───────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        print(f"\n🔁 Epoch {epoch + 1}/{args.epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Entrenando"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"📊 Train Loss: {running_loss:.4f} | Accuracy: {train_acc * 100:.2f}%")

        # ───── Validación ──────────────────────────────────────────────────────
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"✅ Validation Accuracy: {val_acc * 100:.4f}%")

        if val_acc >= 0.99 and train_acc >= 0.99:
            print("🛑 Early stopping triggered: Validation Accuracy >= 99%")
            break

    # ───── Testeo final ────────────────────────────────────────────────────────
    print("\n🧪 Evaluando en set de testeo...")
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_acc = correct / total
    print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")

    # ───── Guardado del modelo ────────────────────────────────────────────────
    task_name = os.path.basename(args.data_dir.rstrip("/"))
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"resnet50_{task_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Modelo guardado en: {save_path}")

    # ───── Guardado de métricas detalladas ────────────────────────────────────
    report = classification_report(
        all_labels,
        all_preds,
        target_names=train_dataset.classes,
        output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    csv_path = os.path.join(args.output_dir, f"resnet50_metrics_{task_name}.csv")
    df_report.to_csv(csv_path)
    print(f"📄 Métricas detalladas guardadas en: {csv_path}")


if __name__ == "__main__":
    main()
