import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class ClipVisionClassifier(nn.Module):
    def __init__(self, vision_model, num_classes, projection_dim):
        super().__init__()
        self.vision_model = vision_model
        self.class_head = nn.Linear(projection_dim, num_classes)

    def forward(self, x):
        outputs = self.vision_model(x)
        # Usar pooler_output (vector global de la imagen)
        pooled = outputs.pooler_output  # [batch, projection_dim]
        return self.class_head(pooled)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Ruta base con train/val/test subcarpetas')
    parser.add_argument('--output_dir', type=str, default='./clipvit-training')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--cuda_devices', type=str, default="", help='Ej: "0,1,2"')
    args = parser.parse_args()

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}, GPUs disponibles: {torch.cuda.device_count()}")

    # CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_size = processor.image_processor.size["shortest_edge"]
    normalize = transforms.Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Cargar datasets
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=transform)
    val_dataset   = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=transform)
    test_dataset  = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"📁 Clases detectadas: {train_dataset.classes}")

    # Modelo base + cabeza de clasificación
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = ClipVisionClassifier(
        vision_model=clip.vision_model,
        num_classes=num_classes,
        projection_dim=clip.vision_model.config.hidden_size  # <--- CORREGIDO
    )

    if torch.cuda.device_count() > 1:
        print(f"🚀 Usando {torch.cuda.device_count()} GPUs con DataParallel")
        model = nn.DataParallel(model)

    model.to(device)

    # Optimizador y loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Entrenamiento
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} training"):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"📊 Epoch {epoch} - Train Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")

        # Validación
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"✅ Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if val_acc >= 0.99 and train_acc >= 0.99:
            print("🛑 Early stopping (val ≥ 99.9%)")
            break

    # Test final
    print("\n🧪 Evaluación final en test")
    model.eval()
    all_labels, all_preds = [], []
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_acc = correct / total
    print(f"🎯 Test Accuracy: {test_acc:.4f}")

    # Guardado del modelo
    os.makedirs(args.output_dir, exist_ok=True)
    task_name = os.path.basename(args.data_dir.rstrip("/"))
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"clipvit_{task_name}.pth"))

    # Métricas detalladas
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    csv_path = os.path.join(args.output_dir, f"clipvit_metrics_{task_name}.csv")
    df.to_csv(csv_path, index=True)
    print(f"📄 Métricas guardadas en: {csv_path}")


if __name__ == "__main__":
    main()
