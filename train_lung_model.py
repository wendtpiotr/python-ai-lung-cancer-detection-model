
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from tqdm import tqdm

# =========================
# KONFIGURACJA
# =========================
DATA_DIR = "./lung_cancer_data/lung_colon_image_set/lung_image_sets"
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "lung_cancer_model.pth")

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 0  # !!! WINDOWS SAFE !!!

# =========================
# DATASET
# =========================
class LungCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        classes = [
            ("lung_n", 0),      # normal
            ("lung_aca", 1),    # cancer
            ("lung_scc", 1),    # cancer
        ]

        for folder_name, label in classes:
            folder = os.path.join(root_dir, folder_name)
            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(folder, file))
                    self.labels.append(label)

        print(f"Znaleziono {len(self.images)} obrazów")
        print(f"- Cancer: {sum(self.labels)}")
        print(f"- Normal: {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# TRANSFORMACJE
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =========================
# TRENING
# =========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUżywam urządzenia: {device}")

    full_dataset = LungCancerDataset(DATA_DIR, transform=None)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n🔄 Rozpoczynam trening...\n")

    for epoch in range(EPOCHS):
        model.train()
        correct = total = 0
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoka {epoch+1}/{EPOCHS}")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*correct/total:.2f}%"
            )

        # ===== VALIDATION =====
        model.eval()
        val_correct = val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.unsqueeze(1).to(device)

                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Walidacja: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Zapisano najlepszy model ({best_acc:.2f}%)")

    print("\n✅ TRENING ZAKOŃCZONY")
    print(f"Najlepsza dokładność: {best_acc:.2f}%")
    print(f"Model zapisany w: {MODEL_PATH}")


# =========================
# POBIERANIE DATASETU (opcjonalne)
# =========================
def download_dataset():
    try:
        import kaggle
    except ImportError:
        os.system("pip install kaggle")
        import kaggle

    print("\n⬇️ Pobieram dataset z Kaggle...")
    kaggle.api.dataset_download_files(
        "andrewmvd/lung-and-colon-cancer-histopathological-images",
        path="./lung_cancer_data",
        unzip=True
    )
    print("✅ Dataset gotowy")


# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("🫁 LUNG CANCER MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print("\nDataset nie znaleziony.")
        choice = input("Pobrać teraz? (t/n): ").lower()
        if choice == "t":
            download_dataset()
        else:
            print("❌ Brak datasetu – koniec.")
            return
    else:
        print("\n📁 Dataset już istnieje – pomijam pobieranie.")

    train()


if __name__ == "__main__":
    main()
