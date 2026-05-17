from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def train_detection(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.det_base_model)
    results = model.train(
        data=str(args.det_data_yaml),
        epochs=args.det_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.det_name,
        patience=args.patience,
        workers=args.workers,
        exist_ok=True,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"[detection] best weights: {best}")
    return best


def train_classifier(args: argparse.Namespace) -> Path:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    train_dir = args.cls_root / "train"
    val_dir = args.cls_root / "valid"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing classification train folder: {train_dir}")
    if not val_dir.exists():
        val_dir = train_dir

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.cls_image_size, args.cls_image_size)),
            transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.15, 0.03)], p=0.7),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((args.cls_image_size, args.cls_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.cls_batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.cls_batch, shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if args.pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(train_ds.classes)))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.cls_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.cls_epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = args.project / args.cls_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_path = out_dir / "best_classifier.pt"
    for epoch in range(args.cls_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
        scheduler.step()

        val_acc = evaluate_classifier(model, val_loader, device)
        print(f"[classifier] epoch={epoch + 1}/{args.cls_epochs} loss={train_loss / max(1, len(train_loader)):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": train_ds.classes,
                    "image_size": args.cls_image_size,
                    "arch": "resnet50",
                    "val_acc": best_acc,
                },
                best_path,
            )
    (out_dir / "classes.json").write_text(json.dumps(train_ds.classes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[classifier] best weights: {best_path}")
    return best_path


def evaluate_classifier(model, loader, device) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    return correct / total if total else 0.0


def train_fewshot(args: argparse.Namespace) -> Path:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms

    class EpisodicLabelDataset(Dataset):
        def __init__(self, root: Path, episodes: int, n_way: int, k_shot: int, query: int, image_size: int) -> None:
            self.root = root
            self.episodes = episodes
            self.n_way = n_way
            self.k_shot = k_shot
            self.query = query
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.15, 0.03)], p=0.6),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            self.classes = {}
            for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                images = [p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
                if len(images) >= k_shot + query:
                    self.classes[class_dir.name] = images
            if len(self.classes) < n_way:
                raise ValueError(f"Need at least {n_way} classes with >= {k_shot + query} images in {root}")

        def __len__(self) -> int:
            return self.episodes

        def __getitem__(self, index: int):
            del index
            selected = random.sample(list(self.classes), self.n_way)
            support_images = []
            query_images = []
            query_labels = []
            for label, class_name in enumerate(selected):
                chosen = random.sample(self.classes[class_name], self.k_shot + self.query)
                for path in chosen[: self.k_shot]:
                    support_images.append(self.transform(Image.open(path).convert("RGB")))
                for path in chosen[self.k_shot :]:
                    query_images.append(self.transform(Image.open(path).convert("RGB")))
                    query_labels.append(label)
            return torch.stack(support_images), torch.stack(query_images), torch.tensor(query_labels)

    class ProtoNet(nn.Module):
        def __init__(self, embedding_dim: int = 128, pretrained: bool = True) -> None:
            super().__init__()
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.projection = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, embedding_dim),
            )

        def encode(self, x):
            return F.normalize(self.projection(self.backbone(x)), dim=1)

        def forward(self, support, query, n_way: int, k_shot: int):
            batch = support.shape[0]
            support = support.view(batch * n_way * k_shot, *support.shape[2:])
            query = query.view(batch * query.shape[1], *query.shape[2:])
            support_emb = self.encode(support).view(batch, n_way, k_shot, -1).mean(dim=2)
            query_emb = self.encode(query).view(batch, -1, support_emb.shape[-1])
            return -torch.cdist(query_emb, support_emb).view(-1, n_way)

    train_root = args.fewshot_root / "train"
    val_root = args.fewshot_root / "valid"
    if not val_root.exists():
        val_root = train_root
    train_ds = EpisodicLabelDataset(train_root, args.episodes, args.n_way, args.k_shot, args.query_num, args.cls_image_size)
    val_ds = EpisodicLabelDataset(val_root, max(16, args.episodes // 5), args.n_way, args.k_shot, args.query_num, args.cls_image_size)
    train_loader = DataLoader(train_ds, batch_size=args.episode_batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.episode_batch, shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    model = ProtoNet(args.embedding_dim, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.fewshot_lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_acc = -1.0
    out_dir = args.project / args.fewshot_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_fewshot_encoder.pt"

    for epoch in range(args.fewshot_epochs):
        model.train()
        for support, query, labels in train_loader:
            support = support.to(device)
            query = query.to(device)
            labels = labels.view(-1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(support, query, args.n_way, args.k_shot)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        val_acc = evaluate_fewshot(model, val_loader, device, args.n_way, args.k_shot)
        print(f"[fewshot] epoch={epoch + 1}/{args.fewshot_epochs} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "protonet_resnet50",
                    "embedding_dim": args.embedding_dim,
                    "image_size": args.cls_image_size,
                    "n_way": args.n_way,
                    "k_shot": args.k_shot,
                    "query_num": args.query_num,
                    "val_acc": best_acc,
                },
                best_path,
            )
    print(f"[fewshot] best weights: {best_path}")
    return best_path


def evaluate_fewshot(model, loader, device, n_way: int, k_shot: int) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for support, query, labels in loader:
            support = support.to(device)
            query = query.to(device)
            labels = labels.view(-1).to(device)
            preds = model(support, query, n_way, k_shot).argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    return correct / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google Colab training entrypoint for the Checking-Label stack.")
    parser.add_argument("--task", choices=["detection", "classifier", "fewshot", "all"], default="all")
    parser.add_argument("--dataset-root", type=Path, default=Path("/content/drive/MyDrive/CheckingLabel/datasets"))
    parser.add_argument("--project", type=Path, default=Path("/content/drive/MyDrive/CheckingLabel/runs"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", default=True)

    parser.add_argument("--det-data-yaml", type=Path, default=None)
    parser.add_argument("--det-base-model", default="yolo11s.pt")
    parser.add_argument("--det-name", default="label_detection_yolo11")
    parser.add_argument("--det-epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--cls-root", type=Path, default=None)
    parser.add_argument("--cls-name", default="label_classifier_resnet50")
    parser.add_argument("--cls-epochs", type=int, default=30)
    parser.add_argument("--cls-image-size", type=int, default=224)
    parser.add_argument("--cls-batch", type=int, default=64)
    parser.add_argument("--cls-lr", type=float, default=1e-4)

    parser.add_argument("--fewshot-root", type=Path, default=None)
    parser.add_argument("--fewshot-name", default="label_identity_fewshot")
    parser.add_argument("--fewshot-epochs", type=int, default=40)
    parser.add_argument("--fewshot-lr", type=float, default=1e-4)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-batch", type=int, default=32)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--query-num", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    args = parser.parse_args()

    args.det_data_yaml = args.det_data_yaml or (args.dataset_root / "dataLabelDetection" / "data.yaml")
    args.cls_root = args.cls_root or (args.dataset_root / "dataLabelClassification_seen")
    args.fewshot_root = args.fewshot_root or (args.dataset_root / "dataLabelClassification_seen")
    return args


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if args.task in {"detection", "all"}:
        outputs["detection"] = str(train_detection(args))
    if args.task in {"classifier", "all"}:
        outputs["classifier"] = str(train_classifier(args))
    if args.task in {"fewshot", "all"}:
        outputs["fewshot"] = str(train_fewshot(args))
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
