from __future__ import annotations

import cv2
import numpy as np


class HistogramIdentityEncoder:
    """Small deterministic label-crop encoder for the first open-set MVP.

    This is intentionally model-free so identity galleries, threshold logic and
    UI flow can be tested before a trained metric-learning checkpoint exists.
    """

    name = "histogram_v1"

    def __init__(self, image_size: int = 224, hist_bins: int = 8) -> None:
        self.image_size = image_size
        self.hist_bins = hist_bins
        self.embedding_dim = hist_bins * hist_bins * hist_bins + 16

    def encode(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0:
            raise ValueError("Cannot encode an empty image")
        resized = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [self.hist_bins, self.hist_bins, self.hist_bins],
            [0, 180, 0, 256, 0, 256],
        ).astype(np.float32)
        hist = hist.reshape(-1)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        cell_size = self.image_size // 4
        edge_features: list[float] = []
        for row in range(4):
            for col in range(4):
                cell = edges[
                    row * cell_size : (row + 1) * cell_size,
                    col * cell_size : (col + 1) * cell_size,
                ]
                edge_features.append(float(np.mean(cell) / 255.0))

        embedding = np.concatenate([hist, np.asarray(edge_features, dtype=np.float32)])
        return normalize_embedding(embedding)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = embedding.astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        return embedding
    return embedding / norm


class TorchFewShotEncoder:
    """Runtime encoder for checkpoints produced by scripts/colab_train_label_stack.py."""

    name = "torch_fewshot"

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import models, transforms

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.image_size = int(checkpoint.get("image_size", 224))
        self.embedding_dim = int(checkpoint.get("embedding_dim", 128))
        self.device = torch.device(device if device != "0" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        class EncoderNet(nn.Module):
            def __init__(self, embedding_dim: int) -> None:
                super().__init__()
                backbone = models.resnet50(weights=None)
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

        self.model = EncoderNet(self.embedding_dim).to(self.device)
        state = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

    def encode(self, image: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode(tensor).squeeze(0).detach().cpu().numpy()
        return normalize_embedding(embedding)
