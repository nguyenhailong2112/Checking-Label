from pathlib import Path

import numpy as np

from edge_inspector.teach.dataset import TeachDatasetWriter, xyxy_to_yolo, yolo_to_xyxy
from edge_inspector.teach.recipe import RecipeStore
from edge_inspector.teach.schemas import ApprovedAnnotation, ProductRecipe


def test_xyxy_to_yolo_round_trip() -> None:
    box = (10, 20, 50, 80)
    yolo_box = xyxy_to_yolo(box, image_width=100, image_height=100)

    assert yolo_box == (0.3, 0.5, 0.4, 0.6)
    assert yolo_to_xyxy(yolo_box, image_width=100, image_height=100) == box


def test_recipe_store_round_trip(tmp_path: Path) -> None:
    store = RecipeStore(root=tmp_path / "recipes")
    recipe = ProductRecipe(recipe_id="SanDisk Ver2", product_name="SanDisk Ver2")

    path = store.save(recipe)
    loaded = store.load("sandisk_ver2")

    assert path.exists()
    assert loaded.recipe_id == "sandisk_ver2"
    assert loaded.product_name == "SanDisk Ver2"


def test_teach_dataset_writer_exports_yolo_sample(tmp_path: Path) -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    writer = TeachDatasetWriter(root=tmp_path / "teach")
    annotation = ApprovedAnnotation(
        target="label",
        class_name="Label",
        xyxy=(20, 10, 120, 60),
    )

    sample = writer.save_sample(
        image=image,
        image_name="sample.jpg",
        recipe_id="default_recipe",
        target="label",
        annotations=[annotation],
        class_names=["Label"],
    )

    image_path = Path(sample.image_path)
    label_path = image_path.parents[2] / "labels" / "train" / f"{sample.sample_id}.txt"
    dataset_yaml = image_path.parents[2] / "dataset.yaml"
    metadata = tmp_path / "teach" / "sessions" / "default_recipe" / "metadata" / f"{sample.sample_id}.json"

    assert image_path.exists()
    assert label_path.exists()
    assert label_path.read_text(encoding="utf-8").startswith("0 0.350000 0.350000 0.500000 0.500000")
    assert dataset_yaml.exists()
    assert metadata.exists()
