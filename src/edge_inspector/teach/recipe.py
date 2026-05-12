from __future__ import annotations

from pathlib import Path

import yaml

from edge_inspector.teach.schemas import ProductRecipe
from edge_inspector.utils.config import AppConfig
from edge_inspector.utils.time import utc_now


class RecipeStore:
    """Persist product recipes as small YAML files."""

    def __init__(self, config: AppConfig | None = None, root: str | Path | None = None) -> None:
        if root is not None:
            self.root = Path(root)
        elif config is not None:
            self.root = Path(config.get("teach.recipe_dir", "data/teach/recipes"))
        else:
            self.root = Path("data/teach/recipes")

    def list_recipes(self, active_only: bool = False) -> list[ProductRecipe]:
        if not self.root.exists():
            return []
        recipes: list[ProductRecipe] = []
        for path in sorted(self.root.glob("*.yaml")):
            recipe = self.load(path.stem)
            if active_only and not recipe.active:
                continue
            recipes.append(recipe)
        return recipes

    def load(self, recipe_id: str) -> ProductRecipe:
        path = self._path(recipe_id)
        if not path.exists():
            raise FileNotFoundError(f"Recipe not found: {recipe_id}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return ProductRecipe(**payload)

    def save(self, recipe: ProductRecipe) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        recipe = recipe.model_copy(update={"updated_at": utc_now()})
        path = self._path(recipe.recipe_id)
        path.write_text(yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
        return path

    def create(self, recipe_id: str, product_name: str) -> ProductRecipe:
        recipe = ProductRecipe(recipe_id=recipe_id, product_name=product_name)
        self.save(recipe)
        return recipe

    def _path(self, recipe_id: str) -> Path:
        normalized = ProductRecipe(recipe_id=recipe_id, product_name="placeholder").recipe_id
        return self.root / f"{normalized}.yaml"
