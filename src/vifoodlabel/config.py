"""Project paths and the VLM model registry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
CONFIGS_DIR = REPO_ROOT / "configs"
PROMPTS_DIR = REPO_ROOT / "prompts"
RESULTS_DIR = REPO_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
SCORED_RESULTS_DIR = RESULTS_DIR / "scored"
FIGURES_DIR = RESULTS_DIR / "figures"
PERTURBED_IMAGES_DIR = RESULTS_DIR / "perturbed_images"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    slug: str
    display_name: str
    group: str
    open_weight: bool
    price_input_per_m: float
    price_output_per_m: float
    base_url: str | None = None  # None => OPENROUTER_BASE_URL
    api_key_env: str | None = None  # None => OPENROUTER_API_KEY

    @property
    def slug_sanitized(self) -> str:
        """Filesystem-safe version of the slug (slugs may contain '/')."""
        return self.slug.replace("/", "__")

    @property
    def resolved_base_url(self) -> str:
        return self.base_url or OPENROUTER_BASE_URL

    @property
    def is_local(self) -> bool:
        return self.base_url is not None

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.price_input_per_m
            + output_tokens / 1_000_000 * self.price_output_per_m
        )


@lru_cache(maxsize=1)
def load_model_registry(path: Path | None = None) -> dict[str, ModelSpec]:
    """Load configs/models.yaml into a dict keyed by ModelSpec.key."""
    path = path or (CONFIGS_DIR / "models.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs = {}
    for entry in raw["models"]:
        spec = ModelSpec(
            key=entry["key"],
            slug=entry["slug"],
            display_name=entry["display_name"],
            group=entry["group"],
            open_weight=entry["open_weight"],
            price_input_per_m=float(entry["price_input_per_m"]),
            price_output_per_m=float(entry["price_output_per_m"]),
            base_url=entry.get("base_url"),
            api_key_env=entry.get("api_key_env"),
        )
        specs[spec.key] = spec
    return specs


def resolve_models(selectors: list[str] | None) -> list[ModelSpec]:
    """Resolve a list of CLI-provided model keys or slugs to ModelSpec objects.

    If `selectors` is None or empty, returns all registered models.
    Accepts either the short `key` (e.g. "gpt-5.4") or the full OpenRouter
    `slug` (e.g. "openai/gpt-5.4").
    """
    registry = load_model_registry()
    if not selectors:
        return list(registry.values())

    by_slug = {spec.slug: spec for spec in registry.values()}
    resolved = []
    for selector in selectors:
        if selector in registry:
            resolved.append(registry[selector])
        elif selector in by_slug:
            resolved.append(by_slug[selector])
        else:
            known = ", ".join(sorted(registry.keys()))
            raise ValueError(f"Unknown model '{selector}'. Known keys: {known}")
    return resolved
