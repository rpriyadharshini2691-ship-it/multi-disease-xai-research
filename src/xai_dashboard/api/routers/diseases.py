"""Config-driven disease registry."""

from pathlib import Path

from fastapi import APIRouter, Depends

from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.core.data.loader import load_dataset
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/diseases", tags=["diseases"])


@router.get("")
def list_diseases(settings: Settings = Depends(get_settings)) -> list[dict[str, object]]:
    configs = load_disease_configs(settings.configs_dir)
    result: list[dict[str, object]] = []
    for config in configs.values():
        try:
            dataset = load_dataset(config, Path(config.dataset_path))
        except (OSError, ValueError):
            dataset = None
        features: list[dict[str, object]] = []
        for name in config.feature_columns:
            options: list[object] = []
            if dataset is not None and name in config.categorical_columns:
                values = dataset[name].dropna().unique().tolist()
                options = [value.item() if hasattr(value, "item") else value for value in values]
                options.sort(key=str)
            features.append(
                {
                    "id": name,
                    "name": config.display_names.get(name, name),
                    "unit": config.units.get(name),
                    "valid_range": config.valid_ranges.get(name),
                    "kind": "numeric" if name in config.numeric_columns else "categorical",
                    "options": options,
                }
            )
        result.append({"id": config.id, "name": config.name, "features": features})
    return result
