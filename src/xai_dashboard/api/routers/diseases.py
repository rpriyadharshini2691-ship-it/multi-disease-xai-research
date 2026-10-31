"""Config-driven disease registry."""

from fastapi import APIRouter, Depends

from xai_dashboard.core.data.diseases import load_disease_configs
from xai_dashboard.settings import Settings, get_settings

router = APIRouter(prefix="/diseases", tags=["diseases"])


@router.get("")
def list_diseases(settings: Settings = Depends(get_settings)) -> list[dict[str, object]]:
    configs = load_disease_configs(settings.configs_dir)
    return [
        {
            "id": config.id,
            "name": config.name,
            "features": [
                {
                    "id": name,
                    "name": config.display_names.get(name, name),
                    "unit": config.units.get(name),
                    "valid_range": config.valid_ranges.get(name),
                    "kind": "numeric" if name in config.numeric_columns else "categorical",
                }
                for name in config.feature_columns
            ],
        }
        for config in configs.values()
    ]
