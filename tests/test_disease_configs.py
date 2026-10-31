from pathlib import Path

from xai_dashboard.core.data.diseases import load_disease_configs


def test_disease_configs_are_unique_and_feature_columns_are_defined() -> None:
    configs = load_disease_configs(Path("configs"))
    assert set(configs) == {"diabetes", "heart", "ckd"}
    for config in configs.values():
        assert config.target_column
        assert config.feature_columns
        assert len(set(config.feature_columns)) == len(config.feature_columns)
