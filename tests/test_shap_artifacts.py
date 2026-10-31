import base64

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from xai_dashboard.core.xai.artifacts import generate_global_explanations
from xai_dashboard.core.xai.shap_service import explain_instance


def test_local_additivity_and_global_artifacts(tmp_path) -> None:
    frame = pd.DataFrame(
        {"signal": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], "context": [1, 0, 1, 0, 1, 0]}
    )
    target = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression().fit(frame, target)

    local = explain_instance(model, frame.iloc[:4], frame.iloc[[4]])
    assert local["additivity_passed"] is True
    assert abs(local["additivity_sum"] - local["positive_class_score"]) < 1e-4
    assert base64.b64decode(local["waterfall_png_base64"]).startswith(b"\x89PNG")

    result = generate_global_explanations(model, frame, tmp_path, sample_size=6)
    assert (tmp_path / result["importance_csv"]).is_file()
    assert (tmp_path / result["beeswarm_png"]).read_bytes().startswith(b"\x89PNG")
