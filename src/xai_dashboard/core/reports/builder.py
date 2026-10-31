"""Build PDF reports only from persisted experiment artifacts."""

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_evaluation_report(disease_name: str, artifact: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"{disease_name} model evaluation", styles["Title"]), Spacer(1, 12)]
    story.append(
        Paragraph(
            "Generated from stored evaluation artifact: "
            f"{artifact.get('generated_at', 'timestamp unavailable')}",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Metrics are research evaluation results, not clinical performance claims.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph("This research and education tool is not a clinical device.", styles["BodyText"])
    )
    story.append(
        Paragraph(
            "Reliability results, where included, measure consistency under tested "
            "conditions only and do not imply medical correctness.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 16))
    table_data = [["Model", "Metric", "Value"]]
    for model_name, values in artifact.get("holdout_metrics", {}).items():
        for metric, value in values.items():
            table_data.append([model_name, metric, f"{float(value):.4f}"])
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#254c45")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d4d0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f2")]),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)
    document.build(story)
    return buffer.getvalue()
