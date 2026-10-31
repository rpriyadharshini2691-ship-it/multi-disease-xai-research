"""Current profile and administrator-only account activity."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from xai_dashboard.api.deps import get_current_user, get_session, require_roles
from xai_dashboard.api.schemas import SCORE_DISCLAIMER
from xai_dashboard.db.models import PredictionLog, User

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
def profile(user: User = Depends(get_current_user)) -> dict[str, object]:
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "created_at": user.created_at,
    }


@router.get("/me/activity")
def personal_activity(
    user: User = Depends(get_current_user), db: Session = Depends(get_session)
) -> dict[str, object]:
    counts = db.execute(
        select(PredictionLog.disease_id, func.count(PredictionLog.id))
        .where(PredictionLog.user_id == user.id)
        .group_by(PredictionLog.disease_id)
    ).all()
    per_disease = {disease_id: count for disease_id, count in counts}
    return {"consented_prediction_count": sum(per_disease.values()), "by_disease": per_disease}


@router.get("")
def list_users(
    _admin: User = Depends(require_roles("admin")), db: Session = Depends(get_session)
) -> list[dict[str, object]]:
    users = db.scalars(select(User).order_by(User.created_at.desc())).all()
    result: list[dict[str, object]] = []
    for user in users:
        count = (
            db.scalar(select(func.count(PredictionLog.id)).where(PredictionLog.user_id == user.id))
            or 0
        )
        logs = db.scalars(
            select(PredictionLog)
            .where(PredictionLog.user_id == user.id)
            .order_by(PredictionLog.created_at.desc())
            .limit(10)
        ).all()
        result.append(
            {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "created_at": user.created_at,
                "prediction_count": count,
                "activity": [
                    {
                        "disease_id": log.disease_id,
                        "created_at": log.created_at,
                        "positive_class_score": log.positive_score,
                        "disclaimer": SCORE_DISCLAIMER,
                    }
                    for log in logs
                ],
            }
        )
    return result
