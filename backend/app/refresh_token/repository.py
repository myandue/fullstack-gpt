from sqlalchemy.orm import Session

from . import model, schemas


def save_refresh_token(db: Session, refresh_token: model.RefreshToken):
    db_refresh_token = model.RefreshToken(
        user_id=refresh_token.user_id,
        token=refresh_token.token,
        user_agent=refresh_token.user_agent,
        ip_address=refresh_token.ip_address,
        expires_at=refresh_token.expires_at,
    )
    db.add(db_refresh_token)
    db.commit()
    db.refresh(db_refresh_token)
    return db_refresh_token


def get_refresh_token_by_token(db: Session, refresh_token: str):
    return (
        db.query(model.RefreshToken)
        .filter(
            model.RefreshToken.token == refresh_token,
            model.RefreshToken.status == "active",
        )
        .first()
    )


def update_refresh_token(db: Session, token_id: int, status: str):
    token = (
        db.query(model.RefreshToken)
        .filter(model.RefreshToken.id == token_id)
        .first()
    )
    if token:
        token.status = status
        db.commit()
        db.refresh(token)
    return token
