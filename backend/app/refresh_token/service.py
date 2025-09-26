import secrets
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from . import model, repository

REFRESH_TOKEN_EXPIRE_DAYS = 7


def create_refresh_token(
    db: Session,
    user_id,
    user_agent,
    ip_address,
    expires_delta: timedelta = None,
):
    print(f"user_id: {user_id}")
    refresh_token = secrets.token_urlsafe(32)
    expire = datetime.utcnow() + (
        expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    return repository.save_refresh_token(
        db,
        model.RefreshToken(
            user_id=user_id,
            token=refresh_token,
            user_agent=user_agent,
            ip_address=ip_address,
            expires_at=expire,
        ),
    ).token


def verify_refresh_token(
    db: Session, refresh_token: str, user_agent: str, ip_address: str
):
    refresh_token = repository.get_refresh_token_by_token(db, refresh_token)

    if not refresh_token:
        raise ValueError("Invalid refresh token")
    elif refresh_token.user_agent != user_agent:
        raise ValueError("User agent does not match")
    elif refresh_token.ip_address != ip_address:
        raise ValueError("IP address does not match")
    elif refresh_token.expires_at < datetime.utcnow():
        repository.update_refresh_token(db, refresh_token.id, "expired")
        raise ValueError("Refresh token has expired")
    else:
        return refresh_token


def regenerate_refresh_token(db: Session, refresh_token: model.RefreshToken):
    revoke_refresh_token(db, token_id=refresh_token.id)
    new_refresh_token = create_refresh_token(
        db,
        {
            "user_id": refresh_token.user_id,
            "user_agent": refresh_token.user_agent,
            "ip_address": refresh_token.ip_address,
        },
    )
    return new_refresh_token


def revoke_refresh_token(db: Session, token_id: int = None, token: str = None):
    if token_id:
        repository.update_refresh_token(db, token_id, "revoked")
    elif token:
        refresh_token = repository.get_refresh_token_by_token(db, token)
        if refresh_token:
            repository.update_refresh_token(db, refresh_token.id, "revoked")
