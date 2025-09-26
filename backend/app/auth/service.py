from sqlalchemy.orm import Session

from backend.app.core.security import create_access_token, verify_password
from backend.app.refresh_token.service import create_refresh_token
from backend.app.user.service import get_user_by_username
from backend.app.refresh_token.service import (
    verify_refresh_token,
    regenerate_refresh_token,
    revoke_refresh_token,
)

from . import schemas


def login(
    db: Session, user: schemas.UserLogin, user_agent: str, ip_address: str
):
    auth_user = get_user_by_username(db, user.username)
    if not auth_user or not verify_password(user.password, auth_user.password):
        raise ValueError("Invalid credentials")
    access_token = create_access_token(data={"user_id": auth_user.id})
    refresh_token = create_refresh_token(
        db,
        auth_user.id,
        user_agent,
        ip_address,
    )
    return {"access_token": access_token, "refresh_token": refresh_token}


def logout(db: Session, refresh_token: str):
    revoke_refresh_token(db, token=refresh_token)


def regenerate_token(
    db: Session, refresh_token: str, user_agent: str, ip_address: str
):
    refresh_token = verify_refresh_token(
        db, refresh_token, user_agent, ip_address
    )
    new_refresh_token = regenerate_refresh_token(refresh_token)
    access_token = create_access_token(data={"user_id": refresh_token.user_id})
    return {"access_token": access_token, "refresh_token": new_refresh_token}
