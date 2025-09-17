from sqlalchemy.orm import Session

from backend.app.core.security import create_access_token, verify_password
from backend.app.user.repository import get_user_by_username
from . import schemas


def login_user(db: Session, user: schemas.UserLogin):
    auth_user = get_user_by_username(db, user.username)
    if not auth_user or not verify_password(user.password, auth_user.password):
        raise ValueError("Invalid credentials")
    access_token = create_access_token(data={"sub": auth_user.email})
    return {"access_token": access_token, "token_type": "bearer"}
