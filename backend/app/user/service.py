from sqlalchemy.orm import Session

from backend.app.core.security import get_password_hash
from . import repository, schemas


def register_user(db: Session, user: schemas.UserCreate):
    db_user = repository.get_user_by_email(db, email=user.email)
    if db_user:
        raise ValueError("Email already registered")
    hashed_pw = get_password_hash(user.password)
    return repository.create_user(db, user, hashed_pw)


def get_user_by_username(db: Session, username: str):
    return repository.get_user_by_username(db, username)
