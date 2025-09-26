from sqlalchemy.orm import Session

from . import model, schemas


def get_user_by_username(db: Session, username: str):
    return db.query(model.User).filter(model.User.username == username).first()


def create_user(db: Session, user: schemas.UserCreate, hashed_pw: str):
    db_user = model.User(
        username=user.username,
        email=user.email,
        password=hashed_pw,
        full_name=user.full_name,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
