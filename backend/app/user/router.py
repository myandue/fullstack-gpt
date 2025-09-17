from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db

from . import service, schemas

router = APIRouter(prefix="/user", tags=["user"])


@router.post("/sign_up", response_model=schemas.UserRead)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    try:
        return service.register_user(db, user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
