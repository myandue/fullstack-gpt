from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    Cookie,
)
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import verify_token

from . import service, schemas

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()


@router.post("/login", response_model=schemas.Token)
def login(
    user: schemas.UserLogin,
    db: Session = Depends(get_db),
    request: Request = None,
    response: Response = None,
):
    try:
        user_agent = request.headers.get("user-agent")
        ip_address = request.client.host

        tokens = service.login(db, user, user_agent, ip_address)
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]

        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            secure=True,
            samesite="strict",
        )

        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/refresh", response_model=schemas.Token)
def refresh_token(
    db: Session = Depends(get_db),
    request: Request = None,
    refresh_token: str = Cookie(None),
    response: Response = None,
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    try:
        user_agent = request.headers.get("user-agent")
        ip_address = request.client.host

        access_token, new_refresh_token = service.regenerate_token(
            db, refresh_token, user_agent, ip_address
        )

        response.set_cookie(
            key="refresh_token",
            value=new_refresh_token,
            httponly=True,
            secure=True,
            samesite="strict",
        )

        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/logout")
def logout(
    db: Session = Depends(get_db),
    credentials=Depends(security),
    refresh_token: str = Cookie(None),
    response: Response = None,
):
    verify_token(credentials.credentials)

    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    try:
        service.logout(db, refresh_token)

        response.delete_cookie(key="refresh_token")

        return {"detail": "Logged out successfully"}
    except HTTPException as e:
        raise HTTPException(status_code=401, detail=str(e))
