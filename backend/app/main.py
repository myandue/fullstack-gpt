from fastapi import FastAPI

from backend.app.core.database import engine, Base

from backend.app.user.router import router as user_router
from backend.app.auth.router import router as auth_router

app = FastAPI(title="My API", version="1.0.0")

# 라우터 등록
app.include_router(user_router)
app.include_router(auth_router)

# DB 테이블 생성
Base.metadata.create_all(bind=engine)


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
