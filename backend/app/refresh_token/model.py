import enum
from sqlalchemy import Column, ForeignKey, Enum, Integer, String, DateTime
from datetime import datetime
from backend.app.core.database import Base


class StatusEnum(enum.Enum):
    active = "ACTIVE"
    revoked = "REVOKED"
    expired = "EXPIRED"


class RefreshToken(Base):
    __tablename__ = "refresh_token"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    token = Column(String(255), nullable=False)
    user_agent = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    status = Column(Enum(StatusEnum), default=StatusEnum.active)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
