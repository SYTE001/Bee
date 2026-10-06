"""SQLAlchemy declarative base and mixins."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Naive UTC now (SQLite-friendly)."""
    return datetime.utcnow()


def new_uuid() -> str:
    """Return a random UUID4 string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base for all models."""


class IdMixin:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )
