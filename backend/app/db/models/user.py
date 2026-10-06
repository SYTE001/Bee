"""User model."""

from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class User(Base, IdMixin, TimestampMixin):
    """A ClipForge user (simple e-mail auth + job quota)."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), default="", server_default="")
    is_active: Mapped[bool] = mapped_column(default=True, server_default="1")
    # usage quota
    jobs_today: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    jobs_quota_per_day: Mapped[int] = mapped_column(
        Integer, default=20, server_default="20"
    )
    quota_reset_date: Mapped[date | None] = mapped_column(Date, nullable=True)
