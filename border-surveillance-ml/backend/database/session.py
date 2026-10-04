"""Database engine and session setup. SQLite for dev/MVP -- change
DATABASE_URL to a postgresql:// connection string for production
central-tier deployment, nothing else in the codebase needs to change.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models.db_models import Base

DATABASE_URL = os.environ.get("IBVAP_DATABASE_URL", "sqlite:///./configs/ibvap_backend.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    os.makedirs("configs", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
