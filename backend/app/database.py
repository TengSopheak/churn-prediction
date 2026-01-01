from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import DATABASE_URL
from .models import Base

# Create SQLAlchemy engine using the DATABASE_URL (Neon Postgres)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ensure tables exist (lightweight auto-create)
Base.metadata.create_all(bind=engine)

def get_db():
    """FastAPI dependency that yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()