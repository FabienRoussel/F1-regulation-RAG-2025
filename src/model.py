from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import TIMESTAMP, TEXT, INTEGER
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()
metadata = Base.metadata


class Regulation(Base):
    __tablename__ = 'regulations'

    id = Column(INTEGER, primary_key=True, autoincrement=True)
    title = Column(TEXT, nullable=False)
    content = Column(TEXT, nullable=False)
    embedding = Column(Vector(1024))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
