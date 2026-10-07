from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from database import Base

class PromptHistory(Base):
    __tablename__ = "prompt_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    temperature = Column(String(50), nullable=True)
    top_k = Column(Integer, nullable=True)
    top_p = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelConfig:
    BLOCK_SIZE = 128