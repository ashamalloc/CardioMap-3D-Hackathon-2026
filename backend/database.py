"""MySQL tables (SQLAlchemy). Mirrors ../database/schema.sql."""
import re
from datetime import datetime

from sqlalchemy import (JSON, BigInteger, Column, DateTime, Float, Integer,
                        String, create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=1800)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def col_name(feature: str) -> str:
    """'Current Smoker' -> 'current_smoker', 'EF-TTE' -> 'ef_tte'."""
    return re.sub(r"[^0-9a-z]+", "_", feature.lower()).strip("_")


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    input_features = Column(JSON, nullable=False)
    cad_probability = Column(Float, nullable=False)
    lad_probability = Column(Float, nullable=False)
    lcx_probability = Column(Float, nullable=False)
    rca_probability = Column(Float, nullable=False)
    highest_vessel = Column(String(3), nullable=False)
    client_ip = Column(String(45))


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    target = Column(String(8), primary_key=True)          # CAD / LAD / LCX / RCA
    label = Column(String(64), nullable=False)
    n_train = Column(Integer, nullable=False)
    n_test = Column(Integer, nullable=False)
    positive_rate = Column(Float)
    cv_auc_train = Column(Float)
    test_auc = Column(Float)
    test_accuracy = Column(Float)
    test_precision = Column(Float)
    test_recall = Column(Float)
    test_specificity = Column(Float)
    test_f1 = Column(Float)
    tn = Column(Integer)
    fp = Column(Integer)
    fn = Column(Integer)
    tp = Column(Integer)
    best_params = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FeatureImportance(Base):
    __tablename__ = "feature_importance"
    id = Column(Integer, primary_key=True, autoincrement=True)
    target = Column(String(8), nullable=False, index=True)
    feature = Column(String(64), nullable=False)
    importance = Column(Float, nullable=False)
    rank_no = Column(Integer, nullable=False)
