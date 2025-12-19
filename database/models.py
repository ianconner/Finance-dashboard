# database/models.py

from sqlalchemy import Column, Integer, String, Float, Date, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base
from .connection import engine

Base = declarative_base()

class MonthlyUpdate(Base):
    __tablename__ = "monthly_updates"
    date = Column(Date, primary_key=True)
    person = Column(String, primary_key=True)
    account_type = Column(String, primary_key=True)
    value = Column(Float)
    __table_args__ = (PrimaryKeyConstraint('date', 'person', 'account_type'),)

class AccountConfig(Base):
    __tablename__ = "account_config"
    person = Column(String, primary_key=True)
    account_type = Column(String, primary_key=True)
    __table_args__ = (PrimaryKeyConstraint('person', 'account_type'),)

class RetirementGoal(Base):
    __tablename__ = "retirement_goal"
    id = Column(Integer, primary_key=True, autoincrement=True)
    target_amount = Column(Float, default=1000000.0)

class AIChat(Base):
    __tablename__ = "ai_chat"
    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String)
    content = Column(String)
    timestamp = Column(Date)

class PortfolioCSV(Base):
    __tablename__ = "portfolio_csv"
    id = Column(Integer, primary_key=True, autoincrement=True)
    csv_data = Column(String)
    uploaded_at = Column(Date)

# Create all tables if they don't exist
Base.metadata.create_all(engine)

# Ensure retirement goal row exists
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
with Session() as sess:
    if sess.query(RetirementGoal).first() is None:
        sess.add(RetirementGoal(target_amount=1000000.0))
        sess.commit()
