# database/operations.py

import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert
from .models import (
    MonthlyUpdate, AccountConfig, RetirementGoal,
    AIChat, PortfolioCSV
)
from .connection import engine

Session = sessionmaker(bind=engine)

def get_session():
    return Session()

def reset_database():
    sess = get_session()
    try:
        from .models import Base
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        
        defaults = {
            'Sean': ['IRA', 'Roth IRA', 'TSP', 'Personal', 'T3W'],
            'Kim': ['Retirement', 'IRA', 'Roth IRA'],
            'Taylor': ['Personal']
        }
        for person, types in defaults.items():
            for acct_type in types:
                sess.merge(AccountConfig(person=person, account_type=acct_type))
        sess.add(RetirementGoal(target_amount=1000000.0))
        sess.commit()
    except Exception as e:
        sess.rollback()
        raise
    finally:
        sess.close()

def load_accounts():
    sess = get_session()
    try:
        cfg = sess.query(AccountConfig).all()
        accounts = {}
        for row in cfg:
            accounts.setdefault(row.person, []).append(row.account_type)
    finally:
        sess.close()
    
    if not accounts:
        reset_database()
        return load_accounts()
    return accounts

def add_monthly_update(date, person, acc_type, value):
    if value < 0:
        raise ValueError("Value cannot be negative")
    
    sess = get_session()
    try:
        stmt = pg_insert(MonthlyUpdate).values(
            date=date, person=person, account_type=acc_type, value=float(value)
        ).on_conflict_do_update(
            index_elements=['date', 'person', 'account_type'],
            set_={'value': float(value)}
        )
        sess.execute(stmt)
        sess.commit()
    except Exception as e:
        sess.rollback()
        raise
    finally:
        sess.close()

def get_monthly_updates():
    sess = get_session()
    try:
        rows = sess.query(MonthlyUpdate).all()
        return pd.DataFrame([
            {'date': r.date, 'person': r.person,
             'account_type': r.account_type, 'value': r.value}
            for r in rows
        ])
    finally:
        sess.close()

def get_retirement_goal():
    sess = get_session()
    try:
        goal = sess.query(RetirementGoal).first()
        return goal.target_amount if goal else 1000000.0
    finally:
        sess.close()

def set_retirement_goal(amount):
    if amount <= 0:
        raise ValueError("Goal must be positive")
    
    sess = get_session()
    try:
        goal = sess.query(RetirementGoal).first()
        if goal:
            goal.target_amount = float(amount)
        else:
            sess.add(RetirementGoal(target_amount=float(amount)))
        sess.commit()
    except Exception as e:
        sess.rollback()
        raise
    finally:
        sess.close()

def save_ai_message(role, content):
    sess = get_session()
    try:
        db_role = "model" if role == "assistant" else role
        sess.add(AIChat(role=db_role, content=content))
        sess.commit()
    except Exception as e:
        sess.rollback()
        raise
    finally:
        sess.close()

def load_ai_history():
    sess = get_session()
    try:
        rows = sess.query(AIChat).order_by(AIChat.id).all()
        return [{"role": r.role, "content": r.content} for r in rows]
    finally:
        sess.close()

def save_portfolio_csv_slot(slot: int, csv_b64: str):
    """Save portfolio CSV to specific slot (1, 2, or 3)"""
    if slot not in [1, 2, 3]:
        raise ValueError("Slot must be 1, 2, or 3")
    
    sess = get_session()
    try:
        # Delete existing in this slot
        sess.query(PortfolioCSV).filter(PortfolioCSV.id == slot).delete()
        sess.add(PortfolioCSV(id=slot, csv_data=csv_b64))
        sess.commit()
    except Exception as e:
        sess.rollback()
        raise
    finally:
        sess.close()

def load_portfolio_csv_slot(slot: int):
    """Load portfolio CSV from specific slot"""
    if slot not in [1, 2, 3]:
        return None
    
    sess = get_session()
    try:
        result = sess.query(PortfolioCSV).filter(PortfolioCSV.id == slot).first()
        return result.csv_data if result else None
    finally:
        sess.close()

def load_all_portfolios():
    """Return dict of all saved portfolios {slot: b64_data}"""
    portfolios = {}
    for slot in [1, 2, 3]:
        data = load_portfolio_csv_slot(slot)
        if data:
            portfolios[slot] = data
    return portfolios

def clear_all_portfolios():
    """Clear all portfolio slots"""
    sess = get_session()
    try:
        sess.query(PortfolioCSV).delete()
        sess.commit()
    except Exception as e:
        sess.rollback()
        raise
    finally:
        sess.close()

# Legacy functions for backwards compatibility
def save_portfolio_csv(csv_b64):
    """Save to slot 1 (backwards compatibility)"""
    save_portfolio_csv_slot(1, csv_b64)

def load_portfolio_csv():
    """Load from slot 1 (backwards compatibility)"""
    return load_portfolio_csv_slot(1)
