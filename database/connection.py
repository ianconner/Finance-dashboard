# database/connection.py

import streamlit as st
from sqlalchemy import create_engine, text

# Global variables for connection status
engine = None
DB_AVAILABLE = True
DB_ERROR = None

def create_db_engine():
    """Create and test connection to Neon PostgreSQL"""
    global engine, DB_AVAILABLE, DB_ERROR
    
    try:
        url = st.secrets["postgres_url"]
        
        # Standardize URL format
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        
        # Ensure SSL is required for Neon
        if "sslmode" not in url:
            url += ("&" if "?" in url else "?") + "sslmode=require"
        
        engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        DB_AVAILABLE = True
        DB_ERROR = None
        
    except Exception as e:
        engine = None
        DB_AVAILABLE = False
        DB_ERROR = str(e)

# Attempt to create engine on import
create_db_engine()

# Export for use in other modules
__all__ = ["engine", "DB_AVAILABLE", "DB_ERROR"]
