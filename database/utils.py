# database/utils.py
import os
from sqlalchemy import create_engine, text # pip install sqlalchemy
from sqlalchemy.orm import sessionmaker # pip install psycopg2-binary
from sqlalchemy.sql import text # pip install sqlalchemy
# from dotenv import load_dotenv # pip install python-dotenv

# load_dotenv()

DATABASE_URI = "postgresql://postgres:Super123@10.2.7.107:5432/cburn"
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)

def get_session():
    return Session()


def init_db():
    from database.models import Base
    # print("Creating schema...")
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS hw1"))
        conn.commit()
    # print("Creating tables...")
    Base.metadata.create_all(engine)
    # print("Database initialization completed")

