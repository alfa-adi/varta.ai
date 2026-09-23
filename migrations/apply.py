"""Run all migrations in order. Usage: python -m migrations.apply"""
from migrations.m001_create_conversation_schema import up as _up_001

if __name__ == "__main__":
    _up_001()
    print("Migrations complete.")
