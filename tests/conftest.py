import os

# Set required env vars before any app modules are imported (config.py validates at import time)
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/test")
os.environ.setdefault("KEYCLOAK_ISSUER", "https://keycloak.example.com/realms/test")
os.environ.setdefault("KEYCLOAK_JWKS_URL", "https://keycloak.example.com/realms/test/protocol/openid-connect/certs")
