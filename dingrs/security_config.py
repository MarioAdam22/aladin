"""
Update #39: Data encryption & security best practices.
- Parole bcrypt hashed
- API keys din environment variables (nu în cod)
- Secrets management
"""
import os
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT VARIABLES — TEMPLATE
# =============================================================================
# Creează fișierul .env (NU îl pune în Git!) cu:
#   TELEGRAM_BOT_TOKEN=your_token
#   TELEGRAM_CHAT_ID=your_chat_id
#   FRED_API_KEY=your_fred_key
#   ALADIN_SMTP_USER=your@gmail.com
#   ALADIN_SMTP_PASS=your_app_password
#   ALADIN_SECRET_KEY=random_secret_for_sessions

ENV_TEMPLATE = """
# ALADIN — Environment Variables (NE CONFIDENȚIALE — nu commit în Git!)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
FRED_API_KEY=
ALADIN_SMTP_USER=
ALADIN_SMTP_PASS=
ALADIN_EMAIL_TO=marioyear@yahoo.com
ALADIN_SECRET_KEY=
""".strip()


def create_env_template(path: str = "/Users/mario/Desktop/Aladin/.env.template"):
    """Creează template pentru .env dacă nu există."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(ENV_TEMPLATE)
        print(f"✅ .env template creat: {path}")
        print("   Completează valorile și redenumește în .env")
    else:
        print(f"ℹ️  .env template deja există: {path}")


def load_env_file(path: str = "/Users/mario/Desktop/Aladin/.env"):
    """Încarcă variabile din .env în os.environ."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()
        logger.info(f"✅ .env încărcat din {path}")
        return True
    except Exception as e:
        logger.warning(f"Nu s-a putut încărca .env: {e}")
        return False


def hash_password(password: str) -> str:
    """Hash password cu bcrypt (sau fallback SHA-256)."""
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        salt = secrets.token_hex(16)
        return f"sha256${salt}${hashlib.sha256((salt + password).encode()).hexdigest()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verifică parolă față de hash."""
    try:
        import bcrypt
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ImportError:
        if hashed.startswith("sha256$"):
            _, salt, stored = hashed.split("$")
            return hashlib.sha256((salt + password).encode()).hexdigest() == stored
        return False


def get_secret(key: str, default: str = "") -> str:
    """Obține secret din environment (niciodată din cod hardcodat)."""
    val = os.environ.get(key, default)
    if not val:
        logger.debug(f"Secret '{key}' lipsă din environment")
    return val


if __name__ == "__main__":
    create_env_template()
    load_env_file()
    print("🔐 Security config OK")
