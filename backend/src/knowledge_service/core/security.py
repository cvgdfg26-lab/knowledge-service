import hashlib
import secrets
from datetime import UTC, datetime

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

passwords = PasswordHasher()


def now():
    return datetime.now(UTC)


def uid(prefix: str):
    return f"{prefix}_{secrets.token_hex(12)}"


def digest(value: str | bytes):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def verify_password(encoded: str, plain: str):
    try:
        return passwords.verify(encoded, plain)
    except (VerificationError, InvalidHashError):
        return False
