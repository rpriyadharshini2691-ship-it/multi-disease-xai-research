import pytest

from xai_dashboard.core.auth.security import PUBLIC_ROLES, hash_password, verify_password


def test_password_hash_roundtrip_and_public_role_allowlist() -> None:
    encoded = hash_password("research-pass-123")
    assert verify_password("research-pass-123", encoded)
    assert not verify_password("wrong-password", encoded)
    assert PUBLIC_ROLES == {"researcher", "clinician"}


def test_password_minimum_length() -> None:
    with pytest.raises(ValueError):
        hash_password("short")
