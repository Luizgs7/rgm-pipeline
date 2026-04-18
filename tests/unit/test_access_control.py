"""Testes do AccessControlService — RBAC, mascaramento e auditoria."""

import pytest

from rgm_pipeline.agents.data_engineer.access_control import (
    AccessControlService,
    AuthenticationError,
    User,
)


@pytest.fixture()
def acl() -> AccessControlService:
    return AccessControlService()


# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------

def test_authenticate_default_users(acl):
    admin = acl.authenticate("U001")
    assert admin.role == "admin"

    analyst = acl.authenticate("U002")
    assert analyst.role == "analyst"

    viewer = acl.authenticate("U003")
    assert viewer.role == "viewer"


def test_authenticate_unknown_user_raises(acl):
    with pytest.raises(AuthenticationError):
        acl.authenticate("UNKNOWN")


def test_register_and_authenticate_new_user(acl):
    user = User("U999", "novo_user", "analyst", "novo@rgm.com")
    acl.register_user(user)
    retrieved = acl.authenticate("U999")
    assert retrieved.username == "novo_user"


def test_register_duplicate_raises(acl):
    with pytest.raises(ValueError):
        acl.register_user(User("U001", "dup", "admin", "dup@rgm.com"))


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        User("U0", "x", "superadmin", "x@x.com")


# ---------------------------------------------------------------------------
# Autorização
# ---------------------------------------------------------------------------

def test_admin_can_read_all_tables(acl, sample_transactions):
    admin = acl.authenticate("U001")
    datasets = {"transactions": sample_transactions}
    df = acl.get_data(admin, "transactions", datasets)
    assert len(df) == len(sample_transactions)


def test_analyst_can_read_transactions(acl, sample_transactions):
    analyst = acl.authenticate("U002")
    df = acl.get_data(analyst, "transactions", {"transactions": sample_transactions})
    assert not df.empty


def test_viewer_cannot_read_transactions(acl, sample_transactions):
    viewer = acl.authenticate("U003")
    with pytest.raises(AuthenticationError):
        acl.get_data(viewer, "transactions", {"transactions": sample_transactions})


def test_viewer_can_read_campaigns(acl, sample_campaigns):
    viewer = acl.authenticate("U003")
    df = acl.get_data(viewer, "campaigns", {"campaigns": sample_campaigns})
    assert not df.empty


# ---------------------------------------------------------------------------
# Mascaramento
# ---------------------------------------------------------------------------

def test_viewer_budget_masked(acl, sample_campaigns):
    viewer = acl.authenticate("U003")
    df = acl.get_data(viewer, "campaigns", {"campaigns": sample_campaigns})
    assert (df["budget"] == "***").all()


def test_admin_budget_not_masked(acl, sample_campaigns):
    admin = acl.authenticate("U001")
    df = acl.get_data(admin, "campaigns", {"campaigns": sample_campaigns})
    assert df["budget"].dtype != object or (df["budget"] != "***").any()


def test_viewer_store_id_hashed(acl, sample_campaigns):
    viewer = acl.authenticate("U003")
    df = acl.get_data(viewer, "campaigns", {"campaigns": sample_campaigns})
    # Hash SHA256 truncado tem 8 caracteres hex
    assert df["store_id"].str.len().eq(8).all()


# ---------------------------------------------------------------------------
# Auditoria
# ---------------------------------------------------------------------------

def test_audit_log_populated(acl, sample_transactions):
    admin = acl.authenticate("U001")
    acl.get_data(admin, "transactions", {"transactions": sample_transactions})
    log = acl.get_audit_log(admin)
    assert len(log) > 0
    assert "username" in log.columns


def test_non_admin_cannot_read_audit_log(acl, sample_transactions):
    analyst = acl.authenticate("U002")
    with pytest.raises(AuthenticationError):
        acl.get_audit_log(analyst)


def test_get_permissions_returns_role_info(acl):
    admin = acl.authenticate("U001")
    perms = acl.get_permissions(admin)
    assert "write" in perms["operations"]
    assert perms["mask_pii"] is False
