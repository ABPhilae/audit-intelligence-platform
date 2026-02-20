"""
Role-Based Access Control.

In a banking environment, not all documents should be accessible to all users.
An auditor on the HK team should not be able to query Paris team documents.

This module implements metadata-based access filtering:
- Each document is tagged with access groups at ingestion time
- Each query is filtered to only return documents the user can access

For this portfolio project, we simulate user roles with a simple
header-based system. In production, this would integrate with Active
Directory, LDAP, or an SSO provider.
"""
from fastapi import Request, HTTPException
import logging

logger = logging.getLogger(__name__)

# Simulated user roles and their document access permissions
USER_ROLES = {
    "admin": {
        "name": "Admin",
        "access_groups": ["GLOBAL_AUDIT", "APAC_AUDIT", "EMEA_AUDIT", "ALL"],
    },
    "apac_auditor": {
        "name": "APAC Auditor",
        "access_groups": ["APAC_AUDIT", "GLOBAL_AUDIT"],
    },
    "emea_auditor": {
        "name": "EMEA Auditor",
        "access_groups": ["EMEA_AUDIT", "GLOBAL_AUDIT"],
    },
    "viewer": {
        "name": "Viewer",
        "access_groups": ["GLOBAL_AUDIT"],
    },
}


def get_current_user(request: Request) -> dict:
    """
    Extract user role from request headers.

    In production, this would validate a JWT token or session cookie.
    For the portfolio demo, we use a simple X-User-Role header.
    """
    role = request.headers.get("X-User-Role", "admin")

    user = USER_ROLES.get(role)
    if not user:
        raise HTTPException(status_code=403, detail=f"Unknown role: {role}")

    return {"role": role, **user}


def build_access_filter(user: dict) -> dict:
    """
    Build a Qdrant metadata filter based on user permissions.

    This filter is passed to the retriever so it only returns
    documents the user is authorised to see.
    """
    access_groups = user.get("access_groups", [])

    if "ALL" in access_groups:
        return {}  # Admin sees everything

    return {
        "must": [{
            "key": "access_group",
            "match": {"any": access_groups},
        }]
    }
