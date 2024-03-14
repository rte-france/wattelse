from typing import List


# FIXME: this is just a test!!! functions have to be properly implemented!

def get_group_for_user(login: str) -> str:
    """
    Returns the group of the given user identified by the given login
    """
    if login in ["admin", "jerome", "guillaume"]:
        return "admin"
    if login == "bob":
        return "maintenance"
    if login == "alice":
        return "rh"

    return "rh"


def get_document_collection_for_group(group_id: str) -> str:
    """We assume that a group has access to one and only one document collection"""
    if group_id == "admin":
        return "drh"
    elif group_id == "rh":
        return "drh"
    elif group_id == "maintenance":
        return "maintenance"
    return "drh"


def get_document_collection_for_user(login: str) -> str:
    """We assume that a user has access to one and only one document"""
    return get_document_collection_for_group(get_group_for_user(login))


def get_all_document_collections() -> List[str]:
    return ["drh", "maintenance"]
