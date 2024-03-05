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


def get_document_collections_for_group(group_id: str) -> List[str]:
    if group_id == "admin":
        return ["rh", "maintenance"]
    elif group_id == "rh":
        return ["drh"]
    elif group_id == "maintenance":
        return ["maintenance"]
    return ["drh"]


def get_document_collections_for_user(login: str) -> List[str]:
    return get_document_collections_for_group(get_group_for_user(login))


def get_all_document_collections() -> List[str]:
    return ["drh", "maintenance"]
