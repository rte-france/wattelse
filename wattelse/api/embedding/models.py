# Pydantic class for FastAPI typing control
from pydantic import BaseModel


class InputText(BaseModel):
    text: str | list[str]
    show_progress_bar: bool
