from pydantic import BaseModel


class RAGConfig(BaseModel):
    config: str | dict


class RAGQuery(BaseModel):
    message: str
    history: list[dict[str, str]] | None
    group_system_prompt: str | None
    selected_files: list[str] | None
    stream: bool = False
