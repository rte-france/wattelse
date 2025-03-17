#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from pydantic import BaseModel


class RAGConfig(BaseModel):
    config: str | dict


class RAGQuery(BaseModel):
    message: str
    history: list[dict[str, str]] | None
    group_system_prompt: str | None
    selected_files: list[str] | None
    stream: bool = False
