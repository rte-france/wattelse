#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from wattelse.chatbot.backend.rag_backend import RAGBackend

# Global session storage
RAG_SESSIONS: dict[str, RAGBackend] = {}
