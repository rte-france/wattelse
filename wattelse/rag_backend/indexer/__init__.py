#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
from langchain_text_splitters import Language

CODE_EXTENSIONS = [
    ".py",
    ".cpp",
    ".java",
    ".kotlin",
    ".js",
    ".php",
    ".ts",
    ".c",
    ".sql",
    ".latex",
    ".tex",
]

CODE_MAPPING = {
    ".py": Language.PYTHON,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".java": Language.JAVA,
    ".kotlin": Language.KOTLIN,
    ".js": Language.JS,
    ".php": Language.PHP,
    ".ts": Language.TS,
    ".proto": Language.PROTO,
    ".python": Language.PYTHON,
    ".tex": Language.LATEX,
    ".latex": Language.LATEX,
    ".sql": "sql",
}

CFG_EXTENSIONS = [".toml", ".cfg"]
