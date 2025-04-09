#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from typing import List

from langchain_core.documents import Document


# these functions are used to concatenate parent-headers to the beginning of each subsection


def get_html_hierarchy(docs: List[Document]):
    processed_documents = []
    buffer_content = ""

    for i in range(len(docs)):
        current_doc = docs[i]
        if current_doc.metadata:
            header = list(docs[i].metadata.keys())[0]
            header_n = int(header.split(" ")[1])

            if header_n > 2:
                buffer_content = ""
                current_header = header_n
                a = 1

                while current_header > 2 and (i - a) >= 0:
                    previous_header = list(docs[i - a].metadata.keys())[0]
                    previous_header_n = int(previous_header.split(" ")[1])

                    if previous_header_n < current_header:
                        buffer_content = (
                            docs[i - a].metadata.get(previous_header)
                            + " / "
                            + buffer_content
                        )
                        a += 1
                        current_header = previous_header_n

                    else:
                        a += 1

                current_doc.page_content = buffer_content + current_doc.page_content
                buffer_content = ""

        processed_documents.append(current_doc)
    return processed_documents


def get_markdown_hierarchy(docs: List[Document]):
    processed_documents = []

    for i in range(len(docs)):
        current_doc = docs[i]
        if current_doc.metadata:
            headers = ""
            for header in list(current_doc.metadata.values()):
                headers += header + "/"  # the separator can be changed

            current_doc.page_content = headers + current_doc.page_content
        processed_documents.append(current_doc)

    return processed_documents
