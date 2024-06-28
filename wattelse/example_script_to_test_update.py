from wattelse.chatbot.backend.rag_backend import RAGBackEnd
from fastapi import UploadFile

rag = RAGBackEnd("group_test")
filepath = "file_path"

with open(filepath, "rb") as f:
	upf = UploadFile(f, filename="file_name")
	rag.add_file_to_collection(upf)

result = rag.query_rag("Question to ask")
print(result["answer"], "\n\n", "-"*50)
print(result["relevant_extracts"])

rag.handle_user_feedback()
rag.update_extract_with_wrong_info()

result2 = rag.query_rag("Same question asked again")
print(result2["answer"], "\n\n")
print(result2["relevant_extracts"])