# How to enable or disable USER_MODE ?

1. Make a file_directory containing all documents the user will have access to (pdf)
2. Initialize a user directory using :
3. ```python initialize_user_RAG.py -i {file_directory} -u {user_name}```
4. This creates a user_directory containing parsed docs in $BASE_DATA_DIR/chatbot/user/{user_name}/docs/
5. Set parameters in wattelse/chatbot/config.cfg : ```USER_MODE=True, USER_NAME={user_name}```
6. Launch streamlit app as usual : ```streamlit run wattelse/chatbot/app.py```