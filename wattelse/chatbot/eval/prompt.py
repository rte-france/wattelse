EVAL_LLM_PROMPT = (
    "You are an evaluator. You will be provided with a a query, "
    "the groundtruth answer and a candidate response. You must "
    "evaluate the candidate response based on the groundtruth "
    "answer and the query. You must provide one of the following "
    "scores:\n"
    "- 0: the candidate response is incorrect and contains wrong information\n"
    "- 1: the candidate response is not incorrect but miss important parts "
    "of the groundtruth answer\n"
    "- 2: the candidate response is correct but miss some details that "
    "do not impact the veracity of the information\n"
    "- 3: the candidate response is correct and provides all the "
    "information from the groundtruth answer\n"
    'You must answer with the score only, using the format "Score: {{0,1,2,3}}"\n\n'
    "Query: {query}\n"
    "Groundtruth answer: {answer}\n"
    "Candidate response: {candidate}\n"
)