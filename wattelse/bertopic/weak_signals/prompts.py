def get_prompt(language, topic_number, content_summary):
    en_prompt = f"""
    Please provide a summary for the evolution of Topic {topic_number} over time based on the following information:
    
    {content_summary}
    
    For each timestamp, generate a title describing the content based on the topic representation and documents. Then provide a summary of the documents in bullet points.
    
    For each timestamp except the first one, also include a section titled "What's New?" that highlights the changes and new information in the topic compared to the previous timestamp.
    
    Base your summary solely on the provided information and do not include any external knowledge or assumptions.
    
    Format the output as follows:
    
    ## Title: [Generated title based on topic representation and documents]
    ### Date: [Timestamp]
    ### Summary : Paragraph summarizing the documents
    ---
    ## Title: [Generated title based on topic representation and documents]
    ### Date: [Timestamp]
    ### Summary : Paragraph summarizing the documents
    ### What's New?
    [Paragraph describing the changes and new information compared to 1st timestamp]
    ---
    ## Title: [Generated title based on topic representation and documents]
    ### Date: [Timestamp]
    ### Summary : Paragraph summarizing the documents
    ### What's New?
    [Paragraph describing the changes and new information compared to 2nd timestamp]
    ---
    ...
    """




    fr_prompt = f"""
    Veuillez fournir un résumé de l'évolution du Sujet {topic_number} au fil du temps en vous basant sur les informations suivantes:
    
    {content_summary}
    
    Pour chaque Date, générez un titre décrivant le contenu en fonction de la représentation du sujet et des documents. Ensuite, fournissez un résumé des documents sous forme de points.
    
    Pour chaque Date, à l'exception de la première, incluez également une section intitulée "Quelles sont les nouveautés ?" qui met en évidence les changements et les nouvelles informations dans le sujet par rapport à la date précédente.
    
    Basez votre résumé uniquement sur les informations fournies et n'incluez aucune connaissance ou hypothèse externe.
    
    Formatez la sortie comme suit:
    
    ## Titre: [Titre généré en fonction de la représentation du sujet et des documents]
    ### Date: [Date]
    ### Résumé
    - [Point 1]
    - [Point 2]
    - ...
    ---
    ## Titre: [Titre généré en fonction de la représentation du sujet et des documents]
    ### Date: [Date]
    ### Résumé
    - [Point 1]
    - [Point 2]
    - ...
    ### Quelles sont les nouveautés ?
    [Paragraphe décrivant les changements et les nouvelles informations par rapport à la 1ere Date]
    ---
    ## Titre: [Titre généré en fonction de la représentation du sujet et des documents]
    ### Date: [Date]
    ### Résumé
    - [Point 1]
    - [Point 2]
    - ...
    ### Quelles sont les nouveautés ?
    [Paragraphe décrivant les changements et les nouvelles informations par rapport à la 2ème Date]
    ---
    ...
    """

    if language == "English":
        return en_prompt
    elif language == "French":
        return fr_prompt
    else:
        raise ValueError(f"Unsupported language: {language}")