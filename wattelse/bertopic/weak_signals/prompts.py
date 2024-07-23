# def get_prompt(language, topic_number, content_summary):
#     en_prompt = f"""
#     Please provide a summary for the evolution of Topic {topic_number} over time based on the following information:
    
#     {content_summary}
    
#     For each timestamp, generate a title describing the content based on the topic representation and documents. Then provide a summary of the documents in bullet points.
    
#     For each timestamp except the first one, also include a section titled "What's New?" that highlights the changes and new information in the topic compared to the previous timestamp.
    
#     Base your summary solely on the provided information and do not include any external knowledge or assumptions.
    
#     Format the output as follows:
    
#     ## Title: [Generated title based on topic representation and documents]
#     ### Date: [Timestamp]
#     ### Summary : Paragraph summarizing the documents
#     ---
#     ## Title: [Generated title based on topic representation and documents]
#     ### Date: [Timestamp]
#     ### Summary : Paragraph summarizing the documents
#     ### What's New?
#     [Paragraph describing the changes and new information compared to 1st timestamp]
#     ---
#     ## Title: [Generated title based on topic representation and documents]
#     ### Date: [Timestamp]
#     ### Summary : Paragraph summarizing the documents
#     ### What's New?
#     [Paragraph describing the changes and new information compared to 2nd timestamp]
#     ---
#     ...
#     """




#     fr_prompt = f"""
#     Veuillez fournir un résumé de l'évolution du Sujet {topic_number} au fil du temps en vous basant sur les informations suivantes:
    
#     {content_summary}
    
#     Pour chaque Date, générez un titre décrivant le contenu en fonction de la représentation du sujet et des documents. Ensuite, fournissez un résumé des documents sous forme de points.
    
#     Pour chaque Date, à l'exception de la première, incluez également une section intitulée "Quelles sont les nouveautés ?" qui met en évidence les changements et les nouvelles informations dans le sujet par rapport à la date précédente.
    
#     Basez votre résumé uniquement sur les informations fournies et n'incluez aucune connaissance ou hypothèse externe.
    
#     Formatez la sortie comme suit:
    
#     ## Titre: [Titre généré en fonction de la représentation du sujet et des documents]
#     ### Date: [Date]
#     ### Résumé
#     - [Point 1]
#     - [Point 2]
#     - ...
#     ---
#     ## Titre: [Titre généré en fonction de la représentation du sujet et des documents]
#     ### Date: [Date]
#     ### Résumé
#     - [Point 1]
#     - [Point 2]
#     - ...
#     ### Quelles sont les nouveautés ?
#     [Paragraphe décrivant les changements et les nouvelles informations par rapport à la 1ere Date]
#     ---
#     ## Titre: [Titre généré en fonction de la représentation du sujet et des documents]
#     ### Date: [Date]
#     ### Résumé
#     - [Point 1]
#     - [Point 2]
#     - ...
#     ### Quelles sont les nouveautés ?
#     [Paragraphe décrivant les changements et les nouvelles informations par rapport à la 2ème Date]
#     ---
#     ...
#     """

#     if language == "English":
#         return en_prompt
#     elif language == "French":
#         return fr_prompt
#     else:
#         raise ValueError(f"Unsupported language: {language}")


def get_prompt(language, prompt_type, topic_number=None, content_summary=None, summary_from_first_prompt=None):
    if language == "English":
        if prompt_type == "topic_summary":
            return f"""
            Analyze the evolution of Topic {topic_number} based solely on the information provided below:

            {content_summary}

            For each period, structure your response as follows:

            ## [Concise and impactful title reflecting the essence of the topic for this period]
            ### Date: [Date]
            ### Summary
            [List of key points summarizing the content of the topic for this period]

            ### Evolution from previous date (except for the first period)
            [2-3 sentences describing significant changes compared to the previous period]

            ---

            After analyzing all periods, conclude with:

            ## Conclusion
            [3-4 sentences summarizing the overall evolution of the topic]

            Important guidelines:
            1. Base your analysis solely on the provided information, without adding external knowledge.
            2. Be concise but precise in your summaries and analyses.
            3. Highlight important changes and trends observed over time.
            """
        elif prompt_type == "weak_signal":
            return f"""
            You are an expert analyst with extensive knowledge across various domains and a keen ability to foresee potential future developments. Your task is to analyze the following summary of Topic {topic_number} and determine if it represents a weak signal:

            {summary_from_first_prompt}

            Guidelines:
            1. A weak signal is an early indicator of a potentially significant change that is not yet widely recognized. Examples include emerging technologies, subtle societal shifts, or nascent security threats.
            2. Use your broad knowledge and foresight to evaluate the topic's potential to become a major issue.
            3. Distinguish between weak signals and ephemeral trends or popular topics with no long-term significance.
            4. While basing your primary analysis on the provided summary, feel free to draw connections to your wider knowledge base.
            5. Be objective in your analysis, but don't hesitate to make informed predictions based on your expertise.

            Provide your analysis strictly according to the following format:

            ## Weak Signal Analysis

            ### Score: [0-10]

            ### Justification
            [3-4 sentences]

            ### Potential Impact
            [2-3 sentences]

            ### Recommendations
            [2-3 recommendations]
            """

    elif language == "French":
        if prompt_type == "topic_summary":
            return f"""
            Analysez l'évolution du Sujet {topic_number} en vous basant uniquement sur les informations fournies ci-dessous :

            {content_summary}

            Pour chaque période, structurez votre réponse comme suit :

            ## [Titre concis et percutant reflétant l'essence du sujet à cette période]
            ### Date : [Date]
            ### Résumé
            [Liste de points clés résumant le contenu du sujet pour cette période]

            ### Évolution par rapport à la date précédente (sauf pour la première période)
            [2-3 phrases décrivant les changements significatifs par rapport à la période précédente]

            ---

            Après avoir analysé toutes les périodes, concluez avec :

            ## Conclusion
            [3-4 phrases résumant l'évolution globale du sujet]

            Consignes importantes :
            1. Basez votre analyse uniquement sur les informations fournies, sans ajouter de connaissances externes.
            2. Soyez concis mais précis dans vos résumés et analyses.
            3. Mettez en évidence les changements importants et les tendances observées au fil du temps.
            """
        elif prompt_type == "weak_signal":
            return f"""
            Vous êtes un analyste expert doté d'une vaste connaissance dans divers domaines et d'une capacité aiguë à anticiper les développements futurs potentiels. Votre tâche est d'analyser le résumé suivant du Sujet {topic_number} et de déterminer s'il représente un signal faible :

            {summary_from_first_prompt}

            Consignes :
            1. Un signal faible est un indicateur précoce d'un changement potentiellement important, mais pas encore largement reconnu. Exemples : technologies émergentes, changements sociétaux subtils, menaces de sécurité naissantes.
            2. Utilisez votre large connaissance et votre capacité d'anticipation pour évaluer le potentiel du sujet à devenir un enjeu majeur.
            3. Distinguez les signaux faibles des tendances éphémères ou des sujets populaires sans importance à long terme.
            4. Tout en basant votre analyse principale sur le résumé fourni, n'hésitez pas à établir des liens avec votre base de connaissances plus large.
            5. Soyez objectif dans votre analyse, mais n'hésitez pas à faire des prédictions éclairées basées sur votre expertise.

            Fournissez votre analyse strictement selon le format suivant :

            ## Analyse du signal

            ### Score : [0-10]

            ### Justification
            [3-4 phrases]

            ### Potentiel d'impact
            [2-3 phrases]

            ### Recommandations
            [2-3 recommandations]
            """
    else:
        raise ValueError(f"Unsupported language: {language}")