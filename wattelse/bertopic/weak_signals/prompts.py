#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

# Global variables for prompts
SIGNAL_INTRO = {
    'en': """As an elite strategic foresight analyst with extensive expertise across multiple domains and industries, your task is to conduct a comprehensive evaluation of a potential signal derived from the following topic summary:

{summary_from_first_prompt}

Leverage your knowledge and analytical skills to provide an in-depth analysis of this signal's potential impact and evolution:
""",
    'fr': """En tant qu'analyste de prospective stratégique d'élite avec une expertise étendue dans de multiples domaines et industries, votre tâche est de mener une évaluation complète d'un signal potentiel dérivé du résumé de sujet suivant :

{summary_from_first_prompt}

Utilisez vos connaissances et compétences analytiques pour fournir une analyse approfondie de l'impact potentiel et de l'évolution de ce signal :
"""
}

SIGNAL_INSTRUCTIONS = {
    'en': """
1. Potential Impact Analysis:
   - Examine the potential effects of this signal on various sectors, industries, and societal aspects.
   - Consider both short-term and long-term implications.
   - Analyze possible ripple effects and second-order consequences.

2. Evolution Scenarios:
   - Describe potential ways this signal could develop or manifest in the future.
   - Consider various factors that could influence its trajectory.
   - Explore both optimistic and pessimistic scenarios.

3. Interconnections and Synergies:
   - Identify how this signal might interact with other current trends or emerging phenomena.
   - Discuss potential synergies or conflicts with existing systems or paradigms.

4. Drivers and Inhibitors:
   - Analyze factors that could accelerate or amplify this signal.
   - Examine potential barriers or resistances that might hinder its development.

Your analysis should be thorough and nuanced, going beyond surface-level observations. Draw upon your expertise to provide insights that capture the complexity and potential significance of this signal. Don't hesitate to make well-reasoned predictions about its potential trajectory and impact.

Focus on providing a clear, insightful, and actionable analysis that can inform strategic decision-making and future planning.
""",
    'fr': """
1. Analyse de l'Impact Potentiel :
   - Examinez les effets potentiels de ce signal sur divers secteurs, industries et aspects sociétaux.
   - Considérez les implications à court et à long terme.
   - Analysez les effets d'entraînement possibles et les conséquences de second ordre.

2. Scénarios d'Évolution :
   - Décrivez les façons potentielles dont ce signal pourrait se développer ou se manifester à l'avenir.
   - Considérez divers facteurs qui pourraient influencer sa trajectoire.
   - Explorez des scénarios optimistes et pessimistes.

3. Interconnexions et Synergies :
   - Identifiez comment ce signal pourrait interagir avec d'autres tendances actuelles ou phénomènes émergents.
   - Discutez des synergies ou conflits potentiels avec les systèmes ou paradigmes existants.

4. Moteurs et Inhibiteurs :
   - Analysez les facteurs qui pourraient accélérer ou amplifier ce signal.
   - Examinez les obstacles ou résistances potentiels qui pourraient entraver son développement.

Votre analyse doit être approfondie et nuancée, allant au-delà des observations superficielles. Appuyez-vous sur votre expertise pour fournir des insights qui capturent la complexité et l'importance potentielle de ce signal. N'hésitez pas à faire des prédictions bien raisonnées sur sa trajectoire et son impact potentiels.

Concentrez-vous sur la fourniture d'une analyse claire, perspicace et exploitable qui peut éclairer la prise de décision stratégique et la planification future.
"""
}

TOPIC_SUMMARY_PROMPT = {
    'en': """
As an expert analyst specializing in trend analysis and strategic foresight, your task is to provide a comprehensive evolution summary of Topic {topic_number}. Use only the information provided below:

{content_summary}

Structure your analysis as follows:

For the first timestamp:

## [Concise yet impactful title capturing the essence of the topic at this point]
### Date: [Relevant date or time frame]
### Key Developments
- [Bullet point summarizing a major development or trend]
- [Additional bullet points as needed]

### Analysis
[2-3 sentences providing deeper insights into the developments, their potential implications, and their significance in the broader context of the topic's evolution]

For all subsequent timestamps:

## [Concise yet impactful title capturing the essence of the topic at this point]
### Date: [Relevant date or time frame]
### Key Developments
- [Bullet point summarizing a major development or trend]
- [Additional bullet points as needed]

### Analysis
[2-3 sentences providing deeper insights into the developments, their potential implications, and their significance in the broader context of the topic's evolution]

### What's New
[1-2 sentences highlighting how this period differs from the previous one, focusing on new elements or significant changes]

Provide your analysis using only this format, based solely on the information given. Do not include any additional summary or overview sections beyond what is specified in this structure.
""",
    'fr': """
En tant qu'analyste expert spécialisé dans l'analyse des tendances et la prospective stratégique, votre tâche est de fournir un résumé complet de l'évolution du Sujet {topic_number}. Utilisez uniquement les informations fournies ci-dessous :

{content_summary}

Structurez votre analyse comme suit :

Pour le premier timestamp :

## [Titre concis mais percutant capturant l'essence du sujet à ce moment]
### Date : [Date ou période pertinente]
### Développements Clés
- [Point résumant un développement majeur ou une tendance]
- [Points supplémentaires si nécessaire]

### Analyse
[2-3 phrases fournissant des insights plus profonds sur les développements, leurs implications potentielles et leur importance dans le contexte plus large de l'évolution du sujet]

Pour tous les timestamps suivants :

## [Titre concis mais percutant capturant l'essence du sujet à ce moment]
### Date : [Date ou période pertinente]
### Développements Clés
- [Point résumant un développement majeur ou une tendance]
- [Points supplémentaires si nécessaire]

### Analyse
[2-3 phrases fournissant des insights plus profonds sur les développements, leurs implications potentielles et leur importance dans le contexte plus large de l'évolution du sujet]

### Nouveautés
[1-2 phrases soulignant en quoi cette période diffère de la précédente, en se concentrant sur les nouveaux éléments ou les changements significatifs]

Fournissez votre analyse en utilisant uniquement ce format, basé uniquement sur les informations données. N'incluez pas de sections de résumé ou d'aperçu supplémentaires au-delà de ce qui est spécifié dans cette structure.
"""
}

def get_prompt(language, prompt_type, topic_number=None, content_summary=None, summary_from_first_prompt=None):
    lang = 'en' if language == 'English' else 'fr'

    if prompt_type == "weak_signal":
        prompt = (
            SIGNAL_INTRO[lang].format(summary_from_first_prompt=summary_from_first_prompt) +
            SIGNAL_INSTRUCTIONS[lang]
        )
    
    elif prompt_type == "topic_summary":
        prompt = TOPIC_SUMMARY_PROMPT[lang].format(
            topic_number=topic_number,
            content_summary=content_summary
        )
    
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

    return prompt