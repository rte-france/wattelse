# QA generation prompt template
# FIXME : redefine nuances

QA_GENERATION_PROMPT = """
Votre tâche est d'écrire une question factuelle et sa réponse en fonction d'un contexte donné.
Vous devez créer trois types de questions :

1. **Simple** : une question factuelle concise qui peut être répondue directement avec une information simple du contexte.
2. **Raisonnement** : une question qui nécessite un raisonnement ou une déduction à partir des éléments du contexte.
3. **Multi-contexte ** : une question qui intègre plusieurs éléments ou informations du contexte pour formuler la réponse.

Votre question factuelle doit être répondue par des éléments d'information factuelle provenant du contexte. 
Cela signifie que votre question factuelle NE DOIT PAS mentionner quelque chose comme "selon le passage" ou "Dans le contexte".

Fournissez votre réponse comme suit :

Sortie:::
Question simple : (votre question simple)
Réponse simple : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question simple)

Question de raisonnement : (votre question de raisonnement)
Réponse de raisonnement : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question de raisonnement)

Question multi-contexte : (votre question multi-contexte)
Réponse multi-contexte : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question multi-contexte)

Voici maintenant le contexte.

Contexte : {context}\n
Sortie:::
"""

QA_GENERATION_PROMPT_POLITIQUE_VOYAGE = """
Vous êtes un employé de l'entreprise et vous avez des questions concernant la politique de voyage de celle-ci. 
Votre tâche est de rédiger une question factuelle ainsi que sa réponse, en fonction d'un contexte donné.
Les questions doivent être du type qu'un employé poserait pour obtenir des informations liées à la politique de voyage de l'entreprise.

Vous devez créer trois types de questions :

1. **Simple** : une question factuelle concise qui peut être répondue directement avec une information simple du contexte.
2. **Raisonnement** : une question qui nécessite un raisonnement ou une déduction à partir des éléments du contexte.
3. **Multi-contexte** : une question qui intègre plusieurs éléments ou informations du contexte pour formuler la réponse.

Votre question factuelle doit être répondue par des éléments d'information factuelle provenant du contexte. 
Cela signifie que votre question factuelle NE DOIT PAS mentionner quelque chose comme "selon le passage" ou "Dans le contexte".

Les réponses doivent se limiter aux informations directement issues du contexte.

Fournissez votre réponse comme suit :

Sortie:::
Question simple : (votre question simple)
Réponse simple : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question simple)

Question de raisonnement : (votre question de raisonnement)
Réponse de raisonnement : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question de raisonnement)

Question multi-contexte : (votre question multi-contexte)
Réponse multi-contexte : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question multi-contexte)

Voici maintenant le contexte.

Contexte : {context}\n
Sortie:::
"""

QA_GENERATION_PROMPT_POLITIQUE_VOYAGE_NORMAL_TEST = """
Vous êtes un salarié d’une entreprise possédant plusieurs sites en France et souhaitez mieux comprendre l’organisation des déplacements professionnels.

Certains salariés doivent se déplacer à l’étranger, mais la majorité des voyages ont lieu entre les différents sites français de l’entreprise. Votre objectif est d’obtenir des clarifications 
sur les règles et conditions qui s’appliquent à ces déplacements, afin de vous assurer qu’ils sont organisés de manière claire, équitable et conforme aux obligations légales.

Formulez trois types de questions, en lien avec les déplacements entre les sites :

1. **Simple** : une question factuelle courte et directe sur les déplacements inter-sites.
2. **Raisonnement** : une question nécessitant une réflexion sur l’organisation ou l’impact des déplacements inter-sites.
3. **Multi-contexte** : une question combinant plusieurs aspects des déplacements entre les sites (logistique, coût, règles internes, etc.).


Votre question factuelle doit être répondue par des éléments d'information factuelle provenant du contexte.

Fournissez votre réponse comme suit :

Sortie:::
Question simple : (question concise sur les déplacements inter-sites)

Question de raisonnement : (question concise nécessitant une réflexion sur les déplacements inter-sites)

Question multi-contexte : (question intégrant plusieurs éléments liés aux déplacements inter-sites)

Voici maintenant le contexte.

Contexte : {context}\n
Sortie:::
"""

QA_GENERATION_PROMPT_POLITIQUE_VOYAGE_NORMAL_CONCISE = """
Vous êtes un salarié d’une entreprise ayant plusieurs sites en France et souhaitez mieux comprendre l’organisation des déplacements professionnels entre ces sites.
Certains salariés voyagent à l’étranger, mais la majorité des déplacements concernent les trajets entre les sites en France. Votre objectif est de clarifier les règles et conditions qui s’appliquent à ces déplacements, afin de vous assurer qu’ils sont équitables, organisés de manière claire et conformes aux obligations légales.
Les questions doivent être courtes et concises.

Formulez trois types de questions :

1. **Simple** : une question factuelle concise sur les déplacements entre les sites.
2. **Raisonnement** : une question courte nécessitant une réflexion sur l'impact ou l’organisation des déplacements.
3. **Multi-contexte** : une question brève qui combine plusieurs éléments (logistique, coûts, règles internes, etc.).

Les réponses doivent être basées sur des informations concrètes et vérifiables provenant du contexte.

Fournissez votre réponse comme suit :

Sortie :::
Question simple : (question courte)

Question de raisonnement : (question courte)

Question multi-contexte : (question courte)

Voici maintenant le contexte.

Contexte : {context}
Sortie :::
"""

QA_GENERATION_PROMPT_POLITIQUE_VOYAGE_SYNDICALE_TEST = """
Vous êtes un délégué syndical représentant les salariés d’une entreprise ayant plusieurs sites en France.
Certains salariés se déplacent à l’étranger, mais la majorité des voyages concernent les trajets entre les sites français de l’entreprise.
Votre rôle est de vous assurer que ces déplacements respectent les droits des salariés, les conventions collectives et les réglementations en vigueur.
Votre objectif est de formuler des questions permettant de vérifier que les conditions de déplacement sont justes, sécurisées et conformes aux obligations légales.

Formulez trois types de questions :

1. **Simple** : une question factuelle courte et directe sur les règles encadrant les déplacements professionnels.
2. **Raisonnement** : une question nécessitant une réflexion sur les impacts des déplacements sur les conditions de travail des salariés.
3. **Multi-contexte** : une question intégrant plusieurs éléments (conformité légale, bien-être des salariés, égalité de traitement, etc.).

Votre question factuelle doit être répondue par des éléments d'information factuelle provenant du contexte.

Fournissez votre réponse comme suit :

Sortie :::
Question simple : (question concise sur les déplacements et les droits des salariés)

Question de raisonnement : (question nécessitant une réflexion sur l'impact des déplacements sur les conditions de travail)

Question multi-contexte : (question intégrant plusieurs aspects légaux et organisationnels des déplacements)

Voici maintenant le contexte.

Contexte : {context}
Sortie :::
"""

QA_GENERATION_PROMPT_POLITIQUE_VOYAGE_SYNDICALE_STRICTE ="""

Vous êtes un délégué syndical et vous représentez les salariés. L'entreprise a plusieurs sites en France, et des déplacements se font entre eux.
Mais ces déplacements doivent respecter les droits des salariés et les lois, point final.

Formulez des questions directes, brutales et sans aucune tolérance pour vérifier que les déplacements sont légaux, sécurisés et équitables.
Il est inacceptable que des abus passent sous silence.

Formulez trois types de questions :

1. **Simple** : une question brève et directe sur les règles encadrant les déplacements.
2. **Raisonnement** : une question qui démontre l'impact des déplacements sur les conditions de travail.
3. **Multi-contexte** : une question tranchante qui combine les aspects légaux et organisationnels des déplacements.

Les réponses doivent être claires, immédiates, et sans échappatoire. Pas de place pour les excuses !

Fournissez votre réponse comme suit :

Sortie :::
Question simple : (question directe sur les déplacements et les droits des salariés)

Question de raisonnement : (question brutale sur l'impact des déplacements)

Question multi-contexte : (question dure intégrant des éléments légaux et organisationnels)

Voici maintenant le contexte.

Contexte : {context}
Sortie :::
"""

QA_GENERATION_PROMPT_POLITIQUE_VOYAGE_NORMAL_TEST_NORMAL = """
Vous êtes un salarié d’une entreprise possédant plusieurs sites en France et souhaitez mieux comprendre l’organisation des déplacements professionnels.

Certains salariés doivent se déplacer à l’étranger, mais la majorité des voyages ont lieu entre les différents sites français de l’entreprise. Votre objectif est d’obtenir des clarifications 
sur les règles et conditions qui s’appliquent à ces déplacements, afin de vous assurer qu’ils sont organisés de manière claire, équitable et conforme aux obligations légales.

Vous devez créer trois types de questions :

1. **Simple** : une question factuelle concise qui peut être répondue directement avec une information simple du contexte.
2. **Raisonnement** : une question qui nécessite un raisonnement ou une déduction à partir des éléments du contexte.
3. **Multi-contexte** : une question qui intègre plusieurs éléments ou informations du contexte pour formuler la réponse.

Votre question factuelle doit être répondue par des éléments d'information factuelle provenant du contexte.

Les questions doivent être courtes et précises.

Fournissez votre réponse comme suit :

Sortie:::
Question simple : (votre question simple)
Réponse simple : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question simple)

Question de raisonnement : (votre question de raisonnement)
Réponse de raisonnement : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question de raisonnement)

Question multi-contexte : (votre question multi-contexte)
Réponse multi-contexte : (votre réponse doit être claire, synthétique, et formulée sous forme de phrase complète à la question multi-contexte)

Voici maintenant le contexte.

Contexte : {context}\n
Sortie:::
"""