#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

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
