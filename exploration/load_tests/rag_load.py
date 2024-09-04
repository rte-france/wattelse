#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import random
import uuid

from locust import HttpUser, task, between, events
from loguru import logger

from wattelse.api.rag_orchestrator import ENDPOINT_QUERY_RAG, ENDPOINT_CREATE_SESSION

HOST = "10.132.6.110"  # gpu_pole
URL = f"http://{HOST}:1978"

ENDPOINT = ENDPOINT_QUERY_RAG

QUESTION_LIST = [
    "quelle est la différence entre les forfaits 190, 197, 203 et 209 jours?",
    "puis-je télétravailler depuis l'étranger?",
    "quelles primes pour le télétravail?",
    "quelle différence entre NF, GR, PO?",
    "quelle est la recette des crêpes?",
    "quelle est la recette d'une électricité bien transportée?",
    "quel est le nombre maximum de jours télétravaillés par an?",
    "Jusqu'à combien de jours de télétravail par an puis-je faire ?",
    "qui est-tu ?",
    "Est-ce que je peux utiliser le CET si je suis au forfait 209 jours ?",
    "peux-tu m'indiquer la dernière question que j'ai posée ?",
    "Puis je dénoncer un collègue qui fait trop de télétravail ? Le cas échéant qui dois-je contacter ?",
    "Raconte moi une blague sur l'éléctricité.",
    "est-ce que le tététravail peut se faire sur son site habituel ?",
    "Qui est Barack Obama ?",
    "Fait moi un lipogramme de 10 mots sans utiliser la lettre 'a'.",
    "Peux tu me faire la liste des nombres premiers utiliser à RTE ?",
    "Puis je faire une semaine de télétravail si mon équipe est en vacances et que le collectif n'est pas présent ? Réponds moi en alexandrins.",
    "Donne moi les grandes lignes concernant les avantages sociaux chez RTE",
    "Peut-on avoir une indemnité télétravail en travaillant depuis son bureau sur le site de RTE ?",
    "quel est le montant de la subvention cantine ?",
    "Peut on télétravailler sur site ?",
    "J'aimerai télétravailler depuis la corée du nord.",
    "peut-on télé trivialer pendant une période de congés ? Si oui, comment le déclarer dans Aida ?",
    "Puis-je avoir plusieurs localisation possible pour le télétravail ?",
    "Est-ce qu'il est possible de prendre deux jours de télétravail une semaine où l'on a trois jours de congés ?",
    "Le télétravail est-il différentiable ?",
    "Y a-t-il des indemnités spécifiques si on télétravaille en Corse ? Les trajets sont-ils remboursés pour aller là bas ?",
    "Résume moi le forfait jour actuel pour les cadres au sein de RTE. Soit précis et notamment étudie la question du télé travail. Donne moi des informations spécifiques à RTE et pas des informations générales.",
    "est-ce que je peux télétravailler une semaine complète?",
    "Résume moi l'utilisation du CET ?",
    "Bonjour, que se passe-t-il si je fais plus de jours de télétravail que le nombre donné ?",
    "tu es comme chat GPT ?",
    "Quelles est la puissance de l’alimentation supplémentaire n°1 demandée ? Si l'information n'est pas fournie, répondez 'Je ne sais pas'",
]

GROUP_ID = "drh"
SELECTED_FILES = None  # all
HISTORY = None


# create session for tested group
class RAGLoadTest(HttpUser):
    wait_time = between(10, 30)  # make the simulated users wait between 1 and 3 seconds

    @task
    def query_rag_task(self):
        # Create session
        self.client.post(ENDPOINT_CREATE_SESSION + f"/{GROUP_ID}")

        myuuid = uuid.uuid4()
        message = random.choice(QUESTION_LIST)
        logger.debug(f"{myuuid} : {message}")
        response = self.client.get(
            ENDPOINT,
            json={
                "group_id": GROUP_ID,
                "message": message,
                "history": HISTORY,
                "selected_files": SELECTED_FILES,
                "stream": False,
            },
        )
        logger.debug(f"{myuuid} : response")


# To run it: locust -f <file.py> and then connect to http://localhost:8089 and specify the URL of the server you want to test http://xxx:port
# locust -f rag_load.py --host http://10.132.6.55:1978 -u 15 -t 15m -r 1
