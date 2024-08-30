#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import json
import random
import uuid

from locust import HttpUser, task, between
from loguru import logger
from randomwordfr import RandomWordFr

ENDPOINT = "/encode"
HOST = "http://10.132.6.110:6465"  # gpu_pole


def generate_text(text_length: int = 5):
    rw = RandomWordFr()
    text = ""
    for i in range(0, text_length):
        result = rw.get()
        text += "# %s\n%s\n" % (result["word"], result["definition"])
    return text


class EmbeddingLoadTest(HttpUser):
    wait_time = between(0.1, 1)  # make the simulated users wait between .1 and 1 second

    @task
    def encode_task(self):
        myuuid = uuid.uuid4()
        show_progress_bar = False
        texts = [
            generate_text(random.randint(1, 10))
            for i in range(0, random.randint(1, 10))
        ]
        logger.debug(f"Sending {myuuid}")
        response = self.client.post(
            ENDPOINT, json={"text": texts, "show_progress_bar": show_progress_bar}
        )
        logger.debug(f"Receiving {myuuid}")


# To run it: locust -f <file.py> and then connect to http://localhost:8089 and specify the URL of the server you want to test http://xxx:port
# locust -f embedding_load.py --host http://10.132.6.110:6465 -u 100 -t 2m -r 1
