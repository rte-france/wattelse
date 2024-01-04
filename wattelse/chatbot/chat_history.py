import datetime
import json

from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

DEFAULT_MEMORY_DELAY = 2  # in minutes

class ChatHistory:

    def __init__(self):
        self.db_table = ChatHistory.initialize_db()

    @classmethod
    def initialize_db(cls):
        # in memory small DB for handling messages
        db = TinyDB(storage=MemoryStorage)
        table = db.table("messages")
        return table


    def get_history(self):
        """Return the history of the conversation"""
        context = self.get_recent_context()
        history = ""
        for entry in context:
            history += "Utilisateur : {query}\nRéponse : {response}\n".format(query=entry["query"], response=entry["response"])
        return history


    def get_recent_context(self, delay=DEFAULT_MEMORY_DELAY):
        """Returns a list of recent answers from the bot that occured during the indicated delay in minutes"""
        current_timestamp = datetime.datetime.now()
        q = Query()
        return self.db_table.search(
            q.timestamp > (current_timestamp - datetime.timedelta(minutes=delay))
        )


    def add_to_database(self, query, response):
        timestamp = datetime.datetime.now()
        self.db_table.insert({"query": query, "response": response, "timestamp": timestamp})


    def export_history(self):
        """Export messages in JSON from the database"""
        return json.dumps(
            [
                {
                    k: v if k != "timestamp" else v.strftime("%d-%m-%Y %H:%M:%S")
                    for k, v in d.items()
                }
                for d in self.db_table.all()
            ]
        )

