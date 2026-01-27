import os
import json
import pandas as pd
from datetime import datetime

class KnowledgeBase:
    """
    Pamięć Długotrwała V2.5 - Versioned Rolling Buffer.
    """
    MAX_RECORDS = 1000

    @staticmethod
    def _get_db_path(user):
        if not os.path.exists("assets"): os.makedirs("assets")
        return os.path.join("assets", f"brain_cortex_{user}.json")

    @staticmethod
    def save_pattern(user, features, label):
        db_file = KnowledgeBase._get_db_path(user)
        data = []
        if os.path.exists(db_file):
            try:
                with open(db_file, "r") as f: data = json.load(f)
            except: data = []

        record = features.copy()
        record['target'] = label
        record['ts'] = datetime.now().isoformat()
        record['ver'] = "V4.5"

        data.append(record)

        if len(data) > KnowledgeBase.MAX_RECORDS:
            excess = len(data) - KnowledgeBase.MAX_RECORDS
            data = data[excess:]

        try:
            with open(db_file, "w") as f: json.dump(data, f, indent=4)
        except Exception: return 0
        return len(data)

    @staticmethod
    def load_training_data(user):
        db_file = KnowledgeBase._get_db_path(user)
        if not os.path.exists(db_file): return pd.DataFrame()
        try:
            with open(db_file, "r") as f: data = json.load(f)
            return pd.DataFrame(data)
        except: return pd.DataFrame()

    @staticmethod
    def clear_memory(user):
        db_file = KnowledgeBase._get_db_path(user)
        if os.path.exists(db_file):
            os.remove(db_file)
            return True
        return False
