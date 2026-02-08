import os
import json
from pathlib import Path

from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv


def json_default(o):
    if isinstance(o, ObjectId):
        return str(o)
    raise TypeError


def load_songs(training: bool = True):
    load_dotenv()
    if training:
        dataset_path = Path("data/training_dataset.json")
    else:
        dataset_path = Path("data/testing_dataset.json")

    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # 2) Altrimenti carica da MongoDB Atlas
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "dataset.json non trovato e MONGODB_URI non impostato. "
            "Vedi .env.example."
        )

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[os.getenv("MONGODB_DB")]
    if training:
        collection = db["songs"]
    else:
        collection = db["testing_songs"]

    client.admin.command("ping")

    songs = list(collection.find({}, {"_id": 0}))

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as f:
        json.dump(songs, f, ensure_ascii=False, indent=2, default=json_default)

    return songs


if __name__ == "__main__":
    load_songs(training=False)
