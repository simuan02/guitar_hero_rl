import json
import random

import librosa
import numpy as np
from bson import ObjectId
from pymongo import MongoClient
import gridfs

""" Script per il caricamento delle canzoni nel DB, creazione delle note attraverso la rilevazione degli onset 
 e la quantizzazione e il calcolo del grado di complessità"""


def song_upload(collection, filename, title, artist):
    with open("music/" + filename, "rb") as f:
        audio_id = fs.put(
            f,
            filename=filename,
            contentType="audio/ogg",
            metadata={
                "title": title,
                "artist": artist
            }
        )

    print(audio_id)

    stats = song_stats_and_complexity("music/" + filename)

    song_doc = {"title": title,
                "artist": artist,
                "audio_file_id": ObjectId(audio_id),
                "duration": stats["duration"],
                "notes_per_second": stats["notes_per_second"],
                "std_ioi": stats["std_ioi"],
                "complexity_score": stats["complexity_score"]}

    db[collection].insert_one(song_doc)

    normalized_score = min(stats["complexity_score"] / 10.0, 1.0)
    return normalized_score


def song_stats_and_complexity(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames", backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Statistiche di Base
    duration = librosa.get_duration(y=y, sr=sr)  # durata in secondi
    note_count = len(onset_times)  # numero totale di note
    notes_per_second = note_count / duration  # densità media

    # Irregolarità ritmica
    ioi = np.diff(onset_times)
    std_ioi = np.std(ioi)

    # Complexity score (peso 70% densità, 30% irregolarità)
    complexity_score = 0.7 * notes_per_second + 0.3 * std_ioi

    stats = {
        "duration": duration,
        "notes_per_second": notes_per_second,
        "std_ioi": std_ioi,
        "complexity_score": complexity_score
    }

    return stats


def creazione_note_canzone(
    collection,
    audio_path,
    song_title,
    complexity_score,
    tempo_percorrenza=2.5
):
    y, sr = librosa.load("music/" + audio_path, sr=None, mono=True)

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="frames",
        backtrack=True,
        delta=0.07,
        wait=10
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    beat_duration = 60.0 / tempo

    # Parametri di Quantizzazione Adattivi
    min_interval = beat_duration * (0.75 - 0.6 * complexity_score)  # da 0.75 a 0.15
    quantize_strength = 0.5 - 0.3 * complexity_score  # da 0.5 a 0.2

    filtered_onsets = [onset_times[0]]
    for t in onset_times[1:]:
        if t - filtered_onsets[-1] >= min_interval:
            filtered_onsets.append(t)

    quantized_onsets = []
    for o in filtered_onsets:
        nearest_beat = min(beat_times, key=lambda b: abs(b - o))
        shift = nearest_beat - o

        if abs(shift) < beat_duration * quantize_strength:
            quantized_onsets.append(nearest_beat)
        else:
            quantized_onsets.append(o)

    colori = ["verde", "rossa", "gialla", "blu", "arancione"]
    notes = []

    for t in quantized_onsets:
        notes.append({
            "colore": random.choice(colori),
            "tempo": int(t * 1000),
            "tempo_percorrenza": tempo_percorrenza
        })

    db[collection].update_one(
        {"title": song_title},
        {"$set": {"note": notes}}
    )

    return notes


client = MongoClient("mongodb://localhost:27017/")
db = client["songs"]

fs = gridfs.GridFS(db)

with open("song_list.json", "r") as f:
    song_list = json.load(f)

print(song_list)

for song in song_list[:10]:
    song_complexity_score = song_upload("songs", song["filename"], song["title"], song["artist"])
    creazione_note_canzone("songs", song["filename"], song["title"], song_complexity_score)

for song in song_list[10:]:
    song_complexity_score = song_upload("testing_songs", song["filename"], song["title"], song["artist"])
    creazione_note_canzone("testing_songs", song["filename"], song["title"], song_complexity_score)
