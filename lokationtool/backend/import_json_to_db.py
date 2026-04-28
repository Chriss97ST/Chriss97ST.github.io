import json
import os
from datetime import datetime

from sqlalchemy.orm import Session

from main import Base, LocationEntry, SessionLocal, engine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_BASE_DIR = os.path.dirname(BASE_DIR)

DATASETS = {
    "empg_bohrungen": "empg_bohrungen.json",
    "empg_schieber": "empg_schieber.json",
    "gasunie_schieber": "gasunie_schieber.json",
}


def load_json_file(file_name: str) -> list[dict]:
    full_path = os.path.join(JSON_BASE_DIR, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Missing JSON file: {full_path}")

    with open(full_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list: {full_path}")
    return data


def replace_dataset(db: Session, dataset: str, rows: list[dict]) -> int:
    db.query(LocationEntry).filter(LocationEntry.dataset == dataset).delete()

    inserted = 0
    for row in rows:
        ort = (row.get("ort") or "").strip()
        if not ort:
            continue

        breite = row.get("breite")
        laenge = row.get("laenge")
        try:
            breite_value = float(breite)
            laenge_value = float(laenge)
        except (TypeError, ValueError):
            continue

        entry = LocationEntry(
            dataset=dataset,
            ort=ort,
            kuerzel=(row.get("kuerzel") or "").strip() or None,
            breite=breite_value,
            laenge=laenge_value,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(entry)
        inserted += 1

    db.commit()
    return inserted


def main() -> None:
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        for dataset, file_name in DATASETS.items():
            rows = load_json_file(file_name)
            inserted = replace_dataset(db, dataset, rows)
            print(f"{dataset}: imported {inserted} entries")
    finally:
        db.close()


if __name__ == "__main__":
    main()
