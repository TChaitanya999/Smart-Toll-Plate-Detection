"""
Create/populate a local SQLite plate registry with 500+ plates.

Run:
  python init_plate_db.py

This will create `plates.csv` in the current folder.
"""

import random
import os
from datetime import date, timedelta
import csv
from plate_registry import bulk_insert, upsert_plate, get_plate_record, DEFAULT_DB_PATH


STATE_CODES = [
    "AN","AP","AR","AS","BR","CH","DD","DL","DN","GA","GJ","HP","HR","JH","JK","KA",
    "KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","PB","PY","RJ","SK","TN","TR",
    "TS","UA","UK","UP","UT","WB"
]


def gen_plates(n: int, seed: int = 42):
    """
    Generate plausible Indian plates:
      AA NN AA NNNN  (10 chars)
    """
    rng = random.Random(seed)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    plates = set()
    while len(plates) < n:
        st = rng.choice(STATE_CODES)
        district = f"{rng.randint(1, 99):02d}"
        series = rng.choice(letters) + rng.choice(letters)
        number = f"{rng.randint(1, 9999):04d}"
        plates.add(f"{st}{district}{series}{number}")
    return sorted(plates)


def gen_owner_name(rng: random.Random) -> str:
    first = [
        "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Ishaan","Krishna","Rohan",
        "Aanya","Diya","Ananya","Ira","Meera","Saanvi","Myra","Aadhya","Kavya","Riya",
    ]
    last = [
        "Sharma","Verma","Gupta","Patel","Reddy","Nair","Iyer","Singh","Khan","Das",
        "Chatterjee","Mehta","Jain","Bose","Menon","Kulkarni","Joshi","Yadav","Roy","Saha",
    ]
    return f"{rng.choice(first)} {rng.choice(last)}"


def gen_registration_date(rng: random.Random) -> str:
    # Random date in last ~12 years
    end = date.today()
    start = end - timedelta(days=365 * 12)
    d = start + timedelta(days=rng.randint(0, (end - start).days))
    return d.isoformat()


def seed_owner_and_dates(plates, seed: int = 42, db_path: str = DEFAULT_DB_PATH) -> None:
    rng = random.Random(seed)
    for p in plates:
        existing = get_plate_record(p, db_path=db_path)
        owner = (existing or {}).get("owner_name", "").strip()
        reg_date = (existing or {}).get("registration_date", "").strip()

        # Only fill missing fields; do not overwrite existing non-empty values.
        if not owner:
            owner = gen_owner_name(rng)
        if not reg_date:
            reg_date = gen_registration_date(rng)

        upsert_plate(p, owner_name=owner, registration_date=reg_date, db_path=db_path)


def list_all_plates(db_path: str) -> list[str]:
    plates: list[str] = []
    with open(db_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and row.get("plate"):
                plates.append(row["plate"])
    return plates
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(here, DEFAULT_DB_PATH)
    plates = gen_plates(600)                       
    total = bulk_insert(plates, db_path=db_path)
    all_plates = list_all_plates(db_path)
    seed_owner_and_dates(all_plates, seed=42, db_path=db_path)
    print(f"Created/updated {db_path} with {total} registered plates.")


