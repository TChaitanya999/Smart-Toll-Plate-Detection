import os
import csv
from typing import Optional, Iterable, Dict, Any, Union


DEFAULT_DB_PATH = "plates.csv"
REQUIRED_COLUMNS = ["plate", "owner_name", "registration_date"]


def ensure_schema(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Ensure the CSV "registry" exists and has the expected header.

    Supports migrating older versions that only had a single `plate` column.
    """
    if not os.path.exists(db_path):
        with open(db_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(REQUIRED_COLUMNS)
        return

    # Migrate header if needed (e.g., legacy header: ["plate"])
    try:
        with open(db_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None) or []
            rows = list(reader)

        header_norm = [h.strip() for h in header]
        if header_norm == REQUIRED_COLUMNS:
            return

        # If file has no header or unexpected header, attempt best-effort migration.
        # Treat first column as plate; fill the rest with empty strings.
        migrated_rows = []
        for row in rows:
            if not row:
                continue
            plate_val = row[0] if len(row) > 0 else ""
            owner_val = row[1] if len(row) > 1 else ""
            date_val = row[2] if len(row) > 2 else ""
            migrated_rows.append([plate_val, owner_val, date_val])

        with open(db_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(REQUIRED_COLUMNS)
            writer.writerows(migrated_rows)
    except FileNotFoundError:
        # Race or removed file; recreate.
        with open(db_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(REQUIRED_COLUMNS)


def normalize_plate(plate: str) -> str:
    return "".join(ch for ch in (plate or "").upper() if ch.isalnum())


def is_registered(plate: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    plate_n = normalize_plate(plate)
    if not plate_n:
        return False
    
    ensure_schema(db_path)
    
    try:
        with open(db_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if normalize_plate(row.get("plate", "")) == plate_n:
                    return True
    except FileNotFoundError:
        return False
        
    return False


def get_plate_record(plate: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Return the registry record for a plate if present.

    Output example:
      {"plate": "...", "owner_name": "...", "registration_date": "YYYY-MM-DD"}
    """
    plate_n = normalize_plate(plate)
    if not plate_n:
        return None

    ensure_schema(db_path)

    try:
        with open(db_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if normalize_plate(row.get("plate", "")) == plate_n:
                    return {
                        "plate": normalize_plate(row.get("plate", "")),
                        "owner_name": (row.get("owner_name") or "").strip(),
                        "registration_date": (row.get("registration_date") or "").strip(),
                    }
    except FileNotFoundError:
        return None

    return None


def upsert_plate(
    plate: str,
    owner_name: str = "",
    registration_date: str = "",
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """
    Insert or update a plate record (by normalized plate value).

    Keeps CSV as the single source of truth to match the current project structure.
    """
    plate_n = normalize_plate(plate)
    if not plate_n:
        return

    ensure_schema(db_path)

    rows = []
    updated = False
    try:
        with open(db_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                row_plate = normalize_plate(row.get("plate", ""))
                if row_plate == plate_n:
                    rows.append(
                        {
                            "plate": plate_n,
                            "owner_name": owner_name.strip(),
                            "registration_date": registration_date.strip(),
                        }
                    )
                    updated = True
                else:
                    rows.append(
                        {
                            "plate": row_plate,
                            "owner_name": (row.get("owner_name") or "").strip(),
                            "registration_date": (row.get("registration_date") or "").strip(),
                        }
                    )
    except FileNotFoundError:
        rows = []

    if not updated:
        rows.append(
            {
                "plate": plate_n,
                "owner_name": owner_name.strip(),
                "registration_date": registration_date.strip(),
            }
        )

    with open(db_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def bulk_insert(plates: Iterable[str], db_path: str = DEFAULT_DB_PATH) -> int:
    ensure_schema(db_path)
    
                                              
    existing_plates = set()
    try:
        with open(db_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and row.get("plate"):
                    existing_plates.add(normalize_plate(row.get("plate", "")))
    except FileNotFoundError:
        pass

    new_rows = []
    count = 0
    for p in plates:
        norm_p = normalize_plate(p)
        if norm_p and norm_p not in existing_plates:
            new_rows.append(
                {
                    "plate": norm_p,
                    "owner_name": "",
                    "registration_date": "",
                }
            )
            existing_plates.add(norm_p)
            count += 1
            
    if new_rows:
        with open(db_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
            # File already has header; just append rows.
            writer.writerows(new_rows)
            
    return len(existing_plates)


