
import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Paths inside aris-api container
sys.path.insert(0, '/app')

# We'll just use raw SQL for inspection/cleanup to be independent of model changes if possible, 
# but importing models is safer if strictly needed. Let's try raw SQL first for simplicity in a script.
# Actually, let's use the shared models if we can run this inside the container.
# The container has /app/shared/models.py

try:
    from shared.models import Transcript, QSO
except ImportError:
    # If running outside container where shared lib isn't in path exactly the same
    pass

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:////data/db/aris.db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

print("Scanning for potential hallucinations...")

# Patterns to check
bad_patterns = ["you", "You", "bye", "Bye", "goodbye", "Goodbye", "Thanks for watching", "subscribe", ". . .", "ww."]
# Exact matches for short words
exact_matches = ["you.", "You.", "you", "You", "bye.", "Bye.", "bye", "Bye"]

# Fetch all transcripts (might be large, but let's limit or stream)
# For inspection, just get count and samples
print("Checking Transcripts...")
bad_count = 0
ids_to_delete = []

# Simple heuristic check in python
transcripts = session.execute(text("SELECT id, text FROM transcripts")).fetchall()

for t in transcripts:
    txt = t[1].strip()
    is_bad = False
    
    # Check exact noise words (stripping punctuation)
    clean_txt = txt.strip(" .!?").lower()
    if clean_txt in ["you", "bye", "goodbye"]:
        is_bad = True
    
    # Check contains
    if not is_bad:
        for p in ["Thanks for watching", "Amara.org", "MBC"]:
            if p.lower() in txt.lower():
                is_bad = True
                break
    
    if is_bad:
        bad_count += 1
        ids_to_delete.append(t[0])
        if bad_count <= 10:
            print(f" [BAD] ID {t[0]}: '{txt}'")

print(f"Total suspicious transcripts found: {bad_count}")

print("\nChecking QSOs (Summaries)...")
qsos = session.execute(text("SELECT session_id, summary FROM qsos")).fetchall()
bad_qso_count = 0
qso_ids_to_delete = []

for q in qsos:
    summary = q[1]
    # Heuristics for bad summaries
    if "only contains one station" in summary and "callsign unknown" in summary and "you" in summary:
        # potentially bad if the transcript was just "you"
        pass
    
    # If summary mentions "Thanks for watching" or seems extremely generic for 1 line
    if "Thanks for watching" in summary:
        print(f" [BAD QSO] {q[0]}: ...Thanks for watching...")
        bad_qso_count += 1
        qso_ids_to_delete.append(q[0])
        continue

    # Identify empty/nonsense summaries
    if len(summary) < 20: 
        print(f" [SHORT QSO] {q[0]}: '{summary}'")
        bad_qso_count += 1
        qso_ids_to_delete.append(q[0])

print(f"Total suspicious QSOs found: {bad_qso_count}")
session.close()
