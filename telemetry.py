import csv
import os
from dataclasses import dataclass, asdict

CSV_PATH = "results_master.csv"
FALLBACK_PATH = "results.csv"
COLUMNS = [
    "request_id", "scenario", "source", "start_time",
    "first_token_ms", "total_ms", "status", "retries",
    "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"
]

@dataclass
class RequestLog:
    request_id: str
    scenario: str
    source: str          # "real" or "stand-in"
    start_time: float    # Unix timestamp (seconds, e.g. time.time())
    first_token_ms: float
    total_ms: float
    status: str          # "ok", "rate_limited", "timeout", "error"
    retries: int
    tokens_in: int
    tokens_out: int
    retrieval_setting: str
    answer_grade: str    # "correct", "wrong", "refused_correctly", "refused_wrongly"

def log_request(entry: RequestLog):
    for path in [CSV_PATH, FALLBACK_PATH]:
        try:
            file_exists = os.path.exists(path)
            with open(path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(asdict(entry))
        except PermissionError:
            continue
