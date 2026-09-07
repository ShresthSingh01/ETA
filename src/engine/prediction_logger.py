"""
prediction_logger.py - Real-Time In-Memory Prediction Evaluation Loop & Durable Audit Store.

Enables continuous self-evaluation: logs predictions when generated, matches them
against subsequent actual arrival observations, computes rolling error metrics,
and optionally records durable audit logs for post-run forensic monitoring.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import math


@dataclass
class PredictionRecord:
    prediction_id: str
    train_id: str
    station_code: str
    station_name: str
    predicted_at: str
    predicted_arrival: str
    predicted_delay_mins: float
    confidence_pct: float
    actual_arrival: Optional[str] = None
    actual_delay_mins: Optional[float] = None
    error_mins: Optional[float] = None
    status: str = "PENDING_ARRIVAL"  # 'PENDING_ARRIVAL', 'EVALUATED'


class LivePredictionLogger:
    """
    Circular in-memory evaluation log holding recent predictions and computing
    rolling metrics with optional durable append-only JSONL logging.
    """

    def __init__(self, capacity: int = 250, log_file_path: Optional[str] = None):
        self.capacity = capacity
        self.records: List[PredictionRecord] = []
        self._counter = 0
        self.log_file_path = log_file_path
        if self.log_file_path:
            Path(self.log_file_path).parent.mkdir(parents=True, exist_ok=True)

    def seed_from_genuine_evaluations(self, evaluated_records: List[PredictionRecord]) -> None:
        """Populates initial evaluation history from verified operational holdout records."""
        for rec in evaluated_records:
            self._counter += 1
            if not rec.prediction_id:
                rec.prediction_id = f"PRED-{self._counter:04d}"
            self.records.append(rec)
            if len(self.records) > self.capacity:
                self.records.pop(0)

    def log_prediction(
        self,
        train_id: str,
        station_code: str,
        station_name: str,
        predicted_arrival: str,
        predicted_delay_mins: float,
        confidence_pct: float
    ) -> str:
        """Logs a newly calculated station prediction."""
        self._counter += 1
        pid = f"PRED-{self._counter:04d}"
        rec = PredictionRecord(
            prediction_id=pid,
            train_id=str(train_id),
            station_code=station_code,
            station_name=station_name,
            predicted_at=datetime.now(timezone.utc).isoformat(),
            predicted_arrival=predicted_arrival,
            predicted_delay_mins=round(predicted_delay_mins, 1),
            confidence_pct=round(confidence_pct, 1)
        )
        self.records.append(rec)
        if len(self.records) > self.capacity:
            self.records.pop(0)
        return pid

    def record_arrival(
        self,
        train_id: str,
        station_code: str,
        actual_delay_mins: float,
        actual_arrival: Optional[str] = None
    ) -> Optional[float]:
        """
        Matches a newly observed station arrival against the most recent pending prediction.
        Computes absolute error and updates state to EVALUATED.
        """
        for rec in reversed(self.records):
            if (
                rec.train_id == str(train_id) and
                rec.station_code == station_code and
                rec.status == "PENDING_ARRIVAL"
            ):
                rec.actual_delay_mins = round(actual_delay_mins, 1)
                rec.actual_arrival = actual_arrival or rec.predicted_arrival
                rec.error_mins = round(abs(rec.predicted_delay_mins - rec.actual_delay_mins), 1)
                rec.status = "EVALUATED"

                # Durable JSONL audit logging if configured
                if self.log_file_path:
                    try:
                        with open(self.log_file_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(asdict(rec)) + "\n")
                    except Exception:
                        pass

                return rec.error_mins
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates rolling accuracy and error metrics across evaluated records."""
        evaluated = [r for r in self.records if r.status == "EVALUATED" and r.error_mins is not None]
        if not evaluated:
            return {
                "total_logged": len(self.records),
                "evaluated_count": 0,
                "rolling_mae_mins": 0.0,
                "rolling_rmse_mins": 0.0,
                "within_3_mins_pct": 100.0,
                "within_5_mins_pct": 100.0,
                "per_train_mae": {},
                "recent_records": [asdict(r) for r in reversed(self.records[-15:])]
            }

        errors = [r.error_mins for r in evaluated]
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
        within_3 = (sum(1 for e in errors if e <= 3.0) / len(errors)) * 100.0
        within_5 = (sum(1 for e in errors if e <= 5.0) / len(errors)) * 100.0

        # Per-train breakdown
        per_train_errors: Dict[str, List[float]] = {}
        for r in evaluated:
            per_train_errors.setdefault(r.train_id, []).append(r.error_mins)
        per_train_mae = {
            t: round(sum(errs) / len(errs), 2)
            for t, errs in per_train_errors.items()
        }

        return {
            "total_logged": len(self.records),
            "evaluated_count": len(evaluated),
            "rolling_mae_mins": round(mae, 2),
            "rolling_rmse_mins": round(rmse, 2),
            "within_3_mins_pct": round(within_3, 1),
            "within_5_mins_pct": round(within_5, 1),
            "per_train_mae": per_train_mae,
            "recent_records": [asdict(r) for r in reversed(self.records[-15:])]
        }
