"""
base.py - Provider Abstraction & Canonical State Definitions.

Decouples external live observation feeds (RailRadar, RTIS, FOIS, GPS)
from the internal ML inference and Railway Rule Engine.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone


class FreshnessLevel(str, Enum):
    FRESH = "FRESH"      # <= 60 seconds old
    AGING = "AGING"      # 61 - 180 seconds old
    STALE = "STALE"      # > 180 seconds old
    UNKNOWN = "UNKNOWN"  # No valid timestamp available


def get_freshness_level(age_seconds: float) -> FreshnessLevel:
    """Classifies telemetry age into operational freshness categories."""
    if age_seconds < 0:
        return FreshnessLevel.UNKNOWN
    if age_seconds <= 60.0:
        return FreshnessLevel.FRESH
    if age_seconds <= 180.0:
        return FreshnessLevel.AGING
    return FreshnessLevel.STALE


@dataclass
class CanonicalTrainState:
    """
    Standardized live train observation schema.
    All external providers (RailRadar, NTES, GPS, or Replay) must normalize to this.
    """
    provider: str
    train_id: str
    train_name: str
    journey_date: str
    timestamp: str  # ISO-8601 or formatted local string

    status: str  # 'RUNNING', 'HALTED', 'NOT_STARTED', 'TERMINATED', 'DIVERTED'
    current_station_code: str
    current_station_name: str
    current_sequence: int
    segment_progress: float  # 0.0 (at from_station) to 1.0 (at next_station)

    speed_kmph: float
    bearing_deg: float
    current_delay_min: float

    next_station_code: str
    next_station_name: str

    source_freshness_sec: float
    is_actual_position: bool  # True if confirmed GPS/signaling, False if extrapolated
    
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    exceptions: List[str] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def freshness_level(self) -> FreshnessLevel:
        return get_freshness_level(self.source_freshness_sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "train_id": self.train_id,
            "train_name": self.train_name,
            "journey_date": self.journey_date,
            "timestamp": self.timestamp,
            "status": self.status,
            "current_station_code": self.current_station_code,
            "current_station_name": self.current_station_name,
            "current_sequence": self.current_sequence,
            "segment_progress": round(self.segment_progress, 3),
            "speed_kmph": round(self.speed_kmph, 1),
            "bearing_deg": round(self.bearing_deg, 1),
            "current_delay_min": round(self.current_delay_min, 1),
            "next_station_code": self.next_station_code,
            "next_station_name": self.next_station_name,
            "source_freshness_sec": round(self.source_freshness_sec, 1),
            "freshness_level": self.freshness_level.value,
            "is_actual_position": self.is_actual_position,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "exceptions": self.exceptions,
            "raw_metadata": self.raw_metadata
        }


@dataclass
class StationBoardEntry:
    """
    Live arrival/departure train record from a station's digital board.
    Used for downstream junction traffic and platform congestion diagnostics.
    """
    train_number: str
    train_name: str
    scheduled_time: str
    expected_time: str
    delay_minutes: float
    platform: str
    status: str  # 'ARRIVED', 'EXPECTED', 'DEPARTED', 'DIVERTED', 'CANCELLED'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_number": self.train_number,
            "train_name": self.train_name,
            "scheduled_time": self.scheduled_time,
            "expected_time": self.expected_time,
            "delay_minutes": round(self.delay_minutes, 1),
            "platform": self.platform,
            "status": self.status
        }


class TrainStateProvider(ABC):
    """
    Abstract Base Class for train telemetry providers.
    Guarantees plug-and-play capability between RailRadar, Replay, or future RTIS.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'RailRadar', 'HistoricalReplay')."""
        pass

    @abstractmethod
    def get_live_state(
        self,
        train_id: str,
        journey_date: Optional[str] = None
    ) -> Optional[CanonicalTrainState]:
        """Fetches and normalizes live observation for a train."""
        pass

    @abstractmethod
    def get_station_board(
        self,
        station_code: str
    ) -> List[StationBoardEntry]:
        """Fetches current live board of trains at a station."""
        pass

    @abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """Returns health diagnostics: latency, status, uptime, error counts."""
        pass
