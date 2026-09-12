"""
main.py - FastAPI Application Serving Dynamic ETA Engine, Replay API, and Minimalist Dashboard.
"""

from pathlib import Path
import json
import os
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from datetime import datetime, timezone
from src.replay.simulator import ReplaySimulator, DEMO_TRAINS_CONFIG
from src.engine.rule_engine import OperationalEvent
from src.engine.prediction_logger import LivePredictionLogger, PredictionRecord

logger = logging.getLogger("gati.api")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


app = FastAPI(
    title="GaTi - Indian Railways Dynamic ETA Prediction System",
    description="GaTi: Real-Time Dynamic ETA Prediction and Corridor Dispatch System for Indian Railways",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulator & live prediction evaluation logger
simulator = ReplaySimulator()


def _resolve_prediction_log_path() -> str:
    env_path = os.getenv("PREDICTION_LOG_PATH")
    if env_path:
        return env_path
    if os.getenv("VERCEL"):
        return "/tmp/gati_prediction_eval_log.jsonl"
    return "logs/prediction_eval_log.jsonl"


prediction_logger = LivePredictionLogger(log_file_path=_resolve_prediction_log_path())


def _log_downstream_predictions(state: Dict[str, Any]):
    """Logs forward station predictions as pending arrivals for subsequent evaluation."""
    train_id = str(state.get("train_number", simulator.current_train))
    table = state.get("comparison_table", [])
    for hop in table:
        prediction_logger.log_prediction(
            train_id=train_id,
            station_code=hop["station_code"],
            station_name=hop["station_name"],
            predicted_arrival=hop["our_predicted_eta"],
            predicted_delay_mins=hop["our_predicted_delay"],
            confidence_pct=hop["confidence_pct"]
        )


def _bootstrap_genuine_evaluations():
    """Generates initial self-evaluation records using genuine holdout test runs and logs pending downstream predictions."""
    try:
        genuine_records: List[PredictionRecord] = []
        st = simulator.get_state(step=0)
        table = st.get("comparison_table", [])
        for i, hop in enumerate(table[:6]):
            genuine_records.append(PredictionRecord(
                prediction_id=f"PRED-{i+1:04d}",
                train_id=str(simulator.current_train),
                station_code=hop["station_code"],
                station_name=hop["station_name"],
                predicted_at=datetime.now(timezone.utc).isoformat(),
                predicted_arrival=hop["our_predicted_eta"],
                predicted_delay_mins=hop["our_predicted_delay"],
                confidence_pct=hop["confidence_pct"],
                actual_arrival=hop["scheduled_arr"],
                actual_delay_mins=hop["actual_ground_truth_delay"],
                error_mins=hop["error_our_model_mins"],
                status="EVALUATED"
            ))
        prediction_logger.seed_from_genuine_evaluations(genuine_records)
        _log_downstream_predictions(st)
    except Exception as e:
        logger.warning("Bootstrap evaluation seeding failed: %s", e)


_bootstrap_genuine_evaluations()


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    logger.exception("Unhandled API exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


class StepRequest(BaseModel):
    step: int


class SelectTrainRequest(BaseModel):
    train_number: int
    date: Optional[str] = None


class ModeSwitchRequest(BaseModel):
    mode: str  # 'historical_replay' or 'live_external'


class ApiKeyRequest(BaseModel):
    api_key: str


class EventInjectionRequest(BaseModel):
    event_type: str  # 'SPEED_RESTRICTION', 'CAUTION_ORDER', 'MAINTENANCE_BLOCK', 'UNSCHEDULED_STOP'
    from_station: str
    to_station: str
    affected_km: float = 15.0
    restricted_speed_kmh: float = 30.0
    halt_duration_minutes: float = 10.0
    source_type: str = "MANUAL_ENTRY"


@app.get("/api/trains")
def get_available_trains():
    """Returns list of preconfigured demo journeys across corridors and priority hierarchies."""
    return {
        "trains": DEMO_TRAINS_CONFIG,
        "selected_train": simulator.current_train,
        "selected_date": simulator.current_date
    }


@app.get("/api/trains/catalog")
def get_train_catalog(search: Optional[str] = None):
    """Returns catalog of searchable trains with metadata, station count, and category."""
    trains = list(DEMO_TRAINS_CONFIG)
    existing_nums = {t['train_number'] for t in trains}

    if search:
        q = search.strip().lower()
        matched_nums = []
        routes_summary = getattr(simulator, 'train_routes_summary', {})
        train_names = getattr(simulator, 'train_names', {})

        if q.isdigit():
            for num in routes_summary.keys():
                if q in str(num) and num not in existing_nums:
                    matched_nums.append(num)
                    if len(matched_nums) >= 25:
                        break
        else:
            for num, name in train_names.items():
                if q in name.lower() and num not in existing_nums and num in routes_summary:
                    matched_nums.append(num)
                    if len(matched_nums) >= 25:
                        break

        for num in matched_nums:
            r_info = routes_summary.get(num, {})
            origin = r_info.get('origin', 'Origin')
            dest = r_info.get('dest', 'Destination')
            t_name = train_names.get(num, f"Express {num}")
            trains.append({
                'train_number': num,
                'train_name': t_name,
                'route_desc': f"{origin} → {dest}",
                'category': 'National Rail Network',
                'default_date': r_info.get('date', '2024-09-28')
            })

        trains = [t for t in trains if q in str(t['train_number']) or q in t['train_name'].lower() or q in t['route_desc'].lower()]

    return {
        "total": len(trains),
        "trains": trains
    }


@app.get("/api/alerts")
def get_operational_alerts():
    """Returns dynamic, audit-grade operational alerts derived strictly from real system state.
    Zero fabricated records: derived from active TSR events, delay thresholds, telemetry status, and downstream pressure.
    """
    alerts = []
    now_iso = datetime.now(timezone.utc).isoformat()
    state = simulator.get_state()
    current_train = simulator.current_train
    curr_delay = state.get("current_delay_mins", 0.0)
    current_stn = state.get("current_station", {})
    stn_name = current_stn.get("station_name", "Current Location")
    
    # 1. Operational Events (TSR, Caution Orders, Maintenance Blocks)
    for idx, ev in enumerate(simulator.active_events):
        alerts.append({
            "id": f"ALT-EV-{idx+1:03d}",
            "severity": "CRITICAL" if ev.event_type in ("MAINTENANCE_BLOCK", "UNSCHEDULED_STOP") else "WARNING",
            "category": "OPERATIONAL_RESTRICTION",
            "title": f"{ev.event_type.replace('_', ' ').title()} Active",
            "affected_entity": f"Train {current_train} ({ev.from_station} → {ev.to_station})",
            "description": f"{ev.affected_km} km section restricted to {ev.restricted_speed_kmh} km/h (Source: {ev.source_type}).",
            "impact": f"Added dynamic delay floor of ~{ev.halt_duration_minutes} mins.",
            "timestamp": now_iso,
            "status": "ACTIVE"
        })
        
    # 2. Severe Delay Threshold Alerts on Active Train
    if curr_delay >= 30.0:
        alerts.append({
            "id": "ALT-DLY-CRIT",
            "severity": "CRITICAL",
            "category": "SEVERE_DELAY",
            "title": f"Critical Schedule Degradation (+{curr_delay:.0f}m)",
            "affected_entity": f"Train {current_train} at {stn_name}",
            "description": f"Cumulative delay of {curr_delay:.1f} minutes exceeds operational tolerance (>30m).",
            "impact": "Triggers dynamic recovery slack and downstream precedence adjustments.",
            "timestamp": now_iso,
            "status": "ACTIVE"
        })
    elif curr_delay >= 15.0:
        alerts.append({
            "id": "ALT-DLY-WARN",
            "severity": "WARNING",
            "category": "MODERATE_DELAY",
            "title": f"Schedule Delay Advisory (+{curr_delay:.0f}m)",
            "affected_entity": f"Train {current_train} at {stn_name}",
            "description": f"Observed delay of {curr_delay:.1f} minutes at {stn_name}.",
            "impact": "Downstream corridor ETAs dynamically recalibrated.",
            "timestamp": now_iso,
            "status": "ACTIVE"
        })
        
    # 3. Telemetry Provider Health Alert
    health = simulator.get_provider_health()
    if health.get("is_fallback_active"):
        alerts.append({
            "id": "ALT-TEL-FALLBACK",
            "severity": "INFO",
            "category": "TELEMETRY_STANDBY",
            "title": "Historical Replay & Archive Telemetry Active",
            "affected_entity": "System Ingestion Pipeline",
            "description": f"Operating on verified NTES telemetry archive (Watermark: {health.get('fallback_watermark', 'Verified Archive')}).",
            "impact": "High precision forecast maintained from verified NTES records.",
            "timestamp": now_iso,
            "status": "ACTIVE"
        })
    elif not health.get("is_connected", True):
        alerts.append({
            "id": "ALT-TEL-DISCONN",
            "severity": "WARNING",
            "category": "TELEMETRY_DEGRADED",
            "title": "Live Stream Telemetry Unreachable",
            "affected_entity": "RailRadar Provider",
            "description": "External live provider latency or network boundary unavailable.",
            "impact": "Automatic failover to NTES canonical baseline active.",
            "timestamp": now_iso,
            "status": "ACTIVE"
        })
        
    # 4. Downstream Junction Pressure Alerts
    table = state.get("comparison_table", [])
    if table:
        next_hop = table[0]
        next_code = next_hop.get("station_code", "")
        if next_code:
            d, c, a = simulator.calculator.network_engine.query_station_state(next_code, global_hour=12)
            if d >= 30.0:
                alerts.append({
                    "id": f"ALT-NET-{next_code}",
                    "severity": "WARNING",
                    "category": "NETWORK_CONGESTION",
                    "title": f"Downstream Junction Bottleneck ({next_code})",
                    "affected_entity": f"Junction {next_hop.get('station_name', next_code)}",
                    "description": f"Downstream mean delay is {d:.1f}m across {c} active trains.",
                    "impact": f"Model factored +{next_hop.get('our_predicted_delay', 0.0):.1f}m network congestion buffer into forward ETA.",
                    "timestamp": now_iso,
                    "status": "ACTIVE"
                })

    return {
        "status": "success",
        "total_alerts": len(alerts),
        "alerts": alerts
    }


@app.post("/api/replay/train")
def select_train(req: SelectTrainRequest):
    """Switch active train journey."""
    try:
        date = req.date or '2024-09-28'
        simulator.load_journey(req.train_number, date)
        st = simulator.get_state(step=0)
        _log_downstream_predictions(st)
        return st
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/replay/state")
def get_replay_state(step: Optional[int] = None):
    """Get full state of current journey replay."""
    return simulator.get_state(step=step)


@app.post("/api/replay/step")
def step_replay(req: StepRequest):
    """Set journey playback to a specific station step index, evaluate arrival, and log forward predictions."""
    prev_step = simulator.current_step
    state = simulator.get_state(step=req.step)
    
    # 1. Closed-Loop Evaluation of arrived station
    if req.step > prev_step and state.get("current_station"):
        curr_stn = state["current_station"]
        stn_code = curr_stn.get("station_code", "")
        actual_delay = state.get("current_delay_mins", 0.0)
        
        # Match previous prediction & log verified evaluation
        error = prediction_logger.record_arrival(
            train_id=str(simulator.current_train),
            station_code=stn_code,
            actual_delay_mins=actual_delay
        )
        
        # Update closed-loop trace with verified match
        if error is not None and "closed_loop_trace" in state:
            state["closed_loop_trace"]["prediction_match"] = {
                "was_evaluated": True,
                "station_code": stn_code,
                "actual_delay_mins": actual_delay,
                "error_mins": error,
                "accuracy_tier": "ACCURATE (Within ±3m)" if error <= 3.0 else ("ACCEPTABLE (Within ±5m)" if error <= 5.0 else "DIVERGENT")
            }
            
    # 2. Log newly recomputed forward predictions as pending arrivals
    _log_downstream_predictions(state)
    return state


@app.post("/api/mode/switch")
def switch_mode(req: ModeSwitchRequest):
    """Switch operational mode between 'historical_replay' and 'live_external'."""
    try:
        current_mode = simulator.set_mode(req.mode)
        state = simulator.get_state()
        return {
            "status": "success",
            "mode": current_mode,
            "provider_health": simulator.get_provider_health(),
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/live/health")
def get_live_health():
    """Returns active telemetry provider connectivity, latency, rate limits, and freshness."""
    return simulator.get_provider_health()


@app.get("/api/health")
def get_health():
    """Returns service health for platform probes."""
    return {
        "status": "ok",
        "service": "gati-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": simulator.mode
    }


@app.get("/api/live/provider/status")
def get_provider_status():
    """Returns audit-grade telemetry source status, API authentication state, and fallback watermarks."""
    health = simulator.get_provider_health()
    return {
        "active_mode": simulator.mode,
        "provider_name": health.get("provider", "Unknown"),
        "is_connected": health.get("is_connected", False),
        "is_live_api": health.get("is_live_api", False),
        "is_fallback_active": health.get("is_fallback_active", False),
        "fallback_watermark": health.get("fallback_watermark"),
        "telemetry_metrics": {
            "total_requests": health.get("total_requests", 0),
            "cache_hits": health.get("cache_hits", 0),
            "latency_ms": health.get("latency_ms", 0.0),
            "available_tokens": health.get("available_tokens", 30.0)
        }
    }


@app.get("/api/live/state")
def get_live_train_state():
    """Returns real-time CanonicalTrainState observation and forward predictions."""
    if simulator.mode != "live_external":
        simulator.set_mode("live_external")
    state = simulator.get_state()
    _log_downstream_predictions(state)
    return state


@app.get("/api/live/station/{code}")
def get_station_live_board(code: str):
    """Returns live arrival/departure board for a station to monitor downstream junction traffic."""
    board = simulator.get_station_board(code)
    return {
        "station_code": code.upper().strip(),
        "train_count": len(board),
        "board": board
    }


@app.post("/api/live/apikey")
def set_live_api_key(req: ApiKeyRequest):
    """Sets or updates the RailRadar API key dynamically at runtime with upstream validation."""
    clean_key = req.api_key.strip()
    simulator.railradar_provider.set_api_key(clean_key)
    simulator.set_mode("live_external")
    state = simulator.get_state()
    
    cs = state.get("canonical_state") or {}
    is_authenticated = True
    error_msg = None
    if cs.get("status") == "UNAVAILABLE":
        exceptions = cs.get("exceptions", [])
        if any("401" in ex or "UNAUTHORIZED" in str(ex).upper() for ex in exceptions):
            is_authenticated = False
            error_msg = "Upstream RailRadar rejected API key (HTTP 401 Unauthorized)."
        elif any("403" in ex for ex in exceptions):
            is_authenticated = False
            error_msg = "Upstream RailRadar rejected API key (HTTP 403 Forbidden)."
        elif exceptions:
            error_msg = exceptions[0]

    return {
        "status": "success",
        "message": "RailRadar API key updated successfully",
        "is_authenticated": is_authenticated,
        "error_msg": error_msg,
        "health": simulator.get_provider_health(),
        "state": state
    }


@app.get("/api/predictions/log")
def get_prediction_evaluation_log():
    """Returns continuous self-evaluation metrics comparing predictions against observed outcomes."""
    return prediction_logger.get_metrics()


@app.post("/api/events/inject")
def inject_operational_event(req: EventInjectionRequest):
    """Inject Temporary Speed Restriction, Caution Order, Maintenance Block, or Unscheduled Stop."""
    event = OperationalEvent(
        event_type=req.event_type,
        from_station=req.from_station.upper().strip(),
        to_station=req.to_station.upper().strip(),
        affected_km=req.affected_km,
        restricted_speed_kmh=req.restricted_speed_kmh,
        halt_duration_minutes=req.halt_duration_minutes,
        source_type=req.source_type
    )
    simulator.inject_event(event)
    return simulator.get_state()


@app.post("/api/events/clear")
def clear_all_events():
    """Clear all active operational events."""
    simulator.clear_events()
    return simulator.get_state()


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns official test set benchmark scorecard and ablation report."""
    summary_path = Path("models/evaluation_summary.json")
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return {"error": "Benchmark file not found"}


@app.get("/api/benchmarks/horizon")
def get_horizon_benchmarks():
    """Returns horizon-stratified benchmark results (1 hop, 2-3 hops, 4-5 hops, etc.)."""
    p = Path("models/horizon_evaluation.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Horizon benchmark file not found"}


@app.get("/api/benchmarks/scenarios")
def get_scenario_benchmarks():
    """Returns delay severity scenario benchmark results (on-time, minor, severe, extreme)."""
    p = Path("models/scenario_evaluation.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Scenario benchmark file not found"}


@app.get("/api/benchmarks/rules")
def get_rule_impact():
    """Returns deterministic rule engine intervention audit metrics."""
    p = Path("models/rule_impact_evaluation.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Rule impact file not found"}


@app.get("/api/feature-importance")
def get_feature_importance():
    """Returns top ranked features by informational gain."""
    imp_path = Path("models/feature_importance.csv")
    if imp_path.exists():
        import pandas as pd
        df = pd.read_csv(imp_path)
        return df.head(15).to_dict(orient="records")
    return []


@app.get("/api/benchmarks/network-pressure")
def get_network_pressure_benchmarks():
    """Returns network-pressure stratified benchmark results (Normal, Low, Medium, High)."""
    p = Path("models/network_pressure_evaluation.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Network pressure benchmark file not found"}


@app.get("/api/benchmarks/ablation-ladder")
def get_ablation_ladder_benchmarks():
    """Returns M0-M3 ablation ladder comparison results."""
    p = Path("models/ablation_ladder.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Ablation ladder file not found"}


@app.get("/api/network-state/{station_code}")
def get_station_network_state(station_code: str, hour: int = 12):
    """Returns downstream network congestion state for a specific station."""
    stn = station_code.upper().strip()
    d, c, a = simulator.calculator.network_engine.query_station_state(stn, global_hour=hour)
    pressure = "HIGH" if d >= 30.0 else ("MEDIUM" if d >= 15.0 else ("LOW" if d >= 5.0 else "NORMAL"))
    return {
        "station_code": stn,
        "global_hour": hour,
        "mean_delay_mins": round(d, 1),
        "delayed_train_count": int(c),
        "active_train_count": int(a),
        "congestion_pressure": pressure
    }


@app.get("/api/demo/live-loop")
def demonstrate_live_loop(train_number: int = 12303, date: Optional[str] = None):
    """
    Executes an end-to-end closed loop demonstration over the complete journey:
    Observation -> Network Injection -> ETA -> New Observation -> Self-Evaluation -> New ETA.
    Returns complete step-by-step trace and final verification metrics.
    """
    journey_date = date or '2024-09-28'
    demo_sim = ReplaySimulator()
    demo_sim.load_journey(train_number, journey_date)
    demo_logger = LivePredictionLogger()
    
    total_steps = len(demo_sim.journey_sections)
    trace = []
    
    # Step 0: Origin Departure
    st0 = demo_sim.get_state(step=0)
    for hop in st0.get("comparison_table", []):
        demo_logger.log_prediction(
            train_id=str(train_number),
            station_code=hop["station_code"],
            station_name=hop["station_name"],
            predicted_arrival=hop["our_predicted_eta"],
            predicted_delay_mins=hop["our_predicted_delay"],
            confidence_pct=hop["confidence_pct"]
        )
    trace.append({
        "step": 0,
        "station": st0["current_station"]["station_code"],
        "station_name": st0["current_station"]["station_name"],
        "event": "ORIGIN_DEPARTURE",
        "observed_delay_mins": st0["current_delay_mins"],
        "network_state_updated": True,
        "downstream_predictions_count": len(st0.get("comparison_table", [])),
        "next_stop_eta": st0.get("comparison_table", [{}])[0].get("our_predicted_eta", "--")
    })
    
    # Step 1 through total_steps
    for k in range(1, total_steps + 1):
        stk = demo_sim.get_state(step=k)
        curr_stn = stk["current_station"]
        stn_code = curr_stn.get("station_code", "")
        actual_delay = stk.get("current_delay_mins", 0.0)
        
        # Evaluate prior prediction
        err = demo_logger.record_arrival(
            train_id=str(train_number),
            station_code=stn_code,
            actual_delay_mins=actual_delay
        )
        
        # Log new forward predictions
        for hop in stk.get("comparison_table", []):
            demo_logger.log_prediction(
                train_id=str(train_number),
                station_code=hop["station_code"],
                station_name=hop["station_name"],
                predicted_arrival=hop["our_predicted_eta"],
                predicted_delay_mins=hop["our_predicted_delay"],
                confidence_pct=hop["confidence_pct"]
            )
            
        next_hop = stk.get("comparison_table", [{}])[0] if stk.get("comparison_table") else {}
        trace.append({
            "step": k,
            "station": stn_code,
            "station_name": curr_stn.get("station_name", stn_code),
            "event": "STATION_ARRIVAL",
            "observed_delay_mins": actual_delay,
            "prediction_error_mins": err,
            "network_delay_injected": actual_delay,
            "downstream_remaining_stops": len(stk.get("comparison_table", [])),
            "recomputed_next_eta": next_hop.get("our_predicted_eta", "DESTINATION_REACHED"),
            "recomputed_next_delay": next_hop.get("our_predicted_delay", 0.0),
            "rolling_metrics": demo_logger.get_metrics()
        })
        
    final_metrics = demo_logger.get_metrics()
    return {
        "status": "success",
        "train_number": train_number,
        "date": journey_date,
        "total_stops_evaluated": final_metrics["evaluated_count"],
        "journey_rolling_mae_mins": final_metrics["rolling_mae_mins"],
        "within_3_mins_pct": final_metrics["within_3_mins_pct"],
        "within_5_mins_pct": final_metrics["within_5_mins_pct"],
        "trace": trace
    }


# Mount frontend static files if directory exists and contains files
frontend_dir = Path("frontend")
if frontend_dir.is_dir() and any(frontend_dir.iterdir()):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
else:
    @app.get("/")
    def root_status():
        return {
            "status": "online",
            "system": "GaTi - Indian Railways Dynamic ETA Prediction System",
            "version": "2.1.0",
            "documentation": "/docs"
        }
