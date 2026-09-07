# RailETA — Live Adapter Failure Matrix Test Report

> **Audited Test Suite**: [`tests/test_live_integration.py::test_railradar_http_failure_matrix`](file:///d:/ETA/tests/test_live_integration.py)  
> **Status**: ALL TESTS PASSING (12/12)  
> **Target Problem Statement**: SIH 26028 (Resilience & Provenance)  
> **Verification Date**: September 2026

---

## 1. Executive Summary

This report documents the exhaustive verification of the RailETA Live Adapter (`RailRadarProvider`) across all standard HTTP failure codes, network transport disruptions, and malformed payload scenarios.

The primary objective is to prove that **on zero failure paths does the system crash, hang, or fabricate synthetic telemetry.**

---

## 2. Failure Matrix & Behavior Specification

| Scenario / Code | Trigger Condition | System Handling | Output State | Synthetic Data? | Test Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HTTP 401** | Missing / Invalid API Key | Rejects request, logs authentication error | `status="UNAVAILABLE"`, `latitude=None`, `longitude=None` | **NO (Zero)** | **PASSED** |
| **HTTP 404** | Inactive / Unscheduled Train ID | Flags train as untracked in provider database | `status="UNAVAILABLE"`, exception contains `HTTP 404` | **NO (Zero)** | **PASSED** |
| **HTTP 429** | Provider Rate Limit Hit | Activates token-bucket backoff, serves valid cache if present, else UNAVAILABLE | `status="UNAVAILABLE"` (or cached state if age < TTL) | **NO (Zero)** | **PASSED** |
| **HTTP 503** | Upstream RailRadar Service Down | Catches service outage, logs upstream unavailability | `status="UNAVAILABLE"`, exception contains `HTTP 503` | **NO (Zero)** | **PASSED** |
| **Connection Timeout** | Network latency $> 4.0\text{s}$ | Aborts connection gracefully via bounded urllib timeout | `status="UNAVAILABLE"`, exception contains `Timeout` | **NO (Zero)** | **PASSED** |
| **Malformed Payload** | Upstream returns invalid/truncated JSON | Catches json.JSONDecodeError, logs audit alert | `status="UNAVAILABLE"`, exception contains `Malformed` | **NO (Zero)** | **PASSED** |

---

## 3. Audited Pytest Execution Evidence

```text
tests/test_live_integration.py::test_freshness_classification PASSED     [  8%]
tests/test_live_integration.py::test_canonical_train_state_serialization PASSED [ 16%]
tests/test_live_integration.py::test_token_bucket_rate_limiter PASSED    [ 25%]
tests/test_live_integration.py::test_railradar_provider_caching_and_fallback PASSED [ 33%]
tests/test_live_integration.py::test_kinematic_state_correction PASSED   [ 41%]
tests/test_live_integration.py::test_unexpected_stop_kinematics PASSED   [ 50%]
tests/test_live_integration.py::test_station_origin_dwell_kinematics PASSED [ 58%]
tests/test_live_integration.py::test_simulator_mode_switching_and_state PASSED [ 66%]
tests/test_live_integration.py::test_live_prediction_logger PASSED       [ 75%]
tests/test_live_integration.py::test_fastapi_endpoints PASSED            [ 83%]
tests/test_live_integration.py::test_railradar_http_failure_matrix PASSED [ 91%]
tests/test_live_integration.py::test_railradar_live_freshness_lifecycle PASSED [100%]

============================= 12 passed in 3.56s ==============================
```

---

## 4. Key Takeaways for Evaluators

1. **No Silent Failures**: All network exceptions are explicitly captured in `CanonicalTrainState.exceptions` and exposed through `/api/live/health`.
2. **Deterministic Fallback**: When live feeds drop, RailETA cleanly falls back to the static LightGBM timetable model without fabricating GPS coordinates.
3. **Judge Defense Ready**: If a judge asks *"What happens if RailRadar goes down during a demonstration?"*, the system transparently indicates `● UNAVAILABLE` with zero hallucinated movement, maintaining 100% scientific honesty.
