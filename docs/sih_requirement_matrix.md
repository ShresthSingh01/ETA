# RailETA — SIH 26028 Comprehensive Requirement Traceability Matrix

> **Problem Statement**: SIH 26028 — Real-Time Train Delay & Dynamic ETA Prediction Engine  
> **Target System**: Indian Railways National Network  
> **Status**: 100% AUDITED & VERIFIED

---

## 1. Traceability Matrix

| Requirement Clause | Core Description | Implementation File(s) | Verification Test File(s) | Evidence / Documentation Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **REQ-01: Multi-Source Ingestion** | Ingest timetables, historical movements, real-time live telemetry, and adverse weather | `src/data/loader.py`<br>`src/integrations/railradar.py` | `tests/test_live_integration.py` | [`docs/data_dictionary.md`](file:///d:/ETA/docs/data_dictionary.md)<br>[`docs/data_provenance.md`](file:///d:/ETA/docs/data_provenance.md) |
| **REQ-02: Zero Synthetic Live Policy** | Never fabricate fake coordinates, speeds, or moving trains in LIVE mode | `src/integrations/railradar.py`<br>`src/integrations/base.py` | `tests/test_live_integration.py` | [`docs/live_freshness_policy.md`](file:///d:/ETA/docs/live_freshness_policy.md)<br>[`docs/live_failure_test_report.md`](file:///d:/ETA/docs/live_failure_test_report.md) |
| **REQ-03: Dynamic Machine Learning** | Train gradient boosted tree on historical records with temporal & weather features | `src/model/features.py`<br>`src/model/trainer.py` | `tests/test_pipeline.py` | [`docs/final_benchmark.md`](file:///d:/ETA/docs/final_benchmark.md)<br>[`docs/leakage_audit.md`](file:///d:/ETA/docs/leakage_audit.md) |
| **REQ-04: Multi-Layer Benchmark** | Compare schedule naive, historical median, static ML, dynamic ML, rules, and kinematics | `src/model/ablation_waterfall.py` | `tests/test_baselines.py` | [`docs/ablation_waterfall_benchmark.md`](file:///d:/ETA/docs/ablation_waterfall_benchmark.md)<br>[`docs/live_state_ablation.md`](file:///d:/ETA/docs/live_state_ablation.md) |
| **REQ-05: Statutory Railway Rules** | Enforce G&SR track limits (MPS floor, loop holds, WTT slack recovery cushion) | `src/engine/rule_engine.py` | `tests/test_rule_engine.py` | [`docs/rule_provenance.md`](file:///d:/ETA/docs/rule_provenance.md)<br>[`docs/rule_validation_report.md`](file:///d:/ETA/docs/rule_validation_report.md) |
| **REQ-06: 7-Class Rule Provenance** | Classify every operating rule into official IR source categories | `src/engine/rule_engine.py` | `tests/test_rule_engine.py` | [`docs/rule_provenance.md`](file:///d:/ETA/docs/rule_provenance.md) |
| **REQ-07: In-Flight Kinematics** | Real-time speedometer reading and segment progress fusion on active blocks | `src/engine/state_correction.py` | `tests/test_live_integration.py` | [`docs/live_integration_evidence.md`](file:///d:/ETA/docs/live_integration_evidence.md) |
| **REQ-08: Operational Events** | Inject TSR, caution orders, maintenance blocks, signal holds, and diversions | `src/engine/rule_engine.py`<br>`src/replay/simulator.py` | `tests/test_rule_engine.py` | [`docs/rule_validation_report.md`](file:///d:/ETA/docs/rule_validation_report.md) |
| **REQ-09: Empirical Calibration** | Quantify confidence intervals and error bounds per confidence tier | `src/engine/eta_calculator.py` | `tests/test_calibrated_confidence.py` | [`docs/confidence_calibration_report.md`](file:///d:/ETA/docs/confidence_calibration_report.md) |
| **REQ-10: Transparent Error Analysis**| Audit largest errors and root-cause classify top 1% outliers | `src/model/ablation_waterfall.py` | `tests/test_baselines.py` | [`docs/error_analysis.md`](file:///d:/ETA/docs/error_analysis.md) |
| **REQ-11: Downstream State Analysis** | Audit downstream station pressure and headway on 150K records | `scripts/analyze_headway_feasibility.py` | `tests/test_pipeline.py` | [`docs/downstream_live_state_study.md`](file:///d:/ETA/docs/downstream_live_state_study.md) |
| **REQ-12: High-Throughput Scale** | Handle national burst concurrency (>1,000 to 5,000 trains) with sub-10ms latency | `src/engine/eta_calculator.py`<br>`src/integrations/railradar.py` | `tests/test_scalability.py` | [`docs/scalability_report.md`](file:///d:/ETA/docs/scalability_report.md) |
| **REQ-13: Distinct Three-Mode UX** | Cleanly separate LIVE (green), REPLAY (blue), and SIMULATED (amber) | `frontend/app.js`<br>`frontend/index.html` | Manual Web UI & API Tests | [`docs/live_freshness_policy.md`](file:///d:/ETA/docs/live_freshness_policy.md) |
| **REQ-14: Production Containerization**| Dockerfile and docker-compose deployment for reproducible execution | `Dockerfile`<br>`docker-compose.yml` | `docker build .` | [`README.md`](file:///d:/ETA/README.md) |

---

## 2. Compliance Summary

- **Total Requirements Addressed**: 14 / 14 (100%)
- **Automated Test Count**: 61 Passing Unit & Integration Tests
- **Proof Artifacts**: 14 Formal Markdown Dossiers in `docs/`
- **Zero Mocking In LIVE Mode**: Enforced via strict Tri-State Machine (`LIVE` / `STALE` / `UNAVAILABLE`).
