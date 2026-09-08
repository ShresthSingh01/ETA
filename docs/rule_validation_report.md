# GaTi — Rule Engine Validation & Boundary Assertion Dossier

> **Audited Test Suite**: [`tests/test_rule_engine.py`](file:///d:/ETA/tests/test_rule_engine.py)  
> **Status**: ALL 21 TESTS PASSING (100% SUCCESS)  
> **Target Problem Statement**: SIH 26028 (Rule Integrity & Boundary Validation)

---

## 1. Executive Summary

This dossier provides verifiable, automated test evidence demonstrating that the GaTi Rule Engine strictly enforces railway physical constraints while rejecting invalid, out-of-scope, expired, or malformed operational events.

---

## 2. Test Coverage & Verification Matrix (21 Passing Tests)

| # | Test Name | Assertion / Scenario Verified | Expected Outcome | Status |
|---|:---|:---|:---|:---:|
| 1 | `test_proof_1_ml_within_bounds_passes_through` | ML prediction (18m) safely within MPS floor (15m) and ceiling | Passed through unmodified, zero delta | **PASSED** |
| 2 | `test_proof_2_impossible_speed_clamped_to_floor` | ML predicts 10m on 100 km/h track (implies 150 km/h) | Clamped to physical MPS floor of 15.0m | **PASSED** |
| 3 | `test_proof_3_physics_derived_speed_restriction` | TSR of 30 km/h over 15 km on 100 km/h track | Exactly +21.0 minutes delay added via kinematics | **PASSED** |
| 4 | `test_proof_4_recovery_capped_at_timetable_allowance` | Late train: ML predicts 25m on 40m section (37.5% recovery) | Capped at 15% WTT cushion limit (34.0m) | **PASSED** |
| 5 | `test_proof_5_simultaneous_rules_audit_trail` | Concurrent MPS clamp, TSR restriction, and ceiling test | Full sequential audit trail generated with deltas | **PASSED** |
| 6 | `test_proof_6_caution_order_and_unscheduled_stop` | Form T/409 caution order + precedence loop hold | Both operational events sequentially compounded | **PASSED** |
| 7 | `test_proof_7_rule_provenance_metadata` | Audit records contain document references & clauses | `source_document` correctly populated | **PASSED** |
| 8 | `test_proof_8_signal_hold_operational_event` | Automatic signal clearance buffer hold (+3.0m) | Signal wait time added with audit reason | **PASSED** |
| 9 | `test_negative_zero_distance_and_zero_speed` | Malformed inputs: distance = 0 km, speed = 0 km/h | Handled gracefully without ZeroDivisionError | **PASSED** |
| 10 | `test_negative_affected_km_isolation` | Malformed TSR with negative affected distance (-5 km) | Rejected; section unaffected | **PASSED** |
| 11 | `test_event_wrong_section_isolation` | Caution order applied to unrelated section (CNB-ALJN) | Section DDU-PRYJ remains unmodified | **PASSED** |
| 12 | `test_duplicate_additive_events_on_same_section` | Multiple simultaneous caution orders on one track | Additive compounding handled cleanly | **PASSED** |
| 13 | `test_inactive_event_ignored` | Event with `active=False` passed to rule engine | Event ignored; zero adjustment | **PASSED** |
| 14 | `test_tsr_zero_affected_distance` | TSR with `affected_km = 0.0` | Event ignored; zero adjustment | **PASSED** |
| 15 | `test_tsr_speed_higher_than_normal` | TSR speed (130 km/h) higher than normal section speed | Disregarded; does not speed up train | **PASSED** |
| 16 | `test_early_train_negative_delay_no_recovery_clamp` | Train running early (negative delay: -10m) | Recovery cap is not triggered | **PASSED** |
| 17 | `test_tsr_affected_distance_clamped_to_section_length`| TSR affected distance (150 km) > section length (60 km)| Clamped strictly to section distance | **PASSED** |
| 18 | `test_rule_classification_registry_provenance` | 7-class rule registry and recovery cap heuristic proof | Verified complete metadata fields | **PASSED** |
| 19 | `test_negative_rule_scoping_and_boundaries` | Inactive, wrong zone, wrong division, expired, future rules | All out-of-scope rules safely rejected | **PASSED** |
| 20 | `test_operational_event_cancellation` | Journey cancellation event triggered | Final time = 0.0, audit trail generated | **PASSED** |
| 21 | `test_operational_event_reschedule_and_diversion` | Reschedule (+45m) and diversion loop (+25m) events | Exact buffer added with formal audit record | **PASSED** |

---

## 3. Audited Pytest Terminal Output

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-7.4.0, pluggy-1.6.0 -- D:\ETA\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\ETA
plugins: anyio-4.11.0, langsmith-0.6.2
collecting ... collected 21 items

tests/test_rule_engine.py::test_proof_1_ml_within_bounds_passes_through PASSED [  4%]
tests/test_rule_engine.py::test_proof_2_impossible_speed_clamped_to_floor PASSED [  9%]
tests/test_rule_engine.py::test_proof_3_physics_derived_speed_restriction PASSED [ 14%]
tests/test_rule_engine.py::test_proof_4_recovery_capped_at_timetable_allowance PASSED [ 19%]
tests/test_rule_engine.py::test_proof_5_simultaneous_rules_audit_trail PASSED [ 23%]
tests/test_rule_engine.py::test_proof_6_caution_order_and_unscheduled_stop PASSED [ 28%]
tests/test_rule_engine.py::test_proof_7_rule_provenance_metadata PASSED  [ 33%]
tests/test_rule_engine.py::test_proof_8_signal_hold_operational_event PASSED [ 38%]
tests/test_rule_engine.py::test_negative_zero_distance_and_zero_speed PASSED [ 42%]
tests/test_rule_engine.py::test_negative_affected_km_isolation PASSED    [ 47%]
tests/test_rule_engine.py::test_event_wrong_section_isolation PASSED     [ 52%]
tests/test_rule_engine.py::test_duplicate_additive_events_on_same_section PASSED [ 57%]
tests/test_rule_engine.py::test_inactive_event_ignored PASSED            [ 61%]
tests/test_rule_engine.py::test_tsr_zero_affected_distance PASSED        [ 66%]
tests/test_rule_engine.py::test_tsr_speed_higher_than_normal PASSED      [ 71%]
tests/test_rule_engine.py::test_early_train_negative_delay_no_recovery_clamp PASSED [ 76%]
tests/test_rule_engine.py::test_tsr_affected_distance_clamped_to_section_length PASSED [ 80%]
tests/test_rule_engine.py::test_rule_classification_registry_provenance PASSED [ 85%]
tests/test_rule_engine.py::test_negative_rule_scoping_and_boundaries PASSED [ 90%]
tests/test_rule_engine.py::test_operational_event_cancellation PASSED    [ 95%]
tests/test_rule_engine.py::test_operational_event_reschedule_and_diversion PASSED [100%]

============================= 21 passed in 0.04s ==============================
```
