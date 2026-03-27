"""
Analyze Layer — Event Correlator
Section 5.4.3 (Phase 1) and Section 7.2 (Phase 2 weighted fusion).

Phase 1: Temporal grouping of co-occurring anomaly flags within a
         5-window co-occurrence window. Severity = max individual score.

Phase 2: Weighted composite severity score combining:
         - mean anomaly score
         - number of contributing metric streams
         - anomaly duration (window count)
         Weighting coefficients empirically optimised on semi-synthetic dataset.
"""

import logging
from collections import deque

logger = logging.getLogger(__name__)

# Phase 2 weighting coefficients (Section 7.2)
W_SCORE = 0.5
W_BREADTH = 0.3
W_DURATION = 0.2


class EventCorrelator:
    """
    Groups temporally proximate anomaly detections into composite
    anomaly events and computes a unified severity score.
    """

    def __init__(self, config: dict):
        self.co_window = config["analyze"]["event_correlator"]["co_occurrence_window"]
        self._buffer = deque(maxlen=self.co_window)
        self._active_event_duration = 0
        logger.info(
            f"EventCorrelator initialised — co-occurrence window={self.co_window}"
        )

    def _phase2_severity(
        self,
        scores: list,
        flagged: list,
        duration: int,
    ) -> float:
        """
        Phase 2 weighted composite severity score — Section 7.2.
        score    ∈ [0,1]: mean IF anomaly score
        breadth  ∈ [0,1]: fraction of metric streams flagged (max 4)
        duration ∈ [0,1]: normalised anomaly duration (cap at 20 windows)
        """
        score_component = sum(scores) / len(scores) if scores else 0.0
        breadth_component = len(set(flagged)) / 4.0
        duration_component = min(duration / 20.0, 1.0)

        composite = (
            W_SCORE * score_component
            + W_BREADTH * breadth_component
            + W_DURATION * duration_component
        )
        return round(composite, 4)

    def correlate(
        self,
        zscore_result: dict,
        if_result: dict,
        window: list,
    ) -> dict | None:
        """
        Receives results from both detection stages.
        Returns a composite anomaly event if one is formed, else None.
        """
        is_anomaly = zscore_result.get("passed_gate", False) and if_result.get(
            "is_anomaly", False
        )

        self._buffer.append(
            {
                "is_anomaly": is_anomaly,
                "anomaly_score": if_result.get("anomaly_score", 0.0),
                "flagged_metrics": zscore_result.get("flagged_metrics", []),
                "timestamp": window[-1]["timestamp"] if window else None,
            }
        )

        # Check if we have co-occurring anomalies in the buffer
        anomalous_entries = [e for e in self._buffer if e["is_anomaly"]]

        if not anomalous_entries:
            self._active_event_duration = 0
            return None

        self._active_event_duration += 1

        # Build composite event
        all_scores = [e["anomaly_score"] for e in anomalous_entries]
        all_flagged = []
        for e in anomalous_entries:
            all_flagged.extend(e["flagged_metrics"])

        severity = self._phase2_severity(
            scores=all_scores,
            flagged=all_flagged,
            duration=self._active_event_duration,
        )

        # Severity band mapping
        if severity >= 0.75:
            severity_band = "CRITICAL"
        elif severity >= 0.50:
            severity_band = "HIGH"
        elif severity >= 0.25:
            severity_band = "MEDIUM"
        else:
            severity_band = "LOW"

        event = {
            "event_type": "composite_anomaly",
            "severity_score": severity,
            "severity_band": severity_band,
            "contributing_metrics": list(set(all_flagged)),
            "anomaly_duration_windows": self._active_event_duration,
            "detection_timestamp": anomalous_entries[-1]["timestamp"],
            "component_scores": all_scores,
        }

        logger.info(
            f"Anomaly event formed — severity={severity} ({severity_band}), "
            f"metrics={event['contributing_metrics']}, "
            f"duration={self._active_event_duration} windows"
        )
        return event
