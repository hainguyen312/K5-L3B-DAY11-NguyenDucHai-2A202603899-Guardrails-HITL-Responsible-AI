"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    redacted_responses: int = 0
    layer_counts: dict[str, int] = field(default_factory=dict)

    def record(self, *, blocked: bool, layer: str | None, redacted: bool = False,
               judge_checked: bool = False, judge_failed: bool = False) -> None:
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1
        if redacted:
            self.redacted_responses += 1
        if judge_checked:
            self.judge_checks += 1
            if judge_failed:
                self.judge_fails += 1
        key = layer or "passed"
        self.layer_counts[key] = self.layer_counts.get(key, 0) + 1

    def check_metrics(self) -> list[Alert]:
        snap = self.snapshot()
        alerts: list[Alert] = []

        if self.total_requests and snap["block_rate"] > self.block_rate_threshold:
            alerts.append(Alert(
                metric="block_rate",
                value=round(snap["block_rate"], 3),
                threshold=self.block_rate_threshold,
                message=(
                    f"Block rate {snap['block_rate']:.0%} > {self.block_rate_threshold:.0%}: "
                    "possible mass attack, or guardrails producing too many false positives."
                ),
            ))

        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            alerts.append(Alert(
                metric="rate_limit_hits",
                value=float(self.rate_limit_hits),
                threshold=float(self.rate_limit_hit_threshold),
                message=(
                    f"{self.rate_limit_hits} requests rate-limited (threshold "
                    f"{self.rate_limit_hit_threshold}): suspected spam / cost attack."
                ),
            ))

        if self.judge_checks and snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            alerts.append(Alert(
                metric="judge_fail_rate",
                value=round(snap["judge_fail_rate"], 3),
                threshold=self.judge_fail_rate_threshold,
                message=(
                    f"LLM judge flagged {snap['judge_fail_rate']:.0%} of responses as UNSAFE: "
                    "model may be jailbroken or producing unreliable answers."
                ),
            ))

        self.alerts = alerts
        return alerts

    def export_json(self, filepath: str | None = None) -> str:
        self.check_metrics()
        data = self.snapshot()
        data.update({
            "redacted_responses": self.redacted_responses,
            "layer_counts": dict(self.layer_counts),
            "thresholds": {
                "block_rate": self.block_rate_threshold,
                "rate_limit_hits": self.rate_limit_hit_threshold,
                "judge_fail_rate": self.judge_fail_rate_threshold,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
