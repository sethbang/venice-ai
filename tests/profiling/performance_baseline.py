"""
Statistical Performance Baseline Tracking

Implements adaptive performance monitoring using statistical analysis
instead of fixed thresholds. Uses z-scores and rolling baselines to
detect performance degradation while accounting for environmental variance.
"""

import json
import statistics
from pathlib import Path
from typing import Any


class PerformanceBaseline:
    """
    Track performance baselines using statistical analysis.

    Uses rolling baselines (last 100 samples) and z-scores to detect
    performance degradation with 95% confidence threshold (z-score = 1.96).
    """

    # Maximum samples to keep for rolling baseline
    MAX_SAMPLES = 100

    def __init__(self, baseline_file: Path):
        """
        Initialize performance baseline tracker.

        Args:
            baseline_file: Path to JSON file storing baseline data
        """
        self.baseline_file = baseline_file
        self.baselines = self._load_baselines(baseline_file)

    def check_degradation(
        self, test_name: str, current_value: float, confidence: float = 0.95
    ) -> tuple[bool, str]:
        """
        Check if current performance value indicates degradation.

        Uses z-score analysis to determine if the current value is within
        the expected confidence interval based on historical baseline.

        Args:
            test_name: Name of the test/metric being tracked
            current_value: Current measured value
            confidence: Confidence level (default 0.95 for 95%)

        Returns:
            Tuple of (is_acceptable, message)
            - is_acceptable: True if performance is acceptable
            - message: Descriptive message about the result
        """
        # Calculate z-score threshold for given confidence level
        # 95% confidence = z-score of 1.96 (two-tailed)
        z_threshold = self._confidence_to_z_score(confidence)

        # Get or create baseline for this test
        if test_name not in self.baselines:
            self.baselines[test_name] = {"samples": [], "mean": 0.0, "std": 0.0}

        baseline = self.baselines[test_name]
        samples = baseline["samples"]

        # First run - no baseline yet
        if not samples:
            samples.append(current_value)
            baseline["mean"] = current_value
            baseline["std"] = 0.0
            self._save_baselines()

            return (
                True,
                f"First run for '{test_name}' - baseline initialized with {current_value:.2f}",
            )

        # Second run - need at least 2 samples for std deviation
        if len(samples) == 1:
            samples.append(current_value)
            baseline["mean"] = statistics.mean(samples)
            baseline["std"] = 0.0  # Can't calculate std with only 2 samples reliably
            self._save_baselines()

            return (
                True,
                f"Second run for '{test_name}' - building baseline (current: {current_value:.2f}, mean: {baseline['mean']:.2f})",
            )

        # Calculate z-score: (current_value - mean) / std
        mean = baseline["mean"]
        std = baseline["std"]

        # Handle edge case where std is 0 (all samples identical)
        if std == 0:
            # If current value matches mean, accept it
            if abs(current_value - mean) < 0.01:
                is_acceptable = True
                z_score = 0.0
            else:
                # First deviation from mean - accept and update baseline
                is_acceptable = True
                z_score = 0.0  # Can't calculate meaningful z-score yet
        else:
            z_score = (current_value - mean) / std
            is_acceptable = abs(z_score) <= z_threshold

        # Update rolling baseline
        samples.append(current_value)
        if len(samples) > self.MAX_SAMPLES:
            samples.pop(0)  # Remove oldest sample

        # Recalculate statistics
        baseline["mean"] = statistics.mean(samples)
        baseline["std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0

        self._save_baselines()

        # Generate informative message
        if is_acceptable:
            msg = (
                f"✓ '{test_name}' within baseline: "
                f"current={current_value:.2f}, mean={mean:.2f}, std={std:.2f}, "
                f"z-score={z_score:.2f} (threshold=±{z_threshold:.2f})"
            )
        else:
            msg = (
                f"✗ '{test_name}' degradation detected: "
                f"current={current_value:.2f}, mean={mean:.2f}, std={std:.2f}, "
                f"z-score={z_score:.2f} (threshold=±{z_threshold:.2f}) "
                f"[{len(samples)} samples]"
            )

        return (is_acceptable, msg)

    def _confidence_to_z_score(self, confidence: float) -> float:
        """
        Convert confidence level to z-score threshold.

        Common values:
        - 0.90 (90%) = 1.645
        - 0.95 (95%) = 1.96
        - 0.99 (99%) = 2.576

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Z-score threshold for two-tailed test
        """
        # Map common confidence levels to z-scores
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}

        if confidence in z_scores:
            return z_scores[confidence]

        # Default to 95% if unknown confidence level
        return 1.96

    def _load_baselines(self, baseline_file: Path) -> dict[str, Any]:
        """
        Load baselines from JSON file.

        Args:
            baseline_file: Path to baseline JSON file

        Returns:
            Dictionary of baseline data, or empty dict if file doesn't exist
        """
        if not baseline_file.exists():
            return {}

        try:
            with open(baseline_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # If file is corrupted, start fresh
            return {}

    def _save_baselines(self) -> None:
        """Save current baselines to JSON file."""
        # Ensure directory exists
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.baseline_file, "w") as f:
                json.dump(self.baselines, f, indent=2)
        except OSError as e:
            # Log error but don't fail the test
            print(f"Warning: Failed to save baselines: {e}")
