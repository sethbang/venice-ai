"""
Benchmark Report Generation for Scheduler Strategy Analysis
==========================================================

This module provides comprehensive reporting capabilities for benchmark results,
supporting multiple output formats and comparative analysis between different
scheduler strategies and configurations.

Key Components:
    * BenchmarkReporter: Main reporting class with multiple output formats
    * Console reporting with formatted tables and color coding
    * JSON reports for automated analysis and CI/CD integration
    * XML reports for integration with testing frameworks
    * Comparative analysis between different strategies

Usage:
    >>> reporter = BenchmarkReporter()
    >>> console_report = reporter.generate_console_report(results)
    >>> json_report = reporter.generate_json_report(results)
    >>> reporter.save_reports(results, output_dir="benchmark_results/")
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from tests.benchmarks.metrics import BenchmarkResults


class BenchmarkReporter:
    """
    Comprehensive benchmark reporting with multiple output formats.

    Generates detailed reports from benchmark results including performance
    metrics, visual formatting, and comparative analysis capabilities.
    Supports console, JSON, and XML output formats.
    """

    def __init__(self, use_colors: bool = True):
        """
        Initialize the benchmark reporter.

        Args:
            use_colors: Whether to use ANSI color codes in console output
        """
        self.use_colors = use_colors

        # Color codes for console output (if enabled)
        if use_colors:
            self.colors = {
                "header": "\033[95m",
                "blue": "\033[94m",
                "cyan": "\033[96m",
                "green": "\033[92m",
                "warning": "\033[93m",
                "fail": "\033[91m",
                "bold": "\033[1m",
                "underline": "\033[4m",
                "end": "\033[0m",
            }
        else:
            self.colors = dict.fromkeys(
                ["header", "blue", "cyan", "green", "warning", "fail", "bold", "underline", "end"],
                "",
            )

    def generate_console_report(self, results: BenchmarkResults) -> str:
        """
        Generate a comprehensive console-friendly report.

        Args:
            results: Benchmark results to report

        Returns:
            Formatted console report as a string
        """
        report_lines = []
        c = self.colors  # Shorthand for colors

        # Header
        report_lines.extend(
            [
                f"{c['bold']}{c['header']}{'=' * 80}{c['end']}",
                f"{c['bold']}{c['header']}BENCHMARK RESULTS: {results.scenario_name.upper()}{c['end']}",
                f"{c['bold']}{c['header']}{'=' * 80}{c['end']}",
                f"{c['bold']}Strategy:{c['end']} {c['cyan']}{results.strategy_name}{c['end']}",
                f"{c['bold']}Duration:{c['end']} {c['cyan']}{results.duration:.2f}s{c['end']}",
                f"{c['bold']}Timestamp:{c['end']} {c['cyan']}{results.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}{c['end']}",
                "",
            ]
        )

        # Throughput Section
        report_lines.extend(
            [
                f"{c['bold']}{c['blue']}📊 THROUGHPUT METRICS{c['end']}",
                f"{c['bold']}{'-' * 40}{c['end']}",
                f"Average RPS:     {c['green']}{results.avg_throughput:>8.2f}{c['end']}",
                f"Peak RPS:        {c['green']}{results.peak_throughput:>8.2f}{c['end']}",
                f"Total Requests:  {c['cyan']}{results.total_requests:>8,d}{c['end']}",
                f"Successful:      {c['green']}{results.successful_requests:>8,d}{c['end']} ({self._percentage(results.successful_requests, results.total_requests):.1f}%)",
                f"Failed:          {c['fail'] if results.failed_requests > 0 else c['green']}{results.failed_requests:>8,d}{c['end']} ({self._percentage(results.failed_requests, results.total_requests):.1f}%)",
                "",
            ]
        )

        # Latency Section
        report_lines.extend(
            [
                f"{c['bold']}{c['blue']}⚡ LATENCY METRICS (milliseconds){c['end']}",
                f"{c['bold']}{'-' * 40}{c['end']}",
                f"Minimum:         {c['green']}{results.min_latency:>8.2f}{c['end']}",
                f"Mean:            {c['cyan']}{results.mean_latency:>8.2f}{c['end']}",
                f"P50 (Median):    {c['cyan']}{results.p50_latency:>8.2f}{c['end']}",
                f"P95:             {c['warning'] if results.p95_latency > results.mean_latency * 3 else c['cyan']}{results.p95_latency:>8.2f}{c['end']}",
                f"P99:             {c['warning'] if results.p99_latency > results.mean_latency * 5 else c['cyan']}{results.p99_latency:>8.2f}{c['end']}",
                f"Maximum:         {c['fail'] if results.max_latency > results.mean_latency * 10 else c['cyan']}{results.max_latency:>8.2f}{c['end']}",
                "",
            ]
        )

        # Rate Limiting Section
        efficiency_color = self._get_efficiency_color(results.rate_limit_efficiency)
        report_lines.extend(
            [
                f"{c['bold']}{c['blue']}🚦 RATE LIMIT PERFORMANCE{c['end']}",
                f"{c['bold']}{'-' * 40}{c['end']}",
                f"Efficiency:      {c[efficiency_color]}{results.rate_limit_efficiency:>8.1f}%{c['end']}",
                f"Theoretical Max: {c['cyan']}{results.theoretical_max_rps:>8.2f}{c['end']} RPS",
                f"Achieved:        {c['cyan']}{results.achieved_percentage:>8.1f}%{c['end']} of theoretical",
                f"Violations:      {c['fail'] if results.rate_limit_violations > 0 else c['green']}{results.rate_limit_violations:>8,d}{c['end']}",
                "",
            ]
        )

        # Concurrency Section
        report_lines.extend(
            [
                f"{c['bold']}{c['blue']}🔄 CONCURRENCY METRICS{c['end']}",
                f"{c['bold']}{'-' * 40}{c['end']}",
                f"Max Concurrent:  {c['cyan']}{results.max_concurrent:>8,d}{c['end']}",
                f"Avg Concurrent:  {c['cyan']}{results.avg_concurrent:>8.2f}{c['end']}",
            ]
        )

        if results.configured_limit > 0:
            utilization = (results.avg_concurrent / results.configured_limit) * 100
            util_color = "green" if utilization > 70 else "warning" if utilization > 30 else "fail"
            report_lines.extend(
                [
                    f"Configured Limit:{c['cyan']}{results.configured_limit:>8,d}{c['end']}",
                    f"Utilization:     {c[util_color]}{utilization:>8.1f}%{c['end']}",
                ]
            )

        report_lines.append("")

        # Queue Performance Section
        if results.avg_queue_wait > 0 or results.max_queue_depth > 0:
            report_lines.extend(
                [
                    f"{c['bold']}{c['blue']}📋 QUEUE PERFORMANCE{c['end']}",
                    f"{c['bold']}{'-' * 40}{c['end']}",
                    f"Avg Queue Wait:  {c['cyan']}{results.avg_queue_wait:>8.2f}{c['end']} ms",
                    f"Max Queue Depth: {c['cyan']}{results.max_queue_depth:>8,d}{c['end']}",
                    "",
                ]
            )

        # Error Breakdown (if any errors)
        if results.error_breakdown:
            report_lines.extend(
                [
                    f"{c['bold']}{c['fail']}❌ ERROR BREAKDOWN{c['end']}",
                    f"{c['bold']}{'-' * 40}{c['end']}",
                ]
            )

            for error_type, count in sorted(
                results.error_breakdown.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = self._percentage(count, results.total_requests)
                report_lines.append(
                    f"{error_type[:30]:.<30} {c['fail']}{count:>5,d}{c['end']} ({percentage:.1f}%)"
                )

            report_lines.append("")

        # Performance Summary
        performance_score = self._calculate_performance_score(results)
        score_color = self._get_score_color(performance_score)

        report_lines.extend(
            [
                f"{c['bold']}{c['blue']}🎯 PERFORMANCE SUMMARY{c['end']}",
                f"{c['bold']}{'-' * 40}{c['end']}",
                f"Overall Score:   {c[score_color]}{performance_score:>8.1f}/100{c['end']}",
                f"Status:          {c[score_color]}{self._get_performance_status(performance_score)}{c['end']}",
                "",
            ]
        )

        # Recommendations
        recommendations = self._generate_recommendations(results)
        if recommendations:
            report_lines.extend(
                [
                    f"{c['bold']}{c['warning']}💡 RECOMMENDATIONS{c['end']}",
                    f"{c['bold']}{'-' * 40}{c['end']}",
                ]
            )
            for rec in recommendations:
                report_lines.append(f"• {rec}")
            report_lines.append("")

        # Footer
        report_lines.extend(
            [
                f"{c['bold']}{c['header']}{'=' * 80}{c['end']}",
                f"{c['bold']}Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{c['end']}",
                f"{c['bold']}{c['header']}{'=' * 80}{c['end']}",
            ]
        )

        return "\n".join(report_lines)

    def generate_json_report(self, results: BenchmarkResults) -> str:
        """
        Generate a JSON report for programmatic analysis.

        Args:
            results: Benchmark results to report

        Returns:
            JSON-formatted report as a string
        """
        # Convert dataclass to dict and add computed metrics
        data = asdict(results)

        # Add computed metrics
        data["computed_metrics"] = {
            "error_rate_percent": self._percentage(results.failed_requests, results.total_requests),
            "success_rate_percent": self._percentage(
                results.successful_requests, results.total_requests
            ),
            "latency_variance": results.p99_latency - results.p50_latency
            if results.p50_latency > 0
            else 0,
            "efficiency_rating": self._get_efficiency_rating(results.rate_limit_efficiency),
            "performance_score": self._calculate_performance_score(results),
            "concurrency_utilization_percent": (
                (results.avg_concurrent / results.configured_limit) * 100
                if results.configured_limit > 0
                else 0
            ),
        }

        # Add metadata
        data["report_metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "generator_version": "1.0.0",
            "format_version": "1.0",
        }

        # Add analysis
        data["analysis"] = {
            "recommendations": self._generate_recommendations(results),
            "performance_issues": self._detect_performance_issues(results),
            "status": self._get_performance_status(self._calculate_performance_score(results)),
        }

        return json.dumps(data, indent=2, default=str)

    def generate_xml_report(self, results: BenchmarkResults) -> str:
        """
        Generate an XML report for integration with testing frameworks.

        Args:
            results: Benchmark results to report

        Returns:
            XML-formatted report as a string
        """
        root = ET.Element("benchmark_results")
        root.set("version", "1.0")
        root.set("generated_at", datetime.now().isoformat())

        # Basic info
        info = ET.SubElement(root, "test_info")
        ET.SubElement(info, "scenario").text = results.scenario_name
        ET.SubElement(info, "strategy").text = results.strategy_name
        ET.SubElement(info, "duration").text = str(results.duration)
        ET.SubElement(info, "start_time").text = results.start_time.isoformat()
        ET.SubElement(info, "end_time").text = results.end_time.isoformat()

        # Throughput metrics
        throughput = ET.SubElement(root, "throughput")
        ET.SubElement(throughput, "average_rps").text = str(results.avg_throughput)
        ET.SubElement(throughput, "peak_rps").text = str(results.peak_throughput)
        ET.SubElement(throughput, "total_requests").text = str(results.total_requests)
        ET.SubElement(throughput, "successful_requests").text = str(results.successful_requests)
        ET.SubElement(throughput, "failed_requests").text = str(results.failed_requests)

        # Latency metrics
        latency = ET.SubElement(root, "latency")
        latency.set("unit", "milliseconds")
        ET.SubElement(latency, "min").text = str(results.min_latency)
        ET.SubElement(latency, "mean").text = str(results.mean_latency)
        ET.SubElement(latency, "p50").text = str(results.p50_latency)
        ET.SubElement(latency, "p95").text = str(results.p95_latency)
        ET.SubElement(latency, "p99").text = str(results.p99_latency)
        ET.SubElement(latency, "max").text = str(results.max_latency)

        # Rate limiting
        rate_limit = ET.SubElement(root, "rate_limiting")
        ET.SubElement(rate_limit, "efficiency_percent").text = str(results.rate_limit_efficiency)
        ET.SubElement(rate_limit, "violations").text = str(results.rate_limit_violations)
        ET.SubElement(rate_limit, "theoretical_max_rps").text = str(results.theoretical_max_rps)

        # Concurrency
        concurrency = ET.SubElement(root, "concurrency")
        ET.SubElement(concurrency, "max_concurrent").text = str(results.max_concurrent)
        ET.SubElement(concurrency, "avg_concurrent").text = str(results.avg_concurrent)
        ET.SubElement(concurrency, "configured_limit").text = str(results.configured_limit)

        # Analysis
        analysis = ET.SubElement(root, "analysis")
        ET.SubElement(analysis, "performance_score").text = str(
            self._calculate_performance_score(results)
        )
        ET.SubElement(analysis, "status").text = self._get_performance_status(
            self._calculate_performance_score(results)
        )

        # Format XML with proper indentation
        self._indent_xml(root)
        return ET.tostring(root, encoding="unicode")

    def compare_results(self, baseline: BenchmarkResults, comparison: BenchmarkResults) -> str:
        """
        Generate a comparison report between two benchmark results.

        Args:
            baseline: Baseline benchmark results
            comparison: Comparison benchmark results

        Returns:
            Formatted comparison report
        """
        c = self.colors
        report_lines = []

        # Header
        report_lines.extend(
            [
                f"{c['bold']}{c['header']}{'=' * 80}{c['end']}",
                f"{c['bold']}{c['header']}BENCHMARK COMPARISON{c['end']}",
                f"{c['bold']}{c['header']}{'=' * 80}{c['end']}",
                f"{c['bold']}Baseline:{c['end']} {c['cyan']}{baseline.strategy_name} ({baseline.scenario_name}){c['end']}",
                f"{c['bold']}Comparison:{c['end']} {c['cyan']}{comparison.strategy_name} ({comparison.scenario_name}){c['end']}",
                "",
            ]
        )

        # Throughput comparison
        throughput_change = self._calculate_percentage_change(
            baseline.avg_throughput, comparison.avg_throughput
        )
        throughput_color = (
            "green" if throughput_change > 0 else "fail" if throughput_change < -5 else "warning"
        )

        report_lines.extend(
            [
                f"{c['bold']}{c['blue']}📊 THROUGHPUT COMPARISON{c['end']}",
                f"{c['bold']}{'-' * 50}{c['end']}",
                f"{'Metric':<20} {'Baseline':<12} {'Comparison':<12} {'Change':<10}",
                f"{'-' * 50}",
                f"{'Avg RPS':<20} {baseline.avg_throughput:<12.2f} {comparison.avg_throughput:<12.2f} {c[throughput_color]}{throughput_change:>+7.1f}%{c['end']}",
            ]
        )

        peak_change = self._calculate_percentage_change(
            baseline.peak_throughput, comparison.peak_throughput
        )
        peak_color = "green" if peak_change > 0 else "fail" if peak_change < -5 else "warning"
        report_lines.append(
            f"{'Peak RPS':<20} {baseline.peak_throughput:<12.2f} {comparison.peak_throughput:<12.2f} {c[peak_color]}{peak_change:>+7.1f}%{c['end']}"
        )

        # Latency comparison (lower is better)
        p50_change = self._calculate_percentage_change(baseline.p50_latency, comparison.p50_latency)
        p50_color = "green" if p50_change < 0 else "fail" if p50_change > 10 else "warning"

        p99_change = self._calculate_percentage_change(baseline.p99_latency, comparison.p99_latency)
        p99_color = "green" if p99_change < 0 else "fail" if p99_change > 10 else "warning"

        report_lines.extend(
            [
                "",
                f"{c['bold']}{c['blue']}⚡ LATENCY COMPARISON (ms){c['end']}",
                f"{c['bold']}{'-' * 50}{c['end']}",
                f"{'P50 Latency':<20} {baseline.p50_latency:<12.2f} {comparison.p50_latency:<12.2f} {c[p50_color]}{p50_change:>+7.1f}%{c['end']}",
                f"{'P99 Latency':<20} {baseline.p99_latency:<12.2f} {comparison.p99_latency:<12.2f} {c[p99_color]}{p99_change:>+7.1f}%{c['end']}",
            ]
        )

        # Summary
        baseline_score = self._calculate_performance_score(baseline)
        comparison_score = self._calculate_performance_score(comparison)
        score_change = comparison_score - baseline_score
        score_color = "green" if score_change > 0 else "fail" if score_change < -5 else "warning"

        report_lines.extend(
            [
                "",
                f"{c['bold']}{c['blue']}🎯 OVERALL COMPARISON{c['end']}",
                f"{c['bold']}{'-' * 50}{c['end']}",
                f"{'Performance Score':<20} {baseline_score:<12.1f} {comparison_score:<12.1f} {c[score_color]}{score_change:>+7.1f}{c['end']}",
                "",
                f"{c['bold']}Winner:{c['end']} {c['green'] if comparison_score > baseline_score else c['fail']}{comparison.strategy_name if comparison_score > baseline_score else baseline.strategy_name}{c['end']}",
                f"{c['bold']}{c['header']}{'=' * 80}{c['end']}",
            ]
        )

        return "\n".join(report_lines)

    def save_reports(
        self, results: BenchmarkResults, output_dir: str = "benchmark_results"
    ) -> dict[str, str]:
        """
        Save all report formats to files.

        Args:
            results: Benchmark results to save
            output_dir: Directory to save reports to

        Returns:
            Dictionary mapping format names to saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{results.strategy_name}_{results.scenario_name}_{timestamp}"

        saved_files = {}

        # Save console report
        console_path = output_path / f"{base_filename}_console.txt"
        with open(console_path, "w", encoding="utf-8") as f:
            f.write(self.generate_console_report(results))
        saved_files["console"] = str(console_path)

        # Save JSON report
        json_path = output_path / f"{base_filename}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.generate_json_report(results))
        saved_files["json"] = str(json_path)

        # Save XML report
        xml_path = output_path / f"{base_filename}.xml"
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(self.generate_xml_report(results))
        saved_files["xml"] = str(xml_path)

        return saved_files

    # Helper methods

    def _percentage(self, part: int, total: int) -> float:
        """Calculate percentage safely."""
        return (part / total * 100) if total > 0 else 0.0

    def _calculate_percentage_change(self, baseline: float, comparison: float) -> float:
        """Calculate percentage change from baseline to comparison."""
        if baseline == 0:
            return 0.0 if comparison == 0 else float("inf")
        return ((comparison - baseline) / baseline) * 100

    def _get_efficiency_color(self, efficiency: float) -> str:
        """Get color code based on efficiency percentage."""
        if efficiency >= 90:
            return "green"
        elif efficiency >= 70:
            return "cyan"
        elif efficiency >= 50:
            return "warning"
        else:
            return "fail"

    def _get_efficiency_rating(self, efficiency: float) -> str:
        """Get text rating based on efficiency."""
        if efficiency >= 95:
            return "Excellent"
        elif efficiency >= 85:
            return "Good"
        elif efficiency >= 70:
            return "Fair"
        elif efficiency >= 50:
            return "Poor"
        else:
            return "Critical"

    def _calculate_performance_score(self, results: BenchmarkResults) -> float:
        """Calculate overall performance score (0-100)."""
        score = 0.0

        # Throughput score (40% weight)
        if results.theoretical_max_rps > 0:
            throughput_efficiency = min(
                100, (results.avg_throughput / results.theoretical_max_rps) * 100
            )
            score += throughput_efficiency * 0.4

        # Latency score (30% weight) - lower latency variance is better
        if results.p50_latency > 0:
            latency_variance = results.p99_latency / results.p50_latency
            latency_score = max(0, 100 - (latency_variance - 2) * 10)  # Penalize high variance
            score += latency_score * 0.3

        # Error rate score (20% weight)
        if results.total_requests > 0:
            error_rate = (results.failed_requests / results.total_requests) * 100
            error_score = max(0, 100 - error_rate * 10)  # Heavy penalty for errors
            score += error_score * 0.2

        # Rate limit compliance (10% weight)
        compliance_score = 100 if results.rate_limit_violations == 0 else 0
        score += compliance_score * 0.1

        return min(100, max(0, score))

    def _get_score_color(self, score: float) -> str:
        """Get color based on performance score."""
        if score >= 90:
            return "green"
        elif score >= 70:
            return "cyan"
        elif score >= 50:
            return "warning"
        else:
            return "fail"

    def _get_performance_status(self, score: float) -> str:
        """Get status text based on performance score."""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 70:
            return "FAIR"
        elif score >= 50:
            return "POOR"
        else:
            return "CRITICAL"

    def _generate_recommendations(self, results: BenchmarkResults) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        # Low efficiency
        if results.rate_limit_efficiency < 70:
            recommendations.append(
                f"Rate limit efficiency is {results.rate_limit_efficiency:.1f}%. "
                "Consider optimizing request scheduling or increasing concurrency."
            )

        # High latency variance
        if results.p50_latency > 0 and results.p99_latency > results.p50_latency * 5:
            recommendations.append(
                f"High latency variance detected (P99/P50 ratio: {results.p99_latency / results.p50_latency:.1f}). "
                "Consider implementing better load balancing or request prioritization."
            )

        # High error rate
        error_rate = self._percentage(results.failed_requests, results.total_requests)
        if error_rate > 5:
            recommendations.append(
                f"Error rate is {error_rate:.1f}%. Investigate failure causes and implement better error handling."
            )

        # Low concurrency utilization
        if results.configured_limit > 0:
            utilization = (results.avg_concurrent / results.configured_limit) * 100
            if utilization < 50:
                recommendations.append(
                    f"Concurrency utilization is {utilization:.1f}%. "
                    "Consider increasing request rate or reducing concurrency limit."
                )

        # Rate limit violations
        if results.rate_limit_violations > 0:
            recommendations.append(
                f"{results.rate_limit_violations} rate limit violations detected. "
                "Implement better rate limit awareness and backoff strategies."
            )

        return recommendations

    def _detect_performance_issues(self, results: BenchmarkResults) -> list[str]:
        """Detect specific performance issues."""
        issues = []

        # Check for common performance problems
        if results.avg_throughput < results.theoretical_max_rps * 0.5:
            issues.append("LOW_THROUGHPUT")

        if results.p99_latency > results.p50_latency * 10:
            issues.append("HIGH_LATENCY_VARIANCE")

        error_rate = self._percentage(results.failed_requests, results.total_requests)
        if error_rate > 10:
            issues.append("HIGH_ERROR_RATE")

        if results.rate_limit_violations > 0:
            issues.append("RATE_LIMIT_VIOLATIONS")

        if results.configured_limit > 0:
            utilization = (results.avg_concurrent / results.configured_limit) * 100
            if utilization < 30:
                issues.append("LOW_CONCURRENCY_UTILIZATION")

        return issues

    def _indent_xml(self, elem, level=0):
        """Add indentation to XML elements for pretty printing."""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent_xml(child, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
