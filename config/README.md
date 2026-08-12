# Configuration Reference

This directory contains **reference** configuration files for Venice AI's
optional subsystems. They are not loaded automatically; copy and adapt them
to your deployment.

| File | Purpose |
|------|---------|
| `rate_limiter.yaml` | Rate limiter tuning (bucket sizes, backoff, TTL) |
| `alerts.yml` | Prometheus alerting rules for Venice metrics |
| `grafana/` | Grafana dashboard JSON for monitoring |
