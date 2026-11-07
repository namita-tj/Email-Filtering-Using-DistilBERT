"""Sends request traces, latency and prediction logs to Azure Application Insights.

Switched on only when APPLICATIONINSIGHTS_CONNECTION_STRING is set (the container app
gets it from its secret store); locally and in tests nothing changes and the Azure packages are
not even imported.

What ends up in Application Insights:
- every HTTP request with its duration and status code (table `requests`)
- the prediction log lines with label, source and spam probability as custom dimensions
  (table `traces`), so spam rate and LLM-fallback rate can be charted
"""

import logging
import os

logger = logging.getLogger("email_filter")

_configured = False


def telemetry_enabled() -> bool:
    return bool(os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip())


def setup_telemetry(app) -> bool:
    """Instruments the FastAPI app if Application Insights is configured. Returns True if it did."""
    global _configured
    if not telemetry_enabled():
        return False
    if not _configured:
        from azure.monitor.opentelemetry import configure_azure_monitor

        # The service name shows up as the "cloud role" in Application Insights.
        os.environ.setdefault("OTEL_SERVICE_NAME", "email-filter-api")
        configure_azure_monitor(logger_name="email_filter")
        _configured = True
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
    logger.info("Application Insights telemetry enabled")
    return True
