"""Offline by default; initialize SDK redaction before tests construct agents."""

import os

os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_unredacted_attributes="
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
