from prometheus_client import Counter, Gauge, Histogram

# Default prometheus buckets top out ~10s and are sparse below 100ms. TTFT on a
# small model routinely lands in the 10-200ms range, so the defaults would put
# nearly every sample in one bucket and make p95/p99 meaningless.
TTFT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0)
ITL_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0, 2.0)

ttft_seconds = Histogram(
    "kvroute_ttft_seconds",
    "Time to first token, measured at the gateway",
    ["backend", "strategy", "cache_state"],
    buckets=TTFT_BUCKETS,
)

itl_seconds = Histogram(
    "kvroute_itl_seconds",
    "Inter-token latency between consecutive response chunks",
    ["backend", "strategy", "cache_state"],
    buckets=ITL_BUCKETS,
)

requests_total = Counter(
    "kvroute_requests_total",
    "Requests routed to a backend",
    ["backend", "strategy"],
)

backend_errors_total = Counter(
    "kvroute_backend_errors_total",
    "Backend connection/request failures",
    ["backend"],
)

cancelled_total = Counter(
    "kvroute_cancelled_total",
    "Requests where the client disconnected before the stream completed",
    ["backend"],
)

inflight = Gauge(
    "kvroute_inflight",
    "In-flight requests currently streaming per backend",
    ["backend"],
)

cache_lookups_total = Counter(
    "kvroute_cache_lookups_total",
    "Cache lookups by layer and outcome",
    ["layer", "outcome"],  # layer: exact|semantic, outcome: hit|miss
)

prefix_routing_total = Counter(
    "kvroute_prefix_routing_total",
    "Prefix-aware routing decisions",
    ["outcome"],  # hit|miss|shed_imbalance
)

admission_total = Counter(
    "kvroute_admission_total",
    "Admission control decisions",
    ["priority", "outcome"],  # outcome: admitted|shed_budget|shed_capacity
)
