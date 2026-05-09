#!/usr/bin/env bash
# Run AnalysisApp integration tests.
# Starts broker + AnalysisApp, runs e2e_test.py, tears down on exit.
set -euo pipefail
cd "$(dirname "$0")"

BROKER_PORT=5672
BROKER_URL="amqp://localhost:${BROKER_PORT}"
TEST="${1:-}"   # optional: single test name

echo "=== Building sdr-analysis:dev ==="
podman build --target runtime -t sdr-analysis:dev ..

echo "=== Starting broker ==="
podman-compose -f compose.yml up -d broker
echo -n "Waiting for broker..."
for i in $(seq 1 30); do
    podman exec sdr-broker \
        /var/lib/artemis-instance/bin/artemis check node --up \
        >/dev/null 2>&1 && break
    echo -n "."
    sleep 2
done
echo " ready"

echo "=== Starting AnalysisApp ==="
podman-compose -f compose.yml up -d analysis
sleep 3   # let it connect to broker

echo "=== Running integration tests ==="
pip install python-qpid-proton numpy --quiet

if [ -n "$TEST" ]; then
    python3 e2e_test.py --broker "$BROKER_URL" --test "$TEST"
else
    python3 e2e_test.py --broker "$BROKER_URL"
fi
RC=$?

echo "=== Tearing down ==="
podman-compose -f compose.yml down --timeout 5

exit $RC
