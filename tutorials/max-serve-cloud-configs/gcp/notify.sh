#!/bin/bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

PROJECT_ID=$1
INSTANCE_NAME=$2
ZONE=$3
MAX_WAIT_MINUTES=30
START_TIME=$(date +%s)

fetch_logs() {
    local instance_id=$1
    local limit=${2:-50}
    echo "=== GCP Instance Logs ==="

    echo "Querying logs for instance_id: ${instance_id}"

    gcloud logging read \
        "resource.type=gce_instance AND \
        resource.labels.instance_id=${instance_id} AND \
        jsonPayload.message:*" \
        --project="${PROJECT_ID}" \
        --format="table(timestamp,jsonPayload.message)" \
        --limit=10

    echo "===================="
}

check_server_status() {
    local logs=$1
    echo "🔍 Checking logs for server status..."

    # Print a sample of logs for debugging
    echo "📋 Log sample for pattern matching:"
    echo "$logs" | tail -n 5

    if echo "$logs" | grep -q "startup-script:.*Uvicorn running on http://0.0.0.0:8000" ||
        echo "$logs" | grep -q "startup-script:.*Application startup complete." ||
        echo "$logs" | grep -q "Uvicorn running on http://0.0.0.0:8000" ||
        echo "$logs" | grep -q "Application startup complete."; then
        echo "✅ Found server running message"
        return 0
    fi

    echo "❌ Server running message not found"
    echo "🔍 Looking for patterns:"
    echo "  - 'startup-script:.*Uvicorn running on http://0.0.0.0:8000'"
    echo "  - 'startup-script:.*Application startup complete.'"
    echo "  - 'Uvicorn running on http://0.0.0.0:8000'"
    echo "  - 'Application startup complete.'"
    return 1
}

echo "🔍 Starting monitoring for MAX server (max wait: ${MAX_WAIT_MINUTES} minutes)..."

# Get instance ID
INSTANCE_ID=$(gcloud compute instances describe "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --format="value(id)")

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Failed to get instance ID. Please check if instance exists."
    exit 1
fi

while true; do
    current_time=$(date +%s)
    elapsed_minutes=$(((current_time - START_TIME) / 60))

    if [ $elapsed_minutes -ge $MAX_WAIT_MINUTES ]; then
        echo "❌ Timeout after ${MAX_WAIT_MINUTES} minutes. Server might still be starting up."
        exit 1
    fi

    echo "⏳ Checking logs... (${elapsed_minutes}/${MAX_WAIT_MINUTES} minutes)"

    LOGS=$(fetch_logs "$INSTANCE_ID")

    if [ -n "$LOGS" ]; then
        if check_server_status "$LOGS"; then
            echo "✅ Server is ready! (took ${elapsed_minutes} minutes)"
            echo "📋 Latest logs:"
            echo "$LOGS"
            exit 0
        fi
    else
        echo "⏳ Logs not yet available..."
    fi

    echo "⏳ Server still starting up... checking again in 60 seconds"
    echo "-------------------------------------------"
    sleep 60
done
