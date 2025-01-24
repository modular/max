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

REGION=$1
STACK_NAME=$2
MAX_WAIT_MINUTES=30
START_TIME=$(date +%s)
LOG_GROUP="/aws/ec2/$STACK_NAME-logs"

fetch_logs() {
    local stream_name=$1
    local stream_type=$2
    local limit=$3
    echo "=== $stream_type Logs ==="

    if [ -n "$limit" ]; then
        aws logs get-log-events \
            --log-group-name "$LOG_GROUP" \
            --log-stream-name "$stream_name" \
            --limit $limit \
            --region $REGION \
            --query 'events[*].[timestamp,message]' \
            --output text
    else
        aws logs get-log-events \
            --log-group-name "$LOG_GROUP" \
            --log-stream-name "$stream_name" \
            --start-time $(($(date +%s) - 60))000 \
            --region $REGION \
            --query 'events[*].[timestamp,message]' \
            --output text
    fi
    echo "===================="
}

check_server_status() {
    local logs=$1
    echo "🔍 Checking logs for server status..."

    # Check for Uvicorn startup message in container logs
    if echo "$logs" | grep -q "Uvicorn running on http://0.0.0.0:8000" ||
        echo "$logs" | grep -q "Application startup complete"; then
        echo "✅ Found server running message"
        return 0
    fi

    echo "❌ Server running message not found"
    return 1
}

echo "🔍 Starting monitoring for MAX server (max wait: ${MAX_WAIT_MINUTES} minutes)..."

while true; do
    current_time=$(date +%s)
    elapsed_minutes=$(((current_time - START_TIME) / 60))

    if [ $elapsed_minutes -ge $MAX_WAIT_MINUTES ]; then
        echo "❌ Timeout after ${MAX_WAIT_MINUTES} minutes. Server might still be starting up."
        exit 1
    fi

    EC2_LOG_STREAM=$(aws logs describe-log-streams \
        --log-group-name "$LOG_GROUP" \
        --log-stream-name-prefix "instance-logs" \
        --region $REGION \
        --query "logStreams[0].logStreamName" \
        --output text)

    echo "⏳ Checking logs... (${elapsed_minutes}/${MAX_WAIT_MINUTES} minutes)"

    if [ "$EC2_LOG_STREAM" != "None" ]; then
        echo "📜 Instance Logs:"
        EC2_LOGS=$(fetch_logs "$EC2_LOG_STREAM" "Instance" 50)
        if check_server_status "$EC2_LOGS"; then
            echo "✅ Server is ready! (took ${elapsed_minutes} minutes)"
            echo "📋 Latest logs:"
            echo "$EC2_LOGS"
            exit 0
        fi
    else
        echo "⏳ Logs not yet available..."
    fi

    echo "⏳ Server still starting up... checking again in 60 seconds"
    echo "-------------------------------------------"
    sleep 60
done
