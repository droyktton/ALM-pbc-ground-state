#!/bin/bash
# Submit array job, capture its ID, then submit aggregator to run after all tasks finish
JID=$(sbatch --parsable submit.sh)
echo "Array job: $JID"
sbatch --dependency=afterok:$JID aggregate.sh
echo "Aggregator queued after $JID"

