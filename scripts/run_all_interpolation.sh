#!/bin/bash
# Full processing script for all HFR interpolation methods

# Configuration
BASE_DIR="$HOME/azureblob/4D-Lung-Interpolated"
PROCESSED_DIR="/mnt/tcia_data/processed/4D-Lung-Cycles"
SCRIPTS_DIR="."

# Processing parameters
MAX_WORKERS=8  # Adjust based on your CPU cores
BATCH_SIZE=100  # Process in batches to monitor progress

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_processing_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log_message() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Start processing
log_message "${BLUE}=== Full HFR Interpolation Processing ===${NC}"
log_message "Start time: $(date)"
log_message "Base directory: $BASE_DIR"
log_message "Processed directory: $PROCESSED_DIR"
log_message "Log file: $LOG_FILE"
log_message ""

# Check directories
if [ ! -d "$PROCESSED_DIR" ]; then
    log_message "${RED}Error: Processed data directory not found: $PROCESSED_DIR${NC}"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    log_message "${RED}Error: Base directory not found. Run init_pipeline.py first${NC}"
    exit 1
fi

# Function to process an experiment
process_experiment() {
    local exp_name=$1
    local priority=$2
    
    log_message "\n${GREEN}Processing $exp_name (Priority $priority)...${NC}"
    
    # First, check how many tasks need processing
    MISSING_COUNT=$(poetry run python $SCRIPTS_DIR/process_batch.py \
        --base-dir "$BASE_DIR" \
        --processed-dir "$PROCESSED_DIR" \
        --experiments "$exp_name" \
        --priority "$priority" \
        --dry-run 2>&1 | grep "Total tasks:" | awk '{print $3}')
    
    if [ -z "$MISSING_COUNT" ] || [ "$MISSING_COUNT" -eq 0 ]; then
        log_message "${YELLOW}No missing tasks for $exp_name${NC}"
        return
    fi
    
    log_message "Found $MISSING_COUNT tasks to process for $exp_name"
    
    # Process in batches
    PROCESSED=0
    while [ $PROCESSED -lt $MISSING_COUNT ]; do
        REMAINING=$((MISSING_COUNT - PROCESSED))
        CURRENT_BATCH=$((REMAINING < BATCH_SIZE ? REMAINING : BATCH_SIZE))
        
        log_message "Processing batch: $((PROCESSED + 1)) to $((PROCESSED + CURRENT_BATCH)) of $MISSING_COUNT"
        
        poetry run python $SCRIPTS_DIR/process_batch.py \
            --base-dir "$BASE_DIR" \
            --processed-dir "$PROCESSED_DIR" \
            --experiments "$exp_name" \
            --priority "$priority" \
            --max-tasks "$CURRENT_BATCH" \
            --workers "$MAX_WORKERS" \
            --phase-ranges "0-100" >> "$LOG_FILE" 2>&1 <<< "y"
        
        if [ $? -eq 0 ]; then
            PROCESSED=$((PROCESSED + CURRENT_BATCH))
            log_message "${GREEN}✓ Batch completed. Progress: $PROCESSED/$MISSING_COUNT${NC}"
        else
            log_message "${RED}✗ Batch failed. Check log for details${NC}"
            return 1
        fi
        
        # Show current status
        poetry run python $SCRIPTS_DIR/check_hfr_status.py \
            --base-dir "$BASE_DIR" \
            --experiment "$exp_name" 2>&1 | tail -n 20 >> "$LOG_FILE"
    done
    
    log_message "${GREEN}✓ $exp_name completed!${NC}"
}

# Priority 1: Most important HFR experiments (8 fps)
log_message "\n${BLUE}=== Processing Priority 1 Experiments ===${NC}"
process_experiment "hfr_linear_8fps" 1
process_experiment "hfr_optical_flow_8fps" 1

# Check intermediate status
log_message "\n${BLUE}=== Intermediate Status Check ===${NC}"
poetry run python $SCRIPTS_DIR/check_hfr_status.py --base-dir "$BASE_DIR" | tee -a "$LOG_FILE"

log_message "\n${YELLOW}Priority 1 experiments completed. Continuing with Priority 2...${NC}"

# Priority 2: Spline experiment
log_message "\n${BLUE}=== Processing Priority 2 Experiments ===${NC}"
process_experiment "hfr_spline_8fps" 2

# Final status report
log_message "\n${BLUE}=== Final Status Report ===${NC}"
poetry run python $SCRIPTS_DIR/check_hfr_status.py \
    --base-dir "$BASE_DIR" \
    --stats | tee -a "$LOG_FILE"

# Storage report
log_message "\n${BLUE}=== Storage Usage ===${NC}"
poetry run python $SCRIPTS_DIR/check_status.py \
    --base-dir "$BASE_DIR" 2>&1 | grep -A 5 "STORAGE USAGE" | tee -a "$LOG_FILE"

# Export detailed report
REPORT_FILE="$BASE_DIR/processing_report_$(date +%Y%m%d_%H%M%S).json"
poetry run python $SCRIPTS_DIR/check_status.py \
    --base-dir "$BASE_DIR" \
    --export-report "$REPORT_FILE"

log_message "\n${GREEN}=== Processing Complete ===${NC}"
log_message "End time: $(date)"
log_message "Detailed report saved to: $REPORT_FILE"
log_message ""
log_message "Next steps:"
log_message "1. Review the processing report"
log_message "2. Check training data availability: poetry run python $SCRIPTS_DIR/check_status.py --base-dir $BASE_DIR --split train"
log_message "3. Start training your flow matching model with the generated HFR data"