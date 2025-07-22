#!/bin/bash
# Test script for small amount of interpolation across all methods

# Configuration
#BASE_DIR="/mnt/tcia_data/interpolated"
BASE_DIR="$HOME/azureblob/4D-Lung-Interpolated"
PROCESSED_DIR="/mnt/tcia_data/processed/4D-Lung-Cycles"
SCRIPTS_DIR="scripts"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== HFR Interpolation Test Script ===${NC}"
echo "This will test all interpolation methods with minimal data"
echo ""

# Check if directories exist
if [ ! -d "$PROCESSED_DIR" ]; then
    echo -e "${RED}Error: Processed data directory not found: $PROCESSED_DIR${NC}"
    exit 1
fi

# Initialize pipeline if needed
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${GREEN}Initializing pipeline...${NC}"
    poetry run python $SCRIPTS_DIR/init_pipeline.py \
        --base-dir "$BASE_DIR" \
        --processed-dir "$PROCESSED_DIR" \
        --verbose
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Pipeline initialization failed${NC}"
        exit 1
    fi
fi

# Get first available patient
echo -e "\n${BLUE}Finding test patient...${NC}"
FIRST_PATIENT=$(ls -1d "$PROCESSED_DIR"/*/ 2>/dev/null | grep -v -E "exports|splits|analysis" | head -1 | xargs basename)

if [ -z "$FIRST_PATIENT" ]; then
    echo -e "${RED}No patients found in processed directory${NC}"
    exit 1
fi

echo "Using patient: $FIRST_PATIENT"

# Test each HFR interpolation method
echo -e "\n${BLUE}Testing HFR interpolation methods...${NC}"

# Array of HFR experiments to test (8 fps only)
declare -a HFR_EXPERIMENTS=(
    "hfr_linear_8fps"
    "hfr_spline_8fps"
    "hfr_optical_flow_8fps"
)

# Process each experiment with minimal data
for exp in "${HFR_EXPERIMENTS[@]}"; do
    echo -e "\n${GREEN}Testing $exp...${NC}"
    
    poetry run python $SCRIPTS_DIR/process_batch.py \
        --base-dir "$BASE_DIR" \
        --processed-dir "$PROCESSED_DIR" \
        --experiments "$exp" \
        --patients "$FIRST_PATIENT" \
        --max-slices 2 \
        --phase-ranges "0-100" \
        --workers 2
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $exp completed successfully${NC}"
    else
        echo -e "${RED}✗ $exp failed${NC}"
    fi
done

# Check status of test run
echo -e "\n${BLUE}Checking HFR status...${NC}"
poetry run python $SCRIPTS_DIR/check_hfr_status.py \
    --base-dir "$BASE_DIR" \
    --stats

# Also test sparse experiments for evaluation
echo -e "\n${BLUE}Testing sparse evaluation experiments...${NC}"
poetry run python $SCRIPTS_DIR/process_batch.py \
    --base-dir "$BASE_DIR" \
    --processed-dir "$PROCESSED_DIR" \
    --experiments "test_sparse_0_50,test_sparse_0_100" \
    --patients "$FIRST_PATIENT" \
    --max-slices 1 \
    --workers 2

echo -e "\n${GREEN}Test completed!${NC}"
echo "Check the results in: $BASE_DIR"
echo ""
echo "If everything looks good, run ./run_all_interpolation.sh for full processing"