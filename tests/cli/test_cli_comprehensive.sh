#!/bin/bash

################################################################################
# Venice AI CLI Comprehensive Test Script
#
# This script thoroughly tests all CLI commands with various flag combinations
# and writes outputs to timestamped directories for manual review.
#
# Usage:
#   ./test_cli_comprehensive.sh [OPTIONS]
#
# Options:
#   --quick         Run only essential tests (faster execution)
#   --cleanup       Remove test outputs after completion
#   --help          Show this help message
#
# Environment:
#   VENICE_API_KEY  If set, runs API-dependent tests. Otherwise skips them.
#
# Output:
#   Creates cli_test_results_YYYYMMDD_HHMMSS/ directory with:
#   - Categorized test outputs
#   - Summary report (test_summary.md)
################################################################################

set -e  # Exit on error in setup, but we'll handle errors in tests

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/cli_test_results_${TIMESTAMP}"
VENICE_CMD="poetry run venice"

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Flags
QUICK_MODE=false
CLEANUP_MODE=false
RUN_API_TESTS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

show_help() {
    cat << EOF
Venice AI CLI Comprehensive Test Script

Usage: $0 [OPTIONS]

Options:
    --quick         Run only essential tests (faster execution)
    --cleanup       Remove test outputs after completion
    --help          Show this help message

Environment Variables:
    VENICE_API_KEY  If set, runs API-dependent tests. Otherwise skips them.

Output:
    Creates cli_test_results_YYYYMMDD_HHMMSS/ with categorized test outputs
    and a summary report (test_summary.md)

Examples:
    # Run full test suite (if API key is set)
    $0

    # Run quick tests only
    $0 --quick

    # Run tests and cleanup outputs
    $0 --cleanup

EOF
    exit 0
}

run_test() {
    local category="$1"
    local test_name="$2"
    local command="$3"
    local should_skip="${4:-false}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    local test_num=$(printf "%03d" $TOTAL_TESTS)
    
    # Create sanitized filename
    local filename=$(echo "${test_name}" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    local output_file="${OUTPUT_DIR}/${category}/${test_num}_${filename}.txt"
    
    # Check if we should skip
    if [ "$should_skip" = "true" ]; then
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        print_warning "[${test_num}/${TOTAL_TESTS}] SKIPPED: ${test_name} (requires API key)"
        echo "SKIPPED: Requires VENICE_API_KEY" > "$output_file"
        return 0
    fi
    
    print_info "[${test_num}] Testing: ${test_name}"
    
    # Run the test and capture output
    {
        echo "========================================="
        echo "Test: ${test_name}"
        echo "Category: ${category}"
        echo "Command: ${command}"
        echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="
        echo ""
        
        # Run with timeout to prevent hanging
        if timeout 30s bash -c "$command" 2>&1; then
            TEST_EXIT_CODE=0
        else
            TEST_EXIT_CODE=$?
        fi
        
        echo ""
        echo "========================================="
        echo "Exit Code: ${TEST_EXIT_CODE}"
        echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="
        
        return $TEST_EXIT_CODE
    } > "$output_file" 2>&1
    
    local result=$?
    
    if [ $result -eq 0 ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        print_success "[${test_num}] PASSED: ${test_name}"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_error "[${test_num}] FAILED: ${test_name} (exit code: ${result})"
    fi
}

################################################################################
# Test Suites
################################################################################

test_core_commands() {
    print_header "Core Commands Tests"
    local category="01_core"
    
    run_test "$category" "version" "$VENICE_CMD --version"
    run_test "$category" "help" "$VENICE_CMD --help"
    run_test "$category" "chat_help" "$VENICE_CMD chat --help"
    run_test "$category" "image_help" "$VENICE_CMD image --help"
    run_test "$category" "models_help" "$VENICE_CMD models --help"
}

test_configure_command() {
    print_header "Configure Command Tests"
    local category="02_configure"
    
    # Note: configure is interactive, so we can only test help
    # Help doesn't require API key
    run_test "$category" "configure_help" "$VENICE_CMD configure --help"
}

test_chat_commands() {
    print_header "Chat Commands Tests"
    local category="03_chat"
    
    # Non-API tests (validation)
    run_test "$category" "chat_start_help" "$VENICE_CMD chat start --help"
    
    if [ "$RUN_API_TESTS" = "true" ]; then
        # Basic single message tests
        run_test "$category" "chat_simple" "$VENICE_CMD chat start 'What is 2+2?'"
        run_test "$category" "chat_no_stream" "$VENICE_CMD chat start --no-stream 'Hello'"
        
        # Temperature variations
        run_test "$category" "chat_temp_0" "$VENICE_CMD chat start --temperature 0.0 --no-stream 'Count to 5'"
        run_test "$category" "chat_temp_0.5" "$VENICE_CMD chat start --temperature 0.5 --no-stream 'Count to 5'"
        run_test "$category" "chat_temp_1.0" "$VENICE_CMD chat start --temperature 1.0 --no-stream 'Count to 5'"
        
        # Max tokens variations
        run_test "$category" "chat_max_tokens_50" "$VENICE_CMD chat start --max-tokens 50 --no-stream 'Explain AI'"
        run_test "$category" "chat_max_tokens_500" "$VENICE_CMD chat start --max-tokens 500 --no-stream 'Explain AI'"
        
        # System prompt
        run_test "$category" "chat_system_prompt" "$VENICE_CMD chat start --system 'You are a helpful assistant' --no-stream 'Hi'"
        
        if [ "$QUICK_MODE" = "false" ]; then
            # Animation modes
            run_test "$category" "chat_anim_none" "$VENICE_CMD chat start --animation none 'Tell a joke'"
            run_test "$category" "chat_anim_smooth" "$VENICE_CMD chat start --animation smooth 'Tell a joke'"
            run_test "$category" "chat_anim_word" "$VENICE_CMD chat start --animation word 'Tell a joke'"
            run_test "$category" "chat_anim_char" "$VENICE_CMD chat start --animation char --animation-speed 0.01 'Hi'"
            run_test "$category" "chat_anim_line" "$VENICE_CMD chat start --animation line 'Count to 3'"
            run_test "$category" "chat_anim_typewriter" "$VENICE_CMD chat start --animation typewriter 'Hi'"
            
            # Stats and thinking
            run_test "$category" "chat_show_stats" "$VENICE_CMD chat start --show-stats 'Hello'"
            run_test "$category" "chat_show_thinking" "$VENICE_CMD chat start --show-thinking --no-stream 'Solve 5+3'"
        fi
    else
        # Skip API tests
        run_test "$category" "chat_simple" "" "true"
        run_test "$category" "chat_animations" "" "true"
    fi
}

test_image_commands() {
    print_header "Image Commands Tests"
    local category="04_image"
    
    # Non-API tests
    run_test "$category" "image_generate_help" "$VENICE_CMD image generate --help"
    run_test "$category" "image_batch_help" "$VENICE_CMD image batch --help"
    
    if [ "$RUN_API_TESTS" = "true" ]; then
        # Basic image generation
        run_test "$category" "image_gen_default" "$VENICE_CMD image generate 'A red circle'"
        run_test "$category" "image_gen_512x512" "$VENICE_CMD image generate --size 512x512 'A blue square'"
        run_test "$category" "image_gen_1024x1024" "$VENICE_CMD image generate --size 1024x1024 'A green triangle'"
        
        if [ "$QUICK_MODE" = "false" ]; then
            run_test "$category" "image_gen_1024x576" "$VENICE_CMD image generate --size 1024x576 'A landscape'"
            run_test "$category" "image_gen_1920x1080" "$VENICE_CMD image generate --size 1920x1080 'A sunset'"
            
            # Multiple images
            run_test "$category" "image_gen_num_2" "$VENICE_CMD image generate --num-images 2 'A star'"
            
            # Custom output
            run_test "$category" "image_gen_output" "$VENICE_CMD image generate --output test_image 'A moon'"
            
            # Timing options
            run_test "$category" "image_gen_no_timing" "$VENICE_CMD image generate --no-show-timing 'A sun'"
            
            # Batch generation (if prompts file exists)
            if [ -f "${SCRIPT_DIR}/test_prompts.txt" ]; then
                run_test "$category" "image_batch" "$VENICE_CMD image batch --prompts-file ${SCRIPT_DIR}/test_prompts.txt"
            fi
        fi
    else
        # Skip API tests
        run_test "$category" "image_generation" "" "true"
    fi
}

test_models_commands() {
    print_header "Models Commands Tests"
    local category="05_models"
    
    if [ "$RUN_API_TESTS" = "true" ]; then
        # Basic listing
        run_test "$category" "models_default" "$VENICE_CMD models"
        run_test "$category" "models_verbose" "$VENICE_CMD models --verbose"
        run_test "$category" "models_json" "$VENICE_CMD models --json"
        
        # Currency options
        run_test "$category" "models_currency_usd" "$VENICE_CMD models --currency usd"
        run_test "$category" "models_currency_diem" "$VENICE_CMD models --currency diem"
        run_test "$category" "models_currency_both" "$VENICE_CMD models --currency both"
        
        # Display options
        run_test "$category" "models_no_legend" "$VENICE_CMD models --no-legend"
        run_test "$category" "models_show_tier" "$VENICE_CMD models --show-tier-info"
        
        # Type filtering
        run_test "$category" "models_type_text" "$VENICE_CMD models --type text"
        run_test "$category" "models_type_image" "$VENICE_CMD models --type image"
        run_test "$category" "models_type_tts" "$VENICE_CMD models --type tts"
        run_test "$category" "models_type_embedding" "$VENICE_CMD models --type embedding"
        
        if [ "$QUICK_MODE" = "false" ]; then
            run_test "$category" "models_type_upscale" "$VENICE_CMD models --type upscale"
            run_test "$category" "models_type_inpaint" "$VENICE_CMD models --type inpaint"
            run_test "$category" "models_type_multiple" "$VENICE_CMD models --type text --type image"
            
            # Capability filtering
            run_test "$category" "models_function_calling" "$VENICE_CMD models --function-calling"
            run_test "$category" "models_vision" "$VENICE_CMD models --vision"
            run_test "$category" "models_reasoning" "$VENICE_CMD models --reasoning"
            run_test "$category" "models_web_search" "$VENICE_CMD models --web-search"
            run_test "$category" "models_code" "$VENICE_CMD models --code"
            run_test "$category" "models_response_schema" "$VENICE_CMD models --response-schema"
            
            # Combined capabilities
            run_test "$category" "models_vision_function" "$VENICE_CMD models --vision --function-calling"
            
            # Sorting
            run_test "$category" "models_sort_name" "$VENICE_CMD models --sort name"
            run_test "$category" "models_sort_id" "$VENICE_CMD models --sort id"
            run_test "$category" "models_sort_price_asc" "$VENICE_CMD models --sort price-asc"
            run_test "$category" "models_sort_price_desc" "$VENICE_CMD models --sort price-desc"
            run_test "$category" "models_sort_context" "$VENICE_CMD models --sort context"
            run_test "$category" "models_sort_created" "$VENICE_CMD models --sort created"
            
            # Status filtering
            run_test "$category" "models_beta" "$VENICE_CMD models --beta"
            run_test "$category" "models_no_beta" "$VENICE_CMD models --no-beta"
            run_test "$category" "models_online" "$VENICE_CMD models --online"
            
            # Price filtering
            run_test "$category" "models_max_input_1" "$VENICE_CMD models --max-input 1.0"
            run_test "$category" "models_max_output_1" "$VENICE_CMD models --max-output 1.0"
            run_test "$category" "models_budget_0.5" "$VENICE_CMD models --budget 0.5"
            
            # Search
            run_test "$category" "models_search" "$VENICE_CMD models --search llama"
            
            # Complex combinations
            run_test "$category" "models_complex_1" "$VENICE_CMD models --type text --vision --sort price-asc --no-beta"
            run_test "$category" "models_complex_2" "$VENICE_CMD models --function-calling --max-input 2.0 --currency usd"
        fi
    else
        # Skip API tests if no key available
        run_test "$category" "models_default" "" "true"
        run_test "$category" "models_listing_tests" "" "true"
    fi
}

################################################################################
# Main Execution
################################################################################

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --cleanup)
                CLEANUP_MODE=true
                shift
                ;;
            --help)
                show_help
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                ;;
        esac
    done
    
    print_header "Venice AI CLI Comprehensive Test Suite"
    
    # Check for API key
    if [[ -n "$VENICE_API_KEY" ]]; then
        RUN_API_TESTS=true
        print_success "API key detected - will run full test suite"
    else
        RUN_API_TESTS=false
        print_warning "No API key detected - skipping API-dependent tests"
        print_info "Set VENICE_API_KEY environment variable to run all tests"
    fi
    
    if [ "$QUICK_MODE" = "true" ]; then
        print_info "Quick mode enabled - running essential tests only"
    fi
    
    # Create output directory structure
    print_info "Creating output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}/01_core"
    mkdir -p "${OUTPUT_DIR}/02_configure"
    mkdir -p "${OUTPUT_DIR}/03_chat"
    mkdir -p "${OUTPUT_DIR}/04_image"
    mkdir -p "${OUTPUT_DIR}/05_models"
    
    # Change to CLI directory
    cd "${SCRIPT_DIR}/.." || exit 1
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run test suites
    test_core_commands
    test_configure_command
    test_chat_commands
    test_image_commands
    test_models_commands
    
    # Record end time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # Generate summary report
    print_header "Generating Summary Report"
    
    SUMMARY_FILE="${OUTPUT_DIR}/test_summary.md"
    
    cat > "$SUMMARY_FILE" << EOF
# Venice AI CLI Test Report

**Generated**: $(date '+%Y-%m-%d %H:%M:%S')
**Duration**: ${DURATION} seconds
**Quick Mode**: ${QUICK_MODE}
**API Tests**: ${RUN_API_TESTS}

## Summary

- **Total Tests**: ${TOTAL_TESTS}
- **Passed**: ${PASSED_TESTS} ✅
- **Failed**: ${FAILED_TESTS} ❌
- **Skipped**: ${SKIPPED_TESTS} ⚠️

## Test Categories

### 01. Core Commands
Basic CLI functionality tests (--version, --help, etc.)

### 02. Configure
Configuration command tests

### 03. Chat Commands
Chat completion tests with various parameters

### 04. Image Commands
Image generation and batch processing tests

### 05. Models Commands
Model listing, filtering, and comparison tests

## Detailed Results

See individual test output files in the category subdirectories.

### Failed Tests
EOF
    
    # Add failed test details
    if [ $FAILED_TESTS -gt 0 ]; then
        echo "" >> "$SUMMARY_FILE"
        find "$OUTPUT_DIR" -name "*.txt" -exec grep -l "Exit Code: [^0]" {} \; | while read -r file; do
            test_name=$(basename "$file" .txt)
            echo "- ${test_name}" >> "$SUMMARY_FILE"
        done
    else
        echo "None! All tests passed. 🎉" >> "$SUMMARY_FILE"
    fi
    
    cat >> "$SUMMARY_FILE" << EOF

## Output Structure

\`\`\`
${OUTPUT_DIR}/
├── 01_core/          - Core command tests
├── 02_configure/     - Configure command tests
├── 03_chat/          - Chat command tests
├── 04_image/         - Image generation tests
├── 05_models/        - Models listing tests
└── test_summary.md   - This file
\`\`\`

## Notes

- Each test output file contains:
  - Test name and category
  - Full command executed
  - Complete stdout/stderr output
  - Exit code
  - Timestamps

- Tests marked as SKIPPED required VENICE_API_KEY to be set
- Failed tests may indicate bugs or API issues

## Next Steps

1. Review failed tests in their respective output files
2. Check for patterns in failures
3. File issues for confirmed bugs
4. Update test cases as needed

EOF
    
    # Final summary
    print_header "Test Execution Complete"
    echo ""
    print_info "Total Tests:  ${TOTAL_TESTS}"
    print_success "Passed:       ${PASSED_TESTS}"
    print_error "Failed:       ${FAILED_TESTS}"
    print_warning "Skipped:      ${SKIPPED_TESTS}"
    echo ""
    print_info "Duration:     ${DURATION} seconds"
    print_info "Results:      ${OUTPUT_DIR}"
    print_info "Summary:      ${SUMMARY_FILE}"
    echo ""
    
    # Cleanup if requested
    if [ "$CLEANUP_MODE" = "true" ]; then
        print_warning "Cleanup mode enabled - removing test outputs..."
        rm -rf "$OUTPUT_DIR"
        print_success "Cleanup complete"
    else
        print_info "Test outputs saved for manual review"
        print_info "Use --cleanup flag to automatically remove outputs"
    fi
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main "$@"