#!/bin/bash
set -e

# Setup test environment
echo "Setting up test environment..."
COVER="/Users/betulyedek/Downloads/BOSSbase_1.01/1.pgm"
TEST_DIR="test_artifacts"
mkdir -p "$TEST_DIR"

echo "Creating dummy files..."
echo "This is a secret payload file." > "$TEST_DIR/payload.txt"
dd if=/dev/urandom of="$TEST_DIR/dummy.bin" bs=1K count=1 2>/dev/null

echo "=========================================="
echo "Scenario 1: Lock (Message Mode)"
echo "=========================================="
.venv/bin/python3 -m stegano_app.product_cli lock -i "$COVER" -o "$TEST_DIR/stego_msg.pgm" -p "testpass" -m "Secret" --force
echo "Scenario 1 PASSED"

echo "=========================================="
echo "Scenario 2: Lock (File Mode)"
echo "=========================================="
.venv/bin/python3 -m stegano_app.product_cli lock -i "$COVER" -o "$TEST_DIR/stego_file.pgm" -p "testpass" -f "$TEST_DIR/payload.txt" --force
echo "Scenario 2 PASSED"

echo "=========================================="
echo "Scenario 3: Lock (Binary File Mode)"
echo "=========================================="
.venv/bin/python3 -m stegano_app.product_cli lock -i "$COVER" -o "$TEST_DIR/stego_bin.pgm" -p "testpass" -f "$TEST_DIR/dummy.bin" --force
echo "Scenario 3 PASSED"

echo "=========================================="
echo "Scenario 4: Unlock (Text Extraction to File)"
echo "=========================================="
.venv/bin/python3 -m stegano_app.product_cli unlock -i "$TEST_DIR/stego_file.pgm" -o "$TEST_DIR/extracted_payload.txt" -p "testpass" --force
cat "$TEST_DIR/extracted_payload.txt"
echo
echo "Scenario 4 PASSED"

echo "=========================================="
echo "Scenario 5: Unlock (Text Extraction with Print)"
echo "=========================================="
.venv/bin/python3 -m stegano_app.product_cli unlock -i "$TEST_DIR/stego_msg.pgm" -o "$TEST_DIR/extracted_msg.txt" -p "testpass" --print --force
echo "Scenario 5 PASSED"

echo "=========================================="
echo "Scenario 6: Unlock (Binary Extraction with Print Warning)"
echo "=========================================="
.venv/bin/python3 -m stegano_app.product_cli unlock -i "$TEST_DIR/stego_bin.pgm" -o "$TEST_DIR/extracted_dummy.bin" -p "testpass" --print --force
echo "Scenario 6 PASSED"

echo "=========================================="
echo "Scenario 7: Inspect"
echo "=========================================="
.venv/bin/python3 -m stegano_app.product_cli inspect --cover "$COVER" --stego "$TEST_DIR/stego_msg.pgm" --output "$TEST_DIR/inspect_output.png" --force
if [ -f "$TEST_DIR/inspect_output.png" ]; then
    echo "Scenario 7 PASSED (inspect_output.png generated)"
else
    echo "Scenario 7 FAILED"
    exit 1
fi

echo "=========================================="
echo "Scenario 8: Edge Cases"
echo "=========================================="
echo "8a: Output to existing file WITHOUT --force"
set +e
OUTPUT=$(.venv/bin/python3 -m stegano_app.product_cli lock -i "$COVER" -o "$TEST_DIR/stego_msg.pgm" -p "testpass" -m "Secret" 2>&1)
set -e
if echo "$OUTPUT" | grep -q "output already exists"; then
    echo "8a PASSED"
else
    echo "8a FAILED"
    echo "$OUTPUT"
    exit 1
fi

echo "8b: Output to existing file WITH --force"
.venv/bin/python3 -m stegano_app.product_cli lock -i "$COVER" -o "$TEST_DIR/stego_msg.pgm" -p "testpass" -m "Secret" --force
echo "8b PASSED"

echo "8c: Unlock with incorrect password"
set +e
OUTPUT=$(.venv/bin/python3 -m stegano_app.product_cli unlock -i "$TEST_DIR/stego_msg.pgm" -o "$TEST_DIR/extracted_fail.txt" -p "wrongpass" --force 2>&1)
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -ne 0 ]; then
    echo "8c PASSED (failed as expected with output: $OUTPUT)"
else
    echo "8c FAILED (expected error but succeeded)"
    exit 1
fi

echo "=========================================="
echo "ALL SCENARIOS COMPLETED SUCCESSFULLY"
echo "=========================================="
