#!/usr/bin/env bash
set -e

# ==========================================
# IRIS Rosbag Processing Script (SSD)
# ==========================================

# SSD mount point
SSD_MOUNT="/media/jerry/SSD"

# Default directories on SSD
BAG_DIR="$SSD_MOUNT/rosbag_data"
OUT_DIR="$SSD_MOUNT/raw_data"

# ----------------------------
# Check SSD mounted
# ----------------------------
if [[ ! -d "$SSD_MOUNT" ]]; then
  echo "❌ ERROR: SSD not found at $SSD_MOUNT"
  echo "Is the external drive mounted?"
  exit 1
fi

# ----------------------------
# Parse arguments
# ----------------------------
BAG_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --bag)
      BAG_NAME="$2"
      shift 2
      ;;
    --dir)
      BAG_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./process_rosbag.sh --bag <bag_name or full_path> [--dir <bag_directory>]"
      exit 1
      ;;
  esac
done

# ----------------------------
# Resolve bag path
# ----------------------------
if [[ -z "$BAG_NAME" ]]; then
  echo "❌ Please specify a bag using --bag"
  exit 1
fi

# If user passed full path
if [[ "$BAG_NAME" == /* ]]; then
  BAG_PATH="$BAG_NAME"
else
  # Append .bag if missing
  if [[ "$BAG_NAME" != *.bag ]]; then
    BAG_NAME="${BAG_NAME}.bag"
  fi
  BAG_PATH="$BAG_DIR/$BAG_NAME"
fi

if [[ ! -f "$BAG_PATH" ]]; then
  echo "❌ Bag not found: $BAG_PATH"
  exit 1
fi

# ----------------------------
# Output folder
# ----------------------------
BAG_BASE=$(basename "$BAG_PATH" .bag)
FINAL_OUT="$OUT_DIR/$BAG_BASE"
mkdir -p "$FINAL_OUT"

# ----------------------------
# Run processing
# ----------------------------
echo "=========================================="
echo " IRIS Rosbag Processing"
echo " Bag:     $BAG_PATH"
echo " Output:  $FINAL_OUT"
echo " SSD:     $SSD_MOUNT"
echo "=========================================="

python3 "$HOME/Desktop/MEng_project/bag_reader/scripts/iris_rosbag_reader.py" \
    --bag "$BAG_PATH" \
    --out "$FINAL_OUT"

echo "=========================================="
echo " Done. Raw data saved in:"
echo " $FINAL_OUT"
echo "=========================================="
