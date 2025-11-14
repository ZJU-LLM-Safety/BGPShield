#!/usr/bin/env bash

set -euo pipefail

LLM="True"
YEAR=""
MONTH=""
DAY=""
DEVICE="0"
SERIAL=""  
HOUR=12
MINUTE=0
COLLECTOR="all_collectors"
RANGE=12
DIM=16
EPOCHES=150
TYPE="updates"
MODEL=0

# MODEL=0: "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/"
# MODEL=1: "/hub/huggingface/models/Qwen/Qwen3-1.7B-Base/"
# MODEL=2: "/hub/huggingface/models/BAAI/bge-m3/"
# MODEL=3: "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"


while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --year)
      YEAR="$2"
      shift 2
      ;;
    --month)
      MONTH=$(printf "%02d" "$2")
      shift 2
      ;;
    --day)
      DAY=$(printf "%02d" "$2")
      shift 2
      ;;
    --hour)
      HOUR="$2"
      shift 2
      ;;
    --minute)
      MINUTE="$2"
      shift 2
      ;;
    # --collector)
    #   COLLECTOR="$2"
    #   shift 2
    #   ;;
    --range)
      RANGE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --serial)
      SERIAL="$2"
      shift 2
      ;;
    # --llm)
    #   LLM="$2"
    #   shift 2
    #   ;;
    --dim)
      DIM="$2"
      shift 2
      ;;
    --epoches)
      EPOCHES="$2"
      shift 2
      ;;
    --type)
      TYPE="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      echo -e "\e[31mUnrecognized argument: $1\e[0m"
      exit 1
      ;;
  esac
done

if [[ -z "$YEAR" || -z "$MONTH" || -z "$DEVICE" ]]; then
  echo -e "\e[31mNeed to specify --year, --month, --device：\e[0m"
  echo -e "\e[33m./run_pipeline.sh --year 2020 --month 6 --device 0 [...]\e[0m"
  exit 1
fi

if [[ "$MODEL" < 0 || "$MODEL" > 3 ]]; then
  echo -e "\e[31mMODEL should be between 0 and 3\e[0m"
  echo -e "0: DeepSeek-R1-0528-Qwen3-8B"
  echo -e "1: Qwen3-1.7B-Base"
  echo -e "2: bge-m3"
  exit 1
fi

DATE_STR="${YEAR}${MONTH}${DAY}"

if [[ -z "$SERIAL" ]]; then
  if [ "$DATE_STR" -lt "20151201" ]; then
    echo -e "\e[33m[INFO] date < 20151201, using serial 1\e[0m"
    SERIAL=1
    BEAM_MODEL="${YEAR}${MONTH}01.as-rel.1000.10.128"
  else
    SERIAL=2
    BEAM_MODEL="${YEAR}${MONTH}01.as-rel${SERIAL}.1000.10.128"
  fi
fi


PRIMARY_DEVICE=$(echo "$DEVICE" | cut -d',' -f1)

echo -e "\e[32m========================================\e[0m"
echo -e "\e[32mDetect Date： $YEAR-$MONTH\e[0m"
echo -e "\e[32mAS Relationship Dataset SERIAL: $SERIAL\e[0m"
echo -e "\e[32mRouteViews Collector: $COLLECTOR\e[0m"
echo -e "\e[32mGPU DEVICE: $DEVICE\e[0m"
echo -e "\e[32mPrimary Device: $PRIMARY_DEVICE\e[0m" 

echo -e "\e[32mLLM_MODEL: \e[31mSpecified in the file (iterative_as_embeds.py & train.py)\e[0m"
if [ "$MODEL" = "0" ]; then
    echo -e "\e[32mLLM Model: DeepSeek-R1-0528-Qwen3-8B\e[0m"
elif [ "$MODEL" = "1" ]; then
    echo -e "\e[32mLLM Model: Qwen3-1.7B-Base\e[0m"
elif [ "$MODEL" = "2" ]; then
    echo -e "\e[32mLLM Model: bge-m3\e[0m"
fi
if [ "$DIM" = "0" ]; then
    echo -e "\e[32mDON'T USE Dimensionality Reduction\e[0m"
else
    echo -e "\e[32mLLM Embedding Dimension: $DIM\e[0m"
fi

echo -e "\e[32m========================================\e[0m"

PY_ARGS_COMMON=()
[[ -n "$YEAR" ]] && PY_ARGS_COMMON+=(--year "$YEAR")
[[ -n "$MONTH" ]] && PY_ARGS_COMMON+=(--month "$MONTH")
[[ -n "$DAY" ]] && PY_ARGS_COMMON+=(--day "$DAY")
[[ -n "$HOUR" ]] && PY_ARGS_COMMON+=(--hour "$HOUR")
[[ -n "$MINUTE" ]] && PY_ARGS_COMMON+=(--minute "$MINUTE")
[[ -n "$COLLECTOR" ]] && PY_ARGS_COMMON+=(--collector "$COLLECTOR")

start_time=$(date +%s.%N)

echo -e "\e[32mStep 0: Constructing LLM embedding\e[0m"
python3 BGPShield/iterative_as_embeds.py \
    --year "$YEAR" \
    --month "$MONTH" \
    --device "$DEVICE" \
    --model "$MODEL" 

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo -e "\e[34m[Time] Step 0 took ${elapsed} seconds\e[0m"

echo -e "\e[32mStep 1: Detecting Route Change\e[0m"
python3 routing_monitor/all_route_monitor.py \
    --data-type "$TYPE" \
    --year "$YEAR" \
    --day "$DAY" \
    --hour "$HOUR" \
    --minute "$MINUTE" \
    --time-range "$RANGE"

start_time=$(date +%s.%N)

echo -e "\e[32mStep 2: Quantizing Path Difference\e[0m"
python3 anomaly_detector/diff_evaluator_routeviews.py --dimension "$DIM" \
    "${PY_ARGS_COMMON[@]}" --model "$MODEL" --dimension "$DIM" --epoches "$EPOCHES"

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo -e "\e[34m[Time] Step 2 took ${elapsed} seconds\e[0m"

echo -e "\e[32mStep 3: Detecting Anomaly\e[0m"
python3 anomaly_detector/llm_report_anomaly_routeviews.py \
    "${PY_ARGS_COMMON[@]}" --model "$MODEL" --dimension "$DIM"

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo -e "\e[34m[Time] Step 3 took ${elapsed} seconds\e[0m"

start_time=$(date +%s.%N)

echo -e "\e[32mStep 4: Alarm Postprocessing\e[0m"
python3 post_processor/alarm_postprocess_routeviews.py \
     "${PY_ARGS_COMMON[@]}" --model "$MODEL" --dimension "$DIM"

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo -e "\e[34m[Time] Step 4 took ${elapsed} seconds\e[0m"

start_time=$(date +%s.%N)

echo -e "\e[32mStep 5: Summarizing Report\e[0m"
python3 post_processor/summary_routeviews.py \
    "${PY_ARGS_COMMON[@]}" --llm "$LLM" --dimension "$DIM" --model "$MODEL"

echo -e "\e[32mThe Detection For $YEAR-$MONTH-$DAY Has Finished\e[0m"

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo -e "\e[34m[Time] Step 5 took ${elapsed} seconds\e[0m"