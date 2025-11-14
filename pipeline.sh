#!/usr/bin/env bash

set -euo pipefail
# set -x

# 默认值（如果你需要也可以添加默认）
YEAR=""
MONTH=""
DAY=""
DEVICE="0"
SERIAL=""  # 如果用户不指定，稍后根据日期判断
HOUR=12
MINUTE=0
COLLECTOR="all_collectors"
RANGE=12
LLM="True"
# REDUCE="True"
DIM=16
EPOCHES=150
TYPE="updates"
# BGE="False"
MODEL=0

# MODEL=0: "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/"
# MODEL=1: "/hub/huggingface/models/Qwen/Qwen3-1.7B-Base/"
# MODEL=2: "/hub/huggingface/models/BAAI/bge-m3/"
# MODEL=3: "/hub/huggingface/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/"


# 参数解析
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
    --collector)
      COLLECTOR="$2"
      shift 2
      ;;
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
    --llm)
      LLM="$2"
      shift 2
      ;;
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
    # --bge)
    #   BGE="$2"
    #   shift 2
    #   ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    # --reduce)
    #   REDUCE="$2"
    #   shift 2
    #   ;;
    *)
      echo -e "\e[31mUnrecognized argument: $1\e[0m"
      exit 1
      ;;
  esac
done

# 参数检查
if [[ -z "$YEAR" || -z "$MONTH" || -z "$DEVICE" ]]; then
  echo -e "\e[31mNeed to specify --year, --month, --device：\e[0m"
  echo -e "\e[33m./run_pipeline.sh --year 2020 --month 6 --device 0 [...]\e[0m"
  exit 1
fi

# MODEL check
if [[ "$MODEL" < 0 || "$MODEL" > 3 ]]; then
  echo -e "\e[31mMODEL should be between 0 and 3\e[0m"
  echo -e "0: DeepSeek-R1-0528-Qwen3-8B"
  echo -e "1: Qwen3-1.7B-Base"
  echo -e "2: bge-m3"
  exit 1
fi

DATE_STR="${YEAR}${MONTH}${DAY}"

# 自动判断 serial（如果未指定）
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

if [ "$LLM" = "True" ]; then
    echo -e "\e[32mDetect BGP Anomaly Using LLM\e[0m"
else
    echo -e "\e[32mDetect BGP Anomaly Using Beam\e[0m"
fi

PRIMARY_DEVICE=$(echo "$DEVICE" | cut -d',' -f1)
# COLLECTOR="all_collectors"   # RouteViews collector，可按需修改

echo -e "\e[32m========================================\e[0m"
echo -e "\e[32mDetect Date： $YEAR-$MONTH\e[0m"
echo -e "\e[32mAS Relationship Dataset SERIAL: $SERIAL\e[0m"
echo -e "\e[32mRouteViews Collector: $COLLECTOR\e[0m"
echo -e "\e[32mGPU DEVICE: $DEVICE\e[0m"
echo -e "\e[32mPrimary Device: $PRIMARY_DEVICE\e[0m" 
if [ "$LLM" = "True" ]; then
    echo -e "\e[32mLLM_MODEL: \e[31mSpecified in the file (iterative_as_embeds.py & train.py)\e[0m"
    # echo -e "\e[32mLLM Model: $MODEL\e[0m"
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
else
    echo -e "\e[32mBEAM_MODEL: $BEAM_MODEL\e[0m"
fi
# echo -e "\e[32mBEAM_MODEL: $BEAM_MODEL\e[0m"
echo -e "\e[32m========================================\e[0m"

PY_ARGS_COMMON=()
[[ -n "$YEAR" ]] && PY_ARGS_COMMON+=(--year "$YEAR")
[[ -n "$MONTH" ]] && PY_ARGS_COMMON+=(--month "$MONTH")
[[ -n "$DAY" ]] && PY_ARGS_COMMON+=(--day "$DAY")
[[ -n "$HOUR" ]] && PY_ARGS_COMMON+=(--hour "$HOUR")
[[ -n "$MINUTE" ]] && PY_ARGS_COMMON+=(--minute "$MINUTE")
[[ -n "$COLLECTOR" ]] && PY_ARGS_COMMON+=(--collector "$COLLECTOR")
# [[ -n "$REDUCE" ]] && PY_ARGS_COMMON+=(--reduce "$REDUCE")

# start_time=$(date +%s.%N)

# if [ "$LLM" = "True" ]; then
#     echo -e "\e[32mStep 0: Constructing LLM embedding\e[0m"
#     python3 BGPShield/iterative_as_embeds.py \
#         --year "$YEAR" \
#         --month "$MONTH" \
#         --device "$DEVICE" \
#         --model "$MODEL" 
#         # --bge "$BGE"

#   if [ "$REDUCE" = "True" ]; then
#     echo -e "\e[32mStep 0.5: Reducing LLM embedding to ${DIM} dims (epochs=$EPOCHES)\e[0m"
#       python3 BGPShield/train.py \
#         --time "${YEAR}${MONTH}01" \
#         --Q 10 \
#         --dimension "$DIM" \
#         --epoches "$EPOCHES" \
#         --model "$MODEL" \
#         --device "$PRIMARY_DEVICE"
#   fi
# else
#     echo -e "\e[32mStep 0: Constructing Beam embedding\e[0m"
#     CUDA_VISIBLE_DEVICES="$DEVICE" \
#     python3 BGPShield/train_beam.py --time "${YEAR}${MONTH}01" \
#         --Q 10 \
#         --dimension 128 \
#         --epoches 1000 \
#         --device "$PRIMARY_DEVICE" \
#         --num-workers 8 
# fi

# end_time=$(date +%s.%N)
# elapsed=$(echo "$end_time - $start_time" | bc)
# echo -e "\e[34m[Time] Step 0 took ${elapsed} seconds\e[0m"

# echo -e "\e[32mStep 1: Detecting Route Change\e[0m"
# if [ "$COLLECTOR" = "all_collectors" ]; then
#     python3 routing_monitor/all_route_monitor.py \
#     --data-type "$TYPE" \
#     --year "$YEAR" \
#     --day "$DAY" \
#     --hour "$HOUR" \
#     --minute "$MINUTE" \
#     --time-range "$RANGE"
# else
#     python3 routing_monitor/route_change_monitor.py \
#       --collector "$COLLECTOR" \
#       --data-type "$TYPE" \
#       --year "$YEAR" \
#       --month "$MONTH" \
#       --day "$DAY" \
#       --hour "$HOUR" \
#       --minute "$MINUTE" \
#       --time-range "$RANGE"
# fi

start_time=$(date +%s.%N)

echo -e "\e[32mStep 2: Quantizing Path Difference\e[0m"
if [ "$LLM" = "True" ]; then
    # PY_ARGS_LLM=(--model "$MODEL" --dimension "$DIM" --epoches "$EPOCHES")
    python3 anomaly_detector/diff_evaluator_routeviews.py --dimension "$DIM" \
    "${PY_ARGS_COMMON[@]}" --model "$MODEL" --dimension "$DIM" --epoches "$EPOCHES"
    # --collector "$COLLECTOR" \
    # --model "$MODEL" \
    # --reduce "$REDUCE" \
    # --epoches "$EPOCHES" \
    # --year "$YEAR" \
    # --month "$MONTH" \
    # --day "$DAY" \
    # --hour "$HOUR" \
    # --minute "$MINUTE" 
else
    python3 anomaly_detector/BEAM_diff_evaluator_routeviews.py \
    "${PY_ARGS_COMMON[@]}" --beam-model "$BEAM_MODEL"
    # --collector "$COLLECTOR" \
    # --beam-model "$BEAM_MODEL" \
    # --year "$YEAR" \
    # --month "$MONTH" \
    # --day "$DAY" \
    # --hour "$HOUR" \
    # --minute "$MINUTE" 
fi

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo -e "\e[34m[Time] Step 2 took ${elapsed} seconds\e[0m"

echo -e "\e[32mStep 3: Detecting Anomaly\e[0m"
if [ "$LLM" = "True" ]; then
    python3 anomaly_detector/llm_report_anomaly_routeviews.py \
    "${PY_ARGS_COMMON[@]}" --model "$MODEL" --dimension "$DIM"
    # --collector "$COLLECTOR" \
    # --model "$MODEL" \
    # --reduce "$REDUCE" \
    # --year "$YEAR" \
    # --month "$MONTH" \
    # --day "$DAY" \
    # --hour "$HOUR" \
    # --minute "$MINUTE" 
else
    python3 anomaly_detector/report_anomaly_routeviews.py \
    "${PY_ARGS_COMMON[@]}"
    # --collector "$COLLECTOR" \
    # --year "$YEAR" \
    # --month "$MONTH" \
    # --day "$DAY" \
    # --hour "$HOUR" \
    # --minute "$MINUTE" 
fi

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo -e "\e[34m[Time] Step 3 took ${elapsed} seconds\e[0m"

start_time=$(date +%s.%N)

echo -e "\e[32mStep 4: Alarm Postprocessing\e[0m"
if [ "$LLM" = "True" ]; then
    python3 post_processor/alarm_postprocess_routeviews.py \
     "${PY_ARGS_COMMON[@]}" --model "$MODEL" --dimension "$DIM"
    # --collector "$COLLECTOR" \
    # --model "$MODEL" \
    # --reduce "$REDUCE" \
    # --year "$YEAR" \
    # --month "$MONTH" \
    # --day "$DAY" \
    # --hour "$HOUR" \
    # --minute "$MINUTE" 
else
    python3 post_processor/BEAM_alarm_postprocess_routeviews.py \
    "${PY_ARGS_COMMON[@]}"
    # --collector "$COLLECTOR" \
    # --year "$YEAR" \
    # --month "$MONTH" \
    # --day "$DAY"    \
    # --hour "$HOUR" \
    # --minute "$MINUTE" 
fi

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo -e "\e[34m[Time] Step 4 took ${elapsed} seconds\e[0m"

start_time=$(date +%s.%N)

echo -e "\e[32mStep 5: Summarizing Report\e[0m"
python3 post_processor/summary_routeviews.py \
"${PY_ARGS_COMMON[@]}" --llm "$LLM" --dimension "$DIM" --model "$MODEL"
#   --collector "$COLLECTOR" \
#   --llm "$LLM" \
#   --reduce "$REDUCE" \
#   --model "$MODEL" \
#   --year "$YEAR" \
#   --month "$MONTH" \
#   --day "$DAY" \
#   --hour "$HOUR" \
#   --minute "$MINUTE" 

echo -e "\e[32mThe Detection For $YEAR-$MONTH-$DAY Has Finished\e[0m"

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo -e "\e[34m[Time] Step 5 took ${elapsed} seconds\e[0m"