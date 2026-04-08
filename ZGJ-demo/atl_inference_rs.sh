#!/bin/bash
set -uo pipefail

ROOT_DIR="/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序"
XCODE_DIR="$ROOT_DIR/X-code"
DEMO_DIR="$XCODE_DIR/mmsegmentation/ZGJ-demo"

CONFIG_PATH="$XCODE_DIR/mmsegmentation/configs_new/ZGJ/2-18类地物/ZGJ-S2-BEiTv2-mask2former-18类-512-new.py"
CHECKPOINT_PATH="$XCODE_DIR/mmsegmentation/checkpoints/ZGJ-S2-BEiTv2-mask2former-18类-512-2-iter_80000.pth"
GPU_ID="${GPU_ID:-1}"

LIST_TXT="/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/X-code/img_txt.txt"
IMAGE_DIR="/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/0-测试数据/2-波段组合结果"
RAW_MASK_DIR="/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/0-测试数据/3-推理raw_mask"
VECTOR_DIR="/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/0-测试数据/4-有效区域矢量"
FINAL_MASK_DIR="/opt/workspace/AI-Tianlong/ZGJ/0-检察遥感平台部署/1-哨兵2号-地物分类程序/0-测试数据/5-18类地物分类mask"

usage() {
    cat << 'EOF'
用法:
  bash atl_inference_rs.sh \
    --LIST_TXT <影像名列表txt> \
    --IMAGE_DIR <输入影像目录> \
    --RAW_MASK_DIR <原始推理mask输出目录> \
    --VECTOR_DIR <有效区域矢量输出目录> \
    --FINAL_MASK_DIR <最终裁切结果目录>

说明:
  1) LIST_TXT 每行一个影像名，支持写完整绝对路径。
  2) 若写影像名，脚本会自动拼接 IMAGE_DIR。
  3) 会生成成功和失败清单，失败不会中断全局批处理。
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --LIST_TXT)
            LIST_TXT="$2"; shift 2 ;;
        --IMAGE_DIR)
            IMAGE_DIR="$2"; shift 2 ;;
        --RAW_MASK_DIR)
            RAW_MASK_DIR="$2"; shift 2 ;;
        --VECTOR_DIR)
            VECTOR_DIR="$2"; shift 2 ;;
        --FINAL_MASK_DIR)
            FINAL_MASK_DIR="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1 ;;
    esac
done

[[ -n "$LIST_TXT" ]] || { echo "缺少参数 --LIST_TXT"; usage; exit 1; }
[[ -n "$IMAGE_DIR" ]] || { echo "缺少参数 --IMAGE_DIR"; usage; exit 1; }
[[ -n "$RAW_MASK_DIR" ]] || { echo "缺少参数 --RAW_MASK_DIR"; usage; exit 1; }
[[ -n "$VECTOR_DIR" ]] || { echo "缺少参数 --VECTOR_DIR"; usage; exit 1; }
[[ -n "$FINAL_MASK_DIR" ]] || { echo "缺少参数 --FINAL_MASK_DIR"; usage; exit 1; }

[[ -f "$LIST_TXT" ]] || { echo "列表文件不存在: $LIST_TXT"; exit 1; }
[[ -d "$IMAGE_DIR" ]] || { echo "输入影像目录不存在: $IMAGE_DIR"; exit 1; }

mkdir -p "$RAW_MASK_DIR" "$VECTOR_DIR" "$FINAL_MASK_DIR"

RUN_LOG_DIR="$FINAL_MASK_DIR/_batch_logs"
mkdir -p "$RUN_LOG_DIR"
SUCCESS_LIST="$RUN_LOG_DIR/success.txt"
FAIL_LIST="$RUN_LOG_DIR/fail.txt"
DETAIL_LOG="$RUN_LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

: > "$SUCCESS_LIST"
: > "$FAIL_LIST"

echo "======================================================" | tee -a "$DETAIL_LOG"
echo "批处理开始: $(date '+%F %T')" | tee -a "$DETAIL_LOG"
echo "LIST_TXT=$LIST_TXT" | tee -a "$DETAIL_LOG"
echo "IMAGE_DIR=$IMAGE_DIR" | tee -a "$DETAIL_LOG"
echo "RAW_MASK_DIR=$RAW_MASK_DIR" | tee -a "$DETAIL_LOG"
echo "VECTOR_DIR=$VECTOR_DIR" | tee -a "$DETAIL_LOG"
echo "FINAL_MASK_DIR=$FINAL_MASK_DIR" | tee -a "$DETAIL_LOG"
echo "======================================================" | tee -a "$DETAIL_LOG"

total=0
success=0
failed=0

while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    total=$((total + 1))

    image_path="$line"
    if [[ "$image_path" != /* ]]; then
        image_path="$IMAGE_DIR/$image_path"
    fi

    image_name="$(basename "$image_path")"
    scene_name="${image_name%.*}"
    raw_mask_path="$RAW_MASK_DIR/$image_name"

    echo "[$total] 开始处理: $image_name" | tee -a "$DETAIL_LOG"

    if [[ ! -f "$image_path" ]]; then
        echo "[$total] 输入影像不存在: $image_path" | tee -a "$DETAIL_LOG"
        echo "$image_name | 输入影像不存在" >> "$FAIL_LIST"
        failed=$((failed + 1))
        continue
    fi

    set +e

    (
      cd "$DEMO_DIR" && \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      python atl_inference_rs.py \
        --config "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --DATA_INPUT_DIR1 "$image_path" \
        --DATA_OUTPUT_DIR "$RAW_MASK_DIR"
    ) >> "$DETAIL_LOG" 2>&1
    rc_infer=$?

    if [[ $rc_infer -ne 0 ]]; then
        echo "[$total] 推理失败: $image_name" | tee -a "$DETAIL_LOG"
        echo "$image_name | 推理失败" >> "$FAIL_LIST"
        failed=$((failed + 1))
        set -e
        continue
    fi

    (
      cd "$XCODE_DIR" && \
      python code-1-有效区域提取.py \
        --IMAGE_PATH "$image_path" \
        --OUTPUT_DIR "$VECTOR_DIR"
    ) >> "$DETAIL_LOG" 2>&1
    rc_vector=$?

    if [[ $rc_vector -ne 0 ]]; then
        echo "[$total] 有效区域提取失败: $image_name" | tee -a "$DETAIL_LOG"
        echo "$image_name | 有效区域提取失败" >> "$FAIL_LIST"
        failed=$((failed + 1))
        set -e
        continue
    fi

    (
      cd "$XCODE_DIR" && \
      python code-3-有效区域后处理裁切.py \
        --IMAGE_PATH "$raw_mask_path" \
        --VECTOR_PATH "$VECTOR_DIR" \
        --OUTPUT_PATH "$FINAL_MASK_DIR"
    ) >> "$DETAIL_LOG" 2>&1
    rc_clip=$?

    set -e

    if [[ $rc_clip -ne 0 ]]; then
        echo "[$total] 裁切失败: $image_name" | tee -a "$DETAIL_LOG"
        echo "$image_name | 裁切失败" >> "$FAIL_LIST"
        failed=$((failed + 1))
        continue
    fi

    echo "[$total] 成功: $image_name" | tee -a "$DETAIL_LOG"
    echo "$image_name" >> "$SUCCESS_LIST"
    success=$((success + 1))

done < "$LIST_TXT"

echo "======================================================" | tee -a "$DETAIL_LOG"
echo "批处理完成: $(date '+%F %T')" | tee -a "$DETAIL_LOG"
echo "总数: $total, 成功: $success, 失败: $failed" | tee -a "$DETAIL_LOG"
echo "成功清单: $SUCCESS_LIST" | tee -a "$DETAIL_LOG"
echo "失败清单: $FAIL_LIST" | tee -a "$DETAIL_LOG"
echo "详细日志: $DETAIL_LOG" | tee -a "$DETAIL_LOG"
echo "======================================================" | tee -a "$DETAIL_LOG"

if [[ $failed -gt 0 ]]; then
    exit 2
fi
