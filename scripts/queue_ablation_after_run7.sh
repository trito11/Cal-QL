#!/usr/bin/env bash
# =============================================================================
# queue_ablation_after_run7.sh
# Hàng đợi tự động: Chờ Run 7 hoàn thành rồi tự động chạy tuần tự Run 0 -> Run 6
# =============================================================================

echo "=========================================================================="
echo "  ABLATION QUEUE WATCHER INITIATED"
echo "  Đang theo dõi tiến trình Run 7 trên GPU..."
echo "=========================================================================="

# Chờ cho đến khi tiến trình Run 7 (conservative_sac_main_sam với run_id=7) kết thúc
while pgrep -f "run_id=7" >/dev/null; do
    sleep 20
done

echo ""
echo "=========================================================================="
echo "  [NOTIFICATION] RUN 7 ĐÃ HOÀN TẤT!"
echo "  Bắt đầu khởi động chuỗi Ablation Study từ Run 0 đến Run 6..."
echo "=========================================================================="

# Chạy lần lượt từ Run 0 đến Run 6
./scripts/run_adroit_action_sam_ablation.sh 0-6

echo ""
echo "=========================================================================="
echo "  TOÀN BỘ CÁC RUN 0 ĐẾN 6 ĐÃ HOÀN THÀNH!"
echo "  Đang tổng hợp bảng kết quả đối sánh tự động..."
echo "=========================================================================="

/home/tri/miniconda3/envs/Cal-QL/bin/python scratch/compare_ablation.py
