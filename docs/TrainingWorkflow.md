# PC Training Workflow – Label Studio / Roboflow / YOLOv11 / TensorRT

Tài liệu này mô tả workflow chuẩn cho Scope Ver 2: **Edge chỉ inference + collect data**, còn PC/Workstation thực hiện annotate, fine-tune, evaluate và export model mới.

---

## 1. Dữ liệu đầu vào từ Edge

Sau khi operator bấm **Collect for Training** hoặc bật auto-collect NG/low-confidence, Edge lưu dữ liệu vào:

```text
data/collected/
  ng/
    images/
    metadata/
    visualizations/
    crops/
  low_confidence/
    images/
    metadata/
    visualizations/
    crops/
  manual/
    images/
    metadata/
    visualizations/
    crops/
```

Khuyến nghị copy toàn bộ folder này từ Jetson/Edge về PC theo ngày hoặc theo batch sản xuất:

```powershell
robocopy \\EDGE_DEVICE\share\data\collected D:\edge_ai_dataset\collected /E
```

Hoặc copy thủ công qua USB/network share.

---

## 2. Lọc dữ liệu trước khi annotate

Trước khi đưa vào annotation tool, nên lọc:

- Ảnh trùng lặp.
- Ảnh out-of-focus quá nặng không thể học.
- Ảnh không đúng product/không đúng line.
- Metadata thiếu hoặc file ảnh lỗi.

Khuyến nghị chia batch:

```text
dataset_batches/
  2026-05-11_batch_001/
    images/
    metadata/
  2026-05-12_batch_002/
    images/
    metadata/
```

---

## 3. Annotate bằng Label Studio

### 3.1 Cài Label Studio trên PC

```bash
python -m venv .venv-labelstudio
.venv-labelstudio\Scripts\activate
pip install label-studio
label-studio start
```

### 3.2 Tạo project annotation

Tạo riêng project cho từng model:

1. **Label Detection**
   - Classes: `PCLabel`, `SanDisk`, `WesternDigitalType1`, `WesternDigitalType2`, `WesternDigitalType3`, `WesternDigitalType4`
2. **Code Detection**
   - Classes: `Code1D`, `Code2D`
3. **Defect Detection**
   - Classes: `DefectNG`

Không nên trộn 3 bài toán vào một project nếu mục tiêu vẫn là 3 model độc lập.

### 3.3 Export dataset

Export sang YOLO format nếu Label Studio setup hỗ trợ trực tiếp. Nếu export format khác, cần viết converter để tạo:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  dataset.yaml
```

---

## 4. Annotate bằng Roboflow

Roboflow phù hợp khi cần:

- Quản lý dataset version.
- Augmentation nhanh.
- Export YOLO format dễ dàng.
- Chia train/val/test tự động.

Workflow:

1. Upload images.
2. Annotate bbox.
3. Generate dataset version.
4. Export YOLOv8/YOLOv11 compatible format.
5. Download về PC training workspace.

---

## 5. Fine-tune YOLOv11 trên PC

Ví dụ train bằng Ultralytics CLI:

```bash
yolo detect train model=weights/label_model.pt data=dataset_label/dataset.yaml imgsz=640 epochs=50 batch=16 device=0 project=runs/train name=label_v2

yolo detect train model=weights/code_model.pt data=dataset_code/dataset.yaml imgsz=640 epochs=50 batch=16 device=0 project=runs/train name=code_v2

yolo detect train model=weights/defect_model.pt data=dataset_defect/dataset.yaml imgsz=640 epochs=80 batch=16 device=0 project=runs/train name=defect_v2
```

Gợi ý:

- Defect thường cần nhiều augmentation hơn label/code.
- Code blur/tilt nên thêm blur, rotation, perspective.
- Không nên train quá ít ảnh nếu bbox annotation chưa chuẩn.

---

## 6. Evaluate model cũ/mới

Chạy validation:

```bash
yolo detect val model=runs/train/label_v2/weights/best.pt data=dataset_label/dataset.yaml imgsz=640 device=0
```

Theo dõi:

- mAP50/mAP50-95.
- Precision/Recall.
- Confusion matrix.
- False NG / False OK trên data thực tế.
- FPS trên PC và sau đó trên Jetson.

Chỉ deploy model mới khi model mới tốt hơn model đang chạy hoặc giải quyết được case NG/low-confidence quan trọng.

---

## 7. Export TensorRT engine

Dùng helper trong repo:

```bash
python scripts/export_tensorrt.py --model runs/train/label_v2/weights/best.pt --imgsz 640 --device 0 --half
python scripts/export_tensorrt.py --model runs/train/code_v2/weights/best.pt --imgsz 640 --device 0 --half
python scripts/export_tensorrt.py --model runs/train/defect_v2/weights/best.pt --imgsz 640 --device 0 --half
```

Kết quả `.engine` nên được test trên đúng Jetson/runtime target vì TensorRT engine phụ thuộc môi trường phần cứng/runtime.

---

## 8. Benchmark trước khi deploy

Dùng benchmark CLI:

```bash
python scripts/benchmark.py --config configs/config.yaml --input datasets/test_images --warmup 5 --output-json reports/benchmark_label_code_defect.json
```

Cần ghi lại:

- Average latency.
- P50/P95 latency.
- FPS average.
- OK/NG distribution.
- Model artifact version.

---

## 9. Deploy lại Edge

1. Copy `.pt` hoặc `.engine` về Edge.
2. Mở Streamlit tab **Quick Teach / Models**.
3. Upload/stage model artifact.
4. Apply runtime.
5. Test 5–10 ảnh sanity check.
6. Nếu model mới lỗi, rollback staged artifact cũ từ history.

---

## 10. Checklist trước khi bàn giao model mới

- [ ] Dataset version rõ ràng.
- [ ] Annotation classes đúng.
- [ ] Validation metrics tốt hơn hoặc giải quyết case quan trọng.
- [ ] Benchmark latency không tệ hơn ngưỡng chấp nhận.
- [ ] Test thực tế trên ảnh OK/NG.
- [ ] Staged artifact có note rõ ràng.
- [ ] Có thể rollback model cũ.