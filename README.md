# Edge AI Label & Barcode Inspection

Hệ thống kiểm tra label, barcode 1D/2D và defect theo kiến trúc cascaded pipeline:

1. Label detection
2. Crop & preprocess
3. Code detection + defect detection + barcode decoding
4. Decision (OK/NG) + JSON output + visualization

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.example.yaml configs/config.yaml
mkdir -p weights
# copy model vào weights: label_model.pt, code_model.pt, defect_model.pt
streamlit run app.py
```

## Fine-tune nhanh trên UI

- Vào tab **Fine-tune Model**
- Chọn target model (`label`, `code`, `defect`)
- Upload 1-10 ảnh mới
- Gán nhãn cho từng ảnh trực tiếp trên UI
- Bấm **Train nhanh & cập nhật model** để chạy fine-tune với pretrained `.pt`
- Sau khi train xong, ứng dụng trả về đường dẫn `best.pt`

## Cấu trúc

- `src/edge_inspector/core/inspector.py`: Pipeline chính `LabelBarcodeInspector`
- `src/edge_inspector/core/models.py`: Wrapper YOLO
- `src/edge_inspector/core/schemas.py`: JSON schema với Pydantic
- `src/edge_inspector/utils/image_ops.py`: Preprocess, deskew, visualize
- `src/edge_inspector/utils/config.py`: Load cấu hình YAML
- `src/edge_inspector/training/fine_tune.py`: Prepare dataset + fine-tune training manager
- `app.py`: Streamlit UI

## Tai lieu chi tiet
- Xem tai lieu day du tai `docs/UserManual.md`