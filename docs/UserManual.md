# Tai lieu van hanh toan dien du an Edge AI Label Barcode Defect Inspection

## 1. Muc tieu tai lieu
Tai lieu nay mo ta day du toan bo du an theo huong su dung thuc te tai hien truong:
- Muc tieu he thong
- Kien truc tong the
- Cac thanh phan da xay dung
- Cach cai dat va khoi dong
- Cach van hanh tren UI
- Quy trinh fine-tune nhanh 1-10 anh
- Quy trinh trien khai demo va kiem thu
- Cau truc thu muc
- Cau hinh
- Logging, output, bao tri va mo rong
- Checklist van hanh cho nguoi dung cuoi va ky su

Tai lieu viet cho doi R&D, ky su trien khai, va user van hanh demo.

---

## 2. Tong quan du an
### 2.1 Bai toan
He thong can xu ly bai toan inspection tren Edge Device:
1. Detect vung label trong anh
2. Crop vung label
3. Detect code 1D/2D va defect trong vung crop
4. Decode barcode/QR
5. Ra quyet dinh OK/NG
6. Xuat ket qua JSON va anh visualization

### 2.2 Dinh huong xay dung
- Uu tien kha nang chay tren Edge
- Uu tien quy trinh thao tac don gian
- Uu tien kha nang adapt nhanh khi gap case moi
- Ho tro fine-tune nhanh tu UI

### 2.3 Doi tuong su dung
- Ky su Vision
- Ky su tu dong hoa
- Ky su trien khai tai line
- User van hanh demo

---

## 3. Kien truc he thong
### 3.1 Pipeline xu ly
Pipeline hien tai su dung luong cascaded:
1. Input image
2. Stage 1: Label detection
3. Stage 2: Crop + preprocess
4. Stage 3: Code detection + defect detection + decode
5. Stage 4: Decision + JSON + visualize

### 3.2 Luong du lieu
- Anh dau vao -> model label -> lay bbox top confidence
- Crop theo bbox -> enhance
- Anh crop -> model code + model defect
- Anh crop -> pyzbar decode
- Tong hop ket qua -> decision
- Tra ket qua cho UI va luu output

### 3.3 Luong adapt case moi
- User upload 1-10 anh case moi
- User gan nhan tren UI
- He thong tao dataset YOLO mini
- He thong train nhanh tren pretrained
- Tra ve best.pt moi
- Cap nhat model path runtime

---

## 4. Thanh phan da trien khai
### 4.1 UI Streamlit (`app.py`)
Chuc nang da co:
- Tab Inference
  - Upload single/multi image
  - Hien thi visualization full + crop
  - Hien thi decision, confidence, decode
  - Hien thi JSON
  - Save result
- Tab Fine-tune Model
  - Chon model target: label/code/defect
  - Upload 1-10 anh
  - Gan nhan tung anh
  - Bam train nhanh
  - Hien thi duong dan model best.pt

### 4.2 Core Pipeline (`src/edge_inspector/core/inspector.py`)
Da trien khai:
- Khoi tao 3 model label/code/defect
- Convert prediction -> boxes
- Chay cascaded pipeline
- Decode barcode co xu ly fallback khi thieu pyzbar
- Tao `InspectionResult`
- Ve visualization
- Save JSON + image

### 4.3 Model wrapper (`src/edge_inspector/core/models.py`)
- Wrapper `YOLOModel`
- Validate model path
- Lazy load model
- predict() thong nhat tham so

### 4.4 Data schema (`src/edge_inspector/core/schemas.py`)
- `BoundingBox`
- `DecodeResult`
- `InspectionResult`
- Rang buoc confidence va decision

### 4.5 Config (`src/edge_inspector/utils/config.py`)
- `load_config()` doc YAML
- `AppConfig.get("a.b.c")` truy cap key nested

### 4.6 Image utils (`src/edge_inspector/utils/image_ops.py`)
- crop bbox
- enhance contrast + sharpen
- draw boxes + labels

### 4.7 Fine-tune manager (`src/edge_inspector/training/fine_tune.py`)
- `FineTuneRequest`: model_type, image_label_pairs, base_model_path, output_dir, epochs, image_size
- `FineTuneManager.prepare_dataset()`:
  - Tao dataset train mini
  - Copy image
  - Tao label YOLO
  - Tao dataset.yaml
- `FineTuneManager.run_training()`:
  - Load pretrained
  - Train ultralytics
  - Tra ve duong dan best.pt

### 4.8 Test (`tests/`)
- test_config
- test_schema
- test_fine_tune prepare dataset

---

## 5. Cau truc thu muc de van hanh

```text
Checking-Label/
  app.py
  requirements.txt
  pyproject.toml
  README.md
  ScopeOfWork.md
  configs/
    config.example.yaml
    config.yaml                 # user tu tao tu example
  weights/
    label_model.pt
    code_model.pt
    defect_model.pt
  outputs/
  runs/
    fine_tune/
  data/
    fine_tune/
  src/edge_inspector/
    core/
    utils/
    training/
  tests/
  docs/
    UserManual.md
```

---

## 6. Yeu cau moi truong
### 6.1 Python
- Python 3.10+

### 6.2 Thu vien chinh
- ultralytics
- opencv-python
- numpy
- Pillow
- streamlit
- pyzbar
- pydantic
- PyYAML

### 6.3 Luu y pyzbar
Can cai dat zbar trong he dieu hanh de decode on dinh.

---

## 7. Huong dan cai dat tu dau
### 7.1 Tao moi truong
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 7.2 Tao config runtime
```bash
cp configs/config.example.yaml configs/config.yaml
```

### 7.3 Chuan bi model
- Tao thu muc `weights/`
- Copy 3 file:
  - `weights/label_model.pt`
  - `weights/code_model.pt`
  - `weights/defect_model.pt`

### 7.4 Chay he thong
```bash
streamlit run app.py
```

---

## 8. Mo ta file cau hinh
Noi dung mau (`configs/config.example.yaml`):
- models
  - label_model_path
  - code_model_path
  - defect_model_path
- inference
  - image_size
  - conf_threshold
  - iou_threshold
  - max_det
  - device
- preprocess
  - enhance_contrast
  - sharpen
  - brightness_alpha
  - brightness_beta
  - deskew
- output
  - save_dir
  - save_visualization
  - save_json
- training
  - output_dir
- ui
  - page_title
  - page_icon

Khuyen nghi:
- Demo PC: de `device: cpu`
- Demo Jetson co GPU: dat device phu hop

---

## 9. Huong dan van hanh Tab Inference
### 9.1 Muc tieu
Danh gia nhanh anh dau vao va cho ket qua inspection.

### 9.2 Cac buoc thao tac
1. Mo tab `Inference`
2. Upload 1 hoac nhieu anh
3. Xem:
   - Anh full voi bbox label
   - Anh crop voi bbox code/defect
   - Decision OK/NG
   - Confidence
   - Decode text
   - JSON
4. Bam `Save Result` neu can luu ket qua

### 9.3 Dinh dang output
- JSON: timestamp, image_name, decision, confidence, label_box, code_boxes, defect_boxes, decode_result, notes
- Image: anh visualize

### 9.4 Luu y
- Neu khong detect duoc label -> ket qua NG voi note thong bao
- Neu pyzbar khong san sang -> decode false, pipeline van chay

---

## 10. Huong dan van hanh Tab Fine-tune Model
### 10.1 Muc tieu
Cho user cuoi tu adapt case moi khi model hien tai chua xu ly tot.

### 10.2 Cac buoc thao tac chi tiet
1. Mo tab `Fine-tune Model`
2. Chon `Target model`
   - label
   - code
   - defect
3. Chon so epoch (khuyen nghi 3-10 cho demo nhanh)
4. Upload 1-10 anh cua case moi
5. Gan nhan cho tung anh theo dropdown
6. Bam `Train nhanh & cap nhat model`
7. Cho den khi hien thi `Fine-tune hoan tat`
8. Lay duong dan `best.pt` moi
9. Refresh app neu can tai lai model runtime sach

### 10.3 Ben trong he thong lam gi
- Luu tam anh upload
- Tao dataset mini theo format YOLO
- Tao label full-image bbox de nhanh
- Tao dataset.yaml
- Train ultralytics tu pretrained
- Tra ve checkpoint best

### 10.4 Luu y chat luong
- Cach label full-image bbox la workflow nhanh cho demo
- Muon dat chat luong cao hon can bbox chinh xac
- Nen thu thap them data da dang

---

## 11. Quy trinh demo de xac nhan he thong
### 11.1 Demo co ban
1. Chuan bi 20-50 anh test
2. Chay tab inference
3. Danh dau case fail
4. Tong hop NG do miss detect / sai decode / false defect

### 11.2 Demo adapt nhanh
1. Lay 1-10 anh fail
2. Fine-tune trong tab fine-tune
3. Chay lai inference tren tap fail
4. So sanh truoc/sau

### 11.3 Tieu chi danh gia
- Ty le detect label
- Ty le decode
- Ty le defect detection
- Ty le quyet dinh dung OK/NG
- Toc do suy luan

---

## 12. Van hanh output va bao cao
### 12.1 Thu muc output
- Ket qua luu vao `outputs/`
- Moi lan save se tao file json va jpg theo timestamp

### 12.2 Cach dung output
- JSON dung de tich hop MES/SCADA/ERP sau nay
- Image dung de review va trace loi

---

## 13. Checklist handover cho user cuoi
### 13.1 Truoc khi ban giao
- [ ] Da copy du 3 model vao `weights/`
- [ ] Da tao `configs/config.yaml`
- [ ] App chay duoc
- [ ] Inference tren anh mau OK
- [ ] Save result tao duoc JSON/JPG
- [ ] Fine-tune 1 case mau thanh cong

### 13.2 Sau khi ban giao
- [ ] User biet upload anh
- [ ] User biet doc ket qua
- [ ] User biet quy trinh fine-tune 1-10 anh
- [ ] User biet thu muc output

---

## 14. Loi thuong gap va cach xu ly
### Loi 1: Khong tim thay model
- Nguyen nhan: sai path trong config hoac chua copy model
- Xu ly: kiem tra `weights/*.pt` va config

### Loi 2: Decode khong ra
- Nguyen nhan: pyzbar/zbar, chat luong anh, crop sai
- Xu ly: cai zbar, tang preprocess, cai thien data

### Loi 3: Fine-tune that bai
- Nguyen nhan: anh loi, label sai, model base loi
- Xu ly: kiem tra file upload, ten nhan, base model

### Loi 4: Toc do cham
- Nguyen nhan: chay CPU, model nang
- Xu ly: giam imgsz, dung GPU, chuyen TensorRT

---

## 15. Dinh huong nang cap tiep theo
1. Them cong cu gan bbox truc tiep tren UI
2. Fine-tune co val split that
3. Quan ly version model
4. One-click deploy Jetson
5. Camera realtime + trigger
6. Dashboard thong ke theo ngay/ca

---

## 16. SOP van hanh ngan gon cho user
1. Mo app
2. Upload anh
3. Xem OK/NG
4. Save output
5. Neu fail: vao fine-tune
6. Upload 1-10 anh fail + gan nhan
7. Train nhanh
8. Chay lai inference xac nhan ket qua

---

## 17. Ghi chu quan trong
- Tai lieu nay tap trung vao van hanh demo va adapt nhanh.
- De dat production grade, can bo sung kiem thu sau, benchmark chuan va quy trinh MLOps day du.
- Tuy nhien voi hien trang code, he thong da co khung day du de demo toan bo workflow end-to-end.