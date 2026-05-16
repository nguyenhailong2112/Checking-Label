# Colab Training Workflow - Label Detection, Classification, Few-shot Identity

Ngay cap nhat: 16/05/2026

Tai lieu nay huong dan train 3 artifact chinh cho checkpoint hien tai:

1. Label Detection YOLO weight.
2. Label Classification closed-set baseline weight.
3. Few-shot Label Identity encoder weight.

Script chinh:

```text
scripts/colab_train_label_stack.py
```

## 1. Cau Truc Dataset Tren Colab

Dat folder `datasets` tren Google Drive giong workspace local:

```text
/content/drive/MyDrive/CheckingLabel/datasets/
  dataLabelDetection/
    data.yaml
    train/images
    train/labels
    valid/images
    valid/labels
    test/images
    test/labels
  dataLabelClassification/
    train/<class_name>/*.jpg
    valid/<class_name>/*.jpg
    test/<class_name>/*.jpg
  dataLabelClassification_seen/
    train/<class_name>/*.jpg
    valid/<class_name>/*.jpg
  dataLabelClassification_unseen/
    train/<class_name>/*.jpg
```

Luu y: `dataLabelDetection/data.yaml` hien dang co path Colab:

```yaml
path: /content/drive/MyDrive/CheckingLabel/datasets/dataLabelDetection
```

Neu Drive cua ban khac path, sua lai field `path` truoc khi train YOLO.

## 2. Setup Colab

Trong notebook Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Clone/copy repo vao Colab, sau do:

```bash
cd /content/drive/MyDrive/CheckingLabel/Checking-Label
pip install -U ultralytics opencv-python-headless pillow pydantic pyyaml
```

Torch/torchvision thuong da co san tren Colab. Neu can:

```bash
pip install -U torch torchvision
```

## 3. Train Tat Ca Artifact

```bash
python scripts/colab_train_label_stack.py \
  --task all \
  --dataset-root /content/drive/MyDrive/CheckingLabel/datasets \
  --project /content/drive/MyDrive/CheckingLabel/runs \
  --device 0 \
  --det-base-model yolo11s.pt \
  --det-epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --cls-epochs 30 \
  --fewshot-epochs 40 \
  --n-way 5 \
  --k-shot 3 \
  --query-num 3
```

Output du kien:

```text
/content/drive/MyDrive/CheckingLabel/runs/label_detection_yolo11/weights/best.pt
/content/drive/MyDrive/CheckingLabel/runs/label_classifier_resnet50/best_classifier.pt
/content/drive/MyDrive/CheckingLabel/runs/label_identity_fewshot/best_fewshot_encoder.pt
```

## 4. Train Rieng Tung Phan

### Detection

```bash
python scripts/colab_train_label_stack.py \
  --task detection \
  --dataset-root /content/drive/MyDrive/CheckingLabel/datasets \
  --project /content/drive/MyDrive/CheckingLabel/runs \
  --device 0
```

Artifact can copy ve app:

```text
weights/label_model.pt
```

### Classification

```bash
python scripts/colab_train_label_stack.py \
  --task classifier \
  --dataset-root /content/drive/MyDrive/CheckingLabel/datasets \
  --project /content/drive/MyDrive/CheckingLabel/runs \
  --device 0
```

Artifact:

```text
runs/label_classifier_resnet50/best_classifier.pt
```

Classification artifact la baseline closed-set de doi chieu. Runtime checkpoint hien tai uu tien open-set identity, khong ep unknown vao classifier head.

### Few-shot Identity

```bash
python scripts/colab_train_label_stack.py \
  --task fewshot \
  --dataset-root /content/drive/MyDrive/CheckingLabel/datasets \
  --project /content/drive/MyDrive/CheckingLabel/runs \
  --device 0 \
  --n-way 5 \
  --k-shot 3 \
  --query-num 3
```

Artifact:

```text
runs/label_identity_fewshot/best_fewshot_encoder.pt
```

Day la artifact quan trong nhat cho lop Label Identity giai do sau. MVP hien tai co encoder histogram nhe de test gallery/UI truoc; encoder ProtoNet se duoc noi vao runtime sau khi weight nay san sang.
Sau khi copy weight ve local, dat trong `configs/config.yaml`:

```yaml
identity:
  enabled: true
  encoder_path: "weights/best_fewshot_encoder.pt"
```

## 5. Evaluate Identity Local

Sau khi co dataset local, co the chay:

```bash
python scripts/evaluate_identity.py \
  --seen-root datasets/dataLabelClassification_seen \
  --unseen-root datasets/dataLabelClassification_unseen \
  --shots 1 3 5 \
  --threshold 0.72 \
  --output-json reports/identity_eval.json
```

Report gom:

- Known accuracy tren seen validation.
- Unknown recall tren unseen train.
- Few-shot enroll accuracy cho unseen classes co du query image.

## 6. Copy Weights Ve App Local

Sau khi train Detection xong, copy:

```text
runs/label_detection_yolo11/weights/best.pt -> weights/label_model.pt
```

Trong app:

1. Chon mode `label_only`.
2. Bat/tat `Enable open-set identity` tuy gallery da co hay chua.
3. Upload anh test.
4. Neu identity gallery rong, vao tab `Identity Gallery` upload crop mau hoac enroll truc tiep tu ket qua inference.

## 7. Ghi Chu Van Hanh

- Detection model phai chay truoc de tao crop label.
- Classification model la artifact tham chieu/backup, khong thay the open-set identity.
- Few-shot encoder can evaluation threshold truoc khi dua vao runtime production.
- Moi model moi nen duoc copy/stage qua tab `Quick Teach / Models` neu dung cho detection/code/defect.
- Identity gallery la du lieu runtime rieng, nam trong `data/identity/galleries`.
