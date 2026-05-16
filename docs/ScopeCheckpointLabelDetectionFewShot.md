# Scope Checkpoint - Label Detection + Open-set Few-shot Label Identity

Ngay cap nhat: 16/05/2026

Trang thai: Scope checkpoint hien tai da thong nhat truoc khi bat dau coding phase moi.

Tai lieu lien quan:

- `ScopeOfWorkVer2.md`
- `docs/ProjectAuditCheckpoint.md`
- `docs/ScopeVer2PlusTeachMode.md`
- `docs/TeachModeImplementationCheckpoint.md`
- `Animal-Classification/README.md`

## 1. Executive Summary

Du an tiep tuc giu chien luoc **Hybrid Edge AI**:

- Edge/Laptop chay inference, visualization, decision, data collection va teach/enroll nhe.
- PC/Colab/Workstation phu trach train/fine-tune nang, evaluate va export model.

Checkpoint moi cua phase hien tai tap trung vao bai toan:

```text
Label Detection + Open-set Few-shot Label Identity Recognition
```

Dieu nay co nghia la:

1. Model Detection van duoc giu va phai chay on dinh de tim dung bbox label.
2. Bbox label duoc crop thanh anh label chinh dien de dua vao module identity phia sau.
3. Module identity khong dung classifier dong kin ep vao `num_classes`.
4. Module identity dung metric learning/few-shot/open-set recognition de tra loi:
   - Crop nay giong class da biet nao?
   - Neu khong giong du nguong, day la `UNKNOWN_LABEL`.
   - Neu operator gan ten class moi, he thong enroll class moi bang 1-20 anh mau.
5. Du lieu enroll/teach duoc luu lai thanh dataset dung chuan de train/fine-tune encoder/backbone tren PC/Colab sau nay.

## 2. Dinh Vi Chien Luoc

Phase truoc da chot Scope Ver 2: Edge khong phai training server. Phase moi khong phu dinh dieu do. Phase moi bo sung mot tang AI nhe sau detection:

```text
Input image/live frame
        |
        v
YOLO Label Detection
        |
        v
Label crop
        |
        v
Few-shot/Open-set Label Identity
        |
        v
Known class / Unknown label / Enroll new class
        |
        v
Decision + Visualization + Active Learning + Dataset accumulation
```

Dung tu chuan cho module moi:

- `Label Detection`: tim vung label tren anh goc.
- `Label Identity`: nhan dien label crop thuoc product/class nao.
- `Open-set Recognition`: co kha nang tu choi khi gap class moi.
- `Few-shot Enroll`: them class moi bang vai mau ma khong can train lai ngay.
- `Gallery/Prototype`: bo nho cac embedding mau cua class da teach.

## 3. Hien Trang Repo Tai Checkpoint

Da co:

- Streamlit app trong `app.py`.
- Core pipeline trong `src/edge_inspector/core/inspector.py`.
- YOLO wrapper trong `src/edge_inspector/core/models.py`.
- DecisionEngine trong `src/edge_inspector/core/decision.py`.
- Active Learning collector trong `src/edge_inspector/core/active_learning.py`.
- Teach Mode MVP trong `src/edge_inspector/teach/`.
- Recipe schema, dataset writer YOLO va scoring helper.
- Model registry stage/apply/rollback `.pt` va `.engine`.
- Benchmark helper va TensorRT export helper.

Chua co:

- Module `identity/` cho label crop recognition.
- Embedding encoder wrapper.
- Gallery/prototype store.
- Open-set threshold calibration.
- UI enroll class moi tu crop label.
- Identity fields trong `InspectionResult`.
- Evaluation script cho seen/unseen identity.

## 4. Hien Trang Dataset Tai Checkpoint

Dataset Detection:

- Path: `datasets/dataLabelDetection`
- Classes: 23 label classes.
- Train: 8193 images / 8193 label files.
- Valid: 251 images / 251 label files.
- Test: 1 image / 1 label file.
- Gan nhu moi anh chi co 1 bbox label.
- Can clean/canh bao 2 case trong train:
  - 1 label file rong.
  - 1 label file co 2 bbox.

Dataset Classification/Identity:

- Path: `datasets/dataLabelClassification`
- Train: 8192 images.
- Valid: 251 images.
- Test: 1 image.

Dataset split cho few-shot/open-set:

- Seen classes: `datasets/dataLabelClassification_seen`
- Unseen classes: `datasets/dataLabelClassification_unseen`

Y nghia:

- Seen classes dung de train/fine-tune embedding encoder.
- Unseen classes dung de danh gia kha nang phat hien label moi va few-shot enroll.
- Dataset bi imbalance manh, nen khong train theo sampling ngau nhien thong thuong.
- Can dung episodic/balanced sampler theo class de tranh class lon ap dao class nho.

## 5. In Scope Cho Phase Hien Tai

### 5.1 Label Detection Baseline

- Chay duoc `label_only` voi `weights/label_model.pt`.
- Khong bat buoc phai co `code_model.pt` va `defect_model.pt` trong phase nay.
- Hien thi full image bbox va label crop.
- Cho phep test upload image local.
- Sau do mo rong camera snapshot/live view.

### 5.2 Label Crop Identity

- Them module identity de xu ly label crop.
- Tao embedding vector tu crop label.
- So sanh crop voi gallery/prototype cac class da biet.
- Tra ve:
  - predicted identity class.
  - similarity/distance score.
  - known/unknown decision.
  - top-k nearest classes.

### 5.3 Open-set Unknown Detection

- Neu best similarity thap hon nguong, tra ve `UNKNOWN_LABEL`.
- Khong ep unknown vao class cu.
- Luu unknown crop vao active learning/identity review queue.
- Cho operator gan ten class moi.

### 5.4 Few-shot Enroll

- Operator co the tao class moi bang 1-20 crop mau.
- He thong luu exemplar crop va embedding vao gallery.
- He thong tinh prototype moi bang mean embedding cua cac exemplar.
- Class moi co the duoc nhan dien ngay trong runtime ma khong train lai model.

### 5.5 Dataset Accumulation

- Luu crop label, metadata, predicted identity, similarity score, operator label.
- Co cau truc export duoc de train/fine-tune tren PC/Colab.
- Tach du lieu:
  - known accepted.
  - unknown pending.
  - operator enrolled.
  - false match/negative correction.

### 5.6 Evaluation Local

- Viet script evaluate identity tren:
  - seen validation.
  - unseen classes.
  - k-shot simulation: 1-shot, 3-shot, 5-shot.
- Bao cao:
  - known accuracy.
  - unknown detection precision/recall.
  - AUROC hoac threshold sweep neu du lieu du.
  - confusion/top-k cho known classes.

## 6. Out Of Scope Cho Phase Hien Tai

Tam thoi chua lam:

- Code Reader 1D/2D full-stack.
- DefectNG inspection full-stack.
- PLC integration.
- Multi-camera industrial.
- Full YOLO training truc tiep tren Edge.
- Auto-apply model moi khong co evaluate.
- Cloud sync tu dong.
- UI annotation day du nhu Label Studio.

Code/Defect van giu trong kien truc, nhung khong phai trong tam cua checkpoint nay.

## 7. Kien Truc Module De Them

De xuat them:

```text
src/edge_inspector/identity/
  __init__.py
  schemas.py
  encoder.py
  gallery.py
  inference.py
  evaluation.py
```

### `schemas.py`

Chua:

- `IdentityPrediction`
- `IdentityMatch`
- `IdentityGalleryClass`
- `IdentityEnrollRecord`
- `IdentityRuntimeSettings`

### `encoder.py`

Chua:

- Load embedding model.
- Preprocess crop label.
- Extract normalized embedding.
- Ho tro CPU truoc, CUDA/ONNX/TensorRT sau.

### `gallery.py`

Chua:

- Luu exemplar image paths.
- Luu embeddings/prototypes.
- Add class moi.
- Add exemplar vao class da co.
- Rebuild prototypes.
- Save/load gallery metadata.

### `inference.py`

Chua:

- Compare query embedding voi prototype.
- Top-k nearest identity.
- Unknown threshold decision.
- Optional margin rule giua top-1 va top-2.

### `evaluation.py`

Chua:

- Tao episodic evaluation.
- Simulate k-shot enroll.
- Sweep threshold.
- Report known/unknown metrics.

## 8. De Xuat Cau Truc Du Lieu Identity

```text
data/identity/
  galleries/
    default/
      gallery.json
      prototypes.npy
      embeddings.npy
      exemplars/
        PCLabel/
        SanDisk1T02/
        NewProductA/
  pending_unknown/
    images/
    metadata/
  enrolled/
    images/
    metadata/
  evaluation/
    reports/
```

Trong `gallery.json` can co:

```json
{
  "gallery_id": "default",
  "encoder_name": "label_identity_encoder",
  "embedding_dim": 128,
  "similarity_metric": "cosine",
  "unknown_threshold": 0.72,
  "classes": {
    "PCLabel": {
      "prototype_index": 0,
      "num_exemplars": 5
    }
  }
}
```

## 9. Model Strategy

### Baseline nhanh nhat

Dung pretrained feature extractor co san:

- ResNet50/efficient backbone tu torchvision.
- Cat classification head.
- Lay embedding sau pooling/projection.
- Normalize embedding.

Uu diem:

- Code nhanh.
- Chay duoc CPU.
- Co baseline de test pipeline.

Nhuoc diem:

- Feature ImageNet co the chua toi uu cho label cong nghiep.

### Ban chuan cho du an

Train/fine-tune metric encoder tren `dataLabelClassification_seen`:

- Prototypical Network style.
- Episodic training: N-way K-shot.
- Projection head ra embedding 128/256 dim.
- Balanced class episodic sampler.
- Augmentation vua phai: brightness, contrast, blur nhe, perspective nhe.

Evaluate bang:

- Seen validation.
- Unseen classes.
- Threshold sweep known vs unknown.
- 1-shot/3-shot/5-shot enroll simulation.

### Ban nang cao sau nay

- Siamese/contrastive training de calibrate unknown threshold tot hon.
- Self-supervised pretrain tren tat ca crop label.
- ONNX/TensorRT export cho encoder.

## 10. Runtime Decision Logic

Output identity can co logic:

```text
best_similarity >= accept_threshold
  -> KNOWN_LABEL

best_similarity < unknown_threshold
  -> UNKNOWN_LABEL

unknown_threshold <= best_similarity < accept_threshold
  -> LOW_CONF_IDENTITY
```

Co the them margin rule:

```text
top1_similarity - top2_similarity < margin_threshold
  -> AMBIGUOUS_LABEL
```

Reason codes du kien:

- `identity_known`
- `identity_unknown`
- `identity_low_confidence`
- `identity_ambiguous`
- `identity_enrolled`
- `identity_gallery_empty`
- `identity_encoder_missing`

## 11. UI Scope

### Inference Tab

Them:

- Label crop identity panel.
- Predicted identity class.
- Similarity score.
- Known/unknown badge.
- Top-k matches.
- Nut `Enroll as new label`.
- Nut `Mark prediction wrong`.

### Teach/Identity Tab

Them:

- Gallery selector.
- Create new identity class.
- Upload/enroll crop images.
- View class prototypes/exemplar count.
- Rebuild gallery.
- Export identity dataset.

### Quick Teach / Models

Them sau:

- Stage/apply identity encoder.
- Show encoder version.
- Show identity evaluation report.

## 12. Roadmap Trien Khai

### Phase I0 - Baseline Detection Ready

- Cho phep `label_only` chay khi chi co label model.
- Hien thi crop label ro rang.
- Test voi `weights/label_model.pt` khi model train xong.
- Luu crop label trong result/collection.

Definition of Done:

- Upload 1 anh, detect label, crop label, hien thi crop thanh cong.

### Phase I1 - Identity Data Foundation

- Them package `identity/`.
- Them schema identity.
- Them gallery store.
- Them embedding extraction baseline.
- Them tests cho gallery save/load va similarity matching.

Definition of Done:

- Co the enroll 1 class voi vai crop mau.
- Co the predict crop moi ra top-k match.

### Phase I2 - Open-set Inference MVP

- Gan identity inference vao pipeline sau label crop.
- Them `IdentityPrediction` vao `InspectionResult`.
- Them unknown threshold.
- Them pending unknown collection.

Definition of Done:

- Crop label known -> match class da enroll.
- Crop label la class moi -> tra `UNKNOWN_LABEL`.

### Phase I3 - Streamlit Identity UI

- UI hien identity result.
- UI enroll unknown crop thanh class moi.
- UI add exemplar vao class co san.
- UI rebuild gallery.

Definition of Done:

- Operator co the day class moi truc tiep tu crop inference.

### Phase I4 - Few-shot Evaluation

- Viet script evaluate k-shot.
- Dung seen/unseen split hien co.
- Bao cao threshold khuyen nghi.

Definition of Done:

- Co report cho 1-shot/3-shot/5-shot.
- Co unknown threshold ban dau.

### Phase I5 - Train Encoder Tren Colab/PC

- Chuyen logic Animal Few-shot thanh Label Identity training.
- Train metric encoder tren seen classes.
- Evaluate voi unseen classes.
- Export checkpoint cho app.

Definition of Done:

- Encoder rieng cho label identity tot hon pretrained baseline.
- Co checkpoint va evaluation report.

### Phase I6 - Edge Optimization

- Benchmark identity latency.
- Export ONNX/TensorRT neu can.
- Cache embeddings va prototypes.

Definition of Done:

- Identity inference them vao pipeline ma van dat latency chap nhan duoc.

## 13. Acceptance Criteria

Phase hien tai duoc xem la thanh cong khi:

1. Label detector detect duoc bbox va crop label on dinh.
2. He thong co identity gallery va prototype store.
3. He thong nhan dien duoc label da enroll bang similarity.
4. He thong khong ep label moi vao class cu, ma tra `UNKNOWN_LABEL`.
5. Operator co the gan ten class moi va enroll ngay trong UI.
6. Class moi co the duoc nhan lai sau khi enroll ma khong train lai model.
7. Tat ca sample identity co metadata de train/fine-tune tren PC/Colab.
8. Co evaluation script cho seen/unseen va k-shot.

## 14. Risk Register

| Rui ro | Tac dong | Giam thieu |
|---|---|---|
| Dataset imbalance manh | Encoder thien ve class nhieu anh | Episodic balanced sampler |
| Unseen class qua it mau | Threshold kho calibrate | Dung k-shot simulation va luu them data |
| Crop label bi lech/blur | Identity sai | Detection threshold + crop padding + preprocess |
| Similar label qua giong nhau | Ambiguous match | Top1-top2 margin rule |
| Pretrained feature chua du | Unknown/known tach kem | Train metric encoder rieng tren label crop |
| Edge CPU cham | Latency tang | Cache gallery, model nho, ONNX/TensorRT sau |
| Auto-enroll sai | Gallery bi nhiem | Bat buoc operator approve |

## 15. Nguyen Tac Coding Cho Phase Moi

- Khong pha pipeline hien co.
- Identity la optional module: neu chua co encoder/gallery, inference label detection van chay.
- Tat ca config moi co default an toan.
- Tests khong phu thuoc model weight nang.
- Khong overwrite production model/galleries neu chua co backup.
- Metadata phai du de audit: image, crop, bbox, model, encoder, threshold, prediction, operator action.

## 16. Ket Luan Checkpoint

Huong du an da duoc chot:

```text
Detection de thay label.
Few-shot/Open-set Identity de hieu label do la ai.
Teach/Enroll de mo rong class moi.
PC/Colab de train lai backbone khi data du.
Edge/Laptop de demo va van hanh thuc te.
```

Day la scope thuc chien va co tinh san pham cao: he thong khong chi detect duoc label, ma con co kha nang nhan ra label da biet, tu choi label la, va duoc day them class moi mot cach co kiem soat.
