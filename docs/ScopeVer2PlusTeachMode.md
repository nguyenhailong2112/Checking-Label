# Scope Ver 2 Plus - Hybrid Edge AI With Teach Mode

Ngay cap nhat: 12/05/2026

Tai lieu nay mo rong `ScopeOfWorkVer2.md` theo huong da thong nhat: giu kien truc Hybrid thuc te, nhung bo sung mot lop **Teach Mode / Recipe Adaptation** de demo co cam giac "hoc tai bien" ma van an toan, nhe va phu hop voi Jetson/Edge.

## 1. Dinh vi chien luoc

Scope Ver 2 goc da dung khi tach Edge va PC:

- Edge uu tien inference on-device, visualization, JSON output, Active Learning va model staging.
- PC/Colab/Workstation phu trach train/fine-tune nang, evaluate va export model.

Ban Plus nay khong phu dinh Scope Ver 2. Ban Plus bo sung mot tang trung gian:

```text
Pretrained YOLO models
        |
        v
Runtime inference
        |
        v
Teach Mode / Product Recipe / Local Adaptation
        |
        v
Decision Engine + Active Learning + Optional Micro Fine-tune
```

Muc tieu la cho operator/engineer co the day he thong 1-20 anh mau cua san pham moi ngay tai Edge, ve bbox/ROI, approve prediction, va thay ket qua detect on-site tot hon ma khong bat buoc train lai YOLO ngay lap tuc.

## 2. Ket qua toi thuong can dat

He thong can du suc demo thuyet phuc cho bai toan label, barcode 1D/2D va defect tren pham vi du lieu co dinh:

1. Edge chay pipeline label -> crop -> code/decode -> defect -> OK/NG.
2. Operator co the tao "recipe" cho san pham moi bang cach ve bbox/ROI tren 1-20 anh.
3. He thong dung recipe de tang do tin cay cua label/code/defect trong runtime.
4. Anh duoc operator approve se duoc luu thanh ground-truth co cau truc, dung duoc cho PC/Colab fine-tune.
5. Neu can, Edge co the co che do Micro Fine-tune rieng trong maintenance mode, co benchmark va rollback.
6. Tat ca thay doi quan trong deu co manifest, version, metric va rollback path.

## 3. Pham vi moi cua Teach Mode

### 3.1 In Scope

- UI ve bbox/ROI tren anh upload hoac camera snapshot.
- Tao product recipe cho tung product/label version.
- Luu teach samples duoi dang YOLO annotation dung format.
- Auto-label goi y bbox cho anh tiep theo dua tren current YOLO + ROI + similarity.
- Human approval loop: approve/sua/xoa bbox truoc khi dua vao dataset.
- Recipe-aware decision: ket hop model confidence voi recipe confidence.
- Active Learning mo rong: NG, low-confidence, manual va teach-approved.
- Optional Edge Micro Fine-tune trong maintenance mode, khong chay song song voi production inference.
- Promote candidate model qua ModelRegistry, co rollback.

### 3.2 Out Of Scope trong giai do gan

- Training full YOLO lien tuc trong production runtime.
- Tu dong auto-approve vo han khong co human check.
- Multi-camera industrial hoan chinh.
- PLC closed-loop reject control.
- Cloud dataset sync tu dong.
- Annotation tool day du nhu Label Studio/Roboflow tich hop trong Streamlit.

## 4. Ba tang thich nghi tai bien

### Tang A - Recipe Adaptation, uu tien lam truoc

Day la tang nhe nhat va gia tri demo cao nhat.

Operator teach:

- Product name: vi du `SanDiskVer2`, `WD_Label_A`, `Label`.
- Label ROI hoac bbox chuan.
- Code ROI neu barcode/QR co vi tri on dinh.
- Defect inspection ROI neu chi can soi vung nhat dinh.
- Expected code pattern neu co: prefix, length, regex, allowed code type.
- Threshold rieng: label accept, code accept, defect reject.

Runtime dung:

- YOLO bbox proposal.
- IoU/overlap voi ROI recipe.
- Kich thuoc/ti le bbox so voi bbox mau.
- Similarity giua crop hien tai va crop mau.
- Decode success va pattern validation.
- Defect confidence trong vung can kiem.

Ket qua:

- Neu YOLO conf chi 0.60 nhung recipe score rat cao, he thong co the accept voi note `accepted_by_recipe`.
- Neu YOLO conf cao nhung bbox nam sai ROI hoac code pattern sai, he thong co the chuyen sang warning/NG.

### Tang B - Auto Label And Approve

Tang nay giong tinh than Box Prompting:

1. Operator ve bbox dau tien.
2. He thong goi y bbox tren anh tiep theo bang current model + recipe.
3. Operator approve/sua.
4. Moi bbox approve tro thanh ground-truth.
5. Dataset duoc luu ngay tai Edge de copy ve Colab/PC train.

Nguyen tac an toan:

- Khong coi prediction la ground-truth neu chua approve.
- Can luu negative samples khi he thong de xuat sai.
- Can audit trail: ai approve, luc nao, recipe nao, model nao.

### Tang C - Edge Micro Fine-tune, lam sau

Day la che do nang cao:

- Chi chay khi operator vao `Maintenance / Retrain Mode`.
- Chi train candidate model, khong overwrite production model.
- Uu tien label model truoc, sau do code model, cuoi cung defect model.
- Epoch it, image size nho/vua, batch nho, early stop.
- Chay sanity benchmark tren golden set + teach set.
- Chi promote neu metric dat nguong.

Neu Jetson nong, cham hoac loi dependency, workflow quay ve PC/Colab train. Day la fallback mac dinh.

## 5. Tac dong len 3 model

### 5.1 Label Model

Label model la trung tam cua Teach Mode.

Can bo sung:

- Product recipe cho label class/version.
- ROI/bbox teach UI.
- Similarity scoring tren label crop.
- Product whitelist: model chi accept class hop le voi recipe hien tai.
- Low-confidence recovery: neu detect label conf thap nhung recipe score cao, danh dau `OK_RECIPE_ASSISTED` hoac note tuong duong trong metadata.

Du lieu can luu:

- Full image.
- Label bbox.
- Label crop.
- Product class name.
- Model prediction cu.
- Operator bbox approved.
- Recipe id/version.

### 5.2 Code Model And Decode

Code model can thich nghi theo vi tri va pattern hon la train lien tuc.

Can bo sung:

- Code ROI gan voi label crop.
- Support multi-code neu mot label co nhieu barcode/QR.
- Decode tung code crop, khong chi code box tot nhat neu recipe yeu cau nhieu code.
- Expected pattern: regex, min/max length, code type.
- Recipe rule: `require_code_count`, `require_decode_all`, `allow_code_types`.

Du lieu can luu:

- Code bbox.
- Code crop.
- Decode text/type.
- Decode success/fail.
- Pattern validation result.
- Operator correction neu decode/prediction sai.

### 5.3 Defect Model

Defect model rat de overfit neu train voi it anh, nen can an toan hon.

Can bo sung:

- Defect ROI trong label crop.
- Threshold rieng cho defect, khong dung chung label/code threshold.
- Defect severity: warning/ng neu can.
- Negative samples OK that nhieu hon positive NG.
- Human confirm cho defect prediction truoc khi dua vao train set.

Du lieu can luu:

- Defect bbox neu co.
- Defect-free crop neu operator xac nhan OK.
- Defect type hien tai: `DefectNG`.
- Lighting/camera notes neu co.
- Operator confirmation.

## 6. Product Recipe du kien

Mot recipe nen la file JSON/YAML co version:

```yaml
recipe_id: sandisk_ver2_default
product_name: SanDiskVer2
created_at: "2026-05-12T00:00:00Z"
active: true
models:
  label_model: weights/label_model.pt
  code_model: weights/code_model.pt
  defect_model: weights/defect_model.pt
label:
  classes: ["SanDiskVer2", "Label"]
  expected_roi_xyxy: [100, 80, 900, 620]
  min_model_conf: 0.25
  min_recipe_score: 0.70
  min_accept_score: 0.75
code:
  enabled: true
  expected_count: 1
  allowed_types: ["QRCODE", "CODE128", "EAN13"]
  pattern: null
  require_decode: true
defect:
  enabled: true
  roi_xyxy: null
  reject_threshold: 0.35
teach:
  approved_samples: 0
  negative_samples: 0
  last_updated: null
```

## 7. Runtime confidence moi

Hien tai `total_confidence` chu yeu den tu model confidence. Ban Plus can tach ro:

- `model_confidence`: diem tu YOLO/decode/defect.
- `recipe_confidence`: diem tu ROI, similarity, size/aspect, pattern.
- `decision_confidence`: diem cuoi cung dung cho OK/NG.

Cong thuc ban dau co the don gian:

```text
label_recipe_score = weighted_mean(
  model_conf,
  roi_overlap,
  crop_similarity,
  aspect_ratio_score
)

code_recipe_score = weighted_mean(
  code_model_conf,
  roi_overlap,
  decode_success,
  pattern_score
)

defect_risk_score = max(defect_model_conf, recipe_defect_roi_risk)
```

Khong can lam qua phuc tap ban dau. Quan trong la co log ro tai sao OK/NG.

## 8. Dataset structure moi

Them folder rieng cho Teach Mode:

```text
data/teach/
  recipes/
    sandisk_ver2_default.yaml
  sessions/
    20260512_001/
      images/
      labels/
      crops/
      metadata/
      negatives/
  exports/
    label_dataset/
    code_dataset/
    defect_dataset/
```

YOLO labels phai dung format that:

```text
class_id x_center y_center width height
```

Khong con dung full-image bbox cho training nghiem tuc, tru khi do la prototype co chu dich.

## 9. UI moi can co

### Tab 1 - Inference

Giu nhu hien tai, nhung them:

- Recipe selector.
- Recipe-aware decision notes.
- Model confidence vs recipe confidence.
- Reason codes: `model_ok`, `recipe_assisted`, `pattern_failed`, `roi_mismatch`, `defect_found`.

### Tab 2 - Camera

Nang tu snapshot len:

- Start/stop.
- Capture frame.
- Inspect frame.
- FPS/latency.
- Later: realtime loop.

### Tab 3 - Teach Mode

Chuc nang moi:

- Create/select recipe.
- Upload/capture teach images.
- Ve bbox/ROI cho label/code/defect.
- Auto-suggest bbox.
- Approve/sua/xoa prediction.
- Export teach dataset.
- Mark negative sample.

### Tab 4 - Models

Tach khoi Quick Teach cu:

- Stage/apply/rollback model artifact.
- Xem model version.
- Xem evaluation summary.
- Promote candidate sau Micro Fine-tune.

### Tab 5 - Micro Fine-tune, optional

Chi hien khi enable advanced mode:

- Chon target: label/code/defect.
- Chon recipe/session.
- Train candidate.
- Benchmark candidate.
- Promote/rollback.

## 10. Roadmap trien khai

### Phase P0 - Chot weights va demo baseline

- Copy 3 model `.pt` vao `weights/`.
- Chay inference upload anh.
- Chay benchmark local.
- Ghi baseline metric.

### Phase P1 - Teach data foundation

- Them schema `ProductRecipe`, `TeachSample`, `TeachSession`.
- Them folder `data/teach`.
- Them helper luu bbox YOLO annotation.
- Them unit tests cho bbox conversion va recipe load/save.

### Phase P2 - Teach UI MVP

- Them tab Teach Mode.
- Upload/capture anh.
- Ve bbox/ROI co ban.
- Luu sample approved.
- Export dataset theo label/code/defect.

### Phase P3 - Recipe-aware decision

- Them recipe selector trong Inference.
- Tinh ROI overlap va crop similarity.
- Ket hop recipe score vao decision.
- Them notes va JSON output.

### Phase P4 - Auto label assist

- Dung current YOLO de suggest bbox.
- Dung recipe ROI/similarity de rank prediction.
- Operator approve/sua.
- Luu negative examples.

### Phase P5 - Edge Micro Fine-tune optional

- Train candidate model rieng.
- Sanity benchmark.
- Stage candidate qua ModelRegistry.
- Promote/rollback co dieu kien.

## 11. Tieu chi thanh cong cua ban Plus

- Operator co the teach label moi bang 1-20 anh.
- Sau teach, cung anh/cung product cho decision on dinh hon so voi YOLO conf don thuan.
- Code 1D/2D duoc kiem tra theo bbox + decode + pattern.
- Defect duoc kiem tra theo ROI/threshold rieng.
- Moi sample approved deu thanh dataset co the train lai tren Colab/PC.
- Khong co model production nao bi overwrite truc tiep.
- Moi candidate model co metric va rollback.

## 12. Ket luan

Scope Ver 2 Plus la diem can bang giua mo mong va thuc chien:

- Van la Edge AI demo that.
- Van chay duoc trong pham vi tai nguyen vua phai.
- Khong bien Jetson thanh training server lon.
- Nhung co kha nang teach/adapt tai bien du suc gay an tuong.

Huong nay gan voi tinh than industrial vision: recipe, ROI, teach object, verify, rollback. Diem moi la chung ta dat YOLO/AI model ben duoi recipe layer, de he thong vua linh hoat nhu smart camera, vua co kha nang hoc tiep qua Active Learning va fine-tune co kiem soat.
