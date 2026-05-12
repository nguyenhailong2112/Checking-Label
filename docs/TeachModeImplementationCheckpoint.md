# Teach Mode Implementation Checkpoint

Ngay cap nhat: 12/05/2026

Tai lieu nay la checklist ky thuat de bien project hien tai thanh ban demo Hybrid Edge AI co Teach Mode. File nay nen doc cung:

- `ScopeOfWorkVer2.md`
- `docs/ScopeVer2PlusTeachMode.md`
- `docs/ProjectAuditCheckpoint.md`

## 1. Hien trang repo

### Da co

- Streamlit UI trong `app.py`.
- Cascaded pipeline trong `LabelBarcodeInspector`.
- 3 model wrapper rieng: label, code, defect.
- `DecisionEngine` tach rieng va co unit tests.
- `InspectionResult`, `BoundingBox`, `DecodeResult`, `RuntimeSettings`.
- Active Learning collector luu NG, low-confidence va manual sample.
- Model registry stage/apply/rollback `.pt` va `.engine`.
- Benchmark CLI va TensorRT export helper.
- PC training workflow doc.

### Chua co

- Product recipe schema.
- Teach session/sample schema.
- UI ve bbox/ROI.
- YOLO label export tu bbox nguoi dung ve.
- Recipe-aware scoring.
- Auto-label suggestion/approval loop.
- Candidate model evaluation manifest.
- Realtime camera loop.
- Edge Micro Fine-tune safety layer.

## 2. Nguyen tac thiet ke

1. Khong pha pipeline hien tai.
2. Teach Mode la tang bo sung, khong thay the YOLO inference.
3. Moi du lieu operator approve phai co metadata.
4. Khong auto-train tu prediction chua duoc approve.
5. Model production khong bi overwrite truc tiep.
6. Moi model moi phai di qua staged artifact va rollback path.
7. Label/code/defect co logic teach rieng, khong ep dung chung mot rule.

## 3. Kien truc module de them

De xuat them cac module:

```text
src/edge_inspector/teach/
  __init__.py
  schemas.py
  recipe.py
  dataset.py
  scoring.py
  auto_label.py
```

### `schemas.py`

Chua:

- `ProductRecipe`
- `RecipeLabelSettings`
- `RecipeCodeSettings`
- `RecipeDefectSettings`
- `TeachSession`
- `TeachSample`
- `ApprovedAnnotation`
- `RecipeScore`

### `recipe.py`

Chua:

- Load/save recipe YAML/JSON.
- List recipes.
- Create/update/deactivate recipe.
- Validate recipe id va version.

### `dataset.py`

Chua:

- Convert bbox xyxy <-> YOLO normalized.
- Luu image + label txt.
- Export dataset cho tung target: label/code/defect.
- Luu negative samples.

### `scoring.py`

Chua:

- ROI overlap score.
- Aspect ratio score.
- Crop similarity score.
- Pattern validation score.
- Recipe confidence aggregator.

### `auto_label.py`

Chua:

- Dung prediction hien tai de suggest bbox.
- Rank bbox theo recipe score.
- Chuan bi payload de UI approve/sua.

## 4. Schema du kien

### ProductRecipe

```python
class ProductRecipe(BaseModel):
    recipe_id: str
    product_name: str
    version: int = 1
    active: bool = True
    label: RecipeLabelSettings
    code: RecipeCodeSettings
    defect: RecipeDefectSettings
    created_at: datetime
    updated_at: datetime | None = None
```

### TeachSample

```python
class TeachSample(BaseModel):
    sample_id: str
    recipe_id: str
    target: Literal["label", "code", "defect"]
    image_name: str
    image_path: str
    annotations: list[ApprovedAnnotation]
    negative: bool = False
    source: Literal["upload", "camera", "auto_collect"]
    approved_by: str | None = None
    model_snapshot: dict[str, str] = {}
    created_at: datetime
```

### RecipeScore

```python
class RecipeScore(BaseModel):
    target: Literal["label", "code", "defect"]
    model_confidence: float
    roi_score: float | None = None
    similarity_score: float | None = None
    pattern_score: float | None = None
    final_score: float
    reason_codes: list[str]
```

## 5. Work Package 1 - Data foundation

Muc tieu: co nen tang recipe + teach sample truoc khi lam UI phuc tap.

Viec can lam:

- Tao `src/edge_inspector/teach/`.
- Them Pydantic schemas.
- Them recipe load/save.
- Them YOLO bbox conversion.
- Them tests:
  - recipe serialize/deserialize.
  - bbox xyxy -> YOLO -> xyxy.
  - invalid bbox bi reject.
  - target label/code/defect dung class map.

Definition of Done:

- Co the tao recipe bang code.
- Co the luu 1 teach sample thanh image + `.txt` YOLO label.
- Test pass khong can model weights.

## 6. Work Package 2 - Teach Mode UI MVP

Muc tieu: operator ve/nhap bbox va tao dataset duoc.

Streamlit limitation:

- Streamlit core khong co canvas bbox native.
- Co the dung `streamlit-drawable-canvas` neu chap nhan them dependency.
- Neu muon tranh dependency truoc, MVP co the cho nhap bbox bang numeric inputs tren anh preview.

De xuat:

Phase UI 1:

- Upload anh.
- Hien anh.
- Nhap bbox xyxy bang sliders/number inputs.
- Chon target: label/code/defect.
- Chon class.
- Save approved sample.

Phase UI 2:

- Them canvas ve bbox.
- Them move/resize bbox.
- Them multiple boxes.

Definition of Done:

- Tao session teach.
- Luu it nhat 1 annotation label/code/defect.
- Export dataset folder co images/labels/dataset.yaml.

## 7. Work Package 3 - Label Teach

Muc tieu: label model co kha nang thich nghi voi product moi bang recipe.

Can lam:

- Recipe label ROI.
- Label class whitelist theo recipe.
- Crop label mau.
- Crop similarity basic:
  - resize crop ve kich thuoc chuan.
  - histogram similarity hoac ORB/template matching basic.
  - ban dau chi can deterministic, nhanh, khong can deep embedding.
- Recipe score cho label:
  - model conf.
  - ROI overlap.
  - aspect ratio.
  - similarity.

Decision change:

- Them note `recipe_assisted_label` khi model conf thap nhung recipe score dat.
- Them note `recipe_roi_mismatch` khi bbox nam sai ROI.

Tests:

- Label bbox trong ROI score cao.
- Label bbox ngoai ROI score thap.
- Aspect ratio qua lech bi giam score.

## 8. Work Package 4 - Code Teach

Muc tieu: code 1D/2D duoc kiem tra chinh xac hon theo recipe.

Can lam:

- Code ROI theo label crop.
- Support expected code count.
- Decode tung code box, khong chi best box khi recipe yeu cau nhieu code.
- Pattern validation:
  - regex optional.
  - allowed code types.
  - min/max length.
- Reason codes:
  - `code_missing`
  - `decode_failed`
  - `pattern_failed`
  - `code_count_mismatch`
  - `code_recipe_ok`

Tests:

- Decode success + pattern dung -> OK.
- Decode success nhung pattern sai -> NG/warning.
- Expected 2 codes nhung detect 1 -> NG.

## 9. Work Package 5 - Defect Teach

Muc tieu: defect check on dinh hon, giam false NG/false OK.

Can lam:

- Defect ROI rieng.
- Defect threshold rieng, vi defect khong nen dung chung threshold voi label/code.
- Positive sample: defect bbox.
- Negative sample: crop OK da duoc operator xac nhan.
- Defect decision:
  - any defect over threshold -> NG.
  - defect trong ROI nguy hiem hon defect ngoai ROI.
  - low defect conf co the warning neu recipe yeu cau.

Tests:

- Defect in ROI -> NG.
- Defect below threshold -> OK/warning tuy config.
- Negative sample export dung folder.

## 10. Work Package 6 - Recipe-aware Inspector

Muc tieu: pipeline runtime nhan recipe va tra output giau thong tin hon.

Can lam:

- Config them `recipes.active_recipe_id`.
- `LabelBarcodeInspector.run(..., recipe=None)` hoac inspector load recipe tu config.
- `InspectionResult` them optional fields:
  - `recipe_id`
  - `model_confidence`
  - `recipe_confidence`
  - `reason_codes`
- DecisionEngine nhan recipe scores.

Can can than:

- Giu backward compatibility voi tests hien tai.
- Neu khong co recipe, pipeline chay y nhu hien tai.

Definition of Done:

- Inference khong recipe khong thay doi behavior.
- Inference co recipe co them notes/scores.

## 11. Work Package 7 - Auto-label assist

Muc tieu: tang toc annotate nhu Box Prompting nhung co human approval.

Can lam:

- Lay predictions tu 3 model.
- Rank predictions theo recipe score.
- De xuat bbox cho target dang teach.
- UI cho approve/sua.
- Luu `suggested_by_model=true` trong metadata.

Definition of Done:

- Operator co the upload 10 anh, he thong goi y bbox, operator approve nhanh thanh dataset.

## 12. Work Package 8 - Edge Micro Fine-tune Optional

Muc tieu: co demo train nhe tai bien nhung an toan.

Can lam:

- Advanced flag trong config: `training.enable_edge_micro_finetune`.
- Dataset min check:
  - label: >= 10 approved images recommended.
  - code: >= 10 approved images recommended.
  - defect: can both positive and negative samples.
- Train candidate output vao `runs/edge_micro_finetune`.
- Benchmark candidate tren:
  - teach validation set.
  - golden sanity set neu co.
- Stage candidate vao ModelRegistry.

Khong lam:

- Khong auto-apply candidate neu chua co metric.
- Khong train khi camera realtime dang chay.

## 13. Output/Metadata can mo rong

`InspectionResult` nen co them trong ban Plus:

```json
{
  "recipe_id": "sandisk_ver2_default",
  "reason_codes": ["recipe_assisted_label", "decode_success", "no_defect"],
  "scores": {
    "label": {
      "model_confidence": 0.61,
      "recipe_confidence": 0.86,
      "final_score": 0.78
    },
    "code": {
      "model_confidence": 0.74,
      "pattern_score": 1.0,
      "final_score": 0.87
    },
    "defect": {
      "model_confidence": 0.03,
      "final_score": 0.97
    }
  }
}
```

## 14. Thu tu uu tien de code

Uu tien 1:

- `teach/schemas.py`
- `teach/dataset.py`
- tests cho bbox/dataset.

Uu tien 2:

- Recipe load/save.
- Teach sample save.
- UI numeric bbox MVP.

Uu tien 3:

- Recipe-aware label scoring.
- Inspector optional recipe.

Uu tien 4:

- Code recipe/pattern validation.
- Multi-code decode.

Uu tien 5:

- Defect ROI/threshold rieng.

Uu tien 6:

- Auto-label assist.

Uu tien 7:

- Edge Micro Fine-tune candidate/promotion.

## 15. Risk register

| Rui ro | Tac dong | Giam thieu |
|---|---|---|
| Overfit voi 1-20 anh | Model moi fail case cu | Recipe adaptation truoc, micro-train sau, benchmark truoc promote |
| Auto-approve sai | Dataset bi nhiem loi | Bat buoc human approve, luu negative sample |
| Jetson train cham/nong | Demo mat on dinh | Maintenance mode, fallback Colab/PC |
| Streamlit bbox UI kho | Cham MVP | Ban dau dung numeric bbox, sau do them canvas |
| Defect sample qua it | False OK/NG cao | Can negative samples va threshold rieng |
| Code decode phu thuoc zbar | Decode fail runtime | Check dependency, fallback warning, pattern validation |

## 16. Demo story de thuyet phuc

1. Chay baseline inference voi 3 model.
2. Chon anh san pham moi, model detect label conf chi khoang 0.55-0.65.
3. Vao Teach Mode, tao recipe, ve bbox label va code ROI tren 5-10 anh.
4. Quay lai Inference, bat recipe.
5. He thong cho decision on dinh hon, giai thich `recipe_assisted_label`.
6. Code decode duoc validate theo pattern.
7. Defect ROI duoc kiem tra rieng.
8. Export teach dataset de train tiep tren Colab.
9. Neu can, demo Micro Fine-tune candidate va rollback.

Day la cau chuyen san pham ro rang: Edge khong chi "doan", ma co kha nang duoc day, ghi nho recipe, giai thich quyet dinh va tich luy dataset dung chuan.

---

## 17. Implementation Update - MVP Started

Cap nhat sau dot code dau tien:

- Da them package `src/edge_inspector/teach/`.
- Da co `ProductRecipe`, `TeachSample`, `ApprovedAnnotation`.
- Da co recipe YAML store.
- Da co YOLO dataset writer tu bbox operator approve.
- Da co bbox conversion va scoring helper co ban.
- Da co tab `Teach Mode` trong Streamlit.
- Da co active recipe control trong sidebar.
- Da co recipe metadata trong `InspectionResult`: `recipe_id`, `model_confidence`, `recipe_confidence`, `reason_codes`, `scores`.
- Da co recipe-aware scoring co ban cho label/code/defect.
- Da co tests/smoke tests cho schema, dataset va scoring.

Nhung phan con lai sau MVP:

1. Canvas ve bbox truc quan thay cho numeric inputs.
2. Auto-label suggestion dua tren current YOLO predictions.
3. Multi-code decode day du hon.
4. Defect negative sample review flow tot hon.
5. Candidate model evaluation manifest.
6. Edge Micro Fine-tune optional.
