Edge AI Label & Barcode Inspection System – Hybrid Edition
Phiên bản: 2.0
Ngày lập: 11 tháng 05 năm 2026
Người thực hiện: Nguyễn Hải Long - R&D Vision Engineer
Trạng thái: Đã thống nhất & Phiên bản thực tế khả thi
1. Giới thiệu & Bối cảnh Dự án
Dự án nhằm phát triển một hệ thống Edge AI chuyên dụng cho việc kiểm tra nhãn mác (label inspection) đọc mã vạch/QR (barcode/2D code reading) và phát hiện lỗi ngoại quan (defect detection) trên các linh kiện thùng hàng sản phẩm.
Học hỏi từ các hãng lớn (Cognex Datalogic Zebra Honeywell Keyence) chúng ta nhận thấy:

Inference nên được thực hiện hoàn toàn tại biên (Edge).
Training & Fine-tuning mạnh thường vẫn cần sự hỗ trợ của PC hoặc server.
Do đó chúng ta chọn kiến trúc Hybrid – cân bằng giữa tính thực tế ổn định và khả năng mở rộng.

Mục tiêu là xây dựng một hệ thống thực chiến ổn định dễ sử dụng có giá trị demo cao và làm nền tảng vững chắc để phát triển tiếp trong tương lai.
2. Mục tiêu Dự án
Mục tiêu chính:

Xây dựng Edge Device (Jetson AGX Orin 32GB) chạy Inference full pipeline ổn định tốc độ cao.
Hỗ trợ Active Learning (thu thập data NG tự động) để cải tiến model liên tục.
Xây dựng quy trình Quick Teach / Fine-tune thực tế qua PC.
Đạt độ chính xác và tốc độ đủ để thuyết phục nội bộ / khách hàng.
Tạo nền tảng dễ mở rộng dễ bàn giao và dễ bảo trì.

Mục tiêu đo lường (Success Criteria):

Tốc độ inference ≥ 20 FPS trên Jetson AGX Orin.
Độ chính xác tổng thể ≥ 95% trên tập test thực tế (có thể điều chỉnh theo data).
Hệ thống chạy ổn định xử lý được các trường hợp NG và Low-confidence.
Có quy trình rõ ràng từ thu thập data → fine-tune → deploy model mới.
Streamlit UI thân thiện trực quan dễ sử dụng bởi operator và engineer.

3. Phạm vi Dự án (Scope)
3.1. In Scope
A. Edge Device (Jetson AGX Orin 32GB)

Full Cascaded Inference Pipeline:
Label Detection
Auto Crop + Preprocess (enhance deskew contrast…)
Code Detection (1D & 2D)
Barcode Decoding (pyzbar)
Defect Detection (single class DefectNG)

Visualization (ảnh gốc + vùng crop với bounding box)
Logging & lưu kết quả (JSON + ảnh)
Active Learning: Nút “Collect for Training” tự động lưu ảnh NG/Low-conf + metadata
Quick Teach cơ bản: Điều chỉnh threshold apply model mới (.pt hoặc TensorRT engine)
Web UI (Streamlit) chạy trên Edge

B. Training Workstation (PC/Laptop)

Hỗ trợ annotate data mới (hướng dẫn dùng Label Studio)
Fine-tune model YOLOv11 với data mới
Export TensorRT engine tối ưu cho Jetson
So sánh performance model cũ/mới

C. Tích hợp & Output

JSON output chuẩn (InspectionResult)
Lưu ảnh & log NG
Hỗ trợ input từ USB camera / webcam (sau này mở rộng GigE/HIK)

3.2. Out of Scope (Giai đoạn này)

Fine-tune / Training trực tiếp trên Jetson AGX Orin
Kết nối PLC đầy đủ (Modbus Ethernet/IP OPC UA) – sẽ làm ở phase sau
Multi-camera đồng thời
Fine-tune LoRA / few-shot trực tiếp trên Edge
Hệ thống cloud sync tự động
Annotation tool tích hợp hoàn chỉnh trong Streamlit

4. Kiến trúc Hệ thống

Hybrid Architecture: Edge chuyên Inference + Data Collection | PC chuyên Training & Fine-tune.
Pipeline: Cascaded (Label → Crop → Code + Defect + Decode).
Model Management: 3 models riêng (Label Code Defect) – dễ thay thế độc lập.
Deployment: Docker / systemd service trên Jetson.

5. Yêu cầu Chi tiết Pipeline

Input: Ảnh từ camera (BGR format)
Label Detection: Trả về top-1 box cao nhất hoặc NG nếu không detect
Crop & Preprocess: Crop tight + enhance (CLAHE sharpen brightness deskew)
Code Inspection: Detect → Decode (pyzbar)
Defect Inspection: Detect DefectNG trên vùng crop
Decision Engine: Kết hợp các stage theo mode (có thể chọn từng phần)
Output: InspectionResult (Pydantic) visualization JSON + ảnh lưu

6. Công nghệ Stack

Backend: Python 3.10+ Ultralytics YOLOv11 OpenCV pyzbar
UI: Streamlit
Model Format: .pt (development) + TensorRT .engine (production trên Jetson)
Config: YAML + Pydantic
Logging: Python logging + file handler
Data Collection: Folder structure có metadata

7. Roadmap & Các Phase Triển khai
Phase 0: Preparation (Đang làm)

Train & export 3 models
Audit & hoàn thiện code core

Phase 1: Core Pipeline & UI (Tuần 1-2)

Hoàn thiện LabelBarcodeInspector
Xây dựng Streamlit UI đầy đủ
Implement Collect Data NG
Selective Inspection Mode

Phase 2: Optimization & Edge Deployment (Tuần 3)

Export TensorRT
Deploy lên Jetson AGX Orin
Benchmark FPS memory stability
Camera integration

Phase 3: Quick Teach & Active Learning (Tuần 4)

Data collection system
Quy trình fine-tune trên PC
Model version management

Phase 4: Polish & Delivery (Tuần 5-6)

UI/UX cải tiến
Documentation & User Manual
Test thực tế & Demo nội bộ
Handover package

8. Rủi ro & Giải pháp

Rủi ro: Model generalization chưa tốt → Giải pháp: Thu thập data thực tế mạnh + augmentation
Rủi ro: Tốc độ chậm trên Edge → Giải pháp: Quantization INT8 + TensorRT + tối ưu preprocess
Rủi ro: Fine-tune phức tạp → Giải pháp: Giữ hybrid dùng PC train
Rủi ro: Hệ thống không ổn định → Giải pháp: Error handling mạnh + logging chi tiết

9. Tiêu chí Nghiệm thu & Delivery

Code sạch modular có comment tốt
Pipeline chạy ổn định trên Jetson
UI demo đẹp và dễ sử dụng
Có quy trình thu thập data → fine-tune → deploy rõ ràng
Documentation đầy đủ
Demo thành công với data thực tế

Dưới đây là so sánh toàn diện giữa Scope of Work Ver 1 và Ver 2 lý do thay đổi và giải pháp mình đã lựa chọn.
1. Tóm tắt sự khác biệt lớn nhất

Ver 1 (phiên bản trước): Mang tính tham vọng cao mong muốn làm một hệ thống All-in-One trên Edge Device (Inference + Fine-tune trực tiếp trên Jetson qua Streamlit).
Ver 2 (phiên bản hiện tại): Chuyển sang kiến trúc Hybrid thực tế tập trung mạnh vào Inference ổn định trên Edge còn phần Training/Fine-tune giao cho PC.

2. Bảng so sánh chi tiết

| Hạng mục | Scope Ver 1 (Cũ)                       | Scope Ver 2 (Mới)                                                | Lý do thay đổi                     |
|---|----------------------------------------|------------------------------------------------------------------|------------------------------------|
| Kiến trúc tổng thể | All-in-One trên Edge                   | Hybrid (Edge + PC)                                               | Khả thi và ổn định hơn             |
| Inference | Full pipeline trên Edge                | Giữ nguyên, ưu tiên mạnh hơn                                     | Không đổi                          |
| Fine-tune / Training | Cố gắng làm trực tiếp trên Edge qua UI | Chỉ làm trên PC, Edge chỉ thu thập data                          | Edge khó train ổn định             |
| Fine-tune trên Edge | Có (mong muốn)                         | Không (chỉ Quick Teach nhẹ)                                      | Tránh rủi ro crash, nóng máy, chậm |
| Active Learning | Có                                     | Mạnh hơn, tập trung thu thập data NG                             | Cốt lõi của Hybrid                 |
| Quick Teach | Fine-tune full qua Streamlit           | Threshold tuning + apply model mới + Collect data                | Thực tế hơn                        |
| Annotation | Tích hợp trong UI                      | Dùng Label Studio/Roboflow trên PC                               | Annotation bbox trên Edge khó      |
| Rủi ro kỹ thuật | Cao                                    | Thấp hơn nhiều                                                   | Tăng độ ổn định                    |
| Thời gian demo | Lâu hơn (vì phức tạp)                  | Nhanh hơn                                                        | Ưu tiên demo sớm                   |
| Khả năng bàn giao | Khó (phụ thuộc Edge mạnh)              | Dễ hơn (Edge chỉ inference)                                      | Dễ sử dụng cho operator                   |
| Tầm nhìn dài hạn | Rất lớn, cao (gần commercial)          | Thực tế hơn nhưng bền vững, vẫn cao nhưng đi từng bước vững chắc | Bền vững hơn             |

3. Lý do chúng ta chuyển sang Scope Ver 2
Sau khi trao đổi sâu và bạn cung cấp code thực tế, mình nhận ra một số vấn đề quan trọng:

Hạn chế phần cứng thực tế của Jetson AGX Orin:
Train YOLO full (object detection) cần VRAM lớn, thời gian dài, và làm máy nóng → dễ throttle hoặc crash.
Fine-tune trực tiếp trên Edge không ổn định cho production.

Fine-tune Object Detection rất phức tạp:
Không chỉ gán class mà cần vẽ bounding box → rất khó làm mượt trên Streamlit + Edge.
Dễ dẫn đến model kém chất lượng nếu annotate vội.

Bài học từ các hãng lớn:
Họ cũng tách biệt rõ: Edge chỉ inference mạnh, Training station riêng.

Tầm nhìn thực tế của bạn:
Bạn muốn có demo thuyết phục sớm, có sản phẩm chạy ổn định trên Edge trước.
Muốn giảm rủi ro, tránh “mơ mộng lớn lao” dẫn đến dự án kéo dài hoặc thất bại.

Tài nguyên hiện tại:
Data của bạn đang tốt cho một “vùng bài toán nhỏ” → nên tập trung làm Inference cực mạnh trước, sau đó mới mở rộng.


4. Giải pháp Hybrid trong Ver 2 được thiết kế như thế nào?
Trên Edge Device (Jetson AGX Orin):

Chạy Inference siêu nhanh và ổn định (đây là điểm mạnh nhất của Edge AI).
Tự động phát hiện và lưu ảnh NG / Low-confidence vào folder riêng (Active Learning).
Có nút “Collect for Training” để operator bấm khi gặp trường hợp khó.
Hỗ trợ Quick Teach nhẹ: thay đổi threshold, load model mới (copy file từ USB).

Trên PC/Laptop:

Operator copy folder data thu thập từ Jetson về.
Annotate bằng Label Studio (offline).
Fine-tune model trên PC (mạnh hơn rất nhiều).
Export TensorRT engine → copy lại vào Jetson.

Quy trình vận hành (rất đơn giản):

Chạy Edge Device → Inference bình thường.
Gặp NG hoặc nghi ngờ → bấm Collect.
Định kỳ (1-2 ngày/lần) copy data về PC → train → deploy model mới.
Model mới thay thế ngay lập tức trên Edge.

Lợi ích lớn của cách này:

Edge Device chạy ổn định, nhanh, ít lỗi.
Dễ demo và bàn giao.
Vẫn có vòng lặp cải tiến model liên tục.
Giảm nguy cơ dự án bị kẹt ở phần Fine-tune.
Dễ scale lên sau này (thêm LoRA, TAO Toolkit…).


Kết luận:
Scope Ver 2 là phiên bản thực tế, chín chắn và có khả năng thành công cao hơn rất nhiều so với Ver 1. Nó không phải là giảm ambition hoàn toàn, mà là điều chỉnh chiến lược để chúng ta có thể tạo ra một sản phẩm chất lượng cao, ổn định và thuyết phục trong thời gian hợp lý.