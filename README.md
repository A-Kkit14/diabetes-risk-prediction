# 🩺 Diabetes Risk Prediction \& Health Indicators Dashboard

Đồ án cuối kỳ môn **Khai phá Dữ liệu (Data Mining)** — Đại học Kinh tế TP. Hồ Chí Minh (UEH)

## 📌 Giới thiệu bài toán

Bệnh tiểu đường (đái tháo đường) là một trong những thách thức y tế toàn cầu lớn nhất, đặc biệt nguy hiểm vì nhóm "tiền tiểu đường" thường không có triệu chứng rõ ràng, dẫn đến bỏ lỡ thời điểm can thiệp sớm.

Dự án này khai thác bộ dữ liệu khảo sát sức khỏe cộng đồng **BRFSS 2015** (Behavioral Risk Factor Surveillance System – CDC Hoa Kỳ) nhằm:

1. **Sàng lọc sớm nguy cơ mắc bệnh** dựa trên các chỉ số sức khỏe phi xâm lấn (không cần xét nghiệm y tế)
2. **Định lượng \& xếp hạng mức độ ảnh hưởng** của từng yếu tố nguy cơ (BMI, huyết áp, tuổi tác, lối sống...) lên khả năng mắc bệnh
3. **Trực quan hóa insight** qua một dashboard tương tác, giúp người dùng không chuyên vẫn hiểu được dữ liệu

**Dataset:** [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) (Kaggle) — 253,680 quan sát, 21 biến đầu vào, biến mục tiêu `Diabetes\\\_binary` (nhị phân).

## 🧠 Công việc đã thực hiện

### 1\. Tiền xử lý dữ liệu (preprocess-for-data-mining-2.ipynb)

* Kiểm tra chất lượng dữ liệu: missing values, dữ liệu trùng lặp, outlier (IQR) trên các biến liên tục (BMI, MentHlth, PhysHlth)
* Kiểm tra tỉ lệ mất cân bằng của biến mục tiêu (86.1% No Diabetes vs 13.9% Diabetes)
* Tự động phân nhóm biến (nhị phân / liên tục / phân loại) và ép kiểu dữ liệu phù hợp để tối ưu bộ nhớ và chuẩn hóa cho bước phân tích tiếp theo

### 2\. Phân tích khám phá dữ liệu & tương quan (EDA_Diabetes_1ipynb)

* Phân tích phân phối cho từng nhóm biến: liên tục (histogram, KDE, boxplot), nhị phân (bar/pie chart), thứ bậc (GenHlth, Education, Income, AgeGroup)
* Áp dụng đúng phương pháp thống kê tương quan theo từng cặp loại biến:

  * **Phi correlation** cho biến nhị phân – nhị phân
  * **Spearman correlation** cho biến thứ bậc và biến liên tục (dữ liệu phân phối lệch)
  * **Point-Biserial correlation** cho biến nhị phân – liên tục
* Kiểm tra **đa cộng tuyến (VIF)** để phát hiện các biến chồng chéo thông tin (Education, Income, GenHlth, BMI)
* Rút ra insight y học: BMI là "ngòi nổ" của chuỗi bệnh lý nền; GenHlth (cảm nhận sức khỏe chủ quan) có giá trị dự báo cao nhất; yếu tố kinh tế-xã hội có tác động bảo vệ gián tiếp

### 3\. Xây dựng \& đánh giá mô hình Machine Learning (machine-learning-for-data-mining-2.ipynb)

* Lấy mẫu phân tầng 100,000 quan sát để huấn luyện
* Thử nghiệm 5 mô hình dạng cây: **Decision Tree, Random Forest, Gradient Boosting, XGBoost, AdaBoost**
* Đánh giá bằng **Stratified K-Fold Cross-Validation**, ưu tiên **Recall** và **F1-Score** thay vì Accuracy (do bài toán y tế cần giảm tối đa False Negative)
* Xử lý mất cân bằng lớp bằng **class weighting** (không dùng resampling để tránh làm sai lệch phân phối thực tế)
* Fine-tune bằng **RandomizedSearchCV**
* Đánh giá mô hình được chọn qua **Precision-Recall Curve** và **ROC-AUC Curve** trên tập Test
* Diễn giải mô hình bằng **Feature Importance** và trực quan hóa cây quyết định để rút ra luật dự đoán dễ hiểu (VD: GenHlth + HighBP là hai yếu tố quyết định mạnh nhất)
* Lựa chọn **Decision Tree** làm mô hình triển khai cuối cùng vì tính đơn giản, dễ diễn giải — phù hợp bối cảnh y tế cần minh bạch trong quyết định, dù đánh đổi F1-Score thấp hơn để đổi lấy Recall cao (\~0.85)
* Model đã huấn luyện được lưu tại `dt\\\_tuned\\\_pipeline.joblib`

### 4\. Xây dựng Dashboard tương tác (diabete.py)

* Ứng dụng web bằng **Streamlit**, gồm 7 trang điều hướng qua sidebar:

  * **Overview** — thống kê tổng quan \& phân phối biến mục tiêu
  * **Ordinal vs Diabetes** — phân tích biến thứ bậc (GenHlth, Education, Income, AgeGroup)
  * **Binary vs Diabetes** — bar chart \& pie chart tương tác cho biến nhị phân
  * **Continuous vs Diabetes** — KDE plot so sánh phân phối BMI/MentHlth/PhysHlth
  * **Correlation** — heatmap tương quan Phi giữa các biến nhị phân
  * **Top Risk Factors** — xếp hạng yếu tố nguy cơ theo tỷ lệ mắc bệnh
  * **Diabetes Risk Prediction** — form nhập liệu (BMI tự tính từ chiều cao/cân nặng, huyết áp, cholesterol, tuổi, đánh giá sức khỏe) để dự đoán nguy cơ theo mô hình đã huấn luyện, trả về xác suất mắc bệnh

## 🛠️ Công nghệ sử dụng

|Hạng mục|Công cụ|
|-|-|
|Ngôn ngữ|Python|
|Xử lý dữ liệu|pandas, numpy|
|Trực quan hóa|matplotlib, seaborn|
|Thống kê|scipy.stats (chi2\_contingency, spearmanr, pointbiserialr), statsmodels (VIF)|
|Machine Learning|scikit-learn (DecisionTree, RandomForest, GradientBoosting, AdaBoost, Pipeline, ColumnTransformer, RandomizedSearchCV), XGBoost|
|Lưu trữ mô hình|joblib|
|Dashboard / Deployment|Streamlit|

## 📂 Cấu trúc project

```
├── preprocess-for-data-mining-2.ipynb   # Bước 1: Tiền xử lý dữ liệu
├── EDA_-_Diabetes__1_.ipynb             # Bước 2: EDA + phân tích tương quan
├── machine-learning-for-data-mining-2.ipynb  # Bước 3: Huấn luyện \\\& đánh giá mô hình
├── dt_tuned_pipeline.joblib             # Mô hình Decision Tree đã fine-tune (dùng cho dashboard)
├── diabete.py                           # Bước 4: Dashboard Streamlit
└── README.md
```

> *Lưu ý:* Cần đặt `data_clean.csv` / `data_decoded.csv` vào thư mục `Code/` (theo path được gọi trong `diabete.py`) trước khi chạy dashboard. Dataset gốc có thể tải từ Kaggle theo link ở trên.

## ▶️ Cách chạy dashboard

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
streamlit run diabete.py
```

## ⚠️ Giới hạn của đề tài

* Dữ liệu tự khảo sát (self-reported) nên có thể tồn tại sai lệch do khai báo không chính xác
* Dữ liệu mất cân bằng lớp nghiêm trọng ảnh hưởng đến độ ổn định của mô hình
* Ứng dụng dự đoán chỉ mang tính tham khảo, không thay thế chẩn đoán y khoa chuyên nghiệp

