# 🍔 Đồ án NLP - Aspect-Based Sentiment Analysis (ABSA)

---

## 📂 1. Cấu trúc thư mục dự án

Để code chạy mượt mà trên mọi hệ điều hành (Windows, macOS, Linux) mà không bị lỗi đường dẫn (Path Error), giữ nguyên cấu trúc thư mục như sau sau khi tải về:

```text
NLP_Final_Project/               <-- Thư mục gốc chứa toàn bộ dự án
│
├── data/                        <-- Thư mục chứa dữ liệu
│   └── Restaurants_Train_v2.csv <-- File dataset gốc (~3.500 dòng)
│
├── ML_BAN_1.ipynb               <-- File Source Code Jupyter Notebook chính
└── README.md                    <-- File hướng dẫn này

2. Cài đặt môi trường và Thư viện cần thiết
Dự án này được viết bằng Python. Trước khi chạy code, hãy đảm bảo máy bạn đã cài đặt Python và các thư viện cần thiết.

Bước 1: Mở Terminal (trên MacBook) hoặc Command Prompt / PowerShell (trên Windows).

Bước 2: Chạy câu lệnh dưới đây để cài đặt toàn bộ các thư viện Toán học và Trực quan hóa:
pip install pandas scikit-learn matplotlib seaborn numpy jupyter

🚀 3. Hướng dẫn chạy Code (One-Click Run)
Sau khi cài đặt xong thư viện và tải thư mục dự án về máy:

Mở phần mềm Jupyter Notebook, VS Code, hoặc Google Colab.

Mở file code có tên ML_final.ipynb.

Trên thanh menu trên cùng, tìm nút Kernel -> Chọn Restart & Run All (hoặc nút Run All / Chạy tất cả).

Ngồi nhâm nhi một ngụm nước và xem hệ thống tự động:

Đọc và làm sạch dữ liệu.

Chia tập dữ liệu theo tỷ lệ chuẩn 80% (Train) - 20% (Test).

Dạy cho 4 "bộ não" SVM học từ vựng.

Tự động làm bài kiểm tra trên hơn 650 câu review lạ.

Xuất ra 4 bảng báo cáo Ma trận nhầm lẫn và Biểu đồ cột cực xịn sò ở cuối trang!

🧠 4. Giải thích nhanh Logic Hệ thống
Để tiện cho việc bảo vệ trước hội đồng, mọi người nắm nhanh luồng xử lý sau:

Tách biệt 4 bộ não: Hệ thống không gộp chung mà tạo ra 4 Pipeline SVM song song chạy độc lập. Mỗi Pipeline chuyên trách đánh giá một khía cạnh duy nhất (Ví dụ: Một mô hình chỉ chuyên tìm xem người ta có nói gì về Food không).

3 Sắc thái đầu ra: Khi đọc 1 câu review, mỗi bộ não sẽ đưa ra 1 trong 3 kết luận:

POSITIVE: Lời khen.

NEGATIVE: Lời chê.

UNKNOWN: Không nhắc đến khía cạnh này (hoặc câu quá khó hiểu/trung lập).

Đánh giá thực tế: Tại bước cuối, nếu thấy các cột chỉ số POSITIVE / NEGATIVE hơi thấp, thì đừng hoảng sợ, đây không phải lỗi code! Đây là hiện tượng toán học bình thường phản ánh điểm yếu chí mạng của Machine Learning truyền thống (nguyên lý đếm từ Bag-of-Words) khi gặp câu mỉa mai hoặc dữ liệu mất cân bằng.