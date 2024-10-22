Đề tài: Trình nhận dạng chữ số viết tay sử dụng Deep Learning


Giảng viên hướng dẫn: ThS. Hồ Nhựt Minh


Thành viên tham gia:
- Lê Chấn Nam
- Võ Hoài Thương
- Triệu Quang Ninh
- Hà Mạnh Trình


![output](https://github.com/user-attachments/assets/bebe03b4-e8f8-4430-b41d-c3aee1a69b61)

**DỰ ĐOÁN TRÊN TỪNG HÌNH ẢNH RIÊNG LẺ**

- **Mục tiêu**

Sau khi đã huấn luyện mô hình mạng nơ-ron tích chập (CNN) trên bộ dữ liệu MNIST, bước tiếp theo là kiểm tra khả năng của mô hình bằng cách dự đoán chữ số trên từng hình ảnh cụ thể.

- **Thực hiện**

Để dự đoán chữ số viết tay bằng mô hình CNN đã huấn luyện, ta cần thực hiện các bước sau:

1. **Chọn ảnh:** Chọn ảnh từ tập kiểm thử hoặc ảnh tự tạo.
2. **Tiền xử lý:** Định dạng lại ảnh về kích thước 28x28 và chuẩn hóa giá trị pixel về khoảng 0-1.
3. **Dự đoán:** Sử dụng mô hình CNN để dự đoán chữ số trong ảnh.
4. **Hiển thị kết quả:** So sánh kết quả dự đoán với nhãn thực tế của ảnh.
5. **Kết quả:** Dưới đây là một ví dụ về kết quả dự đoán trên một số hình ảnh từ tập kiểm thử:

![image (1)](https://github.com/user-attachments/assets/2763c07a-c72a-4211-bb4f-e2486bfdc0dc)
