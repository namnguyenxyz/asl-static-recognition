# Bộ khung LaTeX luận văn/KLTN ASL

Bản thảo này phản ánh các kết quả đã có artifact của đồ án nhận dạng 36 lớp ảnh bàn tay theo quy ước ASL-HG. Hiện gồm sáu chương, dùng class LaTeX tạm thời; cần thay class, bìa và thông số hành chính khi khoa cung cấp template chính thức.

## Biên dịch

Khuyến nghị dùng XeLaTeX:

```bash
cd thesis
xelatex main.tex
 bibtex main
xelatex main.tex
xelatex main.tex
```

Nếu môi trường chưa có Times New Roman, file `main.tex` tạm dùng TeX Gyre Termes để có thể biên dịch; khi nộp bản chính thức phải kiểm tra lại font theo template UIT/khoa.

## Nguyên tắc nội dung

- Chỉ dùng số liệu đã xuất từ notebook hoặc artifact có thể truy vết.
- `mp-mnv4-003` là kết quả MediaPipe ROI + MobileNetV4 đã có artifact.
- Các thông tin hành chính chưa được cung cấp (ví dụ tên giảng viên hướng dẫn) được để trống, không suy đoán.
- Mục/tiểu mục sẽ điều chỉnh sau khi có template chính thức.
