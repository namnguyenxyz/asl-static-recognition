# Bộ khung LaTeX luận văn/KLTN ASL

Bộ khung này là bản nháp nội dung đầu tiên cho đề tài nhận dạng cử chỉ tay tĩnh ASL. Hiện đang dùng phương án 5 chương dành cho đồ án tốt nghiệp; cần thay class, bìa và thông số cuối cùng khi khoa cung cấp template chính thức.

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
- Các kết quả chưa có được đánh dấu `TODO`.
- Mục/tiểu mục sẽ điều chỉnh sau khi có template chính thức.
