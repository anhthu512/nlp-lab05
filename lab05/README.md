# LAB 05 — Sequence Models: RNN, LSTM and GRU

Đề nguồn: `W5 (1).pdf`, 23 trang, mục 1–32. Hạn nộp ghi trong đề: **23:59 ngày 21/10/2026**. Dataset: IMDb Movie Reviews, negative=0, positive=1. Đây là LAB05, không phải W4.

Notebook `sequence_classification.ipynb` chỉ gồm **code và output thực thi**, không có cell Markdown trả lời phân tích. Các kết luận, lời giải tính tay và câu trả lời cá nhân được để cho người học tự làm theo mục 27. Không dùng các bảng số liệu làm câu trả lời thay cho phần giải thích.

## Phân công theo từng mục của W5

| Mục đề | Code / output cần làm | Phần người học tự viết, không dùng AI trả lời | Cần dùng output để viết không? | Nơi chèn scan |
|---|---|---|---|---|
| 1–3 | Bối cảnh, mục tiêu, lịch hoạt động; không có code riêng | Không có câu nộp riêng | Không | — |
| 4–5 | Lý thuyết sequence/RNN; không code giải bài tính tay | Tự hiểu hidden state, recurrence, classification; chuẩn bị cho mục 7–9, 14, 28 | Không bắt buộc | `reflection.md` nếu ghi phần ôn |
| 6 — Bài 1 | **Không dùng code giải bộ số tính tay trong đề** | Tự tính từng phép nhân ma trận, tổng và tanh; trình bày đầy đủ từng bước | **Không**, dùng dữ kiện đề | `calculations.md` |
| 7 — Bài 2 | Không có code bắt buộc riêng | Mô tả hidden state sau mỗi từ của “I did not like this movie”; ảnh hưởng của “not” khi đọc “like”; so sánh Average Word2Vec | **Không**, câu hỏi về cơ chế | `reflection.md` |
| 8 — Bài 3 | Probe thứ tự ở cell mục 25 là bằng chứng bổ sung | So sánh “dog bites man” / “man bites dog” dựa trên cơ chế Average Word2Vec và RNN; không chỉ dựa thực nghiệm | **Không bắt buộc**; output chỉ bổ sung | `reflection.md` |
| 9 | Không yêu cầu chứng minh giải tích hoặc code riêng | Giải thích sequence dài liên quan thế nào tới dependency xa và gradient | **Không** | `reflection.md` |
| 10 | Ghi nhận xác nhận trước khi train, khởi tạo cấu hình | Tự viết **5 prediction**, mỗi câu đủ Prediction / Reason / Confidence | **Không; phải viết trước training** | `prediction.md` |
| 11 | Số mẫu/class, phân bố, min/median/mean/max length, duplicate, missing text/labels, histogram | Chọn max_length nên dựa vào thông tin nào? | **Có**: length summary, quantiles, histogram; không chọn bằng test F1 | `reflection.md` |
| 12 | Tokenizer, vocabulary train-only, token IDs, PAD=0, UNK=1, split audit, OOV coverage | Tự giải thích padding khác truncation, padding không là thông tin ngôn ngữ | **Không bắt buộc**; nên dẫn ví dụ IDs/padding nếu cần | `reflection.md` |
| 13 | Chạy RNN với 50/100/200 tokens ngay tại mục 13; xuất đủ % tokens retained, training time, accuracy, F1 | Tăng max_length có luôn tốt hơn? max_length nhỏ làm mất thông tin nào? | **Có** cho nhận xét thực nghiệm; lý thuyết tự giải thích | `reflection.md` |
| 14 | `rnn_cell.py`: `SimpleRNNCell`, `forward`; input nhỏ độc lập; đối chiếu `torch.nn.RNNCell` | Tự giải thích computation graph và RNN forward pass; người học phải đọc, kiểm tra và hiểu code | **Có** khi trình bày đối chiếu framework; **không** cần output để giải thích công thức/cơ chế | `reflection.md` |
| 15 | Refit TF-IDF unigram + LR LAB04; so sánh với RNN100 đã train tại mục 13; đủ bảng test/validation và bằng chứng thứ tự từ ngay tại mục 15 | RNN có sử dụng sequence mà TF-IDF không dùng không? Phải có evidence | **Có**: baseline, probes và phép hoán vị có kiểm soát | `reflection.md` |
| 16 | Không cần tự implement LSTM từ đầu | Tự học cell state, forget/input/output gates và giải thích bằng lời mình | **Không** | `reflection.md` |
| 17 | **Không code giải bộ số LSTM gates của đề** | Tự tính cell state và trả lời forget gate gần 1 / gần 0 | **Không**, dùng dữ kiện đề | `calculations.md` |
| 18 | Không cần tự implement GRU từ đầu | Tự nắm hidden state, update mechanism, cell state/tham số và so sánh GRU/LSTM | **Không bắt buộc**; số tham số thực đo có thể bổ sung | `reflection.md` |
| 19 | RNN/LSTM/GRU cùng điều kiện; parameters, training time, accuracy, precision, recall, F1 | So kết quả với prediction ban đầu; không chỉ kết luận bằng accuracy | **Có**, cần cả prediction đã viết trước | `reflection.md` |
| 20 | RNN cố định, length 50/100/200; dùng các run mục 13 để lập bảng retention, thời gian, validation loss, test F1, memory và 2 đồ thị ngay tại mục 20 | Có trade-off context/computational cost không? | **Có** | `reflection.md` |
| 21 | Câu có cue ở đầu/giữa/cuối; xác nhận cue còn hay bị truncate; probability từng model | Tự nhận xét ảnh hưởng vị trí; phân biệt observation với giải thích lý thuyết; thí nghiệm nhỏ không chứng minh đầy đủ vanishing gradient | **Có**, kết hợp kiến thức tự học; không suy ra nguyên nhân chỉ từ bảng | `reflection.md` |
| 22 | Accuracy/precision/recall/F1, confusion matrices, parameters, epochs, sequence length, vocab size, training time | Tự diễn giải evaluation khi dùng trong so sánh/phân tích | **Có** | `reflection.md`, `error_analysis.md` |
| 23 | Train/validation loss theo **mọi epoch**, đánh dấu checkpoint | Có dấu hiệu overfitting không? Không chỉ nhìn epoch cuối | **Có**, bắt buộc dùng curves/history | `reflection.md` |
| 24 | Pool nguyên văn test, true/predicted/probability, phần giữ/bỏ, 3 đúng + 3 sai chưa gán nguyên nhân | Tự chọn tối thiểu **3 đúng + 3 sai**, tự phân tích từng mẫu, xác định lỗi có liên quan sequence modeling không và bằng chứng | **Có**, bắt buộc đối chiếu text, nhãn, prefix/suffix, OOV; không mặc định nguyên nhân | `error_analysis.md` |
| 25 | Bảng LAB04 lịch sử + LAB05 hiện tại, probe và hoán vị prefix giữ nguyên multiset tokens | RNN/LSTM/GRU giải quyết vấn đề gì của Avg Word2Vec và vẫn còn vấn đề gì? | **Có** khi nhận xét thực nghiệm; phần cơ chế cần tự suy luận | `reflection.md` |
| 26 | Application LSTM/GRU hoàn chỉnh; review ngắn/dài/negation/contrast; probabilities, retained text; bảng nhiễu UNK theo đoạn | Tự giải thích từ/đoạn context có khả năng ảnh hưởng tới mỗi prediction; không chỉ ghi nhãn | **Có**, bắt buộc xem input thực model nhận; bảng perturbation chỉ là evidence phụ | `reflection.md` |
| 27 | AI assistance statement ở dưới | Tự kiểm tra, bổ sung Modified/Verified và có khả năng giải thích code | Không cần scan riêng; bổ sung README nếu cần | `README.md` |
| 28 | Không code trả lời individual check | Tự ôn các câu trong 3 phút; không dùng AI trả lời hộ | **Không bắt buộc**; câu về curves/truncation có thể dùng output để ôn | Không yêu cầu file nộp riêng; có thể scan ghi chú vào `reflection.md` |
| 29–31 | Đối chiếu deliverables/rubric/peer assessment; verification kỹ thuật | Hoàn thành scan, kiểm tra đủ lời giải, nhận xét, hiểu model | Có đối với các mục định lượng | Bốn file `.md` và PDF scan |
| 32 | Mạch nối LAB01→05→Attention, không có experiment riêng bắt buộc | Tự rút ra reflection nếu viết thêm | Không bắt buộc | `reflection.md` |

## Cấu hình và tính công bằng của thí nghiệm

- Dùng toàn bộ IMDb có nhãn từ revision `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`; hai parquet được kiểm tra SHA256. Không dùng unsupervised split.
- Dùng lại chính xác manifest LAB04 sau clean/deduplicate: **19.920 train / 4.981 validation / 24.675 test**. Kiểm tra sample ID, nhãn, document key và không trùng token document giữa split. Giữ manifest và bản kết quả LAB04 trong `data/` để chạy độc lập.
- Tokenizer giữ negation, vocabulary fit **train only**, tối đa 20.000 gồm PAD/UNK, min count 2. Word embeddings được học cùng classifier, không dùng pretrained.
- Seed 42; embedding 64, hidden 64, 1 layer, unidirectional, batch 128, Adam lr 0.001, 6 epochs, gradient clipping 1.0. Các model dùng cùng embedding initialization, cùng thứ tự batch theo seed, cùng split. Tham số recurrent khác nhau được báo cáo thay vì cố ép bằng nhau.
- Thí nghiệm kiến trúc dùng length=100; thí nghiệm độ dài cố định RNN, length=50/100/200, reset cùng seed/cấu hình. **Mục 13 chạy đủ ba RNN và xuất bảng yêu cầu tại chỗ. Mục 14 kiểm tra RNN cell. Mục 15 chạy baseline và xuất bảng so sánh, probes/hoán vị TF-IDF vs RNN tại chỗ. Mục 19 chạy LSTM/GRU và xuất đầy đủ bảng so sánh ba kiến trúc. Mục 20 dùng ba run RNN mục 13 để xuất bảng mở rộng và hai đồ thị đúng mục.** RNN100 được tái sử dụng trong các so sánh; không train lặp rồi chọn lần tốt hơn.
- Right truncation giữ **prefix**; padding bên phải theo batch. Dùng packed sequences và hidden state cuối cùng hợp lệ; kiểm tra thêm padding không đổi logits. `% tokens retained` là tổng token giữ / tổng token gốc, tính riêng từng split; không nhầm với tỷ lệ review không bị cắt.
- Tất cả model train đủ 6 epochs; checkpoint chọn **minimum validation loss**. Mọi hyperparameter và tập thí nghiệm cố định trước mục 13. Primary model chọn bằng validation F1 trong RNN/LSTM/GRU length100; application chọn bằng validation F1 trong LSTM/GRU length100 tại mục 19. Để có bảng đúng thứ tự đề, RNN được báo cáo test tại mục 13, baseline tại mục 15 và LSTM/GRU tại mục 19; không dùng bất kỳ test metric nào cho lựa chọn. Cell 22 tổng hợp evaluation của tất cả cấu hình vào `results.csv` và confusion matrices. Các bảng test/validation ghi rõ split, không dời kết quả bắt buộc của mục trước sang mục sau.
- Baseline TF-IDF unigram+LR dùng đúng cấu hình LAB04: cap 20.000, min_df=2, sublinear TF, C=1, liblinear, giữ negation. Baseline dùng toàn văn; sequence models chỉ dùng prefix đã cấu hình. Đây là so sánh pipeline, cần người học xét giới hạn này khi nhận xét. Không dùng test chọn C.
- Chỉ báo cáo một seed, không phải kết luận thống kê trên nhiều lần chạy. Thí nghiệm cue/hoán vị đo hành vi của các model đã train; không chứng minh nguyên nhân lỗi hay vanishing gradient.

Training time neural = tổng thời gian training batches qua 6 epochs, có đồng bộ CUDA, không cộng validation/checkpoint/evaluation; `validation_time_seconds` và `fit_wall_seconds` lưu riêng. LR time = TF-IDF fit_transform train + classifier.fit, không gồm test transform. Baseline epochs không áp dụng nên để trống, solver iterations lưu riêng. Các cột precision/recall/F1 chính dùng positive=1, threshold=0.5; có thêm macro F1 và per-class metrics.

GPU memory là peak allocated PyTorch trong mỗi run (không phải toàn bộ VRAM). CPU memory là process RSS lấy mẫu ở ranh giới epoch (không phải peak chính xác, có thể bao gồm dữ liệu/model đã nạp của kernel); không dùng số này như bộ nhớ thuần riêng từng kiến trúc.

## Chạy lại

1. Cài `python -m pip install -r requirements.txt`. Muốn chạy GPU cần bản PyTorch/CUDA phù hợp máy; notebook tự chuyển CPU nếu không có GPU.
2. Đặt working directory ở `lab05/`, mở `sequence_classification.ipynb`, chạy tất cả cell theo thứ tự. Notebook gồm code cells theo mục **10,11,12,13,14,15,19,20,21,22,23,24,25,26,29**; không thêm cell giả cho mục chỉ tự viết.
3. **Trước lần training đầu tiên**, phải tự viết đủ mục 10. Trong phiên này người học đã xác nhận cả 5 prediction W5 hoàn tất trước khi huấn luyện; ghi nhận ở `prediction_confirmation.json`. File này ghi nhận xác nhận, không thay bản prediction hoặc bản scan. Người khác sử dụng lại phải tự hoàn thành prediction và sửa thông tin xác nhận cho đúng thực tế.
4. Hai parquet nằm trong `data/`; nếu bị thiếu, loader tải đúng revision và kiểm checksum. `data/lab04_split_manifest.csv` và `data/lab04_results.csv` cần giữ để chạy độc lập và đối chiếu.

```powershell
python -X utf8 -c "import nbformat; from nbclient import NotebookClient; from pathlib import Path; p=Path('sequence_classification.ipynb'); n=nbformat.read(p,4); c=NotebookClient(n,timeout=3600,kernel_name='python3',resources={'metadata':{'path':str(Path.cwd())}}); c.execute(); nbformat.write(n,p)"
```

## Nguồn output để tự viết

| Câu/mục | Output trong notebook và file |
|---|---|
| 11 | `outputs/dataset_audit_raw.csv`, `class_distribution_raw.csv`, `length_summary_raw.csv`, `length_quantiles_raw.csv`, `sequence_length_histogram.png` |
| 12–13 | `outputs/cleaning_audit.csv`, `split_distribution.csv`, `split_overlap_audit.csv`, `vocabulary_coverage.csv`, `tokens_retained.csv`; IDs/padding trong cell 12 |
| 14 | Cell 14: input/output và framework comparison; `outputs/rnn_cell_verification.csv`. Không dùng thay lời giải bộ số mục 6 |
| 15/19/22 | `outputs/baseline_test_comparison.csv`, `baseline_validation.csv`, `architecture_validation.csv`, `architecture_test_comparison.csv`, `all_main_models_test_comparison.csv`, `results.csv`, `per_class_metrics.csv` |
| 13/20 | `outputs/padding_truncation_results.csv`, `sequence_length_results.csv`, `sequence_length_experiment.png` |
| 21 | `outputs/long_range_position_probes.csv`: vị trí, max_length, cue_retained, probability |
| 23 | `learning_curves.png`, `outputs/learning_history.csv`, `history_<model>.csv` |
| 24 | `confusion_matrix.png`, `outputs/confusion_matrices_comparison.png`, `error_analysis_candidates.csv`, `six_examples_unannotated.csv`, `predictions_<model>.csv` |
| 15 | `outputs/baseline_word_order_probes.csv`, `baseline_word_order_ablation.csv` (TF-IDF/RNN, xuất ngay tại cell 15) |
| 25 | `outputs/word_order_probes.csv`, `word_order_ablation.csv`, `lab04_lab05_comparison.csv` (bổ sung LSTM/GRU) |
| 26 | `outputs/application_predictions.csv`, `application_span_perturbations.csv`, `model_selection.json` |

CSV giữ nguyên văn đầy đủ; notebook có thể rút ngắn hiển thị cột text cho dễ đọc. `six_examples_unannotated.csv` chỉ lấy 3 đúng/3 sai đầu tiên để tiện xem, không gán nguyên nhân hoặc viết Analysis; người học có thể tự chọn mẫu khác trong pool. Phép hoán vị giữ nguyên cùng prefix 100 tokens cho cả bản gốc và shuffled, gồm cả TF-IDF, nhằm tách thay đổi thứ tự khỏi thay đổi token do truncation. Mức nhạy thứ tự không tự chứng minh model hiểu ngữ nghĩa.

Bảng LAB04 có nguồn `historical` và thời gian của lần chạy cũ, không trộn thành thời gian mới. Cùng manifest và số mẫu giúp đối chiếu, nhưng điều kiện representation và chi phí không hoàn toàn đồng nhất. Bảng perturbation thay từng đoạn 5 token bằng UNK, giữ độ dài/vị trí; chỉ là phép đo nhiễu đầu vào, không phải lời giải nguyên nhân hay attribution chắc chắn. Phần explanation phải do người học viết.

## Viết tay, scan và chèn vào Markdown

Tự viết theo câu hỏi để trống tại bốn Markdown:

- `calculations.md` → `calculations_scan.pdf`: mục 6 và 17.
- `prediction.md` → `prediction_scan.pdf`: 5 prediction mục 10 đã viết trước training; giữ nguyên prediction ban đầu.
- `error_analysis.md` → `error_analysis_scan.pdf`: mục 24; kèm diễn giải confusion matrix/evaluation liên quan.
- `reflection.md` → `reflection_scan.pdf`: các phần giải thích/nhận xét còn lại theo checklist trong file.

Đặt PDF cạnh file `.md`. Mỗi `.md` đã có link tương đối và `<object>` nhúng PDF; GitHub/trình xem không hỗ trợ nhúng vẫn mở được link PDF. Phải nộp **cả `.md` và PDF scan**; link không chứa nội dung file scan. Đề W5 yêu cầu các Markdown, còn scan PDF chèn Markdown là quy cách bổ sung người học yêu cầu trong phiên này. Chưa tạo scan giả hoặc trả lời thay. Có thể viết nhiều trang trong mỗi PDF.

## Deliverables và trạng thái

Mục 29 yêu cầu: `README.md`, `calculations.md`, `prediction.md`, `rnn_cell.py`, `sequence_classification.ipynb`, `results.csv`, `learning_curves.png`, `confusion_matrix.png`, `error_analysis.md`, `reflection.md`.

Giữ thêm `preprocessing.py`, `sequence_utils.py`, `requirements.txt`, `prediction_confirmation.json`, `data/`, `outputs/` để notebook chạy lại và có evidence. Checkpoints trong `outputs/checkpoints/` phục vụ application; dữ liệu/cache/model là file hỗ trợ, không phải file đề liệt kê riêng.

Trạng thái kỹ thuật sau khi chạy được ghi ở `outputs/verification.json` và `outputs/notebook_verification.json`; cấu hình/checksums ở `outputs/run_metadata.json`. **Phần bài nộp chỉ hoàn chỉnh khi người học tự hoàn thành bốn PDF scan và có thể giải thích code.** Không xem output máy sinh là phần nhận xét đã làm.

## AI assistance statement — mục 27

- **Tool:** ChatGPT / Codex.
- **Purpose:** Hỗ trợ preprocessing, wiring API NumPy/PyTorch/sklearn, training/evaluation boilerplate, visualization, documentation, debugging và kiểm tra kỹ thuật.
- **Generated:** `rnn_cell.py`, `sequence_utils.py`, code trong notebook, khung Markdown/checklist/link scan, `requirements.txt`, script chạy/kiểm tra ở thư mục `tmp/w5/`; CSV/hình/checkpoint do code thực thi sinh ra. Không tạo lời giải mục 6/17 hoặc lời giải thích RNN/LSTM thay người học.
- **Modified:** Tạo `lab05/`; sao chép helper preprocessing và dữ liệu/manifest/kết quả LAB04 sang LAB05 để tái sử dụng; thêm notebook code-only và các file hỗ trợ. Không sửa nội dung lab trước. Người học tự bổ sung chỉnh sửa của mình sau phiên này.
- **Verified:** Phần kiểm tra máy được ghi bằng output chạy thật trong các JSON verification; người học phải tự kiểm tra lại, hiểu `SimpleRNNCell` và từng thành phần model, tự viết phân tích và bổ sung xác nhận kiểm tra của bản thân. Code RNN cell có hỗ trợ AI, không được trình bày như code tự viết không có hỗ trợ; mục 14 yêu cầu người học tự implement/hiểu cell nên cần tự rà soát và điều chỉnh bằng hiểu biết của mình.

Không dùng AI thay prediction, calculation, giải thích RNN forward pass/LSTM gates, chọn nguyên nhân error, interpretation learning curves, reflection hoặc individual learning check.
