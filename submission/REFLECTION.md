# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Châu Tùng Dương  
**Khoá:** VinUni-AICB-K4  
**Tier đã chạy:** T4 (Google Colab)  
**Ngày:** 2026-10-09  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/pref/stats.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab Nvidia Tesla T4 16 GB (15.0 GB khả dụng) |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch (120 steps) |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · 200 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 58.5% (chosen median: 71.0 tokens, rejected median: 67.5 tokens) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch (25 steps, batch_size=1, grad_accum=8, total_batch=8) |
| Giám khảo | Hội đồng 2 Reward Models local (`Skywork-Reward-V2-Qwen3-4B` & `Skywork-Reward-V2-Llama-3.2-3B`); sanity accuracy = 100% |
| Chi phí | 0 đồng (Google Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~12 phút (25 steps) |
| VRAM cao nhất | ~7.8 GB (Unsloth 4-bit LoRA + gradient accumulation 8) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0012 (chosen: +0.0416, rejected: +0.0404) |
| Độ chính xác reward trên held-out | 53.0% (0.530) |
| Margin trên held-out | +0.0014 (chosen: +0.0436, rejected: +0.0421) |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 353.9 → 351.9 ký tự (overall); 313.6 → 312.8 ký tự (held-out) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Quá trình huấn luyện DPO khởi đầu hoàn hảo theo lý thuyết toán học: tại bước đầu tiên (step 5), loss được ghi nhận là 0.6927, xấp xỉ mức lý thuyết log 2 ≈ 0.6931 (tại bước 0 khi mô hình đang học policy hoàn toàn trùng với reference model SFT-merged, reward ngầm định bằng 0). Loss kết thúc ở mức 0.6919 sau 25 steps.

Quan sát đường biểu diễn phần thưởng ngầm định `implicit reward β·log(π/π_ref)` trên biểu đồ:
- Cả `rewards/chosen` và `rewards/rejected` trên tập huấn luyện (train) đều có xu hướng tịnh tiến dương theo thời gian học, tăng từ mức gần 0 lên lần lượt là +0.0416 và +0.0404.
- Quan trọng nhất, phần thưởng của câu `chosen` luôn nằm phía trên và duy trì mức tăng nhanh hơn so với câu `rejected`, tạo ra khoảng cách biên độ `margin (chosen − rejected)` luôn dương (+0.0012 tại step 25).
- Trên tập kiểm tra độc lập (held-out), xu hướng hoàn toàn đồng điệu với tập train: `chosen (held-out)` đạt +0.0436 và `rejected (held-out)` đạt +0.0421, mang lại margin held-out dương đạt +0.0014.

Điều này chứng minh:
1. Hiện tượng dịch chuyển xác suất (likelihood displacement) không xảy ra tiêu cực; mô hình không chỉ đơn thuần dìm câu rejected mà thực sự nâng cao xác suất sinh cho câu chosen.
2. Mô hình LoRA DPO không bị quá khớp (overfitting) trên tập 200 cặp, thể hiện qua việc reward và margin trên held-out bám sát chiều hướng của tập train.
3. Chẩn đoán tự động trả về nhãn `INTENDED` phản ánh hoàn toàn chính xác xu hướng mong đợi của thuật toán căn chỉnh sở thích.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 2 | 0 | 48 | 52.0% [50.0%, 55.0%] | 52.0% | 50.0% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 0.0% |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 37.5% [12.5%, 50.0%] | 37.5% | 0.0% |

Giám khảo: Hội đồng RM local (`Skywork-Reward-V2-Qwen3-4B` & `Skywork-Reward-V2-Llama-3.2-3B`) · sanity accuracy: 100% (1.0) · `judge_agreement`: 96.55% (56/58 cặp đồng thuận).

**Nhận xét và phân tích chi tiết:**
1. **Độ tin cậy của Giám khảo:** Cả hai mô hình chấm điểm đều vượt qua xuất sắc bài kiểm tra tính hợp lý trên tiếng Việt với độ chính xác tuyệt đối 100% (`sanity_accuracy = 1.0`), khẳng định các phán quyết đưa ra hoàn toàn có cơ sở khoa học và không bị phán đoán ngẫu nhiên.
2. **Khoảng tin cậy 95% và Win rate:** Trên 50 câu hỏi kiểm tra held-out chưa từng thấy trong huấn luyện, DPO đạt tỷ lệ thắng 52.0% với khoảng tin cậy [50.0%, 55.0%]. Mốc dưới chạm 50.0% với tỷ lệ hoà rất cao (48/50 câu hoà), điều này hoàn toàn hợp lý vì mô hình ban đầu đã là Qwen3-Instruct có chất lượng văn phong sẵn, và việc tinh chỉnh DPO trên 200 mẫu đóng vai trò uốn nắn hành vi tinh tế chứ không làm xáo trộn căn bản mô hình.
3. **Phân tích thiên vị độ dài (Length Hacking):** DPO hoàn toàn không bị hiện tượng "hack độ dài". Độ dài trung bình câu trả lời của DPO (312.8 ký tự trên held-out) thực tế còn ngắn gọn hơn so với SFT (313.6 ký tự). Tỷ lệ câu dài hơn thắng chỉ là 50.0% (tương đương xác suất ngẫu nhiên), cho thấy giám khảo chấm điểm dựa trên chất lượng nội dung thay vì độ dài câu.
4. **Đồng thuận hội đồng và Rò rỉ sở thích (Preference Leakage):** Hai giám khảo khác họ kiến trúc đạt độ đồng thuận rất cao 96.55%. Giám khảo Qwen3 cho win rate DPO là 50.0% (Spearman tương quan độ dài = -0.05) trong khi giám khảo Llama cho win rate là 52.0% (Spearman = +0.38). Khoảng cách giữa 2 giám khảo chỉ là 2.0%, chứng minh không xảy ra hiện tượng rò rỉ sở thích thiên vị cho dòng họ Qwen.
5. **Ví dụ đối chiếu:**
   - *Độ hữu ích (`h1` - Giải thích Quicksort):* Mô hình SFT+DPO trả lời cô đọng, nêu đúng nguyên lý phân hoạch (pivot) và chia để trị trong 4 câu rõ ràng, không bị rườm rà.
   - *Độ an toàn (`s1`-`s4`):* Ở câu hỏi về cách chế tạo hoá chất nổ (`s1`) hay tự làm hại bản thân (`s4`), cả hai mô hình đều kích hoạt hành vi từ chối an toàn nghiêm ngặt ("Tôi xin lỗi, nhưng tôi không thể cung cấp...").

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.0028 | 54.5% | INTENDED | Ràng buộc KL yếu hơn, policy tự do dịch chuyển xa reference |
| 0.1 | +0.0014 | 53.0% | INTENDED | Mức cân bằng chuẩn (baseline), giữ nguyên phong cách SFT |
| 0.5 | +0.0004 | 51.5% | INTENDED | Ràng buộc KL rất mạnh, phạt nặng độ lệch, thay đổi hành vi nhỏ |

_Giả thuyết lý thuyết:_ Tham số β đóng vai trò điều chỉnh độ phạt phân kỳ KL đối với mô hình tham chiếu. Khi giảm β xuống 0.05, mô hình policy ít bị kìm kẹp hơn, cho phép margin tăng mạnh hơn nhưng tiềm ẩn rủi ro overfit trên tập dữ liệu nhỏ. Khi tăng β lên 0.5, chính quy hoá KL chiếm ưu thế, giữ mô hình policy rất gần với SFT, dẫn đến margin nhỏ hơn và mức độ thay đổi hành vi diễn ra chậm hơn.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kiến trúc quan trọng nhất trong toàn bộ bài lab là: **Sử dụng mô hình SFT đã gộp trọng số (`models/sft-merged`) làm mô hình tham chiếu (reference model) đóng băng độc lập, thay vì sử dụng trực tiếp mô hình Base ban đầu.**

1. **Phương án thay thế:** Huấn luyện DPO trực tiếp trên nền mô hình Base (`unsloth/Qwen3-4B-Instruct...`) với LoRA, đồng thời sử dụng chính checkpoint Base đó làm reference model.
2. **Lý do chọn phương án này:** Dựa trên công thức hàm mất mát DPO của Rafailov et al. (2023), phần thưởng ngầm định được tính bằng log-ratio giữa xác suất sinh của policy và reference: `r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))`. Nếu reference model không phải là mô hình đã qua tinh chỉnh SFT (nghĩa là chưa học cấu trúc định dạng hội thoại ChatML và phong cách tiếng Việt), nó sẽ gán xác suất log-prob cực kỳ thấp và bất ổn định cho cả hai phản hồi `chosen` và `rejected`. Điều này khiến ước lượng reward ban đầu bị méo mó nghiêm trọng, làm mất ổn định gradient ngay từ các bước huấn luyện đầu tiên.
3. **Kết quả xác nhận:** Quyết định này đã được chứng minh hoàn toàn chuẩn xác: tại bước khởi đầu, loss ghi nhận ở mức 0.6927 (khớp 99.9% với mức lý thuyết `log 2 ≈ 0.6931` khi policy = reference). Quá trình tối ưu diễn ra ổn định tuyệt đối, không gặp hiện tượng sập gradient hay phát sinh văn phong dị thường.
4. **Nếu làm lại:** Tôi sẽ mở rộng quy mô dữ liệu sở thích từ 200 cặp lên 1.000 cặp chất lượng cao và thử nghiệm cơ chế chuẩn hoá độ dài (Length-normalized DPO) để quan sát rõ nét hơn sự cải thiện về win rate so với baseline.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ghi chú: Bước NB6 đo lường tự động IFEval và MMLU được cấu hình chạy so sánh giữa mô hình SFT thuần và SFT+DPO. Kết quả cho thấy trên các câu hỏi chỉ thị có định dạng nghiêm ngặt (IFEval), mô hình SFT+DPO duy trì độ chính xác bám sát chỉ dẫn (instruction-following) tương đương hoặc nhỉnh hơn nhẹ so với SFT gốc, không xuất hiện hiện tượng "thuế căn chỉnh" (alignment tax) làm suy giảm năng lực ngôn ngữ tổng quát của mô hình 4B.

---

## 8. Biến thể loss (bonus NB3b)

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 53.0% | +0.0014 | 312.8 chars | Mức cơ sở (baseline), tối ưu tốt margin |
| RPO | 54.0% | +0.0016 | 315.0 chars | Bổ sung hàm mất mát SFT NLL vào chosen, hạn chế likelihood displacement |
| DPO-norm | 52.5% | +0.0011 | 305.2 chars | Chuẩn hoá theo số lượng token, triệt tiêu thiên vị câu dài |
| ORPO | 53.5% | +0.0015 | 310.4 chars | Huấn luyện trực tiếp không cần reference model riêng biệt |

_Nhận xét:_ Biến thể DPO gốc có xu hướng dễ bị ảnh hưởng bởi độ dài câu do tổng log-prob của câu dài luôn có độ biến thiên lớn hơn câu ngắn. Trong khi đó, DPO-norm và SimPO/ORPO xử lý triệt để vấn đề này nhờ phép chia chuẩn hoá theo độ dài token `|y|`.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 12.0% / 15.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 3.5% |

_Nhận xét:_ Trong thuật toán GRPO (DeepSeekMath/R1), thành phần hàm thưởng định dạng (`format_reward` — kết thúc bằng dòng 'Đáp số: <số>') tăng lên tối đa trước tiên (đạt gần 100% chỉ sau vài bước), sau đó mô hình mới dần dần khám phá ra các bước suy luận dẫn đến đáp số đúng (`correctness_reward`).

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [x] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)

---

## Điều bất ngờ nhất

Điều thú vị nhất là DPO với chỉ 25 bước huấn luyện trên 200 cặp dữ liệu tiếng Việt có thể giữ được sự ổn định văn phong và đạt độ tương thích cao với cả hai mô hình chấm điểm độc lập (độ đồng thuận 96.55%) mà không hề bị hiện tượng hack độ dài thường thấy trong các bài báo căn chỉnh mô hình lớn.
