# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM


- **MSSV:2A202603453
- **Họ và tên:Nguyễn Khôi Nguyên**
- **SEED** (in ở ô đầu notebook): 2518

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.988   0.786      0.876        0.857   
2. BoW + MLP (deep learning)         0.967   0.864      0.913        0.947   
3a. Encoder (MiniLM) + kNN           0.857   0.932      0.893        0.923   
3b. Encoder (MiniLM) + LogReg        0.961   0.961      0.961        0.950   
3c. Encoder (MiniLM) + MLP head      0.942   0.951      0.947        0.923   
4a. LLM nhỏ zero-shot                0.826   0.950      0.884        0.884   
4b. LLM lớn zero-shot                0.905   0.950      0.927        0.927   
4c. LLM nhỏ few-shot                 0.826   0.950      0.884        0.884   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            4.977      15515.026       0.033   
2. BoW + MLP (deep learning)         9.397      53130.122       0.203   
3a. Encoder (MiniLM) + kNN          67.976         37.361      26.954   
3b. Encoder (MiniLM) + LogReg       68.156         42.489      21.906   
3c. Encoder (MiniLM) + MLP head     68.668         48.086      21.449   
4a. LLM nhỏ zero-shot                0.000          1.311     280.781   
4b. LLM lớn zero-shot                0.000          1.307     270.742   
4c. LLM nhỏ few-shot                 0.000          0.616    2319.020   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.047           0.0  
2. BoW + MLP (deep learning)          0.260           0.0  
3a. Encoder (MiniLM) + kNN           37.844           0.0  
3b. Encoder (MiniLM) + LogReg        32.316           0.0  
3c. Encoder (MiniLM) + MLP head      28.240           0.0  
4a. LLM nhỏ zero-shot              2296.496           0.0  
4b. LLM lớn zero-shot              2316.337           0.0  
4c. LLM nhỏ few-shot               2480.485           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

3.1 dl_curve — MLP so với Logistic Regression
Em đoán: val loss không giảm mãi, và MLP sẽ kém LogReg về F1.
Thực tế: Vế đầu đúng, val loss thấp nhất ở epoch 6 rồi tăng lại. Vế sau sai: MLP đạt F1 0,913, cao hơn LogReg (0,876), với recall 0,864 so với 0,786.
Vì sao lệch: Em nghĩ MLP có 128.129 tham số trên dữ liệu nhỏ sẽ overfit nặng. Thực tế LogReg đang underfit: F1 trên train chỉ 0,915, và recall thấp vì nó chỉ cộng trọng số từng từ riêng lẻ. MLP học được tổ hợp từ và được tối ưu bằng Adam, nên bắt thêm được spam. Sau epoch 6, val loss tăng chủ yếu vì model dự đoán quá tự tin trên vài tin sai. Mức đó chưa đủ làm đổi nhiều nhãn ở ngưỡng 0,5, nên F1 test vẫn cao hơn LogReg.
3.2 llm_fewshot — thêm 4 ví dụ vào prompt
Em đoán: F1 tăng 0,02–0,05, chi phí mỗi tin tăng khoảng 2–2,5 lần.
Thực tế: F1 không đổi (0,884 → 0,884, precision 0,826 và recall 0,95 giữ nguyên). Không kiểm được chi phí vì USD = 0. Độ trễ p50 tăng từ 281 ms lên 2.319 ms.
Vì sao lệch:
4 ví dụ được lấy máy móc (2 spam và 2 ham đầu tiên trong train) và đều là case dễ.
Lỗi của model nhỏ nằm ở phía precision, tức đoán nhầm ham thành spam (ví dụ "Nice line said by a broken heart…"). Recall đã 0,95, mà các ví dụ không dạy gì về loại lỗi này.
Về token, prompt few-shot dài khoảng gấp đôi, nên dự đoán "chi phí ×2" có lẽ đúng hướng, nhưng notebook không có giá để xác nhận.
Bài học: few-shot chỉ giúp khi ví dụ nhắm đúng chỗ model sai.
3.3 tradeoff — đường Pareto
Em đoán: BoW + LogReg nằm trên Pareto, LLM có thể cũng nằm trên Pareto nếu có F1 cao nhất, và mọi LLM đều vượt SLA.
Thực tế: Pareto gồm BoW + LogReg, BoW + MLP và 3b. LLM không nằm trên Pareto. Vế "LLM vượt SLA" thì đúng, p95 khoảng 2,3 s.
Vì sao lệch: Em đánh giá quá cao LLM và quá thấp encoder pre-train.
LLM không được train trên phân phối dữ liệu này, nên F1 SUB (0,884 và 0,927) không vượt 3b (0,950), trong khi p95 chậm hơn khoảng 70 lần.
LLM lớn bị 3b lấn át: 3b vừa nhanh hơn vừa ít lỗi hơn.
3a và 3c bị BoW + MLP lấn át.

## 4. Phân tích 50 case khó

- Model tụt nhiều nhất là BoW + MLP, F1 giảm từ 0,947 xuống 0,421 (−0,53). Tiếp theo là BoW + LogReg, từ 0,857 xuống 0,378 (−0,48).
- Nhóm gây tụt:
  + BoW + MLP đúng 0% ở hai nhóm spam giọng tự nhiên và né từ khoá, và chỉ 50% ở nhóm thao túng/lừa đảo.
  + BoW + LogReg chỉ đúng 12–20% ở ba nhóm spam này.
- Điểm cao ở nhóm ham là giả: BoW đạt 83–100% ở các nhóm ham, nhưng đó là vì khi không thấy từ khoá quen, nó gần như luôn đoán ham, chứ không phải vì hiểu nội dung.
- LLM bền nhất: đạt 84%, và lên 90% khi dùng prompt phòng thủ. Prompt này nâng nhóm prompt injection từ 83% lên 100% và nhóm tưởng spam từ 50% lên 70%, không làm nhóm nào tụt.

Case 1 — #29 (né từ khoá): "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"

Hàm tokenize cắt câu này thành các token như c0ngr4tulat10ns, y0u, w0n, cl4im, n0w. Phần lớn số token này không nằm trong vocab 2.000 từ của BoW, nên các cột tương ứng bằng 0.
Những từ còn được nhận ra lại là từ phổ thông như a, at, vốn hơi kéo về phía ham. Bias cũng âm, vì 87% dữ liệu train là ham.
Kết quả là BoW + MLP đoán ham. Cả hai model BoW chỉ đúng 0–17% ở nhóm này.
Encoder MiniLM tách từ thành subword, còn LLM đọc như người, nên vẫn nhận ra nghĩa "trúng thưởng": 3a và 3b đạt 83%, LLM đạt 100%.

Case 2 — #13 (spam giọng tự nhiên): "Hope you're well! Quick reminder that your photos from the event are ready, view & order prints here: snap-prints.co/r/88"

Đây là case duy nhất mà cả 7 model đều sai, kể cả LLM có prompt phòng thủ.
Tin không có từ khoá spam, giọng lịch sự, và có dạng giống một thông báo giao dịch hợp lệ như nhắc lịch hẹn hay báo đơn hàng, vốn là loại tin mà model được dạy là ham.
Dấu hiệu spam duy nhất nằm ngoài văn bản: người nhận chưa từng đặt dịch vụ này. Bộ lọc chỉ dựa vào nội dung thì không thể biết điều đó, nên cần thêm tín hiệu khác như người gửi, độ tuổi của domain trong link, hay lịch sử nhắn tin.

## 5. Thử thách cá nhân

Đề #5 (Embedding): Viết 4 tin nhắn của riêng mình, gồm 2 spam né từ khoá và 2 ham chứa "free"/"win". So sánh dự đoán của BoW với Embedding + LogReg và giải thích từng tin.

Cách làm: Dùng lại w, b của Phần 1 và w_e, b_e, mu, sd của 3b. Dự đoán trên 4 tin tự viết với ngưỡng 0,5.

Tin	
1. "Hey, it's me. I changed my phone number recently. Could you send some money to this account?…"	
2. "Hi! My cousin introduced me to a small online business… The returns are surprisingly good. Check this website…"	
3. "Hey, are you free this afternoon? I was thinking we could grab some coffee after class."	
4. "We finally managed to win the match last night!… Let's play again tomorrow."	
Kết quả: Mỗi model đúng 2/4. Cả hai đều bỏ sót cả 2 tin spam.

Giải thích:
- Tin 1 và 2 (spam, cả hai model sai):
  + Tin 1 là kiểu lừa "mạo danh người thân đổi số", tin 2 là lừa đầu tư/đa cấp.
  + Spam trong SMS Spam Collection (2011) chủ yếu là trúng thưởng, dịch vụ tính phí và hẹn hò, nên hai kiểu lừa này gần như không có trong dữ liệu train.
  + Với BoW, các từ như "money", "account", "business", "website" không có trọng số spam mạnh, còn "me", "i", "you" lại kéo về phía ham.
  + Với Embedding, encoder hiểu nghĩa câu, nhưng đầu LogReg chỉ học ranh giới từ các spam cũ. Vector của tin lừa giọng thân mật nằm gần vùng tin bạn bè.
- So với Thí nghiệm 3.B: Ở đó Enc + LR đúng 6/6 vì các tin spam vẫn mang nghĩa "nhận quà/thưởng" ("complimentary holiday", "no cost", "rewards expire"), tức gần với spam trong train. Hai tin của em không như vậy.
- Tin 3 và 4 (ham, cả hai model đúng):
  + BoW cộng trọng số tất cả các từ. "free" và "win" đẩy về spam, nhưng "afternoon", "coffee", "class", "match", "night", "play", "tomorrow", "we", "i" kéo về ham mạnh hơn.
  + Embedding nhìn nghĩa cả câu, là rủ đi cà phê và kể chuyện thi đấu, nên xếp vào vùng ham.

Kết luận: Kết quả này giống phần 50 case khó: cả hai model nhận ham khá tốt, nhưng hỏng ở spam kiểu mới không có trong dữ liệu train (BoW đúng 0–20%, Embedding 30–40% ở nhóm spam tự nhiên).

## 6. Phương án cho sản phẩm thật
Yêu cầu: 10 triệu SMS/ngày, tức khoảng 116 tin/giây trung bình, với p95 < 50 ms/tin.

Loại LLM làm bộ lọc chính: p95 khoảng 2,3 s, tức chậm gấp khoảng 46 lần SLA.
Chi phí ước tính theo giá gpt-6-luna ở phần tính tay ($10,5 cho 1 triệu tin, với 100 token input và 1 token output):
10 triệu tin/ngày tốn khoảng $105/ngày, tức khoảng $3.150/tháng.
Prompt thật dài khoảng 120 token, nên con số thực tế gần $3.750/tháng.
Đây là ước tính có giả định. Notebook không đo được giá của Groq.
- Chọn: 3b. Encoder MiniLM + LogReg. Số liệu dẫn chứng:
  + F1 test 0,961, cao nhất trên 775 tin, với precision = recall = 0,961.
  + p95 32 ms, đạt SLA.
  + Nằm trên đường Pareto.
  + Bền hơn BoW: đúng 66% trên 50 case khó, so với 56% của BoW + MLP.
- Đánh đổi chấp nhận:
  + Biên độ trễ hẹp: p95 32 ms (đo trên Colab) đã khá sát mốc 50 ms.
  + Tốn hạ tầng: Thông lượng khoảng 42 tin/s mỗi máy (CPU), nên cần ít nhất 3 máy cho tải trung bình. Nếu giờ cao điểm gấp 3 lần tải trung bình, cần khoảng 8–10 máy, hoặc dùng GPU, hoặc chuyển encoder sang ONNX/lượng tử hoá. Ngoài ra phải vận hành encoder 22,7 triệu tham số.
  + Phương án dự phòng: Nếu không có hạ tầng, dùng BoW + MLP. Chỉ 0,26 ms/tin và 53.000 tin/s nên một máy là đủ, nhưng chấp nhận F1 test thấp hơn (0,913) và chỉ đúng 56% case khó.

## 7. AI log và điều đã học
Phần	Em hỏi AI gì	AI trả lời/viết gì	Em đã sửa / kiểm chứng gì
1–3	Nhờ giải thích khái niệm: BoW, sigmoid, gradient descent, precision/recall/F1, MLP, cosine, kNN	Giải thích khái niệm kèm ví dụ số	Tự viết code các TODO. Khi test báo lỗi thì hỏi nguyên nhân (ví dụ return trống, sai ngoặc trong nn.Sequential), tự sửa và chạy lại test
4	Hỏi cách xử lý lỗi 429 và 400 của Gemini	Thêm sleep và tăng base_delay; đổi reasoning_effort sang low	Model lớn vẫn hết quota ngày, nên chuyển sang Groq
4–5	Nhờ gợi ý dự đoán và khung trả lời cho Câu hỏi 4, 5, 6 và phần phản ánh	Đưa ra các giả thuyết để chọn và khung trả lời	Tự chọn giả thuyết, điền số liệu của mình, viết lại bằng lời của mình
Báo cáo	Nhờ soạn báo cáo cá nhân từ notebook, lab1_report.json và ảnh tính tay	Bản nháp báo cáo này	Đối chiếu lại từng số với output notebook và JSON; sửa chỗ "model tụt nhiều nhất" cho khớp số liệu (BoW + MLP chứ không phải BoW + LogReg)

- Điều em hiểu ra hôm nay:
  + Điểm cao trên tập test cùng phân phối với train chưa chứng minh model "hiểu" spam. BoW + MLP có F1 SUB 0,947 nhưng chỉ đúng 56% case khó, vì nó học thuộc từ khoá.
  + Chọn model cho sản phẩm không chỉ nhìn F1. Phải cân cả độ trễ p95, thông lượng, chi phí, và độ nhạy của phép đo: trên 40 tin, chênh nhau dưới 0,05 F1 coi như hoà.
  + LLM mạnh nhất ở spam kiểu mới, nhưng chậm và tốn kém. Vì vậy nó hợp làm lớp xử lý tin lưỡng lự trong router hơn là làm bộ lọc chính.
