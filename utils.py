import pandas as pd
from datetime import datetime
import re
from typing import List, Dict, Any
import csv
import fitz  # pymupdf
import requests
import time
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None
# ==============================================================================
# 1. TEMEL YARDIMCI FONKSİYONLAR
# ==============================================================================

def load_dataset(path: str = "dataset.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decode failed; retrying with windows-1254...")
        return pd.read_csv(path, encoding="windows-1254")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at path: {path}")

def format_prompt(question: str, choices: str, include_explanation: bool = False) -> str:
    if not question or len(question) < 5:
        return "SORU METNİ OKUNAMADI."
        
    base_prompt = f"Soru: {question}\nŞıklar: {choices}\n\n"
    if include_explanation:
        return base_prompt + (
            "Bu çoktan seçmeli sorunun doğru cevabını bul.\n "
            "Açıklama YAPMA veya çok az cümle ile açıkla.\n" 
            "Seçenekler sadece A-E arası olacak.\n"
            "Cevabını şu formatta ver:\nSeçenek: <Sadece Harf>\nAçıklama: <Kısa ve mantıklı açıklama>"
        )
    else:
        return base_prompt + "Sadece doğru HARF'i ver (A/B/C/D/E)."

def normalize_letter_response(text: str) -> str:
    if not text: return ""
    text = text.strip()
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    
    explicit_pattern = re.search(r'(?:Seçenek|Cevap|Answer|Choice)\s*[:\-]?\s*([A-E])\b', text, re.IGNORECASE)
    if explicit_pattern:
        return explicit_pattern.group(1).upper()

    start_pattern = re.search(r'^[\(\s]*([A-E])[\)\.]', text, re.MULTILINE)
    if start_pattern:
        return start_pattern.group(1).upper()    
    
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    
    last_ditch = re.search(r'\b([A-E])\s*[\.\)]?$', text)
    if last_ditch:
        return last_ditch.group(1).upper()

    return ""

def write_llm_sent_questions_log(results: List[Dict], source_filename: str="Bilinmiyor"):
    log_file = "llm_sent_questions.log"
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== TEST LOGU ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===\n\n")
            f.write(f"📄 KAYNAK DOSYA: {source_filename}\n") 
            
            f.write("="*60 + "\n\n")
            for r in results:
                f.write(f"SORU ID: {r['id']} | MODEL: {r['model']}\n")
                q_text = r.get("question", "Metin yok")
                f.write(f"Metin: {q_text}\n") 
                f.write(f"Tahmin: {r.get('predicted_letter')} | Doğru: {r.get('correct_choice')}\n")
                f.write("-" * 50 + "\n")
        print(f"📝 Log dosyası güncellendi: {log_file}")
    except Exception as e:
        print(f"Log yazma hatası: {e}")

# ==============================================================================
# 2. OTOMATİK FİYAT ÇEKME (OPENROUTER API) 💰
# ==============================================================================
def fetch_openrouter_pricing():
    print("🌍 OpenRouter güncel fiyat listesi indiriliyor...")
    url = "https://openrouter.ai/api/v1/models"
    
    pricing_map = {
        
    }
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json().get("data", [])
            print(f"✅ {len(data)} adet modelin fiyat bilgisi çekildi.")
            for model in data:
                mid = model.get("id")
                try:
                    p_in = float(model.get("pricing", {}).get("prompt", 0)) * 1_000_000
                    p_out = float(model.get("pricing", {}).get("completion", 0)) * 1_000_000
                    pricing_map[mid] = {"in": p_in, "out": p_out}
                except:
                    continue
            print("✅ Fiyatlar güncellendi.")
        else:
            print("⚠️ Fiyat sunucusuna erişilemedi, varsayılanlar kullanılacak.")
    except Exception as e:
        print(f"⚠️ Fiyat sunucusuna erişilemedi: {e}")
        
    return pricing_map

# ==============================================================================
# 3. GRAFİK OLUŞTURUCU (YENİ EKLENDİ) 📊
# ==============================================================================
def create_performance_chart(summary_df, output_path):
    if plt is None or sns is None:
        print("⚠️ Grafik çizilemedi: matplotlib veya seaborn yüklü değil.")
        return

    try:
        # Veriyi sırala (Başarılı olan en üstte)
        df_sorted = summary_df.sort_values(by="accuracy", ascending=True)
        
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        # Renk paleti oluştur
        colors = sns.color_palette("viridis", len(df_sorted))
        
        # Bar plot çiz
        bars = plt.barh(df_sorted.index, df_sorted["accuracy"], color=colors)
        
        plt.title("Model Doğruluk Oranları (%)", fontsize=14, fontweight='bold')
        plt.xlabel("Doğruluk (%)")
        plt.ylabel("Modeller")
        plt.xlim(0, 100)
        
        # Değerleri barların ucuna yaz
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, 
                     bar.get_y() + bar.get_height()/2, 
                     f'%{width:.1f}', 
                     va='center', fontsize=10, fontweight='bold', color='black')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"📊 Grafik oluşturuldu: {output_path}")
        plt.close() # Hafızadan sil
    except Exception as e:
        print(f"⚠️ Grafik oluşturulurken hata: {e}")

# ==============================================================================
# 3. SONUÇ KAYDETME VE RAPORLAMA (DÜZELTİLDİ) ✅
# ==============================================================================
def save_results(results: list, out_path: str = "results.csv", source_filename: str = "Bilinmiyor") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = out_path.replace(".csv", f"_{ts}.csv")
    chart_path = out_path.replace(".csv", f"_{ts}_chart.png") # Grafik dosya yolu
    
    df = pd.DataFrame(results)

    # --- KRİTİK DÜZELTME BAŞLANGICI ---

    if not df.empty and "usage" in df.columns:
        # Eğer usage sütunu varsa ve içi doluysa parçala
        df["input_tokens"] = df["usage"].apply(lambda x: x.get("input", 0) if isinstance(x, dict) else 0)
        df["output_tokens"] = df["usage"].apply(lambda x: x.get("output", 0) if isinstance(x, dict) else 0)
    else:
        df["input_tokens"] = 0
        df["output_tokens"] = 0
    
    # 1. CSV YAZ (Standart)
    fieldnames = ["id", "model", "predicted", "correct_choice", "is_correct", "latency_sec", "input_tokens", "output_tokens", "explanation"]
    
    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        last_qid = None
        

        for r in results:
            if last_qid is not None and r.get("id") != last_qid: f.write("\n")
            
            usage = r.get("usage", {"input": 0, "output": 0})
            row_data = r.copy()
            row_data["input_tokens"] = usage.get("input", 0)
            row_data["output_tokens"] = usage.get("output", 0)
            
            writer.writerow(row_data)
            last_qid = r.get("id")

    print(f"\n✅ Sonuçlar kaydedildi: {save_path}")
    
    # 2. RAPOR OLUŞTUR
    if not df.empty and "is_correct" in df.columns:
        LIVE_PRICING = fetch_openrouter_pricing()
        
        summary = df.groupby("model").agg(
            total=("is_correct", "count"),
            correct=("is_correct", "sum"),
            avg_latency=("latency_sec", "mean"),
            sum_input=("input_tokens", "sum"),  
            sum_output=("output_tokens", "sum")  
        )
        summary["accuracy"] = (summary["correct"] / summary["total"] * 100).round(2)
        
        create_performance_chart(summary, chart_path)

        lines = []
        lines.append("\n" + "="*95)
        lines.append(f"📄 ANALİZ EDİLEN DOSYA: {source_filename}")
        lines.append(f"{'MODEL PERFORMANS VE MALİYET RAPORU':^95}")
        lines.append("="*95)
        lines.append(f"{'Model':<20} | {'Doğruluk':<10} | {'Süre (s)':<10} | {'Girdi Tok.':<12} | {'Maliyet ($)':<12}")
        lines.append("-" * 95)
        
        MODEL_MATCH_HINTS = {
            # OpenAI
            "openai_new": "gpt-5.2", 
            "openai_4o": "gpt-4o", 
            "openai_mini": "gpt-4o-mini",
            
            # Anthropic
            "claude_sonnet": "claude-4.5-sonnet", 
            "claude_new_haiku": "claude-4.5-haiku", 
            "claude_old_haiku": "claude-3-haiku",
            
            # Gemini
            "gemini_pro": "gemini-2.5-pro", 
            "gemini_new_flash": "gemini-2.5-flash", 
            "gemini_old_flash": "gemini-2.0-flash",
            
            # Grok
            "grok_old": "grok-3", 
            "grok_old_mini": "grok-3-mini", 
            "grok_new": "grok-4.1-fast",
            
            # DeepSeek
            "deepseek_v3": "deepseek-v3", 
            "deepseek_r1": "deepseek-r1", 
            "deepseek_32": "deepseek-v3.2"
        }

        for model_key, row in summary.iterrows():
            price_in, price_out = 0.0, 0.0
            
            if "free" in model_key or "mock" in model_key:
                price_in, price_out = 0.0, 0.0
            else:
                search_term = MODEL_MATCH_HINTS.get(model_key, model_key)
                for api_id, prices in LIVE_PRICING.items():
                    if search_term in api_id:
                        price_in = prices['in']
                        price_out = prices['out']
                        break
            
            cost = ((row['sum_input'] * price_in) + (row['sum_output'] * price_out)) / 1_000_000
            
            lines.append(f"{model_key:<20} | {row['accuracy']:>9.2f}% | {row['avg_latency']:>9.3f}s | {int(row['sum_input']):<12} | ${cost:>11.5f}")
            
        lines.append("="*95 + "\n")
        
        lines.append("=== ÖZET SIRALAMA ===")
        sorted_acc = summary.sort_values(by="accuracy", ascending=False)
        for m, r in sorted_acc.iterrows():
            lines.append(f"{m:<20}: %{r['accuracy']:.2f}")
        lines.append("====================\n")
        
        report_text = "\n".join(lines)
        print(report_text)
        
        try:
            with open(save_path, "a", encoding="utf-8-sig") as f:
                f.write("\n\n" + report_text)
        except: pass

    return save_path

# ==============================================================================
# 4. GELİŞMİŞ PDF AYIKLAMA (GÜNCELLENMİŞ MARJİNLER)
# ==============================================================================

def clean_text_block(text: str) -> str:
    """Metin içindeki gürültüleri temizler."""
    noise_patterns = [
        r"^\d+\s*$",                        # Sayfa numarası
        r"Diğer sayfaya geçiniz",
        r"Sınavda uyulacak kurallar",
        r"Ö\s*S\s*Y\s*M",
        r"hiçbir kişi, kurum veya kuruluş", 
        r"tarafından kullanılamaz",         
    ]
    clean = text.strip()
    for pat in noise_patterns:
        if re.search(pat, clean, re.IGNORECASE):
            return ""
    return clean

def get_sorted_blocks(page):
    """
    Blokları okuma sırasına göre dizer (Sol Sütun -> Sağ Sütun).
    """
    blocks = page.get_text("blocks")
    width = page.rect.width
    mid_x = width / 2
    
    valid_blocks = []
    for b in blocks:
        if b[6] == 0: # Sadece metin blokları
            txt = clean_text_block(b[4])
            if txt:
                # (x0, y0, text)
                valid_blocks.append((b[0], b[1], txt))
    
    # Sol ve Sağ sütun ayrımı
    left_col = [b for b in valid_blocks if b[0] < mid_x]
    right_col = [b for b in valid_blocks if b[0] >= mid_x]
    
    # Yukarıdan aşağıya sırala (y0 koordinatına göre)
    left_col.sort(key=lambda x: x[1])
    right_col.sort(key=lambda x: x[1])
    
    # Listeleri birleştir
    return [b[2] for b in left_col + right_col]

def extract_answer_key(doc) -> Dict[int, str]:
    key_map = {}
    for page_idx in range(len(doc)-1, max(-1, len(doc)-6), -1):
        text = doc[page_idx].get_text()
        matches = re.findall(r'(\d{1,2})\s*[\.\-]?\s*([A-E])', text)
        temp_map = {}
        for num, choice in matches:
            qid = int(num)
            if 1 <= qid <= 40:
                if qid not in temp_map: temp_map[qid] = choice.upper()
        
        if len(temp_map) > 20: 
            key_map.update(temp_map)
            print(f"✅ Cevap anahtarı bulundu (Sayfa {page_idx+1})")
            break
    return key_map

def extract_mcq_from_pdf(pdf_path: str) -> pd.DataFrame:
    doc = fitz.open(pdf_path)
    answer_key = extract_answer_key(doc)
    
    extracted_questions = []
    
    # --- DEĞİŞKENLER ---
    current_qid = None
    current_text_parts = []
    
    # --- ORTAK PARAGRAF YÖNETİMİ ---
    shared_buffer = []             
    active_range = (0, 0)          
    collecting_mode = False         
    
    # --- REGEXLER ---
    re_q_start = re.compile(r'^(\d+)\s*[\.\-\)]\s*(.*)', re.DOTALL)
    
  
    re_range_trigger = re.compile(r'(?:^|\s)(\d{1,2})\s*[-–ve,]\s*(\d{1,2})[\.\s]*soru', re.IGNORECASE)

    in_turkish_section = False
    
    for page in doc:
        blocks = get_sorted_blocks(page) 
        
        for text in blocks:
            # 1. BÖLÜM KONTROLÜ
            if "TÜRKÇE TESTİ" in text:
                in_turkish_section = True
                continue
            if "SOSYAL BİLİMLER" in text or "TEMEL MATEMATİK" in text:
                if in_turkish_section:
                    in_turkish_section = False
                continue
            if not in_turkish_section: continue

            # 2. ORTAK SORU YÖNERGESİ (TRIGGER)
            range_match = re_range_trigger.search(text)
            if range_match:
                # Eğer önceki soru hala açıksa kapat
                if current_qid is not None:
                    extracted_questions.append({
                        "id": current_qid,
                        "question": "\n".join(current_text_parts),
                        "choices": "Metin içinde",
                        "correct_choice": answer_key.get(current_qid, "")
                    })
                    current_qid = None
                    current_text_parts = []

                # Yeni modu aç
                s_q, e_q = int(range_match.group(1)), int(range_match.group(2))
                active_range = (s_q, e_q)
                
                # Bufferı başlat (Yönerge metnini de ekle)
                shared_buffer = [text]
                collecting_mode = True 
                continue

            # 3. YENİ SORU BAŞLANGICI
            match = re_q_start.match(text)
            if match:
                new_id = int(match.group(1))
                
                if 1 <= new_id <= 40:
                    # Eski Soruyu Kaydet
                    if current_qid is not None:
                        extracted_questions.append({
                            "id": current_qid,
                            "question": "\n".join(current_text_parts),
                            "choices": "Metin içinde",
                            "correct_choice": answer_key.get(current_qid, "")
                        })

                    # Yeni Soruya Geç
                    current_qid = new_id
                    q_body = match.group(2)
                    
                    # Soru numarası geldiği için toplama modu biter
                    collecting_mode = False 
                    
                    # Eğer bu soru, aktif aralıktaysa
                    if active_range[0] <= new_id <= active_range[1]:
                        # Tamponu birleştir ve sorunun BAŞINA ekle
                        full_context = "\n".join(shared_buffer)
                        current_text_parts = [full_context, q_body]
                    else:
                        # Normal soru
                        current_text_parts = [q_body]
                        
                        # Aralık dışına çıktıysak bufferı artık temizle
                        if new_id > active_range[1]:
                            shared_buffer = []
                            active_range = (0, 0)
                else:
                    # Soru numarası değil (örn: I. II. maddeler)
                    if collecting_mode:
                        shared_buffer.append(text)
                    elif current_qid is not None:
                        current_text_parts.append(text)
            
            # 4. NUMARA YOK (DÜZ METİN)
            else:
                if collecting_mode:
                    shared_buffer.append(text)
                
                elif current_qid is not None:
                    # Mevcut sorunun devamı
                    current_text_parts.append(text)
    
    # SON SORUYU KAYDET
    if current_qid is not None:
        extracted_questions.append({
            "id": current_qid,
            "question": "\n".join(current_text_parts),
            "choices": "Metin içinde",
            "correct_choice": answer_key.get(current_qid, "")
        })

    # DataFrame Dönüşü
    df = pd.DataFrame(extracted_questions)
    final_data = []
    
    found_ids = {r['id']: r for r in df.to_dict('records')} if not df.empty else {}
    for i in range(1, 41):
        if i in found_ids:
            final_data.append(found_ids[i])
        else:
            final_data.append({
                "id": i, 
                "question": "SORU BULUNAMADI (Parsing Hatası)", 
                "choices": "", 
                "correct_choice": answer_key.get(i, "")
            })
            
    return pd.DataFrame(final_data)