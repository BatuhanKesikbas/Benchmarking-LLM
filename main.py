# main.py
import argparse
import time
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
import os

from utils import (
    load_dataset, format_prompt, save_results, normalize_letter_response, 
    extract_mcq_from_pdf, write_llm_sent_questions_log
)
from models_api import (
    call_gpt52, call_gpt4o_mini, call_gpt4o,
    call_claude_3_haiku, call_claude_45_haiku, call_claude_45_sonnet,
    call_gemini_20_flash, call_gemini_25_flash, call_gemini_25_pro,
    call_grok_3, call_grok_3_mini, call_grok_41_fast,
    call_deepseek_v3, call_deepseek_r1, call_deepseek_v32,
    mock_model
)

# MODEL HARİTASI
MODEL_MAP = {
   # OpenAI
    "openai_new":  call_gpt52,       # GPT-5.2
    "openai_mini": call_gpt4o_mini,  # GPT-4o Mini
    "openai_4o":   call_gpt4o,       # GPT-4o

    # Anthropic
    "claude_old_haiku": call_claude_3_haiku,   # Claude 3 Haiku
    "claude_new_haiku": call_claude_45_haiku,  # Claude 4.5 Haiku
    "claude_sonnet":    call_claude_45_sonnet, # Claude 4.5 Sonnet

    # Gemini
    "gemini_old_flash": call_gemini_20_flash, # Gemini 2.0 Flash
    "gemini_new_flash": call_gemini_25_flash, # Gemini 2.5 Flash
    "gemini_pro":       call_gemini_25_pro,   # Gemini 2.5 Pro

    # Grok
    "grok_old": call_grok_3,            # Grok 3
    "grok_old_mini": call_grok_3_mini,  # Grok 3 Mini
    "grok_new": call_grok_41_fast,      # Grok 4.1 Fast


    # DeepSeek
    "deepseek_v3":   call_deepseek_v3,  # DeepSeek V3
    "deepseek_r1":   call_deepseek_r1,  # DeepSeek R1
    "deepseek_32":   call_deepseek_v32, # DeepSeek V3.2

    "mock": mock_model  # Mock (Random şık seçiyor)
}
def evaluate(dataset_path: str,
             model_keys: List[str],
             out_path: str = "results.csv",
             per_example_sleep: float = 1.0, 
             pdf_path: str = None) -> List[Dict[str, Any]]:
    source_name = os.path.basename(pdf_path) if pdf_path else os.path.basename(dataset_path)
    # Veri Yükleme
    if pdf_path:
        print(f"📄 PDF işleniyor: {pdf_path}")
        df = extract_mcq_from_pdf(pdf_path)
        dataset = df.to_dict(orient="records")
        print(f"✅ PDF'den {len(dataset)} satır veri çıkarıldı.")
    else:
        print(f"📂 CSV yükleniyor: {dataset_path}")
        df = load_dataset(dataset_path)
        dataset = df.to_dict(orient="records")

    results = []
    print(f"🚀 Test Başlıyor: {len(dataset)} soru x {len(model_keys)} model")

    # İlerleme çubuğu (tqdm)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="İlerliyor"):
        tqdm.write("\n")
        qid = row.get("id")
        question = str(row.get("question", ""))
        choices = str(row.get("choices", ""))
        correct = str(row.get("correct_choice", "")).strip().upper()

        if len(question) < 5: continue

        prompt = format_prompt(question, choices, include_explanation=True)

        for mk in model_keys:
            mk_lower = mk.strip().lower()
            
            # Model eşleştirme
            fn = MODEL_MAP.get(mk_lower)
            if not fn:
                for base_key, func in MODEL_MAP.items():
                    if base_key in mk_lower:
                        fn = func
                        break
            
            if fn is None:
                tqdm.write(f"⚠️ Model bulunamadı: '{mk}'")
                continue

            try:
                # API Çağrısı (4 değer dönüyor: Harf, Süre, Açıklama, Token)
                pred_letter, latency, explanation, usage = fn(prompt)
                
                if not pred_letter and not explanation:
                    explanation = "ERROR: No response"
                    
            except ValueError:
                try:
                    pred_letter, latency, explanation = fn(prompt)
                    usage = {"input": 0, "output": 0}
                except Exception as e:
                    pred_letter, latency, explanation, usage = "", 0.0, f"Error: {e}", {"input": 0, "output": 0}
            except Exception as e:
                pred_letter, latency, explanation, usage = "", 0.0, f"Critical Error: {e}", {"input": 0, "output": 0}

            pred_letter = normalize_letter_response(pred_letter)
            is_correct = (pred_letter == correct) if correct else False

            icon = "✅" if is_correct else "❌"
            if not correct: icon = "❓" # Cevap anahtarı yoksa
            
            msg = f"{icon} [{mk_lower:<10}] Soru {qid:<2}: Tahmin={pred_letter} (Doğru: {correct}) | {latency:.2f}s"
            tqdm.write(msg)

            results.append({
                "id": qid,
                "model": mk_lower,
                "question": question,
                "choices": choices,
                "correct_choice": correct,
                "predicted_letter": pred_letter,
                "is_correct": is_correct,
                "latency_sec": round(latency, 3),
                "explanation": explanation,
                "usage": usage
            })
            
            if per_example_sleep > 0:
                time.sleep(per_example_sleep)

    save_results(results, out_path, source_filename=source_name)      
    write_llm_sent_questions_log(results, source_filename=source_name)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Benchmark")
    parser.add_argument("--dataset", type=str, default="dataset.csv")
    parser.add_argument("--models", type=str, default="openai,gemini,groq",
                        help="Modelleri virgülle ayırın")
    parser.add_argument("--out", type=str, default="results.csv")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--pdf", type=str, default=None)
    
    args = parser.parse_args()
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    
    evaluate(args.dataset, model_keys, out_path=args.out, per_example_sleep=args.sleep, pdf_path=args.pdf)