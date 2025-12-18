import time
import os
import functools
from openai import OpenAI
from dotenv import load_dotenv
from utils import normalize_letter_response

load_dotenv(override=True)



client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"), 
)

EXTRA_HEADERS = {
    "HTTP-Referer": os.getenv("YOUR_SITE_URL", "https://localhost"),
    "X-Title": os.getenv("YOUR_SITE_NAME", "TR-LLM-Benchmark"),
}

# ==============================================================================
# 2. HATA YÖNETİMİ (DECORATOR)
# ==============================================================================
def handle_api_errors(func):
    """
    Tüm API çağrılarını sarar ve hataları standart formatta döndürür.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prompt = kwargs.get('prompt') or (args[0] if args else "")
        
        # Giriş Kontrolü
        if not prompt or len(prompt.strip()) < 5:
            return "", 0.0, "[ERROR_INPUT] Soru metni boş veya çok kısa."


        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            
            letter, _, explanation, usage = result
            
            latency = time.time() - start_time
            
            return letter, latency, explanation, usage
        except Exception as e:
            latency = time.time() - start_time
            error_msg = str(e).lower()
            
            # Hata Kodlarını İnsan Diline Çevir
            if "402" in error_msg: code = "[ERROR_BALANCE] (Kredi Yetersiz)"
            elif "429" in error_msg: code = "[ERROR_RATE_LIMIT] (Çok Hızlı İstek)"
            elif "401" in error_msg: code = "[ERROR_AUTH] (API Anahtarı Geçersiz)"
            elif "timeout" in error_msg: code = "[ERROR_TIMEOUT] (Zaman Aşımı)"
            elif "context_length" in error_msg: code = "[ERROR_CONTEXT] (Metin Çok Uzun)"
            elif "connection" in error_msg: code = "[ERROR_NETWORK] (İnternet/Bağlantı Hatası)"
            else: code = f"[ERROR_UNKNOWN] {type(e).__name__}"

            # Hatanın ilk 200 karakterini döndür
            return "", latency, f"{code}: {str(e)[:200]}"
            
    return wrapper

# ==============================================================================
# 3. GENEL ÇAĞRI FONKSİYONU
# ==============================================================================
def _call_openrouter_generic(prompt: str, model_id: str):
    """
    Tüm modellerin arkasında çalışan asıl işçidir.
    """
    # Sisteme rol vererek başarı oranını artırıyoruz
    messages = [
        {"role": "system", "content": "Sen Üniversite giriş sınavına giren, mantıksal çıkarım yeteneği yüksek bir öğrencisin. Soruları dikkatle çöz. Cevabı 'Seçenek: X' formatında ver. Açıklamanı kısaca yaz."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0, 
        max_tokens=512,         
        extra_headers=EXTRA_HEADERS,
        timeout=60    
    )
    
    text = response.choices[0].message.content.strip()
    usage = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens
    }

    return normalize_letter_response(text), time.time(), text, usage

# ==============================================================================
# 4. MODEL WRAPPERLARI 
# ==============================================================================


# ==============================================================================
# 1. OPENAI (Nesil: 3.5 vs 4o | Boyut: Mini vs Standart)
# ==============================================================================
@handle_api_errors
def call_gpt52(prompt: str):     
    return _call_openrouter_generic(prompt, "openai/gpt-5.2")
@handle_api_errors
def call_gpt4o_mini(prompt: str): 
    return _call_openrouter_generic(prompt, "openai/gpt-4o-mini")
@handle_api_errors
def call_gpt4o(prompt: str):      
    return _call_openrouter_generic(prompt, "openai/gpt-4o")

# ==============================================================================
# 2. ANTHROPIC (Nesil: 3 vs 3.5 | Boyut: Haiku vs Sonnet)
# ==============================================================================
@handle_api_errors
def call_claude_3_haiku(prompt: str):  
    return _call_openrouter_generic(prompt, "anthropic/claude-3-haiku")
@handle_api_errors
def call_claude_45_haiku(prompt: str): 
    return _call_openrouter_generic(prompt, "anthropic/claude-haiku-4.5")
@handle_api_errors
def call_claude_45_sonnet(prompt: str):
    return _call_openrouter_generic(prompt, "anthropic/claude-sonnet-4.5")

# ==============================================================================
# 3. GEMINI (Nesil: 1.5 vs 2.0 | Boyut: Flash vs Pro)
# ==============================================================================
@handle_api_errors
def call_gemini_20_flash(prompt: str): 
    return _call_openrouter_generic(prompt, "google/gemini-2.0-flash-001")
@handle_api_errors
def call_gemini_25_flash(prompt: str): 
    return _call_openrouter_generic(prompt, "google/gemini-2.5-flash")
@handle_api_errors
def call_gemini_25_pro(prompt: str):  
    return _call_openrouter_generic(prompt, "google/gemini-2.5-pro")


# ==============================================================================
# 4. GROK (Nesil: Beta vs 2)
# ==============================================================================
@handle_api_errors
def call_grok_3_mini(prompt: str):
    return _call_openrouter_generic(prompt, "x-ai/grok-3-mini")
@handle_api_errors
def call_grok_3(prompt: str):   
    return _call_openrouter_generic(prompt, "x-ai/grok-3")
@handle_api_errors
def call_grok_41_fast(prompt: str):  
    return _call_openrouter_generic(prompt, "x-ai/grok-4.1-fast")


# ==============================================================================
# 5. DEEPSEEK (Model: V3 vs R1 vs R1-Lite)
# ==============================================================================
@handle_api_errors
def call_deepseek_v3(prompt: str):   
    return _call_openrouter_generic(prompt, "deepseek/deepseek-chat")
@handle_api_errors
def call_deepseek_r1(prompt: str):   
    return _call_openrouter_generic(prompt, "deepseek/deepseek-r1")
@handle_api_errors
def call_deepseek_v32(prompt: str): 
    return _call_openrouter_generic(prompt, "deepseek/deepseek-v3.2")

# ==============================================================================
# 5. MOCK MODEL (TEST İÇİN)
# ==============================================================================
@handle_api_errors
def mock_model(prompt: str, **kwargs):
    import random
    time.sleep(0.1)
    return random.choice(["A","B","C","D","E"]), 0.01, "Mock cevap", {"input": 10, "output": 5}