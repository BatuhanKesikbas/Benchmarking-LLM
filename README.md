# Üniversiteye Giriş Türkçe Sınavı LLM Değerlendirme Sistemi

Bu proje, farklı büyük dil modellerinin (LLM) **Türkçe sorulara verdiği cevapların doğruluk ve kalite performansını** ölçer.

# Sürüm
Önerilen Python Sürümü: 3.09 - 3.12

# Gönderilen Soruların Kontrolü
llm_sent_questions.log dosyasında her llm'e giden soru metini ve cevapları bulunmaktadır. (Max 1000 karakter)

# .env dosyası oluşturup api key girin
OPENROUTER_API_KEY=

# OpenAI Modelleri
openai_new       ->  GPT-5.2
openai_4o        ->  GPT-4o
openai_mini      ->  GPT-4o Mini

# Anthropic Modelleri
claude_sonnet    ->  Claude 4.5 Sonnet
claude_new_haiku ->  Claude 4.5 Haiku
claude_old_haiku ->  Claude 3 Haiku

# Google Gemini Modelleri
gemini_pro       ->  Gemini 2.5 Pro
gemini_new_flash ->  Gemini 2.5 Flash
gemini_old_flash ->  Gemini 2.0 Flash

# Grok (xAI) Modelleri
grok_new         ->  Grok 4.1 Fast
grok_old         ->  Grok 3
grok_old_mini    ->  Grok 3 Mini

# DeepSeek Modelleri
deepseek_v3      ->  DeepSeek V3
deepseek_r1      ->  DeepSeek R1
deepseek_32      ->  DeepSeek V3.2

# Test Amaçlı
mock             ->  Dummy Model (API harcamaz)


## 🚀 Kurulum

```bash
pip install -r requirements.txt

# Pdf dosyasının yılını 2018-2024 değiştirebilirsiniz veya YGS sınavı için ygs-2016.pdf 

# Random şık denemesi
py main.py --pdf documents\tyt_yks_2024.pdf --models "mock"  --out "tyt_sonuclar2024.csv"

# Tüm LLM'ler
py main.py --pdf documents\tyt_yks_2024.pdf --models "openai_new,openai_mini,openai_4o,claude_old_haiku,claude_new_haiku,claude_sonnet,gemini_old_flash,gemini_new_flash,gemini_pro,grok_old,grok_old_mini,grok_new,deepseek_v3,deepseek_r1,deepseek_32,mock" --out "tyt_sonuclar2024.csv"
```
# Örnek
Projeye dahil edilen model haritası ve kısayol anahtarları aşağıdadır. `--models` parametresinde bu anahtarları kullanın. (Örnek: --models "openai_new, claude_sonnet")
