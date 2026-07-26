# 🐾 دليل تشغيل مشروع PetNutri (بالعربية)

دليل تفصيلي خطوة بخطوة لتشغيل وصيانة تطبيق PetNutri — مساعد ذكي لتغذية الحيوانات الأليفة يعتمد على تقنية RAG (الاسترجاع مع التوليد).

---

## 1. البرامج المطلوبة

- **Python** إصدار 3.10 أو أحدث.
- **Git** لاستنساخ المشروع.
- محرر أكواد مثل VS Code (اختياري).
- مفتاح API من موقع [OpenRouter](https://openrouter.ai) (يوجد باقة مجانية).
- حساب على [GitHub](https://github.com) و [Streamlit Community Cloud](https://share.streamlit.io) للنشر العام.

---

## 2. هيكل المجلدات

```
petnutri/
├── 01_documents.py              # تحميل الملفات + البيانات الوصفية
├── 02_preprocessing.py          # تنظيف النصوص
├── 03_chunking.py               # تقسيم النصوص إلى مقاطع (chunks)
├── 04_vector_representation.py  # تمثيل النصوص كمتجهات (embeddings)
├── 05_create_chroma_store.py    # بناء/تحميل قاعدة بيانات ChromaDB
├── 06_retrieve_context.py       # استرجاع السياق (بحث هجين BM25 + دلالي)
├── 07_prompting.py              # تصنيف السؤال + بناء الطلب + استدعاء النموذج
├── streamlit_app.py             # واجهة المستخدم
├── config.py                    # الإعدادات المركزية
├── pipeline_utils.py            # أداة مساعدة لاستيراد الملفات المرقّمة
├── data/nutrition/              # ملفات المعرفة بصيغة Markdown
├── assets/style.css             # التنسيق البصري
├── database/chroma_db/          # قاعدة بيانات المتجهات (تُبنى تلقائيًا)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 3. تثبيت المتطلبات (Dependencies)

```bash
git clone <رابط-المستودع>
cd petnutri
python -m venv .venv
source .venv/bin/activate      # على ويندوز: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. إعداد ملف البيئة (.env)

1. انسخ الملف النموذجي:
   ```bash
   cp .env.example .env
   ```
2. افتح `.env` وضع مفتاحك الحقيقي:
   ```
   OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```
3. **لا ترفع أبدًا** ملف `.env` الحقيقي إلى GitHub — الملف مستثنى تلقائيًا عبر `.gitignore`.

---

## 5. تشغيل التطبيق محليًا

```bash
streamlit run streamlit_app.py
```

سيفتح المتصفح تلقائيًا على `http://localhost:8501`.

---

## 6. التشغيل داخل Google Colab

```python
!git clone <رابط-المستودع>
%cd petnutri
!pip install -r requirements.txt -q

import os
os.environ["OPENROUTER_API_KEY"] = "ضع_مفتاحك_هنا"

!pip install -q pyngrok
from pyngrok import ngrok
!streamlit run streamlit_app.py &>/content/log.txt &
public_url = ngrok.connect(8501)
print(public_url)   # الرابط العام للوصول من أي متصفح
```

---

## 7. بناء قاعدة بيانات ChromaDB

- يتم البناء **تلقائيًا** عند أول تشغيل (أو عند تغيّر محتوى ملفات المعرفة).
- يمكنك أيضًا الضغط على زر **🔄 Rebuild Database** في الشريط الجانبي لإعادة البناء يدويًا في أي وقت.
- آلية العمل: يقوم `05_create_chroma_store.py` بحساب "بصمة" (hash) لمحتوى الملفات، ويقارنها بالبصمة المحفوظة في `database/source_hash.json`؛ إن تغيّرت، تُحذف المجموعة القديمة ويُعاد البناء من الصفر — وإلا يُعاد استخدام القاعدة الحالية دون أي عمل إضافي.

---

## 8. تشغيل واجهة Streamlit

بعد تشغيل الأمر في الخطوة 5:
- الصفحة الرئيسية تعرض زر **Start AI Consultation**.
- بعد الضغط عليه تنتقل إلى صفحة المحادثة حيث يمكنك كتابة سؤالك في الصندوق السفلي والضغط على **Consult AI**.
- الشريط الجانبي يعرض حالة قاعدة المعرفة (عدد الملفات، عدد المقاطع، نموذج التمثيل، حالة القاعدة، تاريخ آخر بناء).

---

## 9. نشر التطبيق للوصول العام (Public Deployment)

1. ادفع المشروع إلى مستودع GitHub (بدون `.env` أو أي مفتاح حقيقي).
2. اذهب إلى [share.streamlit.io](https://share.streamlit.io) وأنشئ تطبيقًا جديدًا يشير إلى `streamlit_app.py` في مستودعك.
3. من قائمة **Manage app → Secrets** أضف:
   ```toml
   OPENROUTER_API_KEY = "مفتاحك_الحقيقي"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```
4. اضغط **Deploy**. ستحصل على رابط عام بصيغة `https://xxxx.streamlit.app` يمكن لأي شخص فتحه دون تسجيل دخول.

### قيود لا يمكن تجاوزها (من جهة مزوّد الاستضافة/الـ API وليست من كود التطبيق)

- الخطة المجانية من Streamlit Cloud "تُنيّم" التطبيق بعد فترة خمول، فيستغرق أول تحميل بعد ذلك بضع ثوانٍ إضافية.
- مفاتيح OpenRouter المجانية لها **حد أقصى لعدد الطلبات** في الدقيقة/اليوم؛ الاستخدام العام المكثف قد يواجه رسائل خطأ 429 (طلبات كثيرة).
- تخزين ChromaDB على قرص الحاوية في Streamlit Cloud **غير دائم** بين عمليات إعادة النشر، لذلك صمّم التطبيق آلية إعادة البناء التلقائي عند كل تشغيل جديد بدل الاعتماد على بقاء البيانات محفوظة إلى الأبد.

---

## 10. الأخطاء الشائعة وحلولها

| الخطأ | السبب المحتمل | الحل |
|---|---|---|
| `Knowledge base folder not found` | مجلد `data/nutrition` غير موجود أو فارغ | تأكد من وجود ملفات `.md` داخل المجلد |
| `No OpenRouter API key configured` | لم يتم ضبط `OPENROUTER_API_KEY` | أضفه في `.env` محليًا أو في Secrets عند النشر |
| `The vector database is empty` | لم تُبنَ القاعدة بعد | اضغط **Rebuild Database** من الشريط الجانبي |
| رسالة اتصال فاشلة بـ OpenRouter | مشكلة إنترنت أو مفتاح غير صالح أو رصيد منتهٍ | تحقق من المفتاح ورصيد الحساب على OpenRouter |
| بطء أول تحميل للنموذج | يتم تنزيل نموذج `all-MiniLM-L6-v2` من الإنترنت أول مرة فقط | طبيعي، ثم يُخزَّن محليًا بعد أول تشغيل |

---

## 11. تحديث قاعدة المعرفة بملفات Markdown جديدة

1. أضف الملف الجديد داخل `data/nutrition/`.
2. إن أردت بيانات وصفية مخصصة (عنوان، مصدر، تصنيف...)، أضف مدخلًا له داخل قاموس `CUSTOM_METADATA` في `01_documents.py`.
3. أعد تشغيل التطبيق أو اضغط **Rebuild Database** — سيكتشف النظام التغيير تلقائيًا عبر مقارنة البصمة (hash) ويعيد البناء.

---

## 12. إعادة بناء قاعدة البيانات الشعاعية يدويًا

- من الواجهة: زر **🔄 Rebuild Database** في الشريط الجانبي.
- من سطر الأوامر:
  ```bash
  python 05_create_chroma_store.py
  ```
- لإجبار إعادة بناء كاملة حتى لو لم يتغيّر المحتوى، احذف المجلد والملف التاليين ثم أعد التشغيل:
  ```bash
  rm -rf database/chroma_db database/source_hash.json
  ```

---

## 13. صيانة المشروع مستقبلًا

- **تحديث النموذج اللغوي**: غيّر قيمة `OPENROUTER_MODEL` في الأسرار (Secrets) أو `.env` دون الحاجة لتعديل أي كود.
- **تحديث نموذج التمثيل الشعاعي**: عدّل `EMBEDDING_MODEL_NAME` داخل `config.py` ثم أعد بناء القاعدة.
- **تعديل منطق الاسترجاع** (عدد النتائج، وزن البحث الهجين): القيم `DEFAULT_TOP_K`, `DEFAULT_ALPHA`, `DEFAULT_MAX_CHUNKS`, `DEFAULT_WORD_BUDGET` في `config.py`.
- **تخصيص الواجهة**: عدّل `assets/style.css` للألوان والخطوط دون المساس بمنطق Python.
- راجع دوريًا تحديثات مكتبات `requirements.txt` (خصوصًا `chromadb` و `sentence-transformers`) للحفاظ على التوافق والأمان.
- احتفظ دائمًا بنسخة من ملفات المعرفة الأصلية (`data/nutrition/`) خارج المستودع كنسخة احتياطية.
