# 📬 Customer-Support Ticket Classifier  
Fine-tuned **GPT-4o-mini** for routing e-mails to the right queue

[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](https://www.python.org/)  
[![OpenAI FT](https://img.shields.io/badge/OpenAI-Fine--Tuning-orange)](https://platform.openai.com/docs/guides/fine-tuning)

---

## 1 · Why this repo exists
*Small teams drown in support mail.*  
Automated triage keeps human agents focused. Here we teach a lightweight OpenAI
model to label each ticket as **Technical**, **Billing**, **Account**, **Product** or **Other**.

The project shows the full life-cycle:

| Stage | What we do |
|-------|------------|
| **Dataset** | Craft 55 realistic training tickets + 12 test tickets → `*.jsonl` |
| **Baseline** | Measure accuracy of vanilla **gpt-4o-mini** with only a prompt |
| **Fine-Tune** | Train 3 epochs, ~5 min, <$0.40 |
| **Evaluate** | Compare new model to baseline |
| **Discuss** | When fine-tuning is worth it vs. prompt engineering |

---

## 2 · Quick start

```bash
git clone https://github.com/<you>/ticket-classifier.git
cd ticket-classifier

# 1 Create & activate virtual-env
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2 Set your secret once
echo "OPENAI_API_KEY=sk-..." > .env

# 3 Build data & baseline
python support_classifier.py build-data
python support_classifier.py baseline          # prints ≈75 % accuracy

# 4 Fine-tune (takes a few minutes)
python support_classifier.py fine-tune         # logs job id, polls until done

# 5 Evaluate the new model
python support_classifier.py evaluate --model ft:gpt-4o-mini:org:ticket:v1
```

## 3 · Repository layout

├── support_classifier.py      # one-stop script: data → train → eval
├── train.jsonl                # 55 labeled examples (generated)
├── test.jsonl                 # 12 held-out examples (generated)
├── requirements.txt           # openai, tqdm, python-dotenv
├── README.md                  # you’re reading it
└── .gitignore                 # ignores venvs, pyc, secrets, artefacts

## 4 · Results
Model	Accuracy	Tokens / req	Cost / 1 k req*
Prompt-only gpt-4o-mini	75 %	~35	$0.015
Fine-tuned model	92 %	~10	$0.003

*At 2025-06-13 pricing: $0.0005 /1K input tokens, $0.0015 /1K output tokens.

Take-away: Fine-tuning slashes both error rate and per-call cost.

VIDEO: https://www.youtube.com/watch?v=Bh1RTCEppIk&t=1573s