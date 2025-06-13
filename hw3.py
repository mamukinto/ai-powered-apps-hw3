"""
support_classifier.py
Build dataset • Baseline • Fine-tune • Evaluate
------------------------------------------------
python support_classifier.py  build-data
python support_classifier.py  baseline
python support_classifier.py  fine-tune
python support_classifier.py  evaluate --model ft:...   # after job finishes
"""
import argparse, os, json, time, random, sys
from pathlib import Path
from typing import List, Tuple

import openai
from tqdm import tqdm
from dotenv import load_dotenv

# ---------- CONFIG ----------
TRAIN_PATH = Path("train.jsonl")
TEST_PATH  = Path("test.jsonl")
CATEGORIES = ["Technical", "Billing", "Account", "Product", "Other"]
BASE_MODEL = "gpt-4o-mini"        # change to o3-mini or gpt-3.5-turbo-0613 if preferred
TEMPERATURE = 0                   # deterministic for classification

load_dotenv()                     # pulls OPENAI_API_KEY from .env if present
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- DATASET ----------
RAW_TICKETS: List[Tuple[str, str]] = [
    # Technical (15)
    ("My laptop won't turn on after the update", "Technical"),
    ("The mobile app crashes whenever I upload a photo", "Technical"),
    ("VPN disconnects every 5 minutes on Windows 11", "Technical"),
    ("I’m seeing a 500 error when hitting /api/v2/orders", "Technical"),
    ("Audio is distorted in video calls since last patch", "Technical"),
    ("Can't pair the smartwatch with my Pixel 8", "Technical"),
    ("Firmware update stuck at 12 %", "Technical"),
    ("Wi-Fi drops whenever microwave is on", "Technical"),
    ("Keyboard backlight stopped working", "Technical"),
    ("Camera shows black screen in low light", "Technical"),
    ("Printer prints half pages only", "Technical"),
    ("Touch-ID fails after I re-enrolled my finger", "Technical"),
    ("Getting 'device not certified' in Play Store", "Technical"),
    ("Smart bulb unresponsive in Apple Home", "Technical"),
    ("Battery drains from 100 → 20 % in two hours", "Technical"),

    # Billing (10)
    ("I was charged twice for the same invoice", "Billing"),
    ("Why did my subscription price rise this month?", "Billing"),
    ("Need a VAT invoice for order 78321", "Billing"),
    ("Refund still not credited after 14 days", "Billing"),
    ("Coupon code applied but discount missing", "Billing"),
    ("Subscription cancelled yet I’m billed again", "Billing"),
    ("Late fee added by mistake, please remove", "Billing"),
    ("How can I switch to annual payment?", "Billing"),
    ("Please change card on file to my Visa …1234", "Billing"),
    ("Charge shows as ‘Pending’ two weeks later", "Billing"),

    # Account (10)
    ("Forgot my password and reset link never arrives", "Account"),
    ("Need to change email from john@old.com to john@new.com", "Account"),
    ("Two-factor code SMS is not coming through", "Account"),
    ("Account locked after too many attempts", "Account"),
    ("Combine my personal and work profiles", "Account"),
    ("Please delete my data under GDPR", "Account"),
    ("Why was my username flagged?", "Account"),
    ("How do I set up SSO with Okta?", "Account"),
    ("Add my colleague as admin", "Account"),
    ("Close my account permanently", "Account"),

    # Product (10)
    ("Feature request: dark mode for dashboard", "Product"),
    ("Roadmap ETA for Linux client?", "Product"),
    ("Do you support Apple silicon natively?", "Product"),
    ("Is there an API limit for free tier?", "Product"),
    ("Can you add Turkish language support?", "Product"),
    ("Need spec sheet for Model X router", "Product"),
    ("Is the smart lock weather-proof?", "Product"),
    ("When will version 3.0 be released?", "Product"),
    ("Does plan Pro include analytics?", "Product"),
    ("Clarify difference between Plus and Premium", "Product"),

    # Other (10)
    ("Great job on the latest webinar!", "Other"),
    ("Interested in partnership opportunities", "Other"),
    ("I found a typo on your homepage", "Other"),
    ("Media inquiry: interview with CEO", "Other"),
    ("Please stop sending me marketing emails", "Other"),
    ("Complaint about courier attitude", "Other"),
    ("Do you have offices in Canada?", "Other"),
    ("Your phone line is always busy", "Other"),
    ("What’s your carbon footprint policy?", "Other"),
    ("Need directions to your HQ", "Other"),
]

random.seed(42)
random.shuffle(RAW_TICKETS)  # mix categories

# Keep first N for train, rest for test
TRAIN_SIZE = 55
TRAIN_TICKETS = RAW_TICKETS[:TRAIN_SIZE]
TEST_TICKETS  = RAW_TICKETS[TRAIN_SIZE:]


def build_data():
    """Write train.jsonl and test.jsonl in OpenAI fine-tune format."""
    for path, split in [(TRAIN_PATH, TRAIN_TICKETS), (TEST_PATH, TEST_TICKETS)]:
        with path.open("w", encoding="utf-8") as f:
            for text, label in split:
                prompt = f"Category options: {', '.join(CATEGORIES)}\n\nTicket: \"{text}\"\n\nCategory:"
                completion = f" {label}"
                json.dump({"messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]}, f)
                f.write("\n")
    print(f"Wrote {TRAIN_PATH} ({len(TRAIN_TICKETS)} lines) and {TEST_PATH} ({len(TEST_TICKETS)} lines)")


# ---------- HELPER ----------
def classify(model: str, prompt: str) -> str:
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()


def evaluate(model: str, tickets: List[Tuple[str, str]]) -> float:
    correct = 0
    for text, gold in tqdm(tickets, desc=f"Evaluating {model}"):
        prompt = f"Category options: {', '.join(CATEGORIES)}\n\nTicket: \"{text}\"\n\nCategory:"
        pred = classify(model, prompt)
        if pred.lower().startswith(gold.lower()):
            correct += 1
    acc = correct / len(tickets)
    print(f"Accuracy: {acc:.2%}  ({correct}/{len(tickets)})")
    return acc


# ---------- BASELINE ----------
def baseline():
    evaluate(BASE_MODEL, TEST_TICKETS)


# ---------- FINE-TUNE ----------
def fine_tune():
    # 1) upload training file
    upload_resp = openai.files.create(
        file=open(TRAIN_PATH, "rb"),
        purpose="fine-tune"
    )
    file_id = upload_resp.id
    print("Uploaded training file:", file_id)

    # 2) create fine-tune job
    job_resp = openai.fine_tuning.jobs.create(
        training_file=file_id,
        model=BASE_MODEL,
        hyperparameters={"n_epochs": 3}
    )
    job_id = job_resp.id
    print("Started job:", job_id)

    # 3) poll until done
    status = job_resp.status
    while status not in ("succeeded", "failed", "cancelled"):
        time.sleep(10)
        job_resp = openai.fine_tuning.jobs.retrieve(job_id)
        status = job_resp.status
        print("…", status)
    if status == "succeeded":
        print("Fine-tune complete. New model:", job_resp.fine_tuned_model)
    else:
        print("Job ended with status:", status)


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-data")
    sub.add_parser("baseline")
    sub.add_parser("fine-tune")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", required=True, help="fine-tuned model name (ft:...)")
    args = parser.parse_args()

    if args.cmd == "build-data":
        build_data()
    elif args.cmd == "baseline":
        baseline()
    elif args.cmd == "fine-tune":
        fine_tune()
    elif args.cmd == "evaluate":
        evaluate(args.model, TEST_TICKETS)


if __name__ == "__main__":
    if not openai.api_key:
        sys.exit("❌  Set OPENAI_API_KEY first!")
    main()
