import os
import re
from flask import Flask, render_template, request, flash, redirect, url_for
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from googletrans import Translator, LANGUAGES
from PyPDF2 import PdfReader

# -------------------------------
# Flask setup
# -------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_only_secret_change_me")

# -------------------------------
# Model setup (T5-small = faster; swap to t5-base if you want higher quality)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = os.environ.get("T5_MODEL_NAME", "t5-small")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# Keep a small rolling cache of last N summaries
previous_summaries = []
MAX_HISTORY = 20

translator = Translator()

# -------------------------------
# Utilities
# -------------------------------
def extract_text_from_pdf(file_storage):
    """
    Extract text from a PDF uploaded via Flask's FileStorage.
    Handles pages that return None for text (e.g., images-only).
    """
    try:
        reader = PdfReader(file_storage)
        chunks = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text()
                if txt:
                    chunks.append(txt)
            except Exception:
                # Skip unreadable page; continue
                continue
        text = "\n".join(chunks).strip()
        return text
    except Exception as e:
        flash(f"PDF parsing error: {str(e)}", "error")
        return ""

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_summary(
    text: str,
    max_input_len: int = 512,
    min_out_len: int = 60,
    max_out_len: int = 200,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
) -> str:
    prompt = "summarize: " + normalize_whitespace(text)
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=max_input_len, truncation=True).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs,
            max_length=max_out_len,
            min_length=min_out_len,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            eos_token_id=tokenizer.eos_token_id,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Light cleanup: fix spaces before punctuation
    summary = re.sub(r"\s+([.,!?;:])", r"\1", summary).strip()
    return summary

def translate_to_language(summary: str, language_code: str) -> str:
    if language_code == "en":
        return ""
    try:
        res = translator.translate(summary, dest=language_code)
        return res.text
    except Exception as e:
        flash(f"Translation failed ({language_code}): {str(e)}", "error")
        return ""

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    summary = ""
    translated_summary = ""
    language = "en"

    # UI-provided desired length (soft target)
    ui_length = 200  # default

    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        language = request.form.get("language", "en")
        try:
            ui_length = int(request.form.get("summary_length", "200"))
            ui_length = max(50, min(ui_length, 512))  # enforce sane bounds
        except ValueError:
            ui_length = 200

        # Prefer uploaded PDF if provided
        if "input_file" in request.files and request.files["input_file"].filename:
            input_text = extract_text_from_pdf(request.files["input_file"])

        # Guardrails
        if not input_text or not input_text.strip():
            flash("Please provide some text or upload a readable PDF.", "error")
            return redirect(url_for("index"))

        if input_text.isdigit():
            flash("Only digits detected. Please provide meaningful text.", "error")
            return redirect(url_for("index"))

        # Summarize
        try:
            # Heuristic: aim min_len ~ 0.4 * desired, max_len ~ desired
            min_len = max(20, int(0.4 * ui_length))
            max_len = max(min_len + 10, ui_length)

            summary = generate_summary(
                input_text,
                max_input_len=512,
                min_out_len=min_len,
                max_out_len=max_len,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )

            translated_summary = translate_to_language(summary, language)

            item = (normalize_whitespace(input_text)[:5000], summary, translated_summary, language)
            if not previous_summaries or previous_summaries[-1] != item:
                previous_summaries.append(item)
                if len(previous_summaries) > MAX_HISTORY:
                    previous_summaries.pop(0)

        except Exception as e:
            flash(f"An error occurred while summarizing: {str(e)}", "error")
            return redirect(url_for("index"))

    return render_template(
        "index.html",
        input_text=input_text,
        summary=summary,
        translated_summary=translated_summary,
        language=language,
        languages=LANGUAGES,          # dict: code -> language name
        summary_length=ui_length,
        previous_summaries=previous_summaries,
    )

if __name__ == "__main__":
    # For local dev only
    app.run(debug=True)
