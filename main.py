import os
import time
import requests
import base64
import re
import math
import traceback
from flask import Flask, request, Response
import google.generativeai as genai

try:
    from langdetect import detect_langs
except Exception:  # pragma: no cover
    detect_langs = None

# --- CONFIGURATION ---
app = Flask(__name__)

# Credentials
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ZD_EMAIL = os.environ.get("ZD_EMAIL")
ZD_SUBDOMAIN = os.environ.get("ZD_SUBDOMAIN")
ZD_TOKEN = os.environ.get("ZD_TOKEN")

# Pricing Constants (Audited)
# Voice: ~$0.0085/min for Inbound, but using standard $0.014 as safe buffer.
# Speech Recognition (Gather): ~$0.035/min (Legacy/Standard mix).
PRICE_TWILIO_VOICE_PER_MIN = float(os.environ.get("TWILIO_VOICE_PER_MIN", "0.014"))
PRICE_TWILIO_STT_PER_MIN = float(os.environ.get("TWILIO_STT_PER_MIN", "0.035"))

PRICE_GEMINI_INPUT_1K = 0.000075
PRICE_GEMINI_OUTPUT_1K = 0.00030

# Configure AI
model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash") 
else:
    print("⚠️ WARNING: GEMINI_API_KEY is missing!")

call_context = {}

# Debug toggle
DEBUG_CALL_FLOW = os.environ.get("DEBUG_CALL_FLOW", "").strip().lower() in {"1", "true", "yes"}

def debug_log(message: str) -> None:
    if DEBUG_CALL_FLOW:
        print(message)

def _mask_email(value: str) -> str:
    if not value:
        return ""
    value = str(value).strip()
    if "@" not in value:
        return value[:2] + "***" if len(value) > 2 else "***"
    local, domain = value.split("@", 1)
    if not local:
        return "***@" + domain
    return f"{local[0]}***@{domain}"

def debug_event(stage: str, call_sid: str | None = None, **fields) -> None:
    if not DEBUG_CALL_FLOW:
        return

    safe_fields = []
    for key, value in fields.items():
        if value is None:
            continue
        if key in {"email", "user_email", "extracted_email", "stored_email"}:
            value = _mask_email(str(value))
        safe_fields.append(f"{key}={value!r}")

    prefix = f"[voice:{stage}]"
    if call_sid:
        prefix += f" CallSid={call_sid}"

    debug_log(prefix + (" " + " ".join(safe_fields) if safe_fields else ""))

def debug_exception(stage: str, call_sid: str | None = None, err: Exception | None = None) -> None:
    if not DEBUG_CALL_FLOW:
        return
    debug_event(stage, call_sid=call_sid, error=str(err) if err else "")
    debug_log(traceback.format_exc())

# High-level ticket taxonomy
TICKET_TYPE_GUIDE = """
Common Aerosus support reasons:
- Order info: order status, tracking number, invoice copy
- Order issues: cancellation, address change, payment failed, VAT not recognized
- Delivery issues: courier/customs/payment issues, delayed delivery
- Returns/RMA: damaged/faulty item, wrong item, warranty, refund requests
- Compatibility: part compatibility check, stock info
- Technical support: installation questions, compressor/strut issues
- B2B: partnership, discount request
""".strip()

# Speech hints for Twilio
SPEECH_HINTS = (
    "Aerosus, air suspension, compressor, strut, shock absorber, relay, "
    "RMA, refund, return, tracking number, order number, invoice, "
    "gmail, outlook, hotmail, yahoo, icloud, protonmail, at, dot, underscore, dash"
)

# Twilio <Say> voice override (Amazon Polly Neural)
VOICE_NAME = os.environ.get("TWILIO_VOICE_NAME", "Polly.Amy-Neural")

# --- LANGUAGE DETECTION (Double Defense Layer 1) ---
_NON_LATIN_SCRIPT_RE = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")
_NON_ENGLISH_MARKERS = {
    # Turkish
    "merhaba", "selam", "evet", "hayir", "yok", "lutfen", "tesekkur", "anlamadim", 
    "turkce", "biliyormusun", "siparis", "kargo", "fatura", "iade",

    # Turkish (common grammar words / short utterances)
    "bir", "ve", "bu", "da", "de", "icin", "cok", "ama", "nasil", "ne", "ben", "sen",

    # German (common function words)
    "hallo", "danke", "bitte", "nein", "ja", "ich", "ist", "und", "der", "die", "das", "nicht", "es",

    # European
    "hola", "gracias", "bonjour", "merci", "salut", "hallo", "danke", "bitte", 
    "ciao", "grazie", "ola", "obrigado", "pedido", "envio", "reembolso", 
    "devolucion", "commande", "livraison", "remboursement", "bestellung", 
    "lieferung", "rueckerstattung",

    # Spanish/Italian/Portuguese (common function words)
    "el", "la", "que", "de", "en", "y", "los", "del", "se", "por",

    # Roman Urdu / Urdu (Latin transliteration)
    # Twilio STT often outputs Urdu speech in Latin characters; add common function words.
    "mujhe", "mujhay", "mera", "meri", "mere", "aap", "ap", "tum", "hum",
    "kya", "ky", "kyun", "kaise", "kis", "kahan", "kab",
    "hai", "hain", "tha", "thi",
    "nahi", "nahin", "haan", "han",
    "shukriya", "meherbani", "mehrbani",
    "salam", "assalam", "walaikum", "waalaikum", "salaam",
    "inshaallah", "mashallah",
}

def is_likely_non_english(text: str) -> bool:
    if not text:
        return False
    # Don't block emails or numeric inputs
    if extract_email_from_utterance(text):
        return False
    if re.fullmatch(r"[\s0-9+\-#()]+", text.strip() or ""):
        return False

    # Language-ID (covers any language, including romanized text)
    # Keep conservative thresholds to avoid false positives on short inputs.
    if detect_langs is not None:
        sample = (text or "").strip()
        if len(sample) >= 12:
            try:
                langs = detect_langs(sample)
                if langs:
                    top = langs[0]
                    top_lang = getattr(top, "lang", "")
                    top_prob = float(getattr(top, "prob", 0.0) or 0.0)
                    # If confidently not English, flag it.
                    if top_lang and top_lang != "en" and top_prob >= 0.80:
                        return True
            except Exception:
                # Fall back to heuristics below
                pass
    # Strong signal: Non-Latin scripts
    if _NON_LATIN_SCRIPT_RE.search(text):
        return True
    # Strong signal: Extended ASCII
    if any(ord(ch) > 127 for ch in text):
        return True
    
    words = re.findall(r"[a-z']+", text.lower())
    marker_hits = sum(1 for w in words if w in _NON_ENGLISH_MARKERS)
    
    # Strict check: even 1 marker triggers the warning
    if marker_hits >= 1:
        return True
    
    return False

# --- EMAIL HANDLING ---
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
EMAIL_IN_TEXT_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.IGNORECASE)
AFFIRMATIVE_RE = re.compile(r"\b(yes|yeah|yep|correct|that's\s+right|that\s+is\s+right|right|sure|affirmative)\b", re.IGNORECASE)
NEGATIVE_RE = re.compile(r"\b(no|nope|nah|incorrect|wrong|negative|not\s+right)\b", re.IGNORECASE)

def normalize_spoken_email(text: str) -> str:
    if not text: return ""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^(uh|um|er|ah|hmm|mmm)[\s,]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(my\s+)?e-?mail\s+is\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bemail\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(uh|um|er|ah|hmm|mmm)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("<", "").replace(">", "").strip(" \t\r\n.,;:!?")
    cleaned = cleaned.replace(" at ", "@").replace(" dot ", ".")
    cleaned = cleaned.replace(" at", "@").replace("at ", "@")
    cleaned = cleaned.replace(" dot", ".").replace("dot ", ".")
    cleaned = cleaned.replace(" ", "")
    cleaned = re.sub(r"[^a-z0-9@._%+\-]", "", cleaned)
    cleaned = cleaned.strip(".,;:!?")
    return cleaned

def is_valid_email(email: str) -> bool:
    return bool(email) and bool(EMAIL_RE.match(email))

def extract_email_from_utterance(text: str) -> str:
    if not text: return ""
    normalized = normalize_spoken_email(text)
    if is_valid_email(normalized): return normalized
    match = EMAIL_IN_TEXT_RE.search(normalized)
    if not match: return ""
    candidate = match.group(0).lower().strip(".,;:!?")
    return candidate if is_valid_email(candidate) else ""

def is_affirmative(text: str) -> bool:
    return bool(text) and bool(AFFIRMATIVE_RE.search(text))

def is_negative(text: str) -> bool:
    return bool(text) and bool(NEGATIVE_RE.search(text))

def spell_email_address(email: str) -> str:
    """Spell local part letter-by-letter with spaces, read domain normally."""
    if not email or "@" not in email: return ""
    local_part, domain = email.strip().lower().split("@", 1)
    token_map = {".": "dot", "_": "underscore", "-": "dash", "+": "plus"}
    
    # Spell local part character by character with spaces
    spoken_local = []
    for ch in local_part:
        if ch.isalnum():
            spoken_local.append(ch)
        elif ch in token_map:
            spoken_local.append(token_map[ch])
    
    # Read domain normally (e.g., "gmail.com" not "g m a i l dot c o m")
    return f"{' '.join(spoken_local)} at {domain}".strip()

def normalize_problem_text(text: str) -> str:
    """Fix common speech-to-text errors for air suspension domain."""
    if not text: return ""
    normalized = text
    
    # Common STT mishearings for air suspension terminology
    normalized = re.sub(r"\bear\s+suspension\b", "air suspension", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\beir\s+suspension\b", "air suspension", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bhere\s+suspension\b", "air suspension", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bhair\s+suspension\b", "air suspension", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bair\s+suspensions\b", "air suspension", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\baero\s+sus\b", "Aerosus", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\baero\s+sauce\b", "Aerosus", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\baero\s+source\b", "Aerosus", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bair\s+o\s+sus\b", "Aerosus", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bcompress\s+or\b", "compressor", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bcompress\s+her\b", "compressor", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bshock\s+absorber\b", "shock absorber", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bshock\s+observer\b", "shock absorber", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\br\s*m\s*a\b", "RMA", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bstreet\b", "strut", normalized, flags=re.IGNORECASE)  # common mishearing
    normalized = re.sub(r"\bstruck\b", "strut", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bleaking\b", "leaking", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bleaking\b", "leaking", normalized, flags=re.IGNORECASE)

    # Domain-aware speech-to-text correction:
    # Twilio/Gather sometimes hears "car" as "card".
    # Only correct it when the utterance is clearly about suspension/leaks, not payments.
    lowered = normalized.lower()
    if re.search(r"\bcard\b", lowered):
        suspension_cues = (
            "suspension",
            "air suspension",
            "strut",
            "shock",
            "compressor",
            "leak",
            "leaking",
            "air bag",
            "airbag",
        )
        payment_cues = (
            "payment",
            "paid",
            "charge",
            "charged",
            "credit",
            "debit",
            "visa",
            "mastercard",
            "cvv",
            "expiry",
            "expiration",
            "card number",
        )
        if any(cue in lowered for cue in suspension_cues) and not any(cue in lowered for cue in payment_cues):
            normalized = re.sub(r"\bcard\b", "car", normalized, flags=re.IGNORECASE)

    return normalized


def is_vague_problem(text: str) -> bool:
    """
    Check if the problem description is too vague and needs more details.
    Returns True if the user just said something like "I have a problem" without specifics.
    """
    if not text:
        return True
    
    lowered = text.lower().strip()
    
    # Specific automotive/suspension terms indicate a real problem description
    specific_terms = (
        "suspension", "air suspension", "compressor", "strut", "shock",
        "leak", "leaking", "broken", "not working", "noise", "clicking",
        "order", "tracking", "delivery", "refund", "return", "rma",
        "invoice", "payment", "warranty", "damaged", "wrong", "missing",
        "install", "compatibility", "fit", "replace", "pump", "valve",
        "air bag", "airbag", "spring", "relay", "fuse", "sensor",
        "height", "lowering", "raising", "stuck", "fault", "error",
        "mercedes", "bmw", "audi", "porsche", "range rover", "land rover",
        "volkswagen", "vw", "bentley", "rolls royce", "jaguar", "lexus",
    )
    
    # If any specific term is present, it's not vague
    if any(term in lowered for term in specific_terms):
        return False
    
    # Vague phrases that need clarification
    vague_patterns = (
        r"\b(there('s|s| is)?\s+)?a\s+problem\b",
        r"\bi\s+(have|got)\s+(a|an|some)?\s*(problem|issue)\b",
        r"\bsomething('s|s| is)?\s+(wrong|broken)\b",
        r"\bit('s|s| is)?\s+not\s+working\b",
        r"\bneed\s+help\b",
        r"\bhaving\s+(an?\s+)?(issue|problem|trouble)\b",
    )
    
    for pattern in vague_patterns:
        if re.search(pattern, lowered):
            # Check if there's more substance beyond the vague phrase
            # Remove the vague part and see if anything meaningful remains
            remaining = re.sub(pattern, "", lowered).strip()
            remaining = re.sub(r"\b(with|in|on|my|the|a|an|car|vehicle)\b", "", remaining).strip()
            if len(remaining) < 5:  # Not much left = vague
                return True
    
    # Very short descriptions are likely vague
    words = [w for w in lowered.split() if len(w) > 2]
    if len(words) <= 4:
        # Check if it's just "problem with my car" type phrases
        non_filler_words = [w for w in words if w not in ("the", "my", "car", "with", "have", "problem", "issue", "there", "help")]
        if len(non_filler_words) == 0:
            return True
    
    return False


def is_order_related(text: str) -> bool:
    """
    Check if the problem is order-related (return, order status, shipping status).
    These require both order number AND email for verification.
    """
    if not text:
        return False
    lowered = text.lower()
    order_keywords = (
        "order status", "order number", "my order", "where is my order",
        "shipping status", "shipping update", "delivery status", "track", "tracking",
        "return", "refund", "rma", "send back", "exchange",
        "cancel order", "cancel my order", "cancellation",
        "when will", "hasn't arrived", "not arrived", "not received",
    )
    return any(kw in lowered for kw in order_keywords)


def extract_order_number(text: str) -> str:
    """
    Extract order number from speech. Order numbers are typically numeric or alphanumeric.
    """
    if not text:
        return ""
    
    # Common patterns: "order 12345", "order number 12345", "#12345", etc.
    # Also handle spelled out: "one two three four five"
    lowered = text.lower().strip()
    
    # Direct numeric extraction after "order" or "number"
    match = re.search(r"(?:order|number|#)\s*[:#]?\s*([a-z0-9\-]+)", lowered)
    if match:
        candidate = match.group(1).strip()
        if len(candidate) >= 3:  # Order numbers are usually at least 3 chars
            return candidate.upper()
    
    # Try to find standalone alphanumeric that looks like an order number
    # (5+ digits or alphanumeric combo)
    match = re.search(r"\b([a-z]{0,3}[0-9]{4,}[a-z0-9]*)\b", lowered)
    if match:
        return match.group(1).upper()
    
    # Handle spelled out numbers
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "oh": "0", "o": "0",
    }
    words = lowered.split()
    digits = []
    for w in words:
        if w in number_words:
            digits.append(number_words[w])
        elif w.isdigit():
            digits.append(w)
    if len(digits) >= 4:
        return "".join(digits)
    
    return ""


def calculate_cost(duration_sec, in_tok, out_tok):
    billable_minutes = max(1, int(math.ceil(float(duration_sec) / 60.0)))
    twilio_cost = billable_minutes * (PRICE_TWILIO_VOICE_PER_MIN + PRICE_TWILIO_STT_PER_MIN)
    ai_cost = (in_tok / 1000 * PRICE_GEMINI_INPUT_1K) + (out_tok / 1000 * PRICE_GEMINI_OUTPUT_1K)
    return round(twilio_cost + ai_cost, 4)

def create_zendesk_ticket(user_email, issue_summary, cost):
    if not all([ZD_EMAIL, ZD_SUBDOMAIN, ZD_TOKEN]):
        print("❌ Zendesk env vars missing")
        return None
    
    url = f"https://{ZD_SUBDOMAIN}.zendesk.com/api/v2/tickets.json"
    creds = f"{ZD_EMAIL}/token:{ZD_TOKEN}"
    encoded_creds = base64.b64encode(creds.encode()).decode()
    clean_email = re.sub(r"[^a-zA-Z0-9@._%+\-]", "", user_email.strip())
    user_name = clean_email.split("@")[0]

    payload = {
        "ticket": {
            "subject": f"Voice AI: {issue_summary[:30]}...",
            "comment": { "body": f"User reported:\n{issue_summary}\n\n💰 Call Cost: ${cost}" },
            "requester": { "name": user_name, "email": clean_email }
        }
    }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Basic {encoded_creds}"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 201:
            return response.json()["ticket"]["id"]
        return None
    except Exception as e:
        print(f"💥 Zendesk request crashed: {e}")
        return None


def intelligently_normalize_problem(raw_problem: str, context_dict: dict) -> str:
    """
    Use Gemini to intelligently rewrite the problem for Zendesk.
    Fixes transcription errors based on air suspension domain context.
    """
    if not raw_problem:
        return raw_problem
    
    # First apply regex-based fixes
    normalized = normalize_problem_text(raw_problem)
    
    # Then use Gemini for deeper context understanding
    if model is None:
        return normalized
    
    try:
        prompt = f"""
You are a text normalizer for Aerosus, an air suspension parts company.

Your job is to fix speech-to-text transcription errors in customer problem descriptions.
The customer called about their car's air suspension system.

COMMON TRANSCRIPTION ERRORS TO FIX:
- "ear suspension" → "air suspension"
- "hair suspension" → "air suspension"
- "here suspension" → "air suspension"
- "aero sus" / "aero sauce" / "air o sus" → "Aerosus"
- "compress or" / "compress her" → "compressor"
- "street" / "struck" (when talking about car parts) → "strut"
- "shock observer" → "shock absorber"
- "card" (when talking about vehicle, not payment) → "car"
- Any other obvious speech-to-text errors in the automotive/suspension context

COMMON AEROSUS ISSUES:
{TICKET_TYPE_GUIDE}

INPUT TEXT:
"{normalized}"

OUTPUT RULES:
- Return ONLY the corrected text, nothing else
- Keep the same meaning, just fix obvious transcription errors
- If no corrections needed, return the text unchanged
- Do NOT add explanations or commentary
- Keep it concise and professional
"""
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Track token usage if available
        if hasattr(response, 'usage_metadata') and context_dict:
            context_dict['input_tokens'] = context_dict.get('input_tokens', 0) + response.usage_metadata.prompt_token_count
            context_dict['output_tokens'] = context_dict.get('output_tokens', 0) + response.usage_metadata.candidates_token_count
        
        # Sanity check: result should be similar length (not a full paragraph explanation)
        if result and len(result) < len(normalized) * 3:
            return result
        return normalized
        
    except Exception as e:
        print(f"⚠️ Gemini normalization failed: {e}")
        return normalized

# --- MAIN ROUTE ---
@app.route("/voice", methods=['GET', 'POST'])
def voice():
    call_sid = request.values.get('CallSid')
    user_input = request.values.get('SpeechResult')
    debug_event("incoming", call_sid=call_sid, has_speech=bool(user_input))

    if not call_sid:
        return Response("Missing CallSid", status=400, mimetype='text/plain')

    try:
        # 1. INITIALIZE CONTEXT
        if call_sid not in call_context:
            call_context[call_sid] = {
                "history": [],
                "problem": "",
                "email": "",
                "email_confirmed": False,
                "awaiting_email_confirmation": False,
                "awaiting_email": False,
                "awaiting_problem_details": False,
                "email_declined": False,
                "email_confirm_attempts": 0,
                "email_request_attempts": 0,
                "order_number": "",
                "order_number_confirmed": False,
                "awaiting_order_number": False,
                "awaiting_order_confirmation": False,
                "order_confirm_attempts": 0,
                "is_order_related_issue": False,
                "english_warning_count": 0,
                "start_time": time.time(),
                "input_tokens": 0,
                "output_tokens": 0,
            }
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>Welcome to Aerosus customer support. How may I help you?</Say></Gather></Response>",
                mimetype='text/xml'
            )

        context = call_context[call_sid]

        # 2. NO INPUT HANDLING
        if not user_input:
            if context.get("problem") and not context.get("email"):
                if context.get("email_declined"):
                    msg = "Ok. I can't create a support ticket without an email. Goodbye."
                    context.setdefault("history", []).append(f"AI: {msg}")
                    call_context.pop(call_sid, None)
                    return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')

                attempts = int(context.get("email_request_attempts", 0))
                prompt_text = "Please say your email one character at a time." if attempts >= 2 else "Ok to proceed with your request provide me your email."
                context["email_request_attempts"] = attempts + 1
                context["awaiting_email"] = True
            else:
                prompt_text = "I am listening. What is the problem?"

            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{prompt_text}</Say></Gather></Response>",
                mimetype='text/xml'
            )

        # 3. LANGUAGE CHECK (Layer 1)
        if is_likely_non_english(user_input):
            context["english_warning_count"] = int(context.get("english_warning_count", 0)) + 1

            if context["english_warning_count"] >= 2:
                msg = "This call can only be processed in English. Please try again later."
                call_context.pop(call_sid, None)
                return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')

            msg = "Please speak in English."
            context.setdefault("history", []).append(f"AI: {msg}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{msg}</Say></Gather></Response>",
                mimetype='text/xml'
            )
        else:
            context["english_warning_count"] = 0

        # 4. TRANSCRIPT LOGGING
        context.setdefault("history", []).append(f"User: {user_input}")

        # 4.5 STABLE EMAIL REQUEST HANDLING
        if context.get("awaiting_email") and not context.get("email"):
            extracted_email_now = extract_email_from_utterance(user_input)
            if extracted_email_now:
                context["awaiting_email"] = False
                context["email"] = extracted_email_now
                context["email_confirmed"] = False
                context["awaiting_email_confirmation"] = True
                context["email_request_attempts"] = 0

                spelled = spell_email_address(context["email"])
                ai_reply = f"I heard {spelled}. Is that correct?"
                context["history"].append(f"AI: {ai_reply}")
                return Response(
                    f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                    mimetype='text/xml'
                )

            if is_negative(user_input):
                # User said something negative but we're waiting for email input, not declining
                # Just re-ask for email
                attempts = int(context.get("email_request_attempts", 0))
                ai_reply = "Please say your email one character at a time." if attempts >= 2 else "Please provide your email."
                context["email_request_attempts"] = attempts + 1
                context["history"].append(f"AI: {ai_reply}")
                return Response(
                    f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                    mimetype='text/xml'
                )

            attempts = int(context.get("email_request_attempts", 0))
            ai_reply = "Please say your email one character at a time." if attempts >= 2 else "Ok to proceed with your request provide me your email."
            context["email_request_attempts"] = attempts + 1
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )

        # 5. EMAIL CONFIRMATION FLOW
        if context.get("email") and not context.get("email_confirmed"):
            extracted_email = extract_email_from_utterance(user_input)
            if extracted_email:
                context["email"] = extracted_email
                context["email_confirmed"] = False
                context["awaiting_email_confirmation"] = True
                context["email_confirm_attempts"] = 0

                spelled = spell_email_address(context["email"])
                ai_reply = f"I heard {spelled}. Is that correct?"
                context["history"].append(f"AI: {ai_reply}")
                return Response(
                    f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                    mimetype='text/xml'
                )

            if context.get("awaiting_email_confirmation") and is_affirmative(user_input):
                context["email_confirmed"] = True
                context["awaiting_email_confirmation"] = False
            elif context.get("awaiting_email_confirmation") and is_negative(user_input):
                # Track how many times user said "No" to email confirmation
                confirm_attempts = int(context.get("email_confirm_attempts", 0)) + 1
                context["email_confirm_attempts"] = confirm_attempts
                
                if confirm_attempts >= 2:
                    # Give up after 2 "No" responses
                    msg = "I am having a problem hearing you, please call later."
                    context["history"].append(f"AI: {msg}")
                    call_context.pop(call_sid, None)
                    return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')
                
                # Ask to repeat email
                context["email"] = ""
                context["email_confirmed"] = False
                context["awaiting_email_confirmation"] = False
                context["awaiting_email"] = True
                ai_reply = "Please repeat your email."
                context["history"].append(f"AI: {ai_reply}")
                return Response(
                    f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                    mimetype='text/xml'
                )
            else:
                spelled = spell_email_address(context.get("email", ""))
                ai_reply = f"I heard {spelled}. Is that correct?"
                context["history"].append(f"AI: {ai_reply}")
                return Response(
                    f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                    mimetype='text/xml'
                )

        # 5.5. ORDER NUMBER HANDLING (for order-related issues)
        if context.get("is_order_related_issue"):
            # Try to extract order number from current input
            extracted_order = extract_order_number(user_input)
            
            if context.get("awaiting_order_number") and not context.get("order_number"):
                if extracted_order:
                    context["order_number"] = extracted_order
                    context["awaiting_order_number"] = False
                    context["awaiting_order_confirmation"] = True
                    context["order_confirm_attempts"] = 0
                    ai_reply = f"I heard order number {extracted_order}. Is that correct?"
                    context["history"].append(f"AI: {ai_reply}")
                    return Response(
                        f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                        mimetype='text/xml'
                    )
                else:
                    ai_reply = "Please tell me your order number."
                    context["history"].append(f"AI: {ai_reply}")
                    return Response(
                        f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                        mimetype='text/xml'
                    )
            
            if context.get("order_number") and not context.get("order_number_confirmed"):
                if extracted_order and extracted_order != context.get("order_number"):
                    # User provided a different order number
                    context["order_number"] = extracted_order
                    context["awaiting_order_confirmation"] = True
                    context["order_confirm_attempts"] = 0
                    ai_reply = f"I heard order number {extracted_order}. Is that correct?"
                    context["history"].append(f"AI: {ai_reply}")
                    return Response(
                        f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                        mimetype='text/xml'
                    )
                
                if context.get("awaiting_order_confirmation") and is_affirmative(user_input):
                    context["order_number_confirmed"] = True
                    context["awaiting_order_confirmation"] = False
                    # Now ask for email if not already captured
                    if not context.get("email"):
                        context["awaiting_email"] = True
                        ai_reply = "Great. Now please provide your email address."
                        context["history"].append(f"AI: {ai_reply}")
                        return Response(
                            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                            mimetype='text/xml'
                        )
                elif context.get("awaiting_order_confirmation") and is_negative(user_input):
                    confirm_attempts = int(context.get("order_confirm_attempts", 0)) + 1
                    context["order_confirm_attempts"] = confirm_attempts
                    
                    if confirm_attempts >= 2:
                        msg = "I am having a problem hearing you, please call later."
                        context["history"].append(f"AI: {msg}")
                        call_context.pop(call_sid, None)
                        return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')
                    
                    context["order_number"] = ""
                    context["order_number_confirmed"] = False
                    context["awaiting_order_confirmation"] = False
                    context["awaiting_order_number"] = True
                    ai_reply = "Please repeat your order number."
                    context["history"].append(f"AI: {ai_reply}")
                    return Response(
                        f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                        mimetype='text/xml'
                    )
                elif context.get("awaiting_order_confirmation"):
                    ai_reply = f"I heard order number {context.get('order_number')}. Is that correct?"
                    context["history"].append(f"AI: {ai_reply}")
                    return Response(
                        f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                        mimetype='text/xml'
                    )

        # 6. LEGACY REGEX EMAIL EXTRACTION (Backup)
        extracted_email = extract_email_from_utterance(user_input)
        if not context.get("email") and extracted_email:
            context["email"] = extracted_email
            context["email_confirmed"] = False
            context["awaiting_email_confirmation"] = True
            context["email_request_attempts"] = 0

            spelled = spell_email_address(context["email"])
            ai_reply = f"I heard {spelled}. Is that correct?"
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )

        # 7. PROBLEM CAPTURE
        if not context.get("email") and not extracted_email:
            if not context.get("problem"):
                captured = normalize_problem_text(user_input.strip())
                
                # Check if this is a vague problem that needs more details
                if is_vague_problem(captured):
                    context["awaiting_problem_details"] = True
                    ai_reply = "Please describe the problem."
                    context["history"].append(f"AI: {ai_reply}")
                    return Response(
                        f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                        mimetype='text/xml'
                    )
                
                context["problem"] = captured
                context["awaiting_problem_details"] = False
                
                # Check if this is an order-related issue
                if is_order_related(captured):
                    context["is_order_related_issue"] = True
                    # Check if order number was already mentioned
                    extracted_order = extract_order_number(captured)
                    if extracted_order:
                        context["order_number"] = extracted_order
                        context["awaiting_order_confirmation"] = True
                        ai_reply = f"I heard order number {extracted_order}. Is that correct?"
                        context["history"].append(f"AI: {ai_reply}")
                        return Response(
                            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                            mimetype='text/xml'
                        )
                    else:
                        context["awaiting_order_number"] = True
                        ai_reply = "Please provide your order number."
                        context["history"].append(f"AI: {ai_reply}")
                        return Response(
                            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                            mimetype='text/xml'
                        )
                        
            elif context.get("awaiting_problem_details"):
                # User is providing details after we asked
                extra = normalize_problem_text(user_input.strip())
                if extra:
                    if context.get("problem"):
                        context["problem"] = f"{context['problem']} {extra}".strip()
                    else:
                        context["problem"] = extra
                context["awaiting_problem_details"] = False
                
                # Check if the detailed problem is order-related
                if is_order_related(context.get("problem", "")):
                    context["is_order_related_issue"] = True
                    extracted_order = extract_order_number(context.get("problem", ""))
                    if extracted_order:
                        context["order_number"] = extracted_order
                        context["awaiting_order_confirmation"] = True
                        ai_reply = f"I heard order number {extracted_order}. Is that correct?"
                        context["history"].append(f"AI: {ai_reply}")
                        return Response(
                            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                            mimetype='text/xml'
                        )
                    else:
                        context["awaiting_order_number"] = True
                        ai_reply = "Please provide your order number."
                        context["history"].append(f"AI: {ai_reply}")
                        return Response(
                            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                            mimetype='text/xml'
                        )
            else:
                extra = normalize_problem_text(user_input.strip())
                if extra and extra.lower() not in context["problem"].lower():
                    context["problem"] = f"{context['problem']} {extra}".strip()

        # 8. ASK FOR EMAIL IF PROBLEM IS CAPTURED
        if context.get("problem") and not context.get("email"):
            # For order-related issues, ensure order number is captured first
            if context.get("is_order_related_issue"):
                if not context.get("order_number"):
                    context["awaiting_order_number"] = True
                    ai_reply = "Please provide your order number."
                    context["history"].append(f"AI: {ai_reply}")
                    return Response(
                        f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                        mimetype='text/xml'
                    )
                if not context.get("order_number_confirmed"):
                    ai_reply = f"I heard order number {context.get('order_number')}. Is that correct?"
                    context["awaiting_order_confirmation"] = True
                    context["history"].append(f"AI: {ai_reply}")
                    return Response(
                        f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                        mimetype='text/xml'
                    )
            
            if context.get("email_declined"):
                msg = "I am having a problem hearing you, please call later."
                context["history"].append(f"AI: {msg}")
                call_context.pop(call_sid, None)
                return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')

            attempts = int(context.get("email_request_attempts", 0))
            ai_reply = "Please say your email one character at a time." if attempts >= 2 else "To proceed with your request, please provide me your email."
            context["email_request_attempts"] = attempts + 1
            context["awaiting_email"] = True
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )

        # 9. TICKET CREATION CHECK
        if context.get("problem") and context.get("email") and context.get("email_confirmed"):
            duration = time.time() - context["start_time"]
            total_cost = calculate_cost(duration, context.get("input_tokens", 0), context.get("output_tokens", 0))

            # Intelligently normalize the problem text using Gemini
            raw_problem = context.get("problem", "")
            cleaned_problem = intelligently_normalize_problem(raw_problem, context)
            
            # Include order number in ticket if present
            order_info = ""
            if context.get("order_number"):
                order_info = f"\n\nOrder Number: {context.get('order_number')}"
            
            transcript = "\n".join(context.get("history", []))
            ticket_id = create_zendesk_ticket(context["email"], f"{cleaned_problem}{order_info}\n\nTranscript:\n{transcript}", total_cost)

            if ticket_id:
                msg = "Thank you for calling Aerosus. Our team will look into your problem and get back to you as soon as possible. Have a nice day!"
            else:
                msg = "I could not create the ticket right now. Thank you for calling Aerosus. Please try again later. Have a nice day!"
            call_context.pop(call_sid, None)
            return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')

        # 10. GEMINI BRAIN
        prompt = f"""
You are a helpful customer support agent for Aerosus.

CRITICAL INSTRUCTION - LANGUAGE CHECK:
Analyze the user's last input: "{user_input}"
1. If the user is speaking a language other than English (e.g., Turkish, Spanish, German), output EXACTLY: ACTION_NON_ENGLISH
2. If the user is speaking English, proceed.

CRITICAL INSTRUCTION - EMAIL CAPTURE:
1. Check if the user provided an email address in the current input.
   - If YES, output ONLY: ACTION_CAPTURE_EMAIL: [the_email_address]
   - Example: ACTION_CAPTURE_EMAIL: akif@gmail.com
2. If NO email is found:
   - If the user says something vague like "there is a problem" or "I have an issue" without describing WHAT the problem is, ask EXACTLY: "Please describe the problem."
   - If the user HAS described the specific problem but has NOT provided an email, ask EXACTLY: "To proceed with your request, please provide me your email."

Context:
- "ear suspension" = "air suspension".
- "aero sus" = "Aerosus".
- Ticket guide: {TICKET_TYPE_GUIDE}

History:
{chr(10).join(context['history'])}

Known:
- Problem captured: {bool(context.get('problem'))}
- Email captured: {bool(context.get('email'))}

OUTPUT RULES:
- If capturing email, use the ACTION format.
- If foreign language, use ACTION_NON_ENGLISH.
- Otherwise, keep your response to one short sentence.
"""

        try:
            if model is None:
                raise RuntimeError("Gemini model not configured")
            response = model.generate_content(prompt)
            ai_reply = response.text.strip()

            if hasattr(response, 'usage_metadata'):
                context['input_tokens'] += response.usage_metadata.prompt_token_count
                context['output_tokens'] += response.usage_metadata.candidates_token_count

            if "ACTION_NON_ENGLISH" in ai_reply:
                context["english_warning_count"] = int(context.get("english_warning_count", 0)) + 1
                if context["english_warning_count"] >= 2:
                    msg = "This call can only be processed in English. Please try again later."
                    call_context.pop(call_sid, None)
                    return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')

                msg = "Please speak in English."
                context["history"].append(f"AI: {msg}")
                return Response(
                    f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{msg}</Say></Gather></Response>",
                    mimetype='text/xml'
                )

            if "ACTION_CAPTURE_EMAIL:" in ai_reply:
                captured_email = ai_reply.split("ACTION_CAPTURE_EMAIL:", 1)[1].strip().strip(".,;:!?")
                context["email"] = captured_email
                context["email_confirmed"] = False
                context["awaiting_email_confirmation"] = True
                context["awaiting_email"] = False
                context["email_request_attempts"] = 0

                spelled = spell_email_address(captured_email)
                ai_reply = f"I heard {spelled}. Is that correct?"
                context["history"].append(f"AI: {ai_reply}")
                return Response(
                    f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                    mimetype='text/xml'
                )

            if "provide me your email" in ai_reply.lower():
                context["email_request_attempts"] = int(context.get("email_request_attempts", 0)) + 1

        except Exception as e:
            debug_exception("gemini_error", call_sid=call_sid, err=e)
            if not context.get("problem"):
                ai_reply = "How may I help you?"
            elif not context.get("email"):
                ai_reply = "Ok to proceed with your request provide me your email."
            else:
                ai_reply = "Could you repeat that, please?"

        context["history"].append(f"AI: {ai_reply}")
        return Response(
            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
            mimetype='text/xml'
        )

    except Exception as e:
        debug_exception("voice_crash", call_sid=call_sid, err=e)
        call_context.pop(call_sid, None)
        return Response(
            f"<Response><Say voice='{VOICE_NAME}'>System error. Please call back later.</Say><Hangup/></Response>",
            mimetype='text/xml'
        )

if __name__ == "__main__":
    app.run(debug=True, port=8080)