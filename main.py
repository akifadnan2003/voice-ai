import os
import time
import requests
import base64
import re
import math
import traceback
from flask import Flask, request, Response
import google.generativeai as genai

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
    # European
    "hola", "gracias", "bonjour", "merci", "salut", "hallo", "danke", "bitte", 
    "ciao", "grazie", "ola", "obrigado", "pedido", "envio", "reembolso", 
    "devolucion", "commande", "livraison", "remboursement", "bestellung", 
    "lieferung", "rueckerstattung"
}

def is_likely_non_english(text: str) -> bool:
    if not text:
        return False
    # Don't block emails or numeric inputs
    if extract_email_from_utterance(text):
        return False
    if re.fullmatch(r"[\s0-9+\-#()]+", text.strip() or ""):
        return False
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
    if not email or "@" not in email: return ""
    local_part, domain = email.strip().lower().split("@", 1)
    token_map = {".": "dot", "_": "underscore", "-": "dash", "+": "plus"}
    common_domains = {"gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com"}
    
    spoken_local = []
    for ch in local_part:
        if ch.isalnum(): spoken_local.append(ch)
        elif ch in token_map: spoken_local.append(token_map[ch])
    
    if domain in common_domains:
        spoken_domain = domain
    else:
        spoken_domain_tokens = []
        for ch in domain:
            if ch.isalnum(): spoken_domain_tokens.append(ch)
            elif ch in token_map: spoken_domain_tokens.append(token_map[ch])
        spoken_domain = " ".join(spoken_domain_tokens)
        
    return f"{' '.join(spoken_local)} at {spoken_domain}".strip()

def normalize_problem_text(text: str) -> str:
    if not text: return ""
    normalized = text
    normalized = re.sub(r"\bear\s+suspension\b", "air suspension", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bair\s+suspensions\b", "air suspension", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\baero\s+sus\b", "Aerosus", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bcompress\s+or\b", "compressor", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bshock\s+absorber\b", "shock absorber", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\br\s*m\s*a\b", "RMA", normalized, flags=re.IGNORECASE)
    return normalized

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

# --- MAIN ROUTE ---
@app.route("/voice", methods=['GET', 'POST'])
def voice():
    call_sid = request.values.get('CallSid')
    user_input = request.values.get('SpeechResult')
    debug_event("incoming", call_sid=call_sid, has_speech=bool(user_input))
    
    # 1. INITIALIZE CONTEXT
    if call_sid not in call_context:
        call_context[call_sid] = {
            "history": [],
            "problem": "",
            "email": "",
            "email_confirmed": False,
            "awaiting_email_confirmation": False,
            "email_confirm_attempts": 0,
            "email_request_attempts": 0,
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
            attempts = int(context.get("email_request_attempts", 0))
            if attempts >= 2:
                prompt_text = "Please say your email one character at a time."
            else:
                prompt_text = "Ok to proceed with your request provide me your email."
            context["email_request_attempts"] = attempts + 1
        else:
            prompt_text = "I am listening. What is the problem?"
        
        return Response(
            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{prompt_text}</Say></Gather></Response>",
            mimetype='text/xml'
        )

    # 3. LANGUAGE CHECK (LAYER 1 - REGEX)
    # Check if the user is clearly speaking a foreign language
    if is_likely_non_english(user_input):
        context["english_warning_count"] = int(context.get("english_warning_count", 0)) + 1
        
        # STRIKE 2
        if context["english_warning_count"] >= 2:
            msg = "This call can only be processed in English. Please try again later."
            del call_context[call_sid]
            return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')
        
        # STRIKE 1
        msg = "Please speak in English."
        context["history"].append(f"AI: {msg}")
        return Response(
            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{msg}</Say></Gather></Response>",
            mimetype='text/xml'
        )
    else:
        # Reset counter on valid English input to avoid false flags accumulating
        context["english_warning_count"] = 0

    # 4. TRANSCRIPT LOGGING
    context["history"].append(f"User: {user_input}")

    # 5. EMAIL CONFIRMATION FLOW (If we already captured an email)
    if context.get("email") and not context.get("email_confirmed"):
        # Check if user is confirming or replacing
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
            context["email"] = ""
            context["email_confirmed"] = False
            context["awaiting_email_confirmation"] = False
            ai_reply = "Ok to proceed with your request provide me your email."
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )
        else:
            # Ambiguous response -> Re-verify
            spelled = spell_email_address(context.get("email", ""))
            ai_reply = f"I heard {spelled}. Is that correct?"
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
        context["email_request_attempts"] = 0 # Reset nagging
        
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
            context["problem"] = normalize_problem_text(user_input.strip())
        else:
            extra = normalize_problem_text(user_input.strip())
            if extra and extra.lower() not in context["problem"].lower():
                context["problem"] = f"{context['problem']} {extra}".strip()

    # 8. HARD STOP: NAGGING FOR EMAIL
    # If we have the problem but no email, we force the flow to ask for email.
    if context.get("problem") and not context.get("email"):
        attempts = int(context.get("email_request_attempts", 0))
        if attempts >= 2:
            ai_reply = "Please say your email one character at a time."
        else:
            ai_reply = "Ok to proceed with your request provide me your email."
        
        # Don't increment attempts here, we increment in the NO_INPUT or GEMINI section
        # actually, let's let Gemini handle the 'asking', but we inject the intent in prompt.
        
        # We allow Gemini to handle the phrasing, but we want to prevent it from ignoring the mission.
        # Flow continues to Gemini...

    # 9. TICKET CREATION CHECK
    if context.get("problem") and context.get("email") and context.get("email_confirmed"):
        duration = time.time() - context["start_time"]
        total_cost = calculate_cost(duration, context.get("input_tokens", 0), context.get("output_tokens", 0))

        transcript = "\n".join(context.get("history", []))
        issue_text = context.get("problem", "")
        ticket_id = create_zendesk_ticket(context["email"], f"{issue_text}\n\nTranscript:\n{transcript}", total_cost)

        msg = "Ok ticket created. Closing the call." if ticket_id else "I could not create the ticket right now. Closing the call."
        del call_context[call_sid]
        return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')

    # 10. GEMINI BRAIN (With Double Defense & Smart Email Extraction)
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
   - If the user has NOT described the problem yet, ask a short question to get the problem.
   - If the user HAS described the problem but has NOT provided an email, ask EXACTLY: "Ok to proceed with your request provide me your email."

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
        response = model.generate_content(prompt)
        ai_reply = response.text.strip()
        
        if hasattr(response, 'usage_metadata'):
            context['input_tokens'] += response.usage_metadata.prompt_token_count
            context['output_tokens'] += response.usage_metadata.candidates_token_count

        # --- LOGIC: HANDLE GEMINI ACTIONS ---

        # 1. Foreign Language Flag (Layer 2)
        if "ACTION_NON_ENGLISH" in ai_reply:
            context["english_warning_count"] = int(context.get("english_warning_count", 0)) + 1
            if context["english_warning_count"] >= 2:
                msg = "This call can only be processed in English. Please try again later."
                del call_context[call_sid]
                return Response(f"<Response><Say voice='{VOICE_NAME}'>{msg}</Say><Hangup/></Response>", mimetype='text/xml')
            
            msg = "Please speak in English."
            context["history"].append(f"AI: {msg}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{msg}</Say></Gather></Response>",
                mimetype='text/xml'
            )

        # 2. Smart Email Extraction
        if "ACTION_CAPTURE_EMAIL:" in ai_reply:
            captured_email = ai_reply.split("ACTION_CAPTURE_EMAIL:", 1)[1].strip()
            captured_email = captured_email.strip(".,;:!?") # Clean extraction
            
            # Store it
            context["email"] = captured_email
            context["email_confirmed"] = False
            context["awaiting_email_confirmation"] = True
            context["email_request_attempts"] = 0
            
            spelled = spell_email_address(captured_email)
            ai_reply = f"I heard {spelled}. Is that correct?"
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say voice='{VOICE_NAME}'>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )

        # 3. Handle Nagging Counter (If AI asked for email standardly)
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

if __name__ == "__main__":
    app.run(debug=True, port=8080)