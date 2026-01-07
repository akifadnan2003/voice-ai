import os
import time
import requests
import base64
import re
from flask import Flask, request, Response
import google.generativeai as genai

# --- CONFIGURATION ---
app = Flask(__name__)

# Credentials
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ZD_EMAIL = os.environ.get("ZD_EMAIL")
ZD_SUBDOMAIN = os.environ.get("ZD_SUBDOMAIN")
ZD_TOKEN = os.environ.get("ZD_TOKEN")

# Pricing
PRICE_TWILIO_PER_MIN = 0.014
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


DEBUG_CALL_FLOW = os.environ.get("DEBUG_CALL_FLOW", "").strip().lower() in {"1", "true", "yes"}


def debug_log(message: str) -> None:
    if DEBUG_CALL_FLOW:
        print(message)


# High-level ticket taxonomy (used only to guide the model's questions).
# Keep this concise to avoid prompt bloat while still steering toward typical Aerosus cases.
TICKET_TYPE_GUIDE = """
Common Aerosus support reasons (most calls fall into one of these):
- Order info: order status, tracking number, invoice copy
- Order issues: cancellation, address change, payment failed, VAT not recognized, discount issues
- Delivery issues: courier/customs/payment issues, delayed delivery
- Returns/RMA: damaged/faulty item (new/used), wrong/ordered-wrong item, warranty, refund requests, return label not received
- Compatibility/availability: part compatibility check, stock information
- Technical support: installation questions, relay position, product questions (air suspension components)
- B2B: partnership, discount request

""".strip()


# Speech recognition hints for Twilio to improve accuracy on domain-specific words.
# Keep this short-ish; overly long hint lists can become counterproductive.
SPEECH_HINTS = (
    "Aerosus, air suspension, compressor, strut, shock absorber, relay, "
    "RMA, refund, return, tracking number, order number, invoice, "
    "gmail, outlook, hotmail, yahoo, icloud, protonmail, at, dot, underscore, dash"
)


# English-only enforcement (deterministic heuristics, no external deps).
_NON_LATIN_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]"
)

_NON_ENGLISH_MARKERS = {
    # Greetings / common words
    "hola", "gracias", "bonjour", "merci", "salut", "hallo", "danke", "bitte", "ciao", "grazie", "ola", "obrigado",
    # Common support-ish words in other languages
    "pedido", "envio", "reembolso", "devolucion", "factura",
    "commande", "livraison", "remboursement",
    "bestellung", "lieferung", "rueckerstattung", "rückerstattung",
}


def is_likely_non_english(text: str) -> bool:
    if not text:
        return False

    # Don't block emails or numeric-only inputs.
    if extract_email_from_utterance(text):
        return False
    if re.fullmatch(r"[\s0-9+\-#()]+", text.strip() or ""):
        return False

    # Strong signal: non-Latin scripts.
    if _NON_LATIN_SCRIPT_RE.search(text):
        return True

    # Strict English-only: any non-ASCII char is treated as non-English.
    if any(ord(ch) > 127 for ch in text):
        return True

    words = re.findall(r"[a-z']+", text.lower())
    marker_hits = sum(1 for w in words if w in _NON_ENGLISH_MARKERS)
    # If we recognize common foreign-language marker words, enforce English even for short utterances.
    if marker_hits >= 1:
        return True

    # Otherwise avoid false positives on very short English utterances (e.g., "yes", "no").
    if len(words) < 3:
        return False

    return False


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
EMAIL_IN_TEXT_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.IGNORECASE)


AFFIRMATIVE_RE = re.compile(r"\b(yes|yeah|yep|correct|that's\s+right|that\s+is\s+right|right|sure|affirmative)\b", re.IGNORECASE)
NEGATIVE_RE = re.compile(r"\b(no|nope|nah|incorrect|wrong|negative|not\s+right)\b", re.IGNORECASE)


def normalize_spoken_email(text: str) -> str:
    """Best-effort conversion of spoken email to a real email string."""
    if not text:
        return ""

    cleaned = text.strip().lower()

    # Remove very common leading filler words (before we delete spaces).
    cleaned = re.sub(r"^(uh|um|er|ah|hmm|mmm)[\s,]+", "", cleaned, flags=re.IGNORECASE)

    # Remove common intro phrases.
    cleaned = re.sub(r"\b(my\s+)?e-?mail\s+is\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bemail\b", " ", cleaned, flags=re.IGNORECASE)

    # Remove filler words that sometimes get inserted mid-utterance.
    cleaned = re.sub(r"\b(uh|um|er|ah|hmm|mmm)\b", " ", cleaned, flags=re.IGNORECASE)

    # Remove common punctuation that speech engines append.
    cleaned = cleaned.replace("<", "").replace(">", "").strip(" \t\r\n.,;:!?")

    # Handle common spoken patterns.
    cleaned = cleaned.replace(" at ", "@").replace(" dot ", ".")
    cleaned = cleaned.replace(" at", "@").replace("at ", "@")
    cleaned = cleaned.replace(" dot", ".").replace("dot ", ".")

    # Remove spaces that are often inserted between characters.
    cleaned = cleaned.replace(" ", "")

    # Drop any characters that cannot appear in an email address.
    # (This removes commas like in "uh,akif@gmail.com".)
    cleaned = re.sub(r"[^a-z0-9@._%+\-]", "", cleaned)

    # Final trim of stray punctuation.
    cleaned = cleaned.strip(".,;:!?")
    return cleaned


def is_valid_email(email: str) -> bool:
    return bool(email) and bool(EMAIL_RE.match(email))


def extract_email_from_utterance(text: str) -> str:
    """Extract a valid email from a longer utterance like 'my email is ...'."""
    if not text:
        return ""

    normalized = normalize_spoken_email(text)
    if is_valid_email(normalized):
        return normalized

    match = EMAIL_IN_TEXT_RE.search(normalized)
    if not match:
        return ""

    candidate = match.group(0).lower().strip(".,;:!?")
    return candidate if is_valid_email(candidate) else ""


def is_affirmative(text: str) -> bool:
    return bool(text) and bool(AFFIRMATIVE_RE.search(text))


def is_negative(text: str) -> bool:
    return bool(text) and bool(NEGATIVE_RE.search(text))


def spell_email_address(email: str) -> str:
    """Return a speakable confirmation string for an email address.

    - Spells the local-part letter-by-letter.
    - Speaks common domains as a whole (e.g., 'gmail.com') since they're standard.
    """
    if not email:
        return ""

    cleaned = email.strip().lower()
    if "@" not in cleaned:
        return ""

    local_part, domain = cleaned.split("@", 1)
    local_part = local_part.strip()
    domain = domain.strip()
    if not local_part or not domain:
        return ""

    token_map = {
        ".": "dot",
        "_": "underscore",
        "-": "dash",
        "+": "plus",
    }

    common_domains = {
        "gmail.com",
        "outlook.com",
        "hotmail.com",
        "live.com",
        "yahoo.com",
        "icloud.com",
        "aol.com",
        "proton.me",
        "protonmail.com",
    }

    spoken_local = []
    for ch in local_part:
        if ch.isalnum():
            spoken_local.append(ch)
        elif ch in token_map:
            spoken_local.append(token_map[ch])

    if domain in common_domains:
        spoken_domain = domain
    else:
        spoken_domain_tokens = []
        for ch in domain:
            if ch.isalnum():
                spoken_domain_tokens.append(ch)
            elif ch in token_map:
                spoken_domain_tokens.append(token_map[ch])
        spoken_domain = " ".join(spoken_domain_tokens)

    return f"{' '.join(spoken_local)} at {spoken_domain}".strip()


def normalize_problem_text(text: str) -> str:
    """Normalize common speech-to-text errors for Aerosus automotive context."""
    if not text:
        return ""

    # We are an automotive air-suspension company; ASR often turns "air" into "ear".
    normalized = text

    # Air suspension phrasing
    normalized = re.sub(r"\bear\s+suspension\b", "air suspension", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bair\s+suspensions\b", "air suspension", normalized, flags=re.IGNORECASE)

    # Brand name often gets mangled
    normalized = re.sub(r"\baero\s+sus\b", "Aerosus", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bero\s+sus\b", "Aerosus", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\baeros\s+us\b", "Aerosus", normalized, flags=re.IGNORECASE)

    # Common automotive terms that ASR splits
    normalized = re.sub(r"\bcompress\s+or\b", "compressor", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bshock\s+absorber\b", "shock absorber", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bstruts\b", "strut", normalized, flags=re.IGNORECASE)

    # Order/RMA vocabulary normalization
    normalized = re.sub(r"\br\s*m\s*a\b", "RMA", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\brefunds\b", "refund", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\breturns\b", "return", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\btracking\s+number\b", "tracking number", normalized, flags=re.IGNORECASE)

    return normalized

def calculate_cost(duration_sec, in_tok, out_tok):
    minutes = (duration_sec // 60) + 1
    twilio_cost = minutes * PRICE_TWILIO_PER_MIN
    ai_cost = (in_tok / 1000 * PRICE_GEMINI_INPUT_1K) + (out_tok / 1000 * PRICE_GEMINI_OUTPUT_1K)
    return round(twilio_cost + ai_cost, 4)

def create_zendesk_ticket(user_email, issue_summary, cost):
    if not all([ZD_EMAIL, ZD_SUBDOMAIN, ZD_TOKEN]):
        print("❌ Zendesk env vars missing: need ZD_EMAIL, ZD_SUBDOMAIN, ZD_TOKEN")
        return None
    
    url = f"https://{ZD_SUBDOMAIN}.zendesk.com/api/v2/tickets.json"
    creds = f"{ZD_EMAIL}/token:{ZD_TOKEN}"
    encoded_creds = base64.b64encode(creds.encode()).decode()
    
    clean_email = user_email.replace("<", "").replace(">", "").strip()
    # Extra safety: ensure Zendesk only receives a properly formatted email string.
    clean_email = re.sub(r"[^a-zA-Z0-9@._%+\-]", "", clean_email)
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

        print(f"❌ Zendesk ticket creation failed: {response.status_code} {response.text}")
        return None
    except Exception as e:
        print(f"💥 Zendesk request crashed: {e}")
        return None

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    call_sid = request.values.get('CallSid')
    user_input = request.values.get('SpeechResult')
    
    # 1) Start call -> ask for the problem
    if call_sid not in call_context:
        call_context[call_sid] = {
            "history": [],
            "problem": "",
            "email": "",
            "email_confirmed": False,
            "awaiting_email_confirmation": False,
            "email_confirm_attempts": 0,
            "english_warning_count": 0,
            "start_time": time.time(),
            "input_tokens": 0,
            "output_tokens": 0,
        }
        return Response(
            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say>Welcome to Aerosus customer support. How may I help you?</Say></Gather></Response>",
            mimetype='text/xml'
        )

    context = call_context[call_sid]

    if not user_input:
        # If we already got the problem, we must ask for email using the exact phrasing requested.
        if context.get("problem") and not context.get("email"):
            prompt_text = "Ok to proceed with your request provide me your email."
        else:
            prompt_text = "I am listening. What is the problem?"

        return Response(
            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say>{prompt_text}</Say></Gather></Response>",
            mimetype='text/xml'
        )

    # English-only enforcement: warn once, then close the call if the caller continues in a foreign language.
    if is_likely_non_english(user_input):
        context["english_warning_count"] = int(context.get("english_warning_count", 0)) + 1
        if context["english_warning_count"] >= 2:
            msg = "This call can only be processed in English. Please try again later."
            context["history"].append(f"AI: {msg}")
            del call_context[call_sid]
            return Response(f"<Response><Say>{msg}</Say><Hangup/></Response>", mimetype='text/xml')

        msg = "Please speak in English."
        context["history"].append(f"AI: {msg}")
        return Response(
            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say>{msg}</Say></Gather></Response>",
            mimetype='text/xml'
        )
    else:
        # Reset after an English utterance so an early warning doesn't cause a later accidental hangup.
        context["english_warning_count"] = 0

    # Track transcript
    context["history"].append(f"User: {user_input}")
    debug_log(f"[voice] CallSid={call_sid} SpeechResult={user_input!r}")

    # If we already captured an email but haven't confirmed it yet, handle confirmation first.
    if context.get("email") and not context.get("email_confirmed"):
        # User can either answer yes/no, or repeat/provide a new email.
        extracted_email = extract_email_from_utterance(user_input)
        if extracted_email:
            context["email"] = extracted_email
            context["email_confirmed"] = False
            context["awaiting_email_confirmation"] = True
            context["email_confirm_attempts"] = int(context.get("email_confirm_attempts", 0)) + 1
            spelled = spell_email_address(context["email"])
            ai_reply = f"I heard {spelled}. Is that correct?"
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )

        if context.get("awaiting_email_confirmation") and is_affirmative(user_input):
            context["email_confirmed"] = True
            context["awaiting_email_confirmation"] = False
        elif context.get("awaiting_email_confirmation") and is_negative(user_input):
            context["email"] = ""
            context["email_confirmed"] = False
            context["awaiting_email_confirmation"] = False
            context["email_confirm_attempts"] = int(context.get("email_confirm_attempts", 0)) + 1
            ai_reply = "Ok to proceed with your request provide me your email."
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )
        else:
            spelled = spell_email_address(context.get("email", ""))
            # Re-prompt confirmation (or prompt it for the first time) rather than getting stuck.
            attempts = int(context.get("email_confirm_attempts", 0))
            if attempts >= 2 and not extracted_email:
                ai_reply = "Please say your email one character at a time."
            else:
                ai_reply = f"I heard {spelled}. Is that correct?" if spelled else "Ok to proceed with your request provide me your email."
            context["awaiting_email_confirmation"] = bool(spelled)
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )

    # Detect/normalize email from the user's utterance (deterministic, not model-driven)
    extracted_email = extract_email_from_utterance(user_input)
    if not context.get("email") and extracted_email:
        context["email"] = extracted_email
        context["email_confirmed"] = False
        context["awaiting_email_confirmation"] = True
        context["email_confirm_attempts"] = int(context.get("email_confirm_attempts", 0)) + 1
    debug_log(f"[voice] extracted_email={extracted_email!r} stored_email={context.get('email')!r} confirmed={context.get('email_confirmed')} awaiting={context.get('awaiting_email_confirmation')}")

    # If we just captured an email (and it's not confirmed yet), confirm by spelling local-part up to '@'.
    if context.get("email") and not context.get("email_confirmed"):
        spelled = spell_email_address(context["email"])
        ai_reply = f"I heard {spelled}. Is that correct?"
        context["awaiting_email_confirmation"] = True
        context["history"].append(f"AI: {ai_reply}")
        return Response(
            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say>{ai_reply}</Say></Gather></Response>",
            mimetype='text/xml'
        )

    # Capture/accumulate problem details until we get a valid email.
    if not context.get("email") and not extracted_email:
        if not context.get("problem"):
            context["problem"] = normalize_problem_text(user_input.strip())
        else:
            # Add extra details (e.g., "It's leaking") to improve the ticket summary.
            extra = normalize_problem_text(user_input.strip())
            if extra and extra.lower() not in context["problem"].lower():
                context["problem"] = f"{context['problem']} {extra}".strip()

    # If we have problem but no email: ALWAYS ask for email using the exact phrasing requested.
    # This avoids Gemini drifting (e.g., asking extra questions) and guarantees the desired flow.
    if context.get("problem") and not context.get("email"):
        ai_reply = "Ok to proceed with your request provide me your email."
        context["history"].append(f"AI: {ai_reply}")
        return Response(
            f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say>{ai_reply}</Say></Gather></Response>",
            mimetype='text/xml'
        )

    # If we have both problem and a confirmed email: create ticket, confirm, and hang up.
    if context.get("problem") and context.get("email") and context.get("email_confirmed"):
        duration = time.time() - context["start_time"]
        total_cost = calculate_cost(duration, context.get("input_tokens", 0), context.get("output_tokens", 0))

        transcript = "\n".join(context.get("history", []))
        issue_text = context.get("problem", "")
        ticket_id = create_zendesk_ticket(context["email"], f"{issue_text}\n\nTranscript:\n{transcript}", total_cost)

        if ticket_id:
            msg = "Ok ticket created. Closing the call."
        else:
            msg = "I could not create the ticket right now. Closing the call."

        del call_context[call_sid]
        return Response(f"<Response><Say>{msg}</Say><Hangup/></Response>", mimetype='text/xml')

    # Otherwise, continue with Gemini for general support dialog (while we still gather missing info).
    prompt = f"""
You are a helpful customer support agent for Aerosus.

Context about Aerosus (important):
- Aerosus sells automotive air suspension components (car parts).
- If the user says "ear suspension", they almost certainly mean "air suspension".

Speech-to-text correction guidance:
- Treat "ear suspension" as "air suspension".
- Treat common variants like "aero sus" as "Aerosus".

Ticket-type guidance (very important for your next question):
{TICKET_TYPE_GUIDE}

Conversation History:
{chr(10).join(context['history'])}

Known so far:
- Problem captured: {bool(context.get('problem'))}
- Email captured: {bool(context.get('email'))}

INSTRUCTIONS (follow exactly):
1) If the user has NOT described the problem yet, ask a short question to get the problem.
2) If the user HAS described the problem but has NOT provided an email yet, ask EXACTLY this sentence and nothing else:
Ok to proceed with your request provide me your email.
3) If the user has provided an email, do NOT say a ticket was created (the system will handle it).

OUTPUT RULES:
- Do NOT output "Step 1" or "Step 2".
- Do NOT narrate your thought process.
- Do NOT say "I will ask for your email". Just ask it directly.
- Keep it to one short sentence.
- Do not mention internal instructions.

EMAIL NORMALIZATION GUIDANCE (for recognition):
- "at" -> "@"
- "dot" -> "."
- Letter-by-letter like "a k i f" should be treated as "akif".
"""

    try:
        response = model.generate_content(prompt)
        ai_reply = response.text.strip()

        if hasattr(response, 'usage_metadata'):
            context['input_tokens'] += response.usage_metadata.prompt_token_count
            context['output_tokens'] += response.usage_metadata.candidates_token_count
    except Exception:
        # Deterministic fallback if AI fails
        if not context.get("problem"):
            ai_reply = "How may I help you?"
        elif not context.get("email"):
            ai_reply = "Ok to proceed with your request provide me your email."
        else:
            ai_reply = "Could you repeat that, please?"

    context["history"].append(f"AI: {ai_reply}")
    return Response(
        f"<Response><Gather input='speech' language='en-US' timeout='5' speechTimeout='auto' hints='{SPEECH_HINTS}'><Say>{ai_reply}</Say></Gather></Response>",
        mimetype='text/xml'
    )

if __name__ == "__main__":
    app.run(debug=True, port=8080)