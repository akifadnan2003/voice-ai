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

If unclear, ask a short clarifying question to categorize (order vs return/refund vs technical).
""".strip()


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
EMAIL_IN_TEXT_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.IGNORECASE)


AFFIRMATIVE_RE = re.compile(r"\b(yes|yeah|yep|correct|that's\s+right|that\s+is\s+right|right|sure|affirmative)\b", re.IGNORECASE)
NEGATIVE_RE = re.compile(r"\b(no|nope|nah|incorrect|wrong|negative|not\s+right)\b", re.IGNORECASE)


def normalize_spoken_email(text: str) -> str:
    """Best-effort conversion of spoken email to a real email string."""
    if not text:
        return ""

    cleaned = text.strip().lower()

    # Remove common punctuation that speech engines append.
    cleaned = cleaned.replace("<", "").replace(">", "").strip(" \t\r\n.,;:!?")

    # Handle common spoken patterns.
    cleaned = cleaned.replace(" at ", "@").replace(" dot ", ".")
    cleaned = cleaned.replace(" at", "@").replace("at ", "@")
    cleaned = cleaned.replace(" dot", ".").replace("dot ", ".")

    # Remove spaces that are often inserted between characters.
    cleaned = cleaned.replace(" ", "")

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
            "start_time": time.time(),
            "input_tokens": 0,
            "output_tokens": 0,
        }
        return Response(
            "<Response><Gather input='speech' timeout='3'><Say>Welcome to Aerosus customer support. How may I help you?</Say></Gather></Response>",
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
            f"<Response><Gather input='speech' timeout='3'><Say>{prompt_text}</Say></Gather></Response>",
            mimetype='text/xml'
        )

    # Track transcript
    context["history"].append(f"User: {user_input}")

    # If we already captured an email but haven't confirmed it yet, handle confirmation first.
    if context.get("email") and not context.get("email_confirmed"):
        # User can either answer yes/no, or repeat/provide a new email.
        extracted_email = extract_email_from_utterance(user_input)
        if extracted_email:
            context["email"] = extracted_email
            spelled = spell_email_address(context["email"])
            ai_reply = f"I heard {spelled}. Is that correct?"
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )

        if is_affirmative(user_input):
            context["email_confirmed"] = True
        elif is_negative(user_input):
            context["email"] = ""
            context["email_confirmed"] = False
            ai_reply = "Ok to proceed with your request provide me your email."
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )
        else:
            ai_reply = "Please say yes or no."
            context["history"].append(f"AI: {ai_reply}")
            return Response(
                f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>",
                mimetype='text/xml'
            )

    # Detect/normalize email from the user's utterance (deterministic, not model-driven)
    extracted_email = extract_email_from_utterance(user_input)
    if not context.get("email") and extracted_email:
        context["email"] = extracted_email

    # If we just captured an email (and it's not confirmed yet), confirm by spelling local-part up to '@'.
    if context.get("email") and not context.get("email_confirmed"):
        spelled = spell_email_address(context["email"])
        ai_reply = f"I heard {spelled}. Is that correct?"
        context["history"].append(f"AI: {ai_reply}")
        return Response(
            f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>",
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
            f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>",
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
    return Response(f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>", mimetype='text/xml')

if __name__ == "__main__":
    app.run(debug=True, port=8080)