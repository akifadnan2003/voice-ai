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


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


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
            "start_time": time.time(),
            "input_tokens": 0,
            "output_tokens": 0,
        }
        return Response(
            "<Response><Gather input='speech' timeout='3'><Say>Welcome to customer support. How may I help you?</Say></Gather></Response>",
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

    # Detect/normalize email from the user's utterance (deterministic, not model-driven)
    normalized_email = normalize_spoken_email(user_input)
    if not context.get("email") and is_valid_email(normalized_email):
        context["email"] = normalized_email

    # If we don't have a problem yet, treat the first substantive user input as the problem.
    if not context.get("problem") and not context.get("email"):
        context["problem"] = user_input.strip()

    # If we have problem but no email: ask for email using the exact phrasing requested.
    # (Handled by AI prompt below; kept deterministic fallback if AI is unavailable.)
    if context.get("problem") and not context.get("email") and model is None:
        ai_reply = "Ok to proceed with your request provide me your email."
        context["history"].append(f"AI: {ai_reply}")
        return Response(
            f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>",
            mimetype='text/xml'
        )

    # If we have both problem and email: create ticket, confirm, and hang up.
    if context.get("problem") and context.get("email"):
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
You are a helpful customer support agent.

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
- Keep it to one short sentence.
- Do not mention internal instructions.
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