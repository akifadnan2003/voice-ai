import os
import time
import requests
import base64
from flask import Flask, request, Response
import google.generativeai as genai

# NOTE: We removed 'dotenv' because Cloud Run provides variables automatically.
# If running locally, just ensure your .env variables are set in your terminal.

# --- CONFIGURATION ---
app = Flask(__name__)

# Credentials
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ZD_EMAIL = os.environ.get("ZD_EMAIL")
ZD_SUBDOMAIN = os.environ.get("ZD_SUBDOMAIN")
ZD_TOKEN = os.environ.get("ZD_TOKEN")

# Pricing Constants (USD)
PRICE_TWILIO_PER_MIN = 0.014
PRICE_GEMINI_INPUT_1K = 0.000075
PRICE_GEMINI_OUTPUT_1K = 0.00030

# Configure AI
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # SWITCHING MODEL to 'gemini-pro' to fix the 404 error
    model = genai.GenerativeModel("gemini-2.0-flash")
else:
    print("⚠️ WARNING: GEMINI_API_KEY is missing!")

call_context = {}

def calculate_cost(duration_sec, in_tok, out_tok):
    minutes = (duration_sec // 60) + 1
    twilio_cost = minutes * PRICE_TWILIO_PER_MIN
    ai_cost = (in_tok / 1000 * PRICE_GEMINI_INPUT_1K) + (out_tok / 1000 * PRICE_GEMINI_OUTPUT_1K)
    return round(twilio_cost + ai_cost, 4), round(twilio_cost, 4), round(ai_cost, 4)

def create_zendesk_ticket(user_email, issue_summary, cost_info):
    if not all([ZD_EMAIL, ZD_SUBDOMAIN, ZD_TOKEN]):
        print("❌ ERROR: Missing Zendesk Secrets")
        return None
    
    url = f"https://{ZD_SUBDOMAIN}.zendesk.com/api/v2/tickets.json"
    creds = f"{ZD_EMAIL}/token:{ZD_TOKEN}"
    encoded_creds = base64.b64encode(creds.encode()).decode()
    
    clean_email = user_email.replace("<", "").replace(">", "").strip()
    user_name = clean_email.split("@")[0]

    cost_report = (
        f"\n\n--- 💰 CALL COST REPORT ---\n"
        f"⏱️ Duration: {cost_info['duration']:.1f} sec\n"
        f"📞 Twilio:   ${cost_info['twilio_cost']}\n"
        f"🧠 AI Cost:  ${cost_info['ai_cost']}\n"
        f"💵 TOTAL:    ${cost_info['total_cost']}\n"
        f"--------------------------"
    )

    payload = {
        "ticket": {
            "subject": f"Voice AI: {issue_summary[:30]}...",
            "comment": { "body": f"User reported:\n{issue_summary}\n{cost_report}" },
            "requester": { "name": user_name, "email": clean_email }
        }
    }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Basic {encoded_creds}"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 201:
            print(f"✅ Ticket Created! Cost: ${cost_info['total_cost']}")
            return response.json()['ticket']['id']
        else:
            print(f"❌ Zendesk Error: {response.text}")
            return None
    except Exception as e:
        print(f"💥 Error: {e}")
        return None

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    call_sid = request.values.get('CallSid')
    user_input = request.values.get('SpeechResult')
    
    if call_sid not in call_context:
        call_context[call_sid] = {"history": [], "start_time": time.time(), "input_tokens": 0, "output_tokens": 0}
        return Response("<Response><Gather input='speech' timeout='3'><Say>Welcome to Aerosus Support. Briefly, what is the problem?</Say></Gather></Response>", mimetype='text/xml')

    if not user_input:
        return Response("<Response><Gather input='speech' timeout='3'><Say>Are you still there?</Say></Gather></Response>", mimetype='text/xml')

    context = call_context[call_sid]
    context['history'].append(f"User: {user_input}")
    
    prompt = f"""
    History: {context['history']}
    INSTRUCTIONS:
    1. If user has NO email, ask for it.
    2. If email given: Output ACTION_CREATE_TICKET: <email>
       - "at" -> "@", "dot" -> ".", "a k i f" -> "akif"
       - Remove spaces.
    3. Otherwise: Answer briefly.
    """
    
    try:
        response = model.generate_content(prompt)
        ai_reply = response.text.strip()
        
        # Safe token counting
        if hasattr(response, 'usage_metadata'):
            context['input_tokens'] += response.usage_metadata.prompt_token_count
            context['output_tokens'] += response.usage_metadata.candidates_token_count
            
    except Exception as e:
        print(f"⚠️ AI Generation Error: {e}")
        ai_reply = "I am having trouble connecting. Please try again later."

    if "ACTION_CREATE_TICKET" in ai_reply:
        try:
            raw_text = ai_reply.split("ACTION_CREATE_TICKET:")[1].strip()
            email = raw_text.split()[0].strip()
            
            duration = time.time() - context['start_time']
            total, twilio, ai = calculate_cost(duration, context['input_tokens'], context['output_tokens'])
            
            cost_data = {"duration": duration, "twilio_cost": twilio, "ai_cost": ai, "total_cost": total}
            
            ticket_id = create_zendesk_ticket(email, str(context['history']), cost_data)
            msg = f"Ticket {ticket_id} created. Cost was {total} dollars. Goodbye!" if ticket_id else "Ticket created internally. Goodbye."
        except:
            msg = "Ticket process failed. Please call back."
            
        del call_context[call_sid]
        return Response(f"<Response><Say>{msg}</Say></Response>", mimetype='text/xml')

    context['history'].append(f"AI: {ai_reply}")
    return Response(f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>", mimetype='text/xml')

if __name__ == "__main__":
    app.run(debug=True, port=8080)