import os
import time
import requests
import base64
from flask import Flask, request, Response
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
app = Flask(__name__)

# Credentials
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ZD_EMAIL = os.environ.get("ZD_EMAIL")
ZD_SUBDOMAIN = os.environ.get("ZD_SUBDOMAIN")
ZD_TOKEN = os.environ.get("ZD_TOKEN")

# Pricing Constants (USD)
PRICE_TWILIO_PER_MIN = 0.014  # Approx $0.014 per min
PRICE_GEMINI_INPUT_1K = 0.000075
PRICE_GEMINI_OUTPUT_1K = 0.00030

# Configure AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Memory Storage
# Format: { call_sid: { 'history': [], 'start_time': 0.0, 'input_tokens': 0, 'output_tokens': 0 } }
call_context = {}

def calculate_cost(duration_sec, in_tok, out_tok):
    """Calculates the estimated cost of the call."""
    # 1. Twilio Cost (Rounded up to nearest minute)
    minutes = (duration_sec // 60) + 1
    twilio_cost = minutes * PRICE_TWILIO_PER_MIN
    
    # 2. AI Cost
    ai_cost = (in_tok / 1000 * PRICE_GEMINI_INPUT_1K) + (out_tok / 1000 * PRICE_GEMINI_OUTPUT_1K)
    
    total = twilio_cost + ai_cost
    return round(total, 4), round(twilio_cost, 4), round(ai_cost, 4)

def create_zendesk_ticket(user_email, issue_summary, cost_info):
    """Creates a ticket in Zendesk using the requests library."""
    if not all([ZD_EMAIL, ZD_SUBDOMAIN, ZD_TOKEN]):
        print("❌ ERROR: Missing Zendesk Secrets in Environment Variables")
        return None
    
    url = f"https://{ZD_SUBDOMAIN}.zendesk.com/api/v2/tickets.json"
    
    # Auth String: email/token:api_token
    creds = f"{ZD_EMAIL}/token:{ZD_TOKEN}"
    encoded_creds = base64.b64encode(creds.encode()).decode()
    
    # Cost Breakdown String
    cost_report = (
        f"\n\n--- 💰 CALL COST REPORT ---\n"
        f"⏱️ Duration: {cost_info['duration']:.1f} sec\n"
        f"📞 Twilio:   ${cost_info['twilio_cost']}\n"
        f"🧠 AI Cost:  ${cost_info['ai_cost']} ({cost_info['tokens']} tokens)\n"
        f"💵 TOTAL:    ${cost_info['total_cost']}\n"
        f"--------------------------"
    )

    # Clean the email string just in case
    clean_email = user_email.replace("<", "").replace(">", "").strip()
    user_name = clean_email.split("@")[0]

    payload = {
        "ticket": {
            "subject": f"Voice AI: {issue_summary[:30]}...",
            "comment": { "body": f"User reported:\n{issue_summary}\n{cost_report}" },
            "requester": { "name": user_name, "email": clean_email }
        }
    }
    
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Basic {encoded_creds}"
    }
    
    try:
        print(f"🚀 Sending ticket for email: {clean_email}")
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 201:
            ticket_id = response.json()['ticket']['id']
            print(f"✅ Ticket #{ticket_id} created! Cost: ${cost_info['total_cost']}")
            return ticket_id
        else:
            print(f"❌ Zendesk Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"💥 Critical Error: {e}")
        return None

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    call_sid = request.values.get('CallSid')
    user_input = request.values.get('SpeechResult')
    
    # 1. Initialize Call (Start the Meter)
    if call_sid not in call_context:
        call_context[call_sid] = {
            "history": [], 
            "start_time": time.time(),
            "input_tokens": 0,
            "output_tokens": 0
        }
        # Initial Greeting
        twiml = "<Response><Gather input='speech' timeout='3'><Say>Welcome to Aerosus Support. Briefly, what is the problem?</Say></Gather></Response>"
        return Response(twiml, mimetype='text/xml')

    # 2. Handle User Silence
    if not user_input:
        twiml = "<Response><Gather input='speech' timeout='3'><Say>Are you still there?</Say></Gather></Response>"
        return Response(twiml, mimetype='text/xml')

    # 3. Chat with AI (and Count Tokens)
    context = call_context[call_sid]
    context['history'].append(f"User: {user_input}")
    
    # --- PROMPT LOGIC ---
    prompt = f"""
    History: {context['history']}
    
    INSTRUCTIONS:
    1. If the user has NOT provided an email yet, ask for it politely.
    2. If the user provided an email, you MUST output the command: ACTION_CREATE_TICKET: <email>
    
    CRITICAL RULES FOR EMAIL TRANSCRIPTION:
    - "at" -> "@"
    - "dot" -> "."
    - JOIN SPELLED LETTERS: If user spells "a k i f", output "akif".
    - REMOVE SPACES: The final email must have NO spaces.
    - Example: "j o h n at gmail dot com" -> "john@gmail.com"
    
    3. If the ticket is created, output: TERMINATE
    4. Otherwise: Answer briefly and helpfully.
    """
    
    response = model.generate_content(prompt)
    ai_reply = response.text.strip()
    
    # 💰 UPDATE TOKEN COUNTS
    if response.usage_metadata:
        context['input_tokens'] += response.usage_metadata.prompt_token_count
        context['output_tokens'] += response.usage_metadata.candidates_token_count

    # 4. Check for Ticket Creation Command
    if "ACTION_CREATE_TICKET" in ai_reply:
        try:
            # Extract Email: Split by the tag, then take the FIRST word to ignore "TERMINATE"
            raw_text = ai_reply.split("ACTION_CREATE_TICKET:")[1].strip()
            email = raw_text.split()[0].strip()  # <--- FIX: Removes 'TERMINATE' or trailing text
            
            # 🧾 CALCULATE FINAL BILL
            duration = time.time() - context['start_time']
            total, twilio, ai = calculate_cost(duration, context['input_tokens'], context['output_tokens'])
            
            cost_data = {
                "duration": duration,
                "twilio_cost": twilio,
                "ai_cost": ai,
                "total_cost": total,
                "tokens": context['input_tokens'] + context['output_tokens']
            }
            
            # Create Ticket
            ticket_id = create_zendesk_ticket(email, str(context['history']), cost_data)
            
            if ticket_id:
                msg = f"Ticket {ticket_id} created. The call cost was {total} dollars. Goodbye!"
            else:
                msg = "I created the ticket internally. Goodbye."
                
        except Exception as e:
            print(f"Error parsing email: {e}")
            msg = "I'm having trouble processing that email. Please call again."

        # Cleanup memory
        del call_context[call_sid] 
        return Response(f"<Response><Say>{msg}</Say></Response>", mimetype='text/xml')

    # 5. Continue Conversation if no ticket yet
    context['history'].append(f"AI: {ai_reply}")
    return Response(f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>", mimetype='text/xml')

if __name__ == "__main__":
    app.run(debug=True, port=8080)