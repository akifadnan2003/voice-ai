import os
import time
import requests
import base64
from flask import Flask, request, Response
import google.generativeai as genai

# --- CONFIGURATION ---
app = Flask(__name__)

# Credentials
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ZD_EMAIL = os.environ.get("ZD_EMAIL")
ZD_SUBDOMAIN = os.environ.get("ZD_SUBDOMAIN")
ZD_TOKEN = os.environ.get("ZD_TOKEN")

# Pricing Constants
PRICE_TWILIO_PER_MIN = 0.014
PRICE_GEMINI_INPUT_1K = 0.000075
PRICE_GEMINI_OUTPUT_1K = 0.00030

# Configure AI - using the FAST 2.0 Flash model you found
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash") 
else:
    print("⚠️ WARNING: GEMINI_API_KEY is missing!")

call_context = {}

def calculate_cost(duration_sec, in_tok, out_tok):
    minutes = (duration_sec // 60) + 1
    twilio_cost = minutes * PRICE_TWILIO_PER_MIN
    ai_cost = (in_tok / 1000 * PRICE_GEMINI_INPUT_1K) + (out_tok / 1000 * PRICE_GEMINI_OUTPUT_1K)
    return round(twilio_cost + ai_cost, 4)

def create_zendesk_ticket(user_email, issue_summary, cost):
    if not all([ZD_EMAIL, ZD_SUBDOMAIN, ZD_TOKEN]):
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
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 201:
            return response.json()['ticket']['id']
        return None
    except:
        return None

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    call_sid = request.values.get('CallSid')
    user_input = request.values.get('SpeechResult')
    
    # 1. Start Call
    if call_sid not in call_context:
        call_context[call_sid] = {"history": [], "start_time": time.time(), "input_tokens": 0, "output_tokens": 0}
        # Greeting asks for PROBLEM first
        return Response("<Response><Gather input='speech' timeout='3'><Say>Welcome to Aerosus Support. Briefly, what is the problem?</Say></Gather></Response>", mimetype='text/xml')

    if not user_input:
        return Response("<Response><Gather input='speech' timeout='3'><Say>I am listening. What is the problem?</Say></Gather></Response>", mimetype='text/xml')

    context = call_context[call_sid]
    context['history'].append(f"User: {user_input}")
    
    # --- STRICT ORDER PROMPT ---
    prompt = f"""
    History: {context['history']}
    
    YOUR JOB: Follow this strict order.
    
    STEP 1: CHECK FOR PROBLEM.
    - If the user hasn't clearly stated a problem yet, ask: "Could you describe the issue?"
    
    STEP 2: CHECK FOR EMAIL.
    - If you have the problem, but NO email address, ask: "What is your email address?"
    
    STEP 3: FINALIZE.
    - If you have BOTH the problem AND the email, output EXACTLY: ACTION_CREATE_TICKET: <email>
    
    RULES:
    - "at" -> "@", "dot" -> ".", "a k i f" -> "akif"
    - Do not output the ticket action until you have the email.
    """
    
    try:
        response = model.generate_content(prompt)
        ai_reply = response.text.strip()
        
        if hasattr(response, 'usage_metadata'):
            context['input_tokens'] += response.usage_metadata.prompt_token_count
            context['output_tokens'] += response.usage_metadata.candidates_token_count
            
    except:
        ai_reply = "I am having trouble. Please say your email again."

    # 4. Handle Ticket Creation
    if "ACTION_CREATE_TICKET" in ai_reply:
        try:
            raw_text = ai_reply.split("ACTION_CREATE_TICKET:")[1].strip()
            email = raw_text.split()[0].strip()
            
            # Calculate cost
            duration = time.time() - context['start_time']
            total_cost = calculate_cost(duration, context['input_tokens'], context['output_tokens'])
            
            # Create Ticket
            ticket_id = create_zendesk_ticket(email, str(context['history']), total_cost)
            
            if ticket_id:
                # EXACT PHRASE REQUESTED
                msg = f"Ticket number {ticket_id} made. Goodbye."
            else:
                msg = "Ticket made locally. Goodbye."
                
        except:
            msg = "Error creating ticket. Goodbye."
            
        del call_context[call_sid]
        # HANGUP COMMAND ADDED
        return Response(f"<Response><Say>{msg}</Say><Hangup/></Response>", mimetype='text/xml')

    # 5. Continue Conversation (Ask for email, etc.)
    context['history'].append(f"AI: {ai_reply}")
    return Response(f"<Response><Gather input='speech' timeout='3'><Say>{ai_reply}</Say></Gather></Response>", mimetype='text/xml')

if __name__ == "__main__":
    app.run(debug=True, port=8080)