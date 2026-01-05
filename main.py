import os
import requests
import base64
from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse
import google.generativeai as genai

app = Flask(__name__)

# CONFIGURATION
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ZD_EMAIL = os.environ.get("ZD_EMAIL")
ZD_SUBDOMAIN = os.environ.get("ZD_SUBDOMAIN")
ZD_TOKEN = os.environ.get("ZD_TOKEN")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('models/gemini-2.0-flash-exp') 

call_context = {"issue": ""}

# Replace your existing create_zendesk_ticket function with this:
def create_zendesk_ticket(user_email, issue_summary):
    if not all([ZD_EMAIL, ZD_SUBDOMAIN, ZD_TOKEN]):
        print("❌ ERROR: Missing ZD_EMAIL, ZD_SUBDOMAIN, or ZD_TOKEN")
        return None
    
    url = f"https://{ZD_SUBDOMAIN}.zendesk.com/api/v2/tickets.json"
    creds = f"{ZD_EMAIL}/token:{ZD_TOKEN}"
    encoded_creds = base64.b64encode(creds.encode()).decode()
    
    # Clean the email just in case
    user_email = user_email.strip().replace("<", "").replace(">", "")
    user_name = user_email.split("@")[0]
    
    payload = {
        "ticket": {
            "subject": f"Voice AI: {issue_summary[:30]}...",
            "comment": { "body": f"User reported:\n\n{issue_summary}" },
            "requester": { "name": user_name, "email": user_email }
        }
    }
    
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Basic {encoded_creds}"
    }
    
    try:
        print(f"🚀 Sending ticket to: {url}")
        print(f"📧 With email: {user_email}")
        
        response = requests.post(url, json=payload, headers=headers)
        
        # --- THE DEBUGGING LINES ---
        print(f"📡 Zendesk Status Code: {response.status_code}")
        print(f"📝 Zendesk Response: {response.text}")
        # ---------------------------

        if response.status_code == 201:
            return response.json()['ticket']['id']
        else:
            return None
            
    except Exception as e:
        print(f"💥 CRITICAL EXCEPTION: {e}")
        return None

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    resp = VoiceResponse()
    gather = resp.gather(input='speech', action='/process_speech', timeout=4)
    gather.say("Welcome to Aerosus Support. Briefly, what is the problem with your part?")
    resp.say("I didn't hear anything. Goodbye.")
    return str(resp)

@app.route("/process_speech", methods=['GET', 'POST'])
def process_speech():
    resp = VoiceResponse()
    user_input = request.values.get('SpeechResult')
    if not user_input:
        resp.redirect('/voice')
        return str(resp)
        
    if not call_context["issue"]:
        call_context["issue"] = user_input

    try:
        prompt = f"""
        User said: "{user_input}"
        Context: "{call_context['issue']}"
        
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
        """
        response = model.generate_content(prompt)
        ai_reply = response.text.strip().replace("*", "")
        
        if "ACTION_CREATE_TICKET" in ai_reply:
            try:
                email = ai_reply.split(":")[1].strip()
                ticket_id = create_zendesk_ticket(email, call_context["issue"])
                if ticket_id:
                    resp.say(f"Ticket {ticket_id} created. Goodbye!")
                else:
                    resp.say("Ticket created. Goodbye.")
            except:
                resp.say("I noted that. Goodbye.")
            resp.hangup()
        elif "TERMINATE" in ai_reply:
            resp.say("Goodbye!")
            resp.hangup()
        else:
            resp.say(ai_reply)
            resp.gather(input='speech', action='/process_speech', timeout=5)
    except:
        resp.say("System error. Please call back.")
    return str(resp)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

