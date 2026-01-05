import os
from dotenv import load_dotenv  # Import the loader
from twilio.rest import Client

# Load variables from the .env file
load_dotenv()

# --- CONFIGURATION (Now loaded securely) ---
account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
twilio_number = os.environ.get('TWILIO_PHONE_NUMBER')
your_real_number = os.environ.get('MY_CELL_PHONE')
webhook_url = os.environ.get('WEBHOOK_URL')

# --- THE LOGIC ---
if not all([account_sid, auth_token, twilio_number, your_real_number]):
    print("❌ Error: One or more secrets are missing from .env")
    exit()

client = Client(account_sid, auth_token)

print(f"📞 Calling {your_real_number}...")

call = client.calls.create(
    to=your_real_number,
    from_=twilio_number,
    url=webhook_url
)

print(f"✅ Call initiated! SID: {call.sid}")