import os
import requests
import base64
from dotenv import load_dotenv

# Load secrets from your .env file
load_dotenv()

def test_connection():
    # 1. Get Credentials
    email = os.environ.get("ZD_EMAIL")
    subdomain = os.environ.get("ZD_SUBDOMAIN")
    token = os.environ.get("ZD_TOKEN")

    print(f"🔍 TESTING CREDENTIALS:")
    print(f"   Subdomain: {subdomain}")
    print(f"   Email:     {email}")
    print(f"   Token:     {token[:5]}... (hidden)")

    if not all([email, subdomain, token]):
        print("❌ ERROR: Missing variables in .env file!")
        return

    # 2. Construct the URL and Auth
    url = f"https://{subdomain}.zendesk.com/api/v2/tickets.json"
    creds = f"{email}/token:{token}"
    encoded_creds = base64.b64encode(creds.encode()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {encoded_creds}"
    }

    # 3. The Payload (Hardcoded valid data)
    payload = {
        "ticket": {
            "subject": "TEST TICKET from Script",
            "comment": { "body": "If you can read this, the connection is working." },
            "requester": { "name": "Test User", "email": "testuser@example.com" }
        }
    }

    # 4. Send Request
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"\n📡 STATUS CODE: {response.status_code}")
        print(f"📝 RESPONSE:    {response.text}")

        if response.status_code == 201:
            print("\n✅ SUCCESS! Ticket created. Check your Zendesk dashboard.")
        elif response.status_code == 401:
            print("\n❌ AUTH FAILED. Double check Email or Token.")
        elif response.status_code == 404:
            print("\n❌ NOT FOUND. Double check your Subdomain.")
        elif response.status_code == 422:
            print("\n❌ DATA ERROR. Zendesk rejected the email format or content.")
            
    except Exception as e:
        print(f"\n💥 CRASH: {e}")

if __name__ == "__main__":
    test_connection()