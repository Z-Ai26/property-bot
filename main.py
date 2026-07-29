import os
import json
import requests
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from openai import OpenAI
import time  

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Memory Management (5 Minutes Session) ---
user_sessions = {}
SESSION_TIMEOUT = 300  # 5 minutes
# ---------------------------------------------

VERIFY_TOKEN = "mytoken777"
META_TOKEN = "EAAVwPvWmhzEBSFHS4nuobPutwJtXfiXgN7nRAHQPhdZBhnPZAYdeVUxhLuBXGispQ6W10CEHxfsE9VuxJvU8HYqVe8mISUARJpyVPQhTPUh78lPafRYsuK2FozI4c691GproYJyr1yd4AhAp31jHSuvCawx1QJ5gbye7jXOaNK1C36YNHcaIkGLtmIBwZDZD"
PHONE_NUMBER_ID = "1303976742789764"
GRAPH_VERSION = os.getenv("GRAPH_API_VERSION", "v22.0")

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1CWEDuKTtjhsNPY-8KnKeapha2q6H7s_ZY5Wzj4ldsA4/export?format=csv&gid=1443660362"
JSON_FILE = "knowledge.json"


def load_properties():
    # 1. Load the data directly from Google Sheets URL
    df = pd.read_csv(GOOGLE_SHEET_URL, on_bad_lines='skip')

    # 2. Set the first row as the proper header and clean up the structure
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)

    # 3. Exclude specific columns by their exact position/indices (I, J, K, L, N, O):
    columns_to_drop_indices = [8, 9, 10, 11, 13, 14]
    # Drop these columns safely
    df = df.drop(df.columns[columns_to_drop_indices], axis=1, errors='ignore')

    # 4. Remove any extra empty/NaN columns
    df = df.loc[:, df.columns.notna()]

    # Fill empty values and convert to dict for bot usage
    df = df.fillna("")
    return df.to_dict(orient="records")


def load_knowledge():
    with open(JSON_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def find_properties(question, properties):
    """
    Advanced search: Handles spaces, exact keyword matching,
    and prevents common words like 'Al' from ruining the search.
    """
    clean_question = question.lower().strip()
    words = clean_question.split()
    results = []

    for property_data in properties:
        # 1. Clean Excel Data: lower() and strip() removes all extra spaces from your sheet
        text = " ".join(str(value).lower().strip() for value in property_data.values())
        
        score = 0
        
        # 2. EXACT MATCH BOOST (Super Important)
        # If the user types "al nahda" and it exactly matches in the text, give it a massive score (100)
        if clean_question in text:
            score += 100

        # 3. SMART WORD-BY-WORD MATCH
        for word in words:
            if word in text:
                # Give very low points to common words so they don't overpower the main location
                if word in ["al", "in", "for", "the", "and"]:
                    score += 1
                else:
                    # Give higher points to specific keywords like "nahda", "studio", "fardan"
                    score += 10 

        if score > 0:
            results.append((score, property_data))

    # Sort the results so the highest scored properties come first
    results.sort(key=lambda item: item[0], reverse=True)
    
    # 4. INCREASED LIMIT: Changed from 10 to 40 so the AI gets all the buildings and units
    return [item[1] for item in results[:40]]


def create_ai_reply(sender, user_question):
    properties = load_properties()
    knowledge = load_knowledge()

    matches = find_properties(user_question, properties)

    context = {
        "matching_properties": matches,
        "company_knowledge": knowledge
    }

    current_time = time.time()
    
    if sender in user_sessions:
        if (current_time - user_sessions[sender]["last_updated"]) > SESSION_TIMEOUT:
            print(f"Session timeout for {sender}. Clearing history.")
            del user_sessions[sender]

    SYSTEM_PROMPT = """
    You are a professional and flawless real estate AI assistant in Dubai. 
    Your goal is to provide accurate information from the provided 'CONTEXT' database smoothly, without ever getting stuck in loops.

    CRITICAL RULES:
    1. ANALYZE THOROUGHLY: Read the ENTIRE context based on the user's query. 
    2. DO NOT HIDE DATA: If there are multiple matching units for a specific query, you MUST show ALL of them in a single message. Do not summarize or hide any available units.
    
    3. CONVERSATION LOGIC (FOLLOW STRICTLY): 
       - CONDITION A (Location Only): IF the user only provides a location and there are multiple buildings, DO NOT show unit details. Simply list the available building names and ask: "Which building would you like to explore?"
       - CONDITION B (Building Only): IF the user specifies a building but NOT the unit type (Studio, 1 BR, etc.), DO NOT show full details. List the available unit types in that building and ask: "Which unit type are you looking for?"
       - CONDITION C (Specific Details OR "Show All"): IF the user specifies BOTH the location/building AND unit type OR if they explicitly ask for all details, DO NOT ask any questions. IMMEDIATELY output the full details (Property Name, Unit No, Price, Size, Status) for ALL matching units.

    4. MISSING DATA: If the exact information or property is not found in the context, politely say: "For more information, please connect with Mr. Zahid at +971562625777."
    5. FORMATTING: Use clean bullet points and line breaks for readability on WhatsApp.
        """

    if sender not in user_sessions:
            user_sessions[sender] = {
                "history": [{"role": "system", "content": SYSTEM_PROMPT}],
                "last_updated": current_time
        }
        
    user_sessions[sender]["last_updated"] = current_time


    user_input = (
            f"CONTEXT:\n{json.dumps(context, ensure_ascii=False)}\n\n"
            f"CUSTOMER MESSAGE:\n{user_question}"
        )

    user_sessions[sender]["history"].append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=user_sessions[sender]["history"]
        )

    ai_reply = response.choices[0].message.content
    
    user_sessions[sender]["history"].append({"role": "assistant", "content": ai_reply})

    return ai_reply


def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {META_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": message
        }
    }

    result = requests.post(url, headers=headers, json=payload, timeout=30)
    print(result.status_code, result.text)


@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge)

    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request):
    data = await request.json()

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        messages = value.get("messages", [])

        if not messages:
            return {"status": "ignored"}

        message = messages[0]

        if message.get("type") != "text":
            return {"status": "unsupported_message_type"}

        sender = message["from"]
        user_text = message["text"]["body"]

        reply = create_ai_reply(sender, user_text)
        send_whatsapp_message(sender, reply)

    except Exception as error:
        print("Webhook error:", error)

    return {"status": "ok"}