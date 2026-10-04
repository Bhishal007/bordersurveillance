import requests
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize Local LLM via Ollama (100% offline)
llm = ChatOllama(model="llama3.2", temperature=0.2)

def evaluate_and_generate_sitrep(camera_id: str, event_type: str, object_class: str, confidence: float) -> dict:
    """
    Evaluates detection context and generates a tactical military SITREP.
    """
    prompt = f"""
You are an autonomous tactical border surveillance C2 AI agent.
Analyze the following sensor alert:
- Sector/Camera: {camera_id}
- Detection Type: {event_type}
- Target Object: {object_class}
- Sensor Confidence: {confidence * 100:.1f}%

Respond in this exact format:
DECISION: [ESCALATE or AUTO_RESOLVE]
SITREP: [A concise 1-2 sentence military situation report summarizing the threat level and recommended protocol]
"""
    try:
        response = llm.invoke([
            SystemMessage(content="You are a tactical command-and-control intelligence officer. Be concise, militaristic, and decisive."),
            HumanMessage(content=prompt)
        ])
        content = response.content.strip()
        
        is_auto_resolve = "DECISION: AUTO_RESOLVE" in content
        
        sitrep = "Tactical evaluation completed."
        for line in content.split("\n"):
            if "SITREP:" in line:
                sitrep = line.replace("SITREP:", "").strip()
                break
        
        if sitrep == "Tactical evaluation completed." and content:
            sitrep = content.replace("DECISION: ESCALATE", "").replace("DECISION: AUTO_RESOLVE", "").strip()

        return {
            "sitrep": sitrep,
            "auto_resolve": is_auto_resolve,
            "evaluated": True
        }
    except Exception as e:
        # Fallback if Ollama service is warming up
        return {
            "sitrep": f"Automated SITREP: {event_type} registered at sector {camera_id}. Confidence index {confidence*100:.0f}%.",
            "auto_resolve": False,
            "evaluated": False
        }

def chat_with_agent(user_query: str) -> str:
    """
    Answers commander questions by dynamically cross-referencing live MongoDB records.
    """
    recent_alerts_summary = "No recent sensor alerts found in ledger."
    try:
        # Fetch recent alerts from your Node.js backend
        r = requests.get("http://localhost:5000/api/v1/alerts", timeout=2)
        if r.status_code == 200:
            alerts = r.json()[:5]
            recent_alerts_summary = "\n".join([
                f"- Cam: {a.get('cameraId')}, Type: {a.get('eventType')}, Level: {a.get('threatLevel')}"
                for a in alerts
            ])
    except Exception:
        pass

    prompt = f"""
Current Tactical Database State (from MongoDB):
{recent_alerts_summary}

Commander Inquiry: "{user_query}"

Provide a tactical, authoritative C2 briefing response. Keep it to 2-3 sentences.
"""
    try:
        response = llm.invoke([
            SystemMessage(content="You are the C2 Tactical Autonomous Assistant. Answer operator inquiries authoritatively and concisely."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()
    except Exception as e:
        return "[AGENT OFFLINE]: Unable to process query. Verify Ollama is running."