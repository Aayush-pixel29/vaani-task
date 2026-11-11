# export_chat_history.py
import json
import os
import re

# Groq (gsk_), OpenAI-style (sk-), GitHub tokens (ghp_/github_pat_), Bearer headers
SECRET_RE = re.compile(r"(gsk_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Bearer\s+[A-Za-z0-9._\-]{20,})")

TRANSCRIPT_PATH = r"C:\Users\Aayush\.gemini\antigravity\brain\74bc0d3c-c4e0-47aa-a509-d7f585515918\.system_generated\logs\transcript.jsonl"
OUT_PATH = "assistant_chat_history.txt"

def export():
    if not os.path.exists(TRANSCRIPT_PATH):
        print(f"Transcript not found at {TRANSCRIPT_PATH}")
        return

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f_in, open(OUT_PATH, "w", encoding="utf-8") as f_out:
        f_out.write("================================================================================\n")
        f_out.write("THE RELAY ASSIGNMENT — CONVERSATION & PROMPT HISTORY WITH ASSISTANT\n")
        f_out.write("Candidate: Aayush Shelar (shelaraayush535@gmail.com)\n")
        f_out.write("================================================================================\n\n")

        for line in f_in:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                step_idx = obj.get("step_index", "?")
                source = obj.get("source", "UNKNOWN")
                step_type = obj.get("type", "UNKNOWN")
                ts = obj.get("created_at", "")
                content = obj.get("content", "")
                # Redact any API keys / secrets before they ever hit disk
                content = SECRET_RE.sub("[REDACTED_API_KEY]", content)

                if step_type == "USER_INPUT":
                    f_out.write(f"\n[STEP {step_idx}] [USER] ({ts})\n")
                    f_out.write("-" * 60 + "\n")
                    f_out.write(content.strip() + "\n")
                elif step_type == "PLANNER_RESPONSE" and content:
                    f_out.write(f"\n[STEP {step_idx}] [ASSISTANT] ({ts})\n")
                    f_out.write("-" * 60 + "\n")
                    f_out.write(content.strip() + "\n")
            except Exception:
                continue

    print(f"Exported clean chat history to {OUT_PATH} (Size: {os.path.getsize(OUT_PATH)} bytes)")

if __name__ == "__main__":
    export()

