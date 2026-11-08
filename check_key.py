"""Check whether the API key is available. Usage:
   python check_key.py          -> is the key loaded?
   python check_key.py --test   -> also make one free call to confirm Google accepts it
"""
import os
import sys
from pathlib import Path

import requests

from run import load_dotenv  # same .env reader as run.py

KEY_NAME = "GEMINI_API_KEY"

here = Path(__file__).resolve().parent
files = load_dotenv(Path.cwd() / ".env", here / ".env")

print("Folder you're running from:", Path.cwd())
print(".env files found:", files or "none")

key = os.environ.get(KEY_NAME, "")
if not key:
    print(f"\nNOT FOUND: {KEY_NAME} is not set.")
    print(f"  Create a file named exactly .env in {Path.cwd()} with one line:")
    print(f"  {KEY_NAME}=your-key-here")
    sys.exit(1)

print(f"\nFOUND: {KEY_NAME} = {key[:4]}...{key[-4:]}  ({len(key)} characters)")
if key != key.strip() or " " in key or '"' in key:
    print("  Warning: the key contains spaces or quotes. Remove them from .env.")

if "--test" in sys.argv:
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
    if r.status_code == 200:
        names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
        flash = [n for n in names if "flash" in n]
        print("\nKEY WORKS. Google accepted it.")
        print("Flash models available to you:", ", ".join(flash[:12]))
    else:
        print(f"\nKEY REJECTED (HTTP {r.status_code}): {r.text[:300]}")
        sys.exit(1)
