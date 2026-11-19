import requests
from pathlib import Path

set_id = "2fd193e2-7aa7-4119-b540-7e28e82fbd13"

url = (
    "https://dailymed.nlm.nih.gov/dailymed/"
    f"services/v2/spls/{set_id}.xml"
)

response = requests.get(url, timeout=30)
response.raise_for_status()

path = Path("ultram.txt")
path.parent.mkdir(exist_ok=True)
path.write_text(response.text, encoding="utf-8")

print("Saved:", path)