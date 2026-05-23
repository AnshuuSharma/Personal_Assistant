with open("data\\raw\\info.txt", "r") as f:
    text = f.read()

chunks = [chunk.strip() for chunk in text.split("# SECTION:") if chunk.strip()]

import json
with open("data/chunks/chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)