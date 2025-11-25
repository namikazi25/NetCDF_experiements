import json
import os
from memory_service import save_memory_entry

# Path to the curated seed data
SEED_FILE = "backend/seed_data.json"
MEMORY_FILE = "backend/code_memory.json"

def ingest():
    print("🚀 Starting Knowledge Ingestion...")
    
    if not os.path.exists(SEED_FILE):
        print(f"❌ Error: {SEED_FILE} not found.")
        return

    with open(SEED_FILE, "r") as f:
        seed_data = json.load(f)

    print(f"📂 Found {len(seed_data)} golden records.")

    # Loop through and save (this triggers embedding generation in memory_service)
    count = 0
    for entry in seed_data:
        try:
            print(f"   Processing: {entry['query']}...")
            save_memory_entry(
                query=entry['query'], 
                code=entry['code'], 
                plan_summary=entry['plan']
            )
            count += 1
        except Exception as e:
            print(f"   ⚠️ Failed to process entry: {e}")

    print(f"✅ Successfully ingested {count} records into {MEMORY_FILE}")

if __name__ == "__main__":
    ingest()
