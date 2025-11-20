import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

# Load env vars explicitly to be sure
load_dotenv()

from llm_service import client, MODEL
from memory_service import get_embedding

def test_connection():
    print(f"Testing connection to: {client.base_url}")
    print(f"Using Chat Model: {MODEL}")
    print(f"Using Embedding Model: {os.getenv('LOCAL_EMBEDDING_MODEL')}")

    # 1. Test Chat
    print("\n--- Testing Chat Completion ---")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Always answer in rhymes. Today is Thursday"},
                {"role": "user", "content": "What day is it today?"}
            ],
            temperature=0.7,
            max_tokens=50
        )
        print("Chat Response:")
        print(response.choices[0].message.content)
        print("✅ Chat Test Passed")
    except Exception as e:
        print(f"❌ Chat Test Failed: {e}")

    # 2. Test Embeddings
    print("\n--- Testing Embeddings ---")
    try:
        # We use the function from memory_service to test the full integration
        vec = get_embedding("Some text to embed")
        if vec and len(vec) > 0:
            print(f"Generated embedding with dimension: {len(vec)}")
            print("✅ Embedding Test Passed")
        else:
            print("❌ Embedding Test Failed: Returned empty vector")
    except Exception as e:
        print(f"❌ Embedding Test Failed: {e}")

if __name__ == "__main__":
    test_connection()
