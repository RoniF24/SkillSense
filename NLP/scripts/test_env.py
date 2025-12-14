import sys
print("Python starting...")
try:
    import requests
    print("Requests imported successfully.")
    import tqdm
    print("Tqdm imported successfully.")
    from assets.skills.globalVector import GLOBAL_SKILL_VECTOR
    print(f"Assets imported successfully. Vector size: {len(GLOBAL_SKILL_VECTOR)}")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")
print("Test complete.")
