import random
import json

TOTAL_SAMPLES = 973
OUTPUT_FILE = "shuffled_indices.json"

def main():
    # 1. Create a sequential list of all valid indices
    indices = list(range(TOTAL_SAMPLES))
    
    # 2. Hardcode a random seed so the random output is perfectly reproducible
    random.seed(42)
    
    # 3. Shuffle exactly once
    random.shuffle(indices)
    
    # 4. Save to JSON so all your HPC nodes can read the exact same shuffle order
    with open(OUTPUT_FILE, "w") as f:
        json.dump(indices, f, indent=2)
        
    print(f"Successfully saved {len(indices)} shuffled indices to {OUTPUT_FILE}")
    print(f"First 10 items for your reference: {indices[:10]}")

if __name__ == "__main__":
    main()
