from datasets import load_dataset

def main():
    print("Loading ccdv/govreport-summarization dataset...")
    dataset = load_dataset("ccdv/govreport-summarization")
    print("\nDataset fully loaded.")
    print("Available splits and their sizes:")
    
    for split in dataset.keys():
        print(f"  - {split}: {len(dataset[split])} samples")

if __name__ == "__main__":
    main()
