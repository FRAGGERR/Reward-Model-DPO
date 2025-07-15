import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_rewarded_dataset import load_rewarded_dataset

if __name__ == "__main__":
    dataset = load_rewarded_dataset()
    print(f"✅ Loaded {len(dataset)} samples")
    print(dataset[0])
