import json
import os

EXAMPLES = [
    {
        "prompt": "What is 17 * 12? Show your steps.",
        "chosen": [
            "Step 1: Break 17 into 10 and 7.",
            "Step 2: 10 * 12 = 120",
            "Step 3: 7 * 12 = 84",
            "Step 4: 120 + 84 = 204"
        ],
        "rejected": [
            "Step 1: Break 17 into 10 and 7.",
            "Step 2: 10 * 12 = 120",
            "Step 3: 7 * 12 = 74",
            "Step 4: 120 + 74 = 194"
        ]
    },
    {
        "prompt": "Is 42 divisible by 7? Show your steps.",
        "chosen": [
            "Step 1: 7 * 6 = 42",
            "Step 2: So yes, 42 is divisible by 7."
        ],
        "rejected": [
            "Step 1: 7 * 5 = 35",
            "Step 2: So yes, 42 is divisible by 7."  # Incorrect logic
        ]
    },
    {
        "prompt": "Is 16 a perfect square? Show steps.",
        "chosen": [
            "Step 1: Find the square root of 16.",
            "Step 2: √16 = 4",
            "Step 3: 4 * 4 = 16, so it is a perfect square."
        ],
        "rejected": [
            "Step 1: Find the square root of 16.",
            "Step 2: √16 = 8",  # Incorrect step
            "Step 3: 8 * 8 = 64, so it is not a perfect square."  # Wrong conclusion
        ]
    },
    {
        "prompt": "Determine whether 35 is an odd number.",
        "chosen": [
            "Step 1: An odd number has remainder 1 when divided by 2.",
            "Step 2: 35 ÷ 2 = 17 remainder 1",
            "Step 3: Therefore, 35 is an odd number."
        ],
        "rejected": [
            "Step 1: 35 ÷ 2 = 17 remainder 0",  # Wrong step
            "Step 2: So 35 is even."  # Wrong conclusion
        ]
    },
    {
        "prompt": "What is the sum of the first 5 even numbers?",
        "chosen": [
            "Step 1: First 5 even numbers are 2, 4, 6, 8, 10.",
            "Step 2: Sum = 2 + 4 + 6 + 8 + 10 = 30"
        ],
        "rejected": [
            "Step 1: First 5 even numbers are 2, 4, 6, 8, 10.",
            "Step 2: Sum = 2 + 4 + 6 + 8 + 12 = 32"  # Wrong number used
        ]
    },
    {
        "prompt": "Write steps to check if 'racecar' is a palindrome.",
        "chosen": [
            "Step 1: Reverse the string.",
            "Step 2: 'racecar' reversed is 'racecar'.",
            "Step 3: Since both are the same, it's a palindrome."
        ],
        "rejected": [
            "Step 1: Reverse the string.",
            "Step 2: 'racecar' reversed is 'raccare'.",  # Wrong reversal
            "Step 3: Not a palindrome."  # Wrong conclusion
        ]
    },
    {
        "prompt": "Does the number 100 end in 0? Show reasoning.",
        "chosen": [
            "Step 1: Look at the last digit of 100.",
            "Step 2: The last digit is 0.",
            "Step 3: So yes, it ends in 0."
        ],
        "rejected": [
            "Step 1: Look at the last digit of 100.",
            "Step 2: The last digit is 1.",  # Wrong digit
            "Step 3: So no, it does not end in 0."  # Wrong conclusion
        ]
    },
    {
        "prompt": "What is the output of: print(2 + 3 * 4)?",
        "chosen": [
            "Step 1: According to order of operations, do multiplication first.",
            "Step 2: 3 * 4 = 12",
            "Step 3: 2 + 12 = 14",
            "Step 4: Output is 14"
        ],
        "rejected": [
            "Step 1: Add first: 2 + 3 = 5",  # Wrong order
            "Step 2: 5 * 4 = 20",
            "Step 3: Output is 20"
        ]
    },
    {
        "prompt": "What is 0 divided by 5?",
        "chosen": [
            "Step 1: 0 divided by any non-zero number is 0.",
            "Step 2: So, 0 ÷ 5 = 0."
        ],
        "rejected": [
            "Step 1: 0 divided by 5 is undefined.",  # Incorrect
            "Step 2: So, the answer is undefined."
        ]
    },
    {
        "prompt": "Check if 19 is a prime number.",
        "chosen": [
            "Step 1: A prime number has only two divisors: 1 and itself.",
            "Step 2: Check divisibility by 2, 3, 5.",
            "Step 3: 19 is not divisible by any of them.",
            "Step 4: So 19 is prime."
        ],
        "rejected": [
            "Step 1: Check divisibility by 2, 3, 5.",
            "Step 2: 19 is divisible by 3.",  # Wrong check
            "Step 3: So 19 is not prime."  # Wrong conclusion
        ]
    },
    {
        "prompt": "Is 81 divisible by 9?",
        "chosen": [
            "Step 1: 9 * 9 = 81",
            "Step 2: So yes, 81 is divisible by 9."
        ],
        "rejected": [
            "Step 1: 9 * 8 = 72",  # Wrong multiplication
            "Step 2: So no, 81 is not divisible by 9."  # Wrong conclusion
        ]
    },
    {
        "prompt": "Is 1 an even number?",
        "chosen": [
            "Step 1: An even number is divisible by 2 with remainder 0.",
            "Step 2: 1 ÷ 2 = 0 remainder 1",
            "Step 3: So 1 is not even."
        ],
        "rejected": [
            "Step 1: 1 ÷ 2 = 0.5",  # Not useful
            "Step 2: 0.5 is even",  # Wrong logic
            "Step 3: So 1 is even."  # Wrong conclusion
        ]
    },
    {
        "prompt": "Evaluate: 5 ** 2",
        "chosen": [
            "Step 1: 5 ** 2 means 5 squared.",
            "Step 2: 5 * 5 = 25",
            "Step 3: Result is 25"
        ],
        "rejected": [
            "Step 1: 5 ** 2 = 10",  # Wrong value
            "Step 2: Result is 10"
        ]
    },
    {
        "prompt": "Is 9 an odd number?",
        "chosen": [
            "Step 1: Divide 9 by 2 → 4.5",
            "Step 2: Since not a whole number, it's odd"
        ],
        "rejected": [
            "Step 1: Divide 9 by 2 → 4.5",
            "Step 2: Since it's decimal, it's even"  # Wrong logic
        ]
    },
    {
        "prompt": "What is the square of 11?",
        "chosen": [
            "Step 1: Multiply 11 by 11",
            "Step 2: 11 * 11 = 121",
            "Step 3: So, 11 squared is 121"
        ],
        "rejected": [
            "Step 1: 11 * 11 = 131",  # Wrong multiplication
            "Step 2: So, 11 squared is 131"
        ]
    }
]

os.makedirs("data", exist_ok=True)

with open("data/synthetic_dataset.jsonl", "w") as f:
    for ex in EXAMPLES:
        json.dump(ex, f)
        f.write("\n")

print(f"Saved {len(EXAMPLES)} examples to ../data/synthetic_dataset.jsonl")