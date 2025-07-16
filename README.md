# Stepwise DPO Fine-Tuning with TinyLlama

This project implements a Stepwise Direct Preference Optimization (DPO) fine-tuning pipeline using the TinyLlama-1.1B-Chat model. The pipeline scores intermediate reasoning steps, trains a preference model on them, and evaluates performance through reward model comparison.

---

## 📦 Dataset & Model

- **Base Model**: [`TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Data Format**: JSONL with:
  - `prompt`: reasoning task
  - `chosen`: correct multi-step response
  - `rejected`: incorrect multi-step response
  - `chosen_scores`, `rejected_scores`: list of stepwise scores (1 or 0)

- **Fine-Tuning Method**: Stepwise-DPO  
  We break down the full responses into reasoning steps and apply DPO loss on mean stepwise scores:
  
  \[
  \text{loss} = -\log(\sigma(R_\text{chosen} - R_\text{rejected}))
  \]

---

## 📈 Reward Score Evaluation

Using a 4-bit TinyLlama reward model (`ctransformers`), we scored both the base and fine-tuned models on 15 examples.

| Metric               | Value       |
|----------------------|-------------|
| Base Avg Score       | 0.0000      |
| Fine-Tuned Avg Score | 0.0667      |
| ✅ Improvement       | +0.0667     |
| Evaluation Size      | 15 examples |

---

## ⚠️ Limitations & Next Steps

- **Limited Data**: Only 15 examples were used. Results may not generalize.
- **No Human Raters**: Stepwise scores are heuristic, not human-labeled.
- **Eval Speed**: Inference using CPU 4-bit models is slow for large-scale eval.
- **Model Type**: GGUF 4-bit model limits compatibility with standard HF APIs.

### ✅ Future Improvements:
- Scale to 1k+ examples
- Use human annotators for stepwise labels
- Evaluate with larger reward models or LLM-as-a-judge

---

## 🚀 How to Run

### 📦 Setup
```bash

# 1. Clone and setup

git clone https://github.com/FRAGGERR/Reward-Model-DPO/tree/main
cd futureagi-ml-intern


# 2. Create virtual env
conda create -n dpo-env python=3.11
conda activate dpo-env

# 3. Install dependencies
pip install -r requirements.txt

🏋️‍♂️ Training

python train.py


🔍 Evaluation

cd evaluation
python eval_reward_scores.py

🧠 Inference

python inference.py

```
--- 

## 📁 Project Structure

futureagi-ml-intern/
│
├── train.py
├── inference.py
│
├── trainer/
│   └── stepwise_dpo_trainer.py
│
├── reward_model/
│   ├── evaluate_steps.py
│   ├── phi2_reward_model.py
│   ├── test_reward.py
│   └── tinyllama_reward_model.py
├── saved_model/
│   └── stepwise_dpo_tinyllama/
├── models/
│   └── tinyllama/
│       └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
├── data/
│   ├── synthetic_dataset.jsonl
│   └── rewarded.jsonl
│
├── utils/
│   ├── build_synthetic_dataset.py
│   ├── load_dataset.py
│   ├── load_rewarded_dataset.py
│   └── test_dataset.py 
├── evaluation/
│   ├── eval_reward_scores.py
│   └── reward_score_comparison.json
│
├── logs/
│   └── training_log.txt
│
├── requirements.txt
├── README.md
└── LLM_USAGE.md

---

## 🙌 Acknowledgments
#### This project was implemented as part of the FutureAGI ML Intern Assignment, combining modern fine-tuning (DPO) with step-level interpretability.
