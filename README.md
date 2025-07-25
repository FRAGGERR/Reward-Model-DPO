<h1 align="center">Stepwise DPO Fine-Tuning with TinyLlama</h1>

<p align="center">
  <img alt="Transformers" src="https://img.shields.io/badge/🤖-Transformers-blue" />
  <img alt="DPO" src="https://img.shields.io/badge/🔍-DPO-orange" />
  <img alt="TinyLlama" src="https://img.shields.io/badge/🦙-TinyLlama-9cf" />
  <img alt="Reward Modeling" src="https://img.shields.io/badge/📊-Reward_Modeling-brightgreen" />
  <img alt="LLM" src="https://img.shields.io/badge/🧠-CausalLM-purple" />
</p>

This project aims to replicate the methodology of OpenAI's “Let’s Verify Step by Step” using LLM-generated step-wise rewards instead of human labels. It not only showcases fine-tuning techniques but also emphasizes reasoning traceability and thoughtful experimentation. The focus is on implementing a reward model pipeline, customizing Hugging Face’s DPOTrainer, and evaluating improvements in reasoning capability on small models like TinyLlama (1.1B).

---

## 📦 Dataset & Model

- **Dataset**: Custom rejection-sampled dataset (`data/rewarded.jsonl`) generated using TinyLlama’s responses to mathematical prompts. Each example contains multiple reasoning steps, with chosen/rejected preferences.
- **Base Model**: [`TinyLlama-1.1B-Chat-v1.0.Q4_K_M`](https://huggingface.co/codellama/CodeLlama-7b-hf) (4-bit GGUF quantized)
- **Fine-Tuned Model**: `saved_model/stepwise_dpo_tinyllama` trained using a custom subclass of Hugging Face's `DPOTrainer`.

---

## 📈 Reward Score Results

Evaluation was done using a step-level scoring model (`TinyLlamaRewardScorer`). The average reward scores before and after fine-tuning:

```json
{
  "base_avg_score_diff": 0.0,
  "tuned_avg_score_diff": 0.666,
  "improvement": 0.666,
  "num_examples": 15
}
```

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

<p>Using a 4-bit TinyLlama reward model (<code>ctransformers</code>), we scored both the base and fine-tuned models on 15 examples.</p>

<div style="display: flex; align-items: center; gap: 50px;">

  <!-- Reward Table -->
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Base Avg Score</strong></td>
        <td><strong>0.0000</strong></td>
      </tr>
      <tr>
        <td><strong>Fine-Tuned Avg Score</strong></td>
        <td><strong>0.667</strong></td>
      </tr>
      <tr>
        <td><strong>✅ Improvement</strong></td>
        <td><strong>+0.667</strong></td>
      </tr>
      <tr>
        <td><strong>Evaluation Size</strong></td>
        <td><strong>15 examples</strong></td>
      </tr>
    </tbody>
  </table>

  <!-- Reward Graph -->
  <img src="https://github.com/user-attachments/assets/115dabfb-f4c9-407b-8e7f-77dc1fb3ad3f" width="500" alt="Reward Score Comparison" />

</div>


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

# 4. Training

python train.py

# 5. Evaluation

cd evaluation
python eval_reward_scores.py

# 6. Inference

python inference.py

```
--- 

## Download Fine-Tuned Model

To use the fine-tuned TinyLlama model (saved after Stepwise DPO training), download it from the following link:

🔗 **[Download Fine-Tuned Model (Google Drive)](https://drive.google.com/drive/folders/1Sqs2_OMrIeB5Q4Ei3PqQDcUwEin5XA_C?usp=drive_link)**

Place the downloaded folder into:

```
saved_model/stepwise_dpo_tinyllama/
```
> This folder must contain all necessary files like `pytorch_model.bin`, `config.json`, `tokenizer.json`, etc.

---

## 📁 Project Structure 
``` 

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
``` 
---

## 🙌 Acknowledgments
#### This project was implemented as part of the FutureAGI ML Intern Assignment, combining modern fine-tuning (DPO) with step-level interpretability.
