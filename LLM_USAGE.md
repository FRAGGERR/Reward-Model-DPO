# LLM_USAGE.md

## 💡 How LLMs (like ChatGPT) Helped in This Project

This document outlines where and how I used large language models (LLMs), such as ChatGPT, during the course of this ML internship project.

---

### 🛠️ 1. Code Guidance & Debugging
- **Trainer Design:** I used GPT to help me understand how to subclass `DPOTrainer` and implement `compute_loss()` correctly using custom reward scores.
- **Bug Fixes:** GPT helped trace and resolve multiple runtime errors related to:
  - `TypeError: expected Tensor but got list` during collation
  - `element does not require grad` in `loss.backward()`
  - Tokenizer/model loading errors from `transformers` and `ctransformers`

---

### 🧠 2. Architecture & Flow Clarification
- GPT provided suggestions on:
  - The overall architecture of Stepwise DPO
  - Custom scoring logic per step (`score_step()`)
  - Evaluation script design comparing base and fine-tuned models

---

### 🧾 3. Script & File Generation
The following files were generated or heavily assisted by GPT:
- `train.py` (including tokenizer logic and dataset processing)
- `stepwise_dpo_trainer.py` (custom trainer class)
- `inference.py` (structured prompts + generation)
- `eval_reward_scores.py` (base vs tuned model scoring logic)

---

### 📝 4. Documentation
- GPT generated this very file, along with a draft for `README.md` (to be finalized after evaluation results).
- GPT also guided what to include in plots and evaluation visualizations.

---

## ✅ Attribution
All external help received was through ChatGPT (OpenAI) during interactive debugging and development.

No external code was copied from blogs or third-party repositories unless explicitly licensed or documented.

---