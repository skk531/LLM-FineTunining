# Finance AI Assistant Fine-Tuning with Unsloth

## 1. Project Title

Finance Domain AI Assistant: Non-Instruction FT + SFT + DPO using Unsloth

## 2. Domain Selected

Finance FAQ Assistant

## 3. Business Problem

Build a domain-specific assistant that answers finance and banking questions with better correctness, clarity, and safety than a general base model.

## 4. Dataset Details

- Raw domain corpus: `data/non_instruction_data.txt`
  - Purpose: Stage-1 non-instruction domain adaptation
  - Count: 56 paragraphs
- Instruction dataset: `data/instruction_dataset.jsonl`
  - Format: `{ instruction, response }`
  - Count: 105 examples
- Preference dataset: `data/preference_dataset.jsonl`
  - Format: `{ prompt, chosen, rejected }`
  - Count: 51 examples

## 5. Base Model Used

Primary model used in notebooks: TODO (example: `Qwen/Qwen2.5-0.5B-Instruct`)

## 6. Non-Instruction Fine-Tuning Approach

- Load and clean finance raw text
- Chunk long text for training
- Apply LoRA/QLoRA adapters with Unsloth
- Train for domain adaptation
- Save adapter checkpoint

See: `notebooks/non_instruction_finetuning.ipynb`

## 7. Instruction Fine-Tuning Approach

- Load instruction-response JSONL
- Format prompts for supervised fine-tuning
- Continue training from stage-1 adapter or base model
- Save SFT adapter/model

See: `notebooks/instruction_finetuning.ipynb`

## 8. DPO Alignment Approach

- Load preference dataset (`prompt`, `chosen`, `rejected`)
- Prepare DPO trainer config
- Align SFT model using preference optimization
- Save DPO-aligned adapter/model

See: `notebooks/dpo_alignment.ipynb`

## 9. LoRA / QLoRA Configuration

Fill exact values from notebook runs:

- LoRA rank (`r`): TODO
- LoRA alpha: TODO
- LoRA dropout: TODO
- Learning rate: TODO
- Batch size: TODO
- Epochs or max steps: TODO

## 10. Training Screenshots or Logs

Add screenshots/log snippets from notebook outputs in this section.

## 11. Before vs After Comparison

- Base evaluation report: `reports/base_model_evaluation.md`
- Base vs SFT: `reports/sft_model_comparison.md`
- Base vs SFT vs DPO: `reports/final_evaluation.md`

## 12. Final Observations

TODO: Summarize quality improvements across each stage.

## 13. Challenges Faced

TODO: Mention compute limits, data cleaning effort, prompt formatting, and alignment tuning issues.

## 14. Future Improvements

- Expand dataset diversity and edge cases
- Add multilingual finance support
- Add retrieval for policy and product updates
- Improve evaluation with automatic scoring and human review

## Project Structure

```
finance-ai-assistant-finetuning/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── src/
│   └── inference.py
├── requirements.txt
└── README.md
```

## Inference Script Usage

Single question:

```bash
python src/inference.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path path/to/adapter --question "How can I improve my credit score?"
```

Interactive mode:

```bash
python src/inference.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path path/to/adapter --interactive
```
