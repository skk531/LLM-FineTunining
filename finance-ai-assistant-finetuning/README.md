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

Primary model used in notebooks: `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`

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

- LoRA rank (`r`): 16
- LoRA alpha: 32
- LoRA dropout: 0.0
- Learning rate: 2e-4 (non-instruction + SFT), 5e-6 (DPO)
- Batch size: 1 per device (with gradient accumulation steps = 4)
- Epochs or max steps: 1 epoch (non-instruction), 2 epochs (SFT), 1 epoch (stage-3 SFT warm-start), 1 epoch (DPO)

## 10. Training Screenshots or Logs

Log snippets captured from executed notebook outputs:

- Runtime and stack (all stages):
  - GPU: Tesla T4
  - Unsloth 2026.7.2, Transformers 4.57.6
  - Torch 2.10.0+cu128, CUDA Toolkit 12.8

- Stage 1: Non-instruction fine-tuning ([notebooks/non_instruction_finetuning.ipynb](notebooks/non_instruction_finetuning.ipynb))
  - Num examples = 15
  - Num Epochs = 1
  - Total steps = 4
  - Batch size per device = 1, gradient accumulation steps = 4
  - Trainable parameters = 8,798,208 of 502,830,976 (1.75% trained)

- Stage 2: Instruction SFT ([notebooks/instruction_finetuning.ipynb](notebooks/instruction_finetuning.ipynb))
  - Num examples = 105
  - Num Epochs = 2
  - Total steps = 54
  - Batch size per device = 1, gradient accumulation steps = 4
  - Trainable parameters = 8,798,208 of 502,830,976 (1.75% trained)

- Stage 3A: Warm-start SFT in DPO notebook ([notebooks/dpo_alignment.ipynb](notebooks/dpo_alignment.ipynb))
  - Num examples = 105
  - Num Epochs = 1
  - Total steps = 27
  - Batch size per device = 1, gradient accumulation steps = 4

- Stage 3B: DPO alignment ([notebooks/dpo_alignment.ipynb](notebooks/dpo_alignment.ipynb))
  - Num examples = 51
  - Num Epochs = 1
  - Total steps = 13
  - Batch size per device = 1, gradient accumulation steps = 4
  - Trainable parameters = 8,798,208 of 502,830,976 (1.75% trained)

## 11. Before vs After Comparison

- Base evaluation report: `reports/base_model_evaluation.md`
- Base vs SFT: `reports/sft_model_comparison.md`
- Base vs SFT vs DPO: `reports/final_evaluation.md`

## 12. Final Observations

The staged pipeline improved response quality progressively:

- Stage 1 (non-instruction fine-tuning) improved domain familiarity. The model began using more finance-specific terms and produced less completely generic answers.
- Stage 2 (instruction fine-tuning) improved structure and usefulness. Responses became more aligned to user questions, clearer, and more procedural for how-to prompts.
- Stage 3 (DPO alignment) improved answer preference quality. Compared with SFT-only outputs, responses were generally safer in sensitive cases (for example OTP/fraud prompts), more professional in tone, and better prioritized practical guidance.
- Across the final comparison, the largest gains were observed in domain specificity, clarity, and safety-oriented phrasing rather than in raw verbosity alone.
- Remaining gap: some answers can still be concise or partially generic for edge-case queries, so high-stakes finance questions should include a final human review.

## 13. Challenges Faced

Key challenges during implementation:

- Compute and memory constraints: even with QLoRA/4-bit loading, training and generation settings required careful tuning to avoid Colab memory/runtime interruptions.
- Dataset preparation effort: creating and validating raw text, instruction pairs, and preference pairs required cleanup for consistency, duplication removal, and finance-domain correctness.
- Prompt/template sensitivity: small changes in instruction formatting affected response style and quality, so prompt structure had to be standardized across stages.
- Multi-stage checkpoint management: keeping base, stage-1, stage-2, and stage-3 artifacts organized was necessary to ensure fair before-vs-after evaluation.
- Alignment tuning tradeoffs: DPO settings (for example learning rate and beta) required balancing helpfulness and safety without over-penalizing informative responses.
- Evaluation consistency: using the same fixed question set across stages was essential, and the comparison/report generation logic had to be hardened to reliably export markdown in Colab.

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
