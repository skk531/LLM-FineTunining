## 1) Why full fine-tuning is expensive

Full fine-tuning updates every model weight, which requires high GPU memory, longer training time, and higher compute cost. For modern LLMs this quickly becomes expensive for individual developers.

## 2) What LoRA does

LoRA (Low-Rank Adaptation) freezes the original model weights and trains small low-rank adapter matrices in selected layers. This reduces trainable parameters and compute while still adapting model behavior.

## 3) What QLoRA does

QLoRA combines quantization and LoRA. The base model is loaded in low precision (commonly 4-bit), and only LoRA adapters are trained.

## 4) Why QLoRA is useful on limited GPU

Because the base model is quantized, VRAM usage is much lower. This makes it practical to fine-tune useful models on consumer-grade GPUs.

## 5) What is non-instruction fine-tuning?

Non-instruction fine-tuning trains on raw domain text so the model learns terminology, style, and contextual knowledge before task-specific Q and A behavior is trained.

## 6) What is instruction fine-tuning?

Instruction fine-tuning trains on instruction-response pairs so the model learns how to answer user prompts directly and in the expected output style.

## 7) What is DPO?

DPO (Direct Preference Optimization) trains the model using preference pairs (`chosen` vs `rejected`) so it learns to rank better responses higher without explicit reward model training.

## 8) Difference between SFT and DPO

- SFT learns to imitate reference answers from supervised instruction-response data.
- DPO learns relative preference between good and bad responses, often improving response quality, tone, and safety after SFT.

## 9) Hyperparameters used

Fill the exact training values from your notebooks:

- LoRA rank (`r`): 16
- LoRA alpha: 32
- LoRA dropout: 0.0
- Learning rate: 2e-4 (non-instruction + SFT), 5e-6 (DPO)
- Batch size: 1 per device (with gradient accumulation steps = 4)
- Epochs or max steps: 1 epoch (non-instruction), 2 epochs (SFT), 1 epoch (stage-3 SFT warm-start), 1 epoch (DPO)

## Practical Summary

This project follows a three-stage flow:

1. Domain adaptation via non-instruction tuning
2. Task behavior learning via SFT
3. Response quality alignment via DPO

This staged process is generally more robust than jumping directly from base model to final aligned model.
