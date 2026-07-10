import argparse
import os
import sys # Import sys

import torch
from peft import PeftModel
from unsloth import FastLanguageModel # NEW IMPORT


def load_model(base_model_name: str, adapter_path: str | None = None):
    # These parameters are taken from the main notebook's global settings for consistency
    _max_seq_length = 1024
    _dtype = None  # Unsloth defaults to torch.float16 for GPUs if None
    _load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=_max_seq_length,
        dtype=_dtype,
        load_in_4bit=_load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question: str, max_new_tokens: int = 256) -> str:
    prompt = (
        "You are a finance FAQ assistant. Provide clear, safe, and domain-specific answers.\n"
        f"Question: {question}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "Answer:" in generated:
        return generated.split("Answer:", 1)[1].strip()
    return generated.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for finance fine-tuned assistant")
    parser.add_argument(
        "--base-model",
        default=os.getenv("BASE_MODEL", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"),
        help="Base model name or local path",
    )
    parser.add_argument(
        "--adapter-path",
        default=os.getenv("ADAPTER_PATH", ""),
        help="Optional path to LoRA/DPO adapter",
    )
    parser.add_argument(
        "--question",
        default="How can I apply for reimbursement?",
        help="Single question to run",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive question-answer loop",
    )

    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    adapter_path = args.adapter_path if args.adapter_path else None
    tokenizer, model = load_model(args.base_model, adapter_path)

    FastLanguageModel.for_inference(model) # NEW CALL

    if args.interactive:
        print("Finance AI Assistant is ready. Type 'exit' to quit.")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            print("Answer:", generate_answer(tokenizer, model, question))
    else:
        print("Answer:", generate_answer(tokenizer, model, args.question))


if __name__ == "__main__":
    main()
