import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(base_model: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

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
        default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
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
    args = parser.parse_args()

    adapter_path = args.adapter_path if args.adapter_path else None
    tokenizer, model = load_model(args.base_model, adapter_path)

    if args.interactive:
        print("Finance AI Assistant is ready. Type 'exit' to quit.")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            print("Answer:", generate_answer(tokenizer, model, question))
    else:
        print(generate_answer(tokenizer, model, args.question))


if __name__ == "__main__":
    main()
