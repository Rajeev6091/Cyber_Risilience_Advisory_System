"""Upload the fine-tuned BERT model to a Hugging Face Hub model repo.

The model is git-ignored (it is a 47 MB binary), so it cannot travel with the
source. Publishing it to the Hub lets a deployed Space load it by id: set the
Space variable BERT_MODEL_PATH to the repo id this script prints.

Usage:
    export HF_TOKEN=hf_...                       # needs write access
    python3 scripts/upload_model_to_hf.py <your-username>/bert-mini-cyber
"""
import os
import sys

from huggingface_hub import HfApi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "llm_finetune_project", "bert-mini-finetuned")

# Only the files transformers needs at inference time. Training checkpoints and
# the confusion-matrix images would otherwise bloat the repo.
ALLOWED = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <namespace>/<model-name>")
    repo_id = sys.argv[1]

    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Create a write token at "
                 "https://huggingface.co/settings/tokens")

    model_dir = os.getenv("BERT_MODEL_PATH", DEFAULT_MODEL_DIR)
    if not os.path.isdir(model_dir):
        sys.exit(f"No model directory at {model_dir}. Train it first "
                 f"(see 'Fine-tuning BERT' in the README).")

    missing = [f for f in ALLOWED if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        sys.exit(f"Model directory is missing {missing}")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=ALLOWED,
    )

    print(f"\nUploaded to https://huggingface.co/{repo_id}")
    print(f"Now set BERT_MODEL_PATH={repo_id} in your Space's variables.")


if __name__ == "__main__":
    main()
