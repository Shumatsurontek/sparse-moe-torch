import sys
from pathlib import Path
from transformers import Trainer, TrainingArguments
from sparse_moe.hf_model import MiniMoEConfig, MiniMoEHFModel
from sparse_moe.dataset import ToyRegressionDataset
import wandb
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SparseMoE")

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Charger les variables d'environnement depuis .env.local
load_dotenv(dotenv_path=".env.local")
logger.info(f"🔧 WANDB_API_KEY: {os.environ.get('WANDB_API_KEY')}")
wandb.init(project="sparse-moe-torch", name="moe-hf-trainer", config={
    "model": "MiniMoEHFModel",
    "dim": 64,
    "depth": 16,
    "heads": 2,
    "ff_hidden": 128,
    "num_experts": 16,
    "top_k": 4})

# Config et modèle
config = MiniMoEConfig(dim=64, depth=16, heads=2, ff_hidden=128, num_experts=16, top_k=4)
model = MiniMoEHFModel(config)

# Datasets
train_dataset = ToyRegressionDataset(num_samples=800, seq_len=16, dim=64)
eval_dataset = ToyRegressionDataset(num_samples=40, seq_len=16, dim=64)

# Trainer
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="no",
    report_to=["wandb"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Run the training
if __name__ == "__main__":
    trainer.train()
    wandb.finish()