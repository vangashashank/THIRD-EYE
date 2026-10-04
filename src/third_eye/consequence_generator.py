import random
from pathlib import Path
import gc
import json
import shutil

import torch
from peft import PeftModel

from src.data.dataset import TextDataset
from src.models.model_loader import load_model_and_tokenizer
from src.training.trainer import train_lora
from src.training.checkpoint import save_adapter
from src.evaluation.consequence_evaluator import evaluate_loss


class ConsequenceGenerator:

    def __init__(
        self,
        output_file="outputs/consequence_labels.jsonl",
    ):
        self.output_file = Path(output_file)

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _get_base_model_name(
        self,
        adapter_path,
    ):

        config_path = (
            Path(adapter_path)
            / "adapter_config.json"
        )

        with open(config_path, "r") as f:
            config = json.load(f)

        return config[
            "base_model_name_or_path"
        ]

    def _cleanup(
        self,
        model=None,
        base_model=None,
        tokenizer=None,
        dataset=None,
    ):

        if model is not None:
            del model

        if base_model is not None:
            del base_model

        if tokenizer is not None:
            del tokenizer

        if dataset is not None:
            del dataset

        gc.collect()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_for_candidate(
        self,
        state_id,
        candidate_id,
        candidate_path,
        train_texts,
        validation_texts,
        batch_size=1,
        learning_rate=1e-4,
        epochs=1,
        max_length=128,
    ):

        candidate_path = Path(
            candidate_path
        )


      # Deterministic trajectory generation
        seed = 1000 + candidate_id

        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


        base_model_name = (
            self._get_base_model_name(
                candidate_path
            )
        )

        print(
            f"\nCandidate {candidate_id}"
        )

        print(
            "Generating t+1 consequence..."
        )

        # ------------------------------------------------
        # t+1
        # ------------------------------------------------

        base_model, tokenizer, device = (
            load_model_and_tokenizer(
                base_model_name
            )
        )

        model = PeftModel.from_pretrained(
            base_model,
            str(candidate_path),
            is_trainable=True,
        )

        model.to(device)

        validation_dataset = TextDataset(
            texts=validation_texts,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        t1_loss = evaluate_loss(
            model=model,
            dataset=validation_dataset,
            device=device,
            batch_size=batch_size,
        )

        print(
            f"t+1 validation loss: "
            f"{t1_loss:.4f}"
        )

        # ------------------------------------------------
        # Simulate another update
        # ------------------------------------------------

        print(
            "Simulating t+2 update..."
        )

        train_dataset = TextDataset(
            texts=train_texts,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        train_lora(
            model=model,
            dataset=train_dataset,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # ------------------------------------------------
        # t+2
        # ------------------------------------------------

        t2_loss = evaluate_loss(
            model=model,
            dataset=validation_dataset,
            device=device,
            batch_size=batch_size,
        )

        print(
            f"t+2 validation loss: "
            f"{t2_loss:.4f}"
        )

        # Save simulated descendant
        descendant_path = (
            Path("outputs")
            / "trajectory_cache"
            / f"state_{state_id:03d}"
            / f"candidate_{candidate_id}"
            / "t2"
        )

        if descendant_path.exists():
            shutil.rmtree(
                descendant_path
            )

        descendant_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_adapter(
            model,
            tokenizer,
            str(descendant_path),
        )

        consequence = {
            "state_id": state_id,
            "candidate_id": candidate_id,

            "candidate_checkpoint":
                str(candidate_path),

            "t1_validation_loss":
                t1_loss,

            "t2_validation_loss":
                t2_loss,

            "t2_checkpoint":
                str(descendant_path),

            "horizon": 2,
        }

        with open(
            self.output_file,
            "a",
        ) as f:

            f.write(
                json.dumps(consequence)
                + "\n"
            )

        self._cleanup(
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
        )

        return consequence
