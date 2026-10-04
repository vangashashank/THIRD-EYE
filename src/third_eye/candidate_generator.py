from pathlib import Path
import gc
import json
import random
import shutil

import torch
from peft import PeftModel

from src.data.dataset import TextDataset
from src.models.model_loader import load_model_and_tokenizer
from src.training.trainer import train_lora
from src.training.checkpoint import save_adapter
from src.third_eye.checkpoint_manager import CheckpointManager


class CandidateGenerator:
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        k_candidates: int = 3,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.k_candidates = k_candidates

    def _get_base_model_name(self, parent_path):
        adapter_config_path = Path(parent_path) / "adapter_config.json"

        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        return adapter_config["base_model_name_or_path"]

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_candidates(
        self,
        state_id: int,
        texts,
        epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        max_length: int = 128,
        temporary_root="outputs/candidate_work",
    ):

        parent = self.checkpoint_manager.get_parent(state_id)

        base_model_name = self._get_base_model_name(parent)

        print("Parent adapter:", parent)
        print("Base model:", base_model_name)

        temporary_root = Path(temporary_root)
        temporary_root.mkdir(parents=True, exist_ok=True)

        generated_candidates = []

        # Different seed for every candidate
        seeds = [11, 22, 33]

        for candidate_id in range(self.k_candidates):

            seed = seeds[candidate_id]

            print("\n--------------------------------")
            print(
                f"Training candidate "
                f"{candidate_id + 1}/{self.k_candidates}"
            )
            print("Seed:", seed)
            print("--------------------------------")

            self._set_seed(seed)

            candidate_work_dir = (
                temporary_root
                / f"state_{state_id:03d}"
                / f"candidate_{candidate_id}"
            )

            if candidate_work_dir.exists():
                shutil.rmtree(candidate_work_dir)

            candidate_work_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # 1. Load fresh base model
            base_model, tokenizer, device = (
                load_model_and_tokenizer(
                    base_model_name
                )
            )

            # 2. Load SAME parent LoRA adapter
            #    and make it trainable
            model = PeftModel.from_pretrained(
                base_model,
                str(parent),
                is_trainable=True,
            )

            model.to(device)

            # 3. Build training dataset
            dataset = TextDataset(
                texts=texts,
                tokenizer=tokenizer,
                max_length=max_length,
            )

            # 4. Apply one candidate update
            train_lora(
                model=model,
                dataset=dataset,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            # 5. Save temporary candidate adapter
            save_adapter(
                model,
                tokenizer,
                str(candidate_work_dir),
            )

            # 6. Register permanent candidate checkpoint
            saved_candidate = (
                self.checkpoint_manager.save_candidate(
                    source_checkpoint=candidate_work_dir,
                    state_id=state_id,
                    candidate_id=candidate_id,
                )
            )

            generated_candidates.append(
                saved_candidate
            )

            print(
                f"Candidate {candidate_id} saved at:"
            )
            print(saved_candidate)

            # Free memory before next candidate
            del model
            del base_model
            del tokenizer
            del dataset

            gc.collect()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return generated_candidates
