from pathlib import Path
import gc
import json
import random
import hashlib

import torch
from safetensors.torch import load_file
from src.third_eye.state_manager import StateManager

from src.data.dataset import TextDataset
from src.models.model_loader import load_model_and_tokenizer
from src.training.trainer import train_lora
from src.training.checkpoint import (
    save_adapter, load_adapter, adapter_state_digest, adapter_state_sha256,
)
from src.third_eye.checkpoint_manager import CheckpointManager


class CandidateGenerator:
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        k_candidates: int = 3,
        metadata_file="outputs/candidate_metadata.jsonl",
        model_load_options=None,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.k_candidates = k_candidates
        self.metadata_file = metadata_file
        if k_candidates not in (1, 2, 3):
            raise ValueError("Expected one to three candidates")
        self.model_load_options = dict(model_load_options or {})

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
        run_id=None,
        trajectory_id=None,
    ):
        if (run_id is None) != (trajectory_id is None):
            raise ValueError(
                "Provide both run_id and trajectory_id, or neither."
            )

        proposal_manager = None

        if run_id is not None:
            proposal_manager = StateManager(
                checkpoint_manager=self.checkpoint_manager,
                metadata_file=self.metadata_file,
            )

        parent = self.checkpoint_manager.get_parent(state_id)
        parent_weights = parent / "adapter_model.safetensors"
        parent_file_hash = hashlib.sha256(parent_weights.read_bytes()).hexdigest()
        parent_state_hash = adapter_state_digest(load_file(str(parent_weights)))

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
            if proposal_manager is not None:
                proposal_manager.record_proposal(
                    run_id=run_id,
                    trajectory_id=trajectory_id,
                    state_id=state_id,
                    candidate_id=candidate_id,
                    parent_checkpoint=parent,
                    seed=seed,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                    max_length=max_length,
                    texts=texts,
                    parent_weights_sha256=parent_file_hash,
                )

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

            candidate_work_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            # 1. Load fresh base model
            base_model, tokenizer, device = (
                load_model_and_tokenizer(
                    base_model_name, **self.model_load_options,
                )
            )

            # 2. Load SAME parent LoRA adapter
            #    and make it trainable
            model = load_adapter(base_model, parent, device, is_trainable=True)
            initial_state_hash = adapter_state_sha256(model)
            if initial_state_hash != parent_state_hash:
                raise AssertionError("Candidate did not start from the saved parent weights")

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
            if proposal_manager is not None:
                proposal_manager.record_candidate(
                    state_id=state_id,
                    candidate_id=candidate_id,
                    seed=seed,
                    training_loss=model.last_training_metrics["mean_training_loss"],
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    checkpoint_path=str(saved_candidate),
                    actual_training={
                        **model.last_training_metrics,
                        "initial_adapter_state_sha256": initial_state_hash,
                        "parent_adapter_state_sha256": parent_state_hash,
                    },
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
