"""Explicit cached-HF/PEFT collector backend, never imported by offline preflight.

All supported feature versions are opt-in proposals pending the recorded
agreements. No generic unknown-task scorer, model download or dataset fetch.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import StoppingCriteria, StoppingCriteriaList

from src.models.model_loader import load_model_and_tokenizer
from .feature_contract import ContractError
from .generation_calibration import final_answer, generation_prompt, verify_answer
from .sprint2_collection import file_sha, write_json
from .sprint2_measurements import (answer_logprobs, answer_nll, candidate_data,
    gradient_measurements, lora_parameters, probe_measurements, score_completions)
from .sprint2_protocol import build_pilot_state_plan, prompt_processing_seed, _digest
from .sprint2_runtime import digest_state, restored_training_state, seeded_stream


class HFCollectionBackend:
    def __init__(self, config, base, spec, populations):
        self.config, self.base, self.spec, self.populations = config, Path(base), spec, populations
        binding = config["model"]
        self.base_model, self.tokenizer, self.device = load_model_and_tokenizer(
            binding["name"], revision=binding["revision"], quantization=binding["quantization"], local_files_only=True)
        if hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest() != config["chat_template_sha256"]:
            raise ContractError("Cached tokenizer template differs from calibration/collection")
        if not self.tokenizer.is_fast:
            raise ContractError("Exact answer-offset alignment requires the pinned fast tokenizer")
        if binding["quantization"]:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        self.model = None
        self.deadline = None
        self.output_path = None
        self.state_output_path = None

    def persist_attempt(self, attempt):
        if self.state_output_path is not None:
            with (self.state_output_path/"generation_attempts.jsonl").open("a") as f:
                f.write(json.dumps(attempt, allow_nan=False)+"\n")
                f.flush()
                os.fsync(f.fileno())

    def check_limits(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise ContractError("Pilot wall-time guard reached; stop, preserve interrupted evidence for resume")
        if self.output_path is not None:
            size = sum(p.stat().st_size for p in self.output_path.rglob("*") if p.is_file())
            if size >= self.config["retention"]["max_output_bytes"]:
                raise ContractError("Pilot artifact limit reached; no pruning or overwriting")

    def load_parent(self, trajectory, snapshot):
        if self.model is None:
            root = self.base/self.config["roots"][trajectory]["path"]
            self.model = PeftModel.from_pretrained(self.base_model, str(root), is_trainable=True,
                                                  local_files_only=True)
            lora_parameters(self.model, self.config["module_manifest"])
        if snapshot is None:
            root = self.base/self.config["roots"][trajectory]["path"]
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.model, load_file(str(root/"adapter_model.safetensors")))
        else:
            payload = torch.load(snapshot, map_location="cpu", weights_only=True)
            if payload["semantic_hash"] != self.spec["semantic_hash"]:
                raise ContractError("Applied snapshot definition/config hash mismatch")
            values = payload["parameters"]
            actual = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            if set(actual) != set(values):
                raise ContractError("Snapshot parameter manifest mismatch")
            with torch.no_grad():
                for name, p in actual.items():
                    if p.shape != values[name].shape or p.dtype != values[name].dtype:
                        raise ContractError("Snapshot parameter shape/dtype mismatch")
                    p.copy_(values[name])
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

    def parent_sha(self):
        # Base identity is pinned by spec/config; checksum represents actual A/B.
        return digest_state({n: p for n, p in self.model.named_parameters() if p.requires_grad})

    def _rows(self, role):
        manifest = self.populations[role]["data"]
        return [{**row, "task_id": manifest["task_id"]} for row in manifest["prompts"]]

    def _rule(self, row):
        return self.config["task_rules"][row["task_id"]]

    def _generate(self, row, *, seed=0):
        self.check_limits()
        rule = self._rule(row)
        prompt = generation_prompt(self.tokenizer, row["prompt"], rule)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False, return_token_type_ids=False).to(self.device)
        prefix = inputs["input_ids"].shape[1]
        cap = self.config["generation_caps"][row["task_id"]]
        if prefix+cap > self.model.config.max_position_embeddings:
            raise ContractError("Generation context exceeds cached model capacity")
        tokenizer = self.tokenizer
        class AnswerStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, prefix:], skip_special_tokens=True)
                return torch.tensor([final_answer(text, rule) is not None], device=input_ids.device)
        self.model.eval()
        with seeded_stream(seed), torch.inference_mode():
            time_options = {"max_time": max(.1, self.deadline-time.monotonic())} if self.deadline is not None else {}
            tokens = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=cap,
                                         stopping_criteria=StoppingCriteriaList([AnswerStop()]), **time_options)
        ids = tokens[0, prefix:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        at_eos = bool(len(ids) and int(ids[-1]) in eos)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        parsed = final_answer(text, rule, at_eos=at_eos)
        self.persist_attempt({**row, "text": text, "generated_tokens": len(ids), "cap": cap,
            "at_eos": at_eos, "seed": seed, "parent_sha256": self.parent_sha(),
            "interrupted": parsed is None and not at_eos and len(ids) < cap})
        if parsed is None and not at_eos and len(ids) < cap:
            raise ContractError("Generation interrupted by resource/time limit; not a completed accuracy measurement")
        return {**row, "text": text, "at_eos": at_eos, "generated_tokens": len(ids), "cap": cap,
                "scorer": rule, "seed": seed, "parent_sha256": self.parent_sha(),
                "prompt_token_ids": inputs["input_ids"][0].tolist(), "generated_token_ids": ids.tolist(),
                "truncated": parsed is None and not at_eos and len(ids) == cap,
                "stopping_reason": "final_answer" if parsed is not None else "eos" if at_eos else "max_new_tokens",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}

    def _score(self, role):
        rows = self._rows(role)
        return {**score_completions([self._generate(r) for r in rows], self._rule(rows[0])),
                "manifest_sha256": self.populations[role]["manifest_sha256"]}

    def _prepare(self, row, answer=None):
        answer = row["reference"] if answer is None else answer
        if not verify_answer(answer, row["reference"], self._rule(row)) and answer == row["reference"]:
            raise ContractError("Invalid reference")
        prompt = generation_prompt(self.tokenizer, row["prompt"], self._rule(row))
        prefix = prompt+"Final answer: "
        text = prefix+answer+"\n"
        encoded = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=False,
                                 return_token_type_ids=False)
        offsets = encoded.pop("offset_mapping")[0]
        lo, hi = len(prefix), len(prefix)+len(answer)
        # A VALUE token may include its single leading separator space. This
        # explicitly versioned span does not admit header/reasoning/special tokens.
        mask = torch.tensor([[int(a) >= lo-1 and int(b) <= hi and int(b) > lo
                              for a, b in offsets]], dtype=torch.bool)
        maximum = self.config["update"]["max_length"]
        encoded = {k: v[:, :maximum].to(self.device) for k, v in encoded.items()}
        mask = mask[:, :maximum].to(self.device)
        if not bool(mask.any()) or hi > int(offsets[min(maximum, len(offsets))-1, 1]):
            raise ContractError("Training serialization truncates/loses answer span; stop, do not zero-fill")
        return {"tokens": encoded, "answer_mask": mask, "text": text,
                "length": int(encoded["attention_mask"].sum()), "row": row,
                "serialized_sha256": hashlib.sha256(text.encode()).hexdigest()}

    def _logprobs(self, examples):
        result = []
        for ex in examples:
            self.check_limits()
            out = self.model(**ex["tokens"])
            result.extend(answer_logprobs(out.logits, ex["tokens"]["input_ids"], ex["answer_mask"],
                                         ex["tokens"]["attention_mask"]))
        return result

    def _loss(self, examples):
        return answer_nll(self._logprobs(examples), reduction=self.config["definitions"]["nll_reduction"])

    def _loss_terms(self, examples):
        counts = [int(ex["answer_mask"].sum()) for ex in examples]
        total = sum(counts)
        for ex, count in zip(examples, counts):
            weight = (1/len(examples) if self.config["definitions"]["nll_reduction"] == "equal_answer_v1"
                      else count/total)
            yield self._loss([ex]), weight

    def _anchor_logits(self, examples):
        logits = []
        for ex in examples:
            self.check_limits()
            out = self.model(**ex["tokens"])
            logits.append(out.logits[0, :-1][ex["answer_mask"][0, 1:]])
        return torch.cat(logits)

    def measure_parent(self):
        scores = {role: self._score(role) for role in ("target", "ood", "retention")}
        accuracy = {role: value["accuracy"] for role, value in scores.items()}
        confidence = []
        for row in self._rows("confidence"):
            attempt = self._generate(row)
            parsed = final_answer(attempt["text"], self._rule(row), at_eos=attempt["at_eos"])
            if parsed is None or attempt["truncated"]:
                raise ContractError("D06 state-confidence parse failure: explicit failure policy still needed")
            # Score the ACTUAL emitted tokens in their original generation
            # context, not a reconstructed correct/incorrect answer context.
            generated = attempt["generated_token_ids"]
            text = attempt["text"]
            start = text.rfind("Final answer: ")+len("Final answer: ")
            end = start+len(parsed)
            prefixes = [self.tokenizer.decode(generated[:i], skip_special_tokens=True)
                        for i in range(len(generated)+1)]
            if any(not text.startswith(p) for p in prefixes):
                raise ContractError("Generated token decoding is not prefix-stable; span alignment requires review")
            positions = [len(p) for p in prefixes]
            prefix = len(attempt["prompt_token_ids"])
            mask = [False]*prefix + [positions[i] >= start-1 and positions[i+1] <= end
                     and positions[i+1] > start
                     and positions[i+1] > positions[i] for i in range(len(generated))]
            token_ids = torch.tensor([attempt["prompt_token_ids"]+generated], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=token_ids, attention_mask=torch.ones_like(token_ids)).logits
                lp = answer_logprobs(logits, token_ids, torch.tensor([mask], device=self.device),
                                    torch.ones_like(token_ids))[0]
                value = float(lp.mean())
            confidence.append({"attempt": attempt, "answer_logprob_mean": value})
        values = [r["answer_logprob_mean"] for r in confidence]
        fields = {"target_val_accuracy": accuracy["target"], "ood_proxy_accuracy": accuracy["ood"],
                  "retention_anchor_accuracy": accuracy["retention"],
                  "gen_confidence_mean": float(np.mean(values)), "gen_confidence_std": float(np.std(values))}
        return accuracy, self._arrays(fields), {"scoring": scores, "confidence": confidence,
                                               "parent_sha256": self.parent_sha()}

    @staticmethod
    def _arrays(fields):
        return {name: np.asarray(value if isinstance(value, list) else [value], dtype=np.float32)
                for name, value in fields.items()}

    def _failure_batch(self, group, plan):
        pool = {r["prompt_id"]: r for r in self._rows("adaptation:"+group)}
        attempts, corrected = [], {}
        correction_attempts = []
        examples = []
        for prompt in plan.prompt_ids:
            row = pool[prompt]
            attempt = self._generate(row, seed=prompt_processing_seed(plan, prompt, stream="generation"))
            parsed = final_answer(attempt["text"], self._rule(row), at_eos=attempt["at_eos"])
            passed = verify_answer(parsed, row["reference"], self._rule(row)) and not attempt["truncated"]
            attempt["correct"] = passed
            attempts.append(attempt)
            if not passed:
                correction_row = {**row, "prompt": row["prompt"] +
                    "\nThe previous answer failed verification:\n" + attempt["text"] +
                    "\nReconsider the problem and provide a corrected final answer."}
                correction = self._generate(correction_row, seed=prompt_processing_seed(plan, prompt, stream="correction"))
                value = final_answer(correction["text"], self._rule(row), at_eos=correction["at_eos"])
                verified = verify_answer(value, row["reference"], self._rule(row)) and not correction["truncated"]
                correction_attempts.append({**correction, "correct": bool(verified)})
                if verified:
                    corrected[prompt] = value
                    examples.append(self._prepare(row, value))
        if not examples:
            raise ContractError("Empty failure batch; no unapproved fallback/no-op update")
        return examples, corrected, {"original": attempts, "corrections": correction_attempts}

    def prepare_candidate(self, group, state, candidate):
        pool = [r["prompt_id"] for r in self._rows("adaptation:"+group)]
        binding = self.config["continuation"]
        draws = build_pilot_state_plan(group_id=group, state=state, recipe=binding["recipe"],
            prompt_pool=pool, candidate_master_seed=binding["candidate_master_seed"],
            continuation_master_seed=binding["master_seed"])
        plan = draws.candidates[candidate]
        examples, corrections, attempts = self._failure_batch(group, plan)
        self.model.eval()
        with torch.no_grad():
            logprobs = self._logprobs(examples)
            embeddings = []
            with self.model.disable_adapter():
                for ex in examples:
                    out = self.model(**ex["tokens"], output_hidden_states=True)
                    token_ids = ex["tokens"]["input_ids"][0]
                    keep = ex["tokens"]["attention_mask"][0].bool()
                    for special in self.tokenizer.all_special_ids:
                        keep = keep & (token_ids != special)
                    if not bool(keep.any()):
                        raise ContractError("Empty frozen-base embedding population")
                    embeddings.append(out.hidden_states[-1][0, keep].float().mean(0).cpu().numpy())
        data, data_proof = candidate_data(logprobs=logprobs, lengths=[ex["length"] for ex in examples],
            correction_passes=[a["correct"] for a in attempts["corrections"]],
            original_passes=[a["correct"] for a in attempts["original"]], embeddings=embeddings,
            texts=[ex["text"] for ex in examples], definitions=self.config["definitions"])
        retention = [self._prepare(r) for r in self._rows("retention_gradient")]
        gradients, diagnostics = gradient_measurements(self.model, self.config["module_manifest"],
            lambda: self._loss_terms(examples), lambda: self._loss_terms(retention),
            zero_epsilon=self.config["definitions"]["cosine_zero_epsilon"])
        anchor = [self._prepare(r) for r in self._rows("anchor_probe")]
        validation = [self._prepare(r) for r in self._rows("val_probe")]
        probe_cfg = self.config["probe"]
        probes, probe_proof = probe_measurements(self.model, self.config["module_manifest"],
            train_loss=lambda step: self._loss([examples[(step-1) % len(examples)]]),
            train_eval_loss=lambda: self._loss(examples),
            anchor_loss=lambda: self._loss(anchor), val_loss=lambda: self._loss(validation),
            anchor_logits=lambda: self._anchor_logits(anchor), steps=probe_cfg["steps"],
            cadence=probe_cfg["cadence"], optimizer_config=probe_cfg["optimizer"], seed=plan.training_seed)
        features = {"candidate_data_stats": self._arrays(data),
                    "gradient_interference_stats": self._arrays(gradients), "short_probe_dynamics": self._arrays(probes)}
        evidence = {"source_plan": plan.sha256, "source_prompt_ids": list(plan.prompt_ids),
                    "excluded_h1_prompt_ids": list(draws.excluded_h1_prompt_ids),
                    "sibling_source_overlaps": json.loads(json.dumps(draws.overlaps)),
                    "attempts": attempts, "corrections": corrections,
                    "data": data_proof, "probe": probe_proof, "definitions": self.config["definitions"],
                    "serialized": [{"sha256": ex["serialized_sha256"], "prompt_id": ex["row"]["prompt_id"]}
                                   for ex in examples]}
        return features, evidence, {"examples": examples, "seed": plan.training_seed}, diagnostics

    def _update(self, examples, seed):
        cfg = self.config["update"]
        params = lora_parameters(self.model, self.config["module_manifest"])
        optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                                      betas=tuple(cfg["betas"]), eps=cfg["eps"])
        losses = []
        with seeded_stream(seed):
            for step in range(cfg["steps"]):
                self.check_limits()
                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss([examples[step % len(examples)]])
                if not bool(torch.isfinite(loss)):
                    raise ContractError("Nonfinite full-update loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        self.model.eval()
        return losses

    def run_sibling(self, batch, plan, candidate, out):
        out.mkdir(parents=True, exist_ok=False)
        started, parent = time.monotonic(), self.parent_sha()
        with restored_training_state(self.model) as snapshot:
            losses1 = self._update(batch["examples"], batch["seed"])
            h1_scores = {role: self._score(role) for role in ("target", "ood", "retention")}
            h1 = {"parent_sha256": parent, "endpoint_sha256": self.parent_sha(),
                  "accuracy": {role: value["accuracy"] for role, value in h1_scores.items()}, "scoring": h1_scores}
            saved = out/"h1.pt"
            with saved.open("xb") as f:
                torch.save({"semantic_hash": self.spec["semantic_hash"],
                    "parameters": {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}}, f)
            write_json(out/"h1_scoring.json", h1)
            examples, corrected, attempts = self._failure_batch(plan.group_id, plan)
            losses2 = self._update(examples, plan.training_seed)
            h2_scores = {role: self._score(role) for role in ("target", "ood", "retention")}
            h2 = {"parent_sha256": h1["endpoint_sha256"], "endpoint_sha256": self.parent_sha(),
                  "accuracy": {role: value["accuracy"] for role, value in h2_scores.items()}, "scoring": h2_scores}
            with (out/"h2.pt").open("xb") as f:
                torch.save({"semantic_hash": self.spec["semantic_hash"],
                    "parameters": {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}}, f)
            write_json(out/"h2_scoring.json", h2)
        runtime = {"h1_step_losses": losses1, "h2_step_losses": losses2,
            "continuation_attempts": attempts, "elapsed_seconds": time.monotonic()-started,
            "restoration_sha256": snapshot.before, "restoration_verified": True,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            "h1_snapshot_sha256": file_sha(saved)}
        write_json(out/"runtime.json", runtime)
        return h1, h2, corrected, saved, runtime
