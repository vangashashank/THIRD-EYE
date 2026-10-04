from pathlib import Path
import copy
import json

import torch

from src.data.dataset import TextDataset


def compute_data_stats(texts):
    """
    Pure text statistics about one candidate's training batch. No model,
    no GPU, no training needed -- just math on the text itself. Safe to
    compute on any machine, including a laptop.

    Returns:
        word_count_mean / word_count_std: length of each example, in words
        duplicate_rate: fraction of examples that are exact duplicates
                         of another example in this same batch
        unique_word_ratio: a cheap diversity proxy -- fraction of all
                            words across the batch that are unique
                            (lower = more repetitive vocabulary)
    """

    if not texts:
        return {
            "word_count_mean": 0.0,
            "word_count_std": 0.0,
            "duplicate_rate": 0.0,
            "unique_word_ratio": 0.0,
        }

    word_counts = [len(t.split()) for t in texts]
    n = len(word_counts)

    mean = sum(word_counts) / n
    variance = sum((w - mean) ** 2 for w in word_counts) / n
    std = variance ** 0.5

    # Exact-duplicate rate: how many texts are literal repeats of another
    seen = {}
    duplicate_count = 0
    for t in texts:
        normalized = t.strip().lower()
        if normalized in seen:
            duplicate_count += 1
        else:
            seen[normalized] = True
    duplicate_rate = duplicate_count / n

    # Cheap diversity proxy: unique words / total words across the batch.
    # Close to 1.0 means almost no word repeats anywhere in the batch;
    # close to 0.0 means the batch reuses the same words heavily.
    all_words = []
    for t in texts:
        all_words.extend(t.lower().split())

    if all_words:
        unique_word_ratio = len(set(all_words)) / len(all_words)
    else:
        unique_word_ratio = 0.0

    return {
        "word_count_mean": mean,
        "word_count_std": std,
        "duplicate_rate": duplicate_rate,
        "unique_word_ratio": unique_word_ratio,
    }


def _grad_norm_and_cosine(model, train_dataset, retention_dataset, device, batch_size=1):
    """
    Computes the LoRA gradient on the candidate's own training batch, and
    separately on a small retention/anchor batch, then compares directions.
    Neither gradient is ever applied — both are discarded via zero_grad
    before returning, so the model is left exactly as it was found.
    """

    model.train()

    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]

    def _one_gradient(dataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        model.zero_grad(set_to_none=True)

        total_loss = 0.0
        steps = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            outputs.loss.backward()
            total_loss += outputs.loss.item()
            steps += 1

        grads = [
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in trainable_params
        ]

        model.zero_grad(set_to_none=True)

        return grads, total_loss / max(steps, 1)

    candidate_grads, candidate_loss = _one_gradient(train_dataset)
    retention_grads, retention_loss = _one_gradient(retention_dataset)

    flat_candidate = torch.cat([g.flatten() for g in candidate_grads])
    flat_retention = torch.cat([g.flatten() for g in retention_grads])

    grad_norm = flat_candidate.norm().item()

    cosine = torch.nn.functional.cosine_similarity(
        flat_candidate.unsqueeze(0),
        flat_retention.unsqueeze(0),
    ).item()

    return {
        "gradient_norm": grad_norm,
        "retention_gradient_cosine": cosine,
        "pre_update_train_loss": candidate_loss,
        "pre_update_retention_loss": retention_loss,
    }


def _short_probe(model, train_dataset, retention_dataset, device, probe_steps=15, batch_size=1, lr=1e-4):
    """
    Trains for a small, fixed number of steps (disposable), watches how
    training loss and retention loss move, then restores the model's
    original weights so this probe never becomes a real update.
    """

    # Save a CPU copy of the trainable (LoRA) weights only, so we can
    # restore them after the probe without re-loading the whole model.
    pre_probe_state = {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    model.train()

    train_losses = []
    retention_losses = []

    retention_loader = torch.utils.data.DataLoader(
        retention_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    step = 0
    while step < probe_steps:
        for batch in loader:
            if step >= probe_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            outputs.loss.backward()
            optimizer.step()

            train_losses.append(outputs.loss.item())

            with torch.no_grad():
                retention_loss_total = 0.0
                retention_steps = 0
                for r_batch in retention_loader:
                    r_outputs = model(
                        input_ids=r_batch["input_ids"].to(device),
                        attention_mask=r_batch["attention_mask"].to(device),
                        labels=r_batch["labels"].to(device),
                    )
                    retention_loss_total += r_outputs.loss.item()
                    retention_steps += 1
                retention_losses.append(retention_loss_total / max(retention_steps, 1))

            step += 1

    # --- restore the model to its pre-probe weights ---
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(pre_probe_state[name].to(device))

    def _slope(values):
        n = len(values)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(values) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
        den = sum((x - mean_x) ** 2 for x in xs)
        return num / den if den > 0 else 0.0

    return {
        "probe_steps": probe_steps,
        "probe_train_loss_slope": _slope(train_losses),
        "probe_retention_loss_slope": _slope(retention_losses),
    }


def extract_candidate_features(
    model,
    tokenizer,
    device,
    train_texts,
    retention_texts,
    state_id,
    candidate_id,
    max_length=128,
    probe_steps=15,
    output_file="outputs/candidate_features.jsonl",
):
    """
    Call this on the UNTOUCHED parent model, right after it's loaded and
    made trainable, BEFORE train_lora() is called in candidate_generator.py.
    Writes one JSON line with everything needed as forecaster input features
    for this (state_id, candidate_id) pair.
    """

    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    retention_dataset = TextDataset(
        texts=retention_texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    data_stats = compute_data_stats(train_texts)

    grad_features = _grad_norm_and_cosine(
        model, train_dataset, retention_dataset, device
    )

    probe_features = _short_probe(
        model, train_dataset, retention_dataset, device, probe_steps=probe_steps
    )

    features = {
        "state_id": state_id,
        "candidate_id": candidate_id,
        **data_stats,
        **grad_features,
        **probe_features,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as f:
        f.write(json.dumps(features) + "\n")

    return features
