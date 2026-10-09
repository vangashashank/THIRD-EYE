from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def prepare_adapter_base(model, *, is_trainable):
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=is_trainable,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.config.use_cache = False
    return model


def attach_lora(
    model,
    r=8,
    alpha=16,
    dropout=0.05,
    target_modules=None
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    model = prepare_adapter_base(model, is_trainable=True)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(
        model,
        lora_config
    )

    return model
