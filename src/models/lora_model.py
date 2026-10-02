from peft import LoraConfig, get_peft_model


def attach_lora(
    model,
    r=8,
    alpha=16,
    dropout=0.05,
    target_modules=None
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

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
