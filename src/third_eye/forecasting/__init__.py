"""Third Eye forecasting stack (Mahitha's modeling component).

Modules
-------
layout      feature layout (which columns exist, grouped for ablations)
features    pinned, numpy-only definitions of the schema's features
data        meta-record validation, arrays, grouped splits, normalisation
models      one-step MLP baseline, Third Eye-Direct v2, Third Eye-Dynamics
train       fitting, prediction, grouped cross-validation, checkpoints
metrics     ranking / regression metrics and cluster bootstrap
selection   candidate-selection policies, offline replay, gates, gated PD
synthetic   planted-signal generator used for positive/negative controls

Nothing here touches model weights of the backbone LLMs or any GPU job.
"""

__version__ = "0.1.0"
