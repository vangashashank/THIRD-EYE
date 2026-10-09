import torch
from torch import nn


class ThirdEyeDirect(nn.Module):
    """
    Predict H=2 changes in:
        [target performance, OOD performance, retention]

    All input features must be available before the proposed update.
    Positive output values mean improved performance.
    """

    def __init__(
        self,
        state_dim,
        candidate_dim,
        history_dim,
        hidden_dim=32,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.history_dim = history_dim

        self.history_encoder = nn.GRU(
            input_size=history_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        combined_dim = state_dim + candidate_dim + hidden_dim

        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, state_features, candidate_features, history=None):
        """
        state_features:     [batch, state_dim]
        candidate_features: [batch, candidate_dim]
        history:            [batch, steps, history_dim], or None

        Returns:
            [batch, 3]

        Use None when no history exists.
        Within a batch, histories must have equal actual lengths.
        """

        if state_features.ndim != 2:
            raise ValueError("state_features must be a 2-D tensor.")

        if candidate_features.ndim != 2:
            raise ValueError("candidate_features must be a 2-D tensor.")

        batch_size = state_features.shape[0]

        if state_features.shape[1] != self.state_dim:
            raise ValueError("Incorrect state feature dimension.")

        if candidate_features.shape != (
            batch_size,
            self.candidate_dim,
        ):
            raise ValueError("Incorrect candidate feature shape.")

        if history is None:
            history_summary = state_features.new_zeros(
                batch_size,
                self.history_encoder.hidden_size,
            )
        else:
            if (
                history.ndim != 3
                or history.shape[0] != batch_size
                or history.shape[1] == 0
                or history.shape[2] != self.history_dim
            ):
                raise ValueError("Incorrect history shape.")

            _, hidden = self.history_encoder(history)
            history_summary = hidden[-1]

        combined = torch.cat(
            [
                state_features,
                candidate_features,
                history_summary,
            ],
            dim=-1,
        )

        return self.predictor(combined)
