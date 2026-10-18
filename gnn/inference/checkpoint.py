from pathlib import Path

import torch

from gnn.models.model import (
    ConPaSSharedEncoder,
    GNNFixingMIPBaseline,
    GraphEBM,
    LinearMIPBCEBaseline,
    LinearMIPContrastiveBaseline,
    PredictAndSearchBaseline,
)


MODEL_CLASSES = {
    "energnn": GraphEBM,
    "linear_mip_bce": LinearMIPBCEBaseline,
    "linear_mip_contrastive": LinearMIPContrastiveBaseline,
    # Compatibility for callers reading legacy run metadata directly.
    "linear_mip": LinearMIPContrastiveBaseline,
    "linear_score_mip": LinearMIPBCEBaseline,
    "gnn_fixing_mip": GNNFixingMIPBaseline,
    "predict_and_search": PredictAndSearchBaseline,
    "conpas_shared_encoder": ConPaSSharedEncoder,
}


def load_trained_model(
    model_type: str,
    model_config: dict,
    checkpoint_path: str | Path,
    device: torch.device,
):
    """Load a checkpoint produced by ``gnn.training.train`` for inference."""
    try:
        model_class = MODEL_CLASSES[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_CLASSES))
        raise ValueError(f"Unsupported model type {model_type!r}; expected one of {supported}.") from exc

    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain the expected 'model_state'."
        )

    model = model_class(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model
