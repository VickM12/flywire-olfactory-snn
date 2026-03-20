from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    seed: int = 7
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_size: int = 256
    snn_steps: int = 20
    snn_dt: float = 1.0
    snn_alpha: float = 100.0
    noise_std: float = 0.08
    train_trials_per_odor: int = 24
    val_trials_per_odor: int = 8
    test_trials_per_odor: int = 8
    heldout_odor_fraction: float = 0.2
    max_olfactory_neurons: int = 800
    data_dir: Path = Path("data")
    result_dir: Path = Path("results")
    connectome_cache: str = "olfactory_connectome.npz"
    annotation_dataset: str = "public"
    materialization: str = "auto"
    rebuild_connectome: bool = False
    require_real_connectome: bool = False

