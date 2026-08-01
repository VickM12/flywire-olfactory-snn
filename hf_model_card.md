---
license: mit
library_name: pytorch
tags:
  - spiking-neural-network
  - neuroscience
  - connectome
  - drosophila
  - olfaction
  - norse
  - biology
datasets:
  - {{HF_USER}}/door-olfactory-responses
pipeline_tag: tabular-classification
---

# FlyWire Olfactory SNN (MaskedRecurrentLIFSNN)

A **connectome-constrained recurrent spiking neural network** for odor identity
classification in *Drosophila melanogaster*, trained on the
[DoOR](https://github.com/ropensci/DoOR.data) olfactory receptor response dataset.

## Model description

The recurrent connectivity of this SNN is fixed to the **FlyWire** connectome
subgraph (antennal lobe projection neurons + mushroom body Kenyon cells). Synaptic
signs (excitatory/inhibitory) come from predicted neurotransmitter types in FlyWire.
Only the **weight magnitudes** are learned; the topology is biological.

### Architecture

```
Input: odor receptor vector (DoOR: ~52 receptors)
  → Linear(input_dim → hidden_dim, no bias)
  → 20 LIF timesteps with:
      • Poisson spike encoding from rate-coded input
      • Recurrent current: spk × (W_rec ⊙ mask ⊙ sign)ᵀ
      • Norse LIFCell (surrogate gradient, α=100)
  → time-averaged spike rates
  → Linear(hidden_dim → num_classes)
```

- **Neuron model:** Leaky Integrate-and-Fire (Norse `LIFCell`, `method="super"`)
- **Recurrent mask:** Binary from FlyWire adjacency (fixed, not learned)
- **Synaptic signs:** ACh/DA/5-HT/OA → +1 (excitatory); GABA/Glu → −1 (inhibitory)
- **Training:** Adam optimizer, CrossEntropyLoss, surrogate gradients through LIF

## Files

| File | Description |
|------|-------------|
| `model.safetensors` | Trained weights (best validation checkpoint) |
| `config.json` | Architecture hyperparameters |
| `connectome_mask.npz` | FlyWire olfactory subgraph (binary adjacency + signs) |
| `connectome_meta.json` | Connectome metadata (neuron count, edge count, source) |
| `modeling_snn.py` | Standalone `MaskedRecurrentLIFSNN` class |

## Usage

```python
import scipy.sparse as sp
import torch
from safetensors.torch import load_file

# Load the model
from modeling_snn import MaskedRecurrentLIFSNN

adjacency = sp.load_npz("connectome_mask.npz")
model = MaskedRecurrentLIFSNN(
    input_dim=52,       # from config.json
    hidden_dim=800,     # from config.json
    num_classes=500,    # from config.json
    adjacency=adjacency,
    steps=20,
    alpha=100.0,
)
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
model.eval()

# Inference
x = torch.randn(1, 52)  # receptor activation vector
logits, spike_sparsity = model(x)
predicted_odor = logits.argmax(dim=1).item()
```

## Training details

- **Dataset:** DoOR (Database of Odorant Responses) — CC BY-SA 4.0
- **Cross-validation:** 5-fold over odor identities × 5 seeds = 25 runs
- **Early stopping:** patience 5 on validation accuracy
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)
- **Batch size:** 32
- **Max epochs:** 80
- **SNN timesteps:** 20
- **Evaluation:** 5× Monte Carlo averaging over stochastic Poisson encoding

## Biological basis

The model's recurrent topology is extracted from the [FlyWire](https://flywire.ai/)
whole-brain connectome of *Drosophila melanogaster* (FAFB dataset). The olfactory
subgraph includes:

- **Antennal Lobe Projection Neurons (ALPN):** relay processed odor information
- **Kenyon Cells (KC):** mushroom body neurons for associative olfactory memory

This captures the AL → PN → KC pathway that the fly uses for odor discrimination
and learning.

## Citation

If you use this model, please cite:

- The **DoOR** database: Münch & Galizia (2016). DoOR 2.0 — Comprehensive mapping
  of *Drosophila melanogaster* odorant responses. *Scientific Reports*, 6, 21841.
  https://doi.org/10.1038/srep21841
- The **FlyWire** connectome: Dorkenwald et al. (2024). Neuronal wiring diagram of
  an adult brain. *Nature*, 634, 124–138. https://doi.org/10.1038/s41586-024-07558-y
