# Context Dreamer

### Probing the Geometry of Sparse Autoencoder Features via Prompt Optimization

This project explores the **geometry and accessibility of Sparse Autoencoder (SAE) features in language models** by performing *DeepDream-style optimization over token embeddings*.

Instead of probing SAE features with fixed prompts, the experiment **optimizes prompts directly** to maximally activate individual SAE features in a language model. This allows us to study:

* Which features are easily activatable
* Which features are resistant to prompt optimization
* The distribution of feature accessibility across a layer

The result is a simple but powerful framework for **generative probing of representation geometry**.

---

## Motivation

Sparse Autoencoders (SAEs) are widely used in mechanistic interpretability to decompose model activations into interpretable features.

However, a key open question is:

> *How accessible are these features from natural language prompts?*

Some SAE features appear strongly tied to semantic concepts, while others may be difficult to trigger through natural language alone.

To explore this, this project introduces **Context Dreamer**, a procedure that:

1. Selects a target SAE feature
2. Optimizes token embeddings to maximize its activation
3. Decodes the optimized embeddings into text
4. Measures the resulting activation gain

This allows us to treat prompt optimization as a **probing mechanism for feature accessibility**.

---

## Method

### Model Setup

The experiment uses:

* **GPT-2 Small**
* **Sparse Autoencoder (SAE)** trained on a middle transformer layer
* **TransformerLens** for interpretability tooling
* **SAE Lens** for feature extraction

Libraries used:

```
transformer_lens
sae_lens
torch
```

---

### Prompt Optimization ("Dreaming")

The core procedure performs **gradient ascent on token embeddings** to maximize a target feature.

The process is similar to **DeepDream**, but applied to **language model representations**.

#### Objective

Maximize activation of feature *f*:

```
maximize SAE_feature_f(residual_stream)
```

while maintaining reasonable embedding structure via regularization.

#### Key Components

**Feature Activation Maximization**

The loss is defined as negative feature activation so that gradient descent increases activation.

**Embedding Regularization**

Regularization prevents the optimized embeddings from drifting too far from the valid embedding manifold.

**Diversity Constraints**

Additional penalties encourage diverse solutions rather than collapsing to a single trivial prompt.

---

## Feature Geometry Scan

To analyze the broader feature landscape, the notebook scans **a range of SAE features**.

Example configuration:

```
START_FEATURE = 0
END_FEATURE = 100
```

For each feature, the system measures:

### Spark Probability

Probability that a random prompt activates the feature above a threshold.

This captures how **naturally accessible** the feature is.

### Optimization Gain

Maximum activation increase achievable through prompt optimization.

This measures **how strongly the feature can be stimulated** when directly optimized.

---

## Observations

The experiment reveals several interesting patterns:

### Heavy-Tailed Accessibility

Feature accessibility appears highly uneven:

* Some features activate easily from natural prompts
* Others remain nearly silent unless optimized directly

This produces a **heavy-tailed distribution** over feature activation.

---

### Super-Stimulus Effects

Certain features can be driven to **extremely high activation levels** when optimized.

In some cases, activation exceeds **1000× baseline levels**, suggesting that prompt optimization can create **super-stimuli** for specific internal concepts.

---

### Dead or Inaccessible Features

Some features show:

* Near-zero spark probability
* Minimal optimization gain

These may represent:

* Rare concepts
* Dataset artifacts
* Features requiring long-range context

---

## Visualization

After scanning features, the notebook produces visualizations of:

* Feature accessibility distributions
* Spark probability histograms
* Optimization gain statistics

These plots help characterize the **activation geometry of the SAE feature space**.

---

## Example: Dreaming a Feature

The notebook includes an example where the system "dreams" prompts that strongly activate a specific feature.

Example call:

```python
text_11 = deep_dream_last_token_robust(
    feature_idx=11,
    steps=150,
    reg_weight=1.0,
    div_weight=100.0
)
```

The optimization process iteratively updates token embeddings until the target feature reaches high activation.

The decoded prompt provides a **human-readable glimpse into what the feature represents**.

---

## Running the Notebook

Install dependencies:

```
pip install transformer_lens sae_lens torch
```

Then open the notebook:

```
ContextDreamer.ipynb
```

The notebook performs the following steps:

1. Load GPT-2 and SAE weights
2. Run prompt optimization for selected features
3. Scan a range of SAE features
4. Save results to `sae_geometry_scan.csv`
5. Visualize the feature accessibility landscape

---

## Research Direction

This project is part of an ongoing effort to explore:

* Representation geometry in language models
* Feature accessibility in sparse representations
* Generative probing methods for interpretability

Future extensions include:

* Scaling to **thousands of SAE features**
* Testing across **multiple transformer layers**
* Studying **sparsity vs optimization gain relationships**
* Comparing accessibility across different models

---

## Repository Structure

```
context-dreamer
│
├── ContextDreamer.ipynb
│   Main experiment notebook
│
└── README.md
    Project overview
```

---

## Author

**Priyanshu Mohanty**

Research interests: Mechanistic Interpretability · Representation Learning · LLM Safety · Controllable Generation

GitHub: [https://github.com/pri-Mohanty](https://github.com/pri-Mohanty)
