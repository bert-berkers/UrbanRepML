"""Clustering probe: predict concat k=10 clusters from UNet MS 192D.

Lifetime: temporary (one_off)
Stage: 3
"""
import copy
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from stage3_analysis.dnn_probe import MLPProbeModel, _make_activation

DATA_ROOT = project_root / "data/study_areas/netherlands"
CONCAT_PATH = DATA_ROOT / "stage2_multimodal/concat/embeddings/netherlands_res9_20mix.parquet"
UNET9_PATH = DATA_ROOT / "stage2_multimodal/unet/embeddings/netherlands_res9_20mix.parquet"
UNET8_PATH = DATA_ROOT / "stage2_multimodal/unet/embeddings/netherlands_res8_20mix.parquet"
UNET7_PATH = DATA_ROOT / "stage2_multimodal/unet/embeddings/netherlands_res7_20mix.parquet"
OUTPUT = DATA_ROOT / "stage2_multimodal/clustering/2026-03-08_clustering_probe"
OUTPUT.mkdir(parents=True, exist_ok=True)

K = 10

print("Loading embeddings...")
concat_emb = pd.read_parquet(CONCAT_PATH)
print(f"  Concat: {concat_emb.shape}")

# Build UNet multiscale
unet9 = pd.read_parquet(UNET9_PATH)
unet8 = pd.read_parquet(UNET8_PATH)
unet7 = pd.read_parquet(UNET7_PATH)

unet9.columns = [f"unet9_{c.split('_', 1)[1]}" for c in unet9.columns]
unet8.columns = [f"unet8_{c.split('_', 1)[1]}" for c in unet8.columns]
unet7.columns = [f"unet7_{c.split('_', 1)[1]}" for c in unet7.columns]


def upsample_to_res9(parent_df, parent_res):
    rows = []
    for parent_id in parent_df.index:
        children = h3.cell_to_children(parent_id, 9)
        for child in children:
            rows.append({"region_id": child, "parent_id": parent_id})
    mapping = pd.DataFrame(rows).set_index("region_id")
    return mapping.join(parent_df, on="parent_id").drop(columns="parent_id")


print("Upsampling res8/7 -> res9...")
unet8_up = upsample_to_res9(unet8, 8)
unet7_up = upsample_to_res9(unet7, 7)
unet_ms = unet9.join(unet8_up, how="inner").join(unet7_up, how="inner")
print(f"  UNet MS: {unet_ms.shape}")

# Align indices
common = concat_emb.index.intersection(unet_ms.index)
print(f"Common hexagons: {len(common):,}")

concat_aligned = concat_emb.loc[common]
unet_aligned = unet_ms.loc[common]

# Cluster concat
print(f"Clustering concat with KMeans k={K}...")
scaler_c = StandardScaler()
concat_scaled = scaler_c.fit_transform(concat_aligned.values)
kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=10_000, n_init=3)
cluster_labels = kmeans.fit_predict(concat_scaled)

# Train DNN classifier
n = len(common)
rng = np.random.RandomState(42)
perm = rng.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = perm[:split], perm[split:]

X = unet_aligned.values.astype(np.float32)
y = cluster_labels

scaler_u = StandardScaler()
X_train = scaler_u.fit_transform(X[train_idx])
X_test = scaler_u.transform(X[test_idx])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training DNN classifier on {device}...")

model = MLPProbeModel(
    input_dim=X.shape[1],
    hidden_dim=128,
    num_layers=3,
    use_layer_norm=True,
    output_dim=K,
    activation=_make_activation("silu"),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

x_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y[train_idx], dtype=torch.long, device=device)
x_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_t = torch.tensor(y[test_idx], dtype=torch.long, device=device)

best_state = None
best_loss = float("inf")

for epoch in range(100):
    model.train()
    # Mini-batch
    perm_t = torch.randperm(len(x_train_t), device=device)
    for start in range(0, len(x_train_t), 8192):
        idx = perm_t[start:start + 8192]
        optimizer.zero_grad()
        out = model(x_train_t[idx])
        loss = criterion(out, y_train_t[idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_out = model(x_test_t)
        test_loss = criterion(test_out, y_test_t).item()
    if test_loss < best_loss:
        best_loss = test_loss
        best_state = copy.deepcopy(model.state_dict())

    if (epoch + 1) % 20 == 0:
        pred = test_out.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y[test_idx], pred)
        print(f"  Epoch {epoch+1}: loss={test_loss:.4f}, acc={acc:.3f}")

if best_state is not None:
    model.load_state_dict(best_state)

model = model.cpu().eval()
with torch.no_grad():
    test_logits = model(torch.tensor(X_test, dtype=torch.float32))
    test_pred = test_logits.argmax(dim=1).numpy()

acc = accuracy_score(y[test_idx], test_pred)
f1_macro = f1_score(y[test_idx], test_pred, average="macro")
f1_weighted = f1_score(y[test_idx], test_pred, average="weighted")

print(f"\nResults: Acc={acc:.3f}, F1-macro={f1_macro:.3f}, F1-weighted={f1_weighted:.3f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
metrics = {"Accuracy": acc, "F1 (macro)": f1_macro, "F1 (weighted)": f1_weighted}
bars = ax.bar(metrics.keys(), metrics.values(),
              color=["steelblue", "coral", "seagreen"], edgecolor="white")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylim(0, 1.0)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(
    f"Clustering Probe: UNet MS 192D -> Concat k={K} Clusters\n"
    f"DNN classifier (80/20 split, {len(common):,} hexagons)",
    fontsize=12,
)
ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
path = OUTPUT / "clustering_probe_bar.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

with open(OUTPUT / "clustering_probe_metrics.json", "w") as f:
    json.dump({"k": K, "accuracy": acc, "f1_macro": f1_macro,
               "f1_weighted": f1_weighted, "n_common": len(common)}, f, indent=2)

print("Done.")
