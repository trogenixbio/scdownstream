#!/usr/bin/env python3

import scvi
import anndata as ad
import pandas as pd
from scvi.model import SCVI
from scvi.external import SOLO
import platform

from threadpoolctl import threadpool_limits

threadpool_limits(int("${task.cpus}"))


def format_yaml_like(data: dict, indent: int = 0) -> str:
    """Formats a dictionary to a YAML-like string.

    Args:
        data (dict): The dictionary to format.
        indent (int): The current indentation level.

    Returns:
        str: A string formatted as YAML.
    """
    yaml_str = ""
    for key, value in data.items():
        spaces = "  " * indent
        if isinstance(value, dict):
            yaml_str += f"{spaces}{key}:\\n{format_yaml_like(value, indent + 1)}"
        else:
            yaml_str += f"{spaces}{key}: {value}\\n"
    return yaml_str


adata = ad.read_h5ad("${h5ad}")

setup_kwargs = {"layer": "counts", "batch_key": "batch"}

# Defaults from SCVI github tutorials scanpy_pbmc3k and harmonization
model_kwargs = {
    "gene_likelihood": "nb",
    "n_layers": 2,
    "n_hidden": 128,
    "n_latent": 30,
}

train_kwargs = {"train_size": 1.0}

n_epochs = int(min([round((20000 / adata.n_obs) * 400), 400]))

SCVI.setup_anndata(adata, **setup_kwargs)
model = SCVI(adata, **model_kwargs)
model.train(max_epochs=n_epochs, **train_kwargs)

unique_labels = set(adata.obs["label"].unique())
unique_labels.discard("unknown")
if len(unique_labels) > 0:
    from scvi.model import SCANVI

    n_epochs = int(min([10, max([2, round(n_epochs / 3.0)])]))

    model = SCANVI.from_scvi_model(
        scvi_model=model, labels_key="label", unlabeled_category="unknown"
    )
    model.train(max_epochs=n_epochs, **train_kwargs)

results = []

batches = adata.obs["batch"].unique()
for batch in batches:
    solo = SOLO.from_scvi_model(
        model, restrict_to_batch=batch if len(batches) > 1 else None
    )
    solo.train()
    result = pd.concat([solo.predict(False), solo.predict(True)], axis=0)

    results.append(result)

df_results = pd.concat(results, axis=1)
df_results = df_results.loc[adata.obs_names]
print(df_results)

df = adata.obs[["doublet"]]
df.columns = ["${prefix}"]
df.to_pickle("${prefix}.pkl")

adata = adata[~adata.obs["doublet"]].copy()
adata.obs.drop("doublet", axis=1, inplace=True)

adata.write_h5ad("${prefix}.h5ad")

# Versions

versions = {
    "${task.process}": {
        "python": platform.python_version(),
        "anndata": ad.__version__,
        "scvi": scvi.__version__,
        "pandas": pd.__version__,
    }
}

with open("versions.yml", "w") as f:
    f.write(format_yaml_like(versions))
