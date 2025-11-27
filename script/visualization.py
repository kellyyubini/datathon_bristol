import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datamapplot
from sentence_transformers import SentenceTransformer
from umap import UMAP


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INPUT_FOLDER = "output_optimized"
OUTPUT_PLOT = "mapa_manual.png"



print("Loading datasets...")
topic_results = pd.read_csv(os.path.join(INPUT_FOLDER, "topic_results.csv"))
topic_info = pd.read_csv(os.path.join(INPUT_FOLDER, "topic_info.csv"))

# Verificamos qué columnas existen
print("Columns in topic_results:", topic_results.columns.tolist())
print("Columns in topic_info:", topic_info.columns.tolist())



label_map = dict(zip(topic_info["Topic"], topic_info["label"]))

topic_results["topic_label"] = topic_results["topic"].map(label_map)
topic_results["topic_label"] = topic_results["topic_label"].fillna("Other")

print("Labels assigned!")



viz_data = topic_results.copy()
viz_data = viz_data[viz_data["topic"] != -1]  # excluir outliers

if len(viz_data) > 6000:
    viz_data = viz_data.sample(6000, random_state=42)

# Elegimos la columna correcta
if "clean_content" in viz_data.columns:
    CONTENT_COL = "clean_content"
else:
    CONTENT_COL = "summary"


print("Generating embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(viz_data[CONTENT_COL].astype(str).tolist(), show_progress_bar=True)



print("Running UMAP 2D...")
umap_2d = UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    metric="cosine",
    random_state=42
).fit_transform(embeddings)


print("Generating the map...")
datamapplot.create_plot(
    umap_2d,
    viz_data["topic_label"].values,
    label_font_size=9,
    label_wrap_width=18,
    use_medoids=True,
    dynamic_label_size=True,
    figsize=(14, 10),
    dpi=150,
    darkmode=False
)

plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nMapa guardado en: {OUTPUT_PLOT}")
