import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datamapplot
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuración global
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Inicializar el generador de etiquetas
print("Loading label generator model...")
try:
    LABEL_GENERATOR = pipeline("text2text-generation", model="google/flan-t5-small")
except:
    LABEL_GENERATOR = None
    print("Could not load label generator, will use simple labels")

def create_stopwords():
    from sklearn.feature_extraction.text import CountVectorizer
    sw = list(CountVectorizer(stop_words="english").get_stop_words())
    additional_sw = [
        "utility", "model", "discloses", "provides", "comprises",
        "invention", "provide", "solution", "jpo", "copyright", "solved",
        "financial", "planning", "help", "services", "accounting",
        "bookkeeping", "including", "range", "products", "company",
        "limited", "ltd", "plc", "inc", "business", "service"
    ]
    sw.extend(additional_sw)
    return sw

def load_data(path):
    """Loads and prepares the dataset."""
    df = pd.read_csv(path)
    
    # Verificar qué columnas están disponibles
    if 'clean_content' in df.columns:
        texts = df["clean_content"].astype(str).tolist()
    elif 'summary' in df.columns:
        texts = df["summary"].astype(str).tolist()
    else:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("No 'clean_content' or 'summary' column found")
    
    return df, texts

def initialize_topic_model(embedding_model):
    """Modelo optimizado con todos los ajustes."""
    
    # UMAP mejorado
    umap_model = UMAP(
        n_neighbors=20,
        n_components=20,  # Más dimensiones
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=False
    )
    
    # HDBSCAN balanceado
    hdbscan_model = HDBSCAN(
        min_cluster_size=25,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
        cluster_selection_epsilon=0.3
    )
    
    # TF-IDF vectorizer
    vectorizer_model = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.5,
        max_features=1000,
        stop_words=create_stopwords(),
        sublinear_tf=True
    )
    
    # Representación mejorada
    representation_model = MaximalMarginalRelevance(diversity=0.3)
    
    topic_model = BERTopic(
        min_topic_size=10,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        language="english",
        verbose=True,
        calculate_probabilities=False,  # Cambiar a False para evitar el error
        nr_topics="auto"
    )
    return topic_model

def generate_smart_labels(topic_info):
    """Genera etiquetas únicas e informativas."""
    
    # Mapeo manual para tópicos conocidos basado en keywords
    topic_mappings = {
        "tax": "Tax & Compliance",
        "bookkeeping": "Bookkeeping Services",
        "accounting": "Accounting Services", 
        "taxi": "Taxi Services",
        "airport": "Airport Transport",
        "vehicle": "Vehicle Services",
        "driver": "Transportation",
        "financial": "Financial Advisory",
        "planning": "Financial Planning",
        "products": "Product Solutions",
        "leasing": "Leasing Services",
        "finance": "Finance Services",
        "help": "Support Services",
        "range": "Product Range"
    }
    
    used_labels = set()
    
    def create_unique_label(row):
        if row["Topic"] == -1:
            return "Miscellaneous"
        
        words = [w.lower() for w in row["Representation"][:10]]
        
        # Buscar mapeo directo
        for key, label in topic_mappings.items():
            if key in words[:5]:
                if label not in used_labels:
                    used_labels.add(label)
                    return label
        
        # Si hay generador de etiquetas, usarlo
        if LABEL_GENERATOR:
            try:
                prompt = f"Create a 3-4 word business category label for: {', '.join(words[:6])}"
                result = LABEL_GENERATOR(prompt, max_length=30, temperature=0.3)
                label = result[0]['generated_text'].strip()
                
                # Verificar que la etiqueta no sea muy genérica
                if len(label.split()) <= 5 and label not in used_labels:
                    used_labels.add(label)
                    return label
            except:
                pass
        
        # Generar etiqueta única basada en palabras clave
        unique_words = [w for w in words[:8] if len(w) > 3]
        label = " ".join(unique_words[:3]).title()
        
        # Asegurar unicidad
        counter = 1
        original_label = label
        while label in used_labels:
            label = f"{original_label} {counter}"
            counter += 1
        
        used_labels.add(label)
        return label
    
    topic_info["label"] = topic_info.apply(create_unique_label, axis=1)
    return topic_info

def create_visualization(topic_results, topic_info, embedding_model, output_dir):
    """Creates and saves a 2D visualization of the topics."""
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar outliers para una visualización más limpia - CREAR COPIA EXPLÍCITA
    viz_data = topic_results[topic_results["topic"] != -1].copy()
    
    # Si hay muchos documentos, tomar una muestra
    if len(viz_data) > 5000:
        viz_data = viz_data.sample(n=5000, random_state=42).copy()
    
    # Top topics por tamaño
    top_topics_info = topic_info[topic_info.Topic != -1].sort_values(
        "Count", ascending=False
    ).head(30)
    
    topic_labels = {
        row["Topic"]: str(row["label"]).strip()
        for _, row in top_topics_info.iterrows()
        if pd.notna(row["label"])
    }
    
    # CORREGIDO: Usar asignación directa sin .loc
    viz_data["topic_label"] = viz_data["topic"].map(topic_labels).fillna("Other")
    
    print("Creating 2D projection for visualization...")
    
    # Determinar qué columna usar para el contenido
    if 'clean_content' in viz_data.columns:
        content_column = 'clean_content'
    else:
        content_column = 'summary'
    
    embeddings_visual = embedding_model.encode(
        viz_data[content_column].astype(str).tolist(),
        show_progress_bar=True
    )
    
    # UMAP 2D para visualización
    coordinates_2d = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        spread=1.5
    ).fit_transform(embeddings_visual)
    
    print("Creating plot...")
    datamapplot.create_plot(
        coordinates_2d,
        viz_data["topic_label"].values,
        label_font_size=8,
        label_wrap_width=15,
        use_medoids=True,
        dynamic_label_size=True,
        figsize=(16, 12),
        dpi=150,
        darkmode=False
    )
    
    plot_path = os.path.join(output_dir, "topics_plot_optimized.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"Plot saved to: {plot_path}")

def main():
    print("Loading data...")
    df, texts = load_data("problem1_2021_finance.csv")
    print(f"Loaded {len(texts)} documents")
    
    print("Creating embeddings...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    print("Initializing optimized topic model...")
    topic_model = initialize_topic_model(embedding_model)
    
    print("Fitting model...")
    topics, probs = topic_model.fit_transform(texts, embeddings)
    
    # Opción 1: Reducir outliers con estrategia diferente (sin probabilidades)
    print("Reducing outliers...")
    try:
        # Usar estrategia 'distributions' que no requiere probabilidades
        new_topics = topic_model.reduce_outliers(texts, topics, strategy="distributions")
        topics = new_topics
    except Exception as e:
        print(f"Could not reduce outliers: {e}")
        print("Continuing without outlier reduction...")
    
    # Crear resultados
    topic_results = df.copy()
    topic_results["topic"] = topics
    
    # Manejar probabilidades de manera segura
    if probs is not None:
        topic_results["probability"] = probs
    else:
        topic_results["probability"] = 0.5  # Valor por defecto
    
    # Obtener información de tópicos
    topic_info = topic_model.get_topic_info()
    print(f"Found {len(topic_info) - 1} topics")
    
    # Generar etiquetas mejoradas
    print("Generating smart labels...")
    topic_info = generate_smart_labels(topic_info)
    
    # Crear visualización
    print("Creating visualization...")
    try:
        create_visualization(topic_results, topic_info, embedding_model, "output_optimized")
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
    
    # Guardar resultados
    os.makedirs("output_optimized", exist_ok=True)
    topic_info.to_csv("output_optimized/topic_info.csv", index=False)
    topic_results.to_csv("output_optimized/topic_results.csv", index=False)
    print("Results saved!")
    
    # Mostrar estadísticas
    print("\n" + "="*50)
    print("TOPIC MODELING RESULTS")
    print("="*50)
    print(f"Total topics found: {len(topic_info) - 1}")
    print(f"Documents in outliers: {(topics == -1).sum()} ({(topics == -1).sum()/len(topics)*100:.1f}%)")
    
    # Verificar que hay topics antes de calcular max/min
    if len(topic_info[topic_info.Topic != -1]) > 0:
        print(f"Largest topic size: {topic_info[topic_info.Topic != -1]['Count'].max()}")
        print(f"Smallest topic size: {topic_info[topic_info.Topic != -1]['Count'].min()}")
    
    print("\n" + "="*50)
    print("TOP 15 TOPICS")
    print("="*50)
    
    # Mostrar top topics
    top_topics = topic_info[topic_info.Topic != -1].head(15)
    if len(top_topics) > 0:
        for idx, row in top_topics.iterrows():
            words = ", ".join(row['Representation'][:5])
            print(f"\nTopic {row['Topic']:2d}: {row['label']}")
            print(f"  Size: {row['Count']} documents")
            print(f"  Keywords: {words}")
    else:
        print("No topics found (all documents might be outliers)")
    
    # Guardar un resumen en texto
    with open("output_optimized/summary.txt", "w", encoding="utf-8") as f:
        f.write("TOPIC MODELING RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Total topics: {len(topic_info) - 1}\n")
        f.write(f"Total documents: {len(topics)}\n")
        f.write(f"Outliers: {(topics == -1).sum()}\n\n")
        
        f.write("ALL TOPICS:\n")
        f.write("-"*50 + "\n")
        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:
                f.write(f"Topic {row['Topic']}: {row['label']} (Count: {row['Count']})\n")
                f.write(f"Keywords: {', '.join(row['Representation'][:10])}\n\n")
    
    print(f"\nSummary saved to: output_optimized/summary.txt")
    
    return topic_model, topic_info, topic_results

# Ejecutar el programa
if __name__ == "__main__":
    topic_model, topic_info, results = main()
    