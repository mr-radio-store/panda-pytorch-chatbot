import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # RPi headless-safe
import matplotlib.pyplot as plt

# ======================================================
# 1. Load FAQ dataset
# ======================================================
def load_faq_data(csv_path=None):
    """
    Load FAQs from CSV or create a small default dataset.
    CSV columns: ['question','answer']
    """
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        data = {
            "question": [
                "What is AI?",
                "What is PyTorch?",
                "How do I use pandas?",
                "What is machine learning?",
                "What is deep learning?"
            ],
            "answer": [
                "AI is the simulation of human intelligence in machines.",
                "PyTorch is an open-source deep learning framework.",
                "Pandas is a Python library for data manipulation.",
                "Machine learning automates analytical model building.",
                "Deep learning is a subset of ML using multi-layer neural networks."
            ]
        }
        df = pd.DataFrame(data)
    return df

# ======================================================
# 2. Simple encoder
# ======================================================
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=50):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.linear(x)

# ======================================================
# 3. TF-IDF features
# ======================================================
def prepare_features(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["question"]).toarray()
    return X, vectorizer

# ======================================================
# 4. Train encoder
# ======================================================
def train_encoder(X, epochs=100):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    encoder = SimpleEncoder(input_dim=X.shape[1], embedding_dim=50)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = encoder(X_tensor)
        loss = ((embeddings - embeddings.mean(dim=0))**2).mean()  # variance maximization
        loss.backward()
        optimizer.step()
    return encoder

# ======================================================
# 5. Query function
# ======================================================
def answer_question(query, df, vectorizer, encoder):
    q_vec = vectorizer.transform([query]).toarray()
    q_tensor = torch.tensor(q_vec, dtype=torch.float32)
    q_emb = encoder(q_tensor)

    X_tensor = torch.tensor(vectorizer.transform(df["question"]).toarray(), dtype=torch.float32)
    X_emb = encoder(X_tensor)

    similarity = F.cosine_similarity(q_emb, X_emb)
    idx = torch.argmax(similarity).item()
    return df.iloc[idx]["answer"], similarity[idx].item(), similarity.detach().numpy()

# ======================================================
# 6. Save results as figure
# ======================================================
def save_similarity_figure(sim_scores, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(8,4))
    plt.bar(range(len(sim_scores)), sim_scores, color="skyblue")
    plt.xlabel("FAQ Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Query Similarity Scores")
    plt.tight_layout()
    plt.savefig(out_dir / f"similarity_{timestamp}.png", dpi=150)
    plt.close()
    print(f"[OK] Similarity figure saved to '{out_dir}'")

# ======================================================
# 7. Save textual report
# ======================================================
def save_report(query, answer, score, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"QnA_report_{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write("=== Q&A SESSION ===\n")
        f.write(f"Query: {query}\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"Similarity Score: {score:.2f}\n")
    print(f"[OK] Report saved to '{report_path}'")

# ======================================================
# 8. Main chatbot loop
# ======================================================
def main():
    df = load_faq_data()
    X, vectorizer = prepare_features(df)
    encoder = train_encoder(X)

    print("Dynamic Q&A Chatbot (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        answer, score, sim_scores = answer_question(query, df, vectorizer, encoder)
        print(f"Bot: {answer} (score={score:.2f})")

        # Save figure & report for RPi5
        save_similarity_figure(sim_scores)
        save_report(query, answer, score)

if __name__ == "__main__":
    main()
