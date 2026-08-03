from core.embeddings import create_embeddings
from core.retriever import collection

RELEVANT_QUERIES = [
    "what projects has Anshu worked on",
    "what is Anshu's experience with RAG systems",
    "what certifications does Anshu have",
    "where did Anshu study",
    "what programming languages does Anshu know",
]

IRRELEVANT_QUERIES = [
    "what's the weather like today",
    "recommend a good pizza place",
    "who won the last world cup",
    "how do I fix my car engine",
    "tell me a joke about cats",
]


def top_dense_similarity(query):
    embedding = create_embeddings(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["distances"]
    )
    dist = results["distances"][0][0]
    return 1 - dist  # valid only if collection space is "cosine" -- verify this


print(f"{'Query':<45} {'Similarity':>10}  Label")
print("-" * 68)

relevant_scores, irrelevant_scores = [], []

for q in RELEVANT_QUERIES:
    s = top_dense_similarity(q)
    relevant_scores.append(s)
    print(f"{q:<45} {s:>10.4f}  relevant")

for q in IRRELEVANT_QUERIES:
    s = top_dense_similarity(q)
    irrelevant_scores.append(s)
    print(f"{q:<45} {s:>10.4f}  irrelevant")

print("\n" + "=" * 68)
min_relevant = min(relevant_scores)
max_irrelevant = max(irrelevant_scores)

print(f"lowest score among RELEVANT queries:    {min_relevant:.4f}")
print(f"highest score among IRRELEVANT queries: {max_irrelevant:.4f}")

if min_relevant > max_irrelevant:
    suggested = (min_relevant + max_irrelevant) / 2
    print(f"\nClean gap found. Suggested DENSE_THRESHOLD ~= {suggested:.4f}")
else:
    print(f"\nWARNING: groups overlap -- no clean separating threshold.")
    print(f"Add more test queries covering edge cases, or accept some error rate.")
    print(f"As a starting point you could still try something between")
    print(f"{max_irrelevant:.4f} and {min_relevant:.4f}, but expect misclassifications near it.")