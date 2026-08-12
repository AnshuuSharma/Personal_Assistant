import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.embeddings import create_embeddings
from core.retriever import collection

RELEVANT_QUERIES = [
    # direct / clear-cut
    "what projects has Anshu worked on",
    "what is Anshu's experience with RAG systems",
    "what certifications does Anshu have",
    "where did Anshu study",
    "what programming languages does Anshu know",
    "tell me about ResumeIQ",
    "what tech stack did Anshu use for her resume analyzer",
    "does Anshu have experience with LangGraph",
    "what is Anshu's GitHub",
    "what internship experience does Anshu have",

    # borderline / paraphrased, softer skill-type questions
    "does Anshu have leadership experience",
    "is Anshu good at working in a team",
    "what are Anshu's strengths as an engineer",
    "how does Anshu approach debugging or problem solving",
    "has Anshu deployed any production systems",
    "does Anshu have experience with evaluation frameworks for LLMs",
    "what does Anshu know about hybrid search",
    "is Anshu familiar with agentic AI systems",
    "what's Anshu's availability for a role",
    "why should we hire Anshu",
]

IRRELEVANT_QUERIES = [
    "what's the weather like today",
    "recommend a good pizza place",
    "who won the last world cup",
    "how do I fix my car engine",
    "tell me a joke about cats",
    "what's the capital of France",
    "how do I invest in stocks",
    "write me a poem about the ocean",
    "what's a good workout routine",
    "translate hello into Spanish",
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


print(f"{'Query':<55} {'Similarity':>10}  Label")
print("-" * 78)

relevant_scores, irrelevant_scores = [], []

for q in RELEVANT_QUERIES:
    s = top_dense_similarity(q)
    relevant_scores.append((q, s))
    print(f"{q:<55} {s:>10.4f}  relevant")

for q in IRRELEVANT_QUERIES:
    s = top_dense_similarity(q)
    irrelevant_scores.append((q, s))
    print(f"{q:<55} {s:>10.4f}  irrelevant")

print("\n" + "=" * 78)

rel_vals = [s for _, s in relevant_scores]
irr_vals = [s for _, s in irrelevant_scores]

min_relevant = min(rel_vals)
max_irrelevant = max(irr_vals)

print(f"lowest score among RELEVANT queries:    {min_relevant:.4f}  "
      f"({[q for q, s in relevant_scores if s == min_relevant][0]})")
print(f"highest score among IRRELEVANT queries: {max_irrelevant:.4f}  "
      f"({[q for q, s in irrelevant_scores if s == max_irrelevant][0]})")

if min_relevant > max_irrelevant:
    suggested = (min_relevant + max_irrelevant) / 2
    print(f"\nClean gap found. Midpoint DENSE_THRESHOLD ~= {suggested:.4f}")
    print(f"Consider setting it closer to {max_irrelevant + 0.02:.4f} "
          f"(near the irrelevant ceiling) to reduce false rejections of real questions.")
else:
    print(f"\nWARNING: groups overlap -- no clean separating threshold.")
    print(f"Look at which specific queries overlap (printed above) to see if it's")
    print(f"a phrasing issue (fixable via corpus wording) or a genuine ambiguity.")