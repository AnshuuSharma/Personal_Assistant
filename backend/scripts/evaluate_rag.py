import json
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.embeddings import create_embeddings
from core.retriever import retrieve, format_context
from core.llm import llm_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── TEST SET ──────────────────────────────────────────────────
test_set = [
    {
        "question": "What is Anshu's CGPA?",
        "expected_keywords": ["7.99"],
        "expected_chunk": "Education"
    },
    {
        "question": "What framework did Anshu use to build ResumeIQ?",
        "expected_keywords": ["LangGraph"],
        "expected_chunk": "ResumeIQ"
    },
    {
        "question": "Where is Anshu currently based?",
        "expected_keywords": ["Pune"],
        "expected_chunk": "Contact"
    },
    {
        "question": "What are Anshu's AI skills?",
        "expected_keywords": ["LangGraph", "RAG", "Groq"],
        "expected_chunk": "Skills - AI"
    },
    {
        "question": "How many LLM calls does ResumeIQ make?",
        "expected_keywords": ["3"],
        "expected_chunk": "ResumeIQ"
    },
    {
        "question": "What certifications does Anshu have?",
        "expected_keywords": ["Azure", "Microsoft"],
        "expected_chunk": "Certifications"
    },
    {
        "question": "What internship did Anshu complete?",
        "expected_keywords": ["Scoutbizz"],
        "expected_chunk": "Experience"
    },
    {
        "question": "What is the personal assistant built with?",
        "expected_keywords": ["FastAPI", "ChromaDB", "Gemini"],
        "expected_chunk": "Personal Assistant"
    },
    {
        "question": "Why did Anshu get into AI?",
        "expected_keywords": ["real world", "problems", "fascinating"],
        "expected_chunk": "Why AI"
    },
    {
        "question": "What kind of roles is Anshu looking for?",
        "expected_keywords": ["backend", "AI"],
        "expected_chunk": "Goals"
    },
    {
        "question": "What was the hardest challenge in building ResumeIQ?",
        "expected_keywords": ["token", "limit"],
        "expected_chunk": "Challenges"
    },
    {
        "question": "What tools does ResumeIQ orchestrate?",
        "expected_keywords": ["Adzuna", "YouTube", "ATS"],
        "expected_chunk": "Tool Orchestration"
    },
    {
        "question": "What is Anshu's educational background?",
        "expected_keywords": ["Truba", "Computer Science", "BTech"],
        "expected_chunk": "Education"
    },
    {
        "question": "What programming languages does Anshu know?",
        "expected_keywords": ["Python", "C++"],
        "expected_chunk": "Skills - Programming"
    },
    {
        "question": "What is the ResumeIQ live link?",
        "expected_keywords": ["netlify", "agentic-resume"],
        "expected_chunk": "ResumeIQ"
    }
]


# ── METRIC 1: RETRIEVAL PRECISION ────────────────────────────
def evaluate_retrieval(test_cases):
    correct = 0
    results = []

    for test in test_cases:
        embedding = create_embeddings(test["question"])
        chunks = retrieve(embedding, n_results=5)

        hit = any(
            test["expected_chunk"].lower() in chunk.lower()
            for chunk in chunks
        )

        correct += int(hit)
        results.append({
            "question": test["question"],
            "expected_chunk": test["expected_chunk"],
            "hit": hit,
            "retrieved_chunks": [c[:60] for c in chunks]
        })

    precision = correct / len(test_cases)
    return precision, results


# ── METRIC 2: ANSWER ACCURACY (keyword based) ────────────────
def evaluate_answer_accuracy(test_cases):
    correct = 0
    results = []

    for test in test_cases:
        embedding = create_embeddings(test["question"])
        chunks = retrieve(embedding, n_results=5)
        context = format_context(chunks)
        answer = llm_response(test["question"], context, memory={})

        keywords_found = [
            kw for kw in test["expected_keywords"]
            if kw.lower() in answer.lower()
        ]
        hit = len(keywords_found) == len(test["expected_keywords"])

        correct += int(hit)
        results.append({
            "question": test["question"],
            "expected_keywords": test["expected_keywords"],
            "keywords_found": keywords_found,
            "hit": hit,
            "answer_preview": answer[:100]
        })

    accuracy = correct / len(test_cases)
    return accuracy, results


# ── METRIC 3: SEMANTIC SIMILARITY ────────────────────────────
ground_truths = [
    "Anshu has a CGPA of 7.99",
    "Anshu used LangGraph to build ResumeIQ",
    "Anshu is based in Pune India",
    "Anshu has experience in LangGraph RAG Groq API and Prompt Engineering",
    "ResumeIQ makes 3 LLM calls per analysis",
    "Anshu holds Microsoft Certified Azure Fundamentals certification",
    "Anshu completed internship at Scoutbizz International",
    "Personal assistant uses FastAPI ChromaDB sentence transformers and Gemini Flash",
    "Anshu was fascinated by how AI solves real world problems",
    "Anshu is looking for backend and AI engineering roles",
    "The hardest challenge was managing token usage across the pipeline",
    "ResumeIQ orchestrates ATS checker Adzuna API and YouTube API",
    "Anshu completed BTech from Truba Institute in Computer Science",
    "Anshu knows Python and C++",
    "ResumeIQ is live at agentic-resume-analyzer.netlify.app"
]

def evaluate_semantic_similarity(test_cases, ground_truths):
    scores = []

    for test, truth in zip(test_cases, ground_truths):
        embedding = create_embeddings(test["question"])
        chunks = retrieve(embedding, n_results=5)
        context = format_context(chunks)
        answer = llm_response(test["question"], context, memory={})

        emb_answer = np.array(create_embeddings(answer))
        emb_truth = np.array(create_embeddings(truth))

        similarity = np.dot(emb_answer, emb_truth) / (
            np.linalg.norm(emb_answer) * np.linalg.norm(emb_truth) + 1e-9
        )
        scores.append(float(similarity))

    avg_similarity = np.mean(scores)
    return avg_similarity, scores


# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("="*55)
    print("RAG PIPELINE EVALUATION")
    print("="*55)

    print("\n[1/3] Evaluating retrieval precision...")
    retrieval_score, retrieval_results = evaluate_retrieval(test_set)
    print(f"Retrieval Precision@5: {retrieval_score:.2%}")

    for r in retrieval_results:
        status = "✅" if r["hit"] else "❌"
        print(f"  {status} {r['question'][:50]}")

    
    print("\n[2/3] Evaluating answer accuracy...")
    accuracy_score, accuracy_results = evaluate_answer_accuracy(test_set)
    print(f"Answer Accuracy: {accuracy_score:.2%}")

    for r in accuracy_results:
        status = "✅" if r["hit"] else "❌"
        print(f"  {status} {r['question'][:50]}")
        if not r["hit"]:
            missing = set(r["expected_keywords"]) - set(r["keywords_found"])
            print(f"     missing keywords: {missing}")

   
    print("\n[3/3] Evaluating semantic similarity...")
    similarity_score, sim_scores = evaluate_semantic_similarity(
        test_set, ground_truths
    )
    print(f"Avg Semantic Similarity: {similarity_score:.2%}")

    
    print("\n" + "="*55)
    print("FINAL RESULTS")
    print("="*55)
    print(f"Retrieval Precision@5:   {retrieval_score:.2%}")
    print(f"Answer Accuracy:         {accuracy_score:.2%}")
    print(f"Semantic Similarity:     {similarity_score:.2%}")
    print(f"Test queries:            {len(test_set)}")
    print("="*55)

    
    results = {
        "retrieval_precision": round(retrieval_score, 4),
        "answer_accuracy": round(accuracy_score, 4),
        "semantic_similarity": round(similarity_score, 4),
        "num_queries": len(test_set),
        "detailed_retrieval": retrieval_results,
        "detailed_accuracy": accuracy_results
    }

    output_path = os.path.join(BASE_DIR, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to eval_results.json")
    print("\nResume bullet point:")
    print(f"\"Evaluated RAG pipeline achieving {retrieval_score:.0%} retrieval")
    print(f"precision, {accuracy_score:.0%} answer accuracy and")
    print(f"{similarity_score:.0%} semantic similarity across {len(test_set)} test queries\"")


if __name__ == "__main__":
    main()