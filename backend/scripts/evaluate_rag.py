import json
import sys
import os
import numpy as np
import time
import argparse
from groq import Groq
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.embeddings import create_embeddings
from core.retriever import retrieve, format_context

load_dotenv()


groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── GROQ LLM CALL ─────────────────────────────────────────────
def eval_llm(prompt, max_tokens=512):
    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            time.sleep(2) 
            return response.choices[0].message.content
        except Exception as e:
            print(f"  Groq attempt {attempt+1} failed: {str(e)[:80]}")
            if "429" in str(e) and attempt < 2:
                time.sleep(10)
                continue
            return ""
    return ""


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
        "expected_chunk": "Tech Stack"
    },
    {
        "question": "Why did Anshu get into AI?",
        "expected_keywords": ["real world", "problems","fascinating "],
        "expected_chunk": "Motivation"
    },
    {
        "question": "What kind of roles is Anshu looking for?",
        "expected_keywords": ["backend", "AI"],
        "expected_chunk": "Goals"
    },
    {
        "question": "What was the hardest challenge in building ResumeIQ?",
        "expected_keywords": ["token"],
        "expected_chunk": "Challenges - Building ResumeIQ"
    },
    {
        "question": "What tools does ResumeIQ orchestrate?",
        "expected_keywords": ["Adzuna", "YouTube"],
        "expected_chunk": "Tool Orchestration"
    },
    {
        "question": "What is Anshu's educational background?",
        "expected_keywords": ["Truba", "Computer Science"],
        "expected_chunk": "Education"
    },
    {
        "question": "What programming languages does Anshu know?",
        "expected_keywords": ["Python", "C++"],
        "expected_chunk": "Skills - Programming"
    },
    {
        "question": "What is the ResumeIQ live link?",
        "expected_keywords": ["netlify"],
        "expected_chunk": "ResumeIQ"
    },
    {
        "question": "Which vector database powers Anshu's personal assistant?",
        "expected_keywords": ["ChromaDB"],
        "expected_chunk": "Tech Stack - Personal Assistant"
    },
    {
        "question": "What technologies are used in the backend of the personal assistant?",
        "expected_keywords": ["FastAPI"],
        "expected_chunk": "Tech Stack - Personal Assistant"
    },
    {
        "question": "What AI-related technologies has Anshu worked with?",
        "expected_keywords": ["LangGraph", "RAG"],
        "expected_chunk": "Skills - AI"
    },
    {
        "question": "Which company provided Anshu's internship opportunity?",
        "expected_keywords": ["Scoutbizz"],
        "expected_chunk": "Experience"
    },
    {
        "question": "Explain what ResumeIQ does.",
        "expected_keywords": ["resume", "ATS"],
        "expected_chunk": "Project - ResumeIQ (Overview)"
    },
    {
        "question": "What problem does the personal assistant solve?",
        "expected_keywords": ["recruiters", "questions"],
        "expected_chunk": "Project - Personal Assistant (Overview)"
    },
    {
        "question": "How does ResumeIQ help job seekers?",
        "expected_keywords": ["resume", "job"],
        "expected_chunk": "Project - ResumeIQ (Overview)"
    },
    {
        "question": "What is the purpose of integrating ResumeIQ into the personal assistant?",
        "expected_keywords": ["job description", "fit"],
        "expected_chunk": "Project - Personal Assistant (ResumeIQ Integration)"
    },
    {
        "question": "Which of Anshu's projects demonstrates agentic AI concepts?",
        "expected_keywords": ["ResumeIQ", "LangGraph"],
        "expected_chunk": ["Project - Personal Assistant (Overview)","Project - ResumeIQ (Overview)"]
    },
    {
        "question": "Which project best showcases Anshu's AI engineering skills?",
        "expected_keywords": ["ResumeIQ"],
        "expected_chunk": ["Project - Personal Assistant (Overview)","Project - ResumeIQ (Overview)"]
    },
    {
        "question": "How do Anshu's AI skills contribute to building ResumeIQ?",
        "expected_keywords": ["LangGraph", "RAG"],
        "expected_chunk": ["Skills - AI","Project - ResumeIQ (Overview)"]
    },
    {
        "question": "What skills did Anshu apply while developing the personal assistant?",
        "expected_keywords": ["FastAPI", "Gemini","rag"],
        "expected_chunk": ["Project - Personal Assistant (Architecture)","Personal Assistant (Overview)"]
    },
    {
        "question": "What is Anshu's current salary?",
        "expected_keywords": [],
        "expected_chunk": None
    },
    {
        "question": "Which FAANG company has Anshu worked for?",
        "expected_keywords": [],
        "expected_chunk": None
    },
    {
        "question": "What is Anshu's master's degree specialization?",
        "expected_keywords": [],
        "expected_chunk": None
    },
    {
        "question": "How many years of professional AI experience does Anshu have?",
        "expected_keywords": [],
        "expected_chunk": None
    },
    {
        "question": "Why should a company hire Anshu for an AI role?",
        "expected_keywords": ["AI", "projects"],
        "expected_chunk": "Goals"
    },
    {
        "question": "What makes ResumeIQ technically challenging?",
        "expected_keywords": ["token"],
        "expected_chunk": "Challenges - Building ResumeIQ"
    },
    {
        "question": "What evidence shows that Anshu has experience building AI applications?",
        "expected_keywords": ["ResumeIQ", "Personal Assistant"],
        "expected_chunk": ["Project - Personal Assistant (Overview)"," Project - ResumeIQ (Overview)"]
    },
    {
        "question": "How does Anshu demonstrate experience with retrieval augmented generation?",
        "expected_keywords": ["RAG"],
        "expected_chunk": "Skills - AI"
    }
]

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


# ── STEP 1: GENERATE ALL ANSWERS ONCE ────────────────────────
def generate_answers(test_cases):
    outputs = []
    print(f"\nGenerating {len(test_cases)} answers using Groq...")

    for i, test in enumerate(test_cases):
        print(f"  {i+1}/{len(test_cases)}: {test['question'][:50]}")

        embedding = create_embeddings(test["question"])
        chunks = retrieve(embedding, n_results=5)
        context = format_context(chunks)
        print(f"  Top chunk: {chunks[0][:80]}")

        prompt = f"""You are an AI assistant for Anshu, a CS graduate.
Answer the question based ONLY on the context provided.
Be concise and accurate.

Context:
{context}

Question: {test["question"]}

Answer:"""

        answer = eval_llm(prompt, max_tokens=300)

        outputs.append({
            "question": test["question"],
            "chunks": chunks,
            "context": context,
            "answer": answer
        })

    return outputs


# ── METRIC 1: RETRIEVAL PRECISION ────────────────────────────
def evaluate_retrieval(test_cases):
    print("\n[1/4] Retrieval Precision (no LLM calls)...")
    correct = 0
    evaluated = 0  
    results = []

    for test in test_cases:
        embedding = create_embeddings(test["question"])
        chunks = retrieve(embedding, n_results=5)

        if test["expected_chunk"] is None:
            continue

        evaluated += 1  
        expected = test["expected_chunk"]
        if isinstance(expected, str):
            expected = [expected]

        hit = any(
            any(exp.lower() in chunk.lower() for exp in expected)
            for chunk in chunks
        )

        correct += int(hit)
        status = "✅" if hit else "❌"

        if not hit:
            print(f"\n  ❌ Q: {test['question'][:50]}")
            print(f"     Expected any of: {expected}")
            print(f"     Retrieved chunks:")
            for i, chunk in enumerate(chunks):
                first_line = chunk.split('\n')[0][:70]
                print(f"       {i+1}. {first_line}")

        results.append({"question": test["question"], "hit": hit})

    score = correct / evaluated 
    print(f"  → Retrieval Precision@5: {score:.2%}")
    return score, results

# ── METRIC 2: ANSWER ACCURACY (LLM-as-Judge) ─────────────────
def evaluate_answer_accuracy(test_cases, outputs):
    print("\n[2/4] Answer Accuracy (LLM-as-judge)...")
    correct = 0
    results = []

    for i, (test, output) in enumerate(zip(test_cases, outputs)):
        print(f"  Judging {i+1}/{len(test_cases)}: {test['question'][:45]}")
        answer = output["answer"]

        if not test["expected_keywords"]:
            judge_prompt = f"""You are evaluating an AI assistant's response to an unanswerable question.
The question asks for information that does not exist in a personal portfolio (e.g. salary, FAANG experience, master's degree).
The assistant should respond by saying it doesn't know or that the information isn't available.

Question: {test["question"]}
Answer: {answer}

Did the assistant correctly decline or admit it doesn't have this information?
Respond with valid JSON only, no markdown:
{{"correct": true, "reason": "brief reason"}}"""

        else:
            expected = ", ".join(test["expected_keywords"])
            judge_prompt = f"""You are evaluating whether an AI assistant's answer correctly addresses a question.
The answer doesn't need to use the exact keywords — it just needs to convey the same meaning or concepts.

Question: {test["question"]}
Answer given: {answer}

Does the answer correctly cover the expected concepts, even if phrased differently?
A partial match counts as correct if the core idea is present.
Respond with valid JSON only, no markdown:
{{"correct": true, "reason": "brief reason"}}"""

        try:
            response = eval_llm(judge_prompt, max_tokens=150)
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()
            parsed = json.loads(clean)
            hit = bool(parsed.get("correct", False))
            reason = parsed.get("reason", "")
        except Exception as e:
            print(f"    Parse error: {e} | response: {response[:100]}")
            hit = False
            reason = "parse error"

        correct += int(hit)
        status = "✅" if hit else "❌"
        print(f"  {status} {test['question'][:45]}")
        if not hit:
            print(f"     reason: {reason}")
            print(f"     answer: {answer[:120]}")

        results.append({
            "question": test["question"],
            "hit": hit,
            "reason": reason
        })

    score = correct / len(test_cases)
    print(f"  → Answer Accuracy: {score:.2%}")
    return score, results


# ── METRIC 3: SEMANTIC SIMILARITY ────────────────────────────
def evaluate_semantic_similarity(test_cases, ground_truths, outputs):
    print("\n[3/4] Semantic Similarity (no LLM calls)...")
    scores = []

    for test, truth, output in zip(test_cases, ground_truths, outputs):
        answer = output["answer"]
        emb_answer = np.array(create_embeddings(answer))
        emb_truth = np.array(create_embeddings(truth))
        similarity = np.dot(emb_answer, emb_truth) / (
            np.linalg.norm(emb_answer) * np.linalg.norm(emb_truth) + 1e-9
        )
        scores.append(float(similarity))

    score = float(np.mean(scores))
    print(f"  → Avg Semantic Similarity: {score:.2%}")
    return score, scores


# ── METRIC 4: FAITHFULNESS ────────────────────────────────────
def evaluate_faithfulness(test_cases, outputs):
    test_cases = test_cases[-15:]
    outputs = outputs[-15:]
    print("\n[4/4] Faithfulness (LLM-as-judge via Groq)...")
    scores = []
    results = []

    for i, (test, output) in enumerate(zip(test_cases, outputs)):
        print(f"  Judging {i+1}/{len(test_cases)}: {test['question'][:45]}")

        judge_prompt = f"""You are evaluating if an AI answer is faithful to its source context.
Faithful means every factual claim in the answer is supported by the context.

Context:
{output['context'][:1500]}

Answer:
{output['answer']}

Task:
1. List the factual claims in the answer
2. Check each against the context
3. Calculate faithfulness = supported_claims / total_claims

Respond with valid JSON only, no markdown, no explanation:
{{"total_claims": 3, "supported_claims": 3, "faithfulness_score": 1.0}}"""

        try:
            response = eval_llm(judge_prompt, max_tokens=200)
            print(f"    Raw response: {response[:200]}")
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()
            scored = json.loads(clean)
            faith_score = float(scored.get("faithfulness_score", 0.0))
        except Exception as e:
            print(f"    Parse error: {e} | response: {response[:100]}")
            faith_score = 0.0

        status = "✅" if faith_score >= 0.8 else "❌"
        print(f"  {status} score: {faith_score:.2%}")

        scores.append(faith_score)
        results.append({
            "question": test["question"],
            "faithfulness_score": faith_score
        })

    score = float(np.mean(scores))
    print(f"  → Avg Faithfulness: {score:.2%}")
    return score, results

# ── METRIC 5: CONTEXT RECALL ─────────────────────────────────
def evaluate_context_recall(test_cases, ground_truths, outputs):
    test_cases = test_cases[-15:]
    ground_truths = ground_truths[-15:]
    outputs = outputs[-15:]
    print("\n[5/5] Context Recall (LLM-as-judge via Groq)...")
    scores = []
    results = []

    for i, (test, truth, output) in enumerate(zip(test_cases, ground_truths, outputs)):
        print(f"  Judging {i+1}/{len(test_cases)}: {test['question'][:45]}")

        judge_prompt = f"""You are evaluating whether the retrieved context contains 
all the information needed to answer a question correctly.

Ground Truth Answer:
{truth}

Retrieved Context:
{output['context'][:1500]}

Task:
1. Break the ground truth into individual statements
2. For each statement check if it can be found in the retrieved context
3. Calculate recall = statements found in context / total statements

Respond with valid JSON only, no markdown:
{{"total_statements": 3, "found_in_context": 3, "recall_score": 1.0}}"""

        try:
            response = eval_llm(judge_prompt, max_tokens=200)
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()
            scored = json.loads(clean)
            recall_score = float(scored.get("recall_score", 0.0))
        except Exception as e:
            print(f"    Parse error: {e}")
            recall_score = 0.0

        status = "✅" if recall_score >= 0.8 else "❌"
        print(f"  {status} score: {recall_score:.2%}")

        scores.append(recall_score)
        results.append({
            "question": test["question"],
            "recall_score": recall_score
        })

    score = float(np.mean(scores))
    print(f"  → Avg Context Recall: {score:.2%}")
    return score, results


# ── MAIN ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        choices=["retrieval", "accuracy", "similarity", "faithfulness","recall" "all"],
        default="all",
        help="Which metric to evaluate"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("RAG PIPELINE EVALUATION")
    print("=" * 55)

    retrieval_score = accuracy_score = similarity_score = faithfulness_score =recall_score = None

    
    if args.metric in ["retrieval"]:
        retrieval_score, _ = evaluate_retrieval(test_set)

   
    elif args.metric in ["accuracy", "similarity", "faithfulness", "all"]:
        outputs = generate_answers(test_set)

        if args.metric in ["retrieval", "all"]:
            retrieval_score, _ = evaluate_retrieval(test_set)

        if args.metric in ["accuracy", "all"]:
            accuracy_score, _ = evaluate_answer_accuracy(test_set, outputs)

        if args.metric in ["similarity", "all"]:
            similarity_score, _ = evaluate_semantic_similarity(
                test_set, ground_truths, outputs
            )
        if args.metric in ["faithfulness", "all"]:
            faithfulness_score, _ = evaluate_faithfulness(test_set, outputs)
        
        if args.metric in ["recall", "all"]:
            recall_score, _ = evaluate_context_recall(
                test_set, ground_truths, outputs
            )

    # summary
    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55) 
    if retrieval_score is not None:
        print(f"Retrieval Precision@5:  {retrieval_score:.2%}")
    if accuracy_score is not None:
        print(f"Answer Accuracy:        {accuracy_score:.2%}")
    if similarity_score is not None:
        print(f"Semantic Similarity:    {similarity_score:.2%}")
    if faithfulness_score is not None:
        print(f"Faithfulness:           {faithfulness_score:.2%}")
    if recall_score is not None:
       print(f"Context Recall:          {recall_score:.2%}")
    
    print(f"Test queries:           {len(test_set)}")
    print("=" * 55)

    # save
    results = {k: v for k, v in {
        "retrieval_precision": round(retrieval_score, 4) if retrieval_score else None,
        "answer_accuracy": round(accuracy_score, 4) if accuracy_score else None,
        "semantic_similarity": round(similarity_score, 4) if similarity_score else None,
        "faithfulness": round(faithfulness_score, 4) if faithfulness_score else None,
        "num_queries": len(test_set),
        "context_recall": round(recall_score, 4) if recall_score else None,
    }.items() if v is not None}

    output_path = os.path.join(BASE_DIR, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to eval_results.json")

    if faithfulness_score and accuracy_score:
        print(f"\nResume bullet:")
        print(f'"Evaluated RAG pipeline achieving {faithfulness_score:.0%} faithfulness')
        print(f'and {accuracy_score:.0%} answer accuracy across {len(test_set)}')
        print(f'test queries using custom LLM-as-judge evaluation framework"')


if __name__ == "__main__":
    main()