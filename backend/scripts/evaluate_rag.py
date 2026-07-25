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
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            time.sleep(4)  # increase to 4 seconds
            return response.choices[0].message.content
        except Exception as e:
            print(f"  Groq attempt {attempt+1} failed: {str(e)[:80]}")
            if "429" in str(e) and attempt < 2:
                time.sleep(60)  # wait full minute on 429
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
        "expected_chunk": ["Project - ResumeIQ (Overview)",
                          "Project - ResumeIQ (Architecture and Workflow)"]
    },
    {
        "question": "Where is Anshu currently based?",
        "expected_keywords": ["Pune"],
        "expected_chunk": "Contact and Availability"
    },
    {
        "question": "What are Anshu's AI skills?",
        "expected_keywords": ["LangGraph", "RAG", "Groq"],
        "expected_chunk": "Skills - AI"
    },
    {
        "question": "How many LLM calls does ResumeIQ make?",
        "expected_keywords": ["3"],
        "expected_chunk": "Project - ResumeIQ (Architecture and Workflow)"
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
        "expected_chunk": "Tech Stack - Personal Assistant"
    },
    {
        "question": "Why did Anshu get into AI?",
        "expected_keywords": ["real world", "problems"],
        "expected_chunk": "Motivation - Why Anshu Chose AI"
    },
    {
        "question": "What kind of roles is Anshu looking for?",
        "expected_keywords": ["backend", "AI"],
        "expected_chunk": "Goals - Roles Anshu is Looking For"
    },
    {
        "question": "What was the hardest challenge in building ResumeIQ?",
        "expected_keywords": ["token"],
        "expected_chunk": "Project Challenges - Building ResumeIQ"
    },
    {
        "question": "What tools does ResumeIQ orchestrate?",
        "expected_keywords": ["Adzuna", "YouTube"],
        "expected_chunk": "Project - ResumeIQ (Tool Orchestration)"
    },
    {
        "question": "What is Anshu's educational background?",
        "expected_keywords": ["Truba", "Computer Science"],
        "expected_chunk": "Education"
    },
    {
        "question": "What programming languages does Anshu know?",
        "expected_keywords": ["Python", "C++"],
        "expected_chunk": "Skills - Programming and Core CS"
    },
    {
        "question": "What is the ResumeIQ live link?",
        "expected_keywords": ["netlify"],
        "expected_chunk": "Project - ResumeIQ (Overview)"
    },
    {
        "question": "Which vector database powers Anshu's personal assistant?",
        "expected_keywords": ["ChromaDB"],
        "expected_chunk": "Tech Stack - Personal Assistant"
    },
    {
        "question": "What technologies are used in the backend of the personal assistant?",
        "expected_keywords": ["FastAPI"],
        "expected_chunk": ["Tech Stack - Personal Assistant",
                          "Project - Personal Assistant (Architecture)"]
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
        "expected_chunk": ["Introduction",
                          "Project - Personal Assistant (Overview)",
                          "Project - ResumeIQ (Overview)"]
    },
    {
        "question": "Which project best showcases Anshu's AI engineering skills?",
        "expected_keywords": ["ResumeIQ"],
        "expected_chunk": ["Project - Personal Assistant (Overview)",
                          "Project - ResumeIQ (Overview)"]
    },
    {
        "question": "How do Anshu's AI skills contribute to building ResumeIQ?",
        "expected_keywords": ["LangGraph", "RAG"],
        "expected_chunk": ["Skills - AI",
                          "Project - ResumeIQ (Overview)"]
    },
    {
        "question": "What skills did Anshu apply while developing the personal assistant?",
        "expected_keywords": ["FastAPI", "Gemini"],
        "expected_chunk": ["Project - Personal Assistant (Architecture)",
                          "Project - Personal Assistant (Overview)"]
    },
    {
        "question": "Why should a company hire Anshu for an AI role?",
        "expected_keywords": ["AI", "projects"],
        "expected_chunk": ["Goals - Roles Anshu is Looking For",
                          "Motivation - Why Anshu Chose AI"]
    },
    {
        "question": "What makes ResumeIQ technically challenging?",
        "expected_keywords": ["token"],
        "expected_chunk": ["Project Challenges - Building ResumeIQ",
                          "Project - ResumeIQ (Architecture and Workflow)",
                          "Project - ResumeIQ (Token Management and Chat)"]
    },
    {
        "question": "What evidence shows that Anshu has experience building AI applications?",
        "expected_keywords": ["ResumeIQ", "Personal Assistant"],
        "expected_chunk": ["Introduction",
                          "Project - Personal Assistant (Overview)",
                          "Project - ResumeIQ (Overview)"]
    },
    {
        "question": "How does Anshu demonstrate experience with retrieval augmented generation?",
        "expected_keywords": ["RAG"],
        "expected_chunk": ["Skills - AI",
                          "Project - Personal Assistant (Architecture)"]
    }
]

ground_truths = [
    # 1
    "Anshu has a CGPA of 7.99",
    # 2
    "Anshu used LangGraph to build ResumeIQ",
    # 3
    "Anshu is currently based in Pune India",
    # 4
    "Anshu's AI skills include LangGraph RAG Groq API and Prompt Engineering",
    # 5
    "ResumeIQ makes 3 LLM calls per full analysis",
    # 6
    "Anshu holds Microsoft Certified Azure Fundamentals certification",
    # 7
    "Anshu completed an internship at Scoutbizz International",
    # 8
    "The personal assistant is built with FastAPI ChromaDB sentence transformers and Gemini Flash",
    # 9
    "Anshu got into AI because she finds it fascinating how it solves real world problems",
    # 10
    "Anshu is looking for backend and AI engineering roles",
    # 11
    "The hardest challenge in building ResumeIQ was managing token limits across the pipeline",
    # 12
    "ResumeIQ orchestrates ATS keyword checker Adzuna API and YouTube Data API",
    # 13
    "Anshu completed BTech in Computer Science from Truba Institute with CGPA 7.99",
    # 14
    "Anshu knows Python and C++ as programming languages",
    # 15
    "ResumeIQ is live at agentic-resume-analyzer.netlify.app",
    # 16
    "ChromaDB is the vector database powering Anshu's personal assistant",
    # 17
    "The personal assistant backend uses FastAPI as the async web framework",
    # 18
    "Anshu has worked with LangGraph for agentic pipelines and RAG for retrieval systems",
    # 19
    "Scoutbizz International provided Anshu's internship opportunity",
    # 20
    "ResumeIQ analyzes resumes against job descriptions providing ATS score and skill gap analysis",
    # 21
    "The personal assistant solves the problem of answering recruiter questions about Anshu interactively",
    # 22
    "ResumeIQ helps job seekers by analyzing their resume against a job description and identifying skill gaps",
    # 23
    "ResumeIQ is integrated into the personal assistant so recruiters can paste a JD and get a live fit analysis",
    # 24
    "ResumeIQ demonstrates agentic AI through its LangGraph pipeline with autonomous tool calling decisions",
    # 25
    "ResumeIQ best showcases AI engineering skills through its stateful agentic LangGraph pipeline",
    # 26
    "Anshu's LangGraph and RAG skills directly contributed to building ResumeIQ's agentic pipeline",
    # 27
    "Anshu applied FastAPI Gemini and RAG skills while developing the personal assistant",
    # 28
    "Anshu's current salary is not mentioned as she is a fresher looking for her first role",
    # 29
    "Anshu has not worked for any FAANG company as she is a fresher",
    # 30
    "Anshu does not have a master's degree she completed BTech in Computer Science",
    # 31
    "Anshu is a fresher with no professional AI experience but has built two AI projects",
    # 32
    "A company should hire Anshu for her hands-on AI projects ResumeIQ and personal assistant",
    # 33
    "ResumeIQ is technically challenging due to token management across multiple LLM calls and tool orchestration",
    # 34
    "Anshu has experience building AI applications through ResumeIQ and the Personal Assistant projects",
    # 35
    "Anshu demonstrates RAG experience through her personal assistant which uses ChromaDB and semantic search"
]


# ── STEP 1: GENERATE ALL ANSWERS ONCE ────────────────────────
def generate_answers(test_cases):
    answers_path = os.path.join(BASE_DIR, "answers_cache.json")

    if os.path.exists(answers_path):
        print("\nLoading cached answers (delete answers_cache.json to regenerate)")
        with open(answers_path, "r") as f:
            return json.load(f)

   
    from core.llm import llm_response

    outputs = []
    print(f"\nGenerating {len(test_cases)} answers using actual Gemini pipeline...")

    for i, test in enumerate(test_cases):
        print(f"  {i+1}/{len(test_cases)}: {test['question'][:50]}")

        embedding = create_embeddings(test["question"])
        chunks = retrieve(
            embedding,
            n_results=8,              
            query_text=test["question"]  
        )
        context = format_context(chunks)

        answer = llm_response(test["question"], context, memory={})

        outputs.append({
            "question": test["question"],
            "chunks": chunks,
            "context": context,
            "answer": answer
        })

        time.sleep(3)  

    with open(answers_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print("Answers cached to answers_cache.json")
    return outputs 


# ── METRIC 1: RETRIEVAL PRECISION ────────────────────────────


def is_chunk_match(expected, chunk):
    first_line = chunk.split('\n')[0].strip().lower()
    expected_lower = expected.strip().lower()
    
    
    return first_line == expected_lower

def evaluate_retrieval(test_cases):
    print("\n[1/4] Retrieval Precision (no LLM calls)...")
    correct = 0
    results = []

    for test in test_cases:
        embedding = create_embeddings(test["question"])
        chunks = retrieve(embedding, n_results=5, query_text=test["question"])

        expected = test["expected_chunk"]

        if expected is None:
            hit = True

        elif isinstance(expected, list):
            hit = any(
                any(is_chunk_match(exp, chunk) for chunk in chunks)
                for exp in expected
            )
        else:
            hit = any(is_chunk_match(expected, chunk) for chunk in chunks)

        correct += int(hit)
        status = "✅" if hit else "❌"
        print(f"  {status} [{expected}] {test['question'][:45]}")
        print("  Retrieved chunks:")
        for i, chunk in enumerate(chunks, start=1):
            first_line = chunk.split("\n")[0][:70]
            print(f"    {i}. {first_line}")

        results.append({"question": test["question"], "hit": hit})

    score = correct / len(test_cases)
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
    print("\n[4/4] Faithfulness (LLM-as-judge via Groq)...")
    scores = []
    factual_scores = []
    opinion_scores = []
    results = []

    opinion_keywords = [
        "should", "best", "showcases", "demonstrates",
        "contributes", "evidence", "why should", "worth",
        "recommend", "suitable", "fit for"
    ]

    for i, (test, output) in enumerate(zip(test_cases, outputs)):
        print(f"  Judging {i+1}/{len(test_cases)}: {test['question'][:45]}")

        # classify question type
        is_opinion = any(
            kw in test["question"].lower()
            for kw in opinion_keywords
        )
        q_type = "opinion" if is_opinion else "factual"

        judge_prompt = f"""You are evaluating if an AI answer is faithful to its source context.

Definition of faithful:
- DIRECT: claim is explicitly stated in context ✅
- INFERRED: claim is a reasonable conclusion from context facts ✅  
- HALLUCINATED: claim cannot be supported or inferred from context ❌

Context:
{output['context'][:3000]}

Answer:
{output['answer']}

Question: {test['question']}

Instructions:
1. List each factual claim in the answer
2. For each claim decide: DIRECT, INFERRED, or HALLUCINATED
3. faithful_claims = DIRECT + INFERRED
4. faithfulness_score = faithful_claims / total_claims

Example:
Context says "Anshu built ResumeIQ using LangGraph"
Answer says "Anshu has experience with agentic frameworks"
→ INFERRED ✅ (LangGraph is an agentic framework, reasonable inference)

Respond with valid JSON only, no markdown:
{{"total_claims": 3, "faithful_claims": 3, "faithfulness_score": 1.0}}"""

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
        print(f"  {status} [{q_type}] score: {faith_score:.2%}")

        scores.append(faith_score)
        if is_opinion:
            opinion_scores.append(faith_score)
        else:
            factual_scores.append(faith_score)

        results.append({
            "question": test["question"],
            "question_type": q_type,
            "faithfulness_score": faith_score
        })

    # print breakdown
    overall = float(np.mean(scores))
    factual = float(np.mean(factual_scores)) if factual_scores else 0.0
    opinion = float(np.mean(opinion_scores)) if opinion_scores else 0.0

    print(f"\n  → Faithfulness breakdown:")
    print(f"     Overall:  {overall:.2%} ({len(scores)} questions)")
    print(f"     Factual:  {factual:.2%} ({len(factual_scores)} questions)")
    print(f"     Opinion:  {opinion:.2%} ({len(opinion_scores)} questions)")

    return overall, results

# ── METRIC 5: CONTEXT RECALL ─────────────────────────────────
def evaluate_context_recall(test_cases, ground_truths, outputs):
    # test_cases = test_cases[:15]
    # ground_truths = ground_truths[:15]
    # outputs = outputs[:15]
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

# ── METRIC 6: ANSWER RELEVANCY ────────────────────────────────
def evaluate_answer_relevancy(test_cases, outputs):
    print("\n[6/6] Answer Relevancy (LLM-as-judge via Groq)...")
    scores = []
    results = []

    for i, (test, output) in enumerate(zip(test_cases, outputs)):
        print(f"  Judging {i+1}/{len(test_cases)}: {test['question'][:45]}")

        judge_prompt = f"""You are evaluating whether an AI assistant's answer 
is relevant to the question asked.

Answer relevancy measures:
1. Does the answer actually address the question?
2. Is the answer focused or does it go off topic?
3. Does the answer answer what was asked, not something adjacent?

Question: {test['question']}
Answer: {output['answer']}

Score the answer relevancy from 0.0 to 1.0:
- 1.0: Answer directly and completely addresses the question
- 0.7: Answer mostly addresses question with minor irrelevant parts
- 0.5: Answer partially addresses question, significant off-topic content
- 0.3: Answer barely addresses question
- 0.0: Answer is completely irrelevant or refuses without reason

Note: An answer saying "I don't have that information" for a question
whose answer is genuinely not available is RELEVANT (score: 0.9+)
because it correctly addresses an unanswerable question.

Respond with valid JSON only, no markdown:
{{"relevancy_score": 0.9, "reason": "brief explanation"}}"""

        try:
            response = eval_llm(judge_prompt, max_tokens=150)
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()
            parsed = json.loads(clean)
            relevancy_score = float(parsed.get("relevancy_score", 0.0))
            reason = parsed.get("reason", "")
        except Exception as e:
            print(f"    Parse error: {e} | response: {response[:100]}")
            relevancy_score = 0.0
            reason = "parse error"

        status = "✅" if relevancy_score >= 0.7 else "❌"
        print(f"  {status} score: {relevancy_score:.2%} — {reason[:60]}")

        scores.append(relevancy_score)
        results.append({
            "question": test["question"],
            "relevancy_score": relevancy_score,
            "reason": reason
        })

    score = float(np.mean(scores))
    print(f"  → Avg Answer Relevancy: {score:.2%}")
    return score, results


# ── MAIN ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        choices=["retrieval", "accuracy", "similarity", "faithfulness","recall","all", "relevancy"],
        default="all",
        help="Which metric to evaluate"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("RAG PIPELINE EVALUATION")
    print("=" * 55)

    retrieval_score = accuracy_score = similarity_score = faithfulness_score =recall_score = relevancy_score = None

    
    if args.metric in ["retrieval"]:
        retrieval_score, _ = evaluate_retrieval(test_set)

   
    # elif args.metric in ["accuracy", "similarity", "faithfulness","recall", "all"]:
    else:
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
        if args.metric in ["relevancy", "all"]:
            relevancy_score, _ = evaluate_answer_relevancy(test_set, outputs)

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
    if relevancy_score is not None:
        print(f"Answer Relevancy:       {relevancy_score:.2%}")
    
    
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
        "answer_relevancy": round(relevancy_score, 4) if relevancy_score else None,
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