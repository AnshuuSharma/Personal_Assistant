
import sys
import os
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from core.embeddings import create_embeddings
from core.retriever import retrieve, format_context, retrieve_with_rerank
from core.llm import llm_response
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def eval_llm(prompt, max_tokens=512):
    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            time.sleep(3)
            return response.choices[0].message.content
        except Exception as e:
            print(f"  Groq attempt {attempt+1} failed: {str(e)[:80]}")
            if "429" in str(e) and attempt < 2:
                time.sleep(30)
                continue
            return ""
    return ""


test_set = [
    {"question": "What is Anshu's CGPA?",
     "ground_truth": "Anshu has a CGPA of 7.99"},
    {"question": "What framework did Anshu use to build ResumeIQ?",
     "ground_truth": "Anshu used LangGraph to build ResumeIQ"},
    {"question": "Where is Anshu currently based?",
     "ground_truth": "Anshu is currently based in Pune India"},
    {"question": "What are Anshu's AI skills?",
     "ground_truth": "Anshu AI skills include LangGraph RAG Groq API and Prompt Engineering"},
    {"question": "How many LLM calls does ResumeIQ make?",
     "ground_truth": "ResumeIQ makes 3 LLM calls per full analysis"},
    {"question": "What certifications does Anshu have?",
     "ground_truth": "Anshu holds Microsoft Certified Azure Fundamentals certification"},
    {"question": "What internship did Anshu complete?",
     "ground_truth": "Anshu completed an internship at Scoutbizz International"},
    {"question": "What is the personal assistant built with?",
     "ground_truth": "The personal assistant is built with FastAPI ChromaDB sentence transformers and Gemini Flash"},
    {"question": "Why did Anshu get into AI?",
     "ground_truth": "Anshu got into AI because she finds it fascinating how it solves real world problems"},
    {"question": "What kind of roles is Anshu looking for?",
     "ground_truth": "Anshu is looking for backend and AI engineering roles"},
    {"question": "What was the hardest challenge in building ResumeIQ?",
     "ground_truth": "The hardest challenge was managing token limits across the pipeline"},
    {"question": "What tools does ResumeIQ orchestrate?",
     "ground_truth": "ResumeIQ orchestrates ATS keyword checker Adzuna API and YouTube Data API"},
    {"question": "What is Anshu's educational background?",
     "ground_truth": "Anshu completed BTech in Computer Science from Truba Institute with CGPA 7.99"},
    {"question": "What programming languages does Anshu know?",
     "ground_truth": "Anshu knows Python and C++ as programming languages"},
    {"question": "What is the ResumeIQ live link?",
     "ground_truth": "ResumeIQ is live at agentic-resume-analyzer.netlify.app"},
]


def build_ragas_dataset():
    cache_path = os.path.join(BASE_DIR, "ragas_cache.json")

    if os.path.exists(cache_path):
        print("Loading cached data...")
        with open(cache_path, "r") as f:
            return json.load(f)

    print(f"\nGenerating pipeline outputs for {len(test_set)} questions...")
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for i, test in enumerate(test_set):
        print(f"  {i+1}/{len(test_set)}: {test['question'][:50]}")

        embedding = create_embeddings(test["question"])
        chunks = retrieve_with_rerank(
            test["question"],
            eval_llm,
            n_results=5
        )
        context = format_context(chunks)
        answer = llm_response(test["question"], context, memory={})
        time.sleep(3)

        data["question"].append(test["question"])
        data["answer"].append(answer if answer else "")
        data["contexts"].append(chunks)
        data["ground_truth"].append(test["ground_truth"])

    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Cached to ragas_cache.json")
    return data


def run_ragas_evaluation():
    print("=" * 55)
    print("RAGAS EVALUATION")
    print("=" * 55)

    data = build_ragas_dataset()
    dataset = Dataset.from_dict(data)

    # point RAGAS at Groq using openai compatibility
    # RAGAS reads standard openai env variables
    os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "")
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

    # tell RAGAS which model to use
    from ragas.metrics import faithfulness as f_metric
    from ragas.metrics import answer_relevancy as ar_metric
    from ragas.metrics import context_precision as cp_metric
    from ragas.metrics import context_recall as cr_metric

    print("\nRunning RAGAS evaluation via Groq...")

    try:
        results = evaluate(
            dataset,
            metrics=[f_metric, ar_metric, cp_metric, cr_metric]
        )

        print("\n" + "=" * 55)
        print("RAGAS RESULTS")
        print("=" * 55)
        print(f"Faithfulness:       {results['faithfulness']:.4f}")
        print(f"Answer Relevancy:   {results['answer_relevancy']:.4f}")
        print(f"Context Precision:  {results['context_precision']:.4f}")
        print(f"Context Recall:     {results['context_recall']:.4f}")
        print("=" * 55)

        output = {
            "faithfulness": float(results["faithfulness"]),
            "answer_relevancy": float(results["answer_relevancy"]),
            "context_precision": float(results["context_precision"]),
            "context_recall": float(results["context_recall"]),
            "num_queries": len(test_set),
            "framework": "RAGAS v0.2.15",
            "judge_model": "openai/gpt-oss-120b via Groq"
        }

        output_path = os.path.join(BASE_DIR, "ragas_results.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nSaved to ragas_results.json")
        return results

    except Exception as e:
        print(f"\nRAGAS evaluation failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure ragas==0.2.15 is installed")
        print("2. Check GROQ_API_KEY is set in .env")
        print("3. Try: pip install ragas==0.2.15 datasets")
        return None


if __name__ == "__main__":
    run_ragas_evaluation()