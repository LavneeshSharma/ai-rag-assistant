import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chains.conversational_rag import create_conversational_rag_chain
from config.settings import DATA_DIR
from langchain_chroma import Chroma
from utils.chunker import chunk_documents
from utils.embeddings import create_embeddings
from utils.pdf_loader import load_all_pdfs


DATASET_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.json"
SUMMARY_PATH = PROJECT_ROOT / "evaluation" / "summary.json"
EVAL_INDEX_PATH = PROJECT_ROOT / "vector_store" / "eval_index"
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0"))


def load_dataset() -> List[Dict[str, Any]]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def apply_eval_limit(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if EVAL_LIMIT <= 0:
        return dataset
    return dataset[:EVAL_LIMIT]


def rebuild_eval_index() -> Dict[str, Any]:
    if EVAL_INDEX_PATH.exists():
        shutil.rmtree(EVAL_INDEX_PATH)

    EVAL_INDEX_PATH.mkdir(parents=True, exist_ok=True)

    documents = load_all_pdfs(DATA_DIR)
    chunks = chunk_documents(documents)

    if not chunks:
        raise RuntimeError(f"No PDF chunks found in {DATA_DIR}.")

    embedding_model = create_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(EVAL_INDEX_PATH),
    )

    return {
        "pdf_count": len({doc.metadata.get("source") for doc in documents}),
        "page_count": len(documents),
        "chunk_count": len(chunks),
    }


def split_answer_and_sources(response: str) -> Dict[str, Any]:
    answer = response.strip()
    sources: List[str] = []

    if answer.startswith("Answer:"):
        answer = answer[len("Answer:") :].strip()

    if "\nSources:" in answer:
        answer_text, sources_text = answer.split("\nSources:", 1)
        answer = answer_text.strip()
        sources = [
            line.strip()
            for line in sources_text.strip().splitlines()
            if line.strip()
        ]

    return {
        "answer_generated": answer,
        "sources_returned": sources,
    }


def has_expected_source(result: Dict[str, Any]) -> bool:
    expected_source = result.get("expected_source_file")
    if not expected_source:
        return False

    return any(
        expected_source in source
        for source in result.get("sources_returned", [])
    )


def calculate_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    items = results.get("results", [])
    total_questions = results.get("total_questions", len(items))
    completed_questions = len(items)

    citation_count = sum(1 for item in items if item.get("sources_returned"))
    source_match_count = sum(1 for item in items if has_expected_source(item))
    answer_present_count = sum(
        1
        for item in items
        if item.get("answer_generated", "").strip()
    )
    failure_count = sum(1 for item in items if item.get("error")) + (
        total_questions - completed_questions
    )

    latencies = [
        item.get("latency_seconds", 0)
        for item in items
        if item.get("latency_seconds") is not None
    ]
    answer_lengths = [
        len(item.get("answer_generated", "").split())
        for item in items
    ]

    denominator = total_questions or 1
    completed_denominator = completed_questions or 1

    return {
        "dataset_path": results.get("dataset_path"),
        "results_path": str(RESULTS_PATH),
        "eval_index_path": results.get("eval_index_path"),
        "total_dataset_questions": results.get("total_dataset_questions"),
        "total_questions": total_questions,
        "eval_limit": results.get("eval_limit"),
        "completed_questions": completed_questions,
        "metrics": {
            "citation_coverage": round(citation_count / denominator, 4),
            "source_match_accuracy": round(source_match_count / denominator, 4),
            "answer_present_rate": round(answer_present_count / denominator, 4),
            "failure_rate": round(failure_count / denominator, 4),
            "average_latency_seconds": round(
                sum(latencies) / completed_denominator,
                3,
            ),
            "average_answer_length_words": round(
                sum(answer_lengths) / completed_denominator,
                2,
            ),
        },
        "counts": {
            "with_citations": citation_count,
            "source_matches": source_match_count,
            "answers_present": answer_present_count,
            "failures": failure_count,
        },
        "error": results.get("error"),
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def evaluate_item(item: Dict[str, Any], eval_index_path: str) -> Dict[str, Any]:
    question = item["question"]
    started_at = time.perf_counter()

    try:
        response = create_conversational_rag_chain(
            question,
            active_index_path=eval_index_path,
            chat_messages=[],
        )
        latency_seconds = time.perf_counter() - started_at
        parsed_response = split_answer_and_sources(response)

        return {
            "question": question,
            "answer_generated": parsed_response["answer_generated"],
            "expected_answer": item["expected_answer"],
            "sources_returned": parsed_response["sources_returned"],
            "expected_source_file": item["expected_source_file"],
            "expected_pages": item["expected_pages"],
            "category": item["category"],
            "latency_seconds": round(latency_seconds, 3),
            "pass_fail": None,
            "error": None,
        }
    except Exception as exc:
        latency_seconds = time.perf_counter() - started_at
        return {
            "question": question,
            "answer_generated": "",
            "expected_answer": item["expected_answer"],
            "sources_returned": [],
            "expected_source_file": item["expected_source_file"],
            "expected_pages": item["expected_pages"],
            "category": item["category"],
            "latency_seconds": round(latency_seconds, 3),
            "pass_fail": None,
            "error": str(exc),
        }


def run_eval() -> Dict[str, Any]:
    full_dataset = load_dataset()
    dataset = apply_eval_limit(full_dataset)
    results = {
        "dataset_path": str(DATASET_PATH),
        "eval_index_path": str(EVAL_INDEX_PATH),
        "index_stats": None,
        "total_dataset_questions": len(full_dataset),
        "total_questions": len(dataset),
        "eval_limit": EVAL_LIMIT or None,
        "results": [],
        "error": None,
    }

    try:
        results["index_stats"] = rebuild_eval_index()
    except Exception as exc:
        results["error"] = str(exc)
        write_json(RESULTS_PATH, results)
        write_json(SUMMARY_PATH, calculate_summary(results))
        return results

    for index, item in enumerate(dataset, start=1):
        print(f"Running eval {index}/{len(dataset)}: {item['question']}")
        results["results"].append(evaluate_item(item, str(EVAL_INDEX_PATH)))

    write_json(RESULTS_PATH, results)
    write_json(SUMMARY_PATH, calculate_summary(results))

    return results


if __name__ == "__main__":
    output = run_eval()
    print(f"\nSaved evaluation results to {RESULTS_PATH}")
    print(f"Saved evaluation summary to {SUMMARY_PATH}")
    print(f"Total questions: {output['total_questions']}")
