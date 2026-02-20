"""
RAGAS Evaluation Service for the Audit Intelligence Platform.

Extends Project 1's evaluation with:
- 32 questions across 6 categories
- Category-level scoring breakdown
- Confidence scoring per answer
- History of evaluation runs for tracking quality over time
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


class EvaluationService:
    def __init__(self, rag_ask_fn, retriever):
        self.rag_ask_fn = rag_ask_fn
        self.retriever = retriever
        self.evaluation_history: list[dict] = []

    def _load_test_questions(self, test_file: str = "tests/eval_data/test_questions.json") -> list[dict]:
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        with open(test_file, "r") as f:
            data = json.load(f)
        questions = data.get("questions", [])
        if not questions:
            raise ValueError("Test file contains no questions")
        logger.info(f"Loaded {len(questions)} test questions")
        return questions

    def _evaluate_questions(self, questions: list[dict]) -> dict:
        eval_questions, eval_answers, eval_contexts, eval_ground_truths = [], [], [], []
        per_question_results = []

        for q in questions:
            try:
                result = self.rag_ask_fn(q["question"])
                answer = result if isinstance(result, str) else result.get("answer", "")
                retrieved_docs = self.retriever.invoke(q["question"])
                contexts = [doc.page_content for doc in retrieved_docs[:5]]

                eval_questions.append(q["question"])
                eval_answers.append(answer)
                eval_contexts.append(contexts)
                eval_ground_truths.append(q["ground_truth"])
                per_question_results.append({
                    "question": q["question"], "category": q.get("category", "unknown"),
                    "generated_answer": answer[:300], "status": "evaluated",
                })
            except Exception as e:
                logger.warning(f"Error evaluating question: {e}")
                per_question_results.append({
                    "question": q["question"], "category": q.get("category", "unknown"),
                    "status": "failed", "error": str(e),
                })

        if not eval_questions:
            return {"error": "No questions could be evaluated"}

        dataset = Dataset.from_dict({
            "question": eval_questions, "answer": eval_answers,
            "contexts": eval_contexts, "ground_truth": eval_ground_truths,
        })

        try:
            results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
            scores = {
                "faithfulness": round(results["faithfulness"], 4),
                "answer_relevancy": round(results["answer_relevancy"], 4),
                "context_precision": round(results["context_precision"], 4),
                "context_recall": round(results["context_recall"], 4),
            }
            scores["overall_score"] = round(sum(scores.values()) / len(scores), 4)
            scores["questions_evaluated"] = len(eval_questions)
            scores["questions_failed"] = len(questions) - len(eval_questions)
            scores["timestamp"] = datetime.now().isoformat()
            scores["per_question"] = per_question_results
            return scores
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e), "per_question": per_question_results}

    def run_evaluation(self, test_file: str = "tests/eval_data/test_questions.json") -> dict:
        logger.info("Starting full RAGAS evaluation...")
        questions = self._load_test_questions(test_file)
        results = self._evaluate_questions(questions)
        if "error" not in results:
            self.evaluation_history.append(results)
        return results

    def run_evaluation_by_category(self, test_file: str = "tests/eval_data/test_questions.json") -> dict:
        questions = self._load_test_questions(test_file)
        categories: dict[str, list] = {}
        for q in questions:
            cat = q.get("category", "unknown")
            categories.setdefault(cat, []).append(q)

        category_results = {}
        for category, cat_questions in categories.items():
            logger.info(f"Evaluating category '{category}' ({len(cat_questions)} questions)...")
            result = self._evaluate_questions(cat_questions)
            category_results[category] = {
                "question_count": len(cat_questions),
                "overall_score": result.get("overall_score"),
                "faithfulness": result.get("faithfulness"),
                "answer_relevancy": result.get("answer_relevancy"),
                "context_precision": result.get("context_precision"),
                "context_recall": result.get("context_recall"),
            }
        return {"category_scores": category_results, "timestamp": datetime.now().isoformat()}

    def get_evaluation_history(self) -> list[dict]:
        return [{
            "timestamp": r["timestamp"], "overall_score": r["overall_score"],
            "faithfulness": r["faithfulness"], "answer_relevancy": r["answer_relevancy"],
            "context_precision": r["context_precision"], "context_recall": r["context_recall"],
            "questions_evaluated": r["questions_evaluated"],
        } for r in self.evaluation_history]
