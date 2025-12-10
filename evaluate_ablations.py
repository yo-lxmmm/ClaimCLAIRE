"""Evaluate ablation studies A0-A4 on dev.json dataset."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from claire_agent.inconsistency.ablation_factory import create_ablation_agent, get_ablation_description
from retrieval.retriever_client import search_before_date_var
from utils.logger import logger


Label = Literal["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
Verdict = Literal["consistent", "inconsistent", "unknown"]


def map_label_to_verdict(label: str) -> Verdict:
    """Map dev.json label to verdict."""
    if label == "Supported":
        return "consistent"
    if label in {"Refuted", "Conflicting Evidence/Cherrypicking"}:
        return "inconsistent"
    return "unknown"


def parse_claim_date(date_str: str | None) -> str | None:
    """Parse claim_date from DD-MM-YYYY to YYYY-MM-DD."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        parts = date_str.strip().split("-")
        if len(parts) == 3:
            day, month, year = parts
            return f"{year}-{month}-{day}"
    except Exception:
        pass
    return None


@dataclass
class ExampleResult:
    claim_id: Any
    claim_text: str
    claim_date: str | None
    gold_label: str
    gold_verdict: Verdict
    system_verdict: Verdict
    report: Any = None


def compute_basic_metrics(results: list[ExampleResult]) -> dict[str, Any]:
    """Compute accuracy and confusion matrix on binary classification (consistent/inconsistent)."""
    # Filter out unknown verdicts
    filtered = [r for r in results if r.gold_verdict != "unknown" and r.system_verdict != "unknown"]
    if not filtered:
        return {"note": "No valid examples in evaluation subset."}

    total = len(filtered)
    correct = sum(1 for r in filtered if r.gold_verdict == r.system_verdict)

    conf = Counter((r.gold_verdict, r.system_verdict) for r in filtered)

    # Binary classification: consistent, inconsistent
    tp_consistent = conf[("consistent", "consistent")]
    fp_consistent = conf[("inconsistent", "consistent")]
    fn_consistent = conf[("consistent", "inconsistent")]

    tp_inconsistent = conf[("inconsistent", "inconsistent")]
    fp_inconsistent = conf[("consistent", "inconsistent")]
    fn_inconsistent = conf[("inconsistent", "consistent")]

    def safe_precision(tp: int, fp: int) -> float:
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def safe_recall(tp: int, fn: int) -> float:
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def safe_f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    p_cons = safe_precision(tp_consistent, fp_consistent)
    r_cons = safe_recall(tp_consistent, fn_consistent)
    f1_cons = safe_f1(p_cons, r_cons)

    p_incons = safe_precision(tp_inconsistent, fp_inconsistent)
    r_incons = safe_recall(tp_inconsistent, fn_inconsistent)
    f1_incons = safe_f1(p_incons, r_incons)

    macro_f1 = (f1_cons + f1_incons) / 2.0

    return {
        "num_examples_total": len(results),
        "num_examples_eval": total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
        "confusion": {
            "consistent": {
                "gold_consistent_pred_consistent": tp_consistent,
                "gold_consistent_pred_inconsistent": fn_consistent,
            },
            "inconsistent": {
                "gold_inconsistent_pred_consistent": fp_inconsistent,
                "gold_inconsistent_pred_inconsistent": tp_inconsistent,
            },
        },
        "per_class": {
            "consistent": {
                "precision": p_cons,
                "recall": r_cons,
                "f1": f1_cons,
            },
            "inconsistent": {
                "precision": p_incons,
                "recall": r_incons,
                "f1": f1_incons,
            },
        },
    }


def compute_label_breakdown(results: list[ExampleResult]) -> dict[str, Any]:
    """Compute metrics broken down by original label categories."""
    label_stats = {}
    
    for label in ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]:
        label_results = [r for r in results if r.gold_label == label]
        if not label_results:
            continue
        
        gold_verdict = map_label_to_verdict(label)
        correct = sum(1 for r in label_results if r.system_verdict == gold_verdict)
        
        verdict_dist = Counter(r.system_verdict for r in label_results)
        
        label_stats[label] = {
            "count": len(label_results),
            "gold_verdict": gold_verdict,
            "correct": correct,
            "accuracy": correct / len(label_results) if label_results else 0.0,
            "system_verdict_distribution": dict(verdict_dist),
        }
    
    return label_stats


async def evaluate_example(
    agent: Any,
    example: dict,
    ablation_name: str,
    claim_id: Any = None,
) -> ExampleResult:
    """Evaluate a single example with the given ablation agent."""
    claim_text = example.get("claim", "").strip()
    if not claim_text:
        raise ValueError("Claim text is required")
    
    raw_label = example.get("label", "")
    gold_label = raw_label
    gold_verdict = map_label_to_verdict(gold_label)
    
    claim_date = parse_claim_date(example.get("claim_date"))
    
    # Set date filter
    token = search_before_date_var.set(claim_date)
    try:
        report = await agent.analyze_claim(claim_text=claim_text, passage=claim_text)
    except Exception as e:
        logger.error(f"Error for {ablation_name} example id={claim_id}: {e}")
        report = None
    finally:
        search_before_date_var.reset(token)
    
    system_verdict: Verdict = (
        report.verdict if report and report.verdict in {"consistent", "inconsistent"} else "unknown"
    )
    
    return ExampleResult(
        claim_id=claim_id,
        claim_text=claim_text,
        claim_date=claim_date,
        gold_label=gold_label,
        gold_verdict=gold_verdict,
        system_verdict=system_verdict,
        report=report,
    )


async def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation studies A0-A4")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dev.json")
    parser.add_argument("--ablation", type=str, choices=["A0", "A1", "A2", "A3", "A4"], required=True, help="Ablation variant")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples to evaluate")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSON path")
    parser.add_argument("--engine", type=str, default="gemini-2.5-flash", help="LLM engine")
    parser.add_argument("--model-provider", type=str, default="google_genai", help="Model provider")
    parser.add_argument("--num-results", type=int, default=10, help="Number of search results per query")
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    logger.info(f"Loading {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    logger.info(f"Loaded {len(examples)} examples.")
    
    # Filter out "Not Enough Evidence"
    filtered = [ex for ex in examples if ex.get("label") != "Not Enough Evidence"]
    logger.info(f"Filtered out {len(examples) - len(filtered)} 'Not Enough Evidence' examples. Remaining: {len(filtered)}")
    
    # Subsample if requested
    if args.max_examples:
        filtered = filtered[:args.max_examples]
        logger.info(f"Subsampled to {len(filtered)} examples (max_examples={args.max_examples}).")
    
    # Initialize agent based on ablation
    ablation_description = get_ablation_description(args.ablation)
    logger.info(f"Initializing {args.ablation} agent: {ablation_description}")

    agent = create_ablation_agent(
        ablation_id=args.ablation,
        engine=args.engine,
        model_provider=args.model_provider,
        num_results_per_query=args.num_results,
    )
    
    # Evaluate
    results = []
    for idx, example in enumerate(filtered):
        claim_id = example.get("id", idx)
        claim_text = example.get("claim", "")[:60]
        logger.info(f"Evaluating {args.ablation} example {idx+1}/{len(filtered)}: '{claim_text}...'")
        result = await evaluate_example(agent, example, args.ablation, claim_id)
        results.append(result)
    
    # Compute metrics
    metrics = compute_basic_metrics(results)
    label_breakdown = compute_label_breakdown(results)
    
    # Save results
    output_path = Path(args.output_path)
    logger.info(f"Writing results to {output_path}...")

    serializable_results = [
        {
            "claim_id": r.claim_id,
            "claim": r.claim_text,
            "claim_date": r.claim_date,
            "gold_label": r.gold_label,
            "gold_verdict": r.gold_verdict,
            "system_verdict": r.system_verdict,
            "correct": r.gold_verdict == r.system_verdict,
        }
        for r in results
    ]

    output_data = {
        "metrics": metrics,
        "label_breakdown": label_breakdown,
        "results": serializable_results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    # Output CSV files (similar to evaluate_dev.py)
    # Handle both .json and .csv output paths
    output_str = str(output_path)
    if output_str.endswith(".json"):
        csv_output_path = Path(output_str.replace(".json", "_results.csv"))
        component_csv_path = Path(output_str.replace(".json", "_components.csv"))
    else:
        # If output_path is already .csv, create component CSV with _components suffix
        csv_output_path = Path(output_str)
        component_csv_path = Path(output_str.replace(".csv", "_components.csv"))

    logger.info(f"Writing CSV results to {csv_output_path} and {component_csv_path}...")

    # Write main results CSV
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "claim_id", "claim", "claim_date", "filter_date", "gold_label", "gold_verdict",
            "predicted_verdict", "correct", "success", "num_components", "num_sources",
            "gap_fill_triggered", "num_gap_fills", "explanation", "wording_feedback", "source_links"
        ])
        writer.writeheader()

        for idx, r in enumerate(results):
            report = r.report
            # Extract source links from search results
            source_links = []
            if report and hasattr(report, 'search_results') and report.search_results:
                for block in report.search_results:
                    if hasattr(block, 'url') and block.url:
                        source_links.append(block.url)
            links_str = "; ".join(source_links) if source_links else ""

            # Determine success (report exists and has verdict)
            success = report is not None and hasattr(report, 'verdict') and report.verdict in {"consistent", "inconsistent"}

            # filter_date is the same as claim_date in this case (the date used for temporal filtering)
            filter_date = r.claim_date or ""

            # Format claim_id as string
            claim_id_str = str(r.claim_id) if r.claim_id is not None else str(idx)

            writer.writerow({
                "claim_id": claim_id_str,
                "claim": r.claim_text,
                "claim_date": r.claim_date or "",
                "filter_date": filter_date,
                "gold_label": r.gold_label,
                "gold_verdict": r.gold_verdict,
                "predicted_verdict": r.system_verdict,
                "correct": r.gold_verdict == r.system_verdict,
                "success": success,
                "num_components": len(report.components) if report and hasattr(report, 'components') else 0,
                "num_sources": len(report.search_results) if report and hasattr(report, 'search_results') else 0,
                "gap_fill_triggered": report.gap_fill_triggered if report and hasattr(report, 'gap_fill_triggered') else False,
                "num_gap_fills": report.num_gap_fills if report and hasattr(report, 'num_gap_fills') else 0,
                "explanation": report.explanation if report and hasattr(report, 'explanation') else "",
                "wording_feedback": report.wording_feedback if report and hasattr(report, 'wording_feedback') else "",
                "source_links": links_str,
            })

    # Write component results CSV
    with open(component_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "claim_id", "component_index", "component_text", "initial_verdict",
            "gap_filled", "verdict", "verdict_changed", "reasoning"
        ])
        writer.writeheader()

        for idx, r in enumerate(results):
            report = r.report
            if report and hasattr(report, 'components') and hasattr(report, 'component_evaluations'):
                components = report.components
                evaluations = report.component_evaluations

                # Format claim_id as string
                claim_id_str = str(r.claim_id) if r.claim_id is not None else str(idx)

                for comp_idx, (component, evaluation) in enumerate(zip(components, evaluations)):
                    writer.writerow({
                        "claim_id": claim_id_str,
                        "component_index": comp_idx,
                        "component_text": component,
                        "initial_verdict": evaluation.initial_verdict if hasattr(evaluation, 'initial_verdict') and evaluation.initial_verdict else evaluation.verdict,
                        "gap_filled": evaluation.gap_filled if hasattr(evaluation, 'gap_filled') else False,
                        "verdict": evaluation.verdict if hasattr(evaluation, 'verdict') else "",
                        "verdict_changed": evaluation.verdict_changed if hasattr(evaluation, 'verdict_changed') else False,
                        "reasoning": evaluation.reasoning if hasattr(evaluation, 'reasoning') else "",
                    })
    
    # Print comprehensive classification report
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 80)
    
    if "per_class" in metrics:
        logger.info(f"\nOverall Metrics:")
        logger.info(f"  Total Examples: {metrics.get('num_examples_total', 0)}")
        logger.info(f"  Evaluated: {metrics.get('num_examples_eval', 0)}")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
        
        logger.info(f"\nPer-Class Metrics:")
        logger.info(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        logger.info("-" * 65)
        
        for class_name, class_metrics in metrics["per_class"].items():
            precision = class_metrics.get("precision", 0.0)
            recall = class_metrics.get("recall", 0.0)
            f1 = class_metrics.get("f1", 0.0)
            
            # Calculate support (true positives + false negatives)
            conf = metrics.get("confusion", {})
            if class_name == "consistent":
                support = conf.get("consistent", {}).get("gold_consistent_pred_consistent", 0) + \
                         conf.get("consistent", {}).get("gold_consistent_pred_inconsistent", 0)
            else:
                support = conf.get("inconsistent", {}).get("gold_inconsistent_pred_inconsistent", 0) + \
                         conf.get("inconsistent", {}).get("gold_inconsistent_pred_consistent", 0)
            
            logger.info(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
        
        logger.info("-" * 65)
        logger.info(f"{'avg / total':<15} {metrics.get('macro_f1', 0):<12.4f} {metrics.get('macro_f1', 0):<12.4f} {metrics.get('macro_f1', 0):<12.4f} {metrics.get('num_examples_eval', 0):<10}")
    
    logger.info(f"\nConfusion Matrix:")
    if "confusion" in metrics:
        conf = metrics["confusion"]
        logger.info(f"                    Predicted")
        logger.info(f"                  consistent  inconsistent")
        logger.info(f"Actual consistent    {conf.get('consistent', {}).get('gold_consistent_pred_consistent', 0):<10} {conf.get('consistent', {}).get('gold_consistent_pred_inconsistent', 0):<10}")
        logger.info(f"Actual inconsistent  {conf.get('inconsistent', {}).get('gold_inconsistent_pred_consistent', 0):<10} {conf.get('inconsistent', {}).get('gold_inconsistent_pred_inconsistent', 0):<10}")
    
    logger.info(f"\nLabel Breakdown:")
    for label, stats in label_breakdown.items():
        logger.info(f"  {label}:")
        logger.info(f"    Count: {stats['count']}")
        logger.info(f"    Accuracy: {stats['accuracy']:.4f}")
        logger.info(f"    Gold Verdict: {stats['gold_verdict']}")
        logger.info(f"    Predictions: {stats['system_verdict_distribution']}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Evaluation complete. Results written to:")
    logger.info(f"  JSON: {output_path}")
    logger.info(f"  CSV: {csv_output_path}")
    logger.info(f"  Components CSV: {component_csv_path}")


if __name__ == "__main__":
    asyncio.run(main())

