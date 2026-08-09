"""ET-BERT×packet cohortでC1–C8をpaired評価する統合ランナー．

C2/C5/C8はraw byteへ復号してからTCP fieldを変更し、整合した重複bigramを
再生成する。C7は[CLS]直後へattention mask対象のPAD位置だけを挿入し、可視な
内容を追加せずに元header tokenのposition IDだけを移動する。C6はC3について
末尾tokenが切り捨てられないsampleのpaired効果として集計する．
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[0]))
sys.path.insert(0, str(_HERE.parents[3]))

from run_classifier import Classifier, count_labels_num  # noqa: E402
from uer.opts import finetune_opts  # noqa: E402
from uer.utils import str2tokenizer  # noqa: E402
from uer.utils.config import load_hyperparam  # noqa: E402
from uer.utils.constants import CLS_TOKEN  # noqa: E402
from uer.utils.seed import set_seed  # noqa: E402

from asnfm.attack.tcp_fields import (  # noqa: E402
    BigramConsistencyError,
    field_unit_indices,
    flow_field_unit_indices,
    rewrite_flow_tcp_fields,
    rewrite_tcp_fields,
    swap_flow_tcp_fields,
    swap_tcp_fields,
)
from asnfm.attack.text_padding import pad_text_a  # noqa: E402
from asnfm.data.audit import load_vocab, sha256_file, target_unit_visibility  # noqa: E402
from asnfm.data.cohort import Sample, read_samples  # noqa: E402
from asnfm.statistics import (  # noqa: E402
    classification_metrics,
    conditional_attack_metrics,
    donor_margins,
    donor_permutation_test,
    factorial_effects,
    mcnemar_exact,
    numeric_summary,
    probabilistic_metrics,
    targeted_donor_metrics,
    true_class_margins,
)

CONDITIONS = ("C1", "C2", "C3", "C4", "C5", "C7", "C8")


def _rng(seed: int, sample_id: str, stream: str) -> np.random.Generator:
    payload = f"{seed}\0{sample_id}\0{stream}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(value)


def _cohort_hash(rows: list[Sample]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.sample_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _choose_donors(rows: list[Sample], seed: int) -> dict[str, Sample]:
    by_label: dict[int, list[Sample]] = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)
    assignments: dict[str, Sample] = {}
    for row in rows:
        candidates = [
            candidate
            for label, group in by_label.items()
            if label != row.label
            for candidate in group
        ]
        if not candidates:
            raise ValueError(f"sample {row.sample_id}: 別クラスdonorが存在しない")
        generator = _rng(seed, row.sample_id, "donor")
        assignments[row.sample_id] = candidates[int(generator.integers(0, len(candidates)))]
    return assignments


def _tokenize(
    args: argparse.Namespace, text_a: str, *, masked_prefix: int = 0
) -> tuple[list[int], list[int], bool]:
    tokens = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_a))
    cls = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN])
    src = cls + ([0] * masked_prefix) + tokens
    seg = [1] + ([0] * masked_prefix) + ([1] * len(tokens))
    truncated = len(src) > args.seq_length
    src = src[: args.seq_length]
    seg = seg[: args.seq_length]
    while len(src) < args.seq_length:
        src.append(0)
        seg.append(0)
    return src, seg, truncated


def _transform(
    row: Sample,
    condition: str,
    *,
    fields: list[str],
    prefix_units: int,
    seed: int,
    donor: Sample,
    flow_proxy: bool = False,
) -> tuple[str, int, tuple[str, ...], str | None, int | None]:
    """text、masked prefix、変更field、donor ID/labelを返す．"""
    packet_unit_counts = (
        [int(value) for value in row.metadata["packet_unit_counts"].split(",")]
        if row.metadata.get("packet_unit_counts")
        else None
    )
    donor_packet_unit_counts = (
        [int(value) for value in donor.metadata["packet_unit_counts"].split(",")]
        if donor.metadata.get("packet_unit_counts")
        else None
    )
    if condition == "C1":
        return row.text_a, 0, (), None, None
    if condition == "C2":
        rewrite = rewrite_flow_tcp_fields if flow_proxy else rewrite_tcp_fields
        kwargs = {"packet_unit_counts": packet_unit_counts} if flow_proxy else {}
        text, changed = rewrite(
            row.text_a, fields, rng=_rng(seed, row.sample_id, "rewrite"), **kwargs
        )
        return text, 0, changed, None, None
    if condition == "C3":
        text = pad_text_a(row.text_a, prefix_units, "pre", rng=_rng(seed, row.sample_id, "prefix"))
        return text, 0, (), None, None
    if condition == "C4":
        text = pad_text_a(row.text_a, prefix_units, "post", rng=_rng(seed, row.sample_id, "prefix"))
        return text, 0, (), None, None
    if condition == "C5":
        rewrite = rewrite_flow_tcp_fields if flow_proxy else rewrite_tcp_fields
        kwargs = {"packet_unit_counts": packet_unit_counts} if flow_proxy else {}
        rewritten, changed = rewrite(
            row.text_a, fields, rng=_rng(seed, row.sample_id, "rewrite"), **kwargs
        )
        text = pad_text_a(rewritten, prefix_units, "pre", rng=_rng(seed, row.sample_id, "prefix"))
        return text, 0, changed, None, None
    if condition == "C7":
        return row.text_a, prefix_units, (), None, None
    if condition == "C8":
        swap = swap_flow_tcp_fields if flow_proxy else swap_tcp_fields
        kwargs = (
            {
                "packet_unit_counts": packet_unit_counts,
                "donor_packet_unit_counts": donor_packet_unit_counts,
            }
            if flow_proxy
            else {}
        )
        text, changed = swap(row.text_a, donor.text_a, fields, **kwargs)
        return text, 0, changed, donor.sample_id, donor.label
    raise ValueError(f"未知の条件: {condition}")


def _predict_scores(
    model: torch.nn.Module,
    device: torch.device,
    src: list[list[int]],
    seg: list[list[int]],
    batch_size: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    src_tensor = torch.LongTensor(src)
    seg_tensor = torch.LongTensor(seg)
    predictions: list[int] = []
    logit_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(src), batch_size):
            source = src_tensor[start : start + batch_size].to(device)
            segments = seg_tensor[start : start + batch_size].to(device)
            _, logits = model(source, None, segments)
            predictions.extend(int(value) for value in torch.argmax(logits, dim=1).cpu().tolist())
            logit_batches.append(logits.detach().cpu().numpy())
            probability_batches.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return predictions, np.concatenate(logit_batches), np.concatenate(probability_batches)


def _predict(
    model: torch.nn.Module,
    device: torch.device,
    src: list[list[int]],
    seg: list[list[int]],
    batch_size: int,
) -> list[int]:
    """既存caller向けにhard predictionだけを返す．"""
    predictions, _, _ = _predict_scores(model, device, src, seg, batch_size)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    finetune_opts(parser)
    parser.add_argument("--pooling", default="first")
    parser.add_argument("--tokenizer", default="bert")
    parser.add_argument("--soft_targets", action="store_true")
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--load_model_path", required=True)
    parser.add_argument("--tcp_fields", default="seq,ack")
    parser.add_argument("--prefix_units", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--flow_proxy",
        action="store_true",
        help="TCP-onlyが外部保証された最大64-bigram連結flowの全packetを介入対象にする",
    )
    parser.add_argument(
        "--filter_full_visibility",
        action="store_true",
        help="対象field unitが全て語彙内かつ入力長内のsampleだけを固定cohortにする",
    )
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--predictions_out", required=True)
    parser.add_argument("--scores_out")
    args = parser.parse_args()

    args = load_hyperparam(args)
    set_seed(args.seed)
    args.labels_num = count_labels_num(args.train_path)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    fields = [value.strip() for value in args.tcp_fields.split(",") if value.strip()]

    rows = read_samples(args.test_path, split="test", tcp_only=True)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit("TCP-only test cohortが空")

    invalid: list[str] = []
    target_indices: dict[str, list[int]] = {}
    for row in rows:
        try:
            if args.flow_proxy:
                counts = (
                    [int(value) for value in row.metadata["packet_unit_counts"].split(",")]
                    if row.metadata.get("packet_unit_counts")
                    else None
                )
                target_indices[row.sample_id] = flow_field_unit_indices(
                    row.text_a, fields, packet_unit_counts=counts
                )
            else:
                target_indices[row.sample_id] = field_unit_indices(row.text_a, fields)
        except BigramConsistencyError:
            invalid.append(row.sample_id)
    if invalid:
        raise SystemExit(f"bigram整合性NG: {len(invalid)}件（例: {invalid[:3]}）")

    vocab = load_vocab(args.vocab_path)
    visibility_n = visibility_d = 0
    fully_visible_rows: list[Sample] = []
    for row in rows:
        visible, total = target_unit_visibility(
            row, target_indices[row.sample_id], seq_length=args.seq_length, vocab=vocab
        )
        visibility_n += visible
        visibility_d += total
        if total > 0 and visible == total:
            fully_visible_rows.append(row)
    prefilter_n = len(rows)
    prefilter_visibility = visibility_n / visibility_d if visibility_d else 0.0
    if args.filter_full_visibility:
        rows = fully_visible_rows
        target_indices = {row.sample_id: target_indices[row.sample_id] for row in rows}
        if not rows:
            raise SystemExit("full-visibility filter後のcohortが空")
        field_visibility = 1.0
    else:
        field_visibility = prefilter_visibility
        if field_visibility != 1.0:
            raise SystemExit(f"対象field可視率が100%でない: {field_visibility:.6f}")

    model = Classifier(args)
    model.load_state_dict(torch.load(args.load_model_path, map_location="cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    donors = _choose_donors(rows, args.seed)

    condition_predictions: dict[str, list[int]] = {}
    condition_logits: dict[str, np.ndarray] = {}
    condition_probabilities: dict[str, np.ndarray] = {}
    records_by_condition: dict[str, list[dict[str, object]]] = {}
    records: list[dict[str, object]] = []
    transform_counts: dict[str, Counter[str]] = {condition: Counter() for condition in CONDITIONS}
    truncation_counts: Counter[str] = Counter()
    for condition in CONDITIONS:
        src: list[list[int]] = []
        seg: list[list[int]] = []
        condition_rows: list[dict[str, object]] = []
        for row in rows:
            assigned_donor = donors[row.sample_id]
            text, masked_prefix, changed, _donor_id, _donor_label = _transform(
                row,
                condition,
                fields=fields,
                prefix_units=args.prefix_units,
                seed=args.seed,
                donor=assigned_donor,
                flow_proxy=args.flow_proxy,
            )
            source, segments, truncated = _tokenize(args, text, masked_prefix=masked_prefix)
            src.append(source)
            seg.append(segments)
            truncation_counts[condition] += int(truncated)
            transform_counts[condition].update(changed)
            condition_rows.append(
                {
                    "sample_id": row.sample_id,
                    "row_index": row.row_index,
                    "flow_id": row.flow_id or "",
                    "session_id": row.session_id or "",
                    "capture_id": row.capture_id or "",
                    "true": row.label,
                    "condition": condition,
                    "donor": assigned_donor.label,
                    "donor_sample_id": assigned_donor.sample_id,
                    "truncated": int(truncated),
                    "c6_no_truncation_eligible": int(
                        1 + len(args.tokenizer.tokenize(row.text_a)) + args.prefix_units
                        <= args.seq_length
                    ),
                    "changed_fields": ",".join(changed),
                }
            )
        predictions, logits, probabilities = _predict_scores(
            model, device, src, seg, args.batch_size
        )
        condition_predictions[condition] = predictions
        condition_logits[condition] = logits
        condition_probabilities[condition] = probabilities
        margins = true_class_margins([row.label for row in rows], logits)
        for index, (record, prediction) in enumerate(zip(condition_rows, predictions, strict=True)):
            record["pred"] = prediction
            record["true_probability"] = float(probabilities[index, rows[index].label])
            record["true_margin"] = float(margins[index])
            donor_label = int(record["donor"])
            record["donor_probability"] = float(probabilities[index, donor_label])
            record["donor_margin"] = float(
                logits[index, donor_label] - logits[index, rows[index].label]
            )
            records.append(record)
        records_by_condition[condition] = condition_rows
        metrics = classification_metrics([row.label for row in rows], predictions)
        print(
            f"[suite] {condition} n={len(rows)} acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} truncated={truncation_counts[condition]}",
            flush=True,
        )

    true = [row.label for row in rows]
    clean_predictions = condition_predictions["C1"]
    for condition, condition_rows in records_by_condition.items():
        for index, record in enumerate(condition_rows):
            clean_correct = clean_predictions[index] == true[index]
            attack_correct = condition_predictions[condition][index] == true[index]
            record["clean_correct"] = int(clean_correct)
            record["attack_success_from_clean"] = int(clean_correct and not attack_correct)
            record["clean_error_recovered"] = int(not clean_correct and attack_correct)
            record["targeted_donor_success"] = int(
                condition == "C8"
                and clean_correct
                and condition_predictions[condition][index] == int(record["donor"])
            )

    output_path = Path(args.predictions_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id",
        "row_index",
        "flow_id",
        "session_id",
        "capture_id",
        "true",
        "condition",
        "pred",
        "true_probability",
        "true_margin",
        "donor",
        "donor_probability",
        "donor_margin",
        "donor_sample_id",
        "clean_correct",
        "attack_success_from_clean",
        "clean_error_recovered",
        "targeted_donor_success",
        "truncated",
        "c6_no_truncation_eligible",
        "changed_fields",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    scores_path = (
        Path(args.scores_out)
        if args.scores_out
        else output_path.with_name(f"{output_path.stem}.scores.npz")
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scores_path,
        sample_ids=np.asarray([row.sample_id for row in rows]),
        true=np.asarray(true, dtype=np.int64),
        donor=np.asarray([donors[row.sample_id].label for row in rows], dtype=np.int64),
        **{f"logits_{condition}": values for condition, values in condition_logits.items()},
        **{
            f"probabilities_{condition}": values
            for condition, values in condition_probabilities.items()
        },
    )

    condition_metrics = {
        condition: classification_metrics(true, predictions)
        for condition, predictions in condition_predictions.items()
    }
    probability_metrics = {
        condition: probabilistic_metrics(true, probabilities)
        for condition, probabilities in condition_probabilities.items()
    }
    margin_values = {
        condition: true_class_margins(true, logits)
        for condition, logits in condition_logits.items()
    }
    margin_metrics = {
        condition: {
            "true_margin": numeric_summary(values),
            "change_from_C1": numeric_summary(values - margin_values["C1"]),
        }
        for condition, values in margin_values.items()
    }
    conditional_metrics = {
        condition: conditional_attack_metrics(
            true, condition_predictions["C1"], condition_predictions[condition]
        )
        for condition in CONDITIONS
        if condition != "C1"
    }
    donor_labels = [donors[row.sample_id].label for row in rows]
    c8_donor_margins = donor_margins(true, donor_labels, condition_logits["C8"])
    c1_donor_margins = donor_margins(true, donor_labels, condition_logits["C1"])
    donor_metrics = {
        "targeted": targeted_donor_metrics(
            true, condition_predictions["C1"], condition_predictions["C8"], donor_labels
        ),
        "permutation": donor_permutation_test(
            true,
            condition_predictions["C8"],
            donor_labels,
            n_permutations=10_000,
            seed=args.seed,
        ),
        "donor_margin": numeric_summary(c8_donor_margins),
        "donor_margin_change_from_C1": numeric_summary(c8_donor_margins - c1_donor_margins),
    }
    accuracies = {
        condition: float(metrics["accuracy"]) for condition, metrics in condition_metrics.items()
    }
    eligible = [
        index
        for index, row in enumerate(rows)
        if 1 + len(args.tokenizer.tokenize(row.text_a)) + args.prefix_units <= args.seq_length
    ]
    c6_clean = float(np.mean([condition_predictions["C1"][i] == true[i] for i in eligible]))
    c6_attack = float(np.mean([condition_predictions["C3"][i] == true[i] for i in eligible]))
    all_attack_effect = accuracies["C1"] - accuracies["C3"]
    c6_attack_effect = c6_clean - c6_attack
    paired_tests = {
        f"C1_vs_{condition}": mcnemar_exact(true, condition_predictions["C1"], predictions)
        for condition, predictions in condition_predictions.items()
        if condition != "C1"
    }
    metrics = {
        "task": "C1-C8 paired causal suite",
        "status": "exploratory" if not all(row.flow_id for row in rows) else "flow_identified",
        "cohort": {
            "n": len(rows),
            "n_before_visibility_filter": prefilter_n,
            "n_excluded_by_visibility_filter": prefilter_n - len(rows),
            "sha256": _cohort_hash(rows),
            "tcp_only": True,
            "flow_id_complete": all(row.flow_id for row in rows),
            "n_flows": len({row.flow_id for row in rows if row.flow_id}),
        },
        "g0": {
            "field_visibility": field_visibility,
            "field_visibility_before_filter": prefilter_visibility,
            "bigram_consistency_rate": 1.0,
            "flow_gate_pass": all(row.flow_id for row in rows),
        },
        "seed": args.seed,
        "tcp_fields": fields,
        "prefix_units": args.prefix_units,
        "flow_proxy": args.flow_proxy,
        "seq_length": args.seq_length,
        "conditions": condition_metrics,
        "probabilistic": probability_metrics,
        "margins": {
            "conditions": margin_metrics,
            "interaction": float(
                np.mean(margin_values["C5"])
                - np.mean(margin_values["C2"])
                - np.mean(margin_values["C3"])
                + np.mean(margin_values["C1"])
            ),
        },
        "conditional_attack": conditional_metrics,
        "targeted_donor": donor_metrics,
        "factorial_effects": factorial_effects(accuracies),
        "c6_truncation_control": {
            "n_eligible": len(eligible),
            "clean_accuracy": c6_clean,
            "attack_accuracy": c6_attack,
            "attack_effect_all": all_attack_effect,
            "attack_effect_no_truncation": c6_attack_effect,
            "effect_difference": c6_attack_effect - all_attack_effect,
        },
        "transformation": {
            condition: {
                "changed_field_counts": dict(transform_counts[condition]),
                "truncated": truncation_counts[condition],
            }
            for condition in CONDITIONS
        },
        "mcnemar_packet_level_descriptive_only": paired_tests,
        "artifacts_sha256": {
            "model": sha256_file(args.load_model_path),
            "test": sha256_file(args.test_path),
            "vocab": sha256_file(args.vocab_path),
            "config": sha256_file(args.config_path),
        },
        "predictions_path": str(output_path),
        "scores_path": str(scores_path),
        "scores_sha256": sha256_file(scores_path),
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[suite] saved predictions: {output_path}", flush=True)
    print(f"[suite] saved scores: {scores_path}", flush=True)
    print(f"[suite] saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
