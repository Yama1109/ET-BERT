"""C1/C3/C8に対する訓練不要attention介入の用量反応runner．"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[0]))
sys.path.insert(0, str(_HERE.parents[3]))

from causal_suite import (  # noqa: E402
    _choose_donors,
    _predict_scores,
    _tokenize,
    _transform,
)
from run_classifier import Classifier, count_labels_num  # noqa: E402
from uer.layers.multi_headed_attn import MultiHeadedAttention  # noqa: E402
from uer.opts import finetune_opts  # noqa: E402
from uer.utils import str2tokenizer  # noqa: E402
from uer.utils.config import load_hyperparam  # noqa: E402
from uer.utils.seed import set_seed  # noqa: E402

from asnfm.attack.tcp_fields import (  # noqa: E402
    BigramConsistencyError,
    field_unit_indices,
    flow_field_unit_indices,
)
from asnfm.data.audit import load_vocab, sha256_file, target_unit_visibility  # noqa: E402
from asnfm.data.cohort import read_samples  # noqa: E402
from asnfm.statistics import (  # noqa: E402
    classification_metrics,
    conditional_attack_metrics,
    donor_margins,
    donor_permutation_test,
    intervention_transition_metrics,
    numeric_summary,
    probabilistic_metrics,
    targeted_donor_metrics,
    true_class_margins,
)


def _parse_indices(value: str, upper: int) -> list[int]:
    if value == "all":
        return list(range(upper))
    return [int(item) for item in value.split(",") if item.strip()]


def _set_intervention(
    modules: list[MultiHeadedAttention],
    *,
    scale: float,
    positions: list[int],
    layers: list[int],
    heads: list[int],
) -> None:
    selected_layers = set(layers)
    for layer, module in enumerate(modules):
        module.intervention_scale = scale if layer in selected_layers else 1.0
        module.intervention_key_positions = tuple(positions) if layer in selected_layers else ()
        module.intervention_heads = tuple(heads)


def _selection_scopes(
    scan_axis: str,
    layers: list[int],
    heads: list[int],
    layer_groups: list[list[int]] | None = None,
) -> list[tuple[str, list[int], list[int]]]:
    if scan_axis == "none":
        return [("joint", layers, heads)]
    if scan_axis == "layer":
        return [(f"layer_{layer}", [layer], heads) for layer in layers]
    if scan_axis == "head":
        return [(f"head_{head}", layers, [head]) for head in heads]
    if scan_axis == "layer_group":
        if not layer_groups:
            raise ValueError("layer_group scanには--layer_groupsが必要")
        allowed = set(layers)
        scopes = []
        for group in layer_groups:
            if not group or not set(group) <= allowed:
                raise ValueError(f"layer groupが--layersの範囲外: {group}")
            name = "layers_" + "_".join(str(layer) for layer in group)
            scopes.append((name, group, heads))
        return scopes
    raise ValueError(f"未知のscan_axis: {scan_axis}")


def _reset_attention_statistics(
    modules: list[MultiHeadedAttention], selected_layers: list[int], enabled: bool
) -> None:
    selected = set(selected_layers)
    for layer, module in enumerate(modules):
        module.collect_attention_stats = enabled and layer in selected
        module.attention_pre_mass_sum = None
        module.attention_post_mass_sum = None
        module.attention_stats_count = 0


def _attention_statistics(
    modules: list[MultiHeadedAttention], selected_layers: list[int]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for layer in selected_layers:
        module = modules[layer]
        count = int(getattr(module, "attention_stats_count", 0))
        pre_sum = getattr(module, "attention_pre_mass_sum", None)
        post_sum = getattr(module, "attention_post_mass_sum", None)
        if count == 0 or pre_sum is None or post_sum is None:
            continue
        pre = (pre_sum / count).numpy()
        post = (post_sum / count).numpy()
        output[str(layer)] = {
            "query": "CLS",
            "n_samples": count,
            "pre_target_mass_mean": float(np.mean(pre)),
            "post_target_mass_mean": float(np.mean(post)),
            "pre_target_mass_per_head": [float(value) for value in pre],
            "post_target_mass_per_head": [float(value) for value in post],
        }
    return output


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
    parser.add_argument("--scales", default="1.0,0.75,0.5,0.25,0.0")
    parser.add_argument("--conditions", default="C1,C3,C8")
    parser.add_argument("--target", choices=["leading", "header"], default="leading")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument(
        "--scan_axis", choices=["none", "layer", "head", "layer_group"], default="none"
    )
    parser.add_argument("--layer_groups", default="")
    parser.add_argument("--collect_attention_stats", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--flow_proxy",
        action="store_true",
        help="packet_unit_counts付きflow入力をpacket境界どおりに復号・介入する",
    )
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--predictions_out", required=True)
    parser.add_argument("--scores_out")
    args = parser.parse_args()

    args = load_hyperparam(args)
    set_seed(args.seed)
    args.labels_num = count_labels_num(args.train_path)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    fields = [item.strip() for item in args.tcp_fields.split(",") if item.strip()]
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    scales = [float(item) for item in args.scales.split(",") if item.strip()]

    rows = read_samples(args.test_path, split="test", tcp_only=True)
    vocab = load_vocab(args.vocab_path)
    target_indices: dict[str, list[int]] = {}
    invalid: list[str] = []
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

    prefilter_n = len(rows)
    rows = [
        row
        for row in rows
        if (lambda counts: counts[0] == counts[1] and counts[1] > 0)(
            target_unit_visibility(
                row,
                target_indices[row.sample_id],
                seq_length=args.seq_length,
                vocab=vocab,
            )
        )
    ]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit("full-visible TCP cohortが空")

    model = Classifier(args)
    model.load_state_dict(torch.load(args.load_model_path, map_location="cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    modules = [module for module in model.modules() if isinstance(module, MultiHeadedAttention)]
    if not modules:
        raise SystemExit("MultiHeadedAttention moduleがない")
    layers = _parse_indices(args.layers, len(modules))
    heads = _parse_indices(args.heads, modules[0].heads_num)
    layer_groups = [
        [int(value) for value in group.split(",") if value.strip()]
        for group in args.layer_groups.split(";")
        if group.strip()
    ]
    donors = _choose_donors(rows, args.seed)
    true = [row.label for row in rows]
    records: list[dict[str, object]] = []
    results: dict[str, object] = {}
    cell_predictions: dict[tuple[str, str, float], list[int]] = {}
    cell_logits: dict[tuple[str, str, float], np.ndarray] = {}
    cell_probabilities: dict[tuple[str, str, float], np.ndarray] = {}
    donor_labels_by_condition: dict[str, list[int | None]] = {}
    inputs_by_condition: dict[str, tuple[list[list[int]], list[list[int]]]] = {}
    scopes = _selection_scopes(args.scan_axis, layers, heads, layer_groups)

    for condition in conditions:
        src: list[list[int]] = []
        seg: list[list[int]] = []
        donor_labels: list[int | None] = []
        for row in rows:
            text, masked_prefix, _, _, donor_label = _transform(
                row,
                condition,
                fields=fields,
                prefix_units=args.prefix_units,
                seed=args.seed,
                donor=donors[row.sample_id],
                flow_proxy=args.flow_proxy,
            )
            source, segments, _ = _tokenize(args, text, masked_prefix=masked_prefix)
            src.append(source)
            seg.append(segments)
            donor_labels.append(donor_label)
        donor_labels_by_condition[condition] = donor_labels
        inputs_by_condition[condition] = (src, seg)

    for scope, scope_layers, scope_heads in scopes:
        for condition in conditions:
            src, seg = inputs_by_condition[condition]
            donor_labels = donor_labels_by_condition[condition]
            for scale in scales:
                if args.target == "leading":
                    positions = list(range(1, 1 + args.prefix_units))
                elif condition == "C3":
                    positions = list(range(1 + args.prefix_units, 1 + 2 * args.prefix_units))
                else:
                    positions = list(range(1, 1 + args.prefix_units))
                _set_intervention(
                    modules,
                    scale=scale,
                    positions=positions,
                    layers=scope_layers,
                    heads=scope_heads,
                )
                _reset_attention_statistics(modules, scope_layers, args.collect_attention_stats)
                predictions, logits, probabilities = _predict_scores(
                    model, device, src, seg, args.batch_size
                )
                cell = (scope, condition, scale)
                cell_predictions[cell] = predictions
                cell_logits[cell] = logits
                cell_probabilities[cell] = probabilities
                classification = classification_metrics(true, predictions)
                margins = true_class_margins(true, logits)
                key = f"{condition}@{scale:g}"
                output_key = key if args.scan_axis == "none" else f"{scope}/{key}"
                result = {
                    "classification": classification,
                    "probabilistic": probabilistic_metrics(true, probabilities),
                    "true_margin": numeric_summary(margins),
                    "positions": positions,
                    "selection": {"scope": scope, "layers": scope_layers, "heads": scope_heads},
                }
                if args.collect_attention_stats:
                    result["attention_mass"] = _attention_statistics(modules, scope_layers)
                if condition == "C8":
                    c8_donors = [int(label) for label in donor_labels if label is not None]
                    donor_permutation = donor_permutation_test(
                        true,
                        predictions,
                        c8_donors,
                        n_permutations=10_000,
                        seed=args.seed,
                    )
                    result["donor_permutation"] = donor_permutation
                    result["donor_excess"] = float(donor_permutation["observed"]) - float(
                        donor_permutation["null_mean"]
                    )
                    result["donor_margin"] = numeric_summary(donor_margins(true, c8_donors, logits))
                results[output_key] = result
                for index, (row, prediction, donor_label) in enumerate(
                    zip(rows, predictions, donor_labels, strict=True)
                ):
                    records.append(
                        {
                            "sample_id": row.sample_id,
                            "flow_id": row.flow_id or "",
                            "scope": scope,
                            "layers": ",".join(str(value) for value in scope_layers),
                            "heads": ",".join(str(value) for value in scope_heads),
                            "true": row.label,
                            "condition": condition,
                            "scale": scale,
                            "pred": prediction,
                            "true_probability": float(probabilities[index, row.label]),
                            "true_margin": float(margins[index]),
                            "donor": "" if donor_label is None else donor_label,
                            "donor_probability": ""
                            if donor_label is None
                            else float(probabilities[index, donor_label]),
                            "donor_margin": ""
                            if donor_label is None
                            else float(logits[index, donor_label] - logits[index, row.label]),
                        }
                    )
                print(
                    f"[intervention] {scope} {key} acc={classification['accuracy']:.4f} "
                    f"positions={positions}",
                    flush=True,
                )

    sample_positions = {row.sample_id: index for index, row in enumerate(rows)}
    for scope, _, _ in scopes:
        if (scope, "C1", 1.0) not in cell_predictions:
            raise SystemExit("条件付き指標にはC1@1が必要")
        clean_predictions = cell_predictions[(scope, "C1", 1.0)]
        for condition in conditions:
            if (scope, condition, 1.0) not in cell_predictions:
                raise SystemExit(f"transition指標には{scope}/{condition}@1が必要")
            baseline_predictions = cell_predictions[(scope, condition, 1.0)]
            for scale in scales:
                key = f"{condition}@{scale:g}"
                output_key = key if args.scan_axis == "none" else f"{scope}/{key}"
                predictions = cell_predictions[(scope, condition, scale)]
                result = results[output_key]
                result["transition_from_scale_1"] = intervention_transition_metrics(
                    true, baseline_predictions, predictions
                )
                result["conditional_attack_from_C1_1"] = conditional_attack_metrics(
                    true, clean_predictions, predictions
                )
                if condition == "C8":
                    c8_donors = [
                        int(label)
                        for label in donor_labels_by_condition[condition]
                        if label is not None
                    ]
                    result["targeted_donor_from_C1_1"] = targeted_donor_metrics(
                        true, clean_predictions, predictions, c8_donors
                    )
                    baseline_donor_margin = donor_margins(
                        true, c8_donors, cell_logits[(scope, "C8", 1.0)]
                    )
                    current_donor_margin = donor_margins(
                        true, c8_donors, cell_logits[(scope, condition, scale)]
                    )
                    result["donor_margin_change_from_scale_1"] = numeric_summary(
                        current_donor_margin - baseline_donor_margin
                    )

    for record in records:
        index = sample_positions[str(record["sample_id"])]
        scope = str(record["scope"])
        condition = str(record["condition"])
        scale = float(record["scale"])
        baseline = cell_predictions[(scope, condition, 1.0)][index]
        current = cell_predictions[(scope, condition, scale)][index]
        clean = cell_predictions[(scope, "C1", 1.0)][index]
        target = true[index]
        record["clean_correct"] = int(clean == target)
        record["attack_success_from_clean"] = int(clean == target and current != target)
        record["recovered_from_scale_1"] = int(baseline != target and current == target)
        record["regressed_from_scale_1"] = int(baseline == target and current != target)
        record["targeted_donor_success"] = int(
            condition == "C8" and clean == target and current == int(record["donor"])
        )

    predictions_path = Path(args.predictions_out)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    scores_path = (
        Path(args.scores_out)
        if args.scores_out
        else predictions_path.with_name(f"{predictions_path.stem}.scores.npz")
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    score_arrays: dict[str, np.ndarray] = {}
    for (scope, condition, scale), values in cell_logits.items():
        prefix = "" if args.scan_axis == "none" else f"{scope}_"
        suffix = f"{prefix}{condition}_scale_{scale:g}".replace(".", "p")
        score_arrays[f"logits_{suffix}"] = values
        score_arrays[f"probabilities_{suffix}"] = cell_probabilities[(scope, condition, scale)]
    np.savez_compressed(
        scores_path,
        sample_ids=np.asarray([row.sample_id for row in rows]),
        true=np.asarray(true, dtype=np.int64),
        donor=np.asarray([donors[row.sample_id].label for row in rows], dtype=np.int64),
        **score_arrays,
    )
    output = {
        "task": "G8 attention intervention dose response",
        "status": "exploratory" if not all(row.flow_id for row in rows) else "flow_identified",
        "n": len(rows),
        "prefilter_n": prefilter_n,
        "field_visibility": 1.0,
        "flow_proxy": args.flow_proxy,
        "target": args.target,
        "layers": layers,
        "heads": heads,
        "scan_axis": args.scan_axis,
        "selection_scopes": [
            {"scope": scope, "layers": scope_layers, "heads": scope_heads}
            for scope, scope_layers, scope_heads in scopes
        ],
        "attention_statistics": args.collect_attention_stats,
        "scales": scales,
        "results": results,
        "predictions_path": str(predictions_path),
        "scores_path": str(scores_path),
        "scores_sha256": sha256_file(scores_path),
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
