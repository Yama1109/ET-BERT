"""M3/M4: 学習済み ET-BERT に random byte padding 攻撃を掛け精度を測る（asnfm 研究スクリプト）.

run_classifier.py の Classifier・tokenizer・トークン化手順をそのまま再利用し，推論のみで
clean と padding 注入後の test 精度を比較する（学習は行わない）．padding は AdvTraffic 公式
``adversal/paddingTo1500.py`` と同じ text_a（hex 文字列）層に注入する（asnfm.attack.text_padding，
公式準拠・単体テスト済）．

1 回の実行で baseline（n_pad=0）と複数の付加量を sweep し，position（pre=M3 / post=M4）は
config で切り替える．byte 内容を random に固定して位置だけを変える対照実験＝先頭/末尾どちらの
付加が精度を下げるか（位置仮説）を検証する．AdvTraffic の RL 最適化攻撃そのものの再現ではない
（RL 再現は Phase 2.A で公式コードを直接実行）．
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # fork root（uer を import）
sys.path.insert(0, str(_HERE.parents[0]))  # fine-tuning（run_classifier を import）
sys.path.insert(0, str(_HERE.parents[3]))  # repo root（asnfm を import）

from uer.utils import str2tokenizer  # noqa: E402
from uer.utils.constants import CLS_TOKEN  # noqa: E402
from uer.utils.config import load_hyperparam  # noqa: E402
from uer.utils.seed import set_seed  # noqa: E402
from uer.opts import finetune_opts  # noqa: E402
from run_classifier import Classifier, count_labels_num  # noqa: E402

from asnfm.attack.text_padding import pad_text_a  # noqa: E402  公式準拠の text_a 注入


def read_labeled_text(path):
    """test TSV を (label, text_a) の列で読む（公式 read_dataset と同じ列解釈）."""
    rows, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, name in enumerate(line.strip().split("\t")):
                    columns[name] = i
                continue
            line = line[:-1].split("\t")
            rows.append((int(line[columns["label"]]), line[columns["text_a"]]))
    return rows


def tokenize_to_src_seg(args, text_a):
    """公式 read_dataset と同一のトークン化・truncate・右 PAD（text_a 単文用）."""
    src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
    seg = [1] * len(src)
    if len(src) > args.seq_length:
        src = src[: args.seq_length]
        seg = seg[: args.seq_length]
    while len(src) < args.seq_length:
        src.append(0)
        seg.append(0)
    return src, seg


def evaluate_accuracy(args, model, device, rows, n_pad, position, seed):
    """rows に n_pad の padding を注入して推論し accuracy を返す."""
    rng = np.random.default_rng(seed)
    src_list, seg_list, gold = [], [], []
    for label, text_a in rows:
        padded = pad_text_a(text_a, n_pad, position, rng=rng) if n_pad > 0 else text_a
        src, seg = tokenize_to_src_seg(args, padded)
        src_list.append(src)
        seg_list.append(seg)
        gold.append(label)
    src = torch.LongTensor(src_list)
    seg = torch.LongTensor(seg_list)
    tgt = torch.LongTensor(gold)

    correct, total = 0, src.size(0)
    bs = args.batch_size
    model.eval()
    with torch.no_grad():
        for i in range(0, total, bs):
            sb = src[i : i + bs].to(device)
            gb = seg[i : i + bs].to(device)
            _, logits = model(sb, None, gb)
            pred = torch.argmax(logits, dim=1).cpu()
            correct += int((pred == tgt[i : i + bs]).sum())
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    finetune_opts(parser)
    parser.add_argument("--pooling", default="first")
    parser.add_argument("--tokenizer", default="bert")
    parser.add_argument("--soft_targets", action="store_true")
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--load_model_path", required=True, help="学習済み（fine-tuned）model")
    parser.add_argument("--position", choices=["pre", "post"], default="pre",
                        help="pre=M3（先頭付加）/ post=M4（末尾付加・対照）")
    parser.add_argument("--n_pads", default="0,8,16,32,48",
                        help="付加量の sweep（カンマ区切り）．0=baseline")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="評価サンプル数の上限（0=全件）")
    parser.add_argument("--metrics_out", required=True)
    args = parser.parse_args()

    args = load_hyperparam(args)
    set_seed(args.seed)
    args.labels_num = count_labels_num(args.train_path)
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = Classifier(args)
    model.load_state_dict(torch.load(args.load_model_path, map_location="cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    rows = read_labeled_text(args.test_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit("test データが空")

    n_pads = [int(x) for x in str(args.n_pads).split(",") if x.strip() != ""]
    results = []
    for n_pad in n_pads:
        acc = evaluate_accuracy(args, model, device, rows, n_pad, args.position, args.seed)
        results.append({"n_pad": n_pad, "accuracy": acc})
        print(f"[attack] position={args.position} n_pad={n_pad:>4} acc={acc:.4f}", flush=True)

    baseline = next((r["accuracy"] for r in results if r["n_pad"] == 0), None)
    for r in results:
        r["acc_drop_vs_baseline"] = (
            None if baseline is None else round(baseline - r["accuracy"], 4)
        )
    metrics = {
        "task": "M3/M4 random-byte padding attack",
        "injection_layer": "text_a (hex string, AdvTraffic paddingTo1500.py 準拠)",
        "position": args.position,
        "n_eval": len(rows),
        "seq_length": args.seq_length,
        "baseline_accuracy": baseline,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[attack] saved: {args.metrics_out}", flush=True)


if __name__ == "__main__":
    main()
