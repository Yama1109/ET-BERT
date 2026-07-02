"""M3/M4/C2/C8: 学習済み ET-BERT に攻撃を掛け精度を測る（asnfm 研究スクリプト）.

run_classifier.py の Classifier・tokenizer・トークン化手順をそのまま再利用し，推論のみで
clean と攻撃後の test 精度を比較する（学習は行わない）．

対応する攻撃モード（``--position`` で切り替え）:
  pre     (M3): 先頭 random byte 付加．byte 内容を random に固定して位置の効果を測る
  post    (M4): 末尾 random byte 付加（M3 との対照）
  rewrite (C2): TCP ヘッダ場（seq/ack/window/urgent）の値書換のみ（挿入なし）．
               n_pad=0=baseline，n_pad=1=書換適用．
  swap    (C8): 先頭 unit を別パケットの実ヘッダ値で差し替え（乱数でなく妥当な値）．
               C2 との差が「位置 vs 値」の答えになる．
               n_pad=0=baseline，n_pad=1=スワップ適用．
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
from asnfm.attack.header_rewrite import rewrite_units, swap_units  # noqa: E402  C2/C8


def _is_tcp_sample(text_a: str) -> bool:
    """text_a byte 8 の上位 4 bit が TCP data offset の妥当域 [5, 15] かで TCP を判定する.

    text_a は ET-BERT の TCP packet_string[76:] 由来（TCP seq 先頭から）．
    byte 8 = TCP byte 12 = data offset フィールド（上位 4 bit）．
    UDP パケットでは同位置が payload バイトになるため妥当域外になりやすい．
    heuristic であり完全な判定ではないが，UDP 混在の交絡を大幅に低減できる．
    """
    units = text_a.split()
    if len(units) < 9:
        return False
    try:
        byte8 = int(units[8][:2], 16)
    except (ValueError, IndexError):
        return False
    return 5 <= (byte8 >> 4) <= 15


def read_labeled_text(path, tcp_only: bool = False):
    """test TSV を (label, text_a) の列で読む（公式 read_dataset と同じ列解釈）.

    tcp_only=True のとき，data offset heuristic で TCP と判定されたサンプルのみ返す．
    """
    rows, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, name in enumerate(line.strip().split("\t")):
                    columns[name] = i
                continue
            line = line[:-1].split("\t")
            text_a = line[columns["text_a"]]
            if tcp_only and not _is_tcp_sample(text_a):
                continue
            rows.append((int(line[columns["label"]]), text_a))
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


def evaluate_accuracy(
    args, model, device, rows, n_pad, position, seed,
    rewrite_idxs=(), donor_pool=None,
):
    """rows に攻撃を注入して推論し accuracy を返す.

    position=="rewrite": n_pad>0 なら rewrite_idxs の unit を random 書換（C2）
    position=="swap":    n_pad>0 なら donor_pool からランダムに donor を選び unit を差替（C8）
    それ以外:            n_pad>0 なら pad_text_a（M3/M4 padding 注入）
    n_pad==0 は常に baseline（無加工）
    """
    rng = np.random.default_rng(seed)
    src_list, seg_list, gold = [], [], []
    pool = donor_pool or []
    for label, text_a in rows:
        if n_pad > 0 and position == "rewrite":
            modified = rewrite_units(text_a, list(rewrite_idxs), rng=rng)
        elif n_pad > 0 and position == "swap" and pool:
            j = int(rng.integers(0, len(pool)))
            if pool[j] is text_a:
                j = (j + 1) % len(pool)
            modified = swap_units(text_a, list(rewrite_idxs), pool[j])
        elif n_pad > 0:
            modified = pad_text_a(text_a, n_pad, position, rng=rng)
        else:
            modified = text_a
        src, seg = tokenize_to_src_seg(args, modified)
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
    parser.add_argument("--position", choices=["pre", "post", "rewrite", "swap"], default="pre",
                        help="pre=M3 / post=M4 / rewrite=C2（乱数書換）/ swap=C8（donor 実値差替）")
    parser.add_argument("--n_pads", default="0,8,16,32,48",
                        help="付加量の sweep（カンマ区切り）．0=baseline．rewrite モードでは 0,1 を推奨")
    parser.add_argument("--rewrite_unit_indices", default="",
                        help="C2 用: 書き換える unit index のカンマ区切り（例 0,1,2,3,4,5,6,7,9,10,11,13,14,15）")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="評価サンプル数の上限（0=全件）")
    parser.add_argument("--tcp_only", action="store_true",
                        help="TCP heuristic（data offset byte が妥当域）のサンプルのみ評価する（C2 交絡排除用）")
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

    rows = read_labeled_text(args.test_path, tcp_only=args.tcp_only)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit("test データが空")
    if args.tcp_only:
        print(f"[attack] tcp_only=True: {len(rows)} サンプルを使用（data offset heuristic で絞り込み）", flush=True)

    rewrite_idxs = (
        [int(x) for x in args.rewrite_unit_indices.split(",") if x.strip()]
        if args.rewrite_unit_indices else []
    )

    # C8: donor プールは tcp_only フィルタ後の全 text_a（スワップ元として使う）
    donor_pool = [text_a for _, text_a in rows] if args.position == "swap" else None

    n_pads = [int(x) for x in str(args.n_pads).split(",") if x.strip() != ""]
    results = []
    for n_pad in n_pads:
        acc = evaluate_accuracy(
            args, model, device, rows, n_pad, args.position, args.seed,
            rewrite_idxs, donor_pool,
        )
        results.append({"n_pad": n_pad, "accuracy": acc})
        print(f"[attack] position={args.position} n_pad={n_pad:>4} acc={acc:.4f}", flush=True)

    baseline = next((r["accuracy"] for r in results if r["n_pad"] == 0), None)
    for r in results:
        r["acc_drop_vs_baseline"] = (
            None if baseline is None else round(baseline - r["accuracy"], 4)
        )
    task_label = {
        "rewrite": "C2 TCP header-field rewrite (seq/ack/window/urgent)",
        "swap":    "C8 TCP header-field swap with donor packet",
    }.get(args.position, "M3/M4 random-byte padding attack")
    metrics = {
        "task": task_label,
        "injection_layer": "text_a (hex string, AdvTraffic paddingTo1500.py 準拠)",
        "position": args.position,
        "rewrite_unit_indices": rewrite_idxs if rewrite_idxs else None,
        "tcp_only": args.tcp_only,
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
