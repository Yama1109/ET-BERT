"""M2: 学習済み ET-BERT から attention map を抽出し Sink を計測する（asnfm 研究スクリプト）.

run_classifier.py のモデル構築・データ読込をそのまま再利用し，推論時に各
MultiHeadedAttention 層の softmax 出力（multi_headed_attn.py の read-only 計装
`self.last_probs`）を集めて [L, H, T, T] を作る．挙動は変えない（観測のみ）．
Sink 指標は本研究の正準実装 asnfm.metrics.sink に委ねる．
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
from uer.utils.config import load_hyperparam  # noqa: E402
from uer.utils.seed import set_seed  # noqa: E402
from uer.opts import finetune_opts  # noqa: E402
from uer.layers.multi_headed_attn import MultiHeadedAttention  # noqa: E402
from run_classifier import Classifier, count_labels_num, read_dataset  # noqa: E402

from asnfm.metrics.sink import anchor_ratio, attention_entropy, sink_mass, sink_rate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    finetune_opts(parser)
    parser.add_argument("--pooling", default="first")
    parser.add_argument("--tokenizer", default="bert")
    parser.add_argument("--soft_targets", action="store_true")  # Classifier が参照
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--load_model_path", required=True, help="学習済み（fine-tuned）model")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--sink_window", type=int, default=32, help="先頭/末尾の Sink 候補幅")
    parser.add_argument("--attn_out", required=True)
    parser.add_argument("--plot_out", required=True)
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

    dataset = read_dataset(args, args.test_path)[: args.num_samples]
    if not dataset:
        raise SystemExit("test データが空")
    src = torch.LongTensor([e[0] for e in dataset]).to(device)
    seg = torch.LongTensor([e[2] for e in dataset]).to(device)

    # 構築順 = 層順で attention モジュールを収集
    attn_modules = [m for m in model.modules() if isinstance(m, MultiHeadedAttention)]
    with torch.no_grad():
        model(src, None, seg)

    # 各層 last_probs [B, H, T, T] を batch 平均 → [L, H, T, T]
    attn = np.stack([m.last_probs.mean(dim=0).cpu().numpy() for m in attn_modules], axis=0)

    for p in (args.attn_out, args.plot_out, args.metrics_out):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    np.save(args.attn_out, attn)

    T = attn.shape[-1]
    w = min(args.sink_window, T - 1)
    head_pos = list(range(w))
    tail_pos = list(range(T - w, T))
    metrics = {
        "shape": list(attn.shape),
        "sink_mass_head": sink_mass(attn, head_pos),
        "sink_mass_tail": sink_mass(attn, tail_pos),
        "sink_rate_first": sink_rate(attn, 0, 0.3),
        "sink_rate_last": sink_rate(attn, T - 1, 0.3),
        "attention_entropy": attention_entropy(attn),
        "anchor_ratio_per_layer": [round(x, 4) for x in anchor_ratio(attn, head_pos, tail_pos)],
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    scalar = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
    print(f"[extract] L,H,T={attn.shape[:3]} samples={len(dataset)}")
    print(f"[extract] sink: {json.dumps(scalar, ensure_ascii=False)}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_attn = attn.mean(axis=(0, 1))  # [T, T] 全層・全 head 平均
    plt.figure(figsize=(6, 5))
    plt.imshow(mean_attn, aspect="auto", cmap="viridis")
    plt.colorbar(label="attention weight")
    plt.xlabel("key position (byte)")
    plt.ylabel("query position (byte)")
    plt.title("ET-BERT mean attention (all layers/heads)")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=120)
    print(f"[extract] saved: {args.attn_out}, {args.plot_out}, {args.metrics_out}")


if __name__ == "__main__":
    main()
