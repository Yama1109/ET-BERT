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
    parser.add_argument("--pad_id", type=int, default=0, help="padding トークン ID（ET-BERT は 0）")
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

    # 各層 last_probs [B, H, T, T]（batch 次元を保持）
    layer_probs = [m.last_probs.cpu().numpy() for m in attn_modules]  # 各 [B, H, T, T]
    src_np = src.cpu().numpy()
    real_len = (src_np != args.pad_id).sum(axis=1)  # [B] 各サンプルの実トークン長（CLS + 実バイト）

    for p in (args.attn_out, args.plot_out, args.metrics_out):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    # 生 attention は batch 平均 [L, H, T, T] で保存（解析用）
    np.save(args.attn_out, np.stack([lp.mean(axis=0) for lp in layer_probs], axis=0))

    # padding を除外し，各サンプルの実トークン領域 [L, H, n, n] で Sink を計測してスカラ平均する．
    # tail は「末尾 32 byte」ではなく「実トークン末尾 32」を使う（padding を末尾と誤認しない）．
    mh, mt, ent, srf, srl, anchors = [], [], [], [], [], []
    B = src_np.shape[0]
    for b in range(B):
        n = int(real_len[b])
        if n < 4:
            continue
        attn_b = np.stack([lp[b, :, :n, :n] for lp in layer_probs], axis=0)  # [L, H, n, n]
        w = min(args.sink_window, n - 1)
        head_pos = list(range(w))
        tail_pos = list(range(n - w, n))
        mh.append(sink_mass(attn_b, head_pos))
        mt.append(sink_mass(attn_b, tail_pos))
        ent.append(attention_entropy(attn_b))
        srf.append(sink_rate(attn_b, 0, 0.3))
        srl.append(sink_rate(attn_b, n - 1, 0.3))  # 実トークン末尾
        anchors.append(anchor_ratio(attn_b, head_pos, tail_pos))  # [L]

    anchor_layer = np.nanmean(np.array(anchors), axis=0)  # [L] 層別 anchor 比率（サンプル平均）
    metrics = {
        "L_H": [len(layer_probs), int(layer_probs[0].shape[1])],
        "n_samples_used": len(mh),
        "mean_real_len": float(np.mean(real_len)),
        "sink_window": args.sink_window,
        "sink_mass_head": float(np.mean(mh)),
        "sink_mass_tail": float(np.mean(mt)),  # 実トークン末尾（padding 除外）
        "sink_rate_first": float(np.mean(srf)),
        "sink_rate_last": float(np.mean(srl)),
        "attention_entropy": float(np.mean(ent)),
        "anchor_ratio_per_layer": [round(float(x), 4) for x in anchor_layer],
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    scalar = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
    print(f"[extract] L,H={metrics['L_H']} samples={metrics['n_samples_used']} "
          f"mean_real_len={metrics['mean_real_len']:.1f}")
    print(f"[extract] sink (padding 除外): {json.dumps(scalar, ensure_ascii=False)}")
    print(f"[extract] anchor_ratio/layer (先頭優勢→1, 末尾優勢→0): {metrics['anchor_ratio_per_layer']}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # padding ブロックが図を占有しないよう content 領域に crop（実トークン長中央値）．
    plot_len = int(min(np.median(real_len), layer_probs[0].shape[-1]))
    plot_len = max(plot_len, 8)
    mean_attn = np.stack([lp.mean(axis=(0, 1)) for lp in layer_probs], axis=0).mean(axis=0)
    mean_attn = mean_attn[:plot_len, :plot_len]  # [plot_len, plot_len] 全層・全 head 平均
    plt.figure(figsize=(6, 5))
    plt.imshow(mean_attn, aspect="auto", cmap="viridis")
    plt.colorbar(label="attention weight")
    plt.xlabel("key position (byte)")
    plt.ylabel("query position (byte)")
    plt.title(f"ET-BERT mean attention (all layers/heads, first {plot_len} pos)")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=120)
    print(f"[extract] saved: {args.attn_out}, {args.plot_out}, {args.metrics_out}")


if __name__ == "__main__":
    main()
