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

from asnfm.metrics.bai import (  # noqa: E402  公式準拠（Bai et al. COLM 2024）
    attention_deviation,
    degree_profile,
    outer_degree,
)


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

    # ===== Bai et al. (COLM 2024) 公式準拠の Sink 計測（padding は mask で除外）=====
    B = src_np.shape[0]
    profiles, layer_deg0, devs, sink_positions = [], [], [], []
    for b in range(B):
        n = int(real_len[b])
        if n < 4:
            continue
        mask_b = (src_np[b] != args.pad_id).astype(np.float64)  # [T] 1=実トークン
        attn_b = np.stack([lp[b] for lp in layer_probs], axis=0)  # [L, H, T, T]
        prof = degree_profile(attn_b, mask_b)  # [T] 位置別 outer degree（padding=nan）
        profiles.append(prof)
        deg = outer_degree(attn_b, mask_b)  # [L, H, T]
        layer_deg0.append(deg[:, :, 0].mean(axis=1))  # [L] 位置0(CLS) の degree
        sp = int(np.nanargmax(prof))
        sink_positions.append(sp)
        devs.append(float(attention_deviation(attn_b, mask_b)[:, :, sp].mean()))

    with np.errstate(invalid="ignore"):  # padding 末尾位置の all-nan 平均を無視
        prof_mean = np.nanmean(np.array(profiles), axis=0)  # [T]
        layer_deg0_mean = np.nanmean(np.array(layer_deg0), axis=0)  # [L]
    mean_real = float(np.mean(real_len))
    uniform = 1.0 / mean_real  # 一様時の outer degree（基準）
    sink_pos = int(np.nanargmax(prof_mean))
    metrics = {
        "metric_source": "Bai et al. COLM 2024 (attention-sink-cl: outer degree + attention deviation)",
        "L_H": [len(layer_probs), int(layer_probs[0].shape[1])],
        "n_samples_used": len(profiles),
        "mean_real_len": mean_real,
        "uniform_degree": uniform,
        "sink_position": sink_pos,  # outer degree 最大の位置（0=CLS かどうかで CLS vs ヘッダを判別）
        "sink_position_mode": int(np.bincount(sink_positions).argmax()),
        "sink_degree": float(prof_mean[sink_pos]),
        "sink_over_uniform": float(prof_mean[sink_pos]) / uniform,  # 一様の何倍＝sink 強度
        "degree_pos0_cls": float(prof_mean[0]),
        "pos0_over_uniform": float(prof_mean[0]) / uniform,
        "sink_deviation": float(np.mean(devs)),  # 低いほど一様＝強い sink
        "degree_profile_first16": [round(float(x), 4) for x in prof_mean[:16]],
        "outer_degree_pos0_per_layer": [round(float(x), 4) for x in layer_deg0_mean],
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[extract] L,H={metrics['L_H']} samples={metrics['n_samples_used']} "
          f"mean_real_len={mean_real:.1f}  uniform_degree={uniform:.4f}")
    print(f"[extract] sink_position={sink_pos} (0=CLS)  sink_over_uniform="
          f"{metrics['sink_over_uniform']:.2f}x  pos0_over_uniform={metrics['pos0_over_uniform']:.2f}x  "
          f"deviation={metrics['sink_deviation']:.3f}")
    print(f"[extract] degree_profile[:16]: {metrics['degree_profile_first16']}")
    print(f"[extract] outer_degree(pos0)/layer: {metrics['outer_degree_pos0_per_layer']}")

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
