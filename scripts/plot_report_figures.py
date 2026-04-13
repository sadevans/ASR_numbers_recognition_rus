from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / 'runs' / 'ctc_baseline' / 'metrics.jsonl'
OUT_DIR = ROOT / 'report' / 'figures'


def load_epoch_rows(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{"epoch":'):
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r['epoch'])
    return rows


def main():
    rows = load_epoch_rows(METRICS)
    if not rows:
        raise SystemExit(f'Нет эпох в {METRICS}')

    epochs = [r['epoch'] for r in rows]
    train_loss = [r['train_loss'] for r in rows]
    val_loss = [r['val_loss'] for r in rows]
    cer_mean = [r['val_cer/mean'] for r in rows]
    cer_in = [r['val_cer/in_domain'] for r in rows]
    cer_out = [r['val_cer/out_of_domain'] for r in rows]
    cer_hmean = [r['val_cer/hmean_in_ood'] for r in rows]
    seq_acc = [r['val_seq_accuracy'] for r in rows]
    cer_f = [r['val_cer/gender/female'] for r in rows]
    cer_m = [r['val_cer/gender/male'] for r in rows]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            'font.size': 10,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'figure.dpi': 120,
            'savefig.dpi': 150,
        }
    )

    # --- Рисунок 1: потери ---
    fig1, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, 'o-', label='train loss', color='#2563eb')
    ax.plot(epochs, val_loss, 's-', label='val loss (CTC)', color='#dc2626')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Loss')
    ax.set_title('Функция потерь по эпохам')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs[::2])
    fig1.tight_layout()
    p1 = OUT_DIR / 'loss_train_val.png'
    fig1.savefig(p1, bbox_inches='tight')
    plt.close(fig1)

    # --- Рисунок 2: CER ---
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, cer_mean, 'o-', label='CER mean (dev)', color='#0f766e', linewidth=2)
    ax.plot(epochs, cer_in, '--', label='CER in-domain', alpha=0.85)
    ax.plot(epochs, cer_out, '--', label='CER out-of-domain', alpha=0.85)
    ax.plot(epochs, cer_hmean, ':', label='CER hmean in/ood', alpha=0.9)
    best_e = min(rows, key=lambda r: r['val_cer/mean'])['epoch']
    best_cer = min(cer_mean)
    ax.axvline(best_e, color='#94a3b8', linestyle='--', linewidth=1, label=f'лучшая эпоха ({best_e})')
    ax.scatter([best_e], [best_cer], color='#b45309', s=80, zorder=5, label=f'min CER={best_cer:.3f}')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('CER')
    ax.set_title('Character Error Rate на dev')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs[::2])
    fig2.tight_layout()
    p2 = OUT_DIR / 'cer_dev.png'
    fig2.savefig(p2, bbox_inches='tight')
    plt.close(fig2)

    # --- Рисунок 3: точность и пол ---
    fig3, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax_a.plot(epochs, seq_acc, 'o-', color='#7c3aed')
    ax_a.set_xlabel('Эпоха')
    ax_a.set_ylabel('Точность')
    ax_a.set_title('Точное совпадение строки цифр (dev)')
    ax_a.set_ylim(0.35, 1.0)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xticks(epochs[::2])

    ax_b.plot(epochs, cer_f, 'o-', label='female', color='#db2777')
    ax_b.plot(epochs, cer_m, 's-', label='male', color='#0369a1')
    ax_b.set_xlabel('Эпоха')
    ax_b.set_ylabel('CER')
    ax_b.set_title('CER по полу (dev)')
    ax_b.legend()
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xticks(epochs[::2])
    fig3.tight_layout()
    p3 = OUT_DIR / 'accuracy_gender.png'
    fig3.savefig(p3, bbox_inches='tight')
    plt.close(fig3)

    print('Saved:', p1, p2, p3, sep='\n')


if __name__ == '__main__':
    main()
