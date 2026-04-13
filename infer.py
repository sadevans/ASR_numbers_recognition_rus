from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio

from src.char_vocab import greedy_ctc_decode
from src.metrics import cer_over_utterances
from src.model import DigitCTCModel
from src.text_normalize import denormalize_digits_for_submission, normalize_transcription


def _checkpoint_args_dict(ckpt):
    raw = ckpt.get('args')
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    return vars(raw)


def load_checkpoint(path, device):
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(ckpt, device):
    train_args = _checkpoint_args_dict(ckpt)
    use_spec_augment = not bool(train_args.get('no_spec_augment', False))
    model = DigitCTCModel(use_spec_augment=use_spec_augment).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model


def load_mono_16k(path):
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    return wav


@torch.no_grad()
def predict_digit_strings(
    model,
    paths,
    device,
):
    waves = [load_mono_16k(p) for p in paths]
    lengths = torch.tensor([w.shape[-1] for w in waves], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in waves])
    log_probs, _ = model(padded.to(device), lengths.to(device))
    t_steps, batch_size, _ = log_probs.shape
    out: list[str] = []
    for b in range(batch_size):
        idx = log_probs[:t_steps, b, :].argmax(dim=-1).detach().cpu().tolist()
        out.append(greedy_ctc_decode(idx))
    return out


def predict_submission_integers(
    model,
    paths,
    device,
):
    digit_strs = predict_digit_strings(model, paths, device)
    return [denormalize_digits_for_submission(s) for s in digit_strs]


def run_dev_sanity(
    model,
    dev_csv,
    data_root,
    device,
    batch_size: int,
):
    dev_df = pd.read_csv(dev_csv)
    refs: list[str] = []
    hyps: list[str] = []
    for start in range(0, len(dev_df), batch_size):
        chunk = dev_df.iloc[start : start + batch_size]
        paths = [data_root / str(fn) for fn in chunk['filename']]
        digit_hyps = predict_digit_strings(model, paths, device)
        for raw_label, hyp_digits in zip(chunk['transcription'], digit_hyps):
            _, ref_digits = normalize_transcription(str(raw_label), 'digits')
            refs.append(ref_digits)
            hyps.append(hyp_digits)
    cer = cer_over_utterances(refs, hyps)
    print(f'[infer] dev CER (digit strings): {cer:.4f}')


def run_test_submission(
    model,
    test_csv,
    data_root,
    output_csv,
    device,
    batch_size: int,
) -> None:
    test_df = pd.read_csv(test_csv)
    pred_ints: list[int] = []
    for start in range(0, len(test_df), batch_size):
        chunk = test_df.iloc[start : start + batch_size]
        paths = [data_root / str(fn) for fn in chunk['filename']]
        pred_ints.extend(predict_submission_integers(model, paths, device))
    out_df = test_df.copy()
    out_df['transcription'] = pred_ints
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df[['filename', 'transcription']].to_csv(output_csv, index=False)
    print(f'[infer] saved {output_csv} ({len(pred_ints)} rows)')


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description='Инференс DigitCTCModel из чекпоинта train.py')
    p.add_argument(
        '--checkpoint',
        type=Path,
        default=root / 'checkpoints' / 'ctc_baseline' / 'best.pt',
        help='Путь к .pt с model_state_dict и args',
    )
    p.add_argument('--device', type=str, default=None, help='cuda / cpu (по умолчанию авто)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument(
        '--audio',
        type=Path,
        nargs='*',
        default=(),
        help='Один или несколько wav/flac и т.п.; печать digit-строки и int для сабмита',
    )
    p.add_argument('--test-csv', type=Path, default=None, help='test.csv — сформировать submission')
    p.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help='Корень: пути = data-root / filename из CSV',
    )
    p.add_argument(
        '--output',
        type=Path,
        default=Path('submission.csv'),
        help='Куда сохранить submission (--test-csv)',
    )
    p.add_argument(
        '--eval-dev-csv',
        type=Path,
        default=None,
        help='Опционально: посчитать CER на dev.csv (нужен --data-root)',
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f'[infer] нет файла чекпоинта: {ckpt_path}', file=sys.stderr)
        sys.exit(1)

    ckpt = load_checkpoint(ckpt_path, device)
    model = build_model(ckpt, device)
    print(
        '[infer] checkpoint:',
        ckpt_path,
        'epoch',
        ckpt.get('epoch'),
        'val_cer_mean',
        ckpt.get('val_cer_mean'),
        'device:',
        device,
    )

    if args.audio:
        paths = [Path(a) for a in args.audio]
        for pth in paths:
            if not pth.is_file():
                print(f'[infer] нет файла: {pth}', file=sys.stderr)
                sys.exit(1)
        digit_strs = predict_digit_strings(model, paths, device)
        ints = [denormalize_digits_for_submission(s) for s in digit_strs]
        for pth, ds, n in zip(paths, digit_strs, ints):
            print(f'{pth}\tdigits={ds!r}\tsubmission_int={n}')

    need_root = args.test_csv is not None or args.eval_dev_csv is not None
    if need_root and args.data_root is None:
        print('[infer] для --test-csv / --eval-dev-csv укажите --data-root', file=sys.stderr)
        sys.exit(1)
    if args.data_root is not None and not Path(args.data_root).is_dir():
        print(f'[infer] data-root не каталог: {args.data_root}', file=sys.stderr)
        sys.exit(1)

    data_root = Path(args.data_root) if args.data_root else None

    if args.eval_dev_csv is not None:
        if data_root is None:
            sys.exit(1)
        run_dev_sanity(model, Path(args.eval_dev_csv), data_root, device, args.batch_size)

    if args.test_csv is not None:
        if data_root is None:
            sys.exit(1)
        run_test_submission(
            model,
            Path(args.test_csv),
            data_root,
            Path(args.output),
            device,
            args.batch_size,
        )

    if not args.audio and args.test_csv is None and args.eval_dev_csv is None:
        print(
            '[infer] укажите --audio … и/или --test-csv … / --eval-dev-csv …',
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
