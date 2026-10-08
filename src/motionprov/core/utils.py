from pathlib import Path

import torch


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device_str)
        print(f"Using device: {device}")

    return device


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    meta: dict | None = None,
    loss_name: str = 'val_total'
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        # Which metric 'loss' holds: best_by_rec.pt records val reconstruction
        # while the other checkpoints record total val loss - at
        # lambda_rec=0.001 they differ by orders of magnitude, so a ranking
        # that mixes them without this field is meaningless.
        'loss_name': loss_name,
    }
    # Adam state triples checkpoint size (1.3 GB -> 4 GB for this model). It is
    # only needed to RESUME training; the lambda sweep never resumes.
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    # Architecture knobs that change tensor SHAPES must travel with the
    # weights: code_dim swaps the codebook between 512x32 and 512x46080 and
    # adds proj_in/proj_out, so a loader that does not know it cannot build a
    # matching module.
    if meta:
        checkpoint['meta'] = dict(meta)

    temp_path = path.parent / f"{path.stem}_temp{path.suffix}"

    try:
        torch.save(checkpoint, temp_path)
        # rename() already replaces atomically on POSIX; unlinking first only
        # opens a window where no checkpoint exists at all.
        temp_path.replace(path)
    except Exception as error:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as cleanup_error:
            error.add_note(f"Temporary checkpoint cleanup failed: {cleanup_error!r}")
        raise


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
):
    if optimizer is not None:
        # Resume is NOT supported; fail loudly BEFORE touching the model.
        raise NotImplementedError(
            "Resuming training from a checkpoint is not supported. Train "
            "from scratch, or implement true resume (optimizer + EMA + "
            "data-init state) first.")

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # A loaded codebook is trained by definition: without this stamp any
    # caller that flips the model to train() would silently re-seed the
    # trained codebook (initialized is a non-persistent buffer).
    quantizer = getattr(model, 'quantizer', None)
    if quantizer is not None and hasattr(quantizer, 'initialized'):
        quantizer.initialized.fill_(1)

    return checkpoint['epoch'], checkpoint['loss']
