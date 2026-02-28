import os
import time
from contextlib import nullcontext
from typing import Dict, Optional

import torch
from diffusers import DDIMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


def make_celeba_loader(
    root: str,
    split: str,
    height: int,
    width: int,
    batch_size: int,
    device: str,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.CelebA(
        root=root,
        split=split,
        download=True,
        transform=transform,
    )
    num_workers = min(8, (os.cpu_count() or 2) // 2)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )


def add_real_images(fid_metric: FrechetInceptionDistance, real_loader: DataLoader, device: str, n_real: int) -> None:
    count = 0
    for x, _ in real_loader:
        x = x.to(device, non_blocking=True)
        fid_metric.update(x, real=True)
        count += x.shape[0]
        if count >= n_real:
            break


def real_stats_cache_path(cache_dir: str, fid_feature: int, num_real: int, height: int, width: int) -> str:
    return os.path.join(
        cache_dir,
        f"celeba_real_stats_f{fid_feature}_n{num_real}_{height}x{width}.pt",
    )


def state_num_samples(state: Dict, key: str) -> int:
    v = state.get(key, None)
    if v is None:
        return 0
    if torch.is_tensor(v):
        return int(v.item())
    return int(v)


def metric_num_samples(metric: FrechetInceptionDistance, key: str) -> int:
    v = getattr(metric, key, None)
    if v is None:
        return 0
    if torch.is_tensor(v):
        return int(v.item())
    return int(v)


def load_or_compute_real_stats(
    cache_dir: str,
    fid_feature: int,
    num_real: int,
    height: int,
    width: int,
    device: str,
    real_loader: DataLoader,
) -> Dict:
    path = real_stats_cache_path(cache_dir, fid_feature, num_real, height, width)
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        if state_num_samples(state, "real_features_num_samples") >= 2:
            return state
        # stale/invalid cache

    fid = FrechetInceptionDistance(feature=fid_feature, normalize=True).to(device)
    add_real_images(fid, real_loader, device, num_real)
    state = {k: v.cpu() for k, v in fid.state_dict().items()}
    torch.save(state, path)
    return state


def init_fid_metric(
    fid_feature: int,
    device: str,
    real_state: Dict,
    real_loader: DataLoader,
    num_real: int,
) -> FrechetInceptionDistance:
    fid = FrechetInceptionDistance(feature=fid_feature, normalize=True).to(device)
    fid.load_state_dict(real_state)
    if metric_num_samples(fid, "real_features_num_samples") < 2:
        add_real_images(fid, real_loader, device, num_real)
    return fid


def init_fid_metrics(
    step_count: int,
    fid_feature: int,
    device: str,
    real_state: Dict,
    real_loader: DataLoader,
    num_real: int,
) -> Dict[int, FrechetInceptionDistance]:
    fids = {}
    for k in range(1, step_count + 1):
        fids[k] = init_fid_metric(fid_feature, device, real_state, real_loader, num_real)
    return fids


def init_results_csv(path: str) -> None:
    with open(path, "w") as f:
        f.write("step,fid,time,img_per_sec,step_per_sec,sec_per_step\n")


def append_results_csv(
    path: str,
    step: int,
    fid_score: float,
    t_sample: float,
    img_per_sec: float,
    step_per_sec: float,
    sec_per_step: float,
) -> None:
    with open(path, "a") as f:
        f.write(
            f"{step},{fid_score:.6f},{t_sample:.6f},{img_per_sec:.6f},{step_per_sec:.6f},{sec_per_step:.6f}\n"
        )


def _autocast_ctx(autocast_ctx):
    return autocast_ctx() if autocast_ctx else nullcontext()


def sample_with_scheduler(
    unet,
    scheduler,
    num_inference_steps: int,
    batch_size: int,
    device: str,
    in_channels: int,
    height: int,
    width: int,
    generator: Optional[torch.Generator] = None,
    autocast_ctx=None,
    eta: float = 0.0,
):
    scheduler.set_timesteps(num_inference_steps)
    init_sigma = getattr(scheduler, "init_noise_sigma", 1.0)
    x = torch.randn(
        (batch_size, in_channels, height, width),
        device=device,
        generator=generator,
    ) * init_sigma

    with _autocast_ctx(autocast_ctx):
        for t in scheduler.timesteps:
            t_int = int(t)
            x_in = scheduler.scale_model_input(x, t_int)
            t_tensor = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
            eps = unet(x_in, t_tensor).sample

            step_kwargs = {"generator": generator}
            if isinstance(scheduler, DDIMScheduler):
                step_kwargs["eta"] = eta
            x = scheduler.step(eps, t_int, x, **step_kwargs).prev_sample
    return x


def sample_stepwise_ddpm(
    unet,
    scheduler,
    batch_size: int,
    fid_map: Dict[int, FrechetInceptionDistance],
    device: str,
    in_channels: int,
    height: int,
    width: int,
    generator: Optional[torch.Generator] = None,
    autocast_ctx=None,
    eta: float = 0.0,
) -> Dict[int, float]:
    # Assume scheduler.set_timesteps(...) already called outside
    init_sigma = getattr(scheduler, "init_noise_sigma", 1.0)
    x = torch.randn(
        (batch_size, in_channels, height, width),
        device=device,
        generator=generator,
    ) * init_sigma

    step_times = {k: 0.0 for k in range(1, len(scheduler.timesteps) + 1)}
    elapsed = 0.0
    with _autocast_ctx(autocast_ctx):
        for i, t in enumerate(scheduler.timesteps, start=1):
            sync()
            t0 = time.perf_counter()
            t_int = int(t)
            x_in = scheduler.scale_model_input(x, t_int)
            t_tensor = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
            eps = unet(x_in, t_tensor).sample

            step_kwargs = {"generator": generator}
            if isinstance(scheduler, DDIMScheduler):
                step_kwargs["eta"] = eta
            x = scheduler.step(eps, t_int, x, **step_kwargs).prev_sample
            sync()
            elapsed += time.perf_counter() - t0

            step_times[i] += elapsed
            # FID update after timing; next step sync() prevents overlap
            fid_map[i].update(to_01(x), real=False)
    return step_times


def sample_stepwise_fors(
    fors_sampler,
    timesteps,
    sigma_start: torch.Tensor,
    batch_size: int,
    fid_map: Dict[int, FrechetInceptionDistance],
    device: str,
    in_channels: int,
    height: int,
    width: int,
    generator: Optional[torch.Generator] = None,
    autocast_ctx=None,
) -> Dict[int, float]:
    x = torch.randn(
        (batch_size, in_channels, height, width),
        device=device,
        generator=generator,
        dtype=fors_sampler.alphas_cumprod.dtype,
    ) * sigma_start

    step_times = {k: 0.0 for k in range(1, len(timesteps))}
    elapsed = 0.0
    with _autocast_ctx(autocast_ctx):
        for i in range(len(timesteps) - 1):
            sync()
            t0 = time.perf_counter()
            t_cur = int(timesteps[i])
            t_prev = int(timesteps[i + 1])
            x = fors_sampler._fors_step(x, t_cur, t_prev)
            sync()
            elapsed += time.perf_counter() - t0

            step_idx = i + 1
            step_times[step_idx] += elapsed
            fid_map[step_idx].update(to_01(x), real=False)
    return step_times
