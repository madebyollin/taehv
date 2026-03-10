"""
Test script for streaming decoder performance and correctness.
Usage: python test_decoder.py
"""
import torch
import time
from taehv import TAEHV, StreamingTAEHV, StreamingOptDecoder

CHECKPOINT = "taehv1_5.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16

N_LATENT_FRAMES = 16   # T dimension of latents
LATENT_H, LATENT_W = 32, 64
N_WARMUP = 2
N_EVAL = 5

taehv = TAEHV(checkpoint_path=CHECKPOINT).to(DEVICE, DTYPE)
taehv.eval()
streaming = StreamingTAEHV(taehv)
opt_streaming = StreamingOptDecoder(taehv, device=DEVICE, dtype=DTYPE).to(DEVICE, DTYPE)
opt_streaming.eval()


def make_latents():
    return torch.randn(1, N_LATENT_FRAMES, taehv.latent_channels, LATENT_H, LATENT_W, device=DEVICE, dtype=DTYPE)


def decode_streaming(latents):
    """Decode latents using StreamingTAEHV. Returns list of N1CHW frames."""
    streaming.reset()
    frames = []
    for t in range(latents.shape[1]):
        lat = latents[:, t:t+1]
        frame = streaming.decode(lat)
        if frame is not None:
            frames.append(frame)
        while True:
            frame = streaming.decode()
            if frame is None:
                break
            frames.append(frame)
    frames.extend(streaming.flush_decoder())
    return frames


def decode_opt(latents):
    """Decode latents using StreamingOptDecoder. Returns list of N4CHW frame chunks."""
    opt_streaming.reset()
    frames = []
    for t in range(latents.shape[1]):
        lat = latents[:, t:t+1]
        chunk = opt_streaming.decode(lat)  # [4, C, H, W]
        # split into individual frames to match streaming format
        for i in range(chunk.shape[0]):
            frames.append(chunk[i:i+1].unsqueeze(1))
    return frames


def compute_deviation(frames_a, frames_b):
    """Mean absolute deviation between two lists of N1CHW frames."""
    a = torch.cat(frames_a, dim=1).float()
    b = torch.cat(frames_b, dim=1).float()
    return (a - b).abs().mean().item()


def benchmark(decode_fn, label):
    print(f"\n--- {label} ---")

    # Warmup
    print(f"Running {N_WARMUP} warmup calls...")
    for _ in range(N_WARMUP):
        with torch.inference_mode():
            decode_fn(make_latents())

    # Eval
    print(f"Running {N_EVAL} eval calls...")
    fps_list = []
    latency_list = []
    ref_frames = None
    ref_latents = None

    for i in range(N_EVAL):
        latents = make_latents()
        with torch.inference_mode():
            t0 = time.perf_counter()
            frames = decode_fn(latents)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        n_latents = latents.shape[1]
        fps = n_latents / elapsed
        latency = elapsed / n_latents
        fps_list.append(fps)
        latency_list.append(latency)

        if i == 0:
            ref_frames = frames
            ref_latents = latents

        print(f"  Run {i+1}: {n_latents} latents in {elapsed:.3f}s → {fps:.2f} latents/s, {latency*1000:.2f}ms/latent")

    avg_fps = sum(fps_list) / len(fps_list)
    avg_latency = sum(latency_list) / len(latency_list)
    print(f"Average latents/s: {avg_fps:.2f}")
    print(f"Average latency:   {avg_latency*1000:.2f}ms/latent")

    return avg_fps, avg_latency, ref_frames, ref_latents


def run():
    print(f"Device: {DEVICE}, dtype: {DTYPE}, checkpoint: {CHECKPOINT}")
    print(f"Latent shape: [1, {N_LATENT_FRAMES}, {taehv.latent_channels}, {LATENT_H}, {LATENT_W}]")

    ref_fps, ref_latency, ref_frames, ref_latents = benchmark(decode_streaming, "StreamingTAEHV (reference)")
    opt_fps, opt_latency, opt_frames, _           = benchmark(lambda l: decode_opt(l), "StreamingOptDecoder")

    # --- Deviation ---
    print("\n--- Deviation Analysis ---")
    with torch.inference_mode():
        random_frames = decode_streaming(make_latents())

    trim = taehv.frames_to_trim
    opt_frames_trimmed = opt_frames[trim:]
    n = min(len(ref_frames), len(random_frames), len(opt_frames_trimmed))
    random_baseline = compute_deviation(ref_frames[:n], random_frames[:n])
    opt_deviation   = compute_deviation(ref_frames[:n], opt_frames_trimmed[:n])
    self_deviation  = compute_deviation(ref_frames[:n], ref_frames[:n])

    print(f"  Self-deviation (sanity, should be 0): {self_deviation:.6f}")
    print(f"  Random baseline deviation:            {random_baseline:.6f}")
    print(f"  OptimizedDecoder deviation:           {opt_deviation:.6f}")
    print()
    if opt_deviation < random_baseline * 0.1:
        print("  OptimizedDecoder output looks correct (deviation << random baseline).")
    else:
        print("  WARNING: OptimizedDecoder deviation is close to random baseline — possible bug.")

    print("\n--- Summary ---")
    print(f"  Reference:  {ref_fps:.2f} latents/s, {ref_latency*1000:.2f}ms/latent")
    print(f"  Optimized:  {opt_fps:.2f} latents/s, {opt_latency*1000:.2f}ms/latent")
    print(f"  Speedup:    {opt_fps/ref_fps:.2f}x")


if __name__ == "__main__":
    run()
