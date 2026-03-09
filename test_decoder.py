"""
Test script for streaming decoder performance and correctness.
Usage: python test_decoder.py
"""
import torch
import time
from taehv import TAEHV, StreamingTAEHV

CHECKPOINT = "taehv1_5.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16

N_LATENT_FRAMES = 16   # T dimension of latents
LATENT_H, LATENT_W = 32, 64
N_WARMUP = 2
N_EVAL = 5


def make_latents(taehv):
    return torch.randn(1, N_LATENT_FRAMES, taehv.latent_channels, LATENT_H, LATENT_W, device=DEVICE, dtype=DTYPE)


def decode_streaming(taehv, latents):
    """Decode latents frame-by-frame using StreamingTAEHV. Returns list of decoded frames."""
    streaming = StreamingTAEHV(taehv)
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


def compute_deviation(frames_a, frames_b):
    """Mean absolute deviation between two lists of N1CHW frames."""
    a = torch.cat(frames_a, dim=1).float()
    b = torch.cat(frames_b, dim=1).float()
    return (a - b).abs().mean().item()


def run():
    print(f"Device: {DEVICE}, dtype: {DTYPE}, checkpoint: {CHECKPOINT}")
    taehv = TAEHV(checkpoint_path=CHECKPOINT).to(DEVICE, DTYPE)
    taehv.eval()
    expected_frames = N_LATENT_FRAMES * taehv.t_upscale - taehv.frames_to_trim

    print(f"\nLatent shape: [1, {N_LATENT_FRAMES}, {taehv.latent_channels}, {LATENT_H}, {LATENT_W}]")
    print(f"Expected output frames: {expected_frames}")

    # --- Warmup ---
    print(f"\nRunning {N_WARMUP} warmup calls...")
    for _ in range(N_WARMUP):
        latents = make_latents(taehv)
        with torch.no_grad():
            decode_streaming(taehv, latents)

    # --- Eval ---
    print(f"Running {N_EVAL} eval calls...")
    fps_list = []
    latency_list = []
    ref_frames = None

    for i in range(N_EVAL):
        latents = make_latents(taehv)
        with torch.no_grad():
            t0 = time.perf_counter()
            frames = decode_streaming(taehv, latents)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        n_frames = len(frames)
        fps = n_frames / elapsed
        fps_list.append(fps)
        latency_list.append(elapsed)

        if i == 0:
            ref_frames = frames  # save first run's output for deviation baseline
            ref_latents = latents

        print(f"  Run {i+1}: {n_frames} frames in {elapsed:.3f}s → {fps:.1f} FPS")

    avg_fps = sum(fps_list) / len(fps_list)
    avg_latency = sum(latency_list) / len(latency_list)
    print(f"\nAverage FPS:     {avg_fps:.2f}")
    print(f"Average latency: {avg_latency:.3f}s")

    # --- Deviation metric ---
    # Baseline: decode a fresh random latent, measure deviation from ref
    print("\nComputing deviation metrics...")
    with torch.no_grad():
        random_latents = make_latents(taehv)
        random_frames = decode_streaming(taehv, random_latents)

    # Align lengths (in case of off-by-one from trimming)
    n = min(len(ref_frames), len(random_frames))
    ref_frames_trimmed = ref_frames[:n]
    random_frames_trimmed = random_frames[:n]

    random_baseline = compute_deviation(ref_frames_trimmed, random_frames_trimmed)
    self_deviation = compute_deviation(ref_frames, ref_frames)  # sanity check: should be 0

    print(f"  Self-deviation (sanity, should be 0): {self_deviation:.6f}")
    print(f"  Random baseline deviation:            {random_baseline:.6f}")
    print()
    print("When testing OptimizedDecoder, compare its deviation against ref_frames.")
    print(f"  A good optimized decoder should have deviation << {random_baseline:.4f} (random baseline).")

    return {
        "avg_fps": avg_fps,
        "avg_latency": avg_latency,
        "random_baseline_deviation": random_baseline,
        "ref_frames": ref_frames,
        "ref_latents": ref_latents,
    }


if __name__ == "__main__":
    results = run()
