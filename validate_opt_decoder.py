"""
Validate OptimizedDecoder on a real latent file, saving output to test.mp4.
"""
import torch
import cv2
from tqdm.auto import tqdm
from taehv import TAEHV, StreamingOptDecoder

CHECKPOINT    = "taehv1_5.pth"
LATENT_PATH   = "/mnt/data/waypoint1_5/encoded/owl_control/720p/0000bd48604141b4/taehv1_5/000000_latent.pt"
OUTPUT_PATH   = "test.mp4"
N_LATENTS     = 256   # take first N latent frames
FPS           = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16


def main():
    # --- Load model ---
    print(f"Loading TAEHV from {CHECKPOINT}...")
    taehv = TAEHV(checkpoint_path=CHECKPOINT).to(DEVICE, DTYPE)
    taehv.eval()

    # --- Load latent ---
    print(f"Loading latent from {LATENT_PATH}...")
    latent = torch.load(LATENT_PATH, map_location="cpu", weights_only=True)
    print(f"  Full latent shape: {latent.shape}")

    latent = latent[:N_LATENTS]  # [T, C, H, W]
    latent = latent.to(DEVICE, DTYPE)
    T, C, H, W = latent.shape
    print(f"  Using first {T} latent frames, shape {latent.shape}")

    # spatial dims of the latent determine FeatCache input_shape
    input_shape = (H, W)

    # --- Build streaming decoder ---
    streaming = StreamingOptDecoder(taehv, device=DEVICE, dtype=DTYPE, input_shape=input_shape).to(DEVICE, DTYPE)
    streaming.decoder.eval()

    # frames_to_trim matches TAEHV convention (t_upscale - 1)
    frames_to_trim = taehv.frames_to_trim
    frames_per_latent = taehv.t_upscale  # 4 for taehv1_5
    print(f"  t_upscale={frames_per_latent}, frames_to_trim={frames_to_trim}")

    # --- Decode ---
    writer = None
    frames_written = 0
    frames_skipped = 0

    with torch.no_grad():
        for t in tqdm(range(T), desc="Decoding latents"):
            x = latent[t].unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
            out = streaming.decode(x)                 # [frames_per_latent, 3, H_out, W_out]

            for frame_idx in range(out.shape[0]):
                if frames_skipped < frames_to_trim:
                    frames_skipped += 1
                    continue

                frame = out[frame_idx]  # [3, H_out, W_out], float in [0,1]
                frame_np = (frame.float().permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8")
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                frame_bgr = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)

                if writer is None:
                    writer = cv2.VideoWriter(
                        OUTPUT_PATH,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        FPS,
                        (1280, 720),
                    )

                writer.write(frame_bgr)
                frames_written += 1

    if writer is not None:
        writer.release()

    print(f"\nDone. Wrote {frames_written} frames to {OUTPUT_PATH}")
    print(f"  (skipped {frames_skipped} startup frames)")


if __name__ == "__main__":
    main()
