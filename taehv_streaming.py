"""
Streaming encoder and decoder for TAEHV.

Enables processing videos in chunks for memory efficiency and supports
autoregressive latent generation for world models.
"""

import torch
import torch.nn.functional as F
from collections import namedtuple

# Import from local module
try:
    from .taehv import TAEHV, MemBlock, TPool, TGrow
except ImportError:
    from taehv import TAEHV, MemBlock, TPool, TGrow

TWorkItem = namedtuple('TWorkItem', ['x', 'layer_idx'])


class StreamingTAEHVEncoder:
    """
    Streaming encoder that processes video frames in chunks while maintaining
    temporal consistency across chunk boundaries via state preservation.

    This allows encoding arbitrarily long videos without loading the entire
    video into memory at once. The encoder maintains MemBlock and TPool state
    across chunks to ensure temporal consistency.

    Example:
        >>> encoder = StreamingTAEHVEncoder(model)
        >>> encoder.reset()
        >>>
        >>> # Process video in chunks
        >>> for chunk in video_chunks:  # chunk: [1, T, 3, H, W]
        >>>     latent = encoder.feed_frames(chunk)
        >>>     if latent is not None:
        >>>         process_latent(latent)
        >>>
        >>> # Finalize to flush buffers
        >>> final_latent = encoder.finalize()
    """

    def __init__(self, taehv_model):
        """
        Args:
            taehv_model: TAEHV model instance
        """
        self.model = taehv_model
        self.encoder = taehv_model.encoder
        self.patch_size = taehv_model.patch_size
        self.t_downscale = taehv_model.t_downscale

        # Initialize state
        self.reset()

    def reset(self):
        """Reset encoder state. Call this before encoding a new video."""
        self.mem = [None] * len(self.encoder)  # Per-layer memory state
        self.work_queue = []  # Work items being processed
        self.output_latents = []  # Accumulated output latents
        self.total_frames_fed = 0

    def feed_frames(self, frames):
        """
        Feed a chunk of frames to the encoder.

        Args:
            frames: torch.Tensor [1, T, C, H, W] in range [0, 1]

        Returns:
            latents: torch.Tensor [1, T', latent_channels, H', W'] or None
                     where T' depends on temporal downsampling and buffering.
                     Returns None if frames are still buffering in TPool.
        """
        assert frames.ndim == 5, "Expected [N, T, C, H, W] tensor"
        N, T, C, H, W = frames.shape

        # Apply pixel unshuffle if needed
        if self.patch_size > 1:
            frames = F.pixel_unshuffle(frames, self.patch_size)

        # Update dimensions after pixel unshuffle
        N, T, C, H, W = frames.shape

        # Add frames to work queue
        for t in range(T):
            frame = frames[:, t:t+1, :, :, :]  # Keep dims: [N, 1, C, H, W]
            frame = frame.reshape(N, C, H, W)  # Remove T dim: [N, C, H, W]
            self.work_queue.append(TWorkItem(frame, 0))
            self.total_frames_fed += 1

        # Process work queue
        chunk_outputs = []

        while self.work_queue:
            xt, layer_idx = self.work_queue.pop(0)

            # Check if we've reached the end (output)
            if layer_idx == len(self.encoder):
                chunk_outputs.append(xt)
                continue

            # Get the layer
            b = self.encoder[layer_idx]

            if isinstance(b, MemBlock):
                # MemBlock needs previous frame state
                if self.mem[layer_idx] is None:
                    # First frame - use zero as previous
                    xt_new = b(xt, xt * 0)
                    self.mem[layer_idx] = xt.clone()
                else:
                    # Use cached previous frame
                    xt_new = b(xt, self.mem[layer_idx])
                    self.mem[layer_idx] = xt.clone()

                self.work_queue.insert(0, TWorkItem(xt_new, layer_idx + 1))

            elif isinstance(b, TPool):
                # TPool accumulates frames before pooling
                if self.mem[layer_idx] is None:
                    self.mem[layer_idx] = []

                self.mem[layer_idx].append(xt)

                if len(self.mem[layer_idx]) == b.stride:
                    # We have enough frames to pool
                    pooled = b(torch.cat(self.mem[layer_idx], 1).view(N * b.stride, -1, xt.shape[-2], xt.shape[-1]))
                    self.mem[layer_idx] = []  # Reset pool buffer
                    self.work_queue.insert(0, TWorkItem(pooled, layer_idx + 1))
                # else: not enough frames yet, wait for more

            elif isinstance(b, TGrow):
                # TGrow shouldn't be in encoder
                raise ValueError("TGrow found in encoder - this shouldn't happen!")

            else:
                # Normal layer
                xt = b(xt)
                self.work_queue.insert(0, TWorkItem(xt, layer_idx + 1))

        # Convert outputs to tensor
        if chunk_outputs:
            latents = torch.stack(chunk_outputs, dim=1)  # [N, T', C, H, W]
            self.output_latents.append(latents)
            return latents
        else:
            # No output this chunk (frames still buffering in TPool)
            return None

    def finalize(self):
        """
        Flush any remaining buffered frames and return all latents.
        Call this after feeding all chunks.

        Returns:
            latents: torch.Tensor [1, T_total, latent_channels, H', W']
        """
        # Pad if needed to flush TPool buffers
        if self.total_frames_fed % self.t_downscale != 0:
            n_pad = self.t_downscale - (self.total_frames_fed % self.t_downscale)

            # Create padding frames (repeat last frame)
            if self.mem[0] is not None:
                last_frame = self.mem[0]  # Last frame seen by first MemBlock
                for _ in range(n_pad):
                    self.work_queue.append(TWorkItem(last_frame, 0))

                # Process remaining queue
                chunk_outputs = []
                while self.work_queue:
                    xt, layer_idx = self.work_queue.pop(0)

                    if layer_idx == len(self.encoder):
                        chunk_outputs.append(xt)
                        continue

                    b = self.encoder[layer_idx]

                    if isinstance(b, MemBlock):
                        if self.mem[layer_idx] is None:
                            xt_new = b(xt, xt * 0)
                        else:
                            xt_new = b(xt, self.mem[layer_idx])
                        self.mem[layer_idx] = xt.clone()
                        self.work_queue.insert(0, TWorkItem(xt_new, layer_idx + 1))

                    elif isinstance(b, TPool):
                        if self.mem[layer_idx] is None:
                            self.mem[layer_idx] = []
                        self.mem[layer_idx].append(xt)

                        if len(self.mem[layer_idx]) == b.stride:
                            N = xt.shape[0]
                            pooled = b(torch.cat(self.mem[layer_idx], 1).view(N * b.stride, -1, xt.shape[-2], xt.shape[-1]))
                            self.mem[layer_idx] = []
                            self.work_queue.insert(0, TWorkItem(pooled, layer_idx + 1))

                    else:
                        xt = b(xt)
                        self.work_queue.insert(0, TWorkItem(xt, layer_idx + 1))

                if chunk_outputs:
                    return torch.stack(chunk_outputs, dim=1)

        return None


class StreamingTAEHVDecoder:
    """
    Streaming decoder for autoregressive world model generation.

    Feed 1 latent frame at a time, get ~4 video frames out (depending on t_upscale).
    Maintains temporal state across calls for MemBlocks, enabling true streaming
    generation where latents are generated one at a time.

    Example:
        >>> decoder = StreamingTAEHVDecoder(model)
        >>> decoder.reset()
        >>>
        >>> # Process latents one at a time (e.g., from world model)
        >>> for latent in latent_sequence:  # latent: [1, 1, C, H, W]
        >>>     frames = decoder.decode_single_latent(latent)  # → [1, 4, 3, H', W']
        >>>     render_frames(frames)
    """

    def __init__(self, taehv_model):
        """
        Args:
            taehv_model: TAEHV model instance
        """
        self.model = taehv_model
        self.decoder = taehv_model.decoder
        self.patch_size = taehv_model.patch_size
        self.t_upscale = taehv_model.t_upscale
        self.frames_to_trim = taehv_model.frames_to_trim

        # Initialize state
        self.reset()

    def reset(self):
        """Reset decoder state. Call this before decoding a new video."""
        self.mem = [None] * len(self.decoder)  # Per-layer memory state
        self.frames_decoded = 0  # Track how many frames we've output

    def decode_single_latent(self, latent):
        """
        Decode a single latent frame to video frames.

        Args:
            latent: torch.Tensor [1, 1, C, H, W] or [1, C, H, W] - single latent frame

        Returns:
            frames: torch.Tensor [1, T, 3, H', W'] - decoded video frames
                    where T depends on t_upscale (typically 4 for taehv1_5).
                    May return fewer frames at the start due to trimming.
        """
        # Handle input shape
        if latent.ndim == 4:
            latent = latent.unsqueeze(1)  # [1, C, H, W] → [1, 1, C, H, W]

        assert latent.ndim == 5, "Expected [N, 1, C, H, W] tensor"
        assert latent.shape[1] == 1, "Expected single latent frame (T=1)"

        N, _, C, H, W = latent.shape
        latent = latent.squeeze(1)  # [1, 1, C, H, W] → [1, C, H, W]

        # Initialize work queue with this latent
        work_queue = [TWorkItem(latent, 0)]
        outputs = []

        # Process through all decoder layers
        while work_queue:
            xt, layer_idx = work_queue.pop(0)

            # Check if we've reached the end (output)
            if layer_idx == len(self.decoder):
                outputs.append(xt)
                continue

            # Get the layer
            b = self.decoder[layer_idx]

            if isinstance(b, MemBlock):
                # MemBlock needs previous frame state
                if self.mem[layer_idx] is None:
                    # First frame - use zero as previous
                    xt_new = b(xt, xt * 0)
                    self.mem[layer_idx] = xt.clone()
                else:
                    # Use cached previous frame
                    xt_new = b(xt, self.mem[layer_idx])
                    self.mem[layer_idx] = xt.clone()

                work_queue.insert(0, TWorkItem(xt_new, layer_idx + 1))

            elif isinstance(b, TGrow):
                # TGrow expands 1 frame into stride frames
                # For taehv1_5: 2 TGrow layers with stride=2 each → 1 latent → 4 frames
                xt = b(xt)
                NT, C, H, W = xt.shape

                # Split into individual frames
                # TGrow outputs [N*stride, C, H, W]
                frames = xt.view(N, -1, C, H, W)  # [N, stride, C, H, W]

                # Add each expanded frame to work queue (in reverse for FIFO processing)
                for i in reversed(range(frames.shape[1])):
                    frame = frames[:, i, :, :, :]  # [N, C, H, W]
                    work_queue.insert(0, TWorkItem(frame, layer_idx + 1))

            elif isinstance(b, TPool):
                # TPool shouldn't be in decoder
                raise ValueError("TPool found in decoder - this shouldn't happen!")

            else:
                # Normal layer (Conv, ReLU, Upsample, etc.)
                xt = b(xt)
                work_queue.insert(0, TWorkItem(xt, layer_idx + 1))

        # Post-process outputs
        if outputs:
            # Stack outputs: [N, T', C, H, W] where T' = t_upscale (typically 4)
            frames = torch.stack(outputs, dim=1)

            # Apply pixel shuffle if needed
            if self.patch_size > 1:
                N, T, C, H, W = frames.shape
                frames = frames.reshape(N * T, C, H, W)
                frames = F.pixel_shuffle(frames, self.patch_size)
                _, C_new, H_new, W_new = frames.shape
                frames = frames.view(N, T, C_new, H_new, W_new)

            # Clamp to [0, 1]
            frames = frames.clamp_(0, 1)

            # Handle frame trimming for first few frames
            # For taehv1_5: frames_to_trim = 3
            if self.frames_to_trim > 0 and self.frames_decoded < self.frames_to_trim:
                frames_to_trim_now = min(self.frames_to_trim - self.frames_decoded, frames.shape[1])
                frames = frames[:, frames_to_trim_now:]
                self.frames_decoded += frames.shape[1]
            else:
                self.frames_decoded += frames.shape[1]

            return frames
        else:
            return None


if __name__ == "__main__":
    print("=" * 60)
    print("TAEHV Streaming Encoder/Decoder")
    print("=" * 60)
    print("\nFor usage examples, see:")
    print("  - examples/TAEHV1.5_Streaming_Demo.ipynb")
    print("  - README.md (streaming section)")
    print("=" * 60)
