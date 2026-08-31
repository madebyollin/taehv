from pathlib import Path

import torch

from taehv import TAEH3, TAEHV


ROOT = Path(__file__).resolve().parents[1]


def test_taeh3_has_explicit_h3_contract() -> None:
    model = TAEH3(checkpoint_path=None)

    assert model.is_h3
    assert model.patch_size == 2
    assert model.latent_channels == 24
    assert model.t_downscale == 4
    assert model.t_upscale == 4
    assert model.frames_to_trim == 3


def test_taeh3_checkpoint_does_not_depend_on_filename(tmp_path: Path) -> None:
    checkpoint = tmp_path / "renamed-weights.pth"
    checkpoint.hardlink_to(ROOT / "taeh3.pth")

    model = TAEH3(checkpoint).eval()

    assert model.is_h3
    assert model.decoder[1].weight.shape[1] == 24
    assert model.decoder[22].weight.shape[0] == 12


def test_taeh3_matches_legacy_filename_inference() -> None:
    explicit = TAEH3(ROOT / "taeh3.pth").eval()
    inferred = TAEHV(str(ROOT / "taeh3.pth")).eval()
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn((1, 5, 24, 1, 1), generator=generator)

    with torch.no_grad():
        explicit_frames = explicit.decode_video(latents, parallel=True, show_progress_bar=False)
        inferred_frames = inferred.decode_video(latents, parallel=True, show_progress_bar=False)

    assert explicit_frames.shape == (1, 5, 3, 16, 16)
    torch.testing.assert_close(explicit_frames, inferred_frames, rtol=0, atol=0)


def test_taeh3_maps_37_latents_to_124_frames() -> None:
    model = TAEH3(ROOT / "taeh3.pth").eval()
    latents = torch.zeros((1, 37, 24, 1, 1))

    with torch.no_grad():
        frames = model.decode_video(latents, parallel=True, show_progress_bar=False)

    assert frames.shape == (1, 124, 3, 16, 16)
