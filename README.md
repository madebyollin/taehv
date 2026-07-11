# 🥮 Tiny AutoEncoder for Hunyuan Video

## What is TAEHV?

TAEHV is a Tiny AutoEncoder for Hunyuan Video (and other similar video models). TAEHV can encode and decode latents into videos more cheaply (in time & memory) than the full-size video VAEs, at the cost of slightly lower quality.

Here's a comparison of the output & memory usage of the Full Hunyuan VAE vs. TAEHV during decoding:

<table>
    <tr><th>Decoder AE</th><th>Full Hunyuan VAE</th><th>TAEHV</th></tr>
<tr>
  <td>Decoded Video<br/><sup>(converted to GIF)</sup></td>
  <td><img src="https://github.com/user-attachments/assets/b9ee3405-c210-4410-95ac-639a4ed09c50"/></td>
  <td><img src="https://github.com/user-attachments/assets/3fe3cb6a-30e5-46fe-9458-f0a39e454b86"/></td>
</tr>
<tr>
  <td>Runtime<br/><sup>(in fp16, on GH200)</sup></td>
  <td><strong>~2-3s</strong> for decoding 61 frames of (512, 320) video</td>
  <td><strong>~0.5s</strong> for decoding 61 frames of (512, 320) video.<br/>Can be even faster with <a href="#how-can-i-make-taehv-even-faster">the right settings</a></td>
</tr>
<tr>
  <td>Memory<br/><sup>(in fp16, on GH200)</sup></td>
  <td><strong>~6-9GB Peak Memory Usage</strong><br/><img src="https://github.com/user-attachments/assets/d7837271-c748-4eef-ab37-eda6cc1e6a69"/></td>
  <td><strong><0.5GB Peak Memory Usage</strong><br/><img src="https://github.com/user-attachments/assets/c71e2ef5-12f1-431f-b193-29d9a5ee6343"/></td>
</tr>
</table>

See the [profiling notebook](./examples/TAEHV_Profiling.ipynb) for details on this comparison or the [example notebook](./examples/TAEHV_T2I_Demo.ipynb) for a simpler demo.

## What video models does TAEHV support?

To use TAEHV with different video models, you can load the different model weight files from this repo:

* For **Hunyuan Video 1.5**, load the `taehv1_5` weights ([example notebook](./examples/TAEHV1.5_Encoding_Decoding_Demo.ipynb)).
    * You can also use `taehv1_5_super` to get higher-quality output at the cost of slightly more compute.
* For **Wan 2.1**, load the `taew2_1` weights (see the [Wan 2.1 example notebook](./examples/TAEW2.1_T2I_Demo.ipynb)).
* For **Wan 2.2**, load different files depending on model scale:
  * For **Wan 2.2 5B**, load the `taew2_2` weights ([example notebook](examples/TAEW2.2_T2I_Demo.ipynb)).
    * You can also use `taew2_2_super` to get higher-quality output at the cost of slightly more compute.
  * For **Wan 2.2 14B**, load the `taew2_1` weights since Wan 2.2 14B [still uses the older Wan 2.1 VAE](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_t2v_A14B.py#L16).
* for **Qwen Image**, load the `taew2_1` weights (since Qwen Image uses the Wan 2.1 VAE encoder).
* For **CogVideoX**,  load the `taecvx` weights ([example notebook](./examples/TAECVX_T2I_Demo.ipynb)).
* For **Hunyuan Video 1**, load the `taehv` weights ([example notebook](./examples/TAEHV_T2I_Demo.ipynb)).
* For **Open-Sora 1.3**, load the `taeos1_3` weights.
* For **LTX-2**, load the `taeltx_2` weights ([example notebook](./examples/TAELTX2_Encoding_Decoding_Demo.ipynb)).
* For **LTX-2.3**, load the `taeltx2_3` weights
    * The standard TAE decoder for LTX2.3 has blurry outputs (see [thread](https://github.com/madebyollin/taehv/issues/20#issuecomment-4048513703)) so I also trained a larger, less-blurry `taeltx2_3_wide` variant. Those `taeltx2_3_wide` weights are [here](https://github.com/madebyollin/taehv/blob/2026_03_11_taeltx23_wide/taeltx2_3_wide.pth) and **ComfyUI-bleh** has `taeltx2_3_wide`-compatible previewing code [here](https://github.com/blepping/ComfyUI-bleh/pull/37).
* For **Mochi 1** and **SVD** (which use different architectures), see the other repos [TAEM1](https://github.com/madebyollin/taem1) and [TAESDV](https://github.com/madebyollin/taesdv).

The main model weight `.pth` files are in the repository root directory. Converted `.safetensors` files are located in the [safetensors](./safetensors) subdirectory.

If there's another open video model that would benefit from a TAEHV version, please file an [issue](https://github.com/madebyollin/taehv/issues) (or, worst-case, try [training your own](https://github.com/madebyollin/seraena/blob/main/TAEHV_Training_Example.ipynb)).

## Where can I get TAEHV?

TAEHV is available:

* In [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
  * In main, thanks to [this PR](https://github.com/comfyanonymous/ComfyUI/pull/10884) from [Kijai](https://github.com/kijai)
  * Via the [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper/commit/ce7917664697ee044db8b697ed775ed25cecd000) + [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) nodes thanks to [Kijai](https://github.com/kijai) and [AustinMroz](https://github.com/AustinMroz) 
  * Via the [ComfyUI-Bleh](https://github.com/blepping/ComfyUI-bleh) nodes thanks to [blepping](https://github.com/blepping)
* In [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) thanks to this [PR](https://github.com/leejet/stable-diffusion.cpp/pull/937) from [stduhpf](https://github.com/stduhpf)
* In [SDNext](https://github.com/vladmandic/sdnext) thanks to [vladmandic](https://github.com/vladmandic)
* In the Wan2.1 [Self-Forcing](https://github.com/guandeh17/Self-Forcing) demo thanks to [Guande He](https://github.com/guandeh17) and [Xun Huang](https://github.com/xunhuang1995)

If you've added TAEHV support elsewhere, LMK and I can add a link here.

## What are the limitations of TAEHV?

TAEHV is fast and well-suited for live previewing or interactive video, but TAEHV doesn't yet match the quality of the full-size VAEs.

You can see detailed quality comparisons of TAEHV vs full-size VAEs [here](https://huggingface.co/datasets/madebyollin/movirec).

<img width="512" alt="Speed vs quality chart from MoViRec benchmark" src="https://github.com/user-attachments/assets/b5f573fa-d21f-45eb-8616-f852ba43f08b" />

## How do I use TAEHV with 🧨 Diffusers?

You can use TAEHV with Diffusers by applying a small bit of wrapper code ([example notebook](https://github.com/madebyollin/taehv/blob/main/examples/TAEW2.1_Diffusers_Encoding_and_Decoding_Demo.ipynb)).
If you're writing new code involving both TAEHV and Diffusers, keep the following conventions in mind:

1. TAEHV stores image values in the range **[0, 1]**, whereas Diffusers uses **[-1, 1]**.
2. TAEHV stores videos in **NTCHW** dimension order (time, then channels), while Diffusers stores videos in **NCTHW** dimension order.
3. TAEHV does not use any latent scales / shifts (TAEHV encodes / decodes exactly what diffusion models use), whereas Diffusers requires **explicitly applying** a `latents_mean` and `latents_std` each time you encode or decode something.

## How can I make TAEHV even faster?

You can disable TAEHV's temporal or spatial upscaling to get even-cheaper decoding.

```python
TAEHV(decoder_time_upscale=(False, False), decoder_space_upscale=(True, True, True))
```

![Image](https://github.com/user-attachments/assets/c517e37b-e53b-4d7d-b282-fbbbce10ade7)

```python
TAEHV(decoder_time_upscale=(False, False), decoder_space_upscale=(False, False, False))
```

![Image](https://github.com/user-attachments/assets/62223493-8cad-427b-b13c-fa9919d3fd7b)

If you have a powerful GPU or are decoding at a reduced resolution, you can also set `parallel=True` in `TAEHV.decode_video` to decode all frames at once (which is faster but requires more memory).

## How can I use TAEHV to encode and decode live video?

TAEHV's default `encode_video` and `decode_video` methods process the entire input sequence before returning the entire result.
For live / real-time scenarios (like video-to-video or world modeling), you'll want to use the `StreamingTAEHV` wrapper, which processes each input timestep sequentially and streams intermediate outputs as soon as they're ready.

You can see example `StreamingTAEHV` usage in the [streaming demo notebook](./examples/TAEHV1.5_Streaming_Demo.ipynb) or in [`taehv.py`](./taehv.py)'s `StreamingTAEHV` docstrings.

![Streaming demo notebook screen recording](https://github.com/user-attachments/assets/90f4cb9f-c056-4c7e-926f-2b10cf072c73)

## How can I make TAEHV decode higher-quality videos?

For `taehv1_5` and `taew2_2`, you can use the larger `taehv1_5_super` and `taew2_2_super` weights to get higher quality video output at the cost of slightly increased compute (and higher parameter count). Thsee `super` models are intended for realtime world modeling and video usecases where TAEHV output serves as the final output (rather than a temporary preview).

## How can I cite TAEHV in a publication?

If you find TAEHV useful in your research, you can cite the TAEHV repo as a web link:

```bibtex
@misc {BoerBohan2025TAEHV,
  author = {Boer Bohan, Ollin},
  title = {TAEHV: Tiny AutoEncoder for Hunyuan Video},
  year = {2025},
  howpublished = {\url{https://github.com/madebyollin/taehv}},
}
```

The TAEHV repo contents change over time, so I recommend also noting the latest commit hash and access date in a note field, e.g.
```bibtex
note = {Commit: \texttt{5ce7381}, Accessed: 2025-09-05}
```
