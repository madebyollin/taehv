# 🥮 Tiny AutoEncoder for Hunyuan Video

## What is TAEHV?

TAEHV is a Tiny AutoEncoder for Hunyuan Video. TAEHV can decode Hunyuan latents into videos more cheaply than the full-size Hunyuan VAE.

Here's a comparison of the full VAE vs. TAEHV using fp16 Diffusers with no CPU offload:

<table>
<tr><th>`pipe.vae`</th><th>Full VAE</th><th>TAEHV</th></tr>
<tr><td>Video</td><td></td></tr>
<tr><td>Memory</td><td></td></tr>
</table>

See the [example notebook](./examples/TAEHV_T2I_Demo.ipynb) for more info.
