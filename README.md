# 🥮 Tiny AutoEncoder for Hunyuan Video

## What is TAEHV?

TAEHV is a Tiny AutoEncoder for Hunyuan Video. TAEHV can decode Hunyuan latents into videos more cheaply (in time & memory) than the full-size Hunyuan VAE, at the cost of slightly lower quality.

Here's a comparison of the output & memory usage of the Full VAE vs. TAEHV:

<table>
<tr><th><tt>pipe.vae</tt></th><th>Full VAE</th><th>TAEHV</th></tr>
<tr>
  <td>Decoded Video</td>
  <td><img src="https://github.com/user-attachments/assets/b9ee3405-c210-4410-95ac-639a4ed09c50"/></td>
  <td><img src="https://github.com/user-attachments/assets/3fe3cb6a-30e5-46fe-9458-f0a39e454b86"/></td>
</tr>
<tr>
  <td>Memory</td>
  <td><img width="281" alt="memory_full" src="https://github.com/user-attachments/assets/d7f80f3b-7030-4e0f-be3d-fad994557105" /></td>
  <td><img width="278" alt="memory_taehv" src="https://github.com/user-attachments/assets/9ce922dc-d4af-4422-8b70-d0d5cc326167" /></td>
</tr>
</table>

See the [example notebook](./examples/TAEHV_T2I_Demo.ipynb) for details on the comparison (it's using the fp16 Diffusers implementation with no CPU offload; I tested on H100 and commented out the `pipe.vae` override to get the Full VAE results).

## Limitations

TAEHV is still pretty experimental (specifically, it's a hacky finetune of [TAEM1](https://github.com/madebyollin/taem1) :) using a fairly limited dataset) and I haven't tested it much yet. Please report quality / performance issues as you discover them.
