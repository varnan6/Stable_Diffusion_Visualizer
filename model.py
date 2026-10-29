"""Model-side code only: SD 1.5 loading, DDIM loop, decoding, latent visualisation."""
import io

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from matplotlib import colormaps
from PIL import Image

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
GUIDANCE = 7.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe.to(DEVICE)


@torch.no_grad()
def decode(pipe, latent):
    """Latent (1,4,h,w) -> PIL image via the VAE."""
    z = latent.to(DEVICE, DTYPE) / pipe.vae.config.scaling_factor
    img = pipe.vae.decode(z).sample
    img = (img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((img * 255).round().astype(np.uint8))


@torch.no_grad()
def run_diffusion(pipe, prompt, steps, seed, size, on_step=None):
    """Run DDIM once; cache per-step latent (after step) and predicted x0.

    on_step(i, total) is called after each step (optional).
    """
    sched = pipe.scheduler
    sched.set_timesteps(steps, device=DEVICE)
    cond, uncond = pipe.encode_prompt(prompt, DEVICE, 1, True)
    emb = torch.cat([uncond, cond])

    gen = torch.Generator().manual_seed(int(seed))
    lat = torch.randn((1, 4, size // 8, size // 8), generator=gen)
    lat = (lat * sched.init_noise_sigma).to(DEVICE, DTYPE)

    after, x0s = [], []
    for i, t in enumerate(sched.timesteps):
        inp = sched.scale_model_input(torch.cat([lat, lat]), t)
        eps = pipe.unet(inp, t, encoder_hidden_states=emb).sample
        e_u, e_c = eps.chunk(2)
        eps = e_u + GUIDANCE * (e_c - e_u)
        out = sched.step(eps, t, lat)
        lat = out.prev_sample
        after.append(lat.float().cpu())
        x0s.append(out.pred_original_sample.float().cpu())
        if on_step:
            on_step(i + 1, steps)

    return {"after": after, "x0": x0s, "final": decode(pipe, lat), "steps": steps}


def raw_grid(latent, cell=256):
    """4 latent channels -> 2x2 viridis heatmap grid (per-channel min-max)."""
    cmap = colormaps["viridis"]
    tiles = []
    for c in range(4):
        ch = latent[0, c].numpy()
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        rgb = (cmap(ch)[..., :3] * 255).astype(np.uint8)
        tiles.append(np.array(Image.fromarray(rgb).resize((cell, cell), Image.NEAREST)))
    return Image.fromarray(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))


def latents_to_bytes(res, meta):
    buf = io.BytesIO()
    torch.save({"after": res["after"], "x0": res["x0"], "meta": meta}, buf)
    return buf.getvalue()
