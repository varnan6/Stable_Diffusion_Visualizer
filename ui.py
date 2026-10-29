"""Frontend-only code: Streamlit widgets and layout. No model imports."""
import io

import streamlit as st

VIEW_X0 = "Predicted x₀"
VIEW_RAW = "Raw latent (4 ch)"


def to_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def setup_page():
    st.set_page_config(page_title="Denoising Step Viewer", layout="wide")


def header(device, guidance):
    st.title("Denoising Step Viewer")
    st.caption(f"SD 1.5 · DDIM · CFG {guidance} · running on {device.upper()}")


def generation_form(default_size_index):
    """Returns a dict of inputs when Generate is pressed, else None."""
    with st.form("gen"):
        prompt = st.text_input("Prompt", "a lighthouse on a cliff at sunset, oil painting")
        c1, c2, c3 = st.columns(3)
        steps = c1.slider("Denoising steps", 10, 50, 20)
        seed = c2.number_input("Seed", 0, 2**31 - 1, 42)
        size = c3.selectbox("Resolution", [384, 512], index=default_size_index)
        go = st.form_submit_button("Generate")
    if not go:
        return None
    return {"prompt": prompt, "steps": steps, "seed": int(seed), "size": size}


def progress_tracker():
    """Returns (on_step callback, clear function)."""
    bar = st.progress(0.0, text="Starting...")

    def on_step(i, total):
        bar.progress(i / total, text=f"Step {i}/{total}")

    return on_step, bar.empty


def viewer(res, step_image_fn, latents_bytes):
    """Step viewer (left) + final image (right).

    step_image_fn(view, k) -> PIL image for the chosen view and step k (1-based).
    """
    left, right = st.columns(2)

    with left:
        view = st.radio("View", [VIEW_X0, VIEW_RAW], horizontal=True)
        k = st.slider("Step", 1, res["steps"], 1)
        img = step_image_fn(view, k)
        st.image(img, caption=f"{view} — step {k}/{res['steps']}", use_container_width=True)
        tag = "x0" if view == VIEW_X0 else "raw"
        st.download_button("Download step view (PNG)", to_png(img),
                           f"step_{k}_{tag}.png", "image/png")

    with right:
        st.subheader("Final image")
        st.image(res["final"], use_container_width=True)
        st.download_button("Download final (PNG)", to_png(res["final"]), "final.png", "image/png")
        st.download_button("Download latents (.pt)", latents_bytes, "latents.pt")
