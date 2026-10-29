"""Entry point: wires model.py and ui.py together. Run: streamlit run app.py"""
import streamlit as st

import model
import ui

ui.setup_page()


@st.cache_resource(show_spinner="Loading Stable Diffusion 1.5...")
def get_pipe():
    return model.load_pipe()


ui.header(model.DEVICE, model.GUIDANCE)

inputs = ui.generation_form(default_size_index=1 if model.DEVICE == "cuda" else 0)

if inputs:
    pipe = get_pipe()
    on_step, clear = ui.progress_tracker()
    st.session_state["res"] = model.run_diffusion(
        pipe, inputs["prompt"], inputs["steps"], inputs["seed"], inputs["size"], on_step
    )
    st.session_state["meta"] = {k: inputs[k] for k in ("prompt", "seed", "size")}
    clear()

res = st.session_state.get("res")
if res:
    pipe = get_pipe()

    def step_image(view, k):
        if view == ui.VIEW_X0:
            return model.decode(pipe, res["x0"][k - 1])
        return model.raw_grid(res["after"][k - 1])

    ui.viewer(res, step_image, model.latents_to_bytes(res, st.session_state["meta"]))
