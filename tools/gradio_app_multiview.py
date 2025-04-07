import gradio as gr

from sam2_depthanything.gradio_ui.mv_sam import mv_sam_block

title = """# Sam2 | VGGT | Video Depth Anything"""
description1 = """
    <a title="Website" href="https://promptda.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
    </a>
    <a title="Github" href="https://github.com/rerun-io/prompt-da" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/rerun-io/prompt-da?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
"""
description2 = "Using a combination of models to track an object in 3D"

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description1)
    gr.Markdown(description2)
    with gr.Tab("Multiview"):
        mv_sam_block.render()

if __name__ == "__main__":
    demo.queue(max_size=2).launch()
