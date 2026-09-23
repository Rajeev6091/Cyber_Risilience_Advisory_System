"""ASGI entrypoint for container and serverless hosts.

`python3 integrated_system.py` runs its own server and is still the way to run
this locally. Platforms that import an app instead of running a process (Vercel,
Cloud Run behind an ASGI server, etc.) load `app` from here.

Kept separate from app.py so importing the RAG module never builds a UI, which
would make integrated_system.py's import of it circular.
"""
import gradio as gr
from fastapi import FastAPI

from integrated_system import IntegratedCyberSecuritySystem, create_gradio_interface

# Built at import: the platform holds the instance across requests, so paying
# the model and index load once per cold start is the point.
_system = IntegratedCyberSecuritySystem()
_demo = create_gradio_interface(_system)

app = gr.mount_gradio_app(FastAPI(), _demo, path="/")
