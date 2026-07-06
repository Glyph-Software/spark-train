FROM nvcr.io/nvidia/pytorch:25.11-py3

# Training stack on top of the NGC torch build (ARM64 + Blackwell SM121).
#   - unsloth/unsloth_zoo/bitsandbytes with --no-deps so they can't drag in a
#     conflicting torch/triton wheel over the NGC nightly.
#   - torchao removed: its release wheels clash with the NGC torch nightly, and
#     the 16-bit LoRA path never uses it.
RUN pip install --no-cache-dir transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1" \
 && pip install --no-cache-dir --no-deps unsloth unsloth_zoo bitsandbytes \
 && pip uninstall -y torchao

WORKDIR /workspace

# Bake the training code into the image (mutable data dirs are excluded via
# .dockerignore and bind-mounted by docker-compose instead). Code changes only
# invalidate this layer, so rebuilds skip the pip installs above.
COPY workspace/ /workspace/

CMD ["python", "dashboard.py"]
