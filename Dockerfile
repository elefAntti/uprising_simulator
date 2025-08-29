# ===== Stable-Baselines3 + Gymnasium training image =====
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: compiler toolchain, ffmpeg (videos), GL libs (headless rendering), swig (box2d),
# and git (optional, for editable installs).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    swig \
    git \
 && rm -rf /var/lib/apt/lists/*

# Keep pip tooling current
RUN python -m pip install --upgrade pip setuptools wheel

# ---- Python deps ----
# Pin NumPy/Pandas to a safe ABI pair for Py3.10 to avoid the pandas/NumPy mismatch you hit.
# SB3 v2 uses Gymnasium (not old Gym), so we install gymnasium + shimmy.
# Add extras you need (e.g., classic-control, box2d). Remove what you don't need.
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "pandas==2.2.2" \
    "gymnasium[classic-control,box2d]==0.29.1" \
    "shimmy>=1.3.0" \
    "stable-baselines3[extra]==2.3.2"

# Create a non-root user (safer; optional)
RUN useradd -ms /bin/bash runner
USER runner

# Workspace
WORKDIR /app

# Copy your project (adjust if you prefer Docker bind mounts)
# If your repo has a requirements.txt, add a separate layer to install it for better caching.
COPY --chown=runner:runner . /app

# (Optional) If your project is a package, you can install it editable:
# RUN pip install -e .
RUN pip install torch

# Default command: run your training script
# Adjust the path if your entry script differs.
CMD ["python", "train_rl.py"]

