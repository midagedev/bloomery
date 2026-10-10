---
name: Bug report
about: bloomery did something wrong (a crash, a wrong answer, a refused load, an API error)
title: "bug: <what went wrong>"
labels: bug
---

**What happened, and what you expected**

**How to reproduce**
- `bloomery-serve --version`:
- The command line, with every `BLOOMERY_*` variable you set:
- The model file (name and quantization) and where it came from:
- The request body, if it was an API call:

**Machine**
- `nvidia-smi` (the whole output):
- CPU (`lscpu | head -20`) and host RAM (`free -g`):

**Log**
- The error text and the last 50 lines of the log:
