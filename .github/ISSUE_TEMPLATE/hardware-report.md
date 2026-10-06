---
name: Hardware report
about: bloomery on a card, CPU or card count we have not run (it worked, or it did not)
title: "hardware: <card> / <CPU> / <model>"
labels: hardware
---

Thank you for running bloomery on hardware we do not have. A report that it **worked** helps as much as one that it
failed.

**Machine**
- `bloomery-serve --version`:
- `nvidia-smi` (the whole output):
- CPU (`lscpu | head -20`):
- Host RAM (`free -g`):
- OS and glibc (`ldd --version | head -1`):

**Run**
- The command line, with every `BLOOMERY_*` variable you set:
- The model and quantization:
- The plan line from the log (it names the cards and the expert split):

**Result**
- It loaded and answered: yes / no
- For one request alone, after one warm-up request: `predicted_per_second`, `prompt_per_second`, `draft_n`,
  `draft_n_accepted` from the response's `timings`, with the request body:
- If it failed, the error text and the last 50 lines of the log:
