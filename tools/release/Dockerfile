# bloomery's server image: the prebuilt release on Ubuntu 24.04 (glibc 2.39, above the
# release's 2.34 floor). The GPU comes from the host's NVIDIA Container Toolkit
# (run with `--gpus all`); libcuda is injected by that runtime, never baked in.
#
# Build:  docker build --platform linux/amd64 -t ghcr.io/midagedev/bloomery:0.2.0 \
#             --build-arg VERSION=0.2.0 \
#             --build-arg SHA256=<the release's own sha256> tools/release
# Run:    docker run --gpus all -p 8080:8080 -v bloomery-cache:/root/.cache/bloomery \
#             ghcr.io/midagedev/bloomery:0.2.0 --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M
# (model downloads land in the bloomery-cache volume; arguments after the image name
# reach bloomery-serve itself).
FROM ubuntu:24.04
ARG VERSION=0.2.0
ARG SHA256=cca1271b093cd2610660c700dd3d48b1ded272b7c8f377faa7dc5a93d2b13de8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    asset="bloomery-${VERSION}-linux-x86_64-cuda-sm86.tar.gz"; \
    curl -fsSL -o "/tmp/${asset}" "https://github.com/midagedev/bloomery/releases/download/v${VERSION}/${asset}"; \
    echo "${SHA256}  /tmp/${asset}" | sha256sum -c -; \
    tar -xzf "/tmp/${asset}" -C /usr/local --strip-components=1; \
    rm "/tmp/${asset}"
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/bloomery-serve"]
CMD ["--port", "8080"]
