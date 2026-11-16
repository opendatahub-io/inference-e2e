FROM registry.access.redhat.com/ubi9/python-311:latest

# UBI base for downstream alignment and to avoid docker.io rate limits.
# The UBI python image defaults to UID 1001; run build steps as root (same
# as the previous python:3.11-slim behavior) and rely on the chmod below
# for non-root runtime use.
USER root

ARG MANIFEST_REF=main
ARG MANIFEST_REPO=https://github.com/opendatahub-io/llm-d-conformance-manifests.git

RUN dnf install -y --setopt=install_weak_deps=False \
    curl git && \
    curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install kubectl /usr/local/bin/kubectl && rm kubectl && \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh && \
    dnf clean all && rm -rf /var/cache/dnf

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .
RUN rm -rf .venv && uv sync --frozen

RUN git clone --depth 1 --branch "${MANIFEST_REF}" "${MANIFEST_REPO}" /tmp/manifests && \
    mkdir -p deploy/manifests && \
    cp /tmp/manifests/*.yaml deploy/manifests/ && \
    printf 'branch: %s\nrepo: %s\n' "${MANIFEST_REF}" "${MANIFEST_REPO}" > deploy/manifests/.manifest-ref && \
    rm -rf /tmp/manifests

# Set up directories and permissions so non-root users can invoke the CLI
# with default runtime locations.
RUN mkdir -p reports .pytest_cache && \
    chmod -R a+rX /app && \
    chmod -R a+rwX deploy/manifests reports .pytest_cache

# UV_NO_SYNC prevents updates to the root-owned virtual environment, avoiding
# permission errors when non-root users invoke the image. This is acceptable
# because all dependencies are installed at image build time.
# Set the runtime cache after build steps so it is created by the runtime UID.
ENV UV_NO_SYNC=1 \
    UV_CACHE_DIR=/tmp/uv-cache

ENTRYPOINT ["uv", "run", "llm-d-e2e"]
