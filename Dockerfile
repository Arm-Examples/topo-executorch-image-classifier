FROM astral/uv:python3.12-bookworm-slim AS model-downloader

WORKDIR /downloader

RUN apt-get update && apt-get install -y --no-install-recommends aria2

COPY uv.lock pyproject.toml .python-version ./
RUN uv sync --locked --no-install-project --only-group downloader
COPY scripts/download_model.py scripts/download_model.py

ARG HF_ENDPOINT
ENV HF_ENDPOINT=${HF_ENDPOINT}
ARG HF_REPO_ID
ENV HF_REPO_ID=${HF_REPO_ID}
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN uv run scripts/download_model.py

FROM astral/uv:python3.12-bookworm-slim AS runtime

WORKDIR /runtime

COPY uv.lock pyproject.toml .python-version ./
RUN uv sync --locked --no-install-project --only-group runtime

COPY --from=model-downloader /downloader/model model
COPY app app

CMD ["uv", "run", "app/server.py"]
