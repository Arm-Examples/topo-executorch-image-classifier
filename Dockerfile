FROM astral/uv:python3.12-bookworm-slim AS model-downloader

WORKDIR /downloader

COPY uv.lock pyproject.toml .python-version ./
RUN uv sync --locked --no-install-project --only-group downloader

ARG HF_ENDPOINT
ARG MODEL
ENV HF_ENDPOINT=${HF_ENDPOINT}
ENV MODEL=${MODEL}
COPY scripts/download_model.py scripts/download_model.py
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN uv run scripts/download_model.py

FROM astral/uv:python3.12-bookworm-slim AS runtime

WORKDIR /runtime

COPY uv.lock pyproject.toml .python-version ./
RUN uv sync --locked --no-install-project --only-group runtime

COPY --from=model-downloader /downloader/model model
COPY app app

CMD ["uv", "run", "app/server.py"]
