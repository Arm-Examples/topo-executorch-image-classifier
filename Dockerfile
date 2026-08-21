FROM alpine:3.24 AS model-downloader

RUN apk add --no-cache bash aria2 curl jq

COPY scripts/hfd.sh hfd.sh

ARG HF_ENDPOINT
ARG MODEL

ENV HF_ENDPOINT=${HF_ENDPOINT}
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN ./hfd.sh ${MODEL} --local-dir /downloader/model --exclude *fp32* --hf_token $HF_TOKEN

FROM astral/uv:python3.12-bookworm-slim AS runtime

WORKDIR /runtime

COPY uv.lock pyproject.toml .python-version ./
RUN uv sync --locked --no-install-project

COPY --from=model-downloader /downloader/model model
COPY app app

CMD ["uv", "run", "app/server.py"]
