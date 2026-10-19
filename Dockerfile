# FailSight web app: React frontend built in a Node stage, served together with the FastAPI backend by one Python process.
# The trained models (models/) and processed/preprocessor.joblib are NOT part of the image or of git: mount them (see docker-compose.yml).
#
#   docker build -t failsight .                      # with TensorFlow (all deep models + stacking)
#   docker build -t failsight --build-arg WITH_DEEP=0 .   # tree / linear models only, much smaller image

FROM node:22-alpine AS frontend
WORKDIR /build/app/frontend
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ARG WITH_DEEP=1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FAILSIGHT_MODELS_DIR=/data/models \
    FAILSIGHT_PREPROCESSOR=/data/preprocessor.joblib
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY app/backend/requirements.txt app/backend/requirements-deep.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && if [ "$WITH_DEEP" = "1" ]; then pip install --no-cache-dir -r /tmp/requirements-deep.txt; fi
# code and the small committed result files the API serves (no data, no models)
COPY scripts/ ./scripts/
COPY app/backend/ ./app/backend/
COPY report_outputs/ ./report_outputs/
COPY results/v1_16epoch/ ./results/v1_16epoch/
COPY processed/feature_names.json ./processed/feature_names.json
COPY --from=frontend /build/app/frontend/dist ./app/frontend/dist
RUN find /srv/app/backend -name tests -prune -exec rm -rf {} + || true
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
