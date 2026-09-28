#!/usr/bin/env bash
# Deploy this app to Google Cloud Run.
#
# Prerequisites:
#   gcloud auth login                 # a PERSONAL account, not a work one
#   gcloud config set project <id>    # a project with billing enabled
#
# Cloud Build builds the Dockerfile remotely, so no local Docker is needed.
set -euo pipefail

SERVICE="${SERVICE:-cras}"
REGION="${REGION:-us-central1}"          # free tier applies in select US regions
MEMORY="${MEMORY:-1Gi}"                  # measured peak is ~543 MB

# Gradio holds queue state in memory: a job is registered by one request and
# its result collected by a later one. If those land on different instances the
# second gets "404: Not Found" and the UI spins forever, which is what happens
# on platforms that spread requests freely. Session affinity plus a ceiling of
# one instance keeps a client talking to the process that holds its job.

PROJECT="$(gcloud config get-value project 2>/dev/null)"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || {
  echo "No project set. Run: gcloud config set project <project-id>" >&2; exit 1; }

# Keys are read from .env and passed as environment variables, never baked into
# the image. Secret Manager is the better home for them long term.
set -a; [ -f .env ] && . ./.env; set +a

ENV_VARS="BERT_MODEL_PATH=rajeev60/bert-mini-cyber,HF_HOME=/tmp/hf,APP_WRITABLE_DIR=/tmp"
for name in OPENAI_API_KEY GOOGLE_API_KEY DEEPSEEK_API_KEY MISTRAL_API_KEY ANTHROPIC_API_KEY; do
  value="${!name-}"
  [ -n "$value" ] && ENV_VARS="$ENV_VARS,$name=$value"
done

echo "Deploying $SERVICE to $PROJECT ($REGION)..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --memory "$MEMORY" \
  --cpu 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 1 \
  --session-affinity \
  --allow-unauthenticated \
  --set-env-vars "$ENV_VARS"

echo
echo "URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format="value(status.url)"
