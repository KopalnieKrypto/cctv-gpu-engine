# Security Policy

## Supported Versions

Only `main` is supported.

## Reporting a Vulnerability

Do not report vulnerabilities through public GitHub issues. Email piotr@sobiecki.org
with details and, if possible, steps to reproduce. Allow reasonable time for a fix
before public disclosure.

## What to Report

- Access to another tenant's input or result objects (presigned URL handling, `tenant_url.py`)
- Unauthenticated access to the gpu-service REST or dashboard endpoints
- Appliance token, R2 credentials or RTSP camera credentials exposed in code, logs or `result.json`
- Footage of people leaking from the recording buffer, uploads or retained inputs
- Command injection through RTSP URLs, camera names or ffmpeg arguments
- Supply-chain issues in the Docker image (unverified model downloads, unpinned dependencies)

## Response

We aim to acknowledge receipt within 48 hours, give an initial assessment within
1 week and release a fix as soon as practical.

## For Contributors

- Never commit secrets or credentials; read them from the environment
- Validate input at every network boundary
