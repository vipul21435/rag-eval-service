# Security Policy

## Supported versions

Security fixes are applied to the latest release and the current `main` branch. Older releases may not receive backports.

| Version | Supported |
| --- | --- |
| current `main` | Yes |
| upstream Local PDF Chat RAG releases | No; report those upstream |

## Reporting a vulnerability

Please do not disclose a vulnerability in a public issue, discussion, pull request, or social post.

Use GitHub's private vulnerability reporting flow:

1. Open the repository's **Security** tab.
2. Select **Report a vulnerability**.
3. Include the affected version, reproduction steps, impact, and any suggested mitigation.

The maintainer will acknowledge a valid report as soon as practical, investigate it, and coordinate disclosure after a fix is available. Please do not include real API keys, private documents, personal data, or third-party credentials in the report.

## Security scope

This educational project processes user-supplied documents and can call third-party model and search APIs. Operators are responsible for access control, secret management, data classification, dependency review, network policy, and deployment hardening in their own environment.

The defaults are local-first: the API binds `127.0.0.1`, sends no CORS headers, caps upload size (`MAX_UPLOAD_MB`) and, when `API_TOKEN` is set, requires a bearer token on every `/api` request. See "Exposing the API" in the README before changing `API_HOST` or `CORS_ALLOW_ORIGINS`.
