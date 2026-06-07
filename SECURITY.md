# Security Policy

QWIP Atlas is a research repository. There is no formal vulnerability response
process yet.

If you find a security issue:

- Do not open a public issue with exploit details.
- Prefer a private GitHub security advisory or direct contact with the maintainer.
- Include the affected file, a short description, and a minimal reproduction if
  available.

The main security risks in this repo are:

- Hardcoded tokens or secrets in scripts.
- Accidental publication of local model weights, prompts, or atlas artifacts.
- Running untrusted model code with `trust_remote_code=True` without checking it.

When in doubt, keep secrets out of the tree and prefer local configuration.
