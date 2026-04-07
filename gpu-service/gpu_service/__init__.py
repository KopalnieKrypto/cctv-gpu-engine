"""GPU service: R2 polling worker that runs the pipeline on pending jobs.

This package lives inside the ``gpu-service/`` docker build context (see
issue #5). The Python module name uses an underscore so it is importable; the
parent directory keeps a hyphen because that is the Docker convention used in
``.github/workflows/docker.yml`` and the README.
"""
