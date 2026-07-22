#!/usr/bin/env bash
#
# deploy-image.sh — pull the CI-built gpu-service image on every GPU node and
# prove each node is actually running it.
#
# Why this exists: the gpu-exchange gpu-agent spawns the CCTV container with a
# plain `docker run $CCTV_IMAGE` and never pulls. Docker only fetches an image
# it does not already have locally, so a node holding an older `:latest` keeps
# serving stale engine code indefinitely — and nothing anywhere says so. That
# is not hypothetical: issue #96 shipped a behaviour change that stayed inert
# on both boxes until `docker pull` was run by hand.
#
# What it checks:
#   1. the target commit HAS a CI-built image      (ghcr tag sha-<short>)
#   2. `:latest` — the tag the agents run — points at that same digest
#   3. after pulling, every node's local `:latest` resolves to that digest
# Any mismatch is a non-zero exit, so this is safe to wire into a deploy chain.
#
# Usage:
#   bash scripts/deploy-image.sh                      # deploy HEAD to the default nodes
#   bash scripts/deploy-image.sh --ref 84392f9        # deploy a specific commit's build
#   bash scripts/deploy-image.sh --nodes "cctv-vps"   # subset of nodes
#   bash scripts/deploy-image.sh --check              # verify only, pull nothing
#   GPU_NODES="a b" bash scripts/deploy-image.sh      # nodes via environment
#
# Needs: git, curl, ssh access to each node. No docker or gh auth on this box —
# digests come from the GHCR registry API, which serves this package anonymously.
set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-kopalniekrypto/cctv-gpu-engine/gpu-service}"
REGISTRY="${REGISTRY:-ghcr.io}"
DEFAULT_NODES="cctv-vps cctv-vps-2"
# The system socket every non-interactive docker call on cctv-vps needs (its
# rootless daemon is broken). Tried only if the node's default context fails.
FALLBACK_DOCKER_HOST="unix:///var/run/docker.sock"

log() { printf '[deploy-image] %s\n' "$*"; }
die() { printf '[deploy-image] ERROR: %s\n' "$*" >&2; exit 1; }

# Digest a node/registry must agree on, or the deploy did not take.
# verdict <node> <expected-digest> <actual-digest>
verdict() {
  local node="$1" expected="$2" actual="$3"
  if [ -z "$actual" ]; then
    printf '  %-14s MISSING  (image not present — pull failed?)\n' "$node"
    return 1
  fi
  if [ "$actual" != "$expected" ]; then
    printf '  %-14s STALE    %s\n' "$node" "$actual"
    return 1
  fi
  printf '  %-14s OK       %s\n' "$node" "$actual"
  return 0
}

# Digest GHCR serves for a tag, or empty if the tag does not exist.
# registry_digest <tag>
registry_digest() {
  local tag="$1" token
  token=$(curl -sf "https://${REGISTRY}/token?scope=repository:${IMAGE_REPO}:pull&service=${REGISTRY}" |
    sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
  [ -n "$token" ] || die "could not get a pull token for ${REGISTRY}/${IMAGE_REPO}"
  curl -sI -H "Authorization: Bearer $token" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json' \
    "https://${REGISTRY}/v2/${IMAGE_REPO}/manifests/${tag}" |
    tr -d '\r' | sed -n 's/^[Dd]ocker-[Cc]ontent-[Dd]igest: //p'
}

# Run a docker command on a node, auto-selecting a socket that works there.
# remote_docker <node> <docker-args...>
remote_docker() {
  local node="$1"; shift
  ssh -o ConnectTimeout=20 "$node" \
    "docker version >/dev/null 2>&1 || export DOCKER_HOST=$FALLBACK_DOCKER_HOST; docker $*"
}

main() {
  local ref="" nodes="${GPU_NODES:-$DEFAULT_NODES}" check_only=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --ref) ref="${2:?--ref needs a commit-ish}"; shift 2 ;;
      --nodes) nodes="${2:?--nodes needs a space-separated list}"; shift 2 ;;
      --check) check_only=1; shift ;;
      -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) die "unknown argument: $1 (try --help)" ;;
    esac
  done

  local short
  short=$(git rev-parse --short "${ref:-HEAD}") || die "not a commit: ${ref:-HEAD}"

  # 1. Did CI build this commit at all?
  local expected
  expected=$(registry_digest "sha-${short}")
  [ -n "$expected" ] || die "no image published for sha-${short}.
       Is the commit pushed, and has 'Build Docker images' finished?
       Watch it with: gh run list --limit 3"

  # 2. Does :latest — what CCTV_IMAGE points at on every node — mean that commit?
  local latest
  latest=$(registry_digest latest)
  [ "$latest" = "$expected" ] ||
    die "sha-${short} is published, but :latest is a different image.
       :latest    $latest
       sha-$short $expected
       You are not on the newest build. Deploy the newest commit, or retag :latest."

  log "deploying ${IMAGE_REPO}:latest @ ${expected} (commit ${short})"
  log "nodes: ${nodes}"

  # 3. Pull on every node, then make each one prove what it is holding.
  local failed=0 node actual
  for node in $nodes; do
    if [ "$check_only" -eq 0 ]; then
      log "pulling on ${node} (15+ GB — first pull on a fresh box is slow)"
      remote_docker "$node" "pull -q ${REGISTRY}/${IMAGE_REPO}:latest" >/dev/null ||
        log "pull on ${node} failed — verifying what it still holds"
    fi
    # RepoDigests is the registry identity of the image; the local image ID is
    # not comparable across hosts (a containerd-backed store reports the
    # manifest digest, the classic store the config digest).
    actual=$(remote_docker "$node" \
      "inspect --format '{{index .RepoDigests 0}}' ${REGISTRY}/${IMAGE_REPO}:latest 2>/dev/null" |
      sed -n 's/.*@//p')
    verdict "$node" "$expected" "$actual" || failed=1
  done

  if [ "$failed" -ne 0 ]; then
    die "at least one node is not on the CI image — the agent there will keep running old code."
  fi
  log "all nodes on the CI image for ${short}. New tasks pick it up on next spawn."
  log "note: a task already running keeps the image it started with."
}

# Sourced with DEPLOY_IMAGE_LIB=1 (tests) → define functions, deploy nothing.
if [ "${DEPLOY_IMAGE_LIB:-}" != "1" ]; then
  main "$@"
fi
