#!/usr/bin/env bash
#
# Publish the rstrt workspace to crates.io in the correct dependency order.
#
# crates.io requires that a crate's dependencies already exist on the registry
# at publish time, so `rstrt` (which depends on `rstrt-sys`) can only go out
# after `rstrt-sys` has been published and synced. This script handles that
# ordering and waits for the sync automatically.
#
# Usage:
#   ./publish.sh              # dry-run both, then confirm before real publish
#   ./publish.sh --dry-run    # only validate, upload nothing
#   ./publish.sh --yes        # skip the interactive confirmation
#
# Requires: CARGO_TOKEN in the environment (e.g. `export CARGO_TOKEN=...`).

set -euo pipefail

# Publish from the workspace root so both crates are addressable with -p.
cd "$(dirname "$0")"

DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    --yes|-y)     ASSUME_YES=1 ;;
    -h|--help)    grep '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# registry flag: this box redirects crates-io to a mirror, so name it explicitly
# (dry-run + publish both need it). Remove --registry if you don't use a mirror.
REG="--registry crates-io"

# --- pre-flight ------------------------------------------------------------

if [[ -z "${CARGO_TOKEN:-}" ]]; then
  if (( DRY_RUN )); then
    echo "note: CARGO_TOKEN not set (fine for --dry-run)"
  else
    echo "error: CARGO_TOKEN is not set. Export it first, e.g.:" >&2
    echo "       export CARGO_TOKEN=\$(cargo login)   # or read it from https://crates.io/me" >&2
    exit 1
  fi
fi

echo "==> dry-run validating both crates (manifest + packaging)"
cargo publish -p rstrt-sys $REG --dry-run
cargo publish -p rstrt     $REG --dry-run

if (( DRY_RUN )); then
  echo "==> dry-run complete, nothing was uploaded."
  exit 0
fi

if (( ! ASSUME_YES )); then
  read -r -p "Publish rstrt-sys then rstrt to crates.io? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 0; }
fi

# --- publish in dependency order ------------------------------------------

echo "==> publishing rstrt-sys"
cargo publish -p rstrt-sys $REG

# Wait until rstrt-sys is visible in the registry so rstrt can resolve it.
echo "==> waiting for rstrt-sys to sync to crates.io ..."
for i in $(seq 1 40); do
  if cargo search rstrt-sys --limit 1 >/dev/null 2>&1; then
    echo "    rstrt-sys is live in the registry (after ~$((i * 15))s)"
    break
  fi
  if (( i == 40 )); then
    echo "error: rstrt-sys did not appear in the registry after 10m." >&2
    echo "       It may still be syncing; re-run ./publish.sh to finish." >&2
    exit 1
  fi
  sleep 15
done

echo "==> publishing rstrt"
cargo publish -p rstrt $REG

# Read a crate's declared version from its manifest (first `version = "..."`).
crate_version() {
  sed -n 's/^version[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$1/Cargo.toml" | head -1
}

SYS_VERSION=$(crate_version rstrt-sys)
RSTRT_VERSION=$(crate_version rstrt)

echo
echo "Done. Published rstrt-sys v${SYS_VERSION} and rstrt v${RSTRT_VERSION} to crates.io."
