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
#   ./publish.sh              # publish rstrt-sys (if needed), then publish rstrt
#   ./publish.sh --dry-run    # validate packaging only, upload nothing
#   ./publish.sh --yes        # skip the interactive confirmation
#
# Requires: `cargo login` to have been run (or a token in ~/.cargo/credentials).
#
# Notes:
#  - `rstrt` can only be packaged/validated once `rstrt-sys` exists in the
#    registry (cargo resolves its path dependency against the registry during
#    publish). So `rstrt` is validated after `rstrt-sys` is live.
#  - Re-running is safe: a `rstrt-sys` version already on crates.io is skipped
#    (crates.io versions are immutable and cannot be re-published).

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

# Read a crate's declared version from its manifest (first `version = "..."`).
crate_version() {
  sed -n 's/^version[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$1/Cargo.toml" | head -1
}

# Query crates.io directly (bypasses the local registry mirror that blocks
# `cargo search`). Returns 0 if $1@version $2 is published, non-zero otherwise.
is_published() {
  curl -sf -A "rstrt-publish" "https://crates.io/api/v1/crates/$1" 2>/dev/null \
    | grep -q "\"num\":\"$2\""
}

SYS_VERSION=$(crate_version rstrt-sys)
RSTRT_VERSION=$(crate_version rstrt)

SYS_LIVE=0
if is_published rstrt-sys "$SYS_VERSION"; then
  SYS_LIVE=1
fi

# --- pre-flight ------------------------------------------------------------

# Only rstrt-sys can be validated up front; rstrt needs it live in the registry
# first (see note in the header).
if (( SYS_LIVE )); then
  echo "==> rstrt-sys v${SYS_VERSION} is already on crates.io (skipping its publish)"
else
  echo "==> validating rstrt-sys (manifest + packaging)"
  cargo publish -p rstrt-sys $REG --dry-run
fi

if (( DRY_RUN )); then
  echo "==> dry-run complete, nothing was uploaded."
  echo "    (rstrt not validated: it requires rstrt-sys in the registry.)"
  exit 0
fi

if (( ! ASSUME_YES )); then
  read -r -p "Publish rstrt-sys then rstrt to crates.io? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 0; }
fi

# --- publish in dependency order ------------------------------------------

if (( SYS_LIVE )); then
  echo "==> rstrt-sys already live in the registry, continuing to rstrt"
else
  echo "==> publishing rstrt-sys"
  cargo publish -p rstrt-sys $REG

  # Wait until rstrt-sys is visible in the registry so rstrt can resolve it.
  echo "==> waiting for rstrt-sys to sync to crates.io ..."
  for i in $(seq 1 40); do
    if is_published rstrt-sys "$SYS_VERSION"; then
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
fi

# Now that rstrt-sys is resolvable from the registry, validate then publish rstrt.
echo "==> validating rstrt (manifest + packaging)"
cargo publish -p rstrt $REG --dry-run

echo "==> publishing rstrt"
cargo publish -p rstrt $REG

echo
echo "Done. rstrt-sys v${SYS_VERSION} and rstrt v${RSTRT_VERSION} are on crates.io."
