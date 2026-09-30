#!/usr/bin/env bash
# Set up a BITS Prayogshala lab VM (Rocky Linux 9) for this project. Safe to re-run.
#
#   bash scripts/vm_bootstrap.sh          # setup (run once per fresh VM)
#   bash scripts/vm_bootstrap.sh --check  # print versions + connectivity only
#
# Only ~/workspace survives VM termination, so durable things live there:
#   ~/workspace/.cache/pip      pip download cache  -> fast venv rebuild
#   ~/workspace/.cache/bin      checksum-verified kubectl download
#   ~/workspace/.config/git     git identity + credential store
# Rebuilt on every fresh VM (local disk, fast): make, the Python venv,
# /usr/local/bin/kubectl, and a marked block in ~/.bashrc.
set -euo pipefail

WS="${WORKSPACE_DIR:-$HOME/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV_DIR:-$HOME/.venvs/heartrisk}"
PY_BASE="${PYTHON_BASE:-python3}"
# minikube v1.35 defaults to Kubernetes v1.32; kubectl should match within one minor.
KUBECTL_MINOR="${KUBECTL_MINOR:-1.32}"
MODE="${1:-setup}"
BEGIN_MARK="# >>> heartrisk lab-vm >>>"
END_MARK="# <<< heartrisk lab-vm <<<"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
can_sudo() { sudo -n true 2>/dev/null; }

pkg_install() {  # pkg_install <package>...
  if ! can_sudo; then warn "no sudo: install manually: $*"; return 0; fi
  if have dnf; then sudo dnf install -y -q "$@"
  elif have yum; then sudo yum install -y -q "$@"
  elif have apt-get; then sudo apt-get update -qq && sudo apt-get install -y -qq "$@"
  else warn "no known package manager: install manually: $*"; fi
}

step_dirs() {
  mkdir -p "$WS/.cache/pip" "$WS/.cache/bin" "$WS/.config/git"
  ok "persistent dirs under $WS"
}

step_system_packages() {
  local missing=()
  for tool in make curl git; do have "$tool" || missing+=("$tool"); done
  if ((${#missing[@]})); then log "installing: ${missing[*]}"; pkg_install "${missing[@]}"; fi
  ok "make/curl/git present"
}

step_git() {
  # Keep the global git config on persistent storage via a symlink.
  local cfg="$WS/.config/git/config"
  touch "$cfg"
  if [ -e "$HOME/.gitconfig" ] && [ ! -L "$HOME/.gitconfig" ]; then
    mv "$HOME/.gitconfig" "$HOME/.gitconfig.pre-bootstrap"
  fi
  ln -sfn "$cfg" "$HOME/.gitconfig"
  if ! git config --global --get credential.helper >/dev/null; then
    git config --global credential.helper "store --file=$WS/.config/git/credentials"
  fi
  git config --global init.defaultBranch main
  git config --global pull.rebase false
  if git config --global --get user.name >/dev/null && git config --global --get user.email >/dev/null; then
    ok "git identity: $(git config --global user.name) <$(git config --global user.email)>"
  else
    warn "git identity not set yet (one-time; it persists in $cfg):"
    warn "  git config --global user.name  \"Your Name\""
    warn "  git config --global user.email \"you@example.com\""
  fi
}

step_docker() {
  if ! have docker; then warn "docker not installed; skipping"; return 0; fi
  if docker info >/dev/null 2>&1; then ok "docker daemon reachable as $(id -un)"; return 0; fi
  if can_sudo; then
    sudo systemctl start docker 2>/dev/null || true
    if ! id -nG | tr ' ' '\n' | grep -qx docker; then
      sudo usermod -aG docker "$(id -un)"
      warn "added $(id -un) to the docker group: run 'newgrp docker', then re-run this script"
      return 0
    fi
  fi
  docker info >/dev/null 2>&1 && ok "docker daemon reachable" || warn "docker daemon not reachable"
}

fetch_kubectl() {  # fetch_kubectl <dest>: download + verify sha256, non-zero on failure
  local ver dest="$1" sum
  ver="$(curl -fsSL -m 20 "https://dl.k8s.io/release/stable-${KUBECTL_MINOR}.txt")" || return 1
  log "downloading kubectl $ver (sha256-verified)"
  curl -fsSL -m 120 -o "$dest.tmp" "https://dl.k8s.io/release/${ver}/bin/linux/amd64/kubectl" || return 1
  sum="$(curl -fsSL -m 20 "https://dl.k8s.io/release/${ver}/bin/linux/amd64/kubectl.sha256")" || return 1
  if ! echo "$sum  $dest.tmp" | sha256sum --check --status; then
    rm -f "$dest.tmp"; warn "kubectl checksum mismatch"; return 1
  fi
  chmod +x "$dest.tmp" && mv "$dest.tmp" "$dest"
}

step_kubectl() {
  if have kubectl; then ok "kubectl present: $(kubectl version --client 2>/dev/null | head -1)"; return 0; fi
  local cache="$WS/.cache/bin/kubectl" dir="$HOME/.local/bin"
  can_sudo && dir=/usr/local/bin
  if [ -x "$cache" ] || fetch_kubectl "$cache"; then
    if [ "$dir" = /usr/local/bin ]; then sudo install -m 0755 "$cache" "$dir/kubectl"
    else mkdir -p "$dir" && install -m 0755 "$cache" "$dir/kubectl"; fi
    ok "kubectl installed: $("$dir/kubectl" version --client 2>/dev/null | head -1)"
  elif have minikube; then
    warn "kubectl download failed; installing a 'minikube kubectl --' wrapper"
    printf '#!/usr/bin/env bash\nexec minikube kubectl -- "$@"\n' > "$HOME/.kubectl-wrapper"
    if [ "$dir" = /usr/local/bin ]; then sudo install -m 0755 "$HOME/.kubectl-wrapper" "$dir/kubectl"
    else mkdir -p "$dir" && install -m 0755 "$HOME/.kubectl-wrapper" "$dir/kubectl"; fi
  else
    warn "kubectl unavailable (download failed and minikube missing)"
  fi
}

step_python() {
  if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys' 2>/dev/null; then
    log "creating venv $VENV from $($PY_BASE --version 2>&1)"
    rm -rf "$VENV"
    "$PY_BASE" -m venv "$VENV"
  fi
  local pyv
  pyv="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  [ "$pyv" = "3.12" ] || warn "venv uses Python $pyv; the project targets 3.12"
  export PIP_CACHE_DIR="$WS/.cache/pip"
  local stamp="$VENV/.requirements.sha256" want
  want="$(cat "$REPO_DIR/requirements.txt" "$REPO_DIR/pyproject.toml" | sha256sum | cut -d' ' -f1)"
  if [ "$(cat "$stamp" 2>/dev/null)" != "$want" ]; then
    log "installing requirements.txt (pip cache: $PIP_CACHE_DIR)"
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q -r "$REPO_DIR/requirements.txt"
    "$VENV/bin/python" -m pip install -q --no-deps -e "$REPO_DIR"
    echo "$want" > "$stamp"
  fi
  ok "venv ready: $VENV (Python $pyv)"
}

step_bashrc() {
  local rc="$HOME/.bashrc" tmp
  touch "$rc"
  tmp="$(mktemp)"
  # Drop any previous block, then append the current one (keeps re-runs idempotent).
  awk -v b="$BEGIN_MARK" -v e="$END_MARK" '$0==b{skip=1;next} $0==e{skip=0;next} !skip' "$rc" > "$tmp"
  cat >> "$tmp" <<RC
$BEGIN_MARK
export PIP_CACHE_DIR="$WS/.cache/pip"
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
alias hr='cd "$REPO_DIR"'
$END_MARK
RC
  cat "$tmp" > "$rc" && rm -f "$tmp"
  ok ".bashrc: venv auto-activates in new terminals; 'hr' jumps to the repo"
}

report() {
  echo
  log "versions"
  printf '  %-10s %s\n' os "$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
  printf '  %-10s %s\n' cpu/ram "$(nproc) CPU / $(free -h | awk '/^Mem:/{print $2}')"
  printf '  %-10s %s\n' python "$("$VENV/bin/python" --version 2>/dev/null || echo 'venv missing')"
  printf '  %-10s %s\n' git "$(git --version 2>/dev/null || echo missing)"
  printf '  %-10s %s\n' docker "$(docker --version 2>/dev/null || echo missing)"
  printf '  %-10s %s\n' minikube "$(minikube version --short 2>/dev/null || echo missing)"
  printf '  %-10s %s\n' kubectl "$(kubectl version --client 2>/dev/null | head -1 || echo missing)"
  log "connectivity (401 = reachable, login required)"
  local u code
  for u in archive.ics.uci.edu github.com pypi.org/simple/ registry-1.docker.io/v2/ registry.k8s.io/v2/; do
    code="$(curl -s -o /dev/null -m 10 -w '%{http_code}' "https://$u" || true)"
    [ "$code" = 000 ] && code=unreachable
    printf '  %-26s %s\n' "$u" "$code"
  done
}

main() {
  case "$MODE" in
    --check) report ;;
    setup)
      step_dirs; step_system_packages; step_git; step_docker; step_kubectl
      step_python; step_bashrc; report
      echo; ok "done. Open a new terminal (or: source ~/.bashrc), then: hr && make help" ;;
    *) echo "usage: $0 [--check]"; exit 2 ;;
  esac
}
main
