#!/usr/bin/env bash
# Set up a Linux shell for this project. Safe to re-run.
#   Local PC (primary): WSL2 Ubuntu 24.04 + Docker Desktop (+ its Kubernetes)
#   BITS lab VM (alternative / clean-machine proof): Rocky Linux 9, minikube
#
#   bash scripts/vm_bootstrap.sh          # setup
#   bash scripts/vm_bootstrap.sh --check  # print versions + connectivity only
#
# Environment is auto-detected (override: BOOTSTRAP_ENV=wsl|labvm|linux).
# Lab VM only: ~/workspace is the one persistent folder, so pip cache, kubectl
# download and git config are kept there and everything else is rebuilt.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV_DIR:-$HOME/.venvs/heartrisk}"
PY_BASE="${PYTHON_BASE:-python3}"
MODE="${1:-setup}"
BEGIN_MARK="# >>> heartrisk >>>"
END_MARK="# <<< heartrisk <<<"
OLD_BEGIN="# >>> heartrisk lab-vm >>>"
OLD_END="# <<< heartrisk lab-vm <<<"

detect_env() {
  if [ -n "${BOOTSTRAP_ENV:-}" ]; then echo "$BOOTSTRAP_ENV"
  elif grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then echo wsl
  elif [ -d "$HOME/workspace" ]; then echo labvm
  else echo linux; fi
}
ENVIRONMENT="$(detect_env)"
if [ "$ENVIRONMENT" = labvm ]; then
  WS="${WORKSPACE_DIR:-$HOME/workspace}"
  CACHE_DIR="$WS/.cache"
  KUBECTL_MINOR="${KUBECTL_MINOR:-1.32}"   # minikube v1.35 -> Kubernetes v1.32
else
  CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/heartrisk"
  KUBECTL_MINOR="${KUBECTL_MINOR:-1.36}"   # Docker Desktop cluster runs Kubernetes v1.36
fi

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

SUDO_STATE=""
ensure_sudo() {  # succeeds if sudo is usable; asks for the password once in a terminal
  if [ -z "$SUDO_STATE" ]; then
    if sudo -n true 2>/dev/null; then SUDO_STATE=yes
    elif [ -t 0 ] && have sudo && sudo -v; then SUDO_STATE=yes
    else SUDO_STATE=no; fi
  fi
  [ "$SUDO_STATE" = yes ]
}

pkg_install() {  # pkg_install <package>...
  if ! ensure_sudo; then warn "no sudo: install manually: $*"; return 0; fi
  if have apt-get; then
    sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@"
  elif have dnf; then sudo dnf install -y -q "$@"
  elif have yum; then sudo yum install -y -q "$@"
  else warn "no known package manager: install manually: $*"; fi
}

step_dirs() {
  mkdir -p "$CACHE_DIR/bin"
  if [ "$ENVIRONMENT" = labvm ]; then mkdir -p "$CACHE_DIR/pip" "$WS/.config/git"; fi
  ok "environment: $ENVIRONMENT (cache: $CACHE_DIR)"
}

step_system_packages() {
  local missing=()
  for tool in make curl git; do have "$tool" || missing+=("$tool"); done
  if ! "$PY_BASE" -c 'import ensurepip, venv' 2>/dev/null; then
    if have apt-get; then missing+=(python3-venv); else warn "$PY_BASE has no venv/ensurepip module"; fi
  fi
  if ((${#missing[@]})); then log "installing: ${missing[*]}"; pkg_install "${missing[@]}"; fi
  ok "system packages: make/curl/git/venv present"
}

step_git() {
  if [ "$ENVIRONMENT" = wsl ]; then
    # Reuse the Windows git identity and Git Credential Manager, so commits and
    # pushes made from WSL just work. Git for Windows ships GCM under ucrt64/
    # (2.5x and later) or mingw64/ (older releases).
    local name email gcm="" candidate
    for candidate in "/mnt/c/Program Files/Git/ucrt64/bin" "/mnt/c/Program Files/Git/mingw64/bin"; do
      if [ -f "$candidate/git-credential-manager.exe" ]; then
        gcm="$candidate/git-credential-manager.exe"; break
      fi
    done
    if ! git config --global --get user.name >/dev/null && have git.exe; then
      name="$(git.exe config --global user.name 2>/dev/null | tr -d '\r' || true)"
      email="$(git.exe config --global user.email 2>/dev/null | tr -d '\r' || true)"
      if [ -n "$name" ]; then git config --global user.name "$name"; fi
      if [ -n "$email" ]; then git config --global user.email "$email"; fi
    fi
    if ! git config --global --get credential.helper >/dev/null && [ -n "$gcm" ]; then
      git config --global credential.helper "${gcm// /\\ }"
    fi
    git config --global init.defaultBranch main
    git config --global pull.rebase false
    if git config --global --get user.name >/dev/null; then
      ok "git identity: $(git config --global user.name) <$(git config --global user.email)>"
    else
      warn "git identity not set: git config --global user.name/user.email"
    fi
    if git config --global --get credential.helper >/dev/null; then
      ok "git credentials: Windows Git Credential Manager"
    else
      warn "no git credential helper (Git for Windows not found): pushes will ask for a token"
    fi
    return 0
  fi
  if [ "$ENVIRONMENT" != labvm ]; then
    ok "git: using the existing global git config"
    return 0
  fi
  # Lab VM: keep the global git config on persistent storage via a symlink.
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
  if [ "$ENVIRONMENT" = wsl ]; then
    if ! have docker; then
      warn "docker CLI missing: Docker Desktop > Settings > Resources > WSL integration > enable this distro > Apply & restart"
      return 0
    fi
    local err
    if err="$(docker info 2>&1 >/dev/null)"; then ok "docker: Docker Desktop engine reachable from WSL"
    elif echo "$err" | grep -qi 'permission denied'; then
      local grp; grp="$(stat -c %G /var/run/docker.sock 2>/dev/null || echo docker)"
      if [ "$grp" != root ] && ensure_sudo; then
        getent group "$grp" >/dev/null || sudo groupadd "$grp"
        sudo usermod -aG "$grp" "$(id -un)"
        warn "added $(id -un) to group '$grp' for the Docker socket. Close Ubuntu, run in PowerShell:"
        warn "  wsl --terminate ${WSL_DISTRO_NAME:-Ubuntu-24.04}"
        warn "then reopen Ubuntu and re-run this script"
      else
        warn "docker socket permission denied (socket group: $grp): check Docker Desktop's WSL integration"
      fi
    else warn "docker CLI present but the engine is unreachable: is Docker Desktop running?"; fi
    return 0
  fi
  if ! have docker; then warn "docker not installed; skipping"; return 0; fi
  if docker info >/dev/null 2>&1; then ok "docker daemon reachable as $(id -un)"; return 0; fi
  if ensure_sudo; then
    sudo systemctl start docker 2>/dev/null || true
    if ! id -nG | tr ' ' '\n' | grep -qx docker; then
      sudo usermod -aG docker "$(id -un)"
      warn "added $(id -un) to the docker group: run 'newgrp docker', then re-run this script"
      return 0
    fi
  fi
  docker info >/dev/null 2>&1 && ok "docker daemon reachable" || warn "docker daemon not reachable"
}

step_kubeconfig() {  # WSL: reuse Docker Desktop's kubeconfig from the Windows profile
  [ "$ENVIRONMENT" = wsl ] || return 0
  if [ -e "$HOME/.kube/config" ]; then ok "kubeconfig: $HOME/.kube/config"; return 0; fi
  local winprofile="" wincfg
  if have cmd.exe && have wslpath; then
    winprofile="$(cd /mnt/c 2>/dev/null && cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r' || true)"
  fi
  if [ -z "$winprofile" ]; then warn "cannot locate the Windows profile; link ~/.kube/config manually"; return 0; fi
  wincfg="$(wslpath -u "$winprofile")/.kube/config"
  if [ -f "$wincfg" ]; then
    mkdir -p "$HOME/.kube" && ln -sfn "$wincfg" "$HOME/.kube/config"
    ok "kubeconfig: linked to Docker Desktop's $wincfg"
  else
    warn "no $wincfg yet: enable Kubernetes in Docker Desktop, then re-run this script"
  fi
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

install_bin() {  # install_bin <src> <name>: to /usr/local/bin with sudo, else ~/.local/bin
  if ensure_sudo; then sudo install -m 0755 "$1" "/usr/local/bin/$2"; echo "/usr/local/bin/$2"
  else mkdir -p "$HOME/.local/bin" && install -m 0755 "$1" "$HOME/.local/bin/$2"; echo "$HOME/.local/bin/$2"; fi
}

step_kubectl() {
  if have kubectl; then ok "kubectl present: $(kubectl version --client 2>/dev/null | head -1)"; return 0; fi
  local cache="$CACHE_DIR/bin/kubectl" dest
  if [ -x "$cache" ] || fetch_kubectl "$cache"; then
    dest="$(install_bin "$cache" kubectl)"
    ok "kubectl installed: $dest ($("$dest" version --client 2>/dev/null | head -1))"
  elif have minikube; then
    warn "kubectl download failed; installing a 'minikube kubectl --' wrapper"
    printf '#!/usr/bin/env bash\nexec minikube kubectl -- "$@"\n' > "$CACHE_DIR/bin/kubectl-wrapper"
    install_bin "$CACHE_DIR/bin/kubectl-wrapper" kubectl >/dev/null
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
  if [ "$ENVIRONMENT" = labvm ]; then export PIP_CACHE_DIR="$CACHE_DIR/pip"; fi
  local stamp="$VENV/.requirements.sha256" want
  want="$(cat "$REPO_DIR/requirements.txt" "$REPO_DIR/pyproject.toml" | sha256sum | cut -d' ' -f1)"
  if [ "$(cat "$stamp" 2>/dev/null)" != "$want" ]; then
    log "installing requirements.txt into the venv (first time takes a few minutes)"
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q -r "$REPO_DIR/requirements.txt"
    echo "$want" > "$stamp"
  fi
  # Put src/ on the venv path with a .pth file instead of `pip install -e .`,
  # so nothing (egg-info) is written into the repo on the Windows drive.
  local site
  site="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "$REPO_DIR/src" > "$site/heartrisk-src.pth"
  "$VENV/bin/python" -c 'import heartrisk' || { warn "heartrisk is not importable from $REPO_DIR/src"; return 1; }
  ok "venv ready: $VENV (Python $pyv)"
}

step_bashrc() {
  local rc="$HOME/.bashrc" tmp
  touch "$rc"
  tmp="$(mktemp)"
  # Drop any previous block (current or old marker), then append the current one.
  awk -v b="$BEGIN_MARK" -v e="$END_MARK" -v ob="$OLD_BEGIN" -v oe="$OLD_END" \
    '$0==b||$0==ob{skip=1;next} $0==e||$0==oe{skip=0;next} !skip' "$rc" > "$tmp"
  {
    echo "$BEGIN_MARK"
    if [ "$ENVIRONMENT" = labvm ]; then echo "export PIP_CACHE_DIR=\"$CACHE_DIR/pip\""; fi
    echo "[ -f \"$VENV/bin/activate\" ] && . \"$VENV/bin/activate\""
    echo "alias hr='cd \"$REPO_DIR\"'"
    echo "$END_MARK"
  } >> "$tmp"
  cat "$tmp" > "$rc" && rm -f "$tmp"
  ok ".bashrc: venv auto-activates in new terminals; 'hr' jumps to the repo"
}

report() {
  local k8s engine
  echo
  log "versions"
  printf '  %-10s %s\n' env "$ENVIRONMENT"
  printf '  %-10s %s\n' os "$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
  printf '  %-10s %s\n' cpu/ram "$(nproc) CPU / $(free -h | awk '/^Mem:/{print $2}')"
  printf '  %-10s %s\n' python "$("$VENV/bin/python" --version 2>/dev/null || echo 'venv missing')"
  printf '  %-10s %s\n' git "$(git --version 2>/dev/null || echo missing)"
  engine="$(docker version --format '{{.Server.Version}}' 2>/dev/null || true)"
  printf '  %-10s %s\n' docker "$(docker --version 2>/dev/null || echo missing)${engine:+ | engine $engine}"
  printf '  %-10s %s\n' minikube "$(minikube version --short 2>/dev/null || echo 'not installed')"
  printf '  %-10s %s\n' kubectl "$(kubectl version --client 2>/dev/null | head -1 || echo missing)"
  k8s="$(kubectl get nodes --no-headers --request-timeout=5s 2>/dev/null | awk '{print $1, $2, $5}' | head -1 || true)"
  printf '  %-10s %s\n' cluster "${k8s:-no cluster reachable}"
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
      step_dirs; step_system_packages; step_git; step_docker; step_kubeconfig; step_kubectl
      step_python; step_bashrc; report
      echo; ok "done. Open a new terminal (or: source ~/.bashrc), then: hr && make help" ;;
    *) echo "usage: $0 [--check]"; exit 2 ;;
  esac
}
main
