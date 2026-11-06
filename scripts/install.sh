#!/usr/bin/env bash
# Rakkan Mesh installer for Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/optimist3301/rakkan-mesh/main/scripts/install.sh | bash
#
# Installs for the current user (no root needed): downloads rakkan-node,
# checks its checksum, writes a JSON config holding the node's settings,
# private key and certificate, and optionally starts it at login.
#
# Run as root, it installs for the whole system instead, after saying so:
# /usr/local/bin, the config in /etc/rakkan (Linux) or /Library/Application
# Support/Rakkan (macOS), and a service started at boot that runs the node
# as its own unprivileged user (rakkan, or _rakkan on macOS), not as root.
#
# Options:
#   --yes           accept the defaults without asking
#   --version V     release to install (default: see VERSION below)
#   --binary PATH   install this local binary instead of downloading
#   --uninstall     remove the service and the binary (add --purge to also
#                   delete the config, key and data)
#   --help
#
# Every question can be answered beforehand with an environment variable,
# which also makes the script usable unattended with --yes:
#   RAKKAN_LANG=en|ru  RAKKAN_NETWORK=public|<network ID>  RAKKAN_CERT=<file>
#   RAKKAN_NODE_TYPE=worker|client|relay|llm  RAKKAN_WORKERS=<n>
#   RAKKAN_STORE_GB=<n>  RAKKAN_MAINLINE=yes|no  RAKKAN_BOOTSTRAP=<multiaddr,...>
#   RAKKAN_AUTOSTART=yes|no

set -euo pipefail

REPO="optimist3301/rakkan-mesh"
VERSION="${RAKKAN_VERSION:-0.006-alpha}"
# The open network run by the project: anyone may join as a guest.
PUBLIC_NETWORK="12D3KooWJRPbdQ6Q8TXSctKQ8B4ZYHtKArqbKKReeQdx4iojLx7s"
SERVICE="rakkan-node"
LABEL="network.rakkan.node"
# The system install's user: the node never runs as root.
SYS_USER="rakkan"
SYSTEM=0 # set when run as root

YES=0
UNINSTALL=0
PURGE=0
LOCAL_BINARY=""
LANG_CODE="${RAKKAN_LANG:-}"

# ---------------------------------------------------------------- output ----

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  c() { printf '\033[38;5;%sm' "$1"; }
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
  c() { :; }
  BOLD=""; DIM=""; RESET=""
fi

t() { if [ "$LANG_CODE" = ru ]; then printf '%s' "$2"; else printf '%s' "$1"; fi; }
say()  { printf '%s\n' "$(t "$1" "$2")"; }
info() { printf '  %s•%s %s\n' "$(c 45)" "$RESET" "$(t "$1" "$2")"; }
ok()   { printf '  %s✔%s %s\n' "$(c 78)" "$RESET" "$(t "$1" "$2")"; }
warn() { printf '  %s!%s %s\n' "$(c 214)" "$RESET" "$(t "$1" "$2")" >&2; }
die()  { printf '\n  %s✘ %s%s\n' "$(c 203)" "$1" "$RESET" >&2; exit 1; }
step() { printf '\n%s%s[%s]%s %s%s%s\n' "$BOLD" "$(c 51)" "$1" "$RESET" "$BOLD" "$(t "$2" "$3")" "$RESET"; }

logo() {
  local l
  local -a art=(
    "██████╗  █████╗ ██╗  ██╗██╗  ██╗ █████╗ ███╗   ██╗"
    "██╔══██╗██╔══██╗██║ ██╔╝██║ ██╔╝██╔══██╗████╗  ██║"
    "██████╔╝███████║█████╔╝ █████╔╝ ███████║██╔██╗ ██║"
    "██╔══██╗██╔══██║██╔═██╗ ██╔═██╗ ██╔══██║██║╚██╗██║"
    "██║  ██║██║  ██║██║  ██╗██║  ██╗██║  ██║██║ ╚████║"
    "╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝"
  )
  local -a hue=(51 45 39 33 99 135)
  local i
  printf '\n'
  for i in "${!art[@]}"; do
    l=${art[$i]}
    printf '  %s%s%s\n' "$(c "${hue[$i]}")" "$l" "$RESET"
  done
  printf '  %s●━━━━●━━━━●━━━━●━━━━●%s  %s%sM E S H%s  %s●━━━━●━━━━●━━━━●━━━━●%s\n' \
    "$(c 135)" "$RESET" "$BOLD" "$(c 255)" "$RESET" "$(c 51)" "$RESET"
  printf '  %s%s%s\n\n' "$DIM" "closed-by-default compute mesh · v$VERSION" "$RESET"
}

# ----------------------------------------------------------------- input ----

INTERACTIVE=1
setup_input() {
  if [ "$YES" = 1 ]; then return; fi
  if [ -t 0 ]; then
    exec 3<&0
  elif { : </dev/tty; } 2>/dev/null; then
    exec 3</dev/tty
  elif [ -f "$0" ]; then
    exec 3<&0 # the script is a file, so stdin carries the answers
  else
    INTERACTIVE=0
  fi
  [ "$INTERACTIVE" = 1 ] || die "No terminal to ask questions on. Download the script and run it, or use --yes with RAKKAN_* variables."
}

readline() { IFS= read -r -u 3 "$1" || die "Input closed."; }

# ask_text VAR "English" "Русский" [default]
ask_text() {
  local __v=$1 en=$2 ru=$3 def=${4-} ans
  if [ "$YES" = 1 ]; then printf -v "$__v" '%s' "$def"; return; fi
  printf '  %s%s%s%s ' "$(c 255)" "$(t "$en" "$ru")" "$RESET" "${def:+ $DIM[$def]$RESET}"
  readline ans
  printf -v "$__v" '%s' "${ans:-$def}"
}

# ask_yn VAR "English" "Русский" y|n
ask_yn() {
  local __v=$1 en=$2 ru=$3 def=$4 ans hint="y/N"
  [ "$def" = y ] && hint="Y/n"
  if [ "$YES" = 1 ]; then printf -v "$__v" '%s' "$def"; return; fi
  while :; do
    printf '  %s%s%s %s[%s]%s ' "$(c 255)" "$(t "$en" "$ru")" "$RESET" "$DIM" "$hint" "$RESET"
    readline ans
    case "${ans:-$def}" in
      y|Y|yes|д|Д|да) printf -v "$__v" y; return ;;
      n|N|no|н|Н|нет) printf -v "$__v" n; return ;;
    esac
  done
}

# ask_choice VAR default "English question" "Русский вопрос" option...
# Options are "English|Русский". Stores the chosen number (1-based).
ask_choice() {
  local __v=$1 def=$2 en=$3 ru=$4 ans i=1 o
  shift 4
  printf '  %s%s%s\n' "$(c 255)" "$(t "$en" "$ru")" "$RESET"
  for o in "$@"; do
    printf '    %s%d)%s %s\n' "$(c 51)" "$i" "$RESET" "$(t "${o%%|*}" "${o##*|}")"
    i=$((i + 1))
  done
  if [ "$YES" = 1 ]; then printf -v "$__v" '%s' "$def"; return; fi
  while :; do
    printf '  %s>%s %s[%s]%s ' "$(c 51)" "$RESET" "$DIM" "$def" "$RESET"
    readline ans
    ans=${ans:-$def}
    if [[ $ans =~ ^[0-9]+$ ]] && [ "$ans" -ge 1 ] && [ "$ans" -le $# ]; then
      printf -v "$__v" '%s' "$ans"; return
    fi
  done
}

# ------------------------------------------------------------- utilities ----

jesc() { local s=$1; s=${s//\\/\\\\}; s=${s//\"/\\\"}; printf '%s' "$s"; }

valid_network() { [[ $1 =~ ^12D3KooW[1-9A-HJ-NP-Za-km-z]{44}$ ]]; }

have() { command -v "$1" >/dev/null 2>&1; }

sha256_of() {
  if have sha256sum; then sha256sum "$1" | cut -d' ' -f1
  elif have shasum; then shasum -a 256 "$1" | cut -d' ' -f1
  else die "Neither sha256sum nor shasum found."; fi
}

fetch() { # url dest
  if have curl; then curl -fL --retry 3 -sS -o "$2" "$1"
  elif have wget; then wget -q -O "$2" "$1"
  else die "Neither curl nor wget found."; fi
}

# ------------------------------------------------------------- platform -----

detect_platform() {
  case "$(uname -s)" in
    Linux)  OS=linux ;;
    Darwin) OS=darwin ;;
    *) die "Unsupported system: $(uname -s). On Windows use install.ps1." ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64)  ARCH=amd64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) die "Unsupported architecture: $(uname -m)" ;;
  esac
  [ "$(id -u)" = 0 ] && SYSTEM=1
  if [ "$SYSTEM" = 1 ]; then
    BIN_DIR="${RAKKAN_BIN_DIR:-/usr/local/bin}"
    if [ "$OS" = linux ]; then
      CONF_DIR=/etc/rakkan
      DATA_DIR=/var/lib/rakkan
      UNIT_DIR=/etc/systemd/system
    else
      SYS_USER="_rakkan"
      CONF_DIR="/Library/Application Support/Rakkan"
      DATA_DIR="$CONF_DIR/data"
      PLIST="/Library/LaunchDaemons/$LABEL.plist"
      LOG_FILE="/Library/Logs/rakkan-node.log"
    fi
    BIN="$BIN_DIR/rakkan-node"
    CONF="$CONF_DIR/node.json"
    return
  fi
  BIN_DIR="${RAKKAN_BIN_DIR:-$HOME/.local/bin}"
  if [ "$OS" = linux ]; then
    CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/rakkan"
    DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/rakkan"
    UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  else
    CONF_DIR="$HOME/Library/Application Support/Rakkan"
    DATA_DIR="$CONF_DIR/data"
    PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
    LOG_FILE="$HOME/Library/Logs/rakkan-node.log"
  fi
  BIN="$BIN_DIR/rakkan-node"
  CONF="$CONF_DIR/node.json"
}

# ------------------------------------------------------------- uninstall ----

do_uninstall() {
  detect_platform
  step "-" "Removing Rakkan node" "Удаление узла Rakkan"
  if [ "$SYSTEM" = 1 ]; then
    warn "Running as root: removing the system-wide install ($BIN, the boot service)." \
         "Запуск от root: удаляю общесистемную установку ($BIN, служба при загрузке)."
    if [ "$OS" = linux ]; then
      if have systemctl; then
        systemctl disable --now "$SERVICE.service" >/dev/null 2>&1 || true
      fi
      rm -f "$UNIT_DIR/$SERVICE.service"
      if have systemctl; then systemctl daemon-reload >/dev/null 2>&1 || true; fi
    else
      launchctl bootout "system/$LABEL" >/dev/null 2>&1 || true
      rm -f "$PLIST"
    fi
  elif [ "$OS" = linux ]; then
    if have systemctl; then
      systemctl --user disable --now "$SERVICE.service" >/dev/null 2>&1 || true
    fi
    rm -f "$UNIT_DIR/$SERVICE.service"
    have systemctl && systemctl --user daemon-reload >/dev/null 2>&1 || true
  else
    launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
    rm -f "$PLIST"
  fi
  rm -f "$BIN"
  ok "Service and binary removed." "Служба и программа удалены."
  if [ "$PURGE" = 1 ]; then
    local sure
    ask_yn sure "Delete $CONF_DIR including the node's private key?" "Удалить $CONF_DIR вместе с приватным ключом узла?" n
    if [ "$sure" = y ]; then
      rm -rf "$CONF_DIR" "$DATA_DIR"
      ok "Config, key and data deleted." "Конфиг, ключ и данные удалены."
      if [ "$SYSTEM" = 1 ]; then remove_system_user; fi
    else
      info "Config and key kept in $CONF_DIR." "Конфиг и ключ сохранены в $CONF_DIR."
    fi
  else
    info "Config and key kept in $CONF_DIR (use --purge to delete them)." \
         "Конфиг и ключ сохранены в $CONF_DIR (--purge удалит и их)."
  fi
}

# ---------------------------------------------------------- system install --

# confirm_system says, before anything is written, what installing as root
# does, and asks to go on.
confirm_system() {
  [ "$SYSTEM" = 1 ] || return 0
  warn "You are running this as root: Rakkan will be installed for the WHOLE SYSTEM, not for one user." \
       "Скрипт запущен от root: Rakkan будет установлен для ВСЕЙ СИСТЕМЫ, а не для одного пользователя."
  info "Program: $BIN" "Программа: $BIN"
  info "Config and node key: $CONF" "Конфиг и ключ узла: $CONF"
  info "Data: $DATA_DIR" "Данные: $DATA_DIR"
  info "Service: started at boot, running as the unprivileged user $SYS_USER (created if missing), not as root." \
       "Служба: стартует при загрузке, работает от непривилегированного пользователя $SYS_USER (будет создан, если его нет), а не от root."
  info "To install for yourself only, run the script as a normal user, without sudo." \
       "Чтобы установить только для себя, запустите скрипт обычным пользователем, без sudo."
  local go
  ask_yn go "Install for the whole system?" "Установить для всей системы?" y
  [ "$go" = y ] || die "Cancelled."
}

# ensure_system_user creates the user the node runs as.
ensure_system_user() {
  if id "$SYS_USER" >/dev/null 2>&1; then return; fi
  if [ "$OS" = linux ]; then
    local nologin=/usr/sbin/nologin
    [ -x "$nologin" ] || nologin=/sbin/nologin
    [ -x "$nologin" ] || nologin=/bin/false
    if have useradd; then
      useradd --system --user-group --home-dir "$DATA_DIR" --no-create-home --shell "$nologin" "$SYS_USER"
    elif have adduser; then # BusyBox, Alpine
      adduser -S -D -H -h "$DATA_DIR" -s "$nologin" "$SYS_USER"
    else
      die "Cannot create the user $SYS_USER: no useradd or adduser."
    fi
  else
    # A hidden system account with a free ID below 500, as macOS daemons have.
    local id=300
    while dscl . -list /Users UniqueID | awk '{print $2}' | grep -qx "$id" ||
      dscl . -list /Groups PrimaryGroupID | awk '{print $2}' | grep -qx "$id"; do
      id=$((id + 1))
      [ "$id" -lt 500 ] || die "No free system user ID for $SYS_USER."
    done
    dscl . -create "/Groups/$SYS_USER" PrimaryGroupID "$id"
    dscl . -create "/Users/$SYS_USER" UniqueID "$id"
    dscl . -create "/Users/$SYS_USER" PrimaryGroupID "$id"
    dscl . -create "/Users/$SYS_USER" UserShell /usr/bin/false
    dscl . -create "/Users/$SYS_USER" NFSHomeDirectory /var/empty
    dscl . -create "/Users/$SYS_USER" RealName "Rakkan node"
    dscl . -create "/Users/$SYS_USER" IsHidden 1
  fi
  ok "Created the user $SYS_USER the node runs as." "Создан пользователь $SYS_USER, от которого работает узел."
}

remove_system_user() {
  id "$SYS_USER" >/dev/null 2>&1 || return 0
  if [ "$OS" = linux ]; then
    userdel "$SYS_USER" >/dev/null 2>&1 || deluser "$SYS_USER" >/dev/null 2>&1 || true
  else
    dscl . -delete "/Users/$SYS_USER" >/dev/null 2>&1 || true
    dscl . -delete "/Groups/$SYS_USER" >/dev/null 2>&1 || true
  fi
  ok "Removed the user $SYS_USER." "Пользователь $SYS_USER удалён."
}

# install_system_service starts the node at boot as SYS_USER.
install_system_service() {
  if [ "$OS" = linux ]; then
    if ! have systemctl; then
      warn "No systemd: start the node by hand (command below)." "Нет systemd: запустите узел вручную (команда ниже)."
      return
    fi
    cat >"$UNIT_DIR/$SERVICE.service" <<UNIT
[Unit]
Description=Rakkan Mesh node
After=network-online.target
Wants=network-online.target

[Service]
User=$SYS_USER
ExecStart="$BIN" daemon -config "$CONF"
WorkingDirectory=$DATA_DIR
Restart=on-failure
RestartSec=5
# The node needs nothing of root's: no new privileges, the system read-only
# but for its data.
NoNewPrivileges=yes
ProtectSystem=full
ProtectHome=yes
ReadWritePaths=$DATA_DIR

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    systemctl enable --now "$SERVICE.service"
    sleep 2
    if systemctl is-active --quiet "$SERVICE.service"; then
      ok "The node is running and will start at boot." "Узел запущен и будет стартовать при загрузке."
    else
      warn "The service did not stay up. See: journalctl -u $SERVICE -e" \
           "Служба не удержалась. Смотрите: journalctl -u $SERVICE -e"
    fi
  else
    touch "$LOG_FILE"
    chown "$SYS_USER:" "$LOG_FILE"
    cat >"$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>UserName</key><string>$SYS_USER</string>
  <key>ProgramArguments</key>
  <array><string>$BIN</string><string>daemon</string><string>-config</string><string>$CONF</string></array>
  <key>WorkingDirectory</key><string>$DATA_DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG_FILE</string>
  <key>StandardErrorPath</key><string>$LOG_FILE</string>
</dict>
</plist>
PLIST_EOF
    chown root:wheel "$PLIST"
    chmod 644 "$PLIST"
    launchctl bootout "system/$LABEL" >/dev/null 2>&1 || true
    if launchctl bootstrap system "$PLIST" 2>/dev/null || launchctl load -w "$PLIST"; then
      ok "The node is running and will start at boot." "Узел запущен и будет стартовать при загрузке."
    else
      warn "Could not load $PLIST" "Не удалось загрузить $PLIST"
    fi
  fi
}

# --------------------------------------------------------------- install ----

install_binary() {
  mkdir -p "$BIN_DIR"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  if [ -n "$LOCAL_BINARY" ]; then
    [ -x "$LOCAL_BINARY" ] || die "Not an executable file: $LOCAL_BINARY"
    cp "$LOCAL_BINARY" "$TMP/rakkan-node"
    ok "Using $LOCAL_BINARY" "Использую $LOCAL_BINARY"
  else
    local name="rakkan-node-$VERSION-$OS-$ARCH" base
    base="https://github.com/$REPO/releases/download/v$VERSION"
    info "Downloading $name ..." "Скачиваю $name ..."
    fetch "$base/$name.tar.gz" "$TMP/$name.tar.gz" || die "Download failed: $base/$name.tar.gz"
    fetch "$base/SHA256SUMS" "$TMP/SHA256SUMS" || die "Could not download SHA256SUMS"
    local want got
    want=$(grep -F "  $name.tar.gz" "$TMP/SHA256SUMS" | cut -d' ' -f1 || true)
    [ -n "$want" ] || die "$name.tar.gz is not listed in SHA256SUMS"
    got=$(sha256_of "$TMP/$name.tar.gz")
    [ "$want" = "$got" ] || die "Checksum mismatch for $name.tar.gz (expected $want, got $got)"
    ok "Checksum verified." "Контрольная сумма совпала."
    tar -xzf "$TMP/$name.tar.gz" -C "$TMP"
    [ -f "$TMP/$name/rakkan-node" ] || die "Archive does not contain rakkan-node"
    mv "$TMP/$name/rakkan-node" "$TMP/rakkan-node"
  fi
  install -m 0755 "$TMP/rakkan-node" "$BIN"
  ok "Installed $BIN" "Установлено: $BIN"
  [ "$SYSTEM" = 1 ] && return
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not in your PATH; add it to your shell profile." \
            "$BIN_DIR нет в PATH; добавьте его в профиль оболочки." ;;
  esac
}

# Linux kernels may forbid the user namespaces strict isolation relies on
# (Ubuntu 24.04 and newer do by default).
strict_isolation_works() {
  [ "$OS" = linux ] || return 1
  have unshare || return 0
  local probe=(unshare -Urm sh -c 'mount --make-rprivate / 2>/dev/null')
  if [ "$SYSTEM" = 1 ]; then
    # Root may always; the service's unprivileged user is what counts.
    if have setpriv; then
      setpriv --reuid=65534 --regid=65534 --clear-groups "${probe[@]}" >/dev/null 2>&1
    elif have runuser; then
      runuser -u nobody -- "${probe[@]}" >/dev/null 2>&1
    else
      return 1
    fi
    return
  fi
  "${probe[@]}" >/dev/null 2>&1
}

main() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --yes|-y) YES=1 ;;
      --version) VERSION="${2:?--version needs a value}"; shift ;;
      --binary) LOCAL_BINARY="${2:?--binary needs a path}"; shift ;;
      --uninstall) UNINSTALL=1 ;;
      --purge) PURGE=1 ;;
      --help|-h) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
      *) die "Unknown option: $1" ;;
    esac
    shift
  done

  logo
  setup_input
  if [ -z "$LANG_CODE" ]; then
    case "${LANG:-}" in ru*) LANG_CODE=ru ;; *) LANG_CODE=en ;; esac
    if [ "$YES" != 1 ]; then
      local lc
      ask_choice lc "$([ "$LANG_CODE" = ru ] && echo 2 || echo 1)" "Language" "Язык" "English|English" "Русский|Русский"
      [ "$lc" = 2 ] && LANG_CODE=ru || LANG_CODE=en
    fi
  fi

  if [ "$UNINSTALL" = 1 ]; then do_uninstall; exit 0; fi
  detect_platform
  info "System: $OS/$ARCH" "Система: $OS/$ARCH"
  confirm_system

  # ------------------------------------------------------------ 1: network
  step "1/6" "Network" "Сеть"
  local NET
  if [ -n "${RAKKAN_NETWORK:-}" ]; then
    NET=$RAKKAN_NETWORK
    [ "$NET" = public ] && NET=$PUBLIC_NETWORK
  else
    local nc
    ask_choice nc 1 "Which network do you want to join?" "К какой сети подключиться?" \
      "The public open network (anyone can join) — $PUBLIC_NETWORK|Публичная открытая сеть (вход свободный) — $PUBLIC_NETWORK" \
      "My own network (I have its network ID)|Своя сеть (у меня есть её network ID)"
    if [ "$nc" = 1 ]; then
      NET=$PUBLIC_NETWORK
    else
      while :; do
        ask_text NET "Network ID of your network (12D3KooW...)" "Network ID вашей сети (12D3KooW...)" ""
        valid_network "$NET" && break
        warn "That does not look like a network ID." "Это не похоже на network ID."
        [ "$YES" = 1 ] && die "Invalid network ID."
      done
    fi
  fi
  valid_network "$NET" || die "Invalid network ID: $NET"
  local PUBLIC=0
  [ "$NET" = "$PUBLIC_NETWORK" ] && PUBLIC=1
  ok "Network: $NET" "Сеть: $NET"

  # keep an existing identity from being clobbered
  if [ -e "$CONF" ]; then
    warn "A config already exists: $CONF" "Конфиг уже существует: $CONF"
    local over
    ask_yn over "Replace it? (a backup is kept; the node gets a new identity)" \
      "Заменить? (копия сохранится; у узла будет новая личность)" n
    [ "$over" = y ] || die "Nothing changed."
  fi

  # --------------------------------------------------- 2: binary and key
  step "2/6" "Program and node identity" "Программа и личность узла"
  install_binary
  local KEYFILE="$TMP/node.key" PEER
  PEER=$("$BIN" id -key "$KEYFILE") || die "Could not create a node key."
  ok "Your peer ID: $PEER" "Ваш peer ID: $PEER"

  # ------------------------------------------------------------ 3: identity
  step "3/6" "How you join" "Как вы входите в сеть"
  local CERT_JSON="" cert_path="${RAKKAN_CERT:-}"
  local mode
  if [ -n "$cert_path" ]; then mode=2
  elif [ "$PUBLIC" = 1 ]; then
    info "The public network is open: you join as a guest and prove some work once." \
         "Публичная сеть открыта: вы входите гостем и один раз подтверждаете работу."
    mode=1
  else
    ask_choice mode 2 "How will this node join your network?" "Как узел войдёт в вашу сеть?" \
      "As a guest (only if the network is open to guests)|Гостем (только если сеть открыта для гостей)" \
      "As a member with a certificate from the network admin|Участником, по сертификату от администратора сети"
  fi
  if [ "$mode" = 2 ]; then
    if [ -z "$cert_path" ]; then
      info "Give this peer ID to the network admin, who issues a certificate for it:" \
           "Передайте этот peer ID администратору сети — он выпустит для него сертификат:"
      printf '      %s%s%s\n' "$BOLD" "$PEER" "$RESET"
      ask_text cert_path "Path to the certificate file (empty: join as a guest for now)" \
        "Путь к файлу сертификата (пусто: пока войти гостем)" ""
    fi
    if [ -n "$cert_path" ]; then
      cert_path="${cert_path/#\~/$HOME}"
      [ -f "$cert_path" ] || die "No such file: $cert_path"
      local shown
      shown=$("$BIN" network show "$cert_path") || die "$cert_path is not a valid certificate."
      grep -qF "$PEER" <<<"$shown" || die "This certificate was issued for a different node, not $PEER."
      grep -qF "$NET" <<<"$shown" || die "This certificate belongs to a different network."
      CERT_JSON=$(cat "$cert_path")
      ok "Certificate accepted." "Сертификат принят."
    else
      warn "Joining as a guest; this works only while the network is open." \
           "Вход гостем: работает, только пока сеть открыта."
    fi
  fi

  # -------------------------------------------------------------- 4: type
  step "4/6" "What this node does" "Что будет делать узел"
  local TYPE
  if [ -n "${RAKKAN_NODE_TYPE:-}" ]; then TYPE=$RAKKAN_NODE_TYPE
  else
    local tc
    ask_choice tc 1 "Node type" "Тип узла" \
      "Worker — runs tasks for the network and can submit its own|Рабочий — выполняет задачи сети и может ставить свои" \
      "Client — only submits tasks, runs nothing for others|Клиент — только ставит задачи, ничего не выполняет для других" \
      "Relay — publicly reachable hub that helps others connect|Ретранслятор — публично доступный узел, помогает другим соединиться" \
      "LLM worker — worker that also serves language-model stages (uses GPU if any)|LLM-узел — рабочий, который ещё обслуживает слои языковой модели (GPU, если есть)"
    TYPE=$(echo worker client relay llm | cut -d' ' -f"$tc")
  fi
  case "$TYPE" in worker|client|relay|llm) ;; *) die "Unknown node type: $TYPE" ;; esac

  local CPUS
  CPUS=$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2) | head -1)
  local WORKERS="${RAKKAN_WORKERS:-}" STORE_GB="${RAKKAN_STORE_GB:-}" LLM_MEM="" GPU="auto" PORT=""
  case "$TYPE" in
    client) WORKERS=0 ;;
    *) [ -n "$WORKERS" ] || ask_text WORKERS "Tasks to run at once (CPU cores: $CPUS)" "Сколько задач выполнять одновременно (ядер CPU: $CPUS)" "$CPUS" ;;
  esac
  [[ $WORKERS =~ ^[0-9]+$ ]] || die "Invalid number of workers: $WORKERS"
  [ -n "$STORE_GB" ] || ask_text STORE_GB "Disk budget for stored data, GiB" "Сколько диска отдать под данные, ГиБ" "10"
  [[ $STORE_GB =~ ^[0-9]+([.][0-9]+)?$ ]] || die "Invalid disk budget: $STORE_GB"
  if [ "$TYPE" = llm ]; then
    ask_text LLM_MEM "Memory for model weights and caches, GiB" "Память под веса модели и кэши, ГиБ" "4"
    [[ $LLM_MEM =~ ^[0-9]+([.][0-9]+)?$ ]] || die "Invalid memory size: $LLM_MEM"
    local gc
    ask_choice gc 1 "GPU for language models" "GPU для языковых моделей" \
      "Pick the best one automatically|Выбрать лучшую автоматически" "Do not use the GPU|Не использовать GPU"
    [ "$gc" = 2 ] && GPU=off
  fi
  if [ "$TYPE" = relay ]; then
    ask_text PORT "Port to listen on (open it in your firewall/router)" "Порт для входящих соединений (откройте его в файрволе/роутере)" "4001"
    [[ $PORT =~ ^[0-9]+$ ]] || die "Invalid port: $PORT"
  fi

  local ISOLATION="process"
  if [ "$OS" = linux ]; then
    if strict_isolation_works; then
      ISOLATION=strict
      ok "Strict task isolation is available." "Строгая изоляция задач доступна."
    else
      warn "This system forbids unprivileged user namespaces, so strict isolation (namespaces + seccomp) cannot run; using process isolation. To enable strict mode: sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0" \
           "Система запрещает user namespaces без привилегий, строгая изоляция (namespaces + seccomp) невозможна; использую изоляцию процессом. Чтобы включить строгую: sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0"
    fi
  else
    info "Strict task isolation exists only on Linux; using process isolation." \
         "Строгая изоляция есть только на Linux; использую изоляцию процессом."
  fi

  # ------------------------------------------------------- 5: connectivity
  step "5/6" "Finding the network" "Поиск сети"
  local MAINLINE="${RAKKAN_MAINLINE:-}"
  if [ -z "$MAINLINE" ]; then
    info "The node finds the network by its ID alone, in the BitTorrent DHT: the network's admin lists a few public members there. The records are encrypted, but DHT nodes see this machine's IP, as with any torrent client." \
         "Узел находит сеть по одному её ID — в DHT BitTorrent: администратор сети перечисляет там несколько публичных участников. Записи зашифрованы, но узлы DHT видят IP этой машины, как у любого торрент-клиента."
    ask_yn MAINLINE "Look for the network in the BitTorrent DHT?" "Искать сеть в DHT BitTorrent?" y
  fi
  case "$MAINLINE" in y|yes|true|1|on) MAINLINE=y ;; *) MAINLINE=n ;; esac
  local BOOT="${RAKKAN_BOOTSTRAP:-}"
  if [ -z "$BOOT" ]; then
    info "Known nodes are optional; they help when the DHT is blocked or the admin listed no one. Format: /ip4/203.0.113.7/udp/4001/quic-v1/p2p/12D3Koo..." \
         "Известные узлы не обязательны: они выручают, если DHT заблокирована или администратор никого не перечислил. Формат: /ip4/203.0.113.7/udp/4001/quic-v1/p2p/12D3Koo..."
    if [ "$MAINLINE" = y ]; then
      ask_text BOOT "Address(es) of known nodes, comma-separated (empty: the DHT is enough)" \
        "Адрес(а) известных узлов через запятую (пусто: хватит DHT)" ""
    else
      ask_text BOOT "Address(es) of known nodes, comma-separated (empty: only the local network)" \
        "Адрес(а) известных узлов через запятую (пусто: только локальная сеть)" ""
    fi
  fi
  if [ -n "$BOOT" ]; then
    local a
    IFS=',' read -ra _addrs <<<"$BOOT"
    for a in "${_addrs[@]}"; do
      [[ $a == */p2p/* ]] || die "Address must end in /p2p/<peer ID>: $a"
    done
  elif [ "$MAINLINE" = n ]; then
    if [ -z "$CERT_JSON" ]; then
      warn "A guest needs a known node or the DHT to learn the network policy; without either the node cannot start." \
           "Гостю нужен известный узел или DHT, чтобы узнать политику сети; без них узел не запустится."
    else
      warn "Without known nodes and the DHT, the node finds members only on the local network." \
           "Без известных узлов и DHT узел найдёт участников только в локальной сети."
    fi
  fi

  # ------------------------------------------------------------- 6: start
  step "6/6" "Autostart" "Автозапуск"
  local AUTO="${RAKKAN_AUTOSTART:-}" LINGER=n
  if [ -z "$AUTO" ]; then
    local what
    if [ "$SYSTEM" = 1 ]; then
      what=$([ "$OS" = linux ] && echo "systemd system service" || echo "launchd daemon")
      ask_yn AUTO "Start the node automatically at boot ($what, as $SYS_USER)?" "Запускать узел автоматически при загрузке ($what, от $SYS_USER)?" y
    else
      what=$([ "$OS" = linux ] && echo "systemd user service" || echo "launchd agent")
      ask_yn AUTO "Start the node automatically at login ($what)?" "Запускать узел автоматически при входе ($what)?" y
    fi
  fi
  case "$AUTO" in y|yes) AUTO=y ;; *) AUTO=n ;; esac
  if [ "$AUTO" = y ] && [ "$SYSTEM" = 0 ] && [ "$OS" = linux ] && have loginctl; then
    ask_yn LINGER "Keep it running from boot, even when you are not logged in?" \
      "Держать запущенным с загрузки системы, даже без вашего входа?" n
  fi

  # ---------------------------------------------------------- write config
  mkdir -p "$CONF_DIR" "$DATA_DIR"
  chmod 700 "$CONF_DIR"
  if [ -e "$CONF" ]; then
    cp -p "$CONF" "$CONF.bak-$(date +%Y%m%d%H%M%S)"
  fi
  local KEY_B64
  KEY_B64=$(base64 <"$KEYFILE" | tr -d '\n')
  local -a F=()
  F+=("  \"network\": \"$(jesc "$NET")\"")
  F+=("  \"private_key\": \"$KEY_B64\"")
  [ -z "$CERT_JSON" ] || F+=("  \"certificate\": $CERT_JSON")
  F+=("  \"isolation\": \"$ISOLATION\"")
  F+=("  \"workers\": $WORKERS")
  F+=("  \"store-gb\": $STORE_GB")
  F+=("  \"data\": \"$(jesc "$DATA_DIR")\"")
  F+=("  \"api\": \"$(jesc "$DATA_DIR/rakkan.sock")\"")
  F+=("  \"peers\": \"$(jesc "$DATA_DIR/peers.json")\"")
  F+=("  \"proof\": \"$(jesc "$DATA_DIR/node.proof")\"")
  [ -z "$BOOT" ] || F+=("  \"bootstrap\": \"$(jesc "$BOOT")\"")
  if [ "$MAINLINE" = y ]; then F+=("  \"mainline\": true"); else F+=("  \"mainline\": false"); fi
  if [ "$TYPE" = relay ]; then
    F+=("  \"relay\": true")
    F+=("  \"reachability\": \"public\"")
    F+=("  \"listen\": \"/ip4/0.0.0.0/udp/$PORT/quic-v1,/ip4/0.0.0.0/tcp/$PORT,/ip6/::/udp/$PORT/quic-v1,/ip6/::/tcp/$PORT\"")
  fi
  if [ "$TYPE" = llm ]; then
    F+=("  \"llm\": true")
    F+=("  \"llm-mem-gb\": $LLM_MEM")
    F+=("  \"gpu\": \"$GPU\"")
  fi
  (
    umask 077
    {
      printf '{\n'
      local i last=$((${#F[@]} - 1))
      for i in "${!F[@]}"; do
        if [ "$i" -lt "$last" ]; then printf '%s,\n' "${F[$i]}"; else printf '%s\n' "${F[$i]}"; fi
      done
      printf '}\n'
    } >"$CONF"
  )
  chmod 600 "$CONF"
  if [ "$SYSTEM" = 1 ]; then
    # The node's own user owns its key, config and data; only it reads them.
    ensure_system_user
    chown "$SYS_USER:" "$CONF"
    chown -R "$SYS_USER:" "$DATA_DIR"
    chmod 700 "$DATA_DIR"
    chmod 755 "$CONF_DIR"
    if [ "$OS" = linux ] && [ "$TYPE" = llm ]; then
      # GPUs: /dev/dri and /dev/kfd belong to these groups.
      local g
      for g in video render; do
        if getent group "$g" >/dev/null 2>&1; then usermod -aG "$g" "$SYS_USER" 2>/dev/null || true; fi
      done
    fi
  fi
  ok "Config written: $CONF (owner-only: it holds the node's private key)" \
     "Конфиг записан: $CONF (доступ только владельцу: внутри приватный ключ узла)"

  # --------------------------------------------------------------- service
  if [ "$AUTO" = y ] && [ "$SYSTEM" = 1 ]; then
    install_system_service
  elif [ "$AUTO" = y ]; then
    if [ "$OS" = linux ]; then
      if have systemctl && systemctl --user show-environment >/dev/null 2>&1; then
        mkdir -p "$UNIT_DIR"
        cat >"$UNIT_DIR/$SERVICE.service" <<EOF
[Unit]
Description=Rakkan Mesh node
After=network-online.target
Wants=network-online.target

[Service]
ExecStart="$BIN" daemon -config "$CONF"
WorkingDirectory=$DATA_DIR
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
        systemctl --user daemon-reload
        systemctl --user enable --now "$SERVICE.service"
        [ "$LINGER" = y ] && loginctl enable-linger "$USER" || true
        sleep 2
        if systemctl --user is-active --quiet "$SERVICE.service"; then
          ok "The node is running and will start at login." "Узел запущен и будет стартовать при входе."
        else
          warn "The service did not stay up. See: journalctl --user -u $SERVICE -e" \
               "Служба не удержалась. Смотрите: journalctl --user -u $SERVICE -e"
        fi
      else
        warn "systemd user services are not available here; start the node by hand (command below)." \
             "Пользовательские службы systemd здесь недоступны; запустите узел вручную (команда ниже)."
      fi
    else
      mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG_FILE")"
      cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$BIN</string><string>daemon</string><string>-config</string><string>$CONF</string></array>
  <key>WorkingDirectory</key><string>$DATA_DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG_FILE</string>
  <key>StandardErrorPath</key><string>$LOG_FILE</string>
</dict>
</plist>
EOF
      launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
      if launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl load -w "$PLIST"; then
        ok "The node is running and will start at login." "Узел запущен и будет стартовать при входе."
      else
        warn "Could not load $PLIST" "Не удалось загрузить $PLIST"
      fi
    fi
  fi

  # --------------------------------------------------------------- summary
  printf '\n%s%s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n' "$BOLD" "$(c 99)" "$RESET"
  say "  Done. Welcome to the mesh!" "  Готово. Добро пожаловать в сеть!"
  printf '  %s%s%s %s\n' "$DIM" "peer ID:" "$RESET" "$PEER"
  printf '  %s%s%s %s\n' "$DIM" "network:" "$RESET" "$NET"
  printf '  %s%s%s %s\n' "$DIM" "config: " "$RESET" "$CONF"
  printf '\n'
  say "  Run by hand:" "  Запуск вручную:"
  if [ "$SYSTEM" = 1 ]; then
    printf '    sudo -u %s "%s" daemon -config "%s"\n' "$SYS_USER" "$BIN" "$CONF"
  else
    printf '    %s daemon -config "%s"\n' "$BIN" "$CONF"
  fi
  if [ "$AUTO" = y ] && [ "$SYSTEM" = 1 ]; then
    if [ "$OS" = linux ]; then
      say "  Logs:  journalctl -u $SERVICE -f" "  Журнал:  journalctl -u $SERVICE -f"
    else
      say "  Logs:  tail -f \"$LOG_FILE\"" "  Журнал:  tail -f \"$LOG_FILE\""
    fi
  elif [ "$AUTO" = y ]; then
    if [ "$OS" = linux ]; then
      say "  Logs:  journalctl --user -u $SERVICE -f" "  Журнал:  journalctl --user -u $SERVICE -f"
    else
      say "  Logs:  tail -f \"$LOG_FILE\"" "  Журнал:  tail -f \"$LOG_FILE\""
    fi
  fi
  if [ "$TYPE" = relay ] && [ "$MAINLINE" = y ]; then
    printf '\n'
    say "  This node is public: the network admin can list it in the DHT, so that newcomers find the network through it:" \
        "  Этот узел публичный: администратор сети может перечислить его в DHT, чтобы новички находили сеть через него:"
    printf '    rakkan-node network announce -key root.key -announcers %s\n' "$PEER"
  fi
  if [ "$SYSTEM" = 1 ]; then
    say "  Remove: sudo install.sh --uninstall" "  Удалить: sudo install.sh --uninstall"
  else
    say "  Remove: install.sh --uninstall" "  Удалить: install.sh --uninstall"
  fi
  printf '%s%s━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%s\n\n' "$BOLD" "$(c 99)" "$RESET"
}

main "$@"
