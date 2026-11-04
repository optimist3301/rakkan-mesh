<#
.SYNOPSIS
  Rakkan Mesh installer for Windows.

.DESCRIPTION
  Installs for the current user (no administrator rights needed): downloads
  rakkan-node, checks its checksum, writes a JSON config holding the node's
  settings, private key and certificate, and optionally starts the node at
  logon through a scheduled task.

    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/optimist3301/rakkan-mesh/main/scripts/install.ps1).TrimStart([char]0xFEFF)))

  (A plain "irm ... | iex" does not work with scripts that take parameters.)
  Or download it and pass options:

    .\install.ps1 [-Yes] [-Version 0.002-alpha] [-Binary C:\path\rakkan-node.exe] [-Uninstall [-Purge]]

  Every question can be answered beforehand with an environment variable,
  which also makes the script usable unattended with -Yes:
    RAKKAN_LANG=en|ru  RAKKAN_NETWORK=public|<network ID>  RAKKAN_CERT=<file>
    RAKKAN_NODE_TYPE=worker|client|relay|llm  RAKKAN_WORKERS=<n>
    RAKKAN_STORE_GB=<n>  RAKKAN_MAINLINE=yes|no  RAKKAN_BOOTSTRAP=<multiaddr,...>
    RAKKAN_AUTOSTART=yes|no  RAKKAN_ADD_TO_PATH=yes|no
#>
[CmdletBinding()]
param(
    [switch]$Yes,
    [string]$Version = $(if ($env:RAKKAN_VERSION) { $env:RAKKAN_VERSION } else { '0.002-alpha' }),
    [string]$Binary = '',
    [switch]$Uninstall,
    [switch]$Purge
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }

$Repo = 'optimist3301/rakkan-mesh'
# The open network run by the project: anyone may join as a guest.
$PublicNetwork = '12D3KooWJRPbdQ6Q8TXSctKQ8B4ZYHtKArqbKKReeQdx4iojLx7s'
$TaskName = 'RakkanNode'

$ConfDir = Join-Path $env:LOCALAPPDATA 'Rakkan'
$BinDir = Join-Path $ConfDir 'bin'
$DataDir = Join-Path $ConfDir 'data'
$Bin = Join-Path $BinDir 'rakkan-node.exe'
$Conf = Join-Path $ConfDir 'node.json'
$LogFile = Join-Path $ConfDir 'rakkan-node.log'

$Lang = $env:RAKKAN_LANG

# ---------------------------------------------------------------- output ----

function T([string]$en, [string]$ru) { if ($script:Lang -eq 'ru') { $ru } else { $en } }
function Say([string]$en, [string]$ru) { Write-Host (T $en $ru) }
function Info([string]$en, [string]$ru) { Write-Host '  • ' -ForegroundColor Cyan -NoNewline; Write-Host (T $en $ru) }
function Ok([string]$en, [string]$ru) { Write-Host '  ✔ ' -ForegroundColor Green -NoNewline; Write-Host (T $en $ru) }
function Warn([string]$en, [string]$ru) { Write-Host '  ! ' -ForegroundColor Yellow -NoNewline; Write-Host (T $en $ru) }
function Die([string]$msg) { Write-Host ''; Write-Host "  ✘ $msg" -ForegroundColor Red; throw $msg }
function Step([string]$n, [string]$en, [string]$ru) {
    Write-Host ''
    Write-Host "[$n] " -ForegroundColor Cyan -NoNewline
    Write-Host (T $en $ru) -ForegroundColor White
}

function Show-Logo {
    $art = @(
        '██████╗  █████╗ ██╗  ██╗██╗  ██╗ █████╗ ███╗   ██╗',
        '██╔══██╗██╔══██╗██║ ██╔╝██║ ██╔╝██╔══██╗████╗  ██║',
        '██████╔╝███████║█████╔╝ █████╔╝ ███████║██╔██╗ ██║',
        '██╔══██╗██╔══██║██╔═██╗ ██╔═██╗ ██╔══██║██║╚██╗██║',
        '██║  ██║██║  ██║██║  ██╗██║  ██╗██║  ██║██║ ╚████║',
        '╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝'
    )
    $hue = @('Cyan', 'Cyan', 'Cyan', 'DarkCyan', 'Magenta', 'Magenta')
    Write-Host ''
    for ($i = 0; $i -lt $art.Count; $i++) { Write-Host ('  ' + $art[$i]) -ForegroundColor $hue[$i] }
    Write-Host '  ●━━━━●━━━━●━━━━●━━━━●  ' -ForegroundColor Magenta -NoNewline
    Write-Host 'M E S H' -ForegroundColor White -NoNewline
    Write-Host '  ●━━━━●━━━━●━━━━●━━━━●' -ForegroundColor Cyan
    Write-Host "  closed-by-default compute mesh · v$Version" -ForegroundColor DarkGray
    Write-Host ''
}

# ----------------------------------------------------------------- input ----

function Read-Line {
    $a = Read-Host
    if ($null -eq $a) { Die 'Input closed.' }
    return $a.Trim()
}

function Ask-Text([string]$en, [string]$ru, [string]$default = '') {
    if ($Yes) { return $default }
    $hint = ''
    if ($default -ne '') { $hint = " [$default]" }
    Write-Host ("  " + (T $en $ru) + $hint + ' ') -NoNewline
    $a = Read-Line
    if ($a -eq '') { return $default }
    return $a
}

function Ask-YN([string]$en, [string]$ru, [bool]$default) {
    if ($Yes) { return $default }
    $hint = 'y/N'
    if ($default) { $hint = 'Y/n' }
    while ($true) {
        Write-Host ("  " + (T $en $ru) + " [$hint] ") -NoNewline
        $a = (Read-Line).ToLower()
        if ($a -eq '') { return $default }
        if ($a -in @('y', 'yes', 'д', 'да')) { return $true }
        if ($a -in @('n', 'no', 'н', 'нет')) { return $false }
    }
}

# Options are "English|Русский". Returns the chosen number (1-based).
function Ask-Choice([string]$en, [string]$ru, [string[]]$options, [int]$default) {
    Write-Host ("  " + (T $en $ru))
    for ($i = 0; $i -lt $options.Count; $i++) {
        $parts = $options[$i] -split '\|', 2
        Write-Host ('    ') -NoNewline
        Write-Host ("$($i + 1))") -ForegroundColor Cyan -NoNewline
        Write-Host (' ' + (T $parts[0] $parts[1]))
    }
    if ($Yes) { return $default }
    while ($true) {
        Write-Host '  > ' -ForegroundColor Cyan -NoNewline
        Write-Host "[$default] " -ForegroundColor DarkGray -NoNewline
        $a = Read-Line
        if ($a -eq '') { return $default }
        $n = 0
        if ([int]::TryParse($a, [ref]$n) -and $n -ge 1 -and $n -le $options.Count) { return $n }
    }
}

function Test-Network([string]$id) { return $id -match '^12D3KooW[1-9A-HJ-NP-Za-km-z]{44}$' }

function Write-Utf8NoBom([string]$path, [string]$text) {
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
}

function Stop-Node {
    try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
    Get-Process -Name 'rakkan-node' -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $Bin } | Stop-Process -Force -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------- uninstall ----

function Remove-Node {
    Step '-' 'Removing Rakkan node' 'Удаление узла Rakkan'
    Stop-Node
    try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue } catch { }
    Remove-Item -Force -ErrorAction SilentlyContinue $Bin
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath) {
        $kept = ($userPath -split ';' | Where-Object { $_ -and $_ -ne $BinDir }) -join ';'
        [Environment]::SetEnvironmentVariable('Path', $kept, 'User')
    }
    Ok 'Task and binary removed.' 'Задача и программа удалены.'
    if ($Purge) {
        if (Ask-YN "Delete $ConfDir including the node's private key?" "Удалить $ConfDir вместе с приватным ключом узла?" $false) {
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ConfDir
            Ok 'Config, key and data deleted.' 'Конфиг, ключ и данные удалены.'
        }
    }
    else {
        Info "Config and key kept in $ConfDir (use -Purge to delete them)." "Конфиг и ключ сохранены в $ConfDir (-Purge удалит и их)."
    }
}

# ------------------------------------------------------------------ main ----

Show-Logo

if (-not $Lang) {
    $Lang = 'en'
    if ((Get-Culture).TwoLetterISOLanguageName -eq 'ru') { $Lang = 'ru' }
    if (-not $Yes) {
        $def = 1
        if ($Lang -eq 'ru') { $def = 2 }
        $lc = Ask-Choice 'Language' 'Язык' @('English|English', 'Русский|Русский') $def
        if ($lc -eq 2) { $Lang = 'ru' } else { $Lang = 'en' }
    }
}

if ($Uninstall) { Remove-Node; return }

$osArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
switch ($osArch) {
    'X64' { $Arch = 'amd64' }
    'Arm64' { $Arch = 'arm64' }
    default { Die "Unsupported architecture: $osArch" }
}
Info "System: windows/$Arch" "Система: windows/$Arch"

# ---------------------------------------------------------------- 1: network
Step '1/6' 'Network' 'Сеть'
if ($env:RAKKAN_NETWORK) {
    $Net = $env:RAKKAN_NETWORK
    if ($Net -eq 'public') { $Net = $PublicNetwork }
}
else {
    $nc = Ask-Choice 'Which network do you want to join?' 'К какой сети подключиться?' @(
        "The public open network (anyone can join) — $PublicNetwork|Публичная открытая сеть (вход свободный) — $PublicNetwork",
        'My own network (I have its network ID)|Своя сеть (у меня есть её network ID)') 1
    if ($nc -eq 1) { $Net = $PublicNetwork }
    else {
        while ($true) {
            $Net = Ask-Text 'Network ID of your network (12D3KooW...)' 'Network ID вашей сети (12D3KooW...)' ''
            if (Test-Network $Net) { break }
            Warn 'That does not look like a network ID.' 'Это не похоже на network ID.'
            if ($Yes) { Die 'Invalid network ID.' }
        }
    }
}
if (-not (Test-Network $Net)) { Die "Invalid network ID: $Net" }
$IsPublic = ($Net -eq $PublicNetwork)
Ok "Network: $Net" "Сеть: $Net"

if (Test-Path $Conf) {
    Warn "A config already exists: $Conf" "Конфиг уже существует: $Conf"
    if (-not (Ask-YN 'Replace it? (a backup is kept; the node gets a new identity)' 'Заменить? (копия сохранится; у узла будет новая личность)' $false)) {
        Die 'Nothing changed.'
    }
}

# --------------------------------------------------------- 2: binary and key
Step '2/6' 'Program and node identity' 'Программа и личность узла'
Stop-Node
New-Item -ItemType Directory -Force -Path $BinDir, $DataDir | Out-Null
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('rakkan-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
try {
    if ($Binary) {
        if (-not (Test-Path $Binary)) { Die "No such file: $Binary" }
        Copy-Item -Force $Binary $Bin
        Ok "Using $Binary" "Использую $Binary"
    }
    else {
        $name = "rakkan-node-$Version-windows-$Arch"
        $base = "https://github.com/$Repo/releases/download/v$Version"
        Info "Downloading $name ..." "Скачиваю $name ..."
        $zip = Join-Path $Tmp "$name.zip"
        Invoke-WebRequest -UseBasicParsing -Uri "$base/$name.zip" -OutFile $zip
        $sums = Join-Path $Tmp 'SHA256SUMS'
        Invoke-WebRequest -UseBasicParsing -Uri "$base/SHA256SUMS" -OutFile $sums
        $line = Get-Content $sums | Where-Object { $_ -match ('\s' + [regex]::Escape("$name.zip") + '$') } | Select-Object -First 1
        if (-not $line) { Die "$name.zip is not listed in SHA256SUMS" }
        $want = ($line -split '\s+')[0].ToLower()
        $got = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
        if ($want -ne $got) { Die "Checksum mismatch for $name.zip (expected $want, got $got)" }
        Ok 'Checksum verified.' 'Контрольная сумма совпала.'
        Expand-Archive -Force -Path $zip -DestinationPath $Tmp
        $exe = Join-Path (Join-Path $Tmp $name) 'rakkan-node.exe'
        if (-not (Test-Path $exe)) { Die 'Archive does not contain rakkan-node.exe' }
        Copy-Item -Force $exe $Bin
    }
    Ok "Installed $Bin" "Установлено: $Bin"

    $KeyFile = Join-Path $Tmp 'node.key'
    $Peer = (& $Bin id -key $KeyFile | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or -not $Peer) { Die 'Could not create a node key.' }
    $Peer = "$Peer".Trim()
    Ok "Your peer ID: $Peer" "Ваш peer ID: $Peer"
    $KeyB64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($KeyFile))
}
finally {
    # the key is kept in $KeyB64 only; remove the temporary files
    if (Test-Path $Tmp) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Tmp }
}

# ------------------------------------------------------------ 3: identity
Step '3/6' 'How you join' 'Как вы входите в сеть'
$CertObj = $null
$certPath = $env:RAKKAN_CERT
if ($certPath) { $mode = 2 }
elseif ($IsPublic) {
    Info 'The public network is open: you join as a guest and prove some work once.' 'Публичная сеть открыта: вы входите гостем и один раз подтверждаете работу.'
    $mode = 1
}
else {
    $mode = Ask-Choice 'How will this node join your network?' 'Как узел войдёт в вашу сеть?' @(
        'As a guest (only if the network is open to guests)|Гостем (только если сеть открыта для гостей)',
        'As a member with a certificate from the network admin|Участником, по сертификату от администратора сети') 2
}
if ($mode -eq 2) {
    if (-not $certPath) {
        Info 'Give this peer ID to the network admin, who issues a certificate for it:' 'Передайте этот peer ID администратору сети — он выпустит для него сертификат:'
        Write-Host "      $Peer" -ForegroundColor White
        $certPath = Ask-Text 'Path to the certificate file (empty: join as a guest for now)' 'Путь к файлу сертификата (пусто: пока войти гостем)' ''
    }
    if ($certPath) {
        $certPath = $certPath.Trim('"')
        if (-not (Test-Path $certPath)) { Die "No such file: $certPath" }
        $shown = (& $Bin network show $certPath) -join "`n"
        if ($LASTEXITCODE -ne 0) { Die "$certPath is not a valid certificate." }
        if (-not $shown.Contains($Peer)) { Die "This certificate was issued for a different node, not $Peer." }
        if (-not $shown.Contains($Net)) { Die 'This certificate belongs to a different network.' }
        $CertObj = Get-Content -Raw -Encoding UTF8 $certPath | ConvertFrom-Json
        Ok 'Certificate accepted.' 'Сертификат принят.'
    }
    else {
        Warn 'Joining as a guest; this works only while the network is open.' 'Вход гостем: работает, только пока сеть открыта.'
    }
}

# -------------------------------------------------------------- 4: type
Step '4/6' 'What this node does' 'Что будет делать узел'
$types = @('worker', 'client', 'relay', 'llm')
if ($env:RAKKAN_NODE_TYPE) { $Type = $env:RAKKAN_NODE_TYPE }
else {
    $tc = Ask-Choice 'Node type' 'Тип узла' @(
        'Worker — runs tasks for the network and can submit its own|Рабочий — выполняет задачи сети и может ставить свои',
        'Client — only submits tasks, runs nothing for others|Клиент — только ставит задачи, ничего не выполняет для других',
        'Relay — publicly reachable hub that helps others connect|Ретранслятор — публично доступный узел, помогает другим соединиться',
        'LLM worker — worker that also serves language-model stages (uses GPU if any)|LLM-узел — рабочий, который ещё обслуживает слои языковой модели (GPU, если есть)') 1
    $Type = $types[$tc - 1]
}
if ($types -notcontains $Type) { Die "Unknown node type: $Type" }

$Cpus = [Environment]::ProcessorCount
$Workers = $env:RAKKAN_WORKERS
if ($Type -eq 'client') { $Workers = '0' }
elseif (-not $Workers) {
    $Workers = Ask-Text "Tasks to run at once (CPU cores: $Cpus)" "Сколько задач выполнять одновременно (ядер CPU: $Cpus)" "$Cpus"
}
if ($Workers -notmatch '^\d+$') { Die "Invalid number of workers: $Workers" }
$StoreGb = $env:RAKKAN_STORE_GB
if (-not $StoreGb) { $StoreGb = Ask-Text 'Disk budget for stored data, GiB' 'Сколько диска отдать под данные, ГиБ' '10' }
if ($StoreGb -notmatch '^\d+(\.\d+)?$') { Die "Invalid disk budget: $StoreGb" }
$inv = [System.Globalization.CultureInfo]::InvariantCulture
$LlmMem = $null; $Gpu = 'auto'; $Port = $null
if ($Type -eq 'llm') {
    $m = Ask-Text 'Memory for model weights and caches, GiB' 'Память под веса модели и кэши, ГиБ' '4'
    if ($m -notmatch '^\d+(\.\d+)?$') { Die "Invalid memory size: $m" }
    $LlmMem = [double]::Parse($m, $inv)
    $gc = Ask-Choice 'GPU for language models' 'GPU для языковых моделей' @(
        'Pick the best one automatically|Выбрать лучшую автоматически', 'Do not use the GPU|Не использовать GPU') 1
    if ($gc -eq 2) { $Gpu = 'off' }
}
if ($Type -eq 'relay') {
    $p = Ask-Text 'Port to listen on (allow it in Windows Firewall and your router)' 'Порт для входящих соединений (разрешите его в брандмауэре Windows и на роутере)' '4001'
    if ($p -notmatch '^\d+$') { Die "Invalid port: $p" }
    $Port = [int]$p
}
Info 'Strict task isolation exists only on Linux; using process isolation.' 'Строгая изоляция есть только на Linux; использую изоляцию процессом.'

# ------------------------------------------------------- 5: connectivity
Step '5/6' 'Finding the network' 'Поиск сети'
if ($env:RAKKAN_MAINLINE) { $Mainline = $env:RAKKAN_MAINLINE -in @('y', 'yes', 'true', '1', 'on') }
else {
    Info "The node finds the network by its ID alone, in the BitTorrent DHT: the network's admin lists a few public members there. The records are encrypted, but DHT nodes see this machine's IP, as with any torrent client." 'Узел находит сеть по одному её ID — в DHT BitTorrent: администратор сети перечисляет там несколько публичных участников. Записи зашифрованы, но узлы DHT видят IP этой машины, как у любого торрент-клиента.'
    $Mainline = Ask-YN 'Look for the network in the BitTorrent DHT?' 'Искать сеть в DHT BitTorrent?' $true
}
$Boot = $env:RAKKAN_BOOTSTRAP
if (-not $Boot) {
    Info 'Known nodes are optional; they help when the DHT is blocked or the admin listed no one. Format: /ip4/203.0.113.7/udp/4001/quic-v1/p2p/12D3Koo...' 'Известные узлы не обязательны: они выручают, если DHT заблокирована или администратор никого не перечислил. Формат: /ip4/203.0.113.7/udp/4001/quic-v1/p2p/12D3Koo...'
    if ($Mainline) {
        $Boot = Ask-Text 'Address(es) of known nodes, comma-separated (empty: the DHT is enough)' 'Адрес(а) известных узлов через запятую (пусто: хватит DHT)' ''
    }
    else {
        $Boot = Ask-Text 'Address(es) of known nodes, comma-separated (empty: only the local network)' 'Адрес(а) известных узлов через запятую (пусто: только локальная сеть)' ''
    }
}
if ($Boot) {
    foreach ($a in ($Boot -split ',')) {
        if ($a.Trim() -notmatch '/p2p/') { Die "Address must end in /p2p/<peer ID>: $a" }
    }
}
elseif (-not $Mainline) {
    if (-not $CertObj) {
        Warn 'A guest needs a known node or the DHT to learn the network policy; without either the node cannot start.' 'Гостю нужен известный узел или DHT, чтобы узнать политику сети; без них узел не запустится.'
    }
    else {
        Warn 'Without known nodes and the DHT, the node finds members only on the local network.' 'Без известных узлов и DHT узел найдёт участников только в локальной сети.'
    }
}

# -------------------------------------------------------------- 6: start
Step '6/6' 'Autostart' 'Автозапуск'
if ($env:RAKKAN_AUTOSTART) { $Auto = $env:RAKKAN_AUTOSTART -in @('y', 'yes', 'true', '1') }
else { $Auto = Ask-YN 'Start the node automatically at logon (scheduled task)?' 'Запускать узел автоматически при входе в систему (задача планировщика)?' $true }
if ($env:RAKKAN_ADD_TO_PATH) { $AddPath = $env:RAKKAN_ADD_TO_PATH -in @('y', 'yes', 'true', '1') }
else { $AddPath = Ask-YN 'Add rakkan-node to your PATH?' 'Добавить rakkan-node в PATH?' $true }

# ---------------------------------------------------------- write config
if (Test-Path $Conf) {
    Copy-Item -Force $Conf ($Conf + '.bak-' + (Get-Date -Format 'yyyyMMddHHmmss'))
}
$cfg = [ordered]@{
    'network'     = $Net
    'private_key' = $KeyB64
}
if ($CertObj) { $cfg['certificate'] = $CertObj }
$cfg['isolation'] = 'process'
$cfg['workers'] = [int]$Workers
$cfg['store-gb'] = [double]::Parse($StoreGb, $inv)
$cfg['data'] = $DataDir
$cfg['api'] = Join-Path $DataDir 'rakkan.sock'
$cfg['peers'] = Join-Path $DataDir 'peers.json'
$cfg['proof'] = Join-Path $DataDir 'node.proof'
if ($Boot) { $cfg['bootstrap'] = (($Boot -split ',') | ForEach-Object { $_.Trim() }) -join ',' }
$cfg['mainline'] = [bool]$Mainline
if ($Type -eq 'relay') {
    $cfg['relay'] = $true
    $cfg['reachability'] = 'public'
    $cfg['listen'] = "/ip4/0.0.0.0/udp/$Port/quic-v1,/ip4/0.0.0.0/tcp/$Port,/ip6/::/udp/$Port/quic-v1,/ip6/::/tcp/$Port"
}
if ($Type -eq 'llm') {
    $cfg['llm'] = $true
    $cfg['llm-mem-gb'] = $LlmMem
    $cfg['gpu'] = $Gpu
}
# Keep the file private to this user before the key goes into it.
Write-Utf8NoBom $Conf ''
& icacls $Conf /inheritance:r /grant:r "$($env:USERNAME):(F)" | Out-Null
Write-Utf8NoBom $Conf (($cfg | ConvertTo-Json -Depth 20) + "`n")
Ok "Config written: $Conf (owner-only: it holds the node's private key)" "Конфиг записан: $Conf (доступ только владельцу: внутри приватный ключ узла)"

if ($AddPath) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { $userPath = '' }
    if (($userPath -split ';') -notcontains $BinDir) {
        [Environment]::SetEnvironmentVariable('Path', (($userPath.TrimEnd(';') + ';' + $BinDir).TrimStart(';')), 'User')
        Ok 'Added to PATH (open a new terminal to use it).' 'Добавлено в PATH (откройте новый терминал).'
    }
}

# --------------------------------------------------------------- service
if ($Auto) {
    $q = { param($s) $s -replace "'", "''" }
    $cmd = "& '$(& $q $Bin)' daemon -config '$(& $q $Conf)' *>> '$(& $q $LogFile)'"
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument ('-NoProfile -WindowStyle Hidden -Command "' + $cmd + '"') -WorkingDirectory $DataDir
    $who = "$($env:USERDOMAIN)\$($env:USERNAME)"
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $who
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
    $principal = New-ScheduledTaskPrincipal -UserId $who -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 2
    $state = (Get-ScheduledTask -TaskName $TaskName).State
    if ($state -eq 'Running') { Ok 'The node is running and will start at logon.' 'Узел запущен и будет стартовать при входе.' }
    else { Warn "The task is $state. See the log: $LogFile" "Задача в состоянии $state. Смотрите журнал: $LogFile" }
}

# --------------------------------------------------------------- summary
$rule = '━' * 59
Write-Host ''
Write-Host $rule -ForegroundColor Magenta
Say '  Done. Welcome to the mesh!' '  Готово. Добро пожаловать в сеть!'
Write-Host '  peer ID: ' -ForegroundColor DarkGray -NoNewline; Write-Host $Peer
Write-Host '  network: ' -ForegroundColor DarkGray -NoNewline; Write-Host $Net
Write-Host '  config:  ' -ForegroundColor DarkGray -NoNewline; Write-Host $Conf
Write-Host ''
Say '  Run by hand:' '  Запуск вручную:'
Write-Host "    & '$Bin' daemon -config '$Conf'"
if ($Auto) { Say "  Log:  Get-Content -Wait '$LogFile'" "  Журнал:  Get-Content -Wait '$LogFile'" }
if ($Type -eq 'relay' -and $Mainline) {
    Write-Host ''
    Say '  This node is public: the network admin can list it in the DHT, so that newcomers find the network through it:' '  Этот узел публичный: администратор сети может перечислить его в DHT, чтобы новички находили сеть через него:'
    Write-Host "    rakkan-node network announce -key root.key -announcers $Peer"
}
Say '  Remove: .\install.ps1 -Uninstall' '  Удалить: .\install.ps1 -Uninstall'
Write-Host $rule -ForegroundColor Magenta
Write-Host ''
