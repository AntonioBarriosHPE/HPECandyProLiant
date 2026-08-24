param(
	[switch]$SkipUpdate,
	# Deployments running a feature branch must track that branch, not main.
	[string]$UpdateBranch = $(if ($env:CANDY_UPDATE_BRANCH) { $env:CANDY_UPDATE_BRANCH } else { "main" })
	[switch]$SkipUpdate
)

# Resolve paths from this script so it works from any checkout location.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoUrl = "https://github.com/AntonioBarriosHPE/HPECandyProLiant.git"
$PythonApp = Join-Path $ProjectRoot "app.py"
$Requirements = Join-Path $ProjectRoot "requirements.txt"
$SupportedPythonVersions = @("3.12", "3.11")
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$VenvDirectory = Join-Path $ProjectRoot "venv"
$PidFile = Join-Path $ProjectRoot "candy-machine.pid"

function Stop-PreviousCandyProcess {
	# Orphaned instances keep the webcam locked, so stop every python running this app.py.
	$AppPathPattern = "*" + [System.Management.Automation.WildcardPattern]::Escape($PythonApp) + "*"
	$CandyProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
		Where-Object { $_.CommandLine -like $AppPathPattern }

	foreach ($CandyProcess in $CandyProcesses) {
		Write-Output "Stopping previous Candy Machine process (PID $($CandyProcess.ProcessId))..."
		Stop-Process -Id $CandyProcess.ProcessId -Force -ErrorAction SilentlyContinue
	}

	if ($CandyProcesses) {
		# Give Windows time to release the camera handle before reopening it.
		Start-Sleep -Seconds 2
	}

	Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

function Update-ApplicationCode {
	if ($SkipUpdate) {
		Write-Output "Code update check skipped by -SkipUpdate."
		return
	}

	if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
		throw "Git is required to check for application updates."
	}

	$CurrentCommit = (& git -C $ProjectRoot rev-parse HEAD 2>$null).Trim()
	$RemoteCommit = (& git ls-remote $RepoUrl "refs/heads/main" 2>$null).Split()[0]
	if (-not $CurrentCommit -or -not $RemoteCommit) {
		throw "Could not determine the local or remote Git commit."
	}

	if ($CurrentCommit -eq $RemoteCommit) {
		Write-Output "Application is already up to date ($CurrentCommit)."
		return
	}

	$TrackedChanges = & git -C $ProjectRoot status --porcelain --untracked-files=no
	if ($TrackedChanges) {
		throw "Application update stopped because tracked local changes exist. Commit or revert them before updating."
	}

	$TempDir = Join-Path $env:TEMP ("HPECandyProLiant_" + [Guid]::NewGuid().ToString())
	try {
		Write-Output "Update available: $CurrentCommit -> $RemoteCommit"
		New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
		$Clone = Start-Process -FilePath "git" -ArgumentList @("clone", "--depth", "1", "--branch", "main", $RepoUrl, $TempDir) -NoNewWindow -Wait -PassThru
		if ($Clone.ExitCode -ne 0) {
			throw "Git clone failed with exit code $($Clone.ExitCode)."
		}

		$RoboCopy = Start-Process -FilePath "robocopy" -ArgumentList @(
			$TempDir, $ProjectRoot, "/E", "/XX", "/R:1", "/W:1",
			"/XD", ".git", "venv", "Logs",
			"/XF", ".env", "candy-machine.pid"
		) -NoNewWindow -Wait -PassThru
		if ($RoboCopy.ExitCode -gt 7) {
			throw "Application file update failed with robocopy exit code $($RoboCopy.ExitCode)."
		}
		Write-Output "Application files updated successfully."
	}
	finally {
		if (Test-Path $TempDir) {
			Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
		}
	}
}

# Create timestamped launcher and Python logs.
$LogDir = Join-Path $ProjectRoot "Logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PSScriptLog = Join-Path $LogDir "Launcher_$Timestamp.txt"
$PythonOutLog = Join-Path $LogDir "Python_Stdout_$Timestamp.txt"
$PythonErrLog = Join-Path $LogDir "Python_Stderr_$Timestamp.txt"

# Keep seven days of logs.
Get-ChildItem -Path $LogDir -Filter "*.txt" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

$ErrorActionPreference = "Stop"
Push-Location $ProjectRoot
$TranscriptStarted = $false
try {
	Start-Transcript -Path $PSScriptLog -Append | Out-Null
	$TranscriptStarted = $true
	Write-Output "--- Execution Started at $(Get-Date) ---"
	Write-Output "Project root: $ProjectRoot"
	Stop-PreviousCandyProcess
	Update-ApplicationCode

	# Select an unused local port so a previous run cannot block startup.
	$PortProbe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
	$PortProbe.Start()
	$HttpPort = $PortProbe.LocalEndpoint.Port
	$PortProbe.Stop()
	$env:CANDY_HTTP_HOST = "127.0.0.1"
	$env:CANDY_HTTP_PORT = "$HttpPort"
	$Url = "http://localhost:$HttpPort"
	Write-Output "Selected HTTP port: $HttpPort"

	if (-not (Get-Command "py" -ErrorAction SilentlyContinue)) {
		throw "The Windows Python launcher is required. Install Python 3.12 or 3.11 and try again."
	}

	$PythonVersion = $null
	foreach ($CandidateVersion in $SupportedPythonVersions) {
		$CandidatePython = & py "-$CandidateVersion" -c "import sys; print(sys.executable)" 2>$null
		if ($LASTEXITCODE -eq 0 -and $CandidatePython) {
			$PythonVersion = $CandidateVersion
			break
		}
	}

	if (-not $PythonVersion) {
		throw "No supported Python version was found. Install Python 3.12 or 3.11 and try again."
	}
	Write-Output "Selected Python $PythonVersion for this machine."

	$VenvNeedsRebuild = $false
	if (Test-Path $VenvPython) {
		$VenvRuntimeVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
		$VenvNeedsRebuild = ($LASTEXITCODE -ne 0 -or $VenvRuntimeVersion.Trim() -ne $PythonVersion)
	}

	if ($VenvNeedsRebuild) {
		Write-Host "Existing virtual environment uses an incompatible Python version. Recreating it..." -ForegroundColor Yellow
		Write-Output "Existing virtual environment uses Python $VenvRuntimeVersion; rebuilding with Python $PythonVersion."
		Remove-Item -Path $VenvDirectory -Recurse -Force
	}

	if (-not (Test-Path $VenvPython)) {
		Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
		& py "-$PythonVersion" -m venv $VenvDirectory
	}

	if (-not (Test-Path $VenvPython)) {
		throw "The virtual environment could not be created at $VenvPython"
	}

	Write-Host "Installing or updating Python dependencies..." -ForegroundColor Cyan
	Write-Output "Installing or updating Python dependencies..."
	& $VenvPython -m pip install --upgrade pip
	& $VenvPython -m pip install --only-binary=:all: -r $Requirements
	if ($LASTEXITCODE -ne 0) {
		throw "Dependency installation failed. Review $PSScriptLog and install a compatible Python version before retrying."
	}

	& $VenvPython -c "import aiohttp, cv2, numpy, openvino"
	if ($LASTEXITCODE -ne 0) {
		throw "Dependency verification failed. Review $PSScriptLog and the Python stderr log $PythonErrLog."
	}

	Write-Host "Starting Candy Machine..." -ForegroundColor Green
	Write-Output "Launching Python application..."
	$CandyProcess = Start-Process -FilePath $VenvPython `
		-ArgumentList @($PythonApp) `
		-WindowStyle Hidden `
		-WorkingDirectory $ProjectRoot `
		-RedirectStandardOutput $PythonOutLog `
		-RedirectStandardError $PythonErrLog `
		-PassThru
	Set-Content -Path $PidFile -Value $CandyProcess.Id -NoNewline
	Write-Output "Candy Machine started with PID $($CandyProcess.Id)."

	Write-Output "Waiting 5 seconds for Python and Arduino initialization..."
	Start-Sleep -Seconds 5
	Start-Process "chrome.exe" "--new-window --start-fullscreen $Url" -WindowStyle Maximized
	Write-Output "--- Launch Sequence Completed Successfully ---"
}
catch {
	Write-Error "An unexpected error occurred during execution: $_"
    throw
}
finally {
	if ($TranscriptStarted) {
		Stop-Transcript | Out-Null
	}
	Pop-Location
}