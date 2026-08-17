$RepoUrl = "https://github.com/AntonioBarriosHPE/HPECandyProLiant.git"
$TargetDir = "$env:USERPROFILE\Desktop\HPECandyProLiant"
$TempDir = Join-Path $env:TEMP ("HPECandyProLiant_" + [Guid]::NewGuid().ToString())

# 1. Locate Git if it's not in the default environment PATH
$GitCmd = "git"
if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
    $CommonGitPaths = @(
        "C:\Program Files\Git\cmd\git.exe",
        "C:\Program Files (x86)\Git\cmd\git.exe",
        "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe"
    )
    foreach ($Path in $CommonGitPaths) {
        if (Test-Path $Path) { $GitCmd = $Path; break }
    }
}

# 2. Safety Check: Ensure Git is actually installed
if (-not (Get-Command $GitCmd -ErrorAction SilentlyContinue)) {
    Write-Error "Git is not installed or could not be found. Please install Git from https://git-scm.com"
    return
}

# 3. Create the temp directory and execute the clone
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

Write-Host "Cloning codebase..." -ForegroundColor Cyan

# Use an explicit argument array to prevent Git from truncating the URL string
$GitArgs = @("clone", "--depth", "1", $RepoUrl, $TempDir)
Start-Process -FilePath $GitCmd -ArgumentList $GitArgs -NoNewWindow -Wait

# 4. Copy missing/updated files (without deleting local variations)
if (Test-Path "$TempDir\.git") {
    Write-Host "Updating code base via Robocopy..." -ForegroundColor Cyan
    robocopy $TempDir $TargetDir /E /XX /R:1 /W:1
} else {
    Write-Error "Git clone failed. Skipping file copy."
}

# 5. Clean up temporary directory safely if it exists
if (Test-Path $TempDir) {
    Write-Host "Cleaning up temporary files..." -ForegroundColor Cyan
    Remove-Item $TempDir -Recurse -Force
}

