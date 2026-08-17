# CONFIGURATION
$ProjectRoot = "$env:USERPROFILE\Desktop\HPECandyProLiant"
$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
$PythonApp = Join-Path $ProjectRoot "app.py"
$Url = "http://localhost:9000"

# CREATE LOGGING DIRECTORY & FILENAMES
$LogDir = Join-Path $ProjectRoot "Logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PSScriptLog = Join-Path $LogDir "Launcher_$Timestamp.txt"
$PythonOutLog = Join-Path $LogDir "Python_Stdout_$Timestamp.txt"
$PythonErrLog = Join-Path $LogDir "Python_Stderr_$Timestamp.txt"

# HOUSEKEEPING: Delete logs older than 7 days to save space
Get-ChildItem -Path $LogDir -Filter "*.txt" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item

# START LOGGING POWERSHELL OUTPUT
Start-Transcript -Path $PSScriptLog -Append

try {
    Write-Output "--- Execution Started at (Get-Date) ---"
    
    # LAUNCH PYTHON BACK-END
    Push-Location $ProjectRoot
    Write-Output "Navigated to Project Root: $ProjectRoot"

    # activate venv
    Write-Output "Activating virtual environment..."
    & $VenvActivate

    # start python with unique, timestamped output logs
    Write-Output "Launching Python application background process..."
    Start-Process -FilePath "python.exe" `
        -ArgumentList "`"$PythonApp`"" `
        -WindowStyle Hidden `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $PythonOutLog `
        -RedirectStandardError $PythonErrLog

    # allow time to startup (connect to port and arduino)
    Write-Output "Waiting 5 seconds for Python and Arduino initialization..."
    Start-Sleep -Seconds 5

    # LAUNCH CHROME
    #Write-Output "Launching Chrome browser..."
    #Start-Process "chrome.exe" "--new-window --start-fullscreen $Url" -WindowStyle Maximized
    # LAUNCH CHROME
    Write-Output "Launching Chrome browser..."
    Start-Process "chrome.exe" -ArgumentList "--new-window", "--start-fullscreen", $Url
    


    
    Write-Output "--- Launch Sequence Completed Successfully ---"
}
catch {
    Write-Error "An unexpected error occurred during execution: $_"
}
finally {
    # Stop the script transcript safely
    Stop-Transcript
}
