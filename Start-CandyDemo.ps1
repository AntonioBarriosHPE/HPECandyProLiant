# CONFIGURATION
$ProjectRoot = "$env:USERPROFILE\Desktop\HPECandyProLiant"
$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
$PythonApp = Join-Path $ProjectRoot "app.py"
$Url = "http://localhost:9000"

# LAUNCH PYTHON BACK-END
Push-Location $ProjectRoot

# activate venv
& $VenvActivate

# start python as a background job
Start-Process -FilePath "python.exe" -ArgumentList "`"$PythonApp`"" -WindowStyle Hidden -WorkingDirectory $ProjectRoot

# allow time to startup (connect to port and arduino)
Start-Sleep -Seconds 5

# LAUNCH CHROME
Start-Process "chrome.exe" "--new-window --start-fullscreen $Url" -WindowStyle Maximized