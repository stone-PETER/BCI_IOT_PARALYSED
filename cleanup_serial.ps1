# Kill any Python processes that might be holding serial ports
Write-Host "Stopping Python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Write-Host "Done! Serial ports should be released." -ForegroundColor Green
