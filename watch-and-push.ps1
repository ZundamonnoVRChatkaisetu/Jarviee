# watch-debug.ps1
$projectDir = "C:\1_Sagyo\BenriToul\AI\Jarviee"
Set-Location $projectDir

# グローバル変数に変更
$Global:lastPushTime = Get-Date "2000-01-01"
$debounceSeconds = 5

$watcher = New-Object System.IO.FileSystemWatcher $projectDir, "*.*"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

$action = {
    $fullPath = $Event.SourceEventArgs.FullPath
    $now = Get-Date

    # ログ出力して何が来てるか確認
    Write-Host "→ Event: $fullPath at $($now.ToString('HH:mm:ss'))"
    Write-Host "→ LastPush: $($Global:lastPushTime.ToString('HH:mm:ss'))"

    # .git 以下を除外
    if ($fullPath -match "\\\.git\\\") {
        Write-Host "  (skipped .git)" 
        return
    }

    # デバウンス判定
    if (($now - $Global:lastPushTime).TotalSeconds -lt $debounceSeconds) {
        Write-Host "  (skipped by debounce)" 
        return
    }

    # 本番アクション
    Start-Sleep -Milliseconds 200
    Write-Host "  -> git add/commit/push"
    try {
        git add .
        $msg = $now.ToString("yyyy-MM-dd_HH-mm-ss")
        git commit -m "auto commit: $msg"
        git push
        $Global:lastPushTime = $now
        Write-Host "  ? Pushed at $msg"
    } catch {
        Write-Warning "  ? Git error: $_"
    }
}

foreach ($e in "Changed","Created","Renamed","Deleted") {
    Register-ObjectEvent $watcher $e -Action $action
}

Write-Host "=== Watching with debounce & exclude ==="
while ($true) { Start-Sleep 1 }
