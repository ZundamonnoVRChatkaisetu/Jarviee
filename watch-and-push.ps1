# watch-debug.ps1
$projectDir = "C:\1_Sagyo\BenriToul\AI\Jarviee"
Set-Location $projectDir

# �O���[�o���ϐ��ɕύX
$Global:lastPushTime = Get-Date "2000-01-01"
$debounceSeconds = 5

$watcher = New-Object System.IO.FileSystemWatcher $projectDir, "*.*"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

$action = {
    $fullPath = $Event.SourceEventArgs.FullPath
    $now = Get-Date

    # ���O�o�͂��ĉ������Ă邩�m�F
    Write-Host "�� Event: $fullPath at $($now.ToString('HH:mm:ss'))"
    Write-Host "�� LastPush: $($Global:lastPushTime.ToString('HH:mm:ss'))"

    # .git �ȉ������O
    if ($fullPath -match "\\\.git\\\") {
        Write-Host "  (skipped .git)" 
        return
    }

    # �f�o�E���X����
    if (($now - $Global:lastPushTime).TotalSeconds -lt $debounceSeconds) {
        Write-Host "  (skipped by debounce)" 
        return
    }

    # �{�ԃA�N�V����
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
