# 監視対象ディレクトリ（プロジェクトのルート）を指定
$projectDir = "C:\1_Sagyo\BenriToul\AI\Jarviee"
Set-Location $projectDir

# FileSystemWatcher の設定
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $projectDir
$watcher.Filter = "*.*"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# イベント発生時のアクション
$action = {
    # 少しウェイトを挟んでファイル書き込み完了を待つ
    Start-Sleep -Milliseconds 200
    Write-Host "Detected change: $($Event.SourceEventArgs.FullPath)"
    
    try {
        git add .
        $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
        git commit -m "auto commit: $timestamp"
        git push
        Write-Host "✅ Pushed at $timestamp"
    } catch {
        Write-Warning "❌ Git 操作中にエラー: $_"
    }
}

# イベント登録
Register-ObjectEvent $watcher Changed -Action $action
Register-ObjectEvent $watcher Created -Action $action
Register-ObjectEvent $watcher Renamed -Action $action
Register-ObjectEvent $watcher Deleted -Action $action

# 無限ループでスクリプトを常駐させる
Write-Host "Watching for file changes in $projectDir ..."
while ($true) { Start-Sleep 1 }
