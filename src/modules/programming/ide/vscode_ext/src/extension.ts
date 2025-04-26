/**
 * Jarviee VS Code拡張
 * 
 * Jarvieeシステムとのインテグレーションを提供するVS Code拡張
 */

import * as vscode from 'vscode';
import * as WebSocket from 'ws';
import * as http from 'http';
import * as path from 'path';
import { URI } from 'vscode-uri';

// 拡張コンテキスト
let extensionContext: vscode.ExtensionContext;

// WebSocketサーバー
let wss: WebSocket.Server | undefined;
let clients: Set<WebSocket.WebSocket> = new Set();

// サーバーステータス
let serverStatus = {
    isRunning: false,
    connectedClients: 0,
    startTime: 0,
    port: 7890
};

// ステータスバーアイテム
let statusBarItem: vscode.StatusBarItem;

/**
 * 拡張のアクティベート
 */
export function activate(context: vscode.ExtensionContext) {
    console.log('Jarviee拡張がアクティベートされました');
    extensionContext = context;

    // ステータスバーアイテムを作成
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = 'Jarviee: 切断';
    statusBarItem.command = 'jarviee.showStatus';
    context.subscriptions.push(statusBarItem);

    // コマンドの登録
    context.subscriptions.push(
        vscode.commands.registerCommand('jarviee.connect', connectToServer),
        vscode.commands.registerCommand('jarviee.disconnect', disconnectFromServer),
        vscode.commands.registerCommand('jarviee.showStatus', showStatus),
        vscode.commands.registerCommand('jarviee.analyzeCode', analyzeCode),
        vscode.commands.registerCommand('jarviee.debugCode', debugCode),
        vscode.commands.registerCommand('jarviee.optimizeCode', optimizeCode)
    );

    // イベントリスナーの設定
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument(onDocumentSave),
        vscode.window.onDidChangeActiveTextEditor(onActiveEditorChange),
        vscode.languages.onDidChangeDiagnostics(onDiagnosticsChange)
    );

    // 設定の更新を監視
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(onConfigurationChange)
    );

    // ステータスバーアイテムの表示設定
    const showStatusBar = vscode.workspace.getConfiguration('jarviee').get<boolean>('showStatusBarItem', true);
    if (showStatusBar) {
        statusBarItem.show();
    } else {
        statusBarItem.hide();
    }

    // 自動接続設定
    const autoConnect = vscode.workspace.getConfiguration('jarviee').get<boolean>('autoConnect', true);
    if (autoConnect) {
        connectToServer();
    }
}

/**
 * 拡張のディアクティベート
 */
export function deactivate() {
    disconnectFromServer();
    console.log('Jarviee拡張がディアクティベートされました');
}

/**
 * サーバーに接続（WebSocketサーバーを起動）
 */
async function connectToServer() {
    if (serverStatus.isRunning) {
        vscode.window.showInformationMessage('Jarvieeサーバーは既に起動中です');
        return;
    }

    try {
        // ポートの取得
        const websocketUrl = vscode.workspace.getConfiguration('jarviee').get<string>('websocketUrl', 'ws://localhost:7890');
        const portMatch = websocketUrl.match(/:(\d+)/);
        const port = portMatch ? parseInt(portMatch[1]) : 7890;

        // WebSocketサーバーを作成
        wss = new WebSocket.Server({ port });

        // WebSocketイベントハンドラの設定
        wss.on('connection', onWebSocketConnection);
        wss.on('error', onWebSocketServerError);

        // ステータスの更新
        serverStatus.isRunning = true;
        serverStatus.startTime = Date.now();
        serverStatus.port = port;
        updateStatusBarItem();

        vscode.window.showInformationMessage(`Jarvieeサーバーを起動しました (Port: ${port})`);
    } catch (error) {
        console.error('Jarvieeサーバー起動中にエラーが発生しました:', error);
        vscode.window.showErrorMessage(`Jarvieeサーバー起動中にエラーが発生しました: ${error instanceof Error ? error.message : String(error)}`);
    }
}

/**
 * サーバーから切断（WebSocketサーバーを停止）
 */
function disconnectFromServer() {
    if (!serverStatus.isRunning) {
        return;
    }

    try {
        // 接続中のクライアントをすべて閉じる
        for (const client of clients) {
            client.close();
        }
        clients.clear();

        // WebSocketサーバーをクローズ
        if (wss) {
            wss.close();
            wss = undefined;
        }

        // ステータスの更新
        serverStatus.isRunning = false;
        serverStatus.connectedClients = 0;
        updateStatusBarItem();

        vscode.window.showInformationMessage('Jarvieeサーバーを停止しました');
    } catch (error) {
        console.error('Jarvieeサーバー停止中にエラーが発生しました:', error);
        vscode.window.showErrorMessage(`Jarvieeサーバー停止中にエラーが発生しました: ${error instanceof Error ? error.message : String(error)}`);
    }
}

/**
 * ステータスの表示
 */
function showStatus() {
    const statusMessage = serverStatus.isRunning
        ? `Jarvieeサーバー稼働中:\n- ポート: ${serverStatus.port}\n- 接続クライアント: ${serverStatus.connectedClients}\n- 起動時間: ${formatUptime(Date.now() - serverStatus.startTime)}`
        : 'Jarvieeサーバーは停止中です';

    vscode.window.showInformationMessage(statusMessage);
}

/**
 * 現在のコードを分析
 */
async function analyzeCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('アクティブなエディタがありません');
        return;
    }

    const selection = editor.selection;
    const text = selection.isEmpty
        ? editor.document.getText()
        : editor.document.getText(selection);

    // 選択範囲またはファイル全体が空でないことを確認
    if (!text.trim()) {
        vscode.window.showWarningMessage('選択範囲またはファイルにコードがありません');
        return;
    }

    // クライアントにコード分析リクエストを送信
    const filePath = editor.document.uri.fsPath;
    const language = editor.document.languageId;

    try {
        // 分析中の通知
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'コード分析中...',
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 0 });

            // クライアントにリクエストを送信
            const response = await broadcastRequest('analyzeCode', {
                filePath,
                language,
                code: text,
                selection: {
                    start: {
                        line: selection.start.line,
                        character: selection.start.character
                    },
                    end: {
                        line: selection.end.line,
                        character: selection.end.character
                    }
                }
            });

            progress.report({ increment: 100 });

            // レスポンスの処理
            if (response && response.success) {
                // 分析結果をWebViewで表示
                showAnalysisResults(response.result, language);
            } else {
                const errorMsg = response ? response.error : '不明なエラー';
                vscode.window.showErrorMessage(`コード分析に失敗しました: ${errorMsg}`);
            }

            return response;
        });
    } catch (error) {
        console.error('コード分析中にエラーが発生しました:', error);
        vscode.window.showErrorMessage(`コード分析中にエラーが発生しました: ${error instanceof Error ? error.message : String(error)}`);
    }
}

/**
 * デバッグヘルプを表示
 */
async function debugCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('アクティブなエディタがありません');
        return;
    }

    // 診断情報の取得
    const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
    if (diagnostics.length === 0) {
        vscode.window.showInformationMessage('このファイルには診断エラーがありません');
        return;
    }

    const selection = editor.selection;
    const text = selection.isEmpty
        ? editor.document.getText()
        : editor.document.getText(selection);

    // 選択範囲またはファイル全体が空でないことを確認
    if (!text.trim()) {
        vscode.window.showWarningMessage('選択範囲またはファイルにコードがありません');
        return;
    }

    // クライアントにデバッグヘルプリクエストを送信
    const filePath = editor.document.uri.fsPath;
    const language = editor.document.languageId;

    try {
        // デバッグヘルプを取得中の通知
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'デバッグヘルプを取得中...',
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 0 });

            // 診断情報をエラーメッセージとして整形
            const errorMessages = diagnostics.map(diag => ({
                severity: diag.severity,
                message: diag.message,
                range: {
                    start: {
                        line: diag.range.start.line,
                        character: diag.range.start.character
                    },
                    end: {
                        line: diag.range.end.line,
                        character: diag.range.end.character
                    }
                }
            }));

            // クライアントにリクエストを送信
            const response = await broadcastRequest('debugCode', {
                filePath,
                language,
                code: text,
                diagnostics: errorMessages
            });

            progress.report({ increment: 100 });

            // レスポンスの処理
            if (response && response.success) {
                // デバッグヘルプをWebViewで表示
                showDebugHelp(response.result, language);
            } else {
                const errorMsg = response ? response.error : '不明なエラー';
                vscode.window.showErrorMessage(`デバッグヘルプの取得に失敗しました: ${errorMsg}`);
            }

            return response;
        });
    } catch (error) {
        console.error('デバッグヘルプの取得中にエラーが発生しました:', error);
        vscode.window.showErrorMessage(`デバッグヘルプの取得中にエラーが発生しました: ${error instanceof Error ? error.message : String(error)}`);
    }
}

/**
 * コードを最適化
 */
async function optimizeCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('アクティブなエディタがありません');
        return;
    }

    const selection = editor.selection;
    const text = selection.isEmpty
        ? editor.document.getText()
        : editor.document.getText(selection);

    // 選択範囲またはファイル全体が空でないことを確認
    if (!text.trim()) {
        vscode.window.showWarningMessage('選択範囲またはファイルにコードがありません');
        return;
    }

    // クライアントにコード最適化リクエストを送信
    const filePath = editor.document.uri.fsPath;
    const language = editor.document.languageId;

    try {
        // 最適化中の通知
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'コード最適化中...',
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 0 });

            // クライアントにリクエストを送信
            const response = await broadcastRequest('optimizeCode', {
                filePath,
                language,
                code: text,
                selection: {
                    start: {
                        line: selection.start.line,
                        character: selection.start.character
                    },
                    end: {
                        line: selection.end.line,
                        character: selection.end.character
                    }
                }
            });

            progress.report({ increment: 100 });

            // レスポンスの処理
            if (response && response.success) {
                // 最適化コードを適用するか確認
                const result = response.result;
                const optimizedCode = result.optimizedCode;

                if (optimizedCode) {
                    const applyChanges = await vscode.window.showInformationMessage(
                        '最適化されたコードを適用しますか？',
                        { modal: true },
                        'はい', 'いいえ', '詳細を表示'
                    );

                    if (applyChanges === 'はい') {
                        // 編集を適用
                        await editor.edit(editBuilder => {
                            const range = selection.isEmpty
                                ? new vscode.Range(0, 0, editor.document.lineCount, 0)
                                : selection;
                            editBuilder.replace(range, optimizedCode);
                        });
                        vscode.window.showInformationMessage('最適化コードを適用しました');
                    } else if (applyChanges === '詳細を表示') {
                        // 最適化コードと説明をWebViewで表示
                        showOptimizationResults(result, language);
                    }
                } else {
                    vscode.window.showInformationMessage('最適化の必要はありません。コードは既に最適化されています');
                }
            } else {
                const errorMsg = response ? response.error : '不明なエラー';
                vscode.window.showErrorMessage(`コード最適化に失敗しました: ${errorMsg}`);
            }

            return response;
        });
    } catch (error) {
        console.error('コード最適化中にエラーが発生しました:', error);
        vscode.window.showErrorMessage(`コード最適化中にエラーが発生しました: ${error instanceof Error ? error.message : String(error)}`);
    }
}

/**
 * 分析結果をWebViewで表示
 */
function showAnalysisResults(results: any, language: string) {
    // WebViewパネルを作成
    const panel = vscode.window.createWebviewPanel(
        'jarvieeCodeAnalysis',
        'Jarviee コード分析',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(extensionContext.extensionPath, 'resources'))]
        }
    );

    // HTMLコンテンツを生成
    panel.webview.html = getAnalysisWebviewContent(results, language);
}

/**
 * デバッグヘルプをWebViewで表示
 */
function showDebugHelp(results: any, language: string) {
    // WebViewパネルを作成
    const panel = vscode.window.createWebviewPanel(
        'jarvieeDebugHelp',
        'Jarviee デバッグヘルプ',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(extensionContext.extensionPath, 'resources'))]
        }
    );

    // HTMLコンテンツを生成
    panel.webview.html = getDebugHelpWebviewContent(results, language);
}

/**
 * 最適化結果をWebViewで表示
 */
function showOptimizationResults(results: any, language: string) {
    // WebViewパネルを作成
    const panel = vscode.window.createWebviewPanel(
        'jarvieeOptimization',
        'Jarviee コード最適化',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.file(path.join(extensionContext.extensionPath, 'resources'))]
        }
    );

    // HTMLコンテンツを生成
    panel.webview.html = getOptimizationWebviewContent(results, language);
}

/**
 * ドキュメント保存イベントハンドラ
 */
function onDocumentSave(document: vscode.TextDocument) {
    // 設定で診断機能が無効な場合は何もしない
    const enableDiagnostics = vscode.workspace.getConfiguration('jarviee').get<boolean>('enableDiagnostics', true);
    if (!enableDiagnostics) {
        return;
    }

    // クライアントにイベントを送信
    broadcastEvent('fileSaved', {
        filePath: document.uri.fsPath,
        language: document.languageId
    });
}

/**
 * アクティブエディタ変更イベントハンドラ
 */
function onActiveEditorChange(editor: vscode.TextEditor | undefined) {
    if (!editor) {
        return;
    }

    // クライアントにイベントを送信
    broadcastEvent('fileOpened', {
        filePath: editor.document.uri.fsPath,
        language: editor.document.languageId
    });
}

/**
 * 診断情報変更イベントハンドラ
 */
function onDiagnosticsChange(event: vscode.DiagnosticChangeEvent) {
    // 設定で診断機能が無効な場合は何もしない
    const enableDiagnostics = vscode.workspace.getConfiguration('jarviee').get<boolean>('enableDiagnostics', true);
    if (!enableDiagnostics) {
        return;
    }

    // URI毎に処理
    event.uris.forEach(uri => {
        const diagnostics = vscode.languages.getDiagnostics(uri);
        
        // 診断情報の変換
        const diagnosticData = diagnostics.map(diag => ({
            severity: diag.severity,
            message: diag.message,
            range: {
                start: {
                    line: diag.range.start.line,
                    character: diag.range.start.character
                },
                end: {
                    line: diag.range.end.line,
                    character: diag.range.end.character
                }
            }
        }));

        // クライアントにイベントを送信
        broadcastEvent('diagnosticsChanged', {
            filePath: uri.fsPath,
            diagnostics: diagnosticData
        });
    });
}

/**
 * 設定変更イベントハンドラ
 */
function onConfigurationChange(event: vscode.ConfigurationChangeEvent) {
    if (event.affectsConfiguration('jarviee.showStatusBarItem')) {
        const showStatusBar = vscode.workspace.getConfiguration('jarviee').get<boolean>('showStatusBarItem', true);
        if (showStatusBar) {
            statusBarItem.show();
        } else {
            statusBarItem.hide();
        }
    }
}

/**
 * WebSocket接続イベントハンドラ
 */
function onWebSocketConnection(ws: WebSocket.WebSocket) {
    console.log('Jarvieeクライアントが接続しました');
    
    // クライアントを追加
    clients.add(ws);
    serverStatus.connectedClients = clients.size;
    updateStatusBarItem();

    // クライアントイベントハンドラの設定
    ws.on('message', (message: WebSocket.RawData) => onWebSocketMessage(ws, message));
    ws.on('close', () => onWebSocketClose(ws));
    ws.on('error', (error: Error) => onWebSocketError(ws, error));
}

/**
 * WebSocketメッセージイベントハンドラ
 */
function onWebSocketMessage(ws: WebSocket.WebSocket, message: WebSocket.RawData) {
    try {
        // メッセージをJSONとしてパース
        const data = JSON.parse(message.toString());
        
        // メッセージタイプに基づいて処理
        if (data.type === 'hello') {
            // クライアント初期接続処理
            handleHelloMessage(ws, data);
        } else if (data.type === 'request') {
            // クライアントからのリクエスト処理
            handleRequestMessage(ws, data);
        } else if (data.type === 'response') {
            // クライアントからのレスポンス処理
            handleResponseMessage(ws, data);
        } else {
            console.warn('未知のメッセージタイプ:', data.type);
        }
    } catch (error) {
        console.error('WebSocketメッセージ処理中にエラーが発生しました:', error);
    }
}

/**
 * WebSocket切断イベントハンドラ
 */
function onWebSocketClose(ws: WebSocket.WebSocket) {
    console.log('Jarvieeクライアントが切断しました');
    
    // クライアントを削除
    clients.delete(ws);
    serverStatus.connectedClients = clients.size;
    updateStatusBarItem();
}

/**
 * WebSocketエラーイベントハンドラ
 */
function onWebSocketError(ws: WebSocket.WebSocket, error: Error) {
    console.error('WebSocketクライアントエラー:', error);
    
    // エラー発生時もクライアントを削除
    clients.delete(ws);
    serverStatus.connectedClients = clients.size;
    updateStatusBarItem();
}

/**
 * WebSocketサーバーエラーイベントハンドラ
 */
function onWebSocketServerError(error: Error) {
    console.error('WebSocketサーバーエラー:', error);
    vscode.window.showErrorMessage(`Jarvieeサーバーエラー: ${error.message}`);
    
    // サーバーを停止
    disconnectFromServer();
}

/**
 * 初期接続メッセージ処理
 */
function handleHelloMessage(ws: WebSocket.WebSocket, data: any) {
    console.log(`Jarvieeクライアント初期接続: ${data.client} v${data.version}`);
    
    // クライアントに応答
    ws.send(JSON.stringify({
        type: 'hello',
        server: 'jarviee-vscode',
        version: '0.1.0',
        capabilities: ['codeEdit', 'diagnostics', 'fileInfo', 'projectStructure']
    }));
}

/**
 * リクエストメッセージ処理
 */
async function handleRequestMessage(ws: WebSocket.WebSocket, data: any) {
    console.log(`Jarvieeクライアントからのリクエスト: ${data.method}`);
    
    // リクエストIDの確認
    if (!data.id || !data.method) {
        console.error('無効なリクエスト形式:', data);
        return;
    }
    
    let response: any = {
        type: 'response',
        id: data.id,
        result: {
            success: false,
            error: '未実装のメソッド'
        }
    };
    
    try {
        // メソッドに基づいて処理
        switch (data.method) {
            case 'getCurrentFile':
                response.result = await getCurrentFile();
                break;
            case 'getProjectStructure':
                response.result = await getProjectStructure(data.params);
                break;
            case 'applyCodeEdit':
                response.result = await applyCodeEdit(data.params);
                break;
            case 'executeCommand':
                response.result = await executeCommand(data.params);
                break;
            case 'showNotification':
                response.result = await showNotification(data.params);
                break;
            case 'getDiagnostics':
                response.result = await getDiagnostics(data.params);
                break;
            default:
                console.warn(`未知のリクエストメソッド: ${data.method}`);
        }
    } catch (error) {
        console.error(`リクエスト処理中にエラーが発生しました: ${error instanceof Error ? error.message : String(error)}`);
        response.result = {
            success: false,
            error: `処理エラー: ${error instanceof Error ? error.message : String(error)}`
        };
    }
    
    // レスポンス送信
    ws.send(JSON.stringify(response));
}

/**
 * レスポンスメッセージ処理
 */
function handleResponseMessage(ws: WebSocket.WebSocket, data: any) {
    console.log(`Jarvieeクライアントからのレスポンス: ID=${data.id}`);
    
    // 処理は他の関数で行われるため、ここでは記録のみ
}

/**
 * イベントをすべてのクライアントに送信
 */
function broadcastEvent(event: string, data: any) {
    if (clients.size === 0) {
        return;
    }
    
    const message = JSON.stringify({
        type: 'event',
        event,
        data
    });
    
    // すべてのクライアントに送信
    for (const client of clients) {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    }
}

/**
 * リクエストをすべてのクライアントに送信し、最初のレスポンスを待機
 */
function broadcastRequest(method: string, params: any, timeout: number = 5000): Promise<any> {
    return new Promise((resolve, reject) => {
        if (clients.size === 0) {
            reject(new Error('接続されているクライアントがありません'));
            return;
        }
        
        // リクエストIDの生成
        const requestId = `req_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
        
        // リクエストメッセージの作成
        const message = JSON.stringify({
            type: 'request',
            id: requestId,
            method,
            params
        });
        
        // レスポンス待機用ハンドラ
        let responseHandler: ((ws: WebSocket.WebSocket, data: WebSocket.RawData) => void) | null = null;
        let timeoutId: NodeJS.Timeout | null = null;
        
        // イベントハンドラのクリーンアップ
        const cleanup = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
                timeoutId = null;
            }
            
            // すべてのクライアントからハンドラを削除
            if (responseHandler) {
                for (const client of clients) {
                    client.removeListener('message', responseHandler);
                }
                responseHandler = null;
            }
        };
        
        // レスポンスハンドラの設定
        responseHandler = (ws: WebSocket.WebSocket, rawData: WebSocket.RawData) => {
            try {
                const data = JSON.parse(rawData.toString());
                
                // 対応するレスポンスの確認
                if (data.type === 'response' && data.id === requestId) {
                    cleanup();
                    resolve(data.result);
                }
            } catch (error) {
                // パースエラーは無視
            }
        };
        
        // タイムアウトの設定
        timeoutId = setTimeout(() => {
            cleanup();
            reject(new Error(`リクエスト '${method}' がタイムアウトしました`));
        }, timeout);
        
        // すべてのクライアントにリクエストを送信し、レスポンスハンドラを設定
        let sentCount = 0;
        for (const client of clients) {
            if (client.readyState === WebSocket.OPEN) {
                client.on('message', responseHandler);
                client.send(message);
                sentCount++;
            }
        }
        
        // 送信先がなければエラー
        if (sentCount === 0) {
            cleanup();
            reject(new Error('利用可能なクライアントがありません'));
        }
    });
}

/**
 * 現在のファイル情報を取得
 */
async function getCurrentFile(): Promise<any> {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        return {
            success: false,
            error: 'アクティブなエディタがありません'
        };
    }
    
    try {
        // ファイル情報の構築
        const document = editor.document;
        const selection = editor.selection;
        
        return {
            success: true,
            path: document.uri.fsPath,
            uri: document.uri.toString(),
            language: document.languageId,
            content: document.getText(),
            selection: {
                start: {
                    line: selection.start.line,
                    character: selection.start.character
                },
                end: {
                    line: selection.end.line,
                    character: selection.end.character
                }
            }
        };
    } catch (error) {
        console.error('ファイル情報取得中にエラーが発生しました:', error);
        return {
            success: false,
            error: `ファイル情報取得エラー: ${error instanceof Error ? error.message : String(error)}`
        };
    }
}

/**
 * プロジェクト構造情報を取得
 */
async function getProjectStructure(params: any): Promise<any> {
    try {
        // ワークスペースフォルダの確認
        if (!vscode.workspace.workspaceFolders || vscode.workspace.workspaceFolders.length === 0) {
            return {
                success: false,
                error: 'ワークスペースが開かれていません'
            };
        }
        
        // 除外パターンの取得
        const excludePatterns = params.excludePatterns || 
            vscode.workspace.getConfiguration('jarviee').get<string[]>('excludePatterns', 
                ['node_modules', '.git', '__pycache__']);
        
        // ワークスペース情報の構築
        const rootFolder = vscode.workspace.workspaceFolders[0];
        const workspaceFolders = vscode.workspace.workspaceFolders.map(folder => ({
            name: folder.name,
            path: folder.uri.fsPath,
            uri: folder.uri.toString()
        }));
        
        // ファイル一覧の取得（上位100件、実際の実装では分割取得などが必要）
        const maxFiles = 100;
        const searchPattern = new vscode.RelativePattern(rootFolder, '**/*');
        const excludeGlob = `{${excludePatterns.join(',')}}/**`;
        const files = await vscode.workspace.findFiles(searchPattern, excludeGlob, maxFiles);
        
        // ファイル情報の変換
        const fileInfo = files.map(file => {
            const relativePath = vscode.workspace.asRelativePath(file);
            const extension = path.extname(file.fsPath).toLowerCase();
            return {
                path: file.fsPath,
                uri: file.toString(),
                relativePath,
                extension
            };
        });
        
        // ファイル種類のカウント
        const fileTypes: Record<string, number> = {};
        for (const file of fileInfo) {
            const ext = file.extension || 'none';
            fileTypes[ext] = (fileTypes[ext] || 0) + 1;
        }
        
        return {
            success: true,
            rootPath: rootFolder.uri.fsPath,
            rootUri: rootFolder.uri.toString(),
            workspaceFolders,
            files: fileInfo,
            fileTypes,
            totalFiles: fileInfo.length,
            hasMoreFiles: fileInfo.length >= maxFiles
        };
    } catch (error) {
        console.error('プロジェクト構造取得中にエラーが発生しました:', error);
        return {
            success: false,
            error: `プロジェクト構造取得エラー: ${error instanceof Error ? error.message : String(error)}`
        };
    }
}

/**
 * コード編集を適用
 */
async function applyCodeEdit(params: any): Promise<any> {
    // パラメータの検証
    if (!params.filePath || !params.changes || !Array.isArray(params.changes)) {
        return {
            success: false,
            error: '無効なパラメータ'
        };
    }
    
    try {
        // ファイルURIの作成
        const fileUri = vscode.Uri.file(params.filePath);
        
        // ファイルが存在するか確認
        try {
            await vscode.workspace.fs.stat(fileUri);
        } catch (error) {
            return {
                success: false,
                error: `ファイルが存在しません: ${params.filePath}`
            };
        }
        
        // ドキュメントを開く
        const document = await vscode.workspace.openTextDocument(fileUri);
        
        // 編集ワークスペースを作成
        const workspaceEdit = new vscode.WorkspaceEdit();
        
        // 変更を適用
        for (const change of params.changes) {
            const startPos = new vscode.Position(change.start.line, change.start.character);
            const endPos = new vscode.Position(change.end.line, change.end.character);
            const range = new vscode.Range(startPos, endPos);
            
            workspaceEdit.replace(fileUri, range, change.text);
        }
        
        // 編集を実行
        const success = await vscode.workspace.applyEdit(workspaceEdit);
        
        return {
            success,
            filePath: params.filePath,
            changesApplied: success ? params.changes.length : 0
        };
    } catch (error) {
        console.error('コード編集適用中にエラーが発生しました:', error);
        return {
            success: false,
            error: `コード編集エラー: ${error instanceof Error ? error.message : String(error)}`
        };
    }
}

/**
 * コマンドを実行
 */
async function executeCommand(params: any): Promise<any> {
    // パラメータの検証
    if (!params.command) {
        return {
            success: false,
            error: 'コマンドが指定されていません'
        };
    }
    
    try {
        // コマンド引数の取得
        const args = params.args || [];
        
        // コマンドの実行
        const result = await vscode.commands.executeCommand(params.command, ...args);
        
        return {
            success: true,
            command: params.command,
            result
        };
    } catch (error) {
        console.error('コマンド実行中にエラーが発生しました:', error);
        return {
            success: false,
            error: `コマンド実行エラー: ${error instanceof Error ? error.message : String(error)}`
        };
    }
}

/**
 * 通知を表示
 */
async function showNotification(params: any): Promise<any> {
    // パラメータの検証
    if (!params.message) {
        return {
            success: false,
            error: 'メッセージが指定されていません'
        };
    }
    
    try {
        // 通知レベルの検証
        const level = params.level || 'info';
        
        // 通知の表示
        if (level === 'error') {
            vscode.window.showErrorMessage(params.message);
        } else if (level === 'warning') {
            vscode.window.showWarningMessage(params.message);
        } else {
            vscode.window.showInformationMessage(params.message);
        }
        
        return {
            success: true,
            level,
            message: params.message
        };
    } catch (error) {
        console.error('通知表示中にエラーが発生しました:', error);
        return {
            success: false,
            error: `通知表示エラー: ${error instanceof Error ? error.message : String(error)}`
        };
    }
}

/**
 * 診断情報を取得
 */
async function getDiagnostics(params: any): Promise<any> {
    try {
        let fileUri: vscode.Uri;
        
        // ファイルパスが指定されている場合はそれを使用
        if (params.filePath) {
            fileUri = vscode.Uri.file(params.filePath);
        } else {
            // アクティブなエディタを使用
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                return {
                    success: false,
                    error: 'アクティブなエディタがありません'
                };
            }
            
            fileUri = editor.document.uri;
        }
        
        // 診断情報の取得
        const diagnostics = vscode.languages.getDiagnostics(fileUri);
        
        // 診断情報の変換
        const diagnosticData = diagnostics.map(diag => ({
            severity: diag.severity,
            message: diag.message,
            source: diag.source,
            code: diag.code,
            range: {
                start: {
                    line: diag.range.start.line,
                    character: diag.range.start.character
                },
                end: {
                    line: diag.range.end.line,
                    character: diag.range.end.character
                }
            }
        }));
        
        return {
            success: true,
            filePath: fileUri.fsPath,
            uri: fileUri.toString(),
            diagnostics: diagnosticData
        };
    } catch (error) {
        console.error('診断情報取得中にエラーが発生しました:', error);
        return {
            success: false,
            error: `診断情報取得エラー: ${error instanceof Error ? error.message : String(error)}`
        };
    }
}

/**
 * ステータスバーアイテムの更新
 */
function updateStatusBarItem() {
    if (serverStatus.isRunning) {
        statusBarItem.text = `Jarviee: 接続中 (${serverStatus.connectedClients})`;
        statusBarItem.tooltip = `Jarvieeサーバー稼働中 - ポート: ${serverStatus.port}, クライアント: ${serverStatus.connectedClients}`;
        statusBarItem.backgroundColor = undefined;
    } else {
        statusBarItem.text = 'Jarviee: 切断';
        statusBarItem.tooltip = 'Jarvieeサーバーは停止中です';
        statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
    }
}

/**
 * 稼働時間のフォーマット
 */
function formatUptime(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (days > 0) {
        return `${days}日 ${hours % 24}時間`;
    } else if (hours > 0) {
        return `${hours}時間 ${minutes % 60}分`;
    } else if (minutes > 0) {
        return `${minutes}分 ${seconds % 60}秒`;
    } else {
        return `${seconds}秒`;
    }
}

/**
 * Webview用のユーティリティ関数
 */

/**
 * メトリクスのHTML生成
 */
function generateMetricsHTML(results: any): string {
    if (!results.metrics) {
        return '<p>メトリクス情報がありません</p>';
    }
    
    let html = '';
    for (const [name, value] of Object.entries(results.metrics)) {
        const metricClass = getMetricClass(name, value as any);
        html += `
            <div class="metric">
                <span class="metric-name">${name}</span>
                <span class="metric-value ${metricClass}">${value}</span>
            </div>
        `;
    }
    
    return html;
}

/**
 * 分析詳細のHTML生成
 */
function generateAnalysisDetailsHTML(results: any): string {
    if (!results.details) {
        return '<p>詳細分析情報がありません</p>';
    }
    
    let html = '';
    for (const [category, details] of Object.entries(results.details)) {
        html += `<h3>${category}</h3>`;
        
        if (Array.isArray(details)) {
            html += '<ul>';
            for (const item of details) {
                html += `<li>${item}</li>`;
            }
            html += '</ul>';
        } else {
            html += `<p>${details}</p>`;
        }
    }
    
    return html;
}

/**
 * 推奨事項のHTML生成
 */
function generateRecommendationsHTML(results: any): string {
    if (!results.recommendations) {
        return '<p>推奨事項がありません</p>';
    }
    
    let html = '';
    if (Array.isArray(results.recommendations)) {
        html += '<ul>';
        for (const item of results.recommendations) {
            html += `<li>${item}</li>`;
        }
        html += '</ul>';
    } else {
        html += `<p>${results.recommendations}</p>`;
    }
    
    return html;
}

/**
 * エラー診断のHTML生成
 */
function generateErrorDiagnosticsHTML(results: any): string {
    if (!results.errors) {
        return '<p>エラー情報がありません</p>';
    }
    
    let html = '';
    for (const error of results.errors) {
        html += `
            <div class="error-box">
                <div class="error-message">${error.message}</div>
                <div class="error-location">場所: ${error.location || '不明'}</div>
                <div>${error.description || ''}</div>
            </div>
        `;
    }
    
    return html;
}

/**
 * 解決策のHTML生成
 */
function generateSolutionsHTML(results: any): string {
    if (!results.solutions) {
        return '<p>解決策情報がありません</p>';
    }
    
    let html = '';
    if (results.fixedCode) {
        html += `
            <h3>修正コード:</h3>
            <pre><code>${escapeHtml(results.fixedCode)}</code></pre>
        `;
    }
    
    if (Array.isArray(results.solutions)) {
        html += '<h3>解決手順:</h3><ol>';
        for (const item of results.solutions) {
            html += `<li>${item}</li>`;
        }
        html += '</ol>';
    } else {
        html += `<p>${results.solutions}</p>`;
    }
    
    return html;
}

/**
 * ベストプラクティスのHTML生成
 */
function generateBestPracticesHTML(results: any): string {
    if (!results.bestPractices) {
        return '<p>ベストプラクティス情報がありません</p>';
    }
    
    let html = '';
    if (Array.isArray(results.bestPractices)) {
        html += '<ul>';
        for (const item of results.bestPractices) {
            html += `<li>${item}</li>`;
        }
        html += '</ul>';
    } else {
        html += `<p>${results.bestPractices}</p>`;
    }
    
    return html;
}

/**
 * 最適化詳細のHTML生成
 */
function generateOptimizationDetailsHTML(results: any): string {
    if (!results.details) {
        return '<p>最適化の詳細情報がありません</p>';
    }
    
    let html = '';
    if (typeof results.details === 'string') {
        html += `<p>${results.details}</p>`;
    } else {
        for (const [category, details] of Object.entries(results.details)) {
            html += `<h3>${category}</h3>`;
            
            if (Array.isArray(details)) {
                html += '<ul>';
                for (const item of details) {
                    html += `<li>${item}</li>`;
                }
                html += '</ul>';
            } else {
                html += `<p>${details}</p>`;
            }
        }
    }
    
    return html;
}

/**
 * 改善点のHTML生成
 */
function generateImprovementsHTML(results: any): string {
    if (!results.improvements) {
        return '<p>パフォーマンス改善点がありません</p>';
    }
    
    let html = '';
    if (Array.isArray(results.improvements)) {
        for (const item of results.improvements) {
            html += `
                <div class="improvement-item">
                    <div><strong>${item.title || 'パフォーマンスの改善'}</strong></div>
                    <div>${item.description || ''}</div>
                </div>
            `;
        }
    } else {
        html += `<p>${results.improvements}</p>`;
    }
    
    return html;
}

/**
 * メトリクスのクラス取得
 */
function getMetricClass(name: string, value: any): string {
    // メトリックの種類や値に基づいて適切なスタイルクラスを返す
    if (typeof value !== 'number') {
        return '';
    }
    
    const lowerName = name.toLowerCase();
    
    if (lowerName.includes('complexity') || lowerName.includes('複雑性')) {
        return value > 15 ? 'metric-bad' : (value > 10 ? 'metric-warning' : 'metric-good');
    }
    
    if (lowerName.includes('coverage') || lowerName.includes('カバレッジ')) {
        return value < 50 ? 'metric-bad' : (value < 80 ? 'metric-warning' : 'metric-good');
    }
    
    if (lowerName.includes('error') || lowerName.includes('エラー') || 
        lowerName.includes('warning') || lowerName.includes('警告')) {
        return value > 0 ? 'metric-bad' : 'metric-good';
    }
    
    return '';
}

/**
 * HTMLエスケープ
 */
function escapeHtml(text: string): string {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

/**
 * 分析結果WebView用のHTMLコンテンツ生成
 */
function getAnalysisWebviewContent(results: any, language: string): string {
    return `
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Jarviee コード分析</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                }
                h1, h2, h3 {
                    font-weight: 500;
                    margin-top: 24px;
                    margin-bottom: 16px;
                    color: var(--vscode-editor-foreground);
                }
                h1 {
                    font-size: 24px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                    padding-bottom: 10px;
                }
                h2 {
                    font-size: 20px;
                }
                h3 {
                    font-size: 16px;
                }
                pre {
                    background-color: var(--vscode-textCodeBlock-background);
                    padding: 12px;
                    border-radius: 3px;
                    overflow: auto;
                }
                code {
                    font-family: 'SF Mono', Monaco, Menlo, Consolas, 'Ubuntu Mono', 'Liberation Mono', 'DejaVu Sans Mono', 'Courier New', monospace;
                    font-size: 13px;
                }
                ul {
                    padding-left: 20px;
                }
                .metric {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 4px;
                    padding: 4px 8px;
                    background-color: var(--vscode-list-hoverBackground);
                    border-radius: 3px;
                }
                .metric-name {
                    font-weight: 500;
                }
                .metric-good {
                    color: var(--vscode-testing-iconPassed);
                }
                .metric-warning {
                    color: var(--vscode-editorWarning-foreground);
                }
                .metric-bad {
                    color: var(--vscode-editorError-foreground);
                }
                .recommendations {
                    margin-top: 16px;
                    padding: 12px;
                    background-color: var(--vscode-editorWidget-background);
                    border-left: 4px solid var(--vscode-activityBarBadge-background);
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <h1>コード分析結果</h1>
            <p>言語: ${language}</p>
            
            <h2>概要</h2>
            <div class="metrics">
                ${generateMetricsHTML(results)}
            </div>
            
            <h2>詳細分析</h2>
            ${generateAnalysisDetailsHTML(results)}
            
            <h2>推奨事項</h2>
            <div class="recommendations">
                ${generateRecommendationsHTML(results)}
            </div>
        </body>
        </html>
    `;
}

/**
 * デバッグヘルプWebView用のHTMLコンテンツ生成
 */
function getDebugHelpWebviewContent(results: any, language: string): string {
    return `
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Jarviee デバッグヘルプ</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                }
                h1, h2, h3 {
                    font-weight: 500;
                    margin-top: 24px;
                    margin-bottom: 16px;
                    color: var(--vscode-editor-foreground);
                }
                h1 {
                    font-size: 24px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                    padding-bottom: 10px;
                }
                h2 {
                    font-size: 20px;
                }
                h3 {
                    font-size: 16px;
                }
                pre {
                    background-color: var(--vscode-textCodeBlock-background);
                    padding: 12px;
                    border-radius: 3px;
                    overflow: auto;
                }
                code {
                    font-family: 'SF Mono', Monaco, Menlo, Consolas, 'Ubuntu Mono', 'Liberation Mono', 'DejaVu Sans Mono', 'Courier New', monospace;
                    font-size: 13px;
                }
                ul {
                    padding-left: 20px;
                }
                .error-box {
                    margin-bottom: 16px;
                    padding: 12px;
                    background-color: var(--vscode-inputValidation-errorBackground);
                    border-left: 4px solid var(--vscode-inputValidation-errorBorder);
                    border-radius: 3px;
                }
                .error-message {
                    font-weight: 500;
                    color: var(--vscode-inputValidation-errorForeground);
                }
                .error-location {
                    font-family: 'SF Mono', Monaco, Menlo, Consolas, monospace;
                    margin-top: 8px;
                    color: var(--vscode-foreground);
                }
                .solution {
                    margin-top: 16px;
                    padding: 12px;
                    background-color: var(--vscode-editorWidget-background);
                    border-left: 4px solid var(--vscode-activityBarBadge-background);
                    border-radius: 3px;
                }
                .best-practices {
                    margin-top: 24px;
                    padding: 12px;
                    background-color: var(--vscode-editorWidget-background);
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <h1>デバッグヘルプ</h1>
            <p>言語: ${language}</p>
            
            <h2>エラー診断</h2>
            ${generateErrorDiagnosticsHTML(results)}
            
            <h2>解決策</h2>
            <div class="solution">
                ${generateSolutionsHTML(results)}
            </div>
            
            <h2>ベストプラクティス</h2>
            <div class="best-practices">
                ${generateBestPracticesHTML(results)}
            </div>
        </body>
        </html>
    `;
}

/**
 * 最適化結果WebView用のHTMLコンテンツ生成
 */
function getOptimizationWebviewContent(results: any, language: string): string {
    return `
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Jarviee コード最適化</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                }
                h1, h2, h3 {
                    font-weight: 500;
                    margin-top: 24px;
                    margin-bottom: 16px;
                    color: var(--vscode-editor-foreground);
                }
                h1 {
                    font-size: 24px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                    padding-bottom: 10px;
                }
                h2 {
                    font-size: 20px;
                }
                h3 {
                    font-size: 16px;
                }
                pre {
                    background-color: var(--vscode-textCodeBlock-background);
                    padding: 12px;
                    border-radius: 3px;
                    overflow: auto;
                }
                code {
                    font-family: 'SF Mono', Monaco, Menlo, Consolas, 'Ubuntu Mono', 'Liberation Mono', 'DejaVu Sans Mono', 'Courier New', monospace;
                    font-size: 13px;
                }
                .code-container {
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                .code-title {
                    font-weight: 500;
                    margin-bottom: 8px;
                }
                .optimization-details {
                    margin-top: 24px;
                    padding: 12px;
                    background-color: var(--vscode-editorWidget-background);
                    border-radius: 3px;
                }
                .improvements {
                    margin-top: 16px;
                }
                .improvement-item {
                    margin-bottom: 8px;
                    padding: 8px;
                    background-color: var(--vscode-list-hoverBackground);
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <h1>コード最適化結果</h1>
            <p>言語: ${language}</p>
            
            <h2>最適化コード</h2>
            <div class="code-container">
                <div>
                    <div class="code-title">オリジナルコード</div>
                    <pre><code>${escapeHtml(results.originalCode || '')}</code></pre>
                </div>
                <div>
                    <div class="code-title">最適化コード</div>
                    <pre><code>${escapeHtml(results.optimizedCode || '')}</code></pre>
                </div>
            </div>
            
            <h2>最適化の詳細</h2>
            <div class="optimization-details">
                ${generateOptimizationDetailsHTML(results)}
            </div>
            
            <h2>パフォーマンス改善点</h2>
            <div class="improvements">
                ${generateImprovementsHTML(results)}
            </div>
        </body>
        </html>
    `;
}
