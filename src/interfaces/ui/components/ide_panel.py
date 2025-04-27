"""
IDE連携パネルコンポーネント

Jarvieeシステムと各種IDEの連携機能を提供するインターフェースコンポーネント
"""

import gradio as gr
import json
import os
from typing import Callable, Dict, List, Tuple, Any, Optional

class IDEPanel:
    """IDE連携インターフェースを管理するクラス"""
    
    def __init__(
        self,
        get_ides_handler: Callable[[], List[Dict]],
        connect_handler: Callable[[str], Dict],
        disconnect_handler: Callable[[], Dict],
        install_handler: Callable[[str], Dict],
        uninstall_handler: Callable[[str], Dict],
        launch_handler: Callable[[str, str], Dict],
        system_state: gr.State
    ):
        """
        IDE連携パネルを初期化
        
        Args:
            get_ides_handler: 利用可能なIDE一覧を取得するコールバック関数
            connect_handler: IDE接続を処理するコールバック関数
            disconnect_handler: IDE切断を処理するコールバック関数
            install_handler: 拡張インストールを処理するコールバック関数
            uninstall_handler: 拡張アンインストールを処理するコールバック関数
            launch_handler: IDE起動を処理するコールバック関数
            system_state: システム状態を保持するgradio.State
        """
        self.get_ides_handler = get_ides_handler
        self.connect_handler = connect_handler
        self.disconnect_handler = disconnect_handler
        self.install_handler = install_handler
        self.uninstall_handler = uninstall_handler
        self.launch_handler = launch_handler
        self.system_state = system_state
        
        # UI要素
        self.ide_status = None
        self.ide_list = None
        self.selected_ide = None
        self.project_path = None
        self.connection_result = None
        self.file_info = None
        self.file_content = None
    
    def create(self) -> gr.Column:
        """
        IDE連携パネルUIを作成
        
        Returns:
            gr.Column: IDE連携パネルを含むカラム
        """
        with gr.Column() as panel:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### IDE連携設定")
                    
                    # IDE選択ドロップダウン
                    self.selected_ide = gr.Dropdown(
                        choices=["vscode", "intellij", "eclipse"],
                        label="IDE",
                        value="vscode"
                    )
                    
                    # プロジェクトパス入力
                    self.project_path = gr.Textbox(
                        label="プロジェクトパス (オプション)",
                        placeholder="例: C:\\Projects\\MyProject"
                    )
                    
                    # 接続／切断ボタン
                    with gr.Row():
                        connect_btn = gr.Button("接続", variant="primary")
                        disconnect_btn = gr.Button("切断", variant="secondary")
                    
                    # 拡張管理ボタン
                    with gr.Row():
                        install_btn = gr.Button("拡張インストール")
                        uninstall_btn = gr.Button("拡張アンインストール")
                    
                    # IDE起動ボタン
                    launch_btn = gr.Button("IDEを起動")
                    
                    # 結果表示
                    self.connection_result = gr.Markdown("IDE連携の状態がここに表示されます")
                
                with gr.Column(scale=2):
                    # IDE状態表示
                    self.ide_status = gr.Markdown("### 接続状態: 未接続")
                    
                    gr.Markdown("### 利用可能なIDE")
                    
                    # IDE一覧表示
                    self.ide_list = gr.Dataframe(
                        headers=["ID", "名前", "バージョン", "インストール済み", "アクティブ"],
                        value=[
                            ["vscode", "Visual Studio Code", "1.0.0", "はい", "いいえ"],
                            ["intellij", "IntelliJ IDEA", "未実装", "いいえ", "いいえ"]
                        ],
                        interactive=False
                    )
                    
                    # 更新ボタン
                    refresh_btn = gr.Button("IDE一覧を更新")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 現在のファイル情報")
                    
                    # ファイル情報表示
                    self.file_info = gr.JSON(
                        label="ファイル情報",
                        value={"path": "", "language": "", "size": 0, "modified": ""}
                    )
                
                with gr.Column():
                    gr.Markdown("### ファイル内容")
                    
                    # ファイル内容表示
                    self.file_content = gr.Code(
                        label="内容",
                        language="python",
                        value="# ファイル内容がここに表示されます",
                        interactive=False
                    )
            
            with gr.Row():
                gr.Markdown("""
                ### IDE連携機能 - プロトタイプ版
                
                現在対応しているIDE:
                - Visual Studio Code (実装済み)
                - IntelliJ IDEA (進行中)
                
                プロトタイプ版では一部機能が制限される場合があります。
                """)
            
            # イベント設定
            connect_btn.click(
                self._handle_connect,
                [self.selected_ide],
                [self.ide_status, self.connection_result]
            )
            
            disconnect_btn.click(
                self._handle_disconnect,
                [],
                [self.ide_status, self.connection_result]
            )
            
            install_btn.click(
                self._handle_install,
                [self.selected_ide],
                [self.connection_result]
            )
            
            uninstall_btn.click(
                self._handle_uninstall,
                [self.selected_ide],
                [self.connection_result]
            )
            
            launch_btn.click(
                self._handle_launch,
                [self.selected_ide, self.project_path],
                [self.connection_result]
            )
            
            refresh_btn.click(
                self._handle_refresh_ides,
                [],
                [self.ide_list]
            )
        
        return panel
    
    def _handle_connect(self, ide_name: str) -> Tuple[str, str]:
        """
        IDE接続を処理
        
        Args:
            ide_name: IDE名
            
        Returns:
            Tuple[str, str]: (IDE状態, 接続結果)
        """
        try:
            # ハンドラーを呼び出し
            result = self.connect_handler(ide_name)
            
            if result.get("success", False):
                status = f"### 接続状態: 接続済み ({ide_name})"
                connection_result = f"**成功**: {ide_name}に接続しました"
                
                # システム状態の更新
                if isinstance(self.system_state.value, dict):
                    self.system_state.value["ide"] = ide_name
                    self.system_state.value["ide_connected"] = True
                
                return status, connection_result
            else:
                error_msg = result.get("error", "不明なエラー")
                status = "### 接続状態: 未接続"
                connection_result = f"**エラー**: {ide_name}への接続に失敗しました: {error_msg}"
                return status, connection_result
        
        except Exception as e:
            return "### 接続状態: エラー", f"**例外**: IDE接続中に例外が発生しました: {str(e)}"
    
    def _handle_disconnect(self) -> Tuple[str, str]:
        """
        IDE切断を処理
        
        Returns:
            Tuple[str, str]: (IDE状態, 切断結果)
        """
        try:
            # ハンドラーを呼び出し
            result = self.disconnect_handler()
            
            if result.get("success", False):
                status = "### 接続状態: 未接続"
                connection_result = "**成功**: IDEから切断しました"
                
                # システム状態の更新
                if isinstance(self.system_state.value, dict):
                    self.system_state.value["ide"] = None
                    self.system_state.value["ide_connected"] = False
                
                return status, connection_result
            else:
                error_msg = result.get("error", "不明なエラー")
                status = "### 接続状態: エラー"
                connection_result = f"**エラー**: IDEからの切断に失敗しました: {error_msg}"
                return status, connection_result
        
        except Exception as e:
            return "### 接続状態: エラー", f"**例外**: IDE切断中に例外が発生しました: {str(e)}"
    
    def _handle_install(self, ide_name: str) -> str:
        """
        拡張インストールを処理
        
        Args:
            ide_name: IDE名
            
        Returns:
            str: インストール結果
        """
        try:
            # ハンドラーを呼び出し
            result = self.install_handler(ide_name)
            
            if result.get("success", False):
                return f"**成功**: {ide_name}拡張をインストールしました"
            else:
                error_msg = result.get("error", "不明なエラー")
                return f"**エラー**: {ide_name}拡張のインストールに失敗しました: {error_msg}"
        
        except Exception as e:
            return f"**例外**: 拡張インストール中に例外が発生しました: {str(e)}"
    
    def _handle_uninstall(self, ide_name: str) -> str:
        """
        拡張アンインストールを処理
        
        Args:
            ide_name: IDE名
            
        Returns:
            str: アンインストール結果
        """
        try:
            # ハンドラーを呼び出し
            result = self.uninstall_handler(ide_name)
            
            if result.get("success", False):
                return f"**成功**: {ide_name}拡張をアンインストールしました"
            else:
                error_msg = result.get("error", "不明なエラー")
                return f"**エラー**: {ide_name}拡張のアンインストールに失敗しました: {error_msg}"
        
        except Exception as e:
            return f"**例外**: 拡張アンインストール中に例外が発生しました: {str(e)}"
    
    def _handle_launch(self, ide_name: str, project_path: str) -> str:
        """
        IDE起動を処理
        
        Args:
            ide_name: IDE名
            project_path: プロジェクトパス
            
        Returns:
            str: 起動結果
        """
        try:
            # パスが指定されているか確認
            if project_path and not os.path.exists(project_path):
                return f"**警告**: 指定されたプロジェクトパスが存在しません: {project_path}"
            
            # ハンドラーを呼び出し
            result = self.launch_handler(ide_name, project_path)
            
            if result.get("success", False):
                path_info = f"（プロジェクト: {project_path}）" if project_path else ""
                return f"**成功**: {ide_name}を起動しました{path_info}"
            else:
                error_msg = result.get("error", "不明なエラー")
                return f"**エラー**: {ide_name}の起動に失敗しました: {error_msg}"
        
        except Exception as e:
            return f"**例外**: IDE起動中に例外が発生しました: {str(e)}"
    
    def _handle_refresh_ides(self) -> List:
        """
        IDE一覧を更新
        
        Returns:
            List: 更新されたIDE一覧
        """
        try:
            # ハンドラーを呼び出し
            ides = self.get_ides_handler()
            
            # データをテーブル形式に変換
            ide_rows = []
            for ide in ides:
                ide_rows.append([
                    ide.get("id", ""),
                    ide.get("name", ""),
                    ide.get("version", ""),
                    "はい" if ide.get("installed", False) else "いいえ",
                    "はい" if ide.get("is_active", False) else "いいえ"
                ])
            
            # IDEが見つからない場合
            if not ide_rows:
                return [["情報なし", "利用可能なIDEがありません", "", "", ""]]
            
            return ide_rows
        
        except Exception as e:
            return [["エラー", f"IDE一覧の取得中に例外が発生しました: {str(e)}", "", "", ""]]
