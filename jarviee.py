#!/usr/bin/env python
"""
Jarviee システムのメインエントリーポイント
GPUサポートとインターフェース選択機能を含む統合版
"""
import os
import sys
import argparse
import traceback
import json
import time
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

def setup_environment():
    """環境のセットアップ"""
    # モジュールパスの設定
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # .env ファイルから環境変数を読み込む
    try:
        from dotenv import load_dotenv
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        load_dotenv(dotenv_path)
        print("環境変数を.envファイルから読み込みました")
    except ImportError:
        print("python-dotenvがインストールされていません。環境変数の読み込みをスキップします。")
    except Exception as e:
        print(f".envファイルの読み込み中にエラーが発生しました: {str(e)}")

def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="Jarviee AIシステム")
    
    # インターフェースモード選択
    parser.add_argument(
        "--mode", "-m",
        choices=["cli", "api", "gui", "interactive"],
        default="interactive",
        help="使用するインターフェースモード（デフォルト: interactive）"
    )
    
    # 設定ファイル
    parser.add_argument(
        "--config", "-c",
        help="使用する設定ファイルのパス"
    )
    
    # デバッグモード
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="デバッグモードを有効化"
    )
    
    # GPU使用設定
    parser.add_argument(
        "--gpu", "-g",
        action="store_true",
        help="GPUを使用してモデルを実行（デフォルト: 自動検出）"
    )
    
    # GPU無効化
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="GPUを使用せずCPUのみで実行"
    )
    
    # GPUレイヤー数
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="GPUで実行するレイヤー数 (-1=全て, 0=なし, n=n層)"
    )
    
    # GPU診断モード
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="GPU診断を実行して終了"
    )
    
    return parser.parse_args()

def check_gpu_support() -> Dict[str, Any]:
    """
    GPU対応状況を診断
    
    Returns:
        診断結果を含む辞書
    """
    result = {
        "has_cuda": False,
        "has_torch": False,
        "gpu_count": 0,
        "gpu_names": [],
        "has_llama_cpp": False,
        "llama_cpp_gpu": False,
        "recommended": False
    }
    
    # PyTorchチェック
    try:
        import torch
        result["has_torch"] = True
        
        # CUDA利用可能性チェック
        result["has_cuda"] = torch.cuda.is_available()
        if result["has_cuda"]:
            result["gpu_count"] = torch.cuda.device_count()
            result["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(result["gpu_count"])]
    except ImportError:
        result["has_torch"] = False
    
    # llama-cpp-pythonチェック
    try:
        from llama_cpp import Llama
        result["has_llama_cpp"] = True
        
        # GPU対応ビルドかチェック
        import inspect
        llama_init_args = inspect.signature(Llama.__init__).parameters
        result["llama_cpp_gpu"] = 'n_gpu_layers' in llama_init_args
    except ImportError:
        result["has_llama_cpp"] = False
    
    # 推奨設定判定
    result["recommended"] = result["has_torch"] and result["has_cuda"] and result["has_llama_cpp"] and result["llama_cpp_gpu"]
    
    return result

def print_gpu_diagnostic(result: Dict[str, Any], verbose: bool = False) -> None:
    """
    GPU診断結果を表示
    
    Args:
        result: 診断結果
        verbose: 詳細表示するかどうか
    """
    try:
        # リッチ出力が使用可能か確認
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        
        console = Console()
        
        # 診断テーブルの作成
        table = Table(title="GPU診断結果", show_header=True, header_style="bold magenta")
        table.add_column("項目", style="cyan")
        table.add_column("状態", style="green")
        
        table.add_row("PyTorch", "✅ インストール済み" if result["has_torch"] else "❌ 未インストール")
        table.add_row("CUDA", "✅ 利用可能" if result["has_cuda"] else "❌ 利用不可")
        
        gpu_text = "なし"
        if result["gpu_count"] > 0:
            gpu_text = f"{result['gpu_count']}台のGPUが見つかりました:"
            for i, name in enumerate(result["gpu_names"]):
                gpu_text += f"\n    - GPU {i}: {name}"
        table.add_row("GPUデバイス", gpu_text)
        
        table.add_row("llama-cpp-python", "✅ インストール済み" if result["has_llama_cpp"] else "❌ 未インストール")
        table.add_row("GPUサポート", "✅ サポート" if result["llama_cpp_gpu"] else "❌ 未サポート")
        
        status = "✅ 推奨設定" if result["recommended"] else "⚠️ 最適化が必要"
        table.add_row("総合判定", status)
        
        console.print(table)
        
        # GPUサポートがない場合のアドバイス
        if not result["recommended"]:
            console.print("\n[bold yellow]GPU対応版のllama-cpp-pythonをインストールするには:[/]")
            console.print("""
pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
            
または、以下の公式リポジトリを参照してください:
https://github.com/abetlen/llama-cpp-python
            """)
    except ImportError:
        # リッチ出力が使用できない場合はシンプルに表示
        print("===== GPU診断結果 =====")
        print(f"PyTorch: {'インストール済み' if result['has_torch'] else '未インストール'}")
        print(f"CUDA: {'利用可能' if result['has_cuda'] else '利用不可'}")
        
        gpu_text = "なし"
        if result["gpu_count"] > 0:
            gpu_text = f"{result['gpu_count']}台のGPUが見つかりました:"
            for i, name in enumerate(result["gpu_names"]):
                gpu_text += f"\n    - GPU {i}: {name}"
        print(f"GPUデバイス: {gpu_text}")
        
        print(f"llama-cpp-python: {'インストール済み' if result['has_llama_cpp'] else '未インストール'}")
        print(f"GPUサポート: {'サポート' if result['llama_cpp_gpu'] else '未サポート'}")
        
        status = "推奨設定" if result["recommended"] else "最適化が必要"
        print(f"総合判定: {status}")
        
        if not result["recommended"]:
            print("\nGPU対応版のllama-cpp-pythonをインストールするには:")
            print("pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117")
            print("\nまたは、以下の公式リポジトリを参照してください:")
            print("https://github.com/abetlen/llama-cpp-python")

def update_config_for_gpu(config_path: str, use_gpu: bool, n_gpu_layers: int = -1) -> bool:
    """
    設定ファイルのGPU設定を更新
    
    Args:
        config_path: 設定ファイルパス
        use_gpu: GPU使用フラグ
        n_gpu_layers: GPUレイヤー数
        
    Returns:
        更新成功したかどうか
    """
    try:
        # 設定ファイルを読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Gemmaプロバイダー設定の更新
        if "llm" in config and "providers" in config["llm"] and "gemma" in config["llm"]["providers"]:
            gemma_config = config["llm"]["providers"]["gemma"]
            gemma_config["use_gpu"] = use_gpu
            gemma_config["n_gpu_layers"] = n_gpu_layers
            
            # 設定を書き込み
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
            return True
    except Exception as e:
        print(f"設定ファイルの更新エラー: {e}")
        return False
    
    return False

def display_interface_selection() -> str:
    """
    インターフェースモードの選択プロンプトを表示
    
    Returns:
        選択されたモード
    """
    try:
        # リッチ表示が使用可能か確認
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt
        
        console = Console()
        
        # バナー表示
        banner = """
         ▄▄▄██▀▀▀▄▄▄       ██▀███   ██▒   █▓ ██▓▓█████ ▓█████ 
           ▒██  ▒████▄    ▓██ ▒ ██▒▓██░   █▒▓██▒▓█   ▀ ▓█   ▀ 
           ░██  ▒██  ▀█▄  ▓██ ░▄█ ▒ ▓██  █▒░▒██▒▒███   ▒███   
        ▓██▄██▓ ░██▄▄▄▄██ ▒██▀▀█▄    ▒██ █░░░██░▒▓█  ▄ ▒▓█  ▄ 
         ▓███▒   ▓█   ▓██▒░██▓ ▒██▒   ▒▀█░  ░██░░▒████▒░▒████▒
         ▒▓▒▒░   ▒▒   ▓▒█░░ ▒▓ ░▒▓░   ░ ▐░  ░▓  ░░ ▒░ ░░░ ▒░ ░
         ▒ ░▒░    ▒   ▒▒ ░  ░▒ ░ ▒░   ░ ░░   ▒ ░ ░ ░  ░ ░ ░  ░
         ░ ░ ░    ░   ▒     ░░   ░      ░░   ▒ ░   ░      ░   
         ░   ░        ░  ░   ░           ░   ░     ░  ░   ░  ░
                                         ░                    
               AI Technologies Integration Framework
        """
        console.print(Panel(banner, border_style="blue", expand=False))
        
        console.print("[bold cyan]「てゅん、ようこそ。ジャーヴィーのインターフェースを選択してください。」[/]")
        
        # モード選択
        choices = {
            "1": ("CLI", "コマンドラインインターフェース - フルコマンド操作"),
            "2": ("GUI", "グラフィカルインターフェース - ブラウザベースUI"),
            "3": ("API", "APIサーバー - 他のアプリケーションから利用")
        }
        
        # 選択肢表示
        for key, (mode, desc) in choices.items():
            console.print(f"[bold]{key}[/] - [cyan]{mode}[/]: {desc}")
        
        # ユーザー入力
        selection = Prompt.ask(
            "インターフェースを選択",
            choices=list(choices.keys()),
            default="1"
        )
        
        # 選択に応じたモードを返す
        mode_map = {"1": "cli", "2": "gui", "3": "api"}
        selected_mode = mode_map[selection]
        
        console.print(f"[bold green]「{choices[selection][0]}モードで起動します...」[/]")
        return selected_mode
        
    except ImportError:
        # リッチ表示が使用できない場合はシンプルに表示
        print("===== Jarviee インターフェース選択 =====")
        print("利用可能なモード:")
        print("1 - CLI: コマンドラインインターフェース")
        print("2 - GUI: グラフィカルインターフェース")
        print("3 - API: APIサーバー")
        
        while True:
            try:
                selection = input("選択 (1-3, デフォルト: 1): ").strip()
                if not selection:
                    selection = "1"
                if selection in ["1", "2", "3"]:
                    break
                print("無効な選択です。1, 2, 3のいずれかを入力してください。")
            except KeyboardInterrupt:
                print("\n終了します。")
                sys.exit(0)
        
        # 選択に応じたモードを返す
        mode_map = {"1": "cli", "2": "gui", "3": "api"}
        selected_mode = mode_map[selection]
        
        print(f"'{selected_mode}'モードで起動します...")
        return selected_mode

def jarvis_say(message: str) -> None:
    """
    ジャーヴィス風のメッセージを表示
    
    Args:
        message: 表示するメッセージ
    """
    try:
        from rich.console import Console
        console = Console()
        console.print(f"[bold cyan]「{message}」[/]")
    except ImportError:
        print(f"「{message}」")

def is_first_run() -> bool:
    """
    初回実行かどうかを確認
    
    Returns:
        初回実行かどうか
    """
    first_run_flag = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".firstrun")
    if not os.path.exists(first_run_flag):
        # 初回実行フラグを作成
        try:
            with open(first_run_flag, 'w') as f:
                f.write(str(int(time.time())))
            return True
        except:
            pass
    return False

def main():
    """メインエントリーポイント"""
    try:
        # 環境のセットアップ
        setup_environment()
        
        # 引数のパース
        args = parse_args()
        
        # デバッグモードの設定
        if args.debug:
            os.environ["JARVIEE_DEBUG"] = "1"
            print("デバッグモードが有効です")
        
        # GPU診断モードの場合
        if args.check_gpu:
            result = check_gpu_support()
            print_gpu_diagnostic(result, verbose=True)
            return
        
        # GPU設定
        if args.cpu_only:
            # CPU強制モード
            os.environ["USE_GPU"] = "false"
            jarvis_say("CPUモードで実行します。計算速度は低下しますが、てゅん。")
        elif args.gpu:
            # GPU明示的有効モード
            os.environ["USE_GPU"] = "true"
            os.environ["GPU_LAYERS"] = str(args.gpu_layers)
            jarvis_say("GPUを最大限活用して実行します。")
        else:
            # 自動判定モード
            result = check_gpu_support()
            if result["recommended"]:
                os.environ["USE_GPU"] = "true"
                os.environ["GPU_LAYERS"] = str(args.gpu_layers)
                if result["gpu_names"]:
                    jarvis_say(f"GPUを検出しました: {', '.join(result['gpu_names'])}")
                else:
                    jarvis_say("GPUを検出しました。")
            elif result["has_cuda"] and result["has_torch"]:
                os.environ["USE_GPU"] = "true"
                os.environ["GPU_LAYERS"] = str(args.gpu_layers)
                jarvis_say("GPUを検出しましたが、最適化されていません。パフォーマンスに影響する可能性があります。")
            else:
                os.environ["USE_GPU"] = "false"
                jarvis_say("利用可能なGPUが見つかりませんでした。CPUモードで実行します。")
        
        # 設定ファイルの設定
        if args.config:
            config_path = os.path.abspath(args.config)
            os.environ["JARVIEE_CONFIG"] = config_path
        else:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.json")
            if os.path.exists(config_path):
                os.environ["JARVIEE_CONFIG"] = config_path
        
        # 設定ファイルのGPU設定を更新
        if os.path.exists(config_path):
            use_gpu = os.environ.get("USE_GPU", "").lower() == "true"
            n_gpu_layers = int(os.environ.get("GPU_LAYERS", "-1"))
            update_config_for_gpu(config_path, use_gpu, n_gpu_layers)
        
        # 初回実行メッセージ
        if is_first_run():
            jarvis_say("初めまして、てゅん。ジャーヴィーシステムを起動しています。設定を最適化中です...")
            # 診断実行
            result = check_gpu_support()
            print_gpu_diagnostic(result)
            time.sleep(1)
        
        # インターフェースモードの選択
        selected_mode = args.mode
        if selected_mode == "interactive":
            selected_mode = display_interface_selection()
        
        # 選択されたモードに基づいてインターフェースを起動
        if selected_mode == "cli":
            jarvis_say("CLIモードでシステムを起動します...")
            
            # CLIを起動（直接モジュールを呼び出す方法に変更）
            from src.interfaces.cli.jarviee_cli import JarvieeCLI
            cli = JarvieeCLI(config_path=os.environ.get("JARVIEE_CONFIG"))
            cli.run_interactive_mode()
            
        elif selected_mode == "api":
            jarvis_say("APIサーバーとしてシステムを起動します...")
            from src.interfaces.api.server import main as api_main
            api_main()
        elif selected_mode == "gui":
            jarvis_say("GUIモードでシステムを起動します。ブラウザインターフェースを準備中...")
            from src.interfaces.ui.app import main as gui_main
            gui_main()
        else:
            print(f"エラー: 不明なモード {selected_mode}")
            sys.exit(1)
            
    except ImportError as e:
        print(f"モジュールインポートエラー: {str(e)}")
        print("必要なモジュールがインストールされていない可能性があります。")
        print("pip install -r requirements.txtを実行してください。")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nシステムを終了します。")
        sys.exit(0)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
