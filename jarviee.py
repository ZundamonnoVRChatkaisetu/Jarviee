#!/usr/bin/env python
"""
Jarviee システムのメインエントリーポイント
"""
import os
import sys
import argparse
import traceback

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
        choices=["cli", "api", "gui"],
        default="cli",
        help="使用するインターフェースモード（デフォルト: cli）"
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
        help="GPUを使用してモデルを実行（デフォルト: .envファイルの設定に従う）"
    )
    
    return parser.parse_args()

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
        
        # GPU設定
        if args.gpu:
            os.environ["USE_GPU"] = "true"
            print("GPUを使用してモデルを実行します")
        
        # 設定ファイルの設定
        if args.config:
            os.environ["JARVIEE_CONFIG"] = os.path.abspath(args.config)
        
        # 選択されたモードに基づいてインターフェースを起動
        if args.mode == "cli":
            from src.interfaces.cli.jarviee_cli import main as cli_main
            cli_main()
        elif args.mode == "api":
            from src.interfaces.api.server import main as api_main
            api_main()
        elif args.mode == "gui":
            from src.interfaces.ui.app import main as gui_main
            gui_main()
        else:
            print(f"エラー: 不明なモード {args.mode}")
            sys.exit(1)
            
    except ImportError as e:
        print(f"モジュールインポートエラー: {str(e)}")
        print("必要なモジュールがインストールされていない可能性があります。")
        print("pip install -r requirements.txtを実行してください。")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
