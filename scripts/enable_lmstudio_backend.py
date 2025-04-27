"""
LMStudioのバックエンドを活用するためのスクリプト
てゅん用に作成しました。
"""

import os
import sys
import shutil
import subprocess
import site
from pathlib import Path

# LMStudioのバックエンドパス
LMSTUDIO_BACKEND_PATH = r"C:\Users\pino1.4080PC\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-1.28.0"
LMSTUDIO_VENDOR_PATH = r"C:\Users\pino1.4080PC\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2"

# サイトパッケージパスを取得する安全な方法
def get_site_packages_path():
    """サイトパッケージのパスを取得"""
    try:
        # 方法1: site.getsitepackages() を使用
        site_packages = site.getsitepackages()
        if site_packages and len(site_packages) > 0:
            return site_packages[0]
    except Exception:
        pass
    
    try:
        # 方法2: site-packages ディレクトリを検索
        python_path = os.path.dirname(sys.executable)
        possible_paths = [
            os.path.join(python_path, 'Lib', 'site-packages'),
            os.path.join(python_path, '..', 'Lib', 'site-packages')
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                return path
    except Exception:
        pass
    
    # 方法3: 実行中のスクリプトのパスから推測
    try:
        current_file = os.path.abspath(__file__)
        venv_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        return os.path.join(venv_path, 'Lib', 'site-packages')
    except Exception:
        pass
    
    # デフォルトとして現在のパスを返す
    return os.getcwd()

# サイトパッケージパスを取得
SITE_PACKAGES_PATH = get_site_packages_path()
print(f"サイトパッケージパス: {SITE_PACKAGES_PATH}")

def check_paths():
    """パスの存在を確認"""
    paths_to_check = [LMSTUDIO_BACKEND_PATH, LMSTUDIO_VENDOR_PATH]
    missing_paths = []
    
    for path in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print(f"警告: 以下のパスが存在しません:")
        for path in missing_paths:
            print(f" - {path}")
        return False
    return True

def list_backend_files():
    """バックエンドファイルの一覧を表示"""
    if not os.path.exists(LMSTUDIO_BACKEND_PATH):
        print(f"バックエンドパスが存在しません: {LMSTUDIO_BACKEND_PATH}")
        return
    
    print(f"LMStudioバックエンドファイル一覧:")
    for item in os.listdir(LMSTUDIO_BACKEND_PATH):
        item_path = os.path.join(LMSTUDIO_BACKEND_PATH, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path) / (1024 * 1024)
            print(f" - {item} ({size:.2f} MB)")
        else:
            print(f" - {item}/ (ディレクトリ)")

def create_symbolic_link():
    """シンボリックリンクを作成"""
    # llama_cpp_pythonパッケージの場所を確認
    llama_cpp_path = os.path.join(SITE_PACKAGES_PATH, "llama_cpp")
    
    if not os.path.exists(llama_cpp_path):
        print("llama_cpp パッケージが見つかりません。先にllama-cpp-pythonをインストールしてください。")
        return False
    
    # バックアップを作成
    if os.path.exists(llama_cpp_path) and not os.path.islink(llama_cpp_path):
        backup_path = f"{llama_cpp_path}_backup"
        if not os.path.exists(backup_path):
            print(f"バックアップを作成中: {backup_path}")
            shutil.move(llama_cpp_path, backup_path)
        else:
            print(f"バックアップはすでに存在します: {backup_path}")
    
    # シンボリックリンクを作成
    try:
        if os.path.exists(llama_cpp_path):
            if os.path.islink(llama_cpp_path):
                os.unlink(llama_cpp_path)
            else:
                print(f"警告: {llama_cpp_path} はシンボリックリンクではありません")
                return False
        
        # Windowsではシンボリックリンク作成に管理者権限が必要
        try:
            os.symlink(LMSTUDIO_BACKEND_PATH, llama_cpp_path)
            print(f"シンボリックリンクを作成しました: {llama_cpp_path} -> {LMSTUDIO_BACKEND_PATH}")
            return True
        except OSError as e:
            print(f"シンボリックリンクの作成に失敗しました: {e}")
            print("管理者権限で実行してみてください。")
            return False
            
    except Exception as e:
        print(f"エラー: {e}")
        return False

def copy_dll_files():
    """必要なDLLファイルをコピー"""
    if not os.path.exists(LMSTUDIO_VENDOR_PATH):
        print(f"ベンダーパスが存在しません: {LMSTUDIO_VENDOR_PATH}")
        return False
    
    # DLLファイル一覧
    dll_files = [f for f in os.listdir(LMSTUDIO_VENDOR_PATH) if f.endswith('.dll')]
    
    if not dll_files:
        print(f"DLLファイルが見つかりません: {LMSTUDIO_VENDOR_PATH}")
        return False
    
    # Python実行ファイルのディレクトリにコピー
    python_dir = os.path.dirname(sys.executable)
    
    print(f"DLLファイルをコピー中: {LMSTUDIO_VENDOR_PATH} -> {python_dir}")
    copied_files = []
    
    for dll_file in dll_files:
        src = os.path.join(LMSTUDIO_VENDOR_PATH, dll_file)
        dst = os.path.join(python_dir, dll_file)
        
        try:
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied_files.append(dll_file)
            else:
                print(f"ファイルはすでに存在します: {dst}")
        except Exception as e:
            print(f"コピー中にエラー ({dll_file}): {e}")
    
    if copied_files:
        print(f"コピーしたDLLファイル: {', '.join(copied_files)}")
        return True
    else:
        print("DLLファイルはコピーされませんでした")
        return False

def set_environment_variables():
    """環境変数を設定"""
    # アプリケーションの設定ファイルに環境変数を追加
    env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    
    env_vars = {
        "USE_GPU": "true",
        "GPU_LAYERS": "-1"
    }
    
    if os.path.exists(env_file_path):
        # 既存のファイルを読み込んで環境変数を更新
        with open(env_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        updated_lines = []
        updated_vars = set()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                updated_lines.append(line)
                continue
                
            parts = line.split('=', 1)
            if len(parts) == 2:
                var_name = parts[0].strip()
                if var_name in env_vars:
                    updated_lines.append(f"{var_name}={env_vars[var_name]}")
                    updated_vars.add(var_name)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # 未追加の環境変数を追加
        for var_name, var_value in env_vars.items():
            if var_name not in updated_vars:
                updated_lines.append(f"{var_name}={var_value}")
        
        # 更新した内容を書き込み
        with open(env_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(updated_lines) + '\n')
        
        print(f"環境変数を更新しました: {env_file_path}")
    else:
        # 新規ファイルを作成
        with open(env_file_path, 'w', encoding='utf-8') as f:
            f.write("# Jarviee GPU設定\n")
            for var_name, var_value in env_vars.items():
                f.write(f"{var_name}={var_value}\n")
        
        print(f"環境変数ファイルを作成しました: {env_file_path}")
    
    # カレントプロセスの環境変数も設定
    for var_name, var_value in env_vars.items():
        os.environ[var_name] = var_value
    
    return True

def test_gpu_availability():
    """GPUの利用可能性をテスト"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "不明"
            print(f"PyTorch GPU状況:")
            print(f" - 利用可能: はい")
            print(f" - デバイス数: {device_count}")
            print(f" - デバイス名: {device_name}")
        else:
            print("PyTorch GPU状況: 利用不可")
        
        # llama-cpp-pythonの確認
        try:
            from llama_cpp import Llama
            # llama-cpp-pythonのバージョン確認
            try:
                import llama_cpp
                print(f"llama-cpp-python バージョン: {llama_cpp.__version__}")
            except:
                print("llama-cpp-pythonのバージョン情報を取得できません")
            
            # GPUサポートの確認
            has_gpu_support = hasattr(Llama, 'eval_gpu_supported')
            print(f"llama-cpp-python GPU状況:")
            print(f" - GPU対応ビルド: {'はい' if has_gpu_support else 'いいえ'}")
            
            # n_gpu_layersパラメータのサポートを確認
            import inspect
            params = inspect.signature(Llama.__init__).parameters
            has_gpu_param = 'n_gpu_layers' in params
            print(f" - n_gpu_layersパラメータ: {'サポート' if has_gpu_param else '未サポート'}")
            
        except ImportError:
            print("llama-cpp-pythonがインストールされていません")
        except Exception as e:
            print(f"llama-cpp-pythonテストエラー: {e}")
            
    except ImportError:
        print("PyTorchがインストールされていません")
    except Exception as e:
        print(f"GPUテストエラー: {e}")

def direct_install_gpu_version():
    """GPU対応版のllama-cpp-pythonを直接インストール"""
    print("\n--- GPU対応版llama-cpp-pythonのインストール ---")
    try:
        # 現在のバージョンをアンインストール
        uninstall_cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"]
        print("実行中: " + " ".join(uninstall_cmd))
        subprocess.run(uninstall_cmd, check=True)
        
        # GPU対応版をインストール
        install_cmd = [
            sys.executable, "-m", "pip", "install", "llama-cpp-python", 
            "--force-reinstall", "--no-cache-dir", 
            "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117"
        ]
        print("実行中: " + " ".join(install_cmd))
        subprocess.run(install_cmd, check=True)
        print("GPU対応版llama-cpp-pythonのインストールが完了しました。")
        return True
    except Exception as e:
        print(f"インストール中にエラーが発生しました: {e}")
        return False

def main():
    """メイン処理"""
    print("=== LMStudioバックエンド活用スクリプト ===")
    
    if not check_paths():
        # LMStudioのパスが見つからない場合は直接インストールを試行
        print("LMStudioのバックエンドが見つかりません。直接GPU対応版をインストールします。")
        direct_install_gpu_version()
    else:
        list_backend_files()
        
        print("\n--- 環境変数設定 ---")
        set_environment_variables()
        
        print("\n--- DLLファイルコピー ---")
        copy_dll_files()
        
        print("\n--- シンボリックリンク作成 ---")
        create_symbolic_link()
    
    print("\n--- GPU利用可能性テスト ---")
    test_gpu_availability()
    
    print("\n=== 完了 ===")
    print("Jarvieeを再起動して変更を適用してください。")

if __name__ == "__main__":
    main()
