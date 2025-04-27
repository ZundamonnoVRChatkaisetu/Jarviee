#!/usr/bin/env python
"""
設定ファイルの診断と表示
"""

import json
import sys
from pathlib import Path
import os

def find_config_path():
    """config.jsonのパスを見つける"""
    # スクリプトの場所からプロジェクトルートを特定
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # 一般的な場所を探す
    possible_paths = [
        project_root / "config" / "config.json",
        project_root / "config.json"
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def main():
    """メイン関数"""
    print("Jarviee 設定診断ユーティリティ")
    
    # config.jsonの位置を特定
    config_path = find_config_path()
    
    if not config_path:
        print("設定ファイルが見つかりませんでした。")
        return 1
    
    print(f"設定ファイル: {config_path}")
    print(f"ファイルサイズ: {os.path.getsize(config_path)} バイト")
    
    # エンコーディングを検出
    encodings = ['utf-8', 'utf-16', 'cp932', 'shift_jis', 'euc-jp', 'iso-2022-jp']
    detected_encoding = None
    
    for encoding in encodings:
        try:
            with open(config_path, 'r', encoding=encoding) as f:
                f.read()
                detected_encoding = encoding
                break
        except UnicodeDecodeError:
            continue
    
    if detected_encoding:
        print(f"検出されたエンコーディング: {detected_encoding}")
    else:
        print("エンコーディングを検出できませんでした。")
        return 1
    
    # 設定ファイルを読み込む
    try:
        with open(config_path, 'r', encoding=detected_encoding) as f:
            config = json.load(f)
    except Exception as e:
        print(f"設定ファイルの読み込みに失敗しました: {e}")
        return 1
    
    # LLM設定を表示
    llm_config = config.get('llm', {})
    default_provider = llm_config.get('default_provider', 'none')
    print(f"デフォルトプロバイダー: {default_provider}")
    
    # プロバイダー一覧
    providers = llm_config.get('providers', {})
    print("\nプロバイダー一覧:")
    for provider_name, provider_config in providers.items():
        print(f"- {provider_name}")
    
    # Gemma設定を詳細表示
    if 'gemma' in providers:
        gemma_config = providers['gemma']
        print("\nGemma設定:")
        print(f"  モデルパス: {gemma_config.get('path', 'なし')}")
        
        models = gemma_config.get('models', {})
        print(f"  デフォルトモデル: {models.get('default', 'なし')}")
        
        gpu_enabled = gemma_config.get('use_gpu', False)
        gpu_layers = gemma_config.get('n_gpu_layers', 0)
        
        print(f"  GPU有効: {gpu_enabled}")
        print(f"  GPUレイヤー数: {gpu_layers}")
        
        # GPUが無効の場合
        if not gpu_enabled or gpu_layers == 0:
            print("\n[警告] GPU設定が無効になっています。")
            print("GPU設定を有効にしますか？ [y/n]")
            choice = input().lower()
            
            if choice == 'y':
                # 設定を更新
                gemma_config['use_gpu'] = True
                gemma_config['n_gpu_layers'] = -1  # すべてのレイヤーをGPUで処理
                providers['gemma'] = gemma_config
                llm_config['providers'] = providers
                config['llm'] = llm_config
                
                # 設定を保存
                try:
                    with open(config_path, 'w', encoding=detected_encoding) as f:
                        json.dump(config, f, indent=4, ensure_ascii=False)
                    print("GPU設定を有効化しました。")
                except Exception as e:
                    print(f"設定ファイルの保存に失敗しました: {e}")
                    return 1
    else:
        print("\n[警告] Gemmaプロバイダーが見つかりません。")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
