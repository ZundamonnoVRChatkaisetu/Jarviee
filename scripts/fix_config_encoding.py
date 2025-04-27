#!/usr/bin/env python
"""
設定ファイルの文字コードを修正するスクリプト
"""

import json
import sys
from pathlib import Path

def main():
    """メイン関数"""
    print("Jarviee設定ファイル修正ユーティリティ")
    
    # config.jsonの位置を特定
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    config_path = project_root / "config" / "config.json"
    
    if not config_path.exists():
        print(f"設定ファイルが見つかりません: {config_path}")
        return 1
    
    # ファイルをUTF-8で読み込み試行
    try:
        # まずutf-8で読み込みを試みる
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            print("設定ファイルはUTF-8で正常に読み込めました。")
    except UnicodeDecodeError:
        # 失敗した場合、CP932で読み込みを試みる
        try:
            with open(config_path, 'r', encoding='cp932') as f:
                config_text = f.read()
                
            # UTF-8で書き直す
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_text)
                
            print("設定ファイルをCP932からUTF-8に変換しました。")
            
            # 変換後のファイルを読み込み
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"設定ファイルの変換に失敗しました: {e}")
            return 1
    except Exception as e:
        print(f"設定ファイルの読み込みに失敗しました: {e}")
        return 1
    
    # GPU設定の確認と修正
    llm_config = config.get('llm', {})
    providers = llm_config.get('providers', {})
    gemma_config = providers.get('gemma', {})
    
    gpu_enabled = gemma_config.get('use_gpu', False)
    gpu_layers = gemma_config.get('n_gpu_layers', 0)
    
    print(f"現在のGPU設定: use_gpu={gpu_enabled}, n_gpu_layers={gpu_layers}")
    
    # 設定を更新
    gemma_config['use_gpu'] = True
    gemma_config['n_gpu_layers'] = -1  # すべてのレイヤーをGPUで処理
    providers['gemma'] = gemma_config
    llm_config['providers'] = providers
    config['llm'] = llm_config
    
    # 設定を保存
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("GPU設定を有効化しました。")
    except Exception as e:
        print(f"設定ファイルの保存に失敗しました: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
