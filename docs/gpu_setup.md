# Jarviee GPU設定ガイド

## 目次
1. [概要](#概要)
2. [要件](#要件)
3. [GPU対応版llama-cpp-pythonのインストール](#gpu対応版llama-cpp-pythonのインストール)
4. [LMStudioのバックエンドを活用する方法](#lmstudioのバックエンドを活用する方法)
5. [手動設定](#手動設定)
6. [トラブルシューティング](#トラブルシューティング)

## 概要

JarvieeはローカルLLMモデル(Gemma)でGPUアクセラレーションを活用できます。GPUを活用することで、モデルの推論速度が大幅に向上し、より高速な応答が可能になります。

このドキュメントでは、JarvieeでGPUアクセラレーションを有効にする方法について説明します。

## 要件

- NVIDIA GPU（CUDA対応）
- CUDA 11.7以降
- CUDAデバイスドライバー最新版
- Python 3.10以降

## GPU対応版llama-cpp-pythonのインストール

標準のllama-cpp-pythonパッケージはCPUのみの対応となっています。GPU対応版をインストールするには以下のコマンドを使用します：

```bash
pip uninstall -y llama-cpp-python
pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
```

### CUDA 12.x を使用する場合

CUDA 12.0以降を使用している場合は、以下のコマンドを使用します：

```bash
pip uninstall -y llama-cpp-python
pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu12
```

### 自分のGPUに合ったバージョンを調べる

使用しているGPUに最適なバージョンを調べるには、診断スクリプトを実行してください：

```bash
python scripts/gpu_diagnostics.py
```

## LMStudioのバックエンドを活用する方法

LMStudioを使用している場合、そのバックエンドを活用してGPU対応を有効化できます：

### 自動設定スクリプトを使用する

```bash
# 管理者権限で実行してください
python scripts/enable_lmstudio_backend.py
```

このスクリプトは以下の処理を行います：

1. LMStudioバックエンドファイルの確認
2. DLLファイルの必要な場所へのコピー
3. シンボリックリンクの作成（必要な場合）
4. 環境変数の設定
5. GPU動作確認テスト

### バックエンドパスの設定

LMStudioのバックエンドパスは以下の場所にあります：

- CUDA 12のバックエンド: `C:\Users\[ユーザー名]\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-1.28.0\`
- ベンダーDLL: `C:\Users\[ユーザー名]\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2\`

## 手動設定

1. **config/config.jsonの設定**

```json
"gemma": {
    "path": "./models/gemma-3-12B-it-QAT-Q4_0.gguf",
    "use_gpu": true,
    "n_gpu_layers": -1,
    "verbose": true
}
```

2. **.envファイルの設定**

```
USE_GPU=true
GPU_LAYERS=-1
```

3. **適切なモデルの確認**

使用するモデルがGPU対応であることを確認してください。GGUF形式の量子化モデルがおすすめです。

## トラブルシューティング

### GPU診断の実行

問題が発生した場合は、診断スクリプトを実行して詳細な情報を確認してください：

```bash
python scripts/gpu_diagnostics.py
```

### よくある問題と解決策

1. **「CUDAエラー: no CUDA-capable device is detected」というエラーが表示される**
   - NVIDIAドライバーが正しくインストールされているか確認します
   - `nvidia-smi` コマンドが動作するか確認します

2. **「cannot find llama.dll」というエラーが表示される**
   - DLLファイルが正しい場所にコピーされているか確認します
   - `scripts/enable_lmstudio_backend.py` を実行してDLLを適切な場所にコピーします

3. **モデルはロードされるがGPUが使用されない**
   - `config.json` で `use_gpu` が `true` に設定されているか確認します
   - `n_gpu_layers` が `-1` または適切な値に設定されているか確認します

4. **メモリエラーが発生する**
   - モデルのサイズを小さくするか、量子化されたモデルを使用します
   - `n_gpu_layers` の値を小さくして一部のレイヤーだけをGPUで実行します

### GPU使用状況の確認方法

NVIDIA GPUの使用状況を確認するには：

```bash
# タスクマネージャーの「パフォーマンス」タブでGPU使用率を確認
# または
nvidia-smi -l 1  # 1秒ごとに更新
```

## 参考情報

- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [GGUF Models on Hugging Face](https://huggingface.co/models?sort=trending&search=gguf)
- [CUDA ToolKit Download](https://developer.nvidia.com/cuda-downloads)
