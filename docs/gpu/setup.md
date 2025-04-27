# Jarviee GPU設定ガイド

このドキュメントでは、JarvieeシステムでGPUを使用するための設定方法と最適化の手順について説明します。

## 目次

1. [要件](#要件)
2. [セットアップ手順](#セットアップ手順)
3. [GPU診断](#gpu診断)
4. [設定オプション](#設定オプション)
5. [トラブルシューティング](#トラブルシューティング)

## 要件

Jarvieeでのモデル実行をGPUで高速化するには、以下が必要です：

- NVIDIA GPU（CUDA対応）
- CUDAツールキット（推奨：CUDA 11.7以上）
- 適切なドライバー（NVIDIAウェブサイトから最新版をインストール）
- GPU対応版のllama-cpp-python

## セットアップ手順

### 1. CUDAインストール

NVIDIAのWebサイトからCUDAツールキットをダウンロードしインストールします：
https://developer.nvidia.com/cuda-downloads

### 2. GPU対応版llama-cpp-pythonインストール

#### 事前ビルド版（推奨）

以下のコマンドで事前ビルドされたGPU対応版をインストールします：

```bash
pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
```

このコマンドはCUDA 11.7用のGPU対応バイナリをインストールします。他のCUDAバージョンの場合は`cu117`を適切なバージョン（例：`cu118`）に変更してください。

#### ソースからのビルド（上級者向け）

カスタム設定が必要な場合は、ソースからビルドできます：

```bash
# 必要なビルドツール
pip install build setuptools wheel pybind11

# CUDA環境変数設定
export LLAMA_CUBLAS=1
export CUDA_HOME=/usr/local/cuda-11.7  # インストールしたCUDAのパスに変更

# インストール
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir
```

### 3. Pytorchインストール

GPU対応版Pytorchをインストールします：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 4. Jarvieeの設定

`config/config.json`内の`gemma`プロバイダー設定で、以下のパラメータを確認します：

```json
"gemma": {
    "use_gpu": true,
    "n_gpu_layers": -1  // -1は全レイヤーをGPUで処理、数値を減らすとメモリ使用量も減少
}
```

## GPU診断

Jarvieeの起動時にGPU使用状況が自動的に診断されます。また、以下のコマンドで明示的に診断を実行できます：

```bash
python jarviee.py --check-gpu
```

このコマンドは以下を確認します：
- Pytorchのインストール状況
- CUDA利用可能性
- 利用可能なGPUデバイス
- llama-cpp-pythonのGPUサポート状況

## 設定オプション

`jarviee.py`起動時に以下のGPU関連オプションが使用できます：

- `--gpu` または `-g`: GPUを明示的に有効化
- `--cpu-only`: GPUを使用せずCPUのみで実行
- `--gpu-layers N`: GPUで実行するレイヤー数を指定（-1=全て、0=なし、n=n層）
- `--check-gpu`: GPU診断を実行して終了

例：
```bash
# すべてのレイヤーでGPUを使用
python jarviee.py --gpu

# 最初の24レイヤーのみGPUで実行（メモリ使用量削減）
python jarviee.py --gpu --gpu-layers 24

# CPU専用モード
python jarviee.py --cpu-only
```

## トラブルシューティング

### 一般的な問題

1. **メモリ不足エラー**
   - GPUレイヤー数を減らす: `--gpu-layers 24`のように指定
   - 小さなモデルを使用する（7BパラメータモデルなどQ4_0など低精度量子化）

2. **CUDA関連エラー**
   - NVIDIAドライバーが最新かを確認
   - `nvidia-smi`コマンドが正常に動作するかを確認
   - CUDAバージョンとドライバーが互換性を持つか確認

3. **llama-cpp-pythonエラー**
   - GPU対応版が正しくインストールされているか確認
   - `python -c "from llama_cpp import Llama; print(dir(Llama.__init__))"`を実行して`n_gpu_layers`が表示されるか確認

4. **遅いパフォーマンス**
   - `--check-gpu`で診断を実行し、適切にGPUが認識されているか確認
   - GPUモニタリングツール（nvidia-smi）でGPU使用率を確認
   - 他のGPUを使用するプロセスを終了して専用にする

### 特定のGPUモデル向け設定

#### NVIDIA RTX 4080/4090向け推奨設定
```bash
python jarviee.py --gpu --gpu-layers -1
```

#### NVIDIA RTX 3060/3070向け推奨設定（VRAM 8-10GB）
```bash
python jarviee.py --gpu --gpu-layers 32
```

#### 古いGPUまたは低VRAM（6GB以下）向け設定
```bash
python jarviee.py --gpu --gpu-layers 24
```

## 上級者向け：GPUパフォーマンス最適化

1. **モデル量子化の選択**
   - 大きなVRAMがある場合: F16精度で最高品質（例：*.gguf）
   - 中程度のVRAM: Q5_K_M（良好な品質と効率）
   - 制限されたVRAM: Q4_0（最小メモリ使用量）

2. **コンテキストウィンドウサイズの調整**
   ```bash
   # コンテキストウィンドウを小さくしてメモリ使用量を削減
   # 設定ファイルで context_window を調整
   ```

3. **バッチサイズとスレッド数の調整**
   - 設定ファイルで必要に応じて調整：
   ```json
   "gemma": {
       "use_gpu": true,
       "n_gpu_layers": -1,
       "n_batch": 512,  // バッチサイズ（大きいほど高速だがメモリ使用量増加）
       "threads": 8     // スレッド数（CPUコア数に合わせる）
   }
   ```
