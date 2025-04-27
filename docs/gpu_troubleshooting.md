# GPU 設定トラブルシューティングガイド

## 状況の確認

llama-serverが起動しているにもかかわらずGPUが使用されていない場合、以下の手順でトラブルシューティングを行います。

## 1. CUDAのインストール状況を確認

```bash
# NVIDIA-SMIでGPUが認識されているか確認
nvidia-smi
```

正しい出力例:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.182.1             Driver Version: 535.182.01                 |
|-------------------------------+----------------------+----------------------+
| GPU  Name             Perf   |     Memory-Usage     |      GPU-Util      |
|      TCC/WDDM          P-States     |    PCI-E     |    Temperature     |
|                                     |                |                    |
|===============================+======================+======================|
|   0  GeForce RTX 4080    On  |    1024MiB / 16384MiB |      0%      |
...
```

## 2. llama.cppのビルド確認

llama.cppが正しくCUDAサポート付きでビルドされているか確認します。

```bash
# CMakeでビルドする場合の正しいコマンド
mkdir -p build
cd build
cmake .. -DLLAMA_CUBLAS=ON
cmake --build . --config Release
```

CMake出力でCUDAが有効になっていることを確認:
```
-- Found CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3 (found version "12.3")
...
-- LLAMA_CUBLAS: ON
```

## 3. 正しいコマンドライン引数

llama-serverの起動時に正しいGPUオプションを使用していることを確認:

```bash
# ショートカット形式
llama-server.exe -m MODEL_PATH -ngl -1

# 完全名形式（問題がある場合はこちらを試す）
llama-server.exe -m MODEL_PATH --n-gpu-layers -1
```

## 4. 使用可能なオプションの確認

ビルド済みの`llama-server.exe`で使用可能なオプションを確認:

```bash
llama-server.exe --help
```

GPUサポートを示すオプションを探します:
- `--n-gpu-layers` or `-ngl`
- `--gpu-layers`
- その他のGPU関連パラメータ

## 5. ビルド設定の確認

バイナリがCUDAサポート付きでビルドされているか確認するには:

```bash
# prints GPU configuration
llama-server.exe --print-gpu-config
```

## 6. GPUメモリ使用量の確認

モデルとGPUメモリの互換性を確認:

```bash
# Check GPU memory usage
nvidia-smi -l 1
```

## 7. 正しい対応法

以下の手順に従ってください:

1. llama.cppを再ビルドする (CUDAサポート付き)
   ```bash
   cmake .. -DLLAMA_CUBLAS=ON
   cmake --build . --config Release
   ```

2. 完全な引数形式を使用する
   ```bash
   start "Llama Server" "%SERVER_PATH%" -m "%MODEL_PATH%" --host 127.0.0.1 --port %PORT% --n-gpu-layers -1 --n-ctx 8192
   ```

3. CUDA環境変数を明示的に設定
   ```bash
   set CUDA_VISIBLE_DEVICES=0
   ```

4. 起動前にGPUの状態を確認
   ```bash
   nvidia-smi -l 1
   ```

## サーバー起動時のGPU使用確認方法

サーバーの起動ログに、以下のようなメッセージが含まれていればGPUが使用されています:

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX XXXX, compute capability X.X, VMM: yes
```

## 一般的な問題と解決策

1. **「CUDA driver version is insufficient for CUDA runtime version」エラー**
   - CUDAドライバーを更新する

2. **「no CUDA-capable device is detected」エラー**
   - NVIDIAドライバが正しくインストールされているか確認
   - GPUが認識されているか確認

3. **GPUメモリ不足エラー**
   - 量子化されたモデルを使用する
   - より小さいコンテキストサイズ (`--n-ctx`) を設定する
   - GPU層の数を減らす (`--n-gpu-layers 40` など)

4. **GPUが検出されているのに使用されない**
   - 明示的に `--n-gpu-layers` オプションを使用する
   - モデルが大きすぎないか確認する

5. **CUDAを使用するバイナリが存在しない**
   - `-DLLAMA_CUBLAS=ON` フラグでllama.cppを再ビルドする
