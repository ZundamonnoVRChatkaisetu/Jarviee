# llama-server GPU設定ガイド

## 概要

このガイドでは、ビルド済みの`llama-server.exe`を使用してGPUアクセラレーションを有効にする方法について説明します。この方法では、`llama.cpp`のビルド済みバイナリを使用してHTTPサーバーを起動し、JarvieeシステムからAPIを通じて通信することで、GPUの処理能力を最大限に活用できます。

## 前提条件

- llama.cppリポジトリをクローン・ビルド済み
- build/bin/release配下にllama-server.exeが生成されている
- NVIDIA GPU搭載のWindows環境
- CUDAがインストール済み

## セットアップ手順

### 1. サーバー起動スクリプトの設定

`scripts/start_llama_server.bat`を以下のように編集します：

1. `LLAMA_BUILD_DIR`変数に、実際のllama.cppビルドディレクトリのパスを設定
2. `MODEL_PATH`が正しいモデルファイルを指しているか確認

```bash
# 例:
set LLAMA_BUILD_DIR=C:\path\to\llama.cpp\build\bin\release
```

### 2. サーバーの起動

コマンドプロンプトを管理者権限で開き、以下のコマンドを実行します：

```bash
cd C:\1_Sagyo\BenriToul\AI\Jarviee
scripts\start_llama_server.bat
```

サーバーが正常に起動すると、以下のような出力が表示されます：

```
Llama Server - GPUモード
モデル: C:\1_Sagyo\BenriToul\AI\Jarviee\models\gemma-3-12B-it-QAT-Q4_0.gguf
ホスト: 127.0.0.1
ポート: 8080
GPU設定: 
 - 全GPU層使用
 - メインGPU: 0
サーバー起動中...
```

### 3. Jarvieeシステムの起動

サーバーが起動した状態で、**別の**コマンドプロンプトを開き、Jarvieeを起動します：

```bash
cd C:\1_Sagyo\BenriToul\AI\Jarviee
python jarviee_cli.py
```

Jarvieeが起動すると、`llama_server`プロバイダーが自動的に検出され、GPUアクセラレーションが有効になります。

## システム構成

```
┌────────────────────┐     ┌───────────────────┐
│  llama-server.exe  │ <── │ Jarviee System    │
│  (GPU対応)         │ ──> │ (LlamaServerProvider) │
└────────────────────┘     └───────────────────┘
     │                             │
     │                             │
┌────────────────────┐     ┌───────────────────┐
│  NVIDIA GPU        │     │  ユーザー         │
└────────────────────┘     └───────────────────┘
```

## GPUパフォーマンスの確認

GPU使用状況を確認するには、以下のコマンドを実行します：

```bash
# Jarviee CLI内で
check_gpu
```

または、NVIDIA-SMIを使用して直接確認します：

```bash
nvidia-smi -l 1
```

## トラブルシューティング

### サーバーが起動しない場合

1. CUDAのインストール状況を確認
2. モデルパスが正しいか確認
3. llama-server.exeのパスが正しいか確認
4. 管理者権限でコマンドプロンプトを実行しているか確認

### Jarvieeがサーバーに接続できない場合

1. サーバーが起動していることを確認
2. ポート番号が正しいか確認
3. ファイアウォールが通信をブロックしていないか確認
4. `.env`ファイルの`LLAMA_SERVER_URL`が正しいか確認

### GPUが使用されていない場合

1. llama-serverの起動パラメータの`-ngl`が`-1`または適切な値に設定されているか確認
2. CUDAのバージョンがGPUと互換性があるか確認
3. モデルサイズがGPUメモリに収まるか確認

## 上級設定

### 複数GPU対応

複数のGPUがある場合、`TENSOR_SPLIT`パラメータでGPU間の負荷分散を設定できます：

```bash
# 例: 2つのGPUに均等に分散
set TENSOR_SPLIT=0.5,0.5
```

### サーバーのパフォーマンス調整

`start_llama_server.bat`にオプションを追加することで、パフォーマンスを調整できます：

```bash
# メモリ使用量の調整
--batch-size 512

# スレッド数の調整
--threads 8

# 量子化の調整
--parallel 4
```

## まとめ

llama-serverとGPUを使用することで、Jarvieeシステムの処理速度が大幅に向上します。特に大規模モデルの高速な推論に効果的です。
