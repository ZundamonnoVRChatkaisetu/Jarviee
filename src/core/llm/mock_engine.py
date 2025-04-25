"""
モックLLMエンジン

このモジュールは、開発とテストを容易にするためのモックLLMエンジンを提供します。
実際のLLM APIへの呼び出しを行わず、事前定義された応答を返します。
"""

import logging
import random
import time
from typing import Dict, List, Optional, Union, Any


class MockLLMEngine:
    """
    モックLLMエンジン
    
    実際のLLM APIを呼び出さずに、事前定義されたパターンに基づいて応答を生成します。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        モックLLMエンジンを初期化
        
        Args:
            config: エンジン設定（オプション）
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 個性パラメータ（0.0〜1.0）
        self.formality = self.config.get("formality", 0.3)  # 低いほど砕けた表現
        self.verbosity = self.config.get("verbosity", 0.5)  # 高いほど詳細な応答
        self.humor = self.config.get("humor", 0.7)  # 高いほどユーモアを含む
        self.technical = self.config.get("technical", 0.8)  # 高いほど専門的
        
        # 応答テンプレート
        self._load_templates()
    
    def _load_templates(self):
        """応答テンプレートを読み込む"""
        # 基本的な会話応答テンプレート
        self.greeting_templates = [
            "いらっしゃいませ、てゅん。お役に立てることはありますか？",
            "おかえりなさい、てゅん。今日はどのようなご用件でしょうか？",
            "システム起動完了しました。なにかお手伝いできることはありますか、てゅん？",
            "Jarvieeシステム準備完了です。本日はどのようなサポートが必要ですか？"
        ]
        
        self.question_templates = [
            "申し訳ありませんが、{query}については完全な情報を持ち合わせていません。",
            "{query}についてですが、現在のデータベースでは十分な情報がありません。",
            "興味深い質問です。{query}に関しては、限定的な情報しかありませんが、お伝えします。",
            "{query}について検索中...限られた情報ではありますが、共有できる内容があります。"
        ]
        
        self.programming_templates = [
            "プログラミングタスクを分析しています... {language}のコードを生成します。",
            "{language}による実装方法を考えています。最適なアプローチを提案します。",
            "このプログラミング課題に取り組みましょう。{language}で最適な解決策を提案します。",
            "{language}のコードを作成します。効率性と可読性を重視した実装を目指します。"
        ]
        
        self.error_templates = [
            "申し訳ありません、処理中にエラーが発生しました。詳細: {error}",
            "エラーが検出されました: {error}。別のアプローチを試してみましょう。",
            "処理を完了できませんでした。エラー: {error}",
            "問題が発生しました: {error}。対応策を検討します。"
        ]
        
        self.jarvis_style_templates = [
            "分析完了しました、てゅん。{result}",
            "処理が完了しました。{result} 他に必要な情報はありますか？",
            "計算結果です：{result}",
            "てゅん、分析が終わりました。{result} 何か質問はありますか？"
        ]
    
    def _add_personality(self, response: str) -> str:
        """
        応答に個性を追加
        
        Args:
            response: 基本応答
            
        Returns:
            個性を追加した応答
        """
        # フォーマル度に応じた言葉遣いの調整
        if self.formality < 0.3:
            response = response.replace("です。", "だよ。")
            response = response.replace("ました", "たよ")
            response = response.replace("ください", "くれ")
        
        # 冗長さの調整（高いほど詳細に）
        if self.verbosity > 0.7 and len(response) < 100:
            details = [
                "詳細な分析結果も用意できます。",
                "必要であれば、さらに詳しい情報も提供可能です。",
                "この結果についてより詳しい説明が必要でしょうか？",
                "追加情報が必要な場合は、お知らせください。"
            ]
            response += " " + random.choice(details)
        
        # ユーモアの追加
        if self.humor > 0.7 and random.random() < 0.3:
            jokes = [
                " 私のAIハートはそう言っています。",
                " 少なくとも私のアルゴリズムはそう考えています。",
                " 量子コンピューターでも同じ答えが出るはず...たぶん。",
                " 冗談抜きで。"
            ]
            response += random.choice(jokes)
        
        # 専門性の調整
        if self.technical > 0.7 and random.random() < 0.3:
            tech_terms = [
                " アルゴリズムの複雑度はO(log n)程度です。",
                " ベイジアン推論に基づく結論です。",
                " 統計的有意性は95%を超えています。",
                " 深層学習モデルの信頼度スコアは0.92です。"
            ]
            response += random.choice(tech_terms)
        
        return response
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成
        
        Args:
            prompt: 入力プロンプト
            **kwargs: 追加パラメータ
            
        Returns:
            生成されたテキスト
        """
        self.logger.debug(f"モックLLMで生成: {prompt[:50]}...")
        
        # 短い遅延を追加して処理時間をシミュレート
        time.sleep(0.5 + random.random() * 1.0)
        
        # プロンプトのタイプを判断
        if "こんにちは" in prompt.lower() or "はじめまして" in prompt.lower() or "調子" in prompt.lower():
            template = random.choice(self.greeting_templates)
            response = template
        elif "?" in prompt or "？" in prompt or "何" in prompt or "教えて" in prompt:
            query = prompt.strip("?？").strip()
            template = random.choice(self.question_templates)
            response = template.format(query=query)
        elif "コード" in prompt or "プログラム" in prompt or "実装" in prompt:
            language = "Python"  # デフォルト言語
            for lang in ["Python", "JavaScript", "Java", "C++", "Ruby", "Go"]:
                if lang.lower() in prompt.lower():
                    language = lang
                    break
            template = random.choice(self.programming_templates)
            response = template.format(language=language)
        else:
            # 一般的な応答
            result = f"「{prompt[:20]}...」についての分析結果"
            template = random.choice(self.jarvis_style_templates)
            response = template.format(result=result)
        
        # 個性を追加
        response = self._add_personality(response)
        
        return response
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成（非同期版）
        
        Args:
            prompt: 入力プロンプト
            **kwargs: 追加パラメータ
            
        Returns:
            生成されたテキスト
        """
        return self.generate(prompt, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        チャット形式での生成
        
        Args:
            messages: 会話履歴
            **kwargs: 追加パラメータ
            
        Returns:
            応答メッセージを含む辞書
        """
        self.logger.debug(f"モックLLMでチャット: {len(messages)}メッセージ")
        
        # 短い遅延を追加して処理時間をシミュレート
        time.sleep(0.5 + random.random() * 1.5)
        
        # 最後のメッセージを取得
        if not messages:
            return {"content": "申し訳ありません、会話コンテキストが見つかりません。"}
        
        last_message = messages[-1]
        prompt = last_message.get("content", "")
        
        # 応答を生成
        response = self.generate(prompt, **kwargs)
        
        return {
            "content": response,
            "role": "assistant",
            "metadata": {
                "timestamp": time.time(),
                "model": "mock-jarviee-model",
                "usage": {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(response) // 4,
                    "total_tokens": (len(prompt) + len(response)) // 4
                }
            }
        }
    
    async def chat_async(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        チャット形式での生成（非同期版）
        
        Args:
            messages: 会話履歴
            **kwargs: 追加パラメータ
            
        Returns:
            応答メッセージを含む辞書
        """
        return self.chat(messages, **kwargs)
    
    def embed(self, text: str) -> List[float]:
        """
        テキストの埋め込みベクトルを生成
        
        Args:
            text: 入力テキスト
            
        Returns:
            埋め込みベクトル（モックのためランダム値）
        """
        # 一貫性のためにテキストから決定論的にシードを設定
        seed = sum(ord(c) for c in text)
        random.seed(seed)
        
        # 128次元のモック埋め込みベクトルを生成
        embedding = [random.uniform(-1, 1) for _ in range(128)]
        
        # ベクトルを正規化
        magnitude = sum(x * x for x in embedding) ** 0.5
        normalized = [x / magnitude for x in embedding]
        
        return normalized
    
    async def embed_async(self, text: str) -> List[float]:
        """
        テキストの埋め込みベクトルを生成（非同期版）
        
        Args:
            text: 入力テキスト
            
        Returns:
            埋め込みベクトル
        """
        return self.embed(text)
