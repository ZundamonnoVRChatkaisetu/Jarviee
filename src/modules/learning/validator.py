"""
知識検証フレームワーク

収集された情報の信頼性を検証し、知識ベースに追加可能な
高品質なデータとして処理するためのシステム。
複数の検証戦略を組み合わせて、情報の正確性、一貫性、関連性を確保します。
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from abc import ABC, abstractmethod
import re
import hashlib
import os
from datetime import datetime

from src.core.knowledge.knowledge_base import KnowledgeBase, KnowledgeEntity
from src.core.llm.engine import LLMEngine


class ValidationStrategy(ABC):
    """検証戦略の抽象基底クラス"""
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        検証戦略を初期化
        
        Args:
            name: 戦略名
            weight: 検証結果の重み付け（0.0～1.0）
        """
        self.name = name
        self.weight = weight
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        データを検証
        
        Args:
            data: 検証対象データ
            context: 検証コンテキスト
            
        Returns:
            (検証通過フラグ, 信頼性スコア, 追加情報)
        """
        pass


class SourceReliabilityStrategy(ValidationStrategy):
    """情報源の信頼性に基づく検証戦略"""
    
    def __init__(self, weight: float = 1.0, config: Dict[str, Any] = None):
        """
        ソース信頼性検証戦略を初期化
        
        Args:
            weight: 検証結果の重み付け
            config: 設定情報
        """
        super().__init__("source_reliability", weight)
        self.config = config or {}
        
        # 信頼性スコアの定義
        self.source_scores = self.config.get("source_scores", {
            # 最も信頼できるソース
            "wikipedia.org": 0.9,
            "arxiv.org": 0.9,
            "github.com": 0.85,
            "stackoverflow.com": 0.8,
            "developer.mozilla.org": 0.9,
            "docs.python.org": 0.95,
            "python.org": 0.9,
            "npmjs.com": 0.85,
            
            # 中程度の信頼性
            "medium.com": 0.7,
            "dev.to": 0.75,
            "towardsdatascience.com": 0.75,
            "hackernoon.com": 0.7,
            
            # 一般的なソース
            "blogspot.com": 0.6,
            "wordpress.com": 0.6,
            "example.com": 0.5
        })
        
        # ドメインカテゴリごとの信頼性
        self.domain_categories = self.config.get("domain_categories", {
            "academic": 0.9,  # .edu, .ac.uk などの学術機関
            "government": 0.85,  # .gov など政府機関
            "organization": 0.8,  # .org 非営利団体
            "commercial": 0.7,  # .com 商用サイト
            "network": 0.7,  # .net ネットワーク
            "info": 0.6,  # .info 情報サイト
            "country": 0.7   # 国別ドメイン (.jp, .uk など)
        })
    
    def _get_domain_score(self, url: str) -> float:
        """
        URLからドメインの信頼性スコアを算出
        
        Args:
            url: 対象URL
            
        Returns:
            信頼性スコア（0.0～1.0）
        """
        try:
            # URLからドメイン部分を抽出
            domain = url.split("//")[-1].split("/")[0]
            
            # 完全なドメイン一致
            if domain in self.source_scores:
                return self.source_scores[domain]
            
            # 部分一致
            for known_domain, score in self.source_scores.items():
                if domain.endswith(f".{known_domain}") or domain == known_domain:
                    return score
            
            # ドメインカテゴリによる判定
            domain_extension = domain.split(".")[-1]
            
            if domain_extension in ["edu", "ac"]:
                return self.domain_categories.get("academic", 0.8)
            elif domain_extension == "gov":
                return self.domain_categories.get("government", 0.85)
            elif domain_extension == "org":
                return self.domain_categories.get("organization", 0.8)
            elif domain_extension == "com":
                return self.domain_categories.get("commercial", 0.7)
            elif domain_extension == "net":
                return self.domain_categories.get("network", 0.7)
            elif domain_extension == "info":
                return self.domain_categories.get("info", 0.6)
            elif len(domain_extension) == 2:  # 国別ドメイン
                return self.domain_categories.get("country", 0.7)
            
            # デフォルト値
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"Error parsing domain from URL {url}: {e}")
            return 0.4  # エラー時は低めのスコア
    
    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        ソースの信頼性に基づく検証
        
        Args:
            data: 検証対象データ
            context: 検証コンテキスト
            
        Returns:
            (検証通過フラグ, 信頼性スコア, 追加情報)
        """
        url = data.get("url")
        source = data.get("source")
        
        if not url and not source:
            self.logger.warning("No URL or source information found in data")
            return False, 0.3, {"reason": "No source information"}
        
        # URLが存在する場合はURLからスコアを算出
        if url:
            score = self._get_domain_score(url)
        else:
            # URLがない場合はソース名から推定
            score = 0.5  # デフォルト値
            
            for known_domain, domain_score in self.source_scores.items():
                if known_domain in source:
                    score = domain_score
                    break
        
        # 閾値の設定（デフォルトは0.4）
        threshold = self.config.get("threshold", 0.4)
        
        # 検証結果
        passed = score >= threshold
        details = {
            "score": score,
            "threshold": threshold,
            "source": source or "unknown",
            "url": url or "not provided"
        }
        
        if not passed:
            details["reason"] = f"Source reliability score ({score}) below threshold ({threshold})"
            
        return passed, score, details


class ContentConsistencyStrategy(ValidationStrategy):
    """内容の一貫性に基づく検証戦略"""
    
    def __init__(self, knowledge_base: KnowledgeBase, weight: float = 0.8, config: Dict[str, Any] = None):
        """
        内容一貫性検証戦略を初期化
        
        Args:
            knowledge_base: 知識ベース
            weight: 検証結果の重み付け
            config: 設定情報
        """
        super().__init__("content_consistency", weight)
        self.knowledge_base = knowledge_base
        self.config = config or {}
    
    def _find_related_entities(self, topic: str, content: str) -> List[KnowledgeEntity]:
        """
        関連するエンティティを検索
        
        Args:
            topic: トピック
            content: コンテンツ
            
        Returns:
            関連エンティティのリスト
        """
        # シンプルな実装：トピックに関連するエンティティを検索
        related = []
        
        for entity_id, entity in self.knowledge_base._entities.items():
            # トピックに関連するか
            if topic.lower() in entity.entity_type.lower() or topic.lower() in entity.content.lower():
                related.append(entity)
                
            # メタデータに関連するか
            elif "topic" in entity.metadata and topic.lower() in entity.metadata["topic"].lower():
                related.append(entity)
                
            # カテゴリに関連するか
            elif "category" in entity.metadata and topic.lower() in entity.metadata["category"].lower():
                related.append(entity)
        
        return related[:10]  # 最大10個まで
    
    def _check_consistency(self, content: str, related_entities: List[KnowledgeEntity]) -> Tuple[bool, float, List[str]]:
        """
        コンテンツと関連エンティティの一貫性を確認
        
        Args:
            content: 検証対象コンテンツ
            related_entities: 関連エンティティ
            
        Returns:
            (一貫性フラグ, 一貫性スコア, 矛盾するエンティティID)
        """
        if not related_entities:
            return True, 0.8, []  # 関連エンティティがなければ一貫しているとみなす
        
        # 単純な実装：キーワードと基本的な事実の一致度を確認
        content_lower = content.lower()
        
        # 矛盾を検出
        conflicts = []
        keywords_count = 0
        matched_keywords = 0
        
        for entity in related_entities:
            entity_content = entity.content.lower()
            
            # 主要キーワードを抽出（簡易実装）
            keywords = re.findall(r'\b[a-z][a-z]{2,}\b', entity_content)
            keywords = [k for k in keywords if k not in ['the', 'and', 'that', 'this', 'with', 'from', 'have', 'for']]
            
            # キーワードの一致率を確認
            for keyword in keywords:
                if len(keyword) > 3:  # 短すぎるキーワードは無視
                    keywords_count += 1
                    if keyword in content_lower:
                        matched_keywords += 1
            
            # 数値の矛盾をチェック（簡易実装）
            numbers_entity = re.findall(r'\b\d+\b', entity_content)
            numbers_content = re.findall(r'\b\d+\b', content_lower)
            
            # 明確な矛盾があるかチェック
            # 実際の実装ではより高度な矛盾検出が必要
            if len(numbers_entity) > 0 and len(numbers_content) > 0:
                for num_entity in numbers_entity:
                    for num_content in numbers_content:
                        if num_entity == num_content:
                            matched_keywords += 1
            
            # 重大な矛盾があるとみなす条件（非常に簡易的）
            if keywords_count > 10 and matched_keywords / keywords_count < 0.3:
                conflicts.append(entity.entity_id)
        
        # 一貫性スコアを計算
        consistency_score = 0.5  # デフォルト値
        
        if keywords_count > 0:
            match_ratio = matched_keywords / keywords_count
            consistency_score = min(0.9, max(0.3, match_ratio))
        
        # 一貫性判定
        consistent = len(conflicts) == 0 and consistency_score >= 0.5
        
        return consistent, consistency_score, conflicts
    
    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        内容の一貫性に基づく検証
        
        Args:
            data: 検証対象データ
            context: 検証コンテキスト
            
        Returns:
            (検証通過フラグ, 信頼性スコア, 追加情報)
        """
        content = data.get("content", "")
        topic = data.get("topic", "")
        
        if not content:
            self.logger.warning("No content found in data")
            return False, 0.0, {"reason": "No content to validate"}
        
        if not topic and context:
            topic = context.get("topic", "")
        
        # 関連エンティティの検索
        related_entities = self._find_related_entities(topic, content)
        
        # 一貫性チェック
        consistent, consistency_score, conflicts = self._check_consistency(content, related_entities)
        
        # 検証結果
        details = {
            "score": consistency_score,
            "related_entities_count": len(related_entities),
            "conflicts": conflicts
        }
        
        if not consistent:
            details["reason"] = f"Content inconsistent with existing knowledge"
            if conflicts:
                details["reason"] += f" (conflicts with {len(conflicts)} entities)"
                
        return consistent, consistency_score, details


class LLMValidationStrategy(ValidationStrategy):
    """LLMを使用した検証戦略"""
    
    def __init__(self, llm_engine: LLMEngine, weight: float = 1.2, config: Dict[str, Any] = None):
        """
        LLM検証戦略を初期化
        
        Args:
            llm_engine: LLMエンジン
            weight: 検証結果の重み付け
            config: 設定情報
        """
        super().__init__("llm_validation", weight)
        self.llm_engine = llm_engine
        self.config = config or {}
        
        # 検証プロンプトテンプレート
        self.validation_template = self.config.get("validation_template", """
あなたは知識検証アシスタントとして、提供されたコンテンツの信頼性、正確性、関連性を評価します。
以下の内容を項目ごとに詳細に分析し、0.0～1.0のスコアを付けてください。

トピック: {topic}
コンテンツ:
{content}

追加コンテキスト:
{context}

評価項目:
1. 正確性: 事実関係の正確さ（0.0=完全に不正確、1.0=完全に正確）
2. 関連性: トピックとの関連度（0.0=無関係、1.0=非常に関連性が高い）
3. 情報価値: 知識としての価値（0.0=価値なし、1.0=非常に価値が高い）
4. 最新性: 情報の鮮度（0.0=古い/時代遅れ、1.0=最新）
5. 論理的一貫性: 内容の論理的整合性（0.0=矛盾だらけ、1.0=完全に一貫）

最終評価:
各項目の評価結果とスコア、そして総合的な信頼性スコア（0.0～1.0）を出力してください。
また、このコンテンツを知識として採用すべきかどうかを「採用」または「不採用」で示してください。
不採用の場合は、その理由を簡潔に説明してください。

回答は以下のJSON形式で出力してください:
{
  "評価": {
    "正確性": スコア,
    "関連性": スコア,
    "情報価値": スコア,
    "最新性": スコア,
    "論理的一貫性": スコア
  },
  "総合スコア": スコア,
  "判定": "採用" または "不採用",
  "理由": "判定理由の説明"
}
""")
    
    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        LLMを使用した検証
        
        Args:
            data: 検証対象データ
            context: 検証コンテキスト
            
        Returns:
            (検証通過フラグ, 信頼性スコア, 追加情報)
        """
        if not self.llm_engine:
            self.logger.warning("LLM engine not available for validation")
            return True, 0.5, {"warning": "LLM validation skipped"}
        
        content = data.get("content", "")
        topic = data.get("topic", "")
        
        if not content:
            self.logger.warning("No content found in data")
            return False, 0.0, {"reason": "No content to validate"}
        
        if not topic and context:
            topic = context.get("topic", "")
        
        # コンテキスト情報の準備
        context_str = ""
        if context:
            context_items = []
            for key, value in context.items():
                if key not in ["topic"] and isinstance(value, (str, int, float, bool)):
                    context_items.append(f"{key}: {value}")
            context_str = "\n".join(context_items)
        
        # プロンプトの作成
        prompt = self.validation_template.format(
            topic=topic,
            content=content[:2000],  # 長すぎる場合は切り詰め
            context=context_str
        )
        
        # LLMを使用した検証
        try:
            # 同期的に呼び出す場合（簡易実装）
            response = ""
            
            if hasattr(self.llm_engine, "generate"):
                response = self.llm_engine.generate(prompt)
            # 非同期呼び出しをサポートする場合
            elif hasattr(self.llm_engine, "chat"):
                messages = [{"role": "user", "content": prompt}]
                response_dict = self.llm_engine.chat(messages)
                response = response_dict.get("content", "")
            
            # JSONレスポンスの抽出
            json_str = self._extract_json(response)
            if not json_str:
                self.logger.warning("Failed to extract JSON from LLM response")
                return True, 0.6, {"warning": "Unable to parse LLM validation results"}
                
            # JSONパース
            result = json.loads(json_str)
            
            # 結果の解析
            evaluations = result.get("評価", {})
            total_score = result.get("総合スコア", 0.6)
            decision = result.get("判定", "不採用")
            reason = result.get("理由", "")
            
            passed = decision == "採用"
            
            # 詳細結果
            details = {
                "score": total_score,
                "evaluations": evaluations,
                "decision": decision,
                "reason": reason
            }
            
            return passed, total_score, details
            
        except Exception as e:
            self.logger.error(f"Error in LLM validation: {e}")
            return True, 0.5, {"warning": f"LLM validation failed: {str(e)}"}
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        テキストからJSON部分を抽出
        
        Args:
            text: 対象テキスト
            
        Returns:
            抽出されたJSON文字列、見つからない場合はNone
        """
        # JSON部分を抽出する正規表現
        json_pattern = r'({[\s\S]*})'
        
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                # JSONとして解析できるか確認
                json.loads(match)
                return match
            except:
                continue
                
        return None


class KnowledgeValidator:
    """複数の検証戦略を組み合わせた知識検証システム"""
    
    def __init__(self, 
                 knowledge_base: KnowledgeBase,
                 llm_engine: Optional[LLMEngine] = None,
                 config: Dict[str, Any] = None):
        """
        知識検証システムを初期化
        
        Args:
            knowledge_base: 知識ベース
            llm_engine: LLMエンジン（省略可）
            config: 設定情報
        """
        self.knowledge_base = knowledge_base
        self.llm_engine = llm_engine
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 検証戦略のリスト
        self.strategies: List[ValidationStrategy] = []
        
        # 検証履歴
        self.validation_history: List[Dict[str, Any]] = []
        
        # 戦略の初期化
        self._init_strategies()
        
        # 総合検証設定
        self.min_total_score = self.config.get("min_total_score", 0.6)
        self.require_all_passed = self.config.get("require_all_passed", False)
        
        self.logger.info("KnowledgeValidator initialized")
    
    def _init_strategies(self) -> None:
        """検証戦略を初期化"""
        # ソース信頼性戦略
        source_config = self.config.get("source_reliability", {})
        self.strategies.append(SourceReliabilityStrategy(
            weight=source_config.get("weight", 1.0),
            config=source_config
        ))
        
        # 内容一貫性戦略
        consistency_config = self.config.get("content_consistency", {})
        self.strategies.append(ContentConsistencyStrategy(
            knowledge_base=self.knowledge_base,
            weight=consistency_config.get("weight", 0.8),
            config=consistency_config
        ))
        
        # LLM検証戦略（LLMエンジンがある場合）
        if self.llm_engine:
            llm_config = self.config.get("llm_validation", {})
            self.strategies.append(LLMValidationStrategy(
                llm_engine=self.llm_engine,
                weight=llm_config.get("weight", 1.2),
                config=llm_config
            ))
        
        self.logger.info(f"Initialized {len(self.strategies)} validation strategies")
    
    async def validate(self, 
                     data: Dict[str, Any], 
                     context: Dict[str, Any] = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        データを検証
        
        Args:
            data: 検証対象データ
            context: 検証コンテキスト
            
        Returns:
            (検証通過フラグ, 信頼性スコア, 詳細情報)
        """
        if not self.strategies:
            self.logger.warning("No validation strategies available")
            return True, 0.5, {"warning": "No validation performed"}
            
        # 各戦略で検証
        strategy_results = []
        total_weight = 0.0
        weighted_score = 0.0
        all_passed = True
        
        for strategy in self.strategies:
            self.logger.debug(f"Running validation strategy: {strategy.name}")
            passed, score, details = await strategy.validate(data, context)
            
            strategy_result = {
                "strategy": strategy.name,
                "passed": passed,
                "score": score,
                "weight": strategy.weight,
                "details": details
            }
            
            strategy_results.append(strategy_result)
            total_weight += strategy.weight
            weighted_score += score * strategy.weight
            
            if not passed:
                all_passed = False
        
        # 総合スコアの計算
        total_score = weighted_score / total_weight if total_weight > 0 else 0.5
        
        # 検証判定
        if self.require_all_passed:
            passed = all_passed and total_score >= self.min_total_score
        else:
            passed = total_score >= self.min_total_score
        
        # 詳細結果
        result = {
            "passed": passed,
            "total_score": total_score,
            "threshold": self.min_total_score,
            "all_passed": all_passed,
            "timestamp": time.time(),
            "content_hash": self._get_content_hash(data),
            "strategies": strategy_results
        }
        
        # 履歴に追加
        self.validation_history.append({
            "timestamp": time.time(),
            "content_hash": result["content_hash"],
            "result": result
        })
        
        self.logger.info(f"Validation result: passed={passed}, score={total_score:.2f}")
        return passed, total_score, result
    
    def _get_content_hash(self, data: Dict[str, Any]) -> str:
        """
        データの内容ハッシュを計算
        
        Args:
            data: 対象データ
            
        Returns:
            ハッシュ文字列
        """
        content = data.get("content", "")
        url = data.get("url", "")
        
        # コンテンツとURLを組み合わせてハッシュ化
        hash_input = f"{content}{url}"
        hash_obj = hashlib.sha256(hash_input.encode())
        
        return hash_obj.hexdigest()
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        検証統計情報を取得
        
        Returns:
            統計情報
        """
        if not self.validation_history:
            return {
                "total_validations": 0,
                "passed_validations": 0,
                "pass_rate": 0.0
            }
            
        # 基本統計
        total = len(self.validation_history)
        passed = sum(1 for entry in self.validation_history if entry["result"]["passed"])
        pass_rate = passed / total if total > 0 else 0.0
        
        # 戦略ごとの統計
        strategy_stats = {}
        for entry in self.validation_history:
            for strategy_result in entry["result"]["strategies"]:
                strategy_name = strategy_result["strategy"]
                
                if strategy_name not in strategy_stats:
                    strategy_stats[strategy_name] = {
                        "count": 0,
                        "passed": 0,
                        "total_score": 0.0
                    }
                    
                stats = strategy_stats[strategy_name]
                stats["count"] += 1
                if strategy_result["passed"]:
                    stats["passed"] += 1
                stats["total_score"] += strategy_result["score"]
        
        # 平均値の計算
        for stats in strategy_stats.values():
            if stats["count"] > 0:
                stats["pass_rate"] = stats["passed"] / stats["count"]
                stats["avg_score"] = stats["total_score"] / stats["count"]
            else:
                stats["pass_rate"] = 0.0
                stats["avg_score"] = 0.0
        
        return {
            "total_validations": total,
            "passed_validations": passed,
            "pass_rate": pass_rate,
            "strategies": strategy_stats,
            "latest_timestamp": self.validation_history[-1]["timestamp"] if self.validation_history else None
        }
    
    def save_history(self, file_path: str = "./data/validation_history.json") -> bool:
        """
        検証履歴をファイルに保存
        
        Args:
            file_path: 保存先ファイルパス
            
        Returns:
            保存が成功したかどうか
        """
        try:
            data = {
                "history": self.validation_history,
                "stats": self.get_validation_stats()
            }
            
            # ディレクトリの作成
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved validation history to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving validation history: {e}")
            return False
    
    def load_history(self, file_path: str = "./data/validation_history.json") -> bool:
        """
        検証履歴をファイルからロード
        
        Args:
            file_path: ロード元ファイルパス
            
        Returns:
            ロードが成功したかどうか
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Validation history file not found: {file_path}")
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.validation_history = data.get("history", [])
            self.logger.info(f"Loaded {len(self.validation_history)} validation history entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading validation history: {e}")
            return False
