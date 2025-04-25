"""
シンボリックAI推論結果解釈モジュール

このモジュールはシンボリックAI推論エンジンの結果を解釈し、
自然言語の説明や応用可能な形式に変換する機能を提供します。
LLMの言語理解能力とシンボリックAIの厳密さを融合させるブリッジとして機能します。
"""

import os
import sys
import re
import json
from typing import Dict, List, Any, Union, Optional, Tuple, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.core.utils.logger import Logger
from src.core.llm.engine import LLMEngine
from src.modules.reasoning.symbolic.logic_transformer import LogicTransformer


class ResultTemplates:
    """推論結果の説明テンプレート"""
    
    # 演繹的推論結果のテンプレート
    DEDUCTION_SUCCESS = """
    演繹的推論により、「{query}」が証明されました。
    証明の確信度: {confidence:.2f}
    
    証明過程:
    {steps}
    
    この結論は論理的に導かれるものであり、前提が真である限り結論も真です。
    """
    
    DEDUCTION_FAILURE = """
    演繹的推論では、「{query}」を証明できませんでした。
    確信度: {confidence:.2f}
    
    推論過程:
    {steps}
    
    これは結論が偽であることを意味するわけではなく、現在の情報からは論理的に導けないということです。
    """
    
    # 帰納的推論結果のテンプレート
    INDUCTION_RESULT = """
    帰納的推論により、次のパターンが見出されました：
    「{rule}」
    
    確信度: {confidence:.2f}
    
    この一般化は{example_count}個の例に基づいています。
    カバレッジ: {coverage:.2f}（例の{coverage_percent:.0f}%をカバー）
    
    この結果は経験的な一般化であり、新しい事例では例外が生じる可能性があります。
    """
    
    # 確率的推論結果のテンプレート
    PROBABILITY_UPDATE = """
    ベイズ更新による確率推論結果：
    
    仮説「{hypothesis}」の確率が{prior:.2f}から{posterior:.2f}に更新されました。
    
    これは提供された{evidence_count}件の証拠に基づくものです。
    尤度（証拠が仮説を支持する強さ）: {likelihood:.2f}
    
    確率推論は不確実性を明示的に扱い、絶対的な真/偽ではなく確率的な判断を提供します。
    """
    
    # アブダクション結果のテンプレート
    ABDUCTION_RESULT = """
    アブダクション（最良説明推論）の結果：
    
    観察された{observation_count}件の事象に対する最良の説明は：
    「{explanation}」
    
    説明スコア: {score:.2f}
    
    説明過程:
    {steps}
    
    これは観察を最もよく説明する仮説であり、真である保証はありませんが、現時点で最も合理的な説明です。
    """
    
    # 類推的推論結果のテンプレート
    ANALOGY_RESULT = """
    類推的推論の結果：
    
    ソースドメインの関係「{source_relation}」から類推されるターゲットドメインの関係：
    「{mapped_relation}」
    
    確信度: {confidence:.2f}
    
    マッピング:
    {mappings}
    
    類推は完全な証明ではなく、ドメイン間の構造的類似性に基づいた推論です。
    """
    
    # 一貫性チェック結果のテンプレート
    CONSISTENCY_RESULT = """
    一貫性検証結果:
    
    {statement_count}個の命題は{consistency_status}。
    
    {contradiction_info}
    
    論理的一貫性は、命題集合に矛盾が含まれていないことを意味します。
    """


class ResultInterpreter:
    """推論結果解釈クラス"""
    
    def __init__(self, llm_engine: LLMEngine, logic_transformer: LogicTransformer = None):
        """
        結果解釈モジュールの初期化
        
        Args:
            llm_engine (LLMEngine): LLM処理エンジン
            logic_transformer (LogicTransformer, optional): 論理変換モジュール
        """
        self.logger = Logger(__name__)
        self.llm = llm_engine
        self.logic_transformer = logic_transformer
        self.templates = ResultTemplates()
        
        self.logger.info("ResultInterpreter initialized")
    
    def interpret_deduction_result(self, result: Dict[str, Any], detail_level: str = "medium") -> str:
        """
        演繹的推論結果の解釈
        
        Args:
            result (Dict[str, Any]): 推論結果
            detail_level (str, optional): 説明の詳細レベル（low/medium/high）
            
        Returns:
            str: 自然言語での説明
        """
        self.logger.info(f"Interpreting deduction result with detail level: {detail_level}")
        
        # 結果の抽出
        query = result.get("query", "不明なクエリ")
        success = result.get("result", False)
        confidence = result.get("confidence", 0.0)
        steps = result.get("steps", [])
        
        # 詳細レベルに基づく推論ステップのフィルタリング
        filtered_steps = self._filter_steps_by_detail(steps, detail_level)
        
        # ステップの整形
        formatted_steps = "\n".join([f"- {step}" for step in filtered_steps])
        
        # テンプレートの選択と値の挿入
        if success:
            explanation = self.templates.DEDUCTION_SUCCESS.format(
                query=query,
                confidence=confidence,
                steps=formatted_steps
            )
        else:
            explanation = self.templates.DEDUCTION_FAILURE.format(
                query=query,
                confidence=confidence,
                steps=formatted_steps
            )
        
        # LLMを使用した自然な文章への変換（オプション）
        if self.llm and detail_level == "high":
            try:
                enhanced_explanation = self._enhance_explanation_with_llm(
                    explanation, "deduction", result
                )
                if enhanced_explanation:
                    explanation = enhanced_explanation
            except Exception as e:
                self.logger.error(f"Error enhancing explanation with LLM: {str(e)}")
        
        return explanation.strip()
    
    def interpret_induction_result(self, result: Dict[str, Any], detail_level: str = "medium") -> str:
        """
        帰納的推論結果の解釈
        
        Args:
            result (Dict[str, Any]): 推論結果
            detail_level (str, optional): 説明の詳細レベル（low/medium/high）
            
        Returns:
            str: 自然言語での説明
        """
        self.logger.info(f"Interpreting induction result with detail level: {detail_level}")
        
        # 結果の抽出
        target_concept = result.get("target_concept", "不明な概念")
        rule = result.get("rule", "")
        confidence = result.get("confidence", 0.0)
        details = result.get("details", {})
        example_count = result.get("example_count", 0)
        
        # 必要な情報の抽出と計算
        coverage = details.get("coverage", 0.0)
        coverage_percent = coverage * 100
        
        # ルールの自然言語変換
        natural_rule = rule
        if self.logic_transformer and rule:
            try:
                natural_rule = self.logic_transformer.logic_to_natural(rule)
            except Exception as e:
                self.logger.error(f"Error converting rule to natural language: {str(e)}")
        
        # テンプレートへの値の挿入
        explanation = self.templates.INDUCTION_RESULT.format(
            rule=natural_rule,
            confidence=confidence,
            example_count=example_count,
            coverage=coverage,
            coverage_percent=coverage_percent
        )
        
        # LLMを使用した自然な文章への変換（オプション）
        if self.llm and detail_level == "high":
            try:
                enhanced_explanation = self._enhance_explanation_with_llm(
                    explanation, "induction", result
                )
                if enhanced_explanation:
                    explanation = enhanced_explanation
            except Exception as e:
                self.logger.error(f"Error enhancing explanation with LLM: {str(e)}")
        
        return explanation.strip()
    
    def interpret_probabilistic_result(self, result: Dict[str, Any], detail_level: str = "medium") -> str:
        """
        確率的推論結果の解釈
        
        Args:
            result (Dict[str, Any]): 推論結果
            detail_level (str, optional): 説明の詳細レベル（low/medium/high）
            
        Returns:
            str: 自然言語での説明
        """
        self.logger.info(f"Interpreting probabilistic result with detail level: {detail_level}")
        
        # 結果の抽出
        hypothesis = result.get("hypothesis", "不明な仮説")
        prior = result.get("prior", 0.0)
        posterior = result.get("posterior", 0.0)
        evidence_count = result.get("evidence_count", 0)
        details = result.get("details", {})
        
        # 詳細情報の抽出
        likelihood = details.get("likelihood", 0.0)
        
        # テンプレートへの値の挿入
        explanation = self.templates.PROBABILITY_UPDATE.format(
            hypothesis=hypothesis,
            prior=prior,
            posterior=posterior,
            evidence_count=evidence_count,
            likelihood=likelihood
        )
        
        # 詳細レベルに応じた追加情報
        if detail_level == "high":
            # 証拠の詳細情報を追加
            evidence_details = result.get("evidence", {})
            if evidence_details:
                evidence_str = "\n証拠の詳細:\n"
                for key, value in evidence_details.items():
                    evidence_str += f"- {key}: {value}\n"
                explanation += evidence_str
        
        # LLMを使用した自然な文章への変換（オプション）
        if self.llm and detail_level == "high":
            try:
                enhanced_explanation = self._enhance_explanation_with_llm(
                    explanation, "probabilistic", result
                )
                if enhanced_explanation:
                    explanation = enhanced_explanation
            except Exception as e:
                self.logger.error(f"Error enhancing explanation with LLM: {str(e)}")
        
        return explanation.strip()
    
    def interpret_abduction_result(self, result: Dict[str, Any], detail_level: str = "medium") -> str:
        """
        アブダクション推論結果の解釈
        
        Args:
            result (Dict[str, Any]): 推論結果
            detail_level (str, optional): 説明の詳細レベル（low/medium/high）
            
        Returns:
            str: 自然言語での説明
        """
        self.logger.info(f"Interpreting abduction result with detail level: {detail_level}")
        
        # 結果の抽出
        explanation = result.get("best_explanation", "説明が見つかりませんでした")
        score = result.get("score", 0.0)
        steps = result.get("explanation_steps", [])
        observation_count = result.get("observation_count", 0)
        
        # 詳細レベルに基づく推論ステップのフィルタリング
        filtered_steps = self._filter_steps_by_detail(steps, detail_level)
        
        # ステップの整形
        formatted_steps = "\n".join([f"- {step}" for step in filtered_steps])
        
        # 自然言語に変換
        natural_explanation = explanation
        if self.logic_transformer and isinstance(explanation, str):
            try:
                natural_explanation = self.logic_transformer.logic_to_natural(explanation)
            except Exception as e:
                self.logger.error(f"Error converting explanation to natural language: {str(e)}")
        
        # テンプレートへの値の挿入
        interpretation = self.templates.ABDUCTION_RESULT.format(
            explanation=natural_explanation,
            score=score,
            steps=formatted_steps,
            observation_count=observation_count
        )
        
        # LLMを使用した自然な文章への変換（オプション）
        if self.llm and detail_level == "high":
            try:
                enhanced_explanation = self._enhance_explanation_with_llm(
                    interpretation, "abduction", result
                )
                if enhanced_explanation:
                    interpretation = enhanced_explanation
            except Exception as e:
                self.logger.error(f"Error enhancing explanation with LLM: {str(e)}")
        
        return interpretation.strip()
    
    def interpret_analogy_result(self, result: Dict[str, Any], detail_level: str = "medium") -> str:
        """
        類推的推論結果の解釈
        
        Args:
            result (Dict[str, Any]): 推論結果
            detail_level (str, optional): 説明の詳細レベル（low/medium/high）
            
        Returns:
            str: 自然言語での説明
        """
        self.logger.info(f"Interpreting analogy result with detail level: {detail_level}")
        
        # 結果の抽出
        source_relation = result.get("source_relation", "不明な関係")
        mapped_relation = result.get("mapped_relation", {})
        confidence = result.get("confidence", 0.0)
        mappings = result.get("mapping_explanation", [])
        
        # マッピングの整形
        formatted_mappings = "\n".join([f"- {mapping}" for mapping in mappings])
        
        # マッピングされた関係の自然言語表現
        mapped_relation_str = "マッピングできませんでした"
        if mapped_relation:
            relation_type = mapped_relation.get("type", "")
            entities = mapped_relation.get("entities", [])
            if relation_type and entities:
                mapped_relation_str = f"{relation_type}({', '.join(entities)})"
        
        # テンプレートへの値の挿入
        explanation = self.templates.ANALOGY_RESULT.format(
            source_relation=source_relation,
            mapped_relation=mapped_relation_str,
            confidence=confidence,
            mappings=formatted_mappings
        )
        
        # LLMを使用した自然な文章への変換（オプション）
        if self.llm and detail_level == "high":
            try:
                enhanced_explanation = self._enhance_explanation_with_llm(
                    explanation, "analogy", result
                )
                if enhanced_explanation:
                    explanation = enhanced_explanation
            except Exception as e:
                self.logger.error(f"Error enhancing explanation with LLM: {str(e)}")
        
        return explanation.strip()
    
    def interpret_consistency_result(self, result: Dict[str, Any], detail_level: str = "medium") -> str:
        """
        一貫性検証結果の解釈
        
        Args:
            result (Dict[str, Any]): 検証結果
            detail_level (str, optional): 説明の詳細レベル（low/medium/high）
            
        Returns:
            str: 自然言語での説明
        """
        self.logger.info(f"Interpreting consistency result with detail level: {detail_level}")
        
        # 結果の抽出
        is_consistent = result.get("is_consistent", False)
        direct_contradictions = result.get("direct_contradictions", [])
        inferred_contradictions = result.get("inferred_contradictions", [])
        statement_count = result.get("statement_count", 0)
        
        # 一貫性状態の文字列
        consistency_status = "論理的に一貫しています" if is_consistent else "矛盾を含んでいます"
        
        # 矛盾情報の整形
        contradiction_info = "矛盾は検出されませんでした。"
        if not is_consistent:
            contradiction_info = "検出された矛盾:\n"
            
            if direct_contradictions:
                contradiction_info += "直接的な矛盾:\n"
                for pair in direct_contradictions:
                    contradiction_info += f"- '{pair[0]}' と '{pair[1]}'\n"
            
            if inferred_contradictions:
                contradiction_info += "推論による矛盾:\n"
                for triple in inferred_contradictions:
                    contradiction_info += f"- '{triple[0]}' と '{triple[1]}'\n"
                    
                    if detail_level == "high":
                        contradiction_info += "  推論ステップ:\n"
                        for step in triple[2]:
                            contradiction_info += f"  - {step}\n"
        
        # テンプレートへの値の挿入
        explanation = self.templates.CONSISTENCY_RESULT.format(
            statement_count=statement_count,
            consistency_status=consistency_status,
            contradiction_info=contradiction_info
        )
        
        # LLMを使用した自然な文章への変換（オプション）
        if self.llm and detail_level == "high":
            try:
                enhanced_explanation = self._enhance_explanation_with_llm(
                    explanation, "consistency", result
                )
                if enhanced_explanation:
                    explanation = enhanced_explanation
            except Exception as e:
                self.logger.error(f"Error enhancing explanation with LLM: {str(e)}")
        
        return explanation.strip()
    
    def generate_application_recommendations(self, inference_result: Dict[str, Any], 
                                           context: Dict[str, Any], 
                                           domain: str = "general") -> List[Dict[str, Any]]:
        """
        推論結果に基づく応用推奨の生成
        
        Args:
            inference_result (Dict[str, Any]): 推論結果
            context (Dict[str, Any]): 適用コンテキスト
            domain (str, optional): ドメイン（例: "programming", "science", "business"）
            
        Returns:
            List[Dict[str, Any]]: 推奨事項のリスト
        """
        self.logger.info(f"Generating application recommendations for domain: {domain}")
        
        if not self.llm:
            self.logger.warning("LLM engine required for generating recommendations")
            return []
            
        # 推論タイプの特定
        inference_type = self._determine_inference_type(inference_result)
        
        # プロンプトの構築
        prompt = f"""
        あなたは推論結果の実用的応用を提案する専門家です。
        以下の{inference_type}推論結果に基づいて、{domain}ドメインでの応用推奨を提案してください。
        
        推論結果:
        {json.dumps(inference_result, ensure_ascii=False, indent=2)}
        
        コンテキスト:
        {json.dumps(context, ensure_ascii=False, indent=2)}
        
        ドメイン: {domain}
        
        以下の形式で、具体的で実用的な推奨事項を3つ提案してください：
        
        [
          {{
            "title": "推奨タイトル",
            "description": "詳細な説明",
            "application_steps": ["ステップ1", "ステップ2", ...],
            "confidence": 0.0 - 1.0の間の信頼度,
            "required_resources": ["必要なリソース1", "リソース2", ...]
          }},
          ...
        ]
        
        JSONのみを出力し、他の説明は含めないでください。
        """
        
        try:
            # LLMによる推奨事項生成
            response = self.llm.generate(prompt, max_tokens=1000)
            
            # JSONの抽出
            recommendations = []
            try:
                # 文字列からJSONを抽出
                json_str = self._extract_json_from_text(response)
                recommendations = json.loads(json_str)
                
                # 結果の検証
                if isinstance(recommendations, list) and len(recommendations) > 0:
                    self.logger.info(f"Generated {len(recommendations)} recommendations")
                else:
                    self.logger.warning("Generated recommendations are empty or invalid")
                    recommendations = []
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse recommendations JSON: {str(e)}")
                recommendations = []
                
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def translate_to_domain_specific(self, inference_result: Dict[str, Any], 
                                    target_domain: str, 
                                    format_type: str = "general") -> Dict[str, Any]:
        """
        推論結果をドメイン固有の表現に変換
        
        Args:
            inference_result (Dict[str, Any]): 推論結果
            target_domain (str): 対象ドメイン（例: "programming", "medicine", "finance"）
            format_type (str, optional): 出力形式（例: "general", "technical", "simplified"）
            
        Returns:
            Dict[str, Any]: ドメイン固有の表現に変換された結果
        """
        self.logger.info(f"Translating inference result to {target_domain} domain")
        
        if not self.llm:
            self.logger.warning("LLM engine required for domain-specific translation")
            return inference_result
            
        # 推論タイプの特定
        inference_type = self._determine_inference_type(inference_result)
        
        # プロンプトの構築
        prompt = f"""
        あなたは論理推論結果を特定のドメインの専門家向けに翻訳する専門家です。
        以下の{inference_type}推論結果を{target_domain}ドメインの{format_type}な表現に変換してください。
        
        推論結果:
        {json.dumps(inference_result, ensure_ascii=False, indent=2)}
        
        対象ドメイン: {target_domain}
        形式タイプ: {format_type}
        
        変換結果は以下の形式でJSON出力してください:
        
        {{
          "domain_specific_result": 変換された主要結果,
          "domain_specific_explanation": ドメイン専門家向けの説明,
          "domain_specific_terminology": 使用された専門用語の説明,
          "implications_for_domain": このドメインへの影響や意義
        }}
        
        JSONのみを出力し、他の説明は含めないでください。
        """
        
        try:
            # LLMによる変換
            response = self.llm.generate(prompt, max_tokens=800)
            
            # JSONの抽出と解析
            try:
                json_str = self._extract_json_from_text(response)
                domain_specific = json.loads(json_str)
                
                # オリジナルの推論結果をコピーし、ドメイン固有情報を追加
                result = inference_result.copy()
                result["domain_translation"] = domain_specific
                
                self.logger.info(f"Successfully translated result to {target_domain} domain")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse domain-specific translation JSON: {str(e)}")
                return inference_result
                
        except Exception as e:
            self.logger.error(f"Error translating to domain-specific format: {str(e)}")
            return inference_result
    
    def combine_inference_results(self, results: List[Dict[str, Any]], 
                                 combination_strategy: str = "weighted") -> Dict[str, Any]:
        """
        複数の推論結果の組み合わせ
        
        Args:
            results (List[Dict[str, Any]]): 推論結果のリスト
            combination_strategy (str, optional): 組み合わせ戦略
            
        Returns:
            Dict[str, Any]: 組み合わせた結果
        """
        self.logger.info(f"Combining {len(results)} inference results using {combination_strategy} strategy")
        
        if not results:
            return {}
            
        if len(results) == 1:
            return results[0]
            
        # 結果の種類を確認
        result_types = [self._determine_inference_type(r) for r in results]
        
        # 全て同じ種類の場合
        if all(t == result_types[0] for t in result_types):
            if result_types[0] == "deduction":
                return self._combine_deduction_results(results, combination_strategy)
            elif result_types[0] == "probabilistic":
                return self._combine_probabilistic_results(results, combination_strategy)
            else:
                # その他の結果タイプに対するフォールバック
                return self._combine_generic_results(results, combination_strategy)
        else:
            # 異なる種類の結果の組み合わせ（より複雑）
            return self._combine_heterogeneous_results(results, combination_strategy)
    
    def explain_contradiction(self, contradiction: Tuple[str, str], 
                            context: Optional[Dict[str, Any]] = None) -> str:
        """
        矛盾の詳細な説明
        
        Args:
            contradiction (Tuple[str, str]): 矛盾する命題のペア
            context (Dict[str, Any], optional): 説明コンテキスト
            
        Returns:
            str: 矛盾の説明
        """
        if not self.llm:
            # LLMがない場合は基本的な説明を提供
            return f"命題「{contradiction[0]}」と「{contradiction[1]}」は互いに矛盾しています。"
            
        # プロンプトの構築
        stmt1, stmt2 = contradiction
        
        context_json = "{}"
        if context:
            context_json = json.dumps(context, ensure_ascii=False, indent=2)
            
        prompt = f"""
        あなたは論理矛盾を説明する専門家です。
        以下の2つの命題間の矛盾を詳細に説明してください。
        
        命題1: {stmt1}
        命題2: {stmt2}
        
        コンテキスト:
        {context_json}
        
        矛盾の性質と、なぜこれらの命題が共存できないのかを説明してください。
        可能であれば、矛盾を解消するための考えられる方法も提案してください。
        
        回答は3つのセクションに分けてください:
        1. 矛盾の種類と説明
        2. なぜこれらの命題が互いに矛盾するのか
        3. 矛盾解消の可能性
        """
        
        try:
            explanation = self.llm.generate(prompt, max_tokens=500)
            return explanation.strip()
        except Exception as e:
            self.logger.error(f"Error generating contradiction explanation: {str(e)}")
            return f"命題「{stmt1}」と「{stmt2}」は互いに矛盾しています。詳細な分析は現在利用できません。"
    
    def _determine_inference_type(self, result: Dict[str, Any]) -> str:
        """
        推論結果の種類を判断
        
        Args:
            result (Dict[str, Any]): 推論結果
            
        Returns:
            str: 推論タイプ
        """
        # 結果の内容に基づいて推論タイプを判断
        if "query" in result and ("result" in result or "confidence" in result):
            return "deduction"
        elif "target_concept" in result and "rule" in result:
            return "induction"
        elif "hypothesis" in result and "posterior" in result:
            return "probabilistic"
        elif "best_explanation" in result:
            return "abduction"
        elif "mapped_relation" in result:
            return "analogy"
        elif "is_consistent" in result:
            return "consistency"
        else:
            return "unknown"
    
    def _filter_steps_by_detail(self, steps: List[str], detail_level: str) -> List[str]:
        """
        詳細レベルに基づく推論ステップのフィルタリング
        
        Args:
            steps (List[str]): 推論ステップ
            detail_level (str): 詳細レベル
            
        Returns:
            List[str]: フィルタリングされたステップ
        """
        if not steps:
            return []
            
        if detail_level == "low":
            # 低詳細: 最初と最後のステップのみ
            if len(steps) <= 2:
                return steps
            else:
                return [steps[0], "...", steps[-1]]
        elif detail_level == "medium":
            # 中詳細: ステップ数に応じて間引き
            if len(steps) <= 5:
                return steps
            else:
                # 間引きアルゴリズム
                result = [steps[0]]
                step_size = max(1, len(steps) // 5)
                for i in range(step_size, len(steps) - 1, step_size):
                    result.append(steps[i])
                result.append(steps[-1])
                return result
        else:  # "high"
            # 高詳細: 全ステップを含む
            return steps
    
    def _enhance_explanation_with_llm(self, basic_explanation: str, 
                                    inference_type: str, 
                                    result: Dict[str, Any]) -> str:
        """
        LLMを使用した説明の強化
        
        Args:
            basic_explanation (str): 基本的な説明
            inference_type (str): 推論タイプ
            result (Dict[str, Any]): 元の推論結果
            
        Returns:
            str: 強化された説明
        """
        prompt = f"""
        あなたは論理推論の結果を一般の人に分かりやすく説明する専門家です。
        以下の{inference_type}推論の結果を、より自然で理解しやすい日本語に書き直してください。
        
        専門用語は必要に応じて平易な言葉に置き換え、具体例を加えて理解を助けてください。
        元の情報の正確さは維持しつつ、より読みやすくしてください。
        
        元の説明:
        {basic_explanation}
        
        推論結果の詳細:
        {json.dumps(result, ensure_ascii=False, indent=2)}
        """
        
        try:
            enhanced = self.llm.generate(prompt, max_tokens=800)
            return enhanced.strip()
        except Exception as e:
            self.logger.error(f"Error enhancing explanation: {str(e)}")
            return ""
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        テキストからJSONを抽出
        
        Args:
            text (str): 入力テキスト
            
        Returns:
            str: 抽出されたJSON文字列
        """
        # JSON部分の抽出
        json_pattern = r'(\{|\[).*?(\}|\])'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            # 最大の一致を使用
            largest_match = max(matches, key=lambda x: len(x[0] + x[1]))
            start_bracket, end_bracket = largest_match
            
            # 開始ブラケットから終了ブラケットまでのテキストを抽出
            start_idx = text.find(start_bracket)
            end_idx = text.rfind(end_bracket) + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                return text[start_idx:end_idx]
        
        # JSON形式が見つからない場合
        return "{}"
    
    def _combine_deduction_results(self, results: List[Dict[str, Any]], 
                                  strategy: str) -> Dict[str, Any]:
        """
        演繹的推論結果の組み合わせ
        
        Args:
            results (List[Dict[str, Any]]): 推論結果のリスト
            strategy (str): 組み合わせ戦略
            
        Returns:
            Dict[str, Any]: 組み合わせた結果
        """
        # 全ての推論が同じクエリに対するものか確認
        queries = [r.get("query", "") for r in results]
        if len(set(queries)) > 1:
            self.logger.warning("Cannot combine deduction results with different queries")
            return results[0]  # 最初の結果を返す
            
        query = queries[0]
        
        # 推論結果のブール値を抽出
        bool_results = [r.get("result", False) for r in results]
        
        # 結果の組み合わせ方法を決定
        if strategy == "conservative":
            # 保守的: 全てが真の場合のみ真
            combined_result = all(bool_results)
            confidence = min([r.get("confidence", 0.0) for r in results]) if combined_result else 0.0
        elif strategy == "liberal":
            # リベラル: 一つでも真ならば真
            combined_result = any(bool_results)
            confidence = max([r.get("confidence", 0.0) for r in results]) if combined_result else 0.0
        else:  # "weighted"
            # 重み付け: 確信度による重み付け
            true_results = [r for r, result in zip(results, bool_results) if result]
            if true_results:
                combined_result = True
                confidence = sum([r.get("confidence", 0.0) for r in true_results]) / len(true_results)
            else:
                combined_result = False
                confidence = 0.0
                
        # 推論ステップの統合
        all_steps = []
        for r in results:
            steps = r.get("steps", [])
            all_steps.extend(steps)
            
        # 重複ステップの除去
        unique_steps = []
        for step in all_steps:
            if step not in unique_steps:
                unique_steps.append(step)
                
        # 結果の構築
        combined = {
            "query": query,
            "result": combined_result,
            "confidence": confidence,
            "steps": unique_steps,
            "combined_from": len(results)
        }
        
        return combined
    
    def _combine_probabilistic_results(self, results: List[Dict[str, Any]], 
                                     strategy: str) -> Dict[str, Any]:
        """
        確率的推論結果の組み合わせ
        
        Args:
            results (List[Dict[str, Any]]): 推論結果のリスト
            strategy (str): 組み合わせ戦略
            
        Returns:
            Dict[str, Any]: 組み合わせた結果
        """
        # 全ての推論が同じ仮説に対するものか確認
        hypotheses = [r.get("hypothesis", "") for r in results]
        if len(set(hypotheses)) > 1:
            self.logger.warning("Cannot combine probabilistic results with different hypotheses")
            return results[0]  # 最初の結果を返す
            
        hypothesis = hypotheses[0]
        
        # 確率の抽出
        priors = [r.get("prior", 0.5) for r in results]
        posteriors = [r.get("posterior", 0.5) for r in results]
        
        # 結果の組み合わせ方法を決定
        if strategy == "conservative":
            # 保守的: 最小の事後確率
            combined_posterior = min(posteriors)
            combined_prior = min(priors)
        elif strategy == "liberal":
            # リベラル: 最大の事後確率
            combined_posterior = max(posteriors)
            combined_prior = max(priors)
        else:  # "weighted"
            # 重み付け: 平均
            combined_posterior = sum(posteriors) / len(posteriors)
            combined_prior = sum(priors) / len(priors)
            
        # 結果の構築
        combined = {
            "hypothesis": hypothesis,
            "prior": combined_prior,
            "posterior": combined_posterior,
            "evidence_count": sum([r.get("evidence_count", 0) for r in results]),
            "combination_strategy": strategy,
            "combined_from": len(results)
        }
        
        return combined
    
    def _combine_generic_results(self, results: List[Dict[str, Any]], 
                               strategy: str) -> Dict[str, Any]:
        """
        一般的な推論結果の組み合わせ
        
        Args:
            results (List[Dict[str, Any]]): 推論結果のリスト
            strategy (str): 組み合わせ戦略
            
        Returns:
            Dict[str, Any]: 組み合わせた結果
        """
        # 最も確信度の高い結果を選択
        if "confidence" in results[0]:
            confidences = [r.get("confidence", 0.0) for r in results]
            best_index = confidences.index(max(confidences))
            return results[best_index]
        else:
            # 確信度が利用できない場合は最初の結果を返す
            return results[0]
    
    def _combine_heterogeneous_results(self, results: List[Dict[str, Any]], 
                                     strategy: str) -> Dict[str, Any]:
        """
        異なるタイプの推論結果の組み合わせ
        
        Args:
            results (List[Dict[str, Any]]): 推論結果のリスト
            strategy (str): 組み合わせ戦略
            
        Returns:
            Dict[str, Any]: 組み合わせた結果
        """
        # 異なるタイプの推論結果のマージは複雑
        # 結果をタイプごとにグループ化
        result_groups = {}
        for r in results:
            inference_type = self._determine_inference_type(r)
            if inference_type not in result_groups:
                result_groups[inference_type] = []
            result_groups[inference_type].append(r)
            
        # 各グループごとに結果を組み合わせ
        combined_results = {}
        for inference_type, group in result_groups.items():
            if len(group) > 1:
                if inference_type == "deduction":
                    combined_results[inference_type] = self._combine_deduction_results(group, strategy)
                elif inference_type == "probabilistic":
                    combined_results[inference_type] = self._combine_probabilistic_results(group, strategy)
                else:
                    combined_results[inference_type] = self._combine_generic_results(group, strategy)
            else:
                combined_results[inference_type] = group[0]
                
        # マルチタイプの結合結果を構築
        multi_type_result = {
            "combined_type": "heterogeneous",
            "strategy": strategy,
            "result_types": list(result_groups.keys()),
            "results": combined_results
        }
        
        return multi_type_result
