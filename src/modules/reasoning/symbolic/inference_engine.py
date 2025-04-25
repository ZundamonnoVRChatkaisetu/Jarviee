"""
シンボリックAI推論エンジン

このモジュールは論理ベースの推論機能を提供します。演繹的推論、帰納的推論、確率的推論、
およびその他の論理推論メカニズムを実装し、LLMの推論能力を補完します。
"""

import os
import sys
import re
import math
import json
import copy
from typing import Dict, List, Any, Union, Optional, Tuple, Set, Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.core.utils.logger import Logger
from src.modules.reasoning.symbolic.kb_interface import SymbolicKnowledgeInterface


class DeductionEngine:
    """演繹的推論エンジン"""
    
    def __init__(self):
        """演繹エンジンの初期化"""
        self.logger = Logger(__name__)
        
        # 推論ルールのパースと処理のための正規表現
        self.rule_patterns = {
            'modus_ponens': r'(.*)\s*→\s*(.*)',  # A → B
            'modus_tollens': r'¬\s*(.*)\s*→\s*¬\s*(.*)',  # ¬B → ¬A
            'conjunction': r'(.*)\s*∧\s*(.*)',  # A ∧ B
            'disjunction': r'(.*)\s*∨\s*(.*)',  # A ∨ B
            'negation': r'¬\s*(.*)'  # ¬A
        }
        
        self.logger.info("DeductionEngine initialized")
    
    def infer(self, premises: List[str], query: str, knowledge_context: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, float, List[str]]:
        """
        演繹的推論の実行
        
        Args:
            premises (List[str]): 前提となる命題のリスト
            query (str): 検証したい結論
            knowledge_context (List[Dict[str, Any]], optional): 利用可能な知識コンテキスト
            
        Returns:
            Tuple[bool, float, List[str]]: 
                - 推論結果の真偽
                - 確信度 (0-1)
                - 推論ステップのリスト
        """
        self.logger.info(f"Starting deduction with {len(premises)} premises")
        
        # 推論の状態を管理
        known_facts = set(premises)  # 既知の事実
        inference_steps = []  # 推論ステップの記録
        
        # 知識コンテキストからルールの抽出
        rules = []
        if knowledge_context:
            for item in knowledge_context:
                if 'rule' in item:
                    rules.append(item['rule'])
        
        # 最大ステップ数の設定（無限ループ防止）
        max_steps = 100
        step_count = 0
        
        # 結論が既に前提に含まれているか確認
        if query in known_facts:
            inference_steps.append(f"直接的前提: {query}")
            return True, 1.0, inference_steps
            
        # 推論ループ
        while step_count < max_steps:
            step_count += 1
            initial_fact_count = len(known_facts)
            
            # ルールの適用
            new_facts = self._apply_rules(known_facts, rules)
            
            # 新しい事実の追加
            for fact, rule in new_facts:
                if fact not in known_facts:
                    known_facts.add(fact)
                    inference_steps.append(f"ルール適用: {rule} → {fact}")
                    
                    # 結論に到達したか確認
                    if fact == query:
                        self.logger.info(f"Deduction succeeded in {step_count} steps")
                        return True, 1.0, inference_steps
            
            # 新しい事実が追加されなかった場合は終了
            if len(known_facts) == initial_fact_count:
                break
        
        # 結論に到達できなかった場合
        self.logger.info(f"Deduction failed after {step_count} steps")
        
        # 部分的な証明の信頼性評価
        # 単純な実装例: 前提と結論の類似性に基づく評価
        confidence = self._estimate_confidence(known_facts, query)
        
        return False, confidence, inference_steps
    
    def _apply_rules(self, facts: Set[str], rules: List[str]) -> List[Tuple[str, str]]:
        """
        ルールを適用して新しい事実を導出
        
        Args:
            facts (Set[str]): 既知の事実
            rules (List[str]): 適用可能なルール
            
        Returns:
            List[Tuple[str, str]]: 新しい事実とその導出ルールのペアのリスト
        """
        new_facts = []
        
        # 含意ルール（A → B）の適用: モーダスポネンス
        for rule in rules:
            match = re.match(self.rule_patterns['modus_ponens'], rule)
            if match:
                antecedent, consequent = match.groups()
                antecedent = antecedent.strip()
                consequent = consequent.strip()
                
                # 前件が既知の場合、後件を追加
                if antecedent in facts and consequent not in facts:
                    new_facts.append((consequent, f"{antecedent} → {consequent}"))
        
        # 連言導入（A, B → A ∧ B）
        for fact1 in facts:
            for fact2 in facts:
                if fact1 != fact2:
                    conjunction = f"{fact1} ∧ {fact2}"
                    if conjunction not in facts:
                        new_facts.append((conjunction, f"{fact1}, {fact2} → {conjunction}"))
        
        # 連言消去（A ∧ B → A, A ∧ B → B）
        for fact in facts:
            match = re.match(self.rule_patterns['conjunction'], fact)
            if match:
                left, right = match.groups()
                left = left.strip()
                right = right.strip()
                
                if left not in facts:
                    new_facts.append((left, f"{fact} → {left}"))
                if right not in facts:
                    new_facts.append((right, f"{fact} → {right}"))
        
        # 選言導入（A → A ∨ B, B → A ∨ B）
        for fact in facts:
            for other_fact in facts.union({rule for rule in rules}):
                if fact != other_fact:
                    disjunction = f"{fact} ∨ {other_fact}"
                    if disjunction not in facts:
                        new_facts.append((disjunction, f"{fact} → {disjunction}"))
        
        # 二重否定除去（¬¬A → A）
        for fact in facts:
            match = re.match(self.rule_patterns['negation'], fact)
            if match:
                inner = match.group(1).strip()
                inner_match = re.match(self.rule_patterns['negation'], inner)
                if inner_match:
                    double_neg_removed = inner_match.group(1).strip()
                    if double_neg_removed not in facts:
                        new_facts.append((double_neg_removed, f"{fact} → {double_neg_removed}"))
        
        # モーダストレンス（A → B, ¬B → ¬A）
        for rule in rules:
            match = re.match(self.rule_patterns['modus_ponens'], rule)
            if match:
                antecedent, consequent = match.groups()
                antecedent = antecedent.strip()
                consequent = consequent.strip()
                
                # 後件の否定が既知の場合、前件の否定を追加
                neg_consequent = f"¬{consequent}"
                neg_antecedent = f"¬{antecedent}"
                
                if neg_consequent in facts and neg_antecedent not in facts:
                    new_facts.append((neg_antecedent, f"{rule}, {neg_consequent} → {neg_antecedent}"))
        
        return new_facts
    
    def _estimate_confidence(self, known_facts: Set[str], query: str) -> float:
        """
        結論への確信度の推定
        
        Args:
            known_facts (Set[str]): 既知の事実
            query (str): 検証したい結論
            
        Returns:
            float: 確信度 (0-1)
        """
        # 単純な実装例: 結論に近い事実があるほど確信度が高い
        
        # クエリの一部が既知の事実に含まれるか確認
        query_parts = set(re.split(r'[∧∨¬→↔]', query))
        query_parts = {part.strip() for part in query_parts if part.strip()}
        
        # 既知の事実をトークン化
        fact_parts = set()
        for fact in known_facts:
            parts = re.split(r'[∧∨¬→↔]', fact)
            fact_parts.update({part.strip() for part in parts if part.strip()})
        
        # 一致するトークンの比率を計算
        if not query_parts:
            return 0.0
            
        match_count = sum(1 for part in query_parts if part in fact_parts)
        return min(1.0, match_count / len(query_parts))


class InductionEngine:
    """帰納的推論エンジン"""
    
    def __init__(self):
        """帰納エンジンの初期化"""
        self.logger = Logger(__name__)
        self.logger.info("InductionEngine initialized")
    
    def generalize(self, examples: List[Dict[str, Any]], target_concept: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        帰納的推論による一般化
        
        Args:
            examples (List[Dict[str, Any]]): 例のリスト
            target_concept (str): 対象となる概念
            
        Returns:
            Tuple[str, float, Dict[str, Any]]:
                - 一般化ルール
                - 確信度 (0-1)
                - 詳細情報
        """
        self.logger.info(f"Starting induction with {len(examples)} examples")
        
        if not examples:
            return "", 0.0, {}
            
        # 例からの特徴抽出
        features = self._extract_features(examples)
        
        # 共通パターンの特定
        common_patterns = self._find_common_patterns(examples, features)
        
        # 一般化ルールの構築
        rule, confidence = self._build_generalization(common_patterns, target_concept)
        
        # 詳細情報の作成
        details = {
            'feature_count': len(features),
            'common_pattern_count': len(common_patterns),
            'example_count': len(examples),
            'coverage': self._calculate_coverage(rule, examples)
        }
        
        self.logger.info(f"Induction completed with confidence: {confidence:.2f}")
        return rule, confidence, details
    
    def _extract_features(self, examples: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        例から特徴を抽出
        
        Args:
            examples (List[Dict[str, Any]]): 例のリスト
            
        Returns:
            Dict[str, List[Any]]: 特徴名→値のリストのマッピング
        """
        # 全例からユニークな特徴を抽出
        features = {}
        
        for example in examples:
            for key, value in example.items():
                if key not in features:
                    features[key] = []
                    
                if value not in features[key]:
                    features[key].append(value)
        
        return features
    
    def _find_common_patterns(self, examples: List[Dict[str, Any]], features: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        例から共通パターンを特定
        
        Args:
            examples (List[Dict[str, Any]]): 例のリスト
            features (Dict[str, List[Any]]): 抽出された特徴
            
        Returns:
            List[Dict[str, Any]]: 共通パターンのリスト
        """
        # 単純な実装: 各特徴の値の分布を分析
        patterns = []
        
        for feature, values in features.items():
            # 値の出現頻度の計算
            value_counts = {}
            for example in examples:
                if feature in example:
                    value = example[feature]
                    value_counts[value] = value_counts.get(value, 0) + 1
            
            # 閾値以上の頻度を持つ値を共通パターンとして抽出
            threshold = len(examples) * 0.7  # 70%以上の例で出現する値
            common_values = {value for value, count in value_counts.items() if count >= threshold}
            
            if common_values:
                patterns.append({
                    'feature': feature,
                    'common_values': list(common_values),
                    'frequency': max(count / len(examples) for value, count in value_counts.items() 
                                     if value in common_values)
                })
        
        return patterns
    
    def _build_generalization(self, patterns: List[Dict[str, Any]], target_concept: str) -> Tuple[str, float]:
        """
        共通パターンから一般化ルールを構築
        
        Args:
            patterns (List[Dict[str, Any]]): 共通パターン
            target_concept (str): 対象概念
            
        Returns:
            Tuple[str, float]: 一般化ルールと確信度
        """
        if not patterns:
            return "", 0.0
            
        # パターンを重要度（頻度）でソート
        sorted_patterns = sorted(patterns, key=lambda p: p.get('frequency', 0), reverse=True)
        
        # ルール構築（最大3つの重要な特徴を使用）
        rule_parts = []
        total_confidence = 0.0
        
        for pattern in sorted_patterns[:3]:
            feature = pattern['feature']
            values = pattern['common_values']
            
            if len(values) == 1:
                # 単一値の場合は等価性
                rule_parts.append(f"{feature}({values[0]})")
            else:
                # 複数値の場合は選言（OR）
                values_str = ' ∨ '.join([f"{feature}({v})" for v in values])
                rule_parts.append(f"({values_str})")
                
            total_confidence += pattern.get('frequency', 0)
        
        # ルールの構築
        if rule_parts:
            antecedent = ' ∧ '.join(rule_parts)
            rule = f"{antecedent} → {target_concept}"
            
            # 確信度の計算 (特徴の平均頻度を使用)
            confidence = total_confidence / len(sorted_patterns[:3]) if sorted_patterns[:3] else 0.0
            
            return rule, confidence
        
        return "", 0.0
    
    def _calculate_coverage(self, rule: str, examples: List[Dict[str, Any]]) -> float:
        """
        生成したルールが例をどれだけカバーするかを計算
        
        Args:
            rule (str): 生成されたルール
            examples (List[Dict[str, Any]]): 例のリスト
            
        Returns:
            float: カバレッジ (0-1)
        """
        if not rule or not examples:
            return 0.0
            
        # ルールをパース（簡易実装）
        # 実際にはより複雑なパーサーが必要
        try:
            antecedent, consequent = rule.split('→', 1)
            antecedent = antecedent.strip()
            
            # 条件の評価
            covered_count = 0
            for example in examples:
                if self._evaluate_condition(antecedent, example):
                    covered_count += 1
                    
            return covered_count / len(examples)
            
        except Exception as e:
            self.logger.error(f"Error calculating coverage: {str(e)}")
            return 0.0
    
    def _evaluate_condition(self, condition: str, example: Dict[str, Any]) -> bool:
        """
        条件が例に対して真かどうかを評価
        
        Args:
            condition (str): 評価する条件
            example (Dict[str, Any]): 評価対象の例
            
        Returns:
            bool: 条件が真かどうか
        """
        # 簡易的な条件評価（実際にはより複雑なパーサーが必要）
        
        # 連言（AND）の分解
        if '∧' in condition:
            conjuncts = [c.strip() for c in condition.split('∧')]
            return all(self._evaluate_condition(c, example) for c in conjuncts)
            
        # 選言（OR）の分解
        elif '∨' in condition:
            disjuncts = [d.strip() for d in condition.split('∨')]
            return any(self._evaluate_condition(d, example) for d in disjuncts)
            
        # 否定の評価
        elif condition.startswith('¬'):
            negated = condition[1:].strip()
            return not self._evaluate_condition(negated, example)
            
        # 述語の評価（例: feature(value)）
        elif '(' in condition and ')' in condition:
            match = re.match(r'(\w+)\(([^)]*)\)', condition)
            if match:
                feature, value = match.groups()
                
                # 文字列と数値の変換処理
                try:
                    if value.isdigit():
                        value = int(value)
                    elif re.match(r'^[-+]?\d*\.\d+$', value):
                        value = float(value)
                except:
                    pass
                    
                return feature in example and example[feature] == value
                
        # その他の場合（単純なキーの存在チェックなど）
        return condition in example


class ProbabilisticReasoner:
    """確率的推論エンジン"""
    
    def __init__(self):
        """確率的推論エンジンの初期化"""
        self.logger = Logger(__name__)
        self.logger.info("ProbabilisticReasoner initialized")
    
    def update_belief(self, prior: float, evidence: Dict[str, Any], hypothesis: str) -> Tuple[float, Dict[str, float]]:
        """
        ベイズ更新による信念の更新
        
        Args:
            prior (float): 事前確率
            evidence (Dict[str, Any]): 証拠
            hypothesis (str): 仮説
            
        Returns:
            Tuple[float, Dict[str, float]]:
                - 更新された確率
                - 計算の詳細
        """
        self.logger.info(f"Updating belief for hypothesis: {hypothesis}")
        
        # ベイズの定理: P(H|E) = P(E|H) * P(H) / P(E)
        
        # P(E|H)の計算: 仮説が真の場合に証拠が観察される確率
        likelihood = self._calculate_likelihood(evidence, hypothesis)
        
        # P(E)の計算: 証拠が観察される確率
        # P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
        not_h_likelihood = self._calculate_likelihood(evidence, f"¬{hypothesis}")
        marginal = likelihood * prior + not_h_likelihood * (1 - prior)
        
        # ゼロ除算回避
        if marginal == 0:
            posterior = prior  # 更新不可能な場合は事前確率を維持
        else:
            # ベイズの定理による更新
            posterior = (likelihood * prior) / marginal
        
        # 計算の詳細
        details = {
            'prior': prior,
            'likelihood': likelihood,
            'marginal': marginal,
            'posterior': posterior,
            'evidence_count': len(evidence)
        }
        
        self.logger.info(f"Updated belief: {prior:.4f} → {posterior:.4f}")
        return posterior, details
    
    def _calculate_likelihood(self, evidence: Dict[str, Any], hypothesis: str) -> float:
        """
        尤度P(E|H)の計算
        
        Args:
            evidence (Dict[str, Any]): 証拠
            hypothesis (str): 仮説
            
        Returns:
            float: 尤度
        """
        # 単純な実装例: 一様な確率分布を仮定
        # 実際の実装ではより複雑なモデルが必要
        
        # 仮説が否定形の場合
        is_negated = hypothesis.startswith('¬')
        base_hypothesis = hypothesis[1:] if is_negated else hypothesis
        
        # デモ用にいくつかの仮説に対する尤度を定義
        # 本来は学習データから推定する
        predefined_likelihoods = {
            'is_rainy': {'umbrella': 0.9, 'wet_ground': 0.95, 'people_running': 0.7},
            'is_sunny': {'clear_sky': 0.95, 'people_outside': 0.8, 'hot': 0.7},
            'is_workday': {'office_full': 0.9, 'traffic': 0.85, 'meetings': 0.8}
        }
        
        # 該当する仮説の尤度マップを取得
        likelihood_map = predefined_likelihoods.get(base_hypothesis, {})
        
        # 証拠ごとの尤度を計算
        likelihoods = []
        for key, value in evidence.items():
            if key in likelihood_map:
                # 値が真の場合の尤度
                if value:
                    l = likelihood_map[key] if not is_negated else 1 - likelihood_map[key]
                else:
                    l = 1 - likelihood_map[key] if not is_negated else likelihood_map[key]
                likelihoods.append(l)
            else:
                # 未知の証拠に対する中立的な値
                likelihoods.append(0.5)
        
        # 証拠がない場合のフォールバック
        if not likelihoods:
            return 0.5
            
        # ナイーブベイズの仮定: 条件付き独立性
        # P(E1,E2,...|H) = P(E1|H) * P(E2|H) * ...
        combined_likelihood = 1.0
        for l in likelihoods:
            combined_likelihood *= l
            
        return combined_likelihood
    
    def naive_bayes(self, features: Dict[str, Any], class_priors: Dict[str, float], 
                   feature_probs: Dict[str, Dict[str, Dict[Any, float]]]) -> Dict[str, float]:
        """
        ナイーブベイズ分類
        
        Args:
            features (Dict[str, Any]): 分類する特徴
            class_priors (Dict[str, float]): クラスの事前確率
            feature_probs (Dict[str, Dict[str, Dict[Any, float]]]): 特徴の条件付き確率
            
        Returns:
            Dict[str, float]: クラスごとの事後確率
        """
        posteriors = {}
        
        for class_name, prior in class_priors.items():
            # クラスごとの事後確率の計算
            posterior = prior
            
            # 各特徴の条件付き確率を乗算
            for feature_name, feature_value in features.items():
                if (feature_name in feature_probs and 
                    class_name in feature_probs[feature_name] and
                    feature_value in feature_probs[feature_name][class_name]):
                    
                    conditional_prob = feature_probs[feature_name][class_name][feature_value]
                    posterior *= conditional_prob
                else:
                    # 未知の特徴値の場合のスムージング
                    posterior *= 0.01
            
            posteriors[class_name] = posterior
            
        # 正規化
        total = sum(posteriors.values())
        if total > 0:
            for class_name in posteriors:
                posteriors[class_name] /= total
                
        return posteriors


class AbductionEngine:
    """アブダクション（最良説明）推論エンジン"""
    
    def __init__(self):
        """アブダクションエンジンの初期化"""
        self.logger = Logger(__name__)
        self.logger.info("AbductionEngine initialized")
    
    def infer_best_explanation(self, observations: List[str], hypotheses: List[str], 
                              knowledge_base: List[Dict[str, Any]]) -> Tuple[str, float, List[str]]:
        """
        観察に対する最良の説明を推論
        
        Args:
            observations (List[str]): 観察された事実
            hypotheses (List[str]): 候補となる仮説
            knowledge_base (List[Dict[str, Any]]): 知識ベース
            
        Returns:
            Tuple[str, float, List[str]]:
                - 最良の説明
                - スコア
                - 説明の詳細ステップ
        """
        self.logger.info(f"Inferring best explanation from {len(hypotheses)} hypotheses")
        
        if not observations or not hypotheses:
            return "", 0.0, []
            
        # 仮説の評価
        scored_hypotheses = []
        for hypothesis in hypotheses:
            score, explanation = self._evaluate_hypothesis(hypothesis, observations, knowledge_base)
            scored_hypotheses.append((hypothesis, score, explanation))
            
        # 最良の仮説を選択
        if scored_hypotheses:
            scored_hypotheses.sort(key=lambda x: x[1], reverse=True)
            best_hypothesis, best_score, best_explanation = scored_hypotheses[0]
            
            self.logger.info(f"Best explanation: {best_hypothesis} with score {best_score:.4f}")
            return best_hypothesis, best_score, best_explanation
            
        return "", 0.0, []
    
    def _evaluate_hypothesis(self, hypothesis: str, observations: List[str], 
                            knowledge_base: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        """
        仮説の評価
        
        Args:
            hypothesis (str): 評価する仮説
            observations (List[str]): 観察された事実
            knowledge_base (List[Dict[str, Any]]): 知識ベース
            
        Returns:
            Tuple[float, List[str]]:
                - 評価スコア
                - 説明の詳細ステップ
        """
        # 仮説から予測される観察の導出
        predicted_observations, steps = self._derive_predictions(hypothesis, knowledge_base)
        
        # 説明力の評価: 仮説がどれだけ観察を説明できるか
        explained_count = sum(1 for obs in observations if obs in predicted_observations)
        explanatory_power = explained_count / len(observations) if observations else 0
        
        # 過剰な予測のペナルティ: 観察されていない予測が多いほど低スコア
        unexplained_predictions = len(predicted_observations) - explained_count
        penalty = 0.1 * (unexplained_predictions / len(predicted_observations) if predicted_observations else 0)
        
        # 仮説の複雑さのペナルティ: 単純な仮説ほど優先（オッカムの剃刀）
        complexity_penalty = 0.05 * len(hypothesis) / 100  # 仮説の長さに基づく簡易指標
        
        # 総合スコア
        score = explanatory_power - penalty - complexity_penalty
        
        # 説明ステップの拡張
        explanation = steps.copy()
        explanation.append(f"説明力: {explanatory_power:.2f} ({explained_count}/{len(observations)} 観察を説明)")
        explanation.append(f"過剰予測ペナルティ: {penalty:.2f} ({unexplained_predictions} 未観測予測)")
        explanation.append(f"複雑さペナルティ: {complexity_penalty:.2f}")
        explanation.append(f"総合スコア: {score:.2f}")
        
        return score, explanation
    
    def _derive_predictions(self, hypothesis: str, knowledge_base: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        仮説から予測される観察の導出
        
        Args:
            hypothesis (str): 仮説
            knowledge_base (List[Dict[str, Any]]): 知識ベース
            
        Returns:
            Tuple[List[str], List[str]]:
                - 予測される観察
                - 導出ステップ
        """
        predictions = []
        steps = [f"仮説: {hypothesis}"]
        
        # 知識ベースからのルール抽出
        rules = [item['rule'] for item in knowledge_base if 'rule' in item]
        
        # 仮説を適用可能なルールの特定
        for rule in rules:
            if '→' in rule:
                antecedent, consequent = rule.split('→', 1)
                antecedent = antecedent.strip()
                consequent = consequent.strip()
                
                # 仮説がルールの前件に一致または含まれる場合
                if hypothesis == antecedent or hypothesis in antecedent:
                    predictions.append(consequent)
                    steps.append(f"ルール適用: {rule} → 予測: {consequent}")
                    
                # 仮説が複合条件の一部として含まれる場合
                elif '∧' in antecedent and hypothesis in antecedent.split('∧'):
                    partial_match = True
                    for condition in antecedent.split('∧'):
                        condition = condition.strip()
                        if condition != hypothesis and condition not in predictions:
                            partial_match = False
                            break
                            
                    if partial_match:
                        predictions.append(consequent)
                        steps.append(f"部分条件一致: {rule} → 予測: {consequent}")
        
        # 循環的な推論（1ステップのみ）
        initial_predictions = predictions.copy()
        for pred in initial_predictions:
            for rule in rules:
                if '→' in rule:
                    antecedent, consequent = rule.split('→', 1)
                    antecedent = antecedent.strip()
                    consequent = consequent.strip()
                    
                    if pred == antecedent and consequent not in predictions:
                        predictions.append(consequent)
                        steps.append(f"二次推論: {pred} → {consequent}")
        
        return predictions, steps


class SymbolicInferenceEngine:
    """シンボリックAI推論エンジンの統合クラス"""
    
    def __init__(self, kb_interface: SymbolicKnowledgeInterface):
        """
        推論エンジンの初期化
        
        Args:
            kb_interface (SymbolicKnowledgeInterface): 知識ベースインターフェース
        """
        self.logger = Logger(__name__)
        self.kb = kb_interface
        
        # 推論コンポーネント
        self.deduction = DeductionEngine()
        self.induction = InductionEngine()
        self.probabilistic = ProbabilisticReasoner()
        self.abduction = AbductionEngine()
        
        self.logger.info("SymbolicInferenceEngine initialized")
    
    def deduce(self, premises: List[str], query: str) -> Dict[str, Any]:
        """
        演繹的推論を実行
        
        Args:
            premises (List[str]): 前提となる命題のリスト
            query (str): 検証したい結論
            
        Returns:
            Dict[str, Any]: 推論結果
        """
        self.logger.info(f"Deduction request: {query}")
        
        # 関連知識の取得
        knowledge_context = self.kb.get_relevant_knowledge(premises + [query])
        
        # 演繹的推論の実行
        result, confidence, steps = self.deduction.infer(premises, query, knowledge_context)
        
        # 結果の構築
        inference_result = {
            'query': query,
            'result': result,
            'confidence': confidence,
            'steps': steps,
            'premises_count': len(premises),
            'knowledge_items_used': len(knowledge_context)
        }
        
        return inference_result
    
    def induce(self, examples: List[Dict[str, Any]], target_concept: str) -> Dict[str, Any]:
        """
        帰納的推論を実行
        
        Args:
            examples (List[Dict[str, Any]]): 例のリスト
            target_concept (str): 対象となる概念
            
        Returns:
            Dict[str, Any]: 推論結果
        """
        self.logger.info(f"Induction request for concept: {target_concept}")
        
        # 帰納的推論の実行
        rule, confidence, details = self.induction.generalize(examples, target_concept)
        
        # 生成されたルールの知識ベースへの追加（オプション）
        if rule and confidence > 0.7:  # 高確信度のルールのみ追加
            metadata = {
                'source': 'induction',
                'confidence': confidence,
                'example_count': len(examples),
                'target_concept': target_concept
            }
            rule_id = self.kb.store_logical_rule(rule, metadata)
            details['rule_id'] = rule_id
        
        # 結果の構築
        inference_result = {
            'target_concept': target_concept,
            'rule': rule,
            'confidence': confidence,
            'details': details,
            'example_count': len(examples)
        }
        
        return inference_result
    
    def reason_with_uncertainty(self, evidence: Dict[str, Any], hypothesis: str) -> Dict[str, Any]:
        """
        確率的推論を実行
        
        Args:
            evidence (Dict[str, Any]): 証拠
            hypothesis (str): 仮説
            
        Returns:
            Dict[str, Any]: 推論結果
        """
        self.logger.info(f"Probabilistic reasoning for hypothesis: {hypothesis}")
        
        # 事前確率の取得（知識ベースから）
        # 単純な実装ではデフォルト値を使用
        prior = 0.5  # 中立的な事前確率
        
        # 確率的推論の実行
        posterior, details = self.probabilistic.update_belief(prior, evidence, hypothesis)
        
        # 結果の構築
        inference_result = {
            'hypothesis': hypothesis,
            'prior': prior,
            'posterior': posterior,
            'evidence_count': len(evidence),
            'details': details
        }
        
        return inference_result
    
    def find_best_explanation(self, observations: List[str]) -> Dict[str, Any]:
        """
        アブダクション（最良説明）推論を実行
        
        Args:
            observations (List[str]): 観察された事実
            
        Returns:
            Dict[str, Any]: 推論結果
        """
        self.logger.info(f"Abduction request for {len(observations)} observations")
        
        # 関連知識の取得
        knowledge_context = self.kb.get_relevant_knowledge(observations)
        
        # 候補仮説の生成/取得
        hypotheses = self._generate_hypotheses(observations, knowledge_context)
        
        # 最良説明の推論
        best_hypothesis, score, explanation = self.abduction.infer_best_explanation(
            observations, hypotheses, knowledge_context
        )
        
        # 結果の構築
        inference_result = {
            'best_explanation': best_hypothesis,
            'score': score,
            'explanation_steps': explanation,
            'observation_count': len(observations),
            'hypothesis_count': len(hypotheses),
            'knowledge_items_used': len(knowledge_context)
        }
        
        return inference_result
    
    def _generate_hypotheses(self, observations: List[str], knowledge_context: List[Dict[str, Any]]) -> List[str]:
        """
        観察に基づく候補仮説の生成
        
        Args:
            observations (List[str]): 観察された事実
            knowledge_context (List[Dict[str, Any]]): 知識コンテキスト
            
        Returns:
            List[str]: 候補仮説のリスト
        """
        hypotheses = []
        
        # 知識ベースからのルール抽出
        rules = [item['rule'] for item in knowledge_context if 'rule' in item]
        
        # 観察を説明しうるルールの特定
        for rule in rules:
            if '→' in rule:
                antecedent, consequent = rule.split('→', 1)
                antecedent = antecedent.strip()
                consequent = consequent.strip()
                
                # ルールの後件が観察と一致する場合、前件を仮説候補とする
                for obs in observations:
                    if obs == consequent or obs in consequent:
                        if antecedent not in hypotheses:
                            hypotheses.append(antecedent)
        
        # 知識が不十分な場合のフォールバック
        if not hypotheses:
            # 単純な仮説を生成
            for obs in observations:
                if '(' in obs and ')' in obs:
                    # 述語形式の観察からの仮説生成
                    match = re.match(r'(\w+)\(([^)]*)\)', obs)
                    if match:
                        predicate, args = match.groups()
                        # 原因となりうる述語の推測
                        cause_predicates = {
                            'Wet': 'Rainy',
                            'Hot': 'Sunny',
                            'Cold': 'Winter',
                            'Green': 'Spring',
                            'ColorChange': 'Autumn'
                        }
                        if predicate in cause_predicates:
                            hypotheses.append(f"{cause_predicates[predicate]}({args})")
                else:
                    # 単純な観察からの仮説生成
                    hypotheses.append(f"Cause({obs})")
        
        return hypotheses
    
    def check_consistency(self, statements: List[str]) -> Dict[str, Any]:
        """
        論理的一貫性の検証
        
        Args:
            statements (List[str]): 検証する命題のリスト
            
        Returns:
            Dict[str, Any]: 一貫性検証結果
        """
        self.logger.info(f"Consistency check for {len(statements)} statements")
        
        # 矛盾の検出（簡易実装）
        contradictions = []
        
        # 命題とその否定の検出
        for i, stmt1 in enumerate(statements):
            for stmt2 in statements[i+1:]:
                if stmt2 == f"¬{stmt1}" or stmt1 == f"¬{stmt2}":
                    contradictions.append((stmt1, stmt2))
        
        # 推論による矛盾の検出
        inferred_contradictions = []
        
        for i, stmt in enumerate(statements):
            remaining = statements[:i] + statements[i+1:]
            # 命題の否定が残りの命題から導出可能か確認
            negation = f"¬{stmt}"
            result, confidence, steps = self.deduction.infer(remaining, negation, [])
            if result and confidence > 0.8:
                inferred_contradictions.append((stmt, negation, steps))
        
        # 結果の構築
        is_consistent = len(contradictions) == 0 and len(inferred_contradictions) == 0
        
        result = {
            'is_consistent': is_consistent,
            'direct_contradictions': contradictions,
            'inferred_contradictions': inferred_contradictions,
            'statement_count': len(statements)
        }
        
        return result
    
    def analogy_reasoning(self, source_domain: Dict[str, Any], target_domain: Dict[str, Any], 
                         relation_to_map: str) -> Dict[str, Any]:
        """
        類推的推論の実行
        
        Args:
            source_domain (Dict[str, Any]): ソースドメイン（既知の領域）
            target_domain (Dict[str, Any]): ターゲットドメイン（推論対象の領域）
            relation_to_map (str): マッピングする関係
            
        Returns:
            Dict[str, Any]: 推論結果
        """
        self.logger.info(f"Analogy reasoning for relation: {relation_to_map}")
        
        # ドメイン間の対応関係の特定
        mappings = self._find_domain_mappings(source_domain, target_domain)
        
        # 関係のマッピング
        mapped_relation = None
        confidence = 0.0
        mapping_explanation = []
        
        if mappings:
            # 最も確信度の高いマッピングを使用
            best_mapping = max(mappings, key=lambda m: m.get('confidence', 0))
            
            # 関係のマッピング
            if relation_to_map in source_domain.get('relations', {}):
                relation_info = source_domain['relations'][relation_to_map]
                source_entities = relation_info.get('entities', [])
                relation_type = relation_info.get('type', '')
                
                # 対応するターゲットエンティティの特定
                target_entities = []
                for src_entity in source_entities:
                    for mapping in best_mapping.get('entity_mappings', []):
                        if mapping.get('source') == src_entity:
                            target_entities.append(mapping.get('target'))
                            mapping_explanation.append(
                                f"{src_entity} → {mapping.get('target')} "
                                f"(類似度: {mapping.get('similarity', 0):.2f})"
                            )
                
                # マッピングされた関係の構築
                if target_entities and len(target_entities) == len(source_entities):
                    mapped_relation = {
                        'type': relation_type,
                        'entities': target_entities
                    }
                    
                    # 確信度の算出
                    mapping_confidences = [m.get('similarity', 0) 
                                         for m in best_mapping.get('entity_mappings', [])]
                    confidence = sum(mapping_confidences) / len(mapping_confidences) if mapping_confidences else 0
        
        # 結果の構築
        result = {
            'source_relation': relation_to_map,
            'mapped_relation': mapped_relation,
            'confidence': confidence,
            'mapping_explanation': mapping_explanation,
            'mapping_count': len(mappings)
        }
        
        return result
    
    def _find_domain_mappings(self, source_domain: Dict[str, Any], target_domain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ドメイン間の対応関係を特定
        
        Args:
            source_domain (Dict[str, Any]): ソースドメイン
            target_domain (Dict[str, Any]): ターゲットドメイン
            
        Returns:
            List[Dict[str, Any]]: 可能なマッピングのリスト
        """
        mappings = []
        
        # 簡易実装: エンティティの属性に基づく類似性計算
        source_entities = source_domain.get('entities', {})
        target_entities = target_domain.get('entities', {})
        
        if not source_entities or not target_entities:
            return mappings
            
        # 可能なマッピングの列挙（全ての組み合わせは計算量が大きいため、簡易的なアプローチ）
        entity_mappings = []
        
        for src_name, src_entity in source_entities.items():
            best_match = None
            best_similarity = 0
            
            for tgt_name, tgt_entity in target_entities.items():
                # 属性に基づく類似度計算
                similarity = self._calculate_entity_similarity(src_entity, tgt_entity)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'source': src_name,
                        'target': tgt_name,
                        'similarity': similarity
                    }
            
            if best_match and best_similarity > 0.3:  # 最低類似度閾値
                entity_mappings.append(best_match)
        
        # 全体的なマッピングの構築
        if entity_mappings:
            overall_confidence = sum(m.get('similarity', 0) for m in entity_mappings) / len(entity_mappings)
            
            mappings.append({
                'entity_mappings': entity_mappings,
                'confidence': overall_confidence
            })
        
        return mappings
    
    def _calculate_entity_similarity(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """
        エンティティ間の類似度を計算
        
        Args:
            entity1 (Dict[str, Any]): 1つ目のエンティティ
            entity2 (Dict[str, Any]): 2つ目のエンティティ
            
        Returns:
            float: 類似度 (0-1)
        """
        # 共通の属性を特定
        common_attrs = set(entity1.keys()) & set(entity2.keys())
        
        if not common_attrs:
            return 0.0
            
        # 属性値の類似度を計算
        similarities = []
        
        for attr in common_attrs:
            val1 = entity1[attr]
            val2 = entity2[attr]
            
            # 型に応じた類似度計算
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数値の類似度
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarities.append(1.0 - min(1.0, abs(val1 - val2) / max_val))
                else:
                    similarities.append(1.0)  # 両方とも0の場合
            elif isinstance(val1, str) and isinstance(val2, str):
                # 文字列の類似度（単純なアプローチ）
                total_chars = len(val1) + len(val2)
                if total_chars > 0:
                    # 共通文字の割合
                    common_chars = sum(1 for c in set(val1) if c in val2)
                    similarities.append(2 * common_chars / total_chars)
                else:
                    similarities.append(1.0)  # 両方とも空文字の場合
            else:
                # その他の型: 等価性のみ判定
                similarities.append(1.0 if val1 == val2 else 0.0)
        
        # 平均類似度
        return sum(similarities) / len(similarities) if similarities else 0.0
