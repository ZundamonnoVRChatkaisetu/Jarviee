"""
論理表現変換モジュール

このモジュールは自然言語とシンボリックAIの論理表現の間の変換を担当します。
LLMを活用して自然言語から形式的な論理表現への変換、および逆方向の変換を行います。
"""

import os
import sys
import re
import json
from typing import Dict, List, Any, Union, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.core.utils.logger import Logger
from src.core.llm.engine import LLMEngine


class LogicNormalizer:
    """論理表現の正規化を行うクラス"""
    
    def __init__(self):
        """論理正規化器の初期化"""
        self.logger = Logger(__name__)
        
        # 論理演算子のマッピング（様々な表記を標準形式に統一）
        self.operator_map = {
            # 論理積（AND）
            '&': '∧',
            'and': '∧',
            '&&': '∧',
            '.': '∧',
            
            # 論理和（OR）
            '|': '∨',
            'or': '∨',
            '||': '∨',
            '+': '∨',
            
            # 否定（NOT）
            '!': '¬',
            'not': '¬',
            '~': '¬',
            
            # 含意（IMPLIES）
            '->': '→',
            '=>': '→',
            'implies': '→',
            
            # 同値（IFF）
            '<->': '↔',
            '<=>': '↔',
            'iff': '↔',
            
            # 全称量化子（FORALL）
            'forall': '∀',
            'all': '∀',
            
            # 存在量化子（EXISTS）
            'exists': '∃',
            'some': '∃'
        }
        
        # 標準形式の演算子
        self.std_operators = {'∧', '∨', '¬', '→', '↔', '∀', '∃'}
        
        # 演算子の優先順位
        self.precedence = {
            '¬': 4,  # 否定
            '∧': 3,  # 論理積
            '∨': 2,  # 論理和
            '→': 1,  # 含意
            '↔': 0   # 同値
        }
        
        self.logger.info("LogicNormalizer initialized")
    
    def normalize(self, logic_expr: str) -> str:
        """
        論理表現を正規化
        
        Args:
            logic_expr (str): 入力論理表現
            
        Returns:
            str: 正規化された論理表現
        """
        if not logic_expr or not isinstance(logic_expr, str):
            self.logger.warning(f"Invalid input for normalization: {logic_expr}")
            return ""
        
        # 前処理: スペースの正規化と不要な文字の削除
        expr = logic_expr.strip()
        
        # 演算子の標準化
        for op, std_op in self.operator_map.items():
            # 単語境界を考慮（例：'and'は単語として置換、'candy'の一部として置換しない）
            if len(op) > 1 and op.isalpha():
                expr = re.sub(r'\b' + re.escape(op) + r'\b', std_op, expr, flags=re.IGNORECASE)
            else:
                expr = expr.replace(op, std_op)
        
        # カッコの正規化
        expr = self._normalize_parentheses(expr)
        
        # 冗長な空白の削除
        expr = re.sub(r'\s+', ' ', expr)
        
        # 変数名の正規化（必要に応じて）
        
        self.logger.debug(f"Normalized logic expression: {expr}")
        return expr
    
    def _normalize_parentheses(self, expr: str) -> str:
        """
        括弧の正規化
        
        Args:
            expr (str): 入力表現
            
        Returns:
            str: 括弧が正規化された表現
        """
        # 括弧の対応チェック
        open_count = expr.count('(')
        close_count = expr.count(')')
        
        if open_count != close_count:
            self.logger.warning(f"Unbalanced parentheses in expression: {expr}")
            # 不足している括弧を追加
            if open_count > close_count:
                expr += ')' * (open_count - close_count)
            else:
                expr = '(' * (close_count - open_count) + expr
        
        return expr
    
    def to_cnf(self, logic_expr: str) -> str:
        """
        論理式を連言標準形(CNF)に変換
        
        Args:
            logic_expr (str): 入力論理式
            
        Returns:
            str: CNF形式の論理式
        """
        # TODO: CNF変換ロジックの実装
        # 現段階では基本的な正規化のみ実施
        normalized = self.normalize(logic_expr)
        
        self.logger.info("CNF conversion is currently a placeholder")
        return normalized
    
    def to_dnf(self, logic_expr: str) -> str:
        """
        論理式を選言標準形(DNF)に変換
        
        Args:
            logic_expr (str): 入力論理式
            
        Returns:
            str: DNF形式の論理式
        """
        # TODO: DNF変換ロジックの実装
        # 現段階では基本的な正規化のみ実施
        normalized = self.normalize(logic_expr)
        
        self.logger.info("DNF conversion is currently a placeholder")
        return normalized
    
    def is_satisfiable(self, logic_expr: str) -> bool:
        """
        論理式の充足可能性を判定
        
        Args:
            logic_expr (str): 入力論理式
            
        Returns:
            bool: 充足可能かどうか
        """
        # TODO: SAT解析ロジックの実装
        # 現段階ではデフォルト値を返す
        self.logger.info("Satisfiability check is currently a placeholder")
        return True


class LogicTransformer:
    """自然言語と論理表現の間の変換を行うクラス"""
    
    def __init__(self, llm_engine: LLMEngine):
        """
        論理変換器の初期化
        
        Args:
            llm_engine (LLMEngine): LLM処理エンジン
        """
        self.logger = Logger(__name__)
        self.llm = llm_engine
        self.normalizer = LogicNormalizer()
        
        # プロンプトテンプレート
        self.nl_to_logic_template = """
        あなたは自然言語を形式論理に変換する専門家です。
        以下の文を形式論理に変換してください。
        
        使用する記号:
        - 論理積 (AND): ∧
        - 論理和 (OR): ∨
        - 否定 (NOT): ¬
        - 含意 (IMPLIES): →
        - 同値 (IFF): ↔
        - 全称量化子 (FORALL): ∀
        - 存在量化子 (EXISTS): ∃
        
        変換する際は、述語論理の形式を使用し、一貫性のある変数名を選んでください。
        最終的な論理式のみを出力してください。
        
        文: {text}
        
        論理式:
        """
        
        self.logic_to_nl_template = """
        あなたは形式論理を自然な日本語に変換する専門家です。
        以下の論理式を平易な日本語に変換してください。
        
        形式論理の記号:
        - ∧: かつ (AND)
        - ∨: または (OR)
        - ¬: ではない (NOT)
        - →: ならば (IMPLIES)
        - ↔: 同値 (IFF)
        - ∀: すべての (FORALL)
        - ∃: 存在する (EXISTS)
        
        できるだけ自然な日本語になるように変換し、論理的な意味を正確に保ってください。
        
        論理式: {logic_expr}
        
        日本語:
        """
        
        self.logger.info("LogicTransformer initialized")
    
    def natural_to_logic(self, text: str) -> str:
        """
        自然言語テキストから論理表現への変換
        
        Args:
            text (str): 自然言語テキスト
            
        Returns:
            str: 論理表現
        """
        if not text:
            self.logger.warning("Empty text input for natural_to_logic")
            return ""
        
        # LLMを使用して初期変換
        prompt = self.nl_to_logic_template.format(text=text)
        
        try:
            raw_logic = self.llm.generate(prompt, max_tokens=200)
            
            # 出力から論理式のみを抽出
            raw_logic = raw_logic.strip()
            
            # 論理式の正規化
            normalized = self.normalizer.normalize(raw_logic)
            
            self.logger.info(f"Converted text to logic: {normalized}")
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error converting natural language to logic: {str(e)}")
            return ""
    
    def logic_to_natural(self, logic_expr: str) -> str:
        """
        論理表現から自然言語への変換
        
        Args:
            logic_expr (str): 論理表現
            
        Returns:
            str: 自然言語テキスト
        """
        if not logic_expr:
            self.logger.warning("Empty logic expression input for logic_to_natural")
            return ""
        
        # 入力の正規化
        normalized_expr = self.normalizer.normalize(logic_expr)
        
        # LLMを使用して自然言語に変換
        prompt = self.logic_to_nl_template.format(logic_expr=normalized_expr)
        
        try:
            natural_text = self.llm.generate(prompt, max_tokens=300)
            
            # 出力から日本語部分のみを抽出
            natural_text = natural_text.strip()
            
            self.logger.info(f"Converted logic to natural language: {natural_text}")
            return natural_text
            
        except Exception as e:
            self.logger.error(f"Error converting logic to natural language: {str(e)}")
            return ""
    
    def extract_predicates(self, logic_expr: str) -> List[str]:
        """
        論理式から述語を抽出
        
        Args:
            logic_expr (str): 論理表現
            
        Returns:
            List[str]: 述語のリスト
        """
        if not logic_expr:
            return []
        
        # 正規化
        normalized = self.normalizer.normalize(logic_expr)
        
        # 述語パターンの抽出 (例: P(x), Q(x,y) など)
        predicates = re.findall(r'[A-Z][a-zA-Z]*\([^)]*\)', normalized)
        
        return list(set(predicates))  # 重複除去
    
    def extract_variables(self, logic_expr: str) -> List[str]:
        """
        論理式から変数を抽出
        
        Args:
            logic_expr (str): 論理表現
            
        Returns:
            List[str]: 変数のリスト
        """
        if not logic_expr:
            return []
        
        # 正規化
        normalized = self.normalizer.normalize(logic_expr)
        
        # 述語内の変数を抽出
        # まず述語パターンを見つける
        predicates = re.findall(r'[A-Z][a-zA-Z]*\(([^)]*)\)', normalized)
        
        # 各述語から変数を抽出
        variables = []
        for pred_vars in predicates:
            # カンマで区切られた変数を分割
            vars_in_pred = [v.strip() for v in pred_vars.split(',')]
            variables.extend(vars_in_pred)
        
        # 量化子内の変数も抽出 (例: ∀x, ∃y など)
        quantified_vars = re.findall(r'[∀∃]\s*([a-z][a-zA-Z0-9]*)', normalized)
        variables.extend(quantified_vars)
        
        return list(set(variables))  # 重複除去
    
    def convert_to_prolog(self, logic_expr: str) -> str:
        """
        論理式をPrologコードに変換
        
        Args:
            logic_expr (str): 論理表現
            
        Returns:
            str: Prologコード
        """
        # 自然言語→論理→Prologの変換は複雑なため、現段階では簡易実装
        self.logger.info("Prolog conversion is currently a placeholder")
        
        # 正規化
        normalized = self.normalizer.normalize(logic_expr)
        
        # 簡易変換（完全な実装ではない）
        prolog_code = normalized.replace('∧', ',')
        prolog_code = prolog_code.replace('∨', ';')
        prolog_code = prolog_code.replace('¬', 'not ')
        prolog_code = prolog_code.replace('→', ':-')
        
        # 量化子の処理はより複雑なため省略
        
        return prolog_code
    
    def generate_rule_based_logic(self, domain_description: str) -> str:
        """
        ドメイン記述からルールベースの論理を生成
        
        Args:
            domain_description (str): ドメインの説明文
            
        Returns:
            str: 生成された論理ルール
        """
        prompt = f"""
        あなたは自然言語からルールベースの論理を抽出する専門家です。
        以下のドメイン説明文から、形式論理のルールセットを生成してください。

        ドメイン説明:
        {domain_description}

        形式論理のルールセットとして出力してください。
        各ルールは以下の形式で表現してください:
        - 述語論理形式
        - 使用する記号: ∧(AND), ∨(OR), ¬(NOT), →(IMPLIES), ↔(IFF), ∀(FORALL), ∃(EXISTS)
        - 変数と述語は意味がわかりやすい名前を使用

        ルールセット:
        """
        
        try:
            logic_rules = self.llm.generate(prompt, max_tokens=500)
            return logic_rules.strip()
        except Exception as e:
            self.logger.error(f"Error generating rule-based logic: {str(e)}")
            return ""
