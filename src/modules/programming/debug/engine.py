"""
デバッグエンジンモジュール

エラー診断・修正のためのデバッグエンジン実装
"""

import os
import re
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# プロジェクトルートへのパス追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.core.llm.engine import LLMEngine
from src.modules.programming.code_analyzer import CodeAnalyzer
from src.core.knowledge.query_engine import QueryEngine
from src.core.utils.config import Config

# ロガー設定
logger = logging.getLogger(__name__)


class ErrorPattern:
    """エラーパターンを表現するクラス"""
    
    def __init__(
        self, 
        name: str, 
        regex_pattern: str, 
        description: str, 
        common_causes: List[str],
        solutions: List[str]
    ):
        """
        エラーパターンを初期化
        
        Args:
            name: エラーパターンの名前
            regex_pattern: エラーメッセージにマッチする正規表現パターン
            description: エラーの説明
            common_causes: 一般的な原因のリスト
            solutions: 解決策のリスト
        """
        self.name = name
        self.regex = re.compile(regex_pattern, re.MULTILINE | re.DOTALL)
        self.description = description
        self.common_causes = common_causes
        self.solutions = solutions
    
    def matches(self, error_message: str) -> bool:
        """
        エラーメッセージがこのパターンにマッチするか確認
        
        Args:
            error_message: チェックするエラーメッセージ
            
        Returns:
            bool: マッチすればTrue
        """
        return bool(self.regex.search(error_message))
    
    def to_dict(self) -> Dict:
        """
        辞書形式に変換
        
        Returns:
            Dict: エラーパターン情報を含む辞書
        """
        return {
            "name": self.name,
            "description": self.description,
            "common_causes": self.common_causes,
            "solutions": self.solutions
        }


class DebugEngine:
    """プログラミング言語のデバッグを支援するエンジン"""
    
    def __init__(
        self, 
        llm_engine: Optional[LLMEngine] = None,
        code_analyzer: Optional[CodeAnalyzer] = None,
        query_engine: Optional[QueryEngine] = None,
        config: Optional[Config] = None
    ):
        """
        デバッグエンジンを初期化
        
        Args:
            llm_engine: 使用するLLMエンジン（オプション）
            code_analyzer: コード解析エンジン（オプション）
            query_engine: 知識クエリエンジン（オプション）
            config: 設定オブジェクト（オプション）
        """
        self.llm_engine = llm_engine
        self.code_analyzer = code_analyzer
        self.query_engine = query_engine
        self.config = config or Config()
        
        # エラーパターンデータベースの初期化
        self.error_patterns = self._load_error_patterns()
        
        logger.info("デバッグエンジンが初期化されました")
    
    def _load_error_patterns(self) -> Dict[str, Dict[str, List[ErrorPattern]]]:
        """
        エラーパターンデータベースを読み込み
        
        Returns:
            Dict: 言語ごとのエラーパターンマップ
        """
        # 実際の実装では外部ファイルやDBから読み込むが、
        # プロトタイプでは一部のパターンをハードコード
        python_syntax_patterns = [
            ErrorPattern(
                "SyntaxError: invalid syntax",
                r"SyntaxError: invalid syntax",
                "Python構文エラー：無効な構文",
                [
                    "括弧、括弧、引用符などの不一致",
                    "不正な演算子の使用",
                    "キーワードの不適切な使用"
                ],
                [
                    "エラー位置の前後のコードを確認する",
                    "括弧、括弧、引用符のペアを確認する",
                    "Python構文リファレンスを参照する"
                ]
            ),
            ErrorPattern(
                "IndentationError: expected an indented block",
                r"IndentationError: expected an indented block",
                "インデントエラー：インデントされたブロックが必要",
                [
                    "コロン(:)の後にインデントが不足している",
                    "if、for、whileなどの後に適切なインデントがない",
                    "タブとスペースの混在"
                ],
                [
                    "コロン(:)の後の行を適切にインデントする",
                    "一貫したインデント方法（スペースまたはタブ）を使用する",
                    "PEP 8に従って4スペースのインデントを使用する"
                ]
            )
        ]
        
        python_runtime_patterns = [
            ErrorPattern(
                "NameError: name is not defined",
                r"NameError: name '([^']+)' is not defined",
                "名前エラー：変数または関数名が定義されていない",
                [
                    "変数を使用前に定義していない",
                    "変数名のタイプミス",
                    "スコープ外での変数アクセス"
                ],
                [
                    "変数を使用前に定義する",
                    "変数名のスペルを確認する",
                    "グローバル変数を使用する場合はglobal宣言を使用する"
                ]
            ),
            ErrorPattern(
                "TypeError: unsupported operand type(s)",
                r"TypeError: unsupported operand type\(s\) for ([^:]+): '([^']+)' and '([^']+)'",
                "型エラー：サポートされていないオペランド型",
                [
                    "互換性のない型間での演算",
                    "文字列と数値の不正な連結",
                    "メソッドへの誤った型の引数"
                ],
                [
                    "型を確認し、必要に応じて変換する",
                    "文字列連結には+ではなく、formatまたはf文字列を使用する",
                    "変数の型をprint(type(var))で確認する"
                ]
            )
        ]
        
        javascript_patterns = [
            ErrorPattern(
                "ReferenceError: variable is not defined",
                r"ReferenceError: ([^ ]+) is not defined",
                "参照エラー：変数が定義されていない",
                [
                    "変数を使用前に定義していない",
                    "変数名のタイプミス",
                    "スコープ外での変数アクセス"
                ],
                [
                    "変数を使用前に宣言する（let、const、var）",
                    "変数名のスペルを確認する",
                    "変数のスコープを確認する"
                ]
            ),
            ErrorPattern(
                "SyntaxError: Unexpected token",
                r"SyntaxError: Unexpected token ([^ ]+)",
                "構文エラー：予期しないトークン",
                [
                    "括弧、括弧、引用符などの不一致",
                    "JSONの構文エラー",
                    "不正な演算子の使用"
                ],
                [
                    "構文をチェックし、括弧のバランスを確認する",
                    "JSONデータの場合、JSON.parse前に有効なJSONかを確認する",
                    "コードの整形ツールを使用する"
                ]
            )
        ]
        
        return {
            "python": {
                "syntax": python_syntax_patterns,
                "runtime": python_runtime_patterns
            },
            "javascript": {
                "general": javascript_patterns
            }
        }
    
    def diagnose_error(
        self, 
        error_message: str, 
        language: str, 
        code: Optional[str] = None
    ) -> Dict:
        """
        エラーメッセージを診断し、原因と解決策を提案
        
        Args:
            error_message: 診断するエラーメッセージ
            language: プログラミング言語（'python'、'javascript'など）
            code: エラーが発生したコード（オプション）
            
        Returns:
            Dict: 診断結果を含む辞書
        """
        logger.info(f"{language}のエラーを診断: {error_message[:100]}...")
        
        # 言語が対応しているか確認
        if language.lower() not in self.error_patterns:
            return {
                "status": "error",
                "message": f"言語 '{language}' は現在サポートされていません",
                "supported_languages": list(self.error_patterns.keys())
            }
        
        # マッチするエラーパターンを検索
        matched_patterns = []
        error_categories = self.error_patterns[language.lower()]
        
        for category, patterns in error_categories.items():
            for pattern in patterns:
                if pattern.matches(error_message):
                    matched_patterns.append(pattern.to_dict())
        
        # LLMを使用した詳細診断（LLMエンジンが利用可能な場合）
        llm_diagnosis = None
        if self.llm_engine and code:
            llm_diagnosis = self._get_llm_diagnosis(error_message, language, code)
        
        # 結果の作成
        result = {
            "status": "success",
            "language": language,
            "error_message": error_message,
            "matched_patterns": matched_patterns,
            "llm_diagnosis": llm_diagnosis,
            "has_solution": bool(matched_patterns or llm_diagnosis)
        }
        
        return result
    
    def _get_llm_diagnosis(self, error_message: str, language: str, code: str) -> Dict:
        """
        LLMを使用してエラーを詳細に診断
        
        Args:
            error_message: 診断するエラーメッセージ
            language: プログラミング言語
            code: エラーが発生したコード
            
        Returns:
            Dict: LLM診断結果を含む辞書
        """
        try:
            # LLMへのプロンプトを構築
            prompt = f"""
            以下の{language}コードでエラーが発生しました。エラーの原因と解決策を詳細に説明してください。

            ## コード:
            ```{language}
            {code}
            ```

            ## エラーメッセージ:
            ```
            {error_message}
            ```

            以下の構造で回答してください:
            1. 根本的な問題は何か
            2. エラーの正確な位置と原因
            3. 具体的な修正案（コードサンプル付き）
            4. 同様のエラーを避けるためのベストプラクティス
            """
            
            # LLMエンジンを呼び出し
            response = self.llm_engine.generate(prompt)
            
            # 簡単なパース（実際の実装ではより堅牢なパースが必要）
            diagnosis = {
                "problem": "不明",
                "location": "不明",
                "fix": "不明",
                "best_practices": []
            }
            
            # LLMの出力から情報を抽出（プロトタイプでは簡易実装）
            lines = response.strip().split("\n")
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if "問題" in line or "Problem" in line:
                    current_section = "problem"
                    diagnosis["problem"] = ""
                elif "位置" in line or "Location" in line:
                    current_section = "location"
                    diagnosis["location"] = ""
                elif "修正" in line or "Fix" in line:
                    current_section = "fix"
                    diagnosis["fix"] = ""
                elif "ベストプラクティス" in line or "Best Practices" in line:
                    current_section = "best_practices"
                    diagnosis["best_practices"] = []
                elif current_section:
                    if current_section == "problem":
                        diagnosis["problem"] += line + " "
                    elif current_section == "location":
                        diagnosis["location"] += line + " "
                    elif current_section == "fix":
                        diagnosis["fix"] += line + " "
                    elif current_section == "best_practices":
                        if line.startswith("- "):
                            diagnosis["best_practices"].append(line[2:])
                        else:
                            diagnosis["best_practices"].append(line)
            
            # 各セクションの整理
            diagnosis["problem"] = diagnosis["problem"].strip()
            diagnosis["location"] = diagnosis["location"].strip()
            diagnosis["fix"] = diagnosis["fix"].strip()
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"LLM診断中にエラーが発生しました: {str(e)}")
            return {
                "error": str(e),
                "message": "LLMを使用した詳細診断に失敗しました"
            }
    
    def suggest_fixes(
        self, 
        code: str, 
        error_message: str, 
        language: str
    ) -> Dict:
        """
        エラーの修正案を提案
        
        Args:
            code: 修正するコード
            error_message: エラーメッセージ
            language: プログラミング言語
            
        Returns:
            Dict: 修正案を含む辞書
        """
        logger.info(f"{language}コードの修正案を生成: {error_message[:100]}...")
        
        # エラー診断
        diagnosis = self.diagnose_error(error_message, language, code)
        
        # LLMが利用可能ならより詳細な修正案を生成
        if self.llm_engine:
            try:
                prompt = f"""
                以下の{language}コードでエラーが発生しました。コードを修正してください。

                ## コード:
                ```{language}
                {code}
                ```

                ## エラーメッセージ:
                ```
                {error_message}
                ```

                修正したコード全体を提供し、変更箇所にコメントを追加してください。
                """
                
                response = self.llm_engine.generate(prompt)
                
                # コード部分の抽出（```で囲まれた部分）
                code_match = re.search(r"```(?:[a-zA-Z]*\n)?(.*?)```", response, re.DOTALL)
                fixed_code = code_match.group(1).strip() if code_match else None
                
                # 説明部分の抽出
                explanation = response
                if code_match:
                    # コードブロックを除去
                    explanation = re.sub(r"```.*?```", "", response, flags=re.DOTALL)
                    explanation = explanation.strip()
                
                return {
                    "status": "success",
                    "original_code": code,
                    "fixed_code": fixed_code,
                    "explanation": explanation,
                    "diagnosis": diagnosis
                }
                
            except Exception as e:
                logger.error(f"修正案生成中にエラーが発生しました: {str(e)}")
                return {
                    "status": "error",
                    "message": f"修正案生成に失敗しました: {str(e)}",
                    "diagnosis": diagnosis
                }
        else:
            # LLMなしの場合、基本的な診断結果のみを返す
            return {
                "status": "partial",
                "message": "LLMエンジンが利用できないため、詳細な修正案を生成できません",
                "diagnosis": diagnosis
            }
    
    def analyze_runtime_context(
        self, 
        variables: Dict,
        stacktrace: str,
        code_context: Optional[str] = None
    ) -> Dict:
        """
        実行時コンテキストを分析してデバッグを支援
        
        Args:
            variables: 実行時の変数状態
            stacktrace: スタックトレース
            code_context: 関連するコード（オプション）
            
        Returns:
            Dict: 分析結果を含む辞書
        """
        logger.info("実行時コンテキストを分析...")
        
        # 変数の状態分析
        variable_analysis = self._analyze_variables(variables)
        
        # スタックトレース分析
        stack_analysis = self._analyze_stacktrace(stacktrace)
        
        # コード分析（コードが提供されている場合）
        code_analysis = None
        if code_context and self.code_analyzer:
            try:
                code_analysis = self.code_analyzer.analyze_code(code_context)
            except Exception as e:
                logger.error(f"コード分析中にエラーが発生しました: {str(e)}")
        
        return {
            "status": "success",
            "variable_analysis": variable_analysis,
            "stack_analysis": stack_analysis,
            "code_analysis": code_analysis
        }
    
    def _analyze_variables(self, variables: Dict) -> Dict:
        """
        変数の状態を分析
        
        Args:
            variables: 変数名と値の辞書
            
        Returns:
            Dict: 分析結果
        """
        analysis = {
            "observations": [],
            "potential_issues": []
        }
        
        # 各変数を確認
        for name, value in variables.items():
            # 値の型を確認
            var_type = type(value).__name__
            
            # None値のチェック
            if value is None:
                analysis["observations"].append(f"変数 '{name}' はNoneです")
                analysis["potential_issues"].append(f"変数 '{name}' が予期せずNoneになっている可能性があります")
            
            # 空コンテナのチェック
            if hasattr(value, "__len__") and len(value) == 0:
                analysis["observations"].append(f"変数 '{name}' は空の{var_type}です")
            
            # 文字列のチェック
            if isinstance(value, str):
                if not value.strip():
                    analysis["observations"].append(f"文字列 '{name}' は空白のみか空です")
                
                # 特殊文字を含むかチェック
                if re.search(r"[^\w\s]", value):
                    analysis["observations"].append(f"文字列 '{name}' は特殊文字を含みます")
            
            # 数値のチェック
            if isinstance(value, (int, float)):
                if value == 0:
                    analysis["observations"].append(f"数値 '{name}' はゼロです")
                elif value < 0:
                    analysis["observations"].append(f"数値 '{name}' は負の値です")
                elif isinstance(value, float) and not value.is_integer():
                    analysis["observations"].append(f"数値 '{name}' は小数です")
        
        return analysis
    
    def _analyze_stacktrace(self, stacktrace: str) -> Dict:
        """
        スタックトレースを分析
        
        Args:
            stacktrace: スタックトレース文字列
            
        Returns:
            Dict: 分析結果
        """
        analysis = {
            "error_type": "不明",
            "error_message": "",
            "error_location": "",
            "call_stack": [],
            "observations": []
        }
        
        # 行ごとに処理
        lines = stacktrace.strip().split("\n")
        
        # 最後の行にエラーメッセージがあることが多い
        if lines:
            error_line = lines[-1]
            
            # エラータイプとメッセージの抽出
            error_match = re.search(r"([A-Za-z]+Error|Exception):\s*(.*)", error_line)
            if error_match:
                analysis["error_type"] = error_match.group(1)
                analysis["error_message"] = error_match.group(2)
        
        # スタックトレースからファイル名と行番号を抽出
        file_line_matches = re.findall(r"File \"([^\"]+)\", line (\d+)", stacktrace)
        
        if file_line_matches:
            # 最後のエントリがエラーの場所
            last_file, last_line = file_line_matches[-1]
            analysis["error_location"] = f"{last_file}:{last_line}"
            
            # コールスタックの構築
            for file, line in file_line_matches:
                analysis["call_stack"].append(f"{file}:{line}")
            
            # 共通のパターンの観察
            if len(file_line_matches) > 1:
                # 再帰の可能性をチェック
                files = [file for file, _ in file_line_matches]
                duplicates = {file: files.count(file) for file in set(files)}
                for file, count in duplicates.items():
                    if count > 2:
                        analysis["observations"].append(f"ファイル '{file}' が{count}回呼び出されています - 再帰の可能性があります")
        
        return analysis
    
    def generate_debug_guide(
        self, 
        code: str, 
        error_type: str, 
        language: str
    ) -> Dict:
        """
        特定のエラータイプに対するデバッグガイドを生成
        
        Args:
            code: 対象のコード
            error_type: エラータイプ
            language: プログラミング言語
            
        Returns:
            Dict: デバッグガイド
        """
        logger.info(f"{language}の{error_type}に対するデバッグガイドを生成...")
        
        # 基本的なエラータイプ情報（実際の実装では知識ベースから取得）
        error_info = {
            "python": {
                "SyntaxError": {
                    "description": "コードの構文が正しくありません",
                    "common_causes": [
                        "括弧のミスマッチ",
                        "構文のタイプミス",
                        "インデントの問題"
                    ],
                    "debug_steps": [
                        "エラーメッセージで指定された行を確認する",
                        "コード整形ツールを使用する",
                        "構文要素を一つずつ追加して問題を特定する"
                    ]
                },
                "TypeError": {
                    "description": "操作や関数が不適切な型の引数で使用されています",
                    "common_causes": [
                        "型が合わない操作",
                        "関数への誤った型の引数",
                        "オブジェクトに存在しないメソッドの呼び出し"
                    ],
                    "debug_steps": [
                        "print(type(var))を使用して変数の型を確認する",
                        "値の変換が必要かチェックする",
                        "変数が期待した型かチェックする条件を追加する"
                    ]
                }
            },
            "javascript": {
                "ReferenceError": {
                    "description": "存在しない変数が参照されています",
                    "common_causes": [
                        "変数が宣言されていない",
                        "変数名のタイプミス",
                        "スコープ外の変数アクセス"
                    ],
                    "debug_steps": [
                        "console.logを使用して変数の存在を確認する",
                        "変数の宣言と使用の場所を確認する",
                        "console.log(typeof var)で型を確認する"
                    ]
                }
            }
        }
        
        # 言語とエラータイプの存在確認
        if language not in error_info or error_type not in error_info.get(language, {}):
            # LLMを使用して一般的なガイドを生成
            if self.llm_engine:
                return self._generate_llm_debug_guide(code, error_type, language)
            else:
                return {
                    "status": "error",
                    "message": f"言語 '{language}' のエラータイプ '{error_type}' に対する情報がありません"
                }
        
        # エラー情報の取得
        info = error_info[language][error_type]
        
        # コード分析（可能な場合）
        code_analysis = None
        if self.code_analyzer:
            try:
                code_analysis = self.code_analyzer.analyze_code(code, language)
            except Exception as e:
                logger.error(f"コード分析中にエラーが発生しました: {str(e)}")
        
        # カスタマイズされたデバッグステップ
        custom_steps = info["debug_steps"].copy()
        
        # コード分析結果に基づいてカスタマイズ
        if code_analysis and "potential_issues" in code_analysis:
            for issue in code_analysis["potential_issues"]:
                if "インデント" in issue and error_type == "SyntaxError":
                    custom_steps.append("インデントのレベルを確認し、一貫したインデント（スペースまたはタブ）を使用する")
                elif "未定義" in issue and error_type in ["NameError", "ReferenceError"]:
                    custom_steps.append(f"変数が適切に定義されているか確認する: {issue}")
        
        return {
            "status": "success",
            "error_type": error_type,
            "language": language,
            "description": info["description"],
            "common_causes": info["common_causes"],
            "debug_steps": custom_steps,
            "code_analysis": code_analysis
        }
    
    def _generate_llm_debug_guide(self, code: str, error_type: str, language: str) -> Dict:
        """
        LLMを使用してデバッグガイドを生成
        
        Args:
            code: 対象のコード
            error_type: エラータイプ
            language: プログラミング言語
            
        Returns:
            Dict: デバッグガイド
        """
        try:
            # LLMへのプロンプトを構築
            prompt = f"""
            {language}プログラミングにおける{error_type}エラーに対するデバッグガイドを作成してください。
            
            以下のコードを例として使用してください:
            ```{language}
            {code}
            ```
            
            ガイドには以下の情報を含めてください:
            1. このエラーの簡潔な説明
            2. 一般的な原因（少なくとも3つ）
            3. デバッグする際の具体的なステップ（少なくとも5つ）
            4. このコードに特化したデバッグのヒント
            5. このエラーを防ぐためのベストプラクティス
            
            回答は構造化されたフォーマットで提供してください。
            """
            
            # LLMエンジンを呼び出し
            response = self.llm_engine.generate(prompt)
            
            # 応答を構造化（プロトタイプでは簡易実装）
            guide = {
                "description": "",
                "common_causes": [],
                "debug_steps": [],
                "specific_hints": [],
                "best_practices": []
            }
            
            # LLMの出力から情報を抽出
            lines = response.strip().split("\n")
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if "説明" in line or "Description" in line:
                    current_section = "description"
                    guide["description"] = ""
                elif "原因" in line or "Causes" in line:
                    current_section = "common_causes"
                    guide["common_causes"] = []
                elif "ステップ" in line or "Steps" in line:
                    current_section = "debug_steps"
                    guide["debug_steps"] = []
                elif "ヒント" in line or "Hints" in line:
                    current_section = "specific_hints"
                    guide["specific_hints"] = []
                elif "ベストプラクティス" in line or "Best Practices" in line:
                    current_section = "best_practices"
                    guide["best_practices"] = []
                elif current_section:
                    if current_section == "description":
                        guide["description"] += line + " "
                    elif line.startswith("- ") or line.startswith("* ") or re.match(r"^\d+\.", line):
                        item = re.sub(r"^[- *]\s*|^\d+\.\s*", "", line)
                        if current_section == "common_causes":
                            guide["common_causes"].append(item)
                        elif current_section == "debug_steps":
                            guide["debug_steps"].append(item)
                        elif current_section == "specific_hints":
                            guide["specific_hints"].append(item)
                        elif current_section == "best_practices":
                            guide["best_practices"].append(item)
            
            # 説明の整理
            guide["description"] = guide["description"].strip()
            
            return {
                "status": "success",
                "error_type": error_type,
                "language": language,
                "description": guide["description"],
                "common_causes": guide["common_causes"],
                "debug_steps": guide["debug_steps"],
                "specific_hints": guide["specific_hints"],
                "best_practices": guide["best_practices"],
                "generated_by_llm": True
            }
            
        except Exception as e:
            logger.error(f"LLMデバッグガイド生成中にエラーが発生しました: {str(e)}")
            return {
                "status": "error",
                "message": f"デバッグガイド生成に失敗しました: {str(e)}"
            }
