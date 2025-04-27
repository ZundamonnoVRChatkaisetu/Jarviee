"""
Jarviee デバッグエンジン

このモジュールは、プログラミング特化機能の一部として、
コードのデバッグ支援を提供します。LLMの推論能力と
コード解析エンジンを組み合わせて、エラーの診断と修正を行います。
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import traceback
import ast
import re

from src.core.llm.engine import LLMEngine
from src.core.knowledge.query_engine import KnowledgeQueryEngine
from src.modules.programming.code_analyzer import CodeAnalyzer

logger = logging.getLogger(__name__)

class DebuggingSession:
    """デバッグセッションを表すクラス。問題の追跡と解決過程を管理します。"""
    
    def __init__(self, code: str, error_message: Optional[str] = None, 
                 language: str = "python", context: Optional[Dict] = None):
        """
        デバッグセッションを初期化します。
        
        Args:
            code: デバッグ対象のコード
            error_message: エラーメッセージ（存在する場合）
            language: プログラミング言語
            context: 追加のコンテキスト情報（環境、関連ファイルなど）
        """
        self.code = code
        self.error_message = error_message
        self.language = language
        self.context = context or {}
        self.hypothesis = []  # 考えられる問題の仮説リスト
        self.fixes_attempted = []  # 試した修正のリスト
        self.solution = None  # 確認された解決策
        
    def add_hypothesis(self, hypothesis: str, confidence: float, reasoning: str) -> None:
        """
        デバッグ仮説を追加します。
        
        Args:
            hypothesis: 問題の考えられる原因
            confidence: 仮説の確信度（0.0～1.0）
            reasoning: この仮説を提案する理由
        """
        self.hypothesis.append({
            "hypothesis": hypothesis, 
            "confidence": confidence,
            "reasoning": reasoning,
            "verified": False
        })
        
    def add_fix_attempt(self, fix_code: str, hypothesis_index: int, 
                        result: Optional[str] = None) -> None:
        """
        修正の試みを追加します。
        
        Args:
            fix_code: 修正されたコード
            hypothesis_index: 対応する仮説のインデックス
            result: 修正の結果（成功、失敗、または新しいエラー）
        """
        self.fixes_attempted.append({
            "fix_code": fix_code,
            "hypothesis_index": hypothesis_index,
            "result": result,
            "success": result is None or "success" in result.lower()
        })
        
    def set_solution(self, fix_index: int) -> None:
        """
        確認された解決策を設定します。
        
        Args:
            fix_index: 解決策となる修正試行のインデックス
        """
        if 0 <= fix_index < len(self.fixes_attempted):
            self.solution = self.fixes_attempted[fix_index]
            # 対応する仮説を検証済みにマーク
            hyp_idx = self.solution["hypothesis_index"]
            if 0 <= hyp_idx < len(self.hypothesis):
                self.hypothesis[hyp_idx]["verified"] = True

    def get_debug_summary(self) -> Dict:
        """デバッグセッションの要約を取得します。"""
        return {
            "language": self.language,
            "error_message": self.error_message,
            "hypotheses_count": len(self.hypothesis),
            "fixes_attempted": len(self.fixes_attempted),
            "solution_found": self.solution is not None,
            "verified_hypotheses": sum(1 for h in self.hypothesis if h.get("verified"))
        }


class ErrorDiagnosticEngine:
    """エラー診断エンジン。異なるタイプのエラーを分析し、根本原因を特定します。"""
    
    def __init__(self, llm_engine: LLMEngine, code_analyzer: CodeAnalyzer,
                 knowledge_engine: KnowledgeQueryEngine):
        """
        エラー診断エンジンを初期化します。
        
        Args:
            llm_engine: LLMエンジンインスタンス
            code_analyzer: コード解析エンジンインスタンス
            knowledge_engine: 知識クエリエンジンインスタンス
        """
        self.llm = llm_engine
        self.analyzer = code_analyzer
        self.knowledge = knowledge_engine
        self.error_patterns = self._load_error_patterns()
        
    def _load_error_patterns(self) -> Dict:
        """既知のエラーパターンデータベースを読み込みます。"""
        # 実際の実装では、外部ファイルまたはデータベースから読み込む
        # 現在は、基本的なパターンをハードコードで提供
        return {
            "python": {
                "syntax": [
                    {"pattern": r"SyntaxError: invalid syntax", 
                     "common_causes": ["括弧の不一致", "コロンの欠落", "インデントエラー"]},
                    {"pattern": r"IndentationError", 
                     "common_causes": ["インデントの一貫性がない", "タブとスペースの混在"]},
                    {"pattern": r"NameError: name '(.+)' is not defined", 
                     "common_causes": ["変数の未定義", "タイプミス", "スコープの問題"]}
                ],
                "runtime": [
                    {"pattern": r"TypeError: (.+)", 
                     "common_causes": ["不適切な型の使用", "型変換の欠落"]},
                    {"pattern": r"IndexError: (.+)", 
                     "common_causes": ["配列インデックスの範囲外アクセス", "空のリスト"]}
                ],
                "logic": [
                    {"pattern": r"RecursionError: (.+)", 
                     "common_causes": ["無限再帰", "基底ケースの欠落"]}
                ]
            },
            "javascript": {
                "syntax": [
                    {"pattern": r"SyntaxError: (.+)", 
                     "common_causes": ["括弧の不一致", "セミコロンの問題"]}
                ],
                "runtime": [
                    {"pattern": r"TypeError: (.+)", 
                     "common_causes": ["undefined または null の参照", "不適切なメソッド呼び出し"]},
                    {"pattern": r"ReferenceError: (.+)", 
                     "common_causes": ["未宣言の変数", "スコープの問題"]}
                ]
            },
            # 他の言語のパターンをここに追加
        }
    
    def match_error_patterns(self, error_message: str, language: str) -> List[Dict]:
        """
        エラーメッセージを既知のパターンと照合します。
        
        Args:
            error_message: エラーメッセージ
            language: プログラミング言語
            
        Returns:
            一致したパターンのリスト
        """
        matches = []
        if not error_message:
            return matches
            
        # 言語が知られているパターンにある場合のみ処理
        if language not in self.error_patterns:
            return matches
            
        # 各カテゴリのパターンを確認
        for category, patterns in self.error_patterns[language].items():
            for pattern_info in patterns:
                if re.search(pattern_info["pattern"], error_message):
                    matches.append({
                        "category": category,
                        "pattern": pattern_info["pattern"],
                        "common_causes": pattern_info["common_causes"]
                    })
        
        return matches
    
    def analyze_error(self, code: str, error_message: Optional[str], 
                     language: str) -> List[Dict]:
        """
        エラーを分析し、考えられる原因のリストを返します。
        
        Args:
            code: 問題のあるコード
            error_message: エラーメッセージ（存在する場合）
            language: プログラミング言語
            
        Returns:
            考えられる原因と確信度のリスト
        """
        # 静的解析を実行
        static_issues = self.analyzer.analyze_code(code, language)
        
        # エラーパターンのマッチング
        pattern_matches = []
        if error_message:
            pattern_matches = self.match_error_patterns(error_message, language)
        
        # LLMによる深い診断
        llm_diagnosis = self._llm_error_diagnosis(code, error_message, language, 
                                                static_issues, pattern_matches)
        
        # 関連する知識ベースのクエリ
        if error_message:
            knowledge_results = self.knowledge.query(
                f"debugging {language} error: {error_message}",
                limit=3
            )
        else:
            knowledge_results = []
        
        # すべての情報を統合
        return self._integrate_diagnostic_results(
            static_issues, pattern_matches, llm_diagnosis, knowledge_results
        )
    
    def _llm_error_diagnosis(self, code: str, error_message: Optional[str], 
                            language: str, static_issues: List, 
                            pattern_matches: List) -> List[Dict]:
        """
        LLMを使用してエラーの深い診断を行います。
        
        Args:
            code: 問題のあるコード
            error_message: エラーメッセージ
            language: プログラミング言語
            static_issues: 静的解析からの問題
            pattern_matches: マッチしたエラーパターン
            
        Returns:
            LLMによる診断結果
        """
        # プロンプトの構築
        prompt = f"""
        あなたは高度なプログラミングデバッグアシスタントです。
        以下の{language}コードを分析し、考えられるエラーの原因を特定してください。
        
        コード:
        ```{language}
        {code}
        ```
        
        {f'エラーメッセージ: {error_message}' if error_message else '特定のエラーメッセージはありませんが、コードに問題があると思われます。'}
        
        {"静的解析で検出された問題: " + str(static_issues) if static_issues else ""}
        {"一致するエラーパターン: " + str(pattern_matches) if pattern_matches else ""}
        
        以下の形式で回答してください:
        1. 最も可能性の高い問題（確信度を0から1で表示）
           理由:
           修正案:
        
        2. 次に可能性のある問題（確信度を0から1で表示）
           理由:
           修正案:
        
        3. 他に考慮すべき問題（確信度を0から1で表示）
           理由:
           修正案:
        """
        
        # LLMで診断を生成
        response = self.llm.generate(prompt)
        
        # 応答を構造化されたリストに解析
        diagnoses = self._parse_llm_diagnosis(response)
        return diagnoses
    
    def _parse_llm_diagnosis(self, llm_response: str) -> List[Dict]:
        """
        LLMの診断応答を構造化された形式に解析します。
        
        Args:
            llm_response: LLMからの応答テキスト
            
        Returns:
            構造化された診断リスト
        """
        diagnoses = []
        
        # 正規表現でフォーマットされた回答を解析
        pattern = r"(\d+)\.\s+(.*?)（確信度[を:]\s*([0-9.]+)[）\)].*?）"
        matches = re.finditer(pattern, llm_response, re.DOTALL)
        
        for match in matches:
            index = match.group(1)
            problem = match.group(2).strip()
            confidence = float(match.group(3))
            
            # 理由と修正案を抽出
            reason_match = re.search(r"理由[：:](.*?)(?:修正案[：:]|$)", 
                                    llm_response[match.end():], re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else ""
            
            fix_match = re.search(r"修正案[：:](.*?)(?:\d+\.\s+|$)", 
                                 llm_response[match.end():], re.DOTALL)
            fix = fix_match.group(1).strip() if fix_match else ""
            
            diagnoses.append({
                "problem": problem,
                "confidence": confidence,
                "reasoning": reason,
                "suggested_fix": fix
            })
        
        return diagnoses
    
    def _integrate_diagnostic_results(self, static_issues: List, 
                                     pattern_matches: List, 
                                     llm_diagnosis: List[Dict],
                                     knowledge_results: List) -> List[Dict]:
        """
        すべての診断ソースの結果を統合します。
        
        Args:
            static_issues: 静的解析からの問題
            pattern_matches: マッチしたエラーパターン
            llm_diagnosis: LLMによる診断
            knowledge_results: 知識ベースからの関連結果
            
        Returns:
            統合された診断結果
        """
        # 基本的にはLLM診断を主要な結果として使用
        integrated_results = llm_diagnosis.copy()
        
        # 静的解析とパターンマッチの結果がLLM診断に含まれていない場合、追加
        for issue in static_issues:
            if not any(issue["message"] in diag["problem"] for diag in integrated_results):
                integrated_results.append({
                    "problem": issue["message"],
                    "confidence": 0.7,  # 静的解析は比較的信頼性が高い
                    "reasoning": f"静的コード解析による検出: {issue.get('description', '')}",
                    "suggested_fix": issue.get("fix", "")
                })
        
        for match in pattern_matches:
            causes = match.get("common_causes", [])
            if causes and not any(any(cause in diag["problem"] for cause in causes) 
                                for diag in integrated_results):
                integrated_results.append({
                    "problem": f"{match['category']}エラー: {', '.join(causes)}",
                    "confidence": 0.6,  # パターンマッチは一般的なケースをカバー
                    "reasoning": f"既知のエラーパターンに一致: {match['pattern']}",
                    "suggested_fix": "特定のコードコンテキストに基づく修正が必要です。"
                })
        
        # 知識ベースからの関連情報を追加
        for result in knowledge_results:
            if not any(result["title"] in diag["problem"] for diag in integrated_results):
                integrated_results.append({
                    "problem": result["title"],
                    "confidence": 0.5,  # 知識ベースは一般的な情報
                    "reasoning": f"知識ベースからの関連情報: {result['snippet']}",
                    "suggested_fix": result.get("solution", "")
                })
        
        # 確信度でソート
        integrated_results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return integrated_results


class FixGenerator:
    """コード修正生成エンジン。診断された問題に基づいて修正案を生成します。"""
    
    def __init__(self, llm_engine: LLMEngine, code_analyzer: CodeAnalyzer):
        """
        修正生成エンジンを初期化します。
        
        Args:
            llm_engine: LLMエンジンインスタンス
            code_analyzer: コード解析エンジンインスタンス
        """
        self.llm = llm_engine
        self.analyzer = code_analyzer
        
    def generate_fix(self, code: str, diagnostic: Dict, language: str) -> str:
        """
        診断に基づいて修正されたコードを生成します。
        
        Args:
            code: 元のコード
            diagnostic: 診断結果
            language: プログラミング言語
            
        Returns:
            修正されたコード
        """
        # 診断に既に修正案が含まれている場合は使用
        if diagnostic.get("suggested_fix") and len(diagnostic["suggested_fix"]) > 10:
            suggested_fix = diagnostic["suggested_fix"]
            # コードブロックから実際のコードを抽出
            code_pattern = r"```.*?\n(.*?)```"
            code_match = re.search(code_pattern, suggested_fix, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            # 明示的なコードブロックがない場合は、テキスト全体を使用
            if "```" not in suggested_fix and len(suggested_fix.strip()) > 0:
                return suggested_fix.strip()
        
        # LLMを使用して修正を生成
        prompt = f"""
        あなた専門的なプログラミングアシスタントです。以下の{language}コードを修正してください。

        元のコード:
        ```{language}
        {code}
        ```
        
        問題: {diagnostic["problem"]}
        理由: {diagnostic["reasoning"]}
        
        修正されたコード全体を提供してください。変更部分にはコメントを追加してください。
        コードブロックのみを返し、追加の説明は不要です。
        """
        
        response = self.llm.generate(prompt)
        
        # 応答からコードブロックを抽出
        code_pattern = r"```.*?\n(.*?)```"
        code_match = re.search(code_pattern, response, re.DOTALL)
        if code_match:
            fixed_code = code_match.group(1).strip()
        else:
            # コードブロックが見つからない場合、全体を使用
            fixed_code = response.strip()
        
        return fixed_code
    
    def verify_fix(self, original_code: str, fixed_code: str, 
                  language: str) -> Tuple[bool, Optional[str]]:
        """
        修正が問題を解決するかどうかを検証します。
        
        Args:
            original_code: 元のコード
            fixed_code: 修正されたコード
            language: プログラミング言語
            
        Returns:
            (成功したかどうか, 検証結果または新しいエラーメッセージ)
        """
        # 修正が元のコードと同じでないことを確認
        if original_code.strip() == fixed_code.strip():
            return False, "修正されたコードが元のコードと同じです。"
        
        # 構文的に有効であることを確認
        syntax_check = self._check_syntax(fixed_code, language)
        if not syntax_check[0]:
            return False, f"修正されたコードに構文エラーがあります: {syntax_check[1]}"
        
        # 静的解析でチェック
        issues = self.analyzer.analyze_code(fixed_code, language)
        critical_issues = [i for i in issues if i.get("severity") == "error"]
        if critical_issues:
            return False, f"修正されたコードに問題があります: {critical_issues[0]['message']}"
        
        # 注：実際の実行テストは、安全なサンドボックス環境で行うべき
        # ここでは単純な静的チェックのみを実施
        
        return True, "修正が有効と思われます。"
    
    def _check_syntax(self, code: str, language: str) -> Tuple[bool, Optional[str]]:
        """
        コードの構文的な有効性をチェックします。
        
        Args:
            code: チェックするコード
            language: プログラミング言語
            
        Returns:
            (有効かどうか, エラーメッセージ)
        """
        if language == "python":
            try:
                ast.parse(code)
                return True, None
            except SyntaxError as e:
                return False, str(e)
        
        # JavaScript、TypeScriptなど他の言語のチェックはそれぞれの
        # パーサーを使用して実装する
        
        # 未サポートの言語の場合
        return True, None  # デフォルトで有効と仮定


class DebugEngine:
    """デバッグエンジンのメインクラス。診断と修正プロセスを調整します。"""
    
    def __init__(self, llm_engine: LLMEngine, code_analyzer: CodeAnalyzer,
                 knowledge_engine: KnowledgeQueryEngine):
        """
        デバッグエンジンを初期化します。
        
        Args:
            llm_engine: LLMエンジンインスタンス
            code_analyzer: コード解析エンジンインスタンス
            knowledge_engine: 知識クエリエンジンインスタンス
        """
        self.diagnostic_engine = ErrorDiagnosticEngine(
            llm_engine, code_analyzer, knowledge_engine
        )
        self.fix_generator = FixGenerator(llm_engine, code_analyzer)
        self.sessions = {}  # sessionId -> DebuggingSession
        
    def start_debugging(self, code: str, error_message: Optional[str] = None,
                       language: str = "python", context: Optional[Dict] = None) -> str:
        """
        デバッグセッションを開始します。
        
        Args:
            code: デバッグするコード
            error_message: エラーメッセージ（存在する場合）
            language: プログラミング言語
            context: 追加コンテキスト情報
            
        Returns:
            セッションID
        """
        # セッションIDを生成
        import uuid
        session_id = str(uuid.uuid4())
        
        # セッションを作成
        session = DebuggingSession(code, error_message, language, context)
        self.sessions[session_id] = session
        
        # 初期診断を実行
        self._run_diagnostics(session_id)
        
        return session_id
    
    def _run_diagnostics(self, session_id: str) -> None:
        """
        指定されたセッションの診断を実行します。
        
        Args:
            session_id: デバッグセッションID
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return
            
        # エラー診断を実行
        diagnostics = self.diagnostic_engine.analyze_error(
            session.code, session.error_message, session.language
        )
        
        # 診断結果をセッションに追加
        for i, diag in enumerate(diagnostics[:3]):  # 上位3つの診断のみ処理
            session.add_hypothesis(
                diag["problem"],
                diag["confidence"],
                diag["reasoning"]
            )
    
    def get_diagnostics(self, session_id: str) -> List[Dict]:
        """
        セッションの診断結果を取得します。
        
        Args:
            session_id: デバッグセッションID
            
        Returns:
            診断結果リスト
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return []
            
        return session.hypothesis
    
    def generate_fix(self, session_id: str, hypothesis_index: int) -> Optional[str]:
        """
        特定の仮説に基づいて修正を生成します。
        
        Args:
            session_id: デバッグセッションID
            hypothesis_index: 修正を生成する仮説のインデックス
            
        Returns:
            生成された修正コード、またはエラー時はNone
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return None
            
        if not (0 <= hypothesis_index < len(session.hypothesis)):
            logger.error(f"Hypothesis index {hypothesis_index} out of range")
            return None
            
        hypothesis = session.hypothesis[hypothesis_index]
        
        # 診断から修正を生成
        diagnostic = {
            "problem": hypothesis["hypothesis"],
            "reasoning": hypothesis["reasoning"],
            "suggested_fix": ""  # 修正候補はまだない
        }
        
        try:
            fixed_code = self.fix_generator.generate_fix(
                session.code, diagnostic, session.language
            )
            
            # 修正を検証
            is_valid, verify_result = self.fix_generator.verify_fix(
                session.code, fixed_code, session.language
            )
            
            # セッションに修正試行を記録
            session.add_fix_attempt(fixed_code, hypothesis_index, verify_result)
            
            # 検証が成功した場合、解決策として設定
            if is_valid:
                session.set_solution(len(session.fixes_attempted) - 1)
                
            return fixed_code
            
        except Exception as e:
            logger.error(f"Error generating fix: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def test_fix(self, session_id: str, fix_index: int, test_result: str) -> None:
        """
        修正の試行結果を記録します。
        
        Args:
            session_id: デバッグセッションID
            fix_index: テストする修正のインデックス
            test_result: テスト結果（"success" または新しいエラーメッセージ）
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return
            
        if not (0 <= fix_index < len(session.fixes_attempted)):
            logger.error(f"Fix index {fix_index} out of range")
            return
            
        # 修正の結果を更新
        fix = session.fixes_attempted[fix_index]
        fix["result"] = test_result
        fix["success"] = "success" in test_result.lower()
        
        # 成功した場合、解決策として設定
        if fix["success"]:
            session.set_solution(fix_index)
    
    def get_debug_summary(self, session_id: str) -> Dict:
        """
        デバッグセッションの要約を取得します。
        
        Args:
            session_id: デバッグセッションID
            
        Returns:
            セッション要約
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return {}
            
        return session.get_debug_summary()
    
    def get_solution(self, session_id: str) -> Optional[Dict]:
        """
        解決策を取得します（存在する場合）。
        
        Args:
            session_id: デバッグセッションID
            
        Returns:
            解決策情報、または未解決の場合はNone
        """
        session = self.sessions.get(session_id)
        if not session or not session.solution:
            return None
            
        solution_index = session.fixes_attempted.index(session.solution)
        hypothesis_index = session.solution["hypothesis_index"]
        
        return {
            "fixed_code": session.solution["fix_code"],
            "problem": session.hypothesis[hypothesis_index]["hypothesis"],
            "reasoning": session.hypothesis[hypothesis_index]["reasoning"],
            "test_result": session.solution["result"]
        }
    
    def clean_up_session(self, session_id: str) -> None:
        """
        セッションをクリーンアップします。
        
        Args:
            session_id: デバッグセッションID
        """
        if session_id in self.sessions:
            del self.sessions[session_id]


# 互換性のためのエイリアス
DebuggingEngine = DebugEngine
