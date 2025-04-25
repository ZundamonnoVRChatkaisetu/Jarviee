"""
シンボリックAI結果解釈モジュールのテスト
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from src.modules.reasoning.symbolic.result_interpreter import ResultInterpreter, ResultTemplates


class TestResultInterpreter(unittest.TestCase):
    """結果解釈モジュールのテストクラス"""
    
    def setUp(self):
        """テスト環境のセットアップ"""
        # LLMエンジンのモック
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = "LLMによって強化された説明"
        
        # LogicTransformerのモック
        self.mock_transformer = MagicMock()
        self.mock_transformer.logic_to_natural.return_value = "自然言語に変換された論理表現"
        
        # テスト対象のインスタンス化
        self.interpreter = ResultInterpreter(self.mock_llm, self.mock_transformer)
    
    def test_interpret_deduction_result_success(self):
        """演繹的推論の成功結果解釈テスト"""
        # テスト用の推論結果
        deduction_result = {
            "query": "すべての人間は死ぬ。ソクラテスは人間である。よってソクラテスは死ぬ。",
            "result": True,
            "confidence": 1.0,
            "steps": [
                "前提: すべての人間は死ぬ",
                "前提: ソクラテスは人間である",
                "演繹: ソクラテスは死ぬ"
            ]
        }
        
        # 結果解釈の実行
        explanation = self.interpreter.interpret_deduction_result(deduction_result, "medium")
        
        # 結果の検証
        self.assertIsInstance(explanation, str)
        self.assertIn("証明されました", explanation)
        self.assertIn("ソクラテスは死ぬ", explanation)
        self.assertIn("確信度: 1.00", explanation)
    
    def test_interpret_deduction_result_failure(self):
        """演繹的推論の失敗結果解釈テスト"""
        # テスト用の推論結果
        deduction_result = {
            "query": "ソクラテスは不死である",
            "result": False,
            "confidence": 0.2,
            "steps": [
                "前提: すべての人間は死ぬ",
                "前提: ソクラテスは人間である",
                "矛盾検出: ソクラテスは不死であるという主張は既知の前提と矛盾"
            ]
        }
        
        # 結果解釈の実行
        explanation = self.interpreter.interpret_deduction_result(deduction_result, "medium")
        
        # 結果の検証
        self.assertIsInstance(explanation, str)
        self.assertIn("証明できませんでした", explanation)
        self.assertIn("ソクラテスは不死である", explanation)
        self.assertIn("確信度: 0.20", explanation)
    
    def test_interpret_induction_result(self):
        """帰納的推論結果解釈テスト"""
        # テスト用の推論結果
        induction_result = {
            "target_concept": "鳥は飛ぶ",
            "rule": "∀x (鳥(x) ∧ 健康(x) → 飛ぶ(x))",
            "confidence": 0.8,
            "example_count": 100,
            "details": {
                "coverage": 0.95
            }
        }
        
        # 結果解釈の実行
        explanation = self.interpreter.interpret_induction_result(induction_result, "low")
        
        # 結果の検証
        self.assertIsInstance(explanation, str)
        self.assertIn("帰納的推論", explanation)
        self.assertIn("自然言語に変換された論理表現", explanation)
        self.assertIn("確信度: 0.80", explanation)
        self.assertIn("100個の例", explanation)
    
    def test_interpret_probabilistic_result(self):
        """確率的推論結果解釈テスト"""
        # テスト用の推論結果
        probabilistic_result = {
            "hypothesis": "明日は雨が降る",
            "prior": 0.3,
            "posterior": 0.7,
            "evidence_count": 3,
            "details": {
                "likelihood": 0.8
            }
        }
        
        # 結果解釈の実行
        explanation = self.interpreter.interpret_probabilistic_result(probabilistic_result, "medium")
        
        # 結果の検証
        self.assertIsInstance(explanation, str)
        self.assertIn("ベイズ更新", explanation)
        self.assertIn("明日は雨が降る", explanation)
        self.assertIn("0.30から0.70に更新", explanation)
        self.assertIn("3件の証拠", explanation)
    
    def test_interpret_abduction_result(self):
        """アブダクション推論結果解釈テスト"""
        # テスト用の推論結果
        abduction_result = {
            "best_explanation": "雨が降った(昨日)",
            "score": 0.85,
            "explanation_steps": [
                "観察: 地面が濡れている",
                "観察: 空が曇っている",
                "仮説生成: 雨が降った可能性がある",
                "仮説評価: 地面が濡れる原因として雨は最も可能性が高い"
            ],
            "observation_count": 2
        }
        
        # 結果解釈の実行
        explanation = self.interpreter.interpret_abduction_result(abduction_result, "high")
        
        # 結果の検証
        self.assertIsInstance(explanation, str)
        self.assertIn("アブダクション", explanation)
        self.assertIn("最良の説明", explanation)
        self.assertIn("自然言語に変換された論理表現", explanation)
        self.assertIn("スコア: 0.85", explanation)
    
    def test_interpret_consistency_result_consistent(self):
        """一貫性検証結果（一貫性あり）解釈テスト"""
        # テスト用の検証結果
        consistency_result = {
            "is_consistent": True,
            "direct_contradictions": [],
            "inferred_contradictions": [],
            "statement_count": 5
        }
        
        # 結果解釈の実行
        explanation = self.interpreter.interpret_consistency_result(consistency_result, "low")
        
        # 結果の検証
        self.assertIsInstance(explanation, str)
        self.assertIn("一貫性検証", explanation)
        self.assertIn("5個の命題", explanation)
        self.assertIn("論理的に一貫しています", explanation)
    
    def test_interpret_consistency_result_inconsistent(self):
        """一貫性検証結果（矛盾あり）解釈テスト"""
        # テスト用の検証結果
        consistency_result = {
            "is_consistent": False,
            "direct_contradictions": [
                ("すべての鳥は飛ぶ", "ペンギンは鳥であり、飛ばない")
            ],
            "inferred_contradictions": [],
            "statement_count": 3
        }
        
        # 結果解釈の実行
        explanation = self.interpreter.interpret_consistency_result(consistency_result, "medium")
        
        # 結果の検証
        self.assertIsInstance(explanation, str)
        self.assertIn("一貫性検証", explanation)
        self.assertIn("3個の命題", explanation)
        self.assertIn("矛盾を含んでいます", explanation)
        self.assertIn("すべての鳥は飛ぶ", explanation)
        self.assertIn("ペンギンは鳥であり、飛ばない", explanation)
    
    @patch('src.modules.reasoning.symbolic.result_interpreter.ResultInterpreter._enhance_explanation_with_llm')
    def test_llm_enhancement(self, mock_enhance):
        """LLMによる説明強化のテスト"""
        # モックの設定
        mock_enhance.return_value = "LLMによって大幅に強化された説明"
        
        # テスト用の推論結果
        result = {
            "query": "単純な推論クエリ",
            "result": True,
            "confidence": 1.0,
            "steps": ["単純なステップ"]
        }
        
        # 結果解釈の実行（high詳細レベルでLLM強化が呼ばれる）
        explanation = self.interpreter.interpret_deduction_result(result, "high")
        
        # LLM強化関数が呼ばれたことを確認
        mock_enhance.assert_called_once()
        
        # 強化された説明が返されたことを確認
        self.assertEqual(explanation, "LLMによって大幅に強化された説明")
    
    def test_detail_level_filtering(self):
        """詳細レベルによるステップフィルタリングのテスト"""
        # テスト用の長い推論ステップリスト
        long_steps = [f"ステップ{i}" for i in range(1, 11)]
        
        # テスト用の推論結果
        result = {
            "query": "長いステップを持つクエリ",
            "result": True,
            "confidence": 1.0,
            "steps": long_steps
        }
        
        # 低詳細レベルでの実行（最初と最後のステップのみ）
        low_explanation = self.interpreter.interpret_deduction_result(result, "low")
        
        # 中詳細レベルでの実行（間引きされたステップ）
        med_explanation = self.interpreter.interpret_deduction_result(result, "medium")
        
        # 高詳細レベルでの実行（全ステップ）
        high_explanation = self.interpreter.interpret_deduction_result(result, "high")
        
        # 低詳細レベルでは最初と最後のステップだけ含まれていることを確認
        self.assertIn("ステップ1", low_explanation)
        self.assertIn("ステップ10", low_explanation)
        self.assertIn("...", low_explanation)
        
        # 中詳細レベルでは間引きされていることを確認
        self.assertIn("ステップ1", med_explanation)
        
        # 高詳細レベルでは全ステップが含まれていることを確認
        for i in range(1, 11):
            self.assertIn(f"ステップ{i}", high_explanation)


if __name__ == '__main__':
    unittest.main()
