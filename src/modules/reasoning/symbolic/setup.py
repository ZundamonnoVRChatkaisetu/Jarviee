"""
シンボリックAI推論エンジン設定モジュール

このモジュールは、シンボリックAI推論エンジンの各コンポーネントを
適切に初期化して連携させるためのヘルパー関数を提供します。
"""

import os
import sys
from typing import Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.core.utils.logger import Logger
from src.core.llm.engine import LLMEngine
from src.core.knowledge.graph import KnowledgeGraph

from src.modules.reasoning.symbolic.logic_transformer import LogicTransformer
from src.modules.reasoning.symbolic.kb_interface import SymbolicKnowledgeInterface
from src.modules.reasoning.symbolic.inference_engine import SymbolicInferenceEngine
from src.modules.reasoning.symbolic.result_interpreter import ResultInterpreter


def setup_symbolic_reasoning(llm_engine: LLMEngine, 
                            knowledge_graph: KnowledgeGraph,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    シンボリックAI推論エンジンコンポーネントの設定と初期化
    
    Args:
        llm_engine (LLMEngine): LLM処理エンジン
        knowledge_graph (KnowledgeGraph): 知識グラフエンジン
        config (Dict[str, Any], optional): 設定パラメータ
        
    Returns:
        Dict[str, Any]: 初期化されたコンポーネント
    """
    logger = Logger(__name__)
    logger.info("Setting up symbolic reasoning components")
    
    # デフォルト設定
    default_config = {
        "normalize_rules": True,
        "enable_llm_enhancement": True,
        "abduction_max_hypotheses": 10,
        "preferred_inference_strategy": "weighted"
    }
    
    # 設定のマージ
    if config:
        for key, value in config.items():
            default_config[key] = value
    
    config = default_config
    
    try:
        # コンポーネントの初期化
        # 1. 論理変換モジュール
        logger.info("Initializing logic transformer")
        logic_transformer = LogicTransformer(llm_engine)
        
        # 2. 知識ベースインターフェース
        logger.info("Initializing knowledge base interface")
        kb_interface = SymbolicKnowledgeInterface(knowledge_graph)
        
        # 3. 推論エンジン
        logger.info("Initializing inference engine")
        inference_engine = SymbolicInferenceEngine(kb_interface)
        
        # 4. 結果解釈モジュール
        logger.info("Initializing result interpreter")
        result_interpreter = ResultInterpreter(llm_engine, logic_transformer)
        
        # コンポーネントの組み合わせ
        components = {
            "logic_transformer": logic_transformer,
            "kb_interface": kb_interface,
            "inference_engine": inference_engine,
            "result_interpreter": result_interpreter,
            "config": config
        }
        
        logger.info("Symbolic reasoning components successfully initialized")
        return components
        
    except Exception as e:
        logger.error(f"Error setting up symbolic reasoning components: {str(e)}")
        # 最小限のコンポーネントを返す
        return {
            "error": str(e),
            "config": config
        }


def perform_symbolic_inference(components: Dict[str, Any], 
                              inference_type: str, 
                              params: Dict[str, Any],
                              detail_level: str = "medium") -> Dict[str, Any]:
    """
    シンボリックAI推論の実行と結果の解釈を行う総合関数
    
    Args:
        components (Dict[str, Any]): 初期化されたコンポーネント
        inference_type (str): 推論タイプ (deduction/induction/probabilistic/abduction/analogy/consistency)
        params (Dict[str, Any]): 推論に必要なパラメータ
        detail_level (str, optional): 説明の詳細レベル（low/medium/high）
        
    Returns:
        Dict[str, Any]: 推論結果と解釈
    """
    logger = Logger(__name__)
    logger.info(f"Performing {inference_type} inference")
    
    # エラーチェック
    if "error" in components:
        return {
            "success": False,
            "error": components["error"],
            "inference_type": inference_type
        }
        
    # コンポーネントの取得
    inference_engine = components["inference_engine"]
    result_interpreter = components["result_interpreter"]
    
    try:
        # 推論タイプに応じた処理
        inference_result = None
        
        if inference_type == "deduction":
            # 演繹的推論
            premises = params.get("premises", [])
            query = params.get("query", "")
            
            if not premises or not query:
                raise ValueError("Deduction requires 'premises' and 'query' parameters")
                
            inference_result = inference_engine.deduce(premises, query)
            
        elif inference_type == "induction":
            # 帰納的推論
            examples = params.get("examples", [])
            target_concept = params.get("target_concept", "")
            
            if not examples or not target_concept:
                raise ValueError("Induction requires 'examples' and 'target_concept' parameters")
                
            inference_result = inference_engine.induce(examples, target_concept)
            
        elif inference_type == "probabilistic":
            # 確率的推論
            evidence = params.get("evidence", {})
            hypothesis = params.get("hypothesis", "")
            
            if not evidence or not hypothesis:
                raise ValueError("Probabilistic reasoning requires 'evidence' and 'hypothesis' parameters")
                
            inference_result = inference_engine.reason_with_uncertainty(evidence, hypothesis)
            
        elif inference_type == "abduction":
            # アブダクション推論
            observations = params.get("observations", [])
            
            if not observations:
                raise ValueError("Abduction requires 'observations' parameter")
                
            inference_result = inference_engine.find_best_explanation(observations)
            
        elif inference_type == "analogy":
            # 類推的推論
            source_domain = params.get("source_domain", {})
            target_domain = params.get("target_domain", {})
            relation_to_map = params.get("relation", "")
            
            if not source_domain or not target_domain or not relation_to_map:
                raise ValueError("Analogy requires 'source_domain', 'target_domain', and 'relation' parameters")
                
            inference_result = inference_engine.analogy_reasoning(source_domain, target_domain, relation_to_map)
            
        elif inference_type == "consistency":
            # 一貫性検証
            statements = params.get("statements", [])
            
            if not statements:
                raise ValueError("Consistency check requires 'statements' parameter")
                
            inference_result = inference_engine.check_consistency(statements)
            
        else:
            raise ValueError(f"Unknown inference type: {inference_type}")
            
        # 結果の解釈
        explanation = ""
        
        if inference_result:
            if inference_type == "deduction":
                explanation = result_interpreter.interpret_deduction_result(inference_result, detail_level)
            elif inference_type == "induction":
                explanation = result_interpreter.interpret_induction_result(inference_result, detail_level)
            elif inference_type == "probabilistic":
                explanation = result_interpreter.interpret_probabilistic_result(inference_result, detail_level)
            elif inference_type == "abduction":
                explanation = result_interpreter.interpret_abduction_result(inference_result, detail_level)
            elif inference_type == "analogy":
                explanation = result_interpreter.interpret_analogy_result(inference_result, detail_level)
            elif inference_type == "consistency":
                explanation = result_interpreter.interpret_consistency_result(inference_result, detail_level)
                
        # 結果の構築
        result = {
            "success": True,
            "inference_type": inference_type,
            "inference_result": inference_result,
            "explanation": explanation,
            "detail_level": detail_level
        }
        
        # オプション: ドメイン特化変換
        if "target_domain" in params:
            target_domain = params.get("target_domain")
            format_type = params.get("format_type", "general")
            
            domain_specific = result_interpreter.translate_to_domain_specific(
                inference_result, target_domain, format_type
            )
            
            result["domain_specific"] = domain_specific
            
        # オプション: 応用推奨
        if params.get("generate_recommendations", False):
            context = params.get("context", {})
            domain = params.get("domain", "general")
            
            recommendations = result_interpreter.generate_application_recommendations(
                inference_result, context, domain
            )
            
            result["recommendations"] = recommendations
            
        logger.info(f"{inference_type.capitalize()} inference completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error performing {inference_type} inference: {str(e)}")
        return {
            "success": False,
            "inference_type": inference_type,
            "error": str(e)
        }
