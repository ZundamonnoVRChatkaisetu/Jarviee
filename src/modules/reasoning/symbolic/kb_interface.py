"""
シンボリックAIの知識ベース連携インターフェース

このモジュールは論理ルールの知識ベース保存・検索・クエリ実行機能を提供し、
LLMとシンボリックAIの推論エンジンの橋渡しを行います。
"""

import os
import sys
import json
import uuid
from typing import Dict, List, Any, Union, Optional, Tuple, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.core.utils.logger import Logger
from src.core.knowledge.graph import KnowledgeGraph
from src.modules.reasoning.symbolic.logic_transformer import LogicTransformer


class RuleConverter:
    """論理ルールとグラフパターンの相互変換を行うクラス"""
    
    def __init__(self):
        """ルール変換器の初期化"""
        self.logger = Logger(__name__)
        
    def rule_to_graph_pattern(self, rule: str) -> Dict[str, Any]:
        """
        論理ルールからグラフパターンへの変換
        
        Args:
            rule (str): 論理ルール表現
            
        Returns:
            Dict[str, Any]: グラフパターン表現
        """
        # 基本的なパターン変換を実装
        # 実際にはより複雑なパース/変換処理が必要
        
        # 簡易的な実装例
        pattern = {
            'type': 'rule',
            'rule_text': rule,
            'nodes': [],
            'edges': []
        }
        
        # TODO: 本格的なルール解析とグラフパターン生成
        # ここでは単純化のためスケルトン実装のみ
        
        try:
            # 含意関係の基本的な解析（A → B形式）
            if '→' in rule:
                antecedent, consequent = rule.split('→', 1)
                antecedent = antecedent.strip()
                consequent = consequent.strip()
                
                # 前件のノード・エッジの抽出
                pattern['nodes'].append({
                    'id': 'antecedent',
                    'labels': ['Condition'],
                    'properties': {'text': antecedent}
                })
                
                # 後件のノード・エッジの抽出
                pattern['nodes'].append({
                    'id': 'consequent',
                    'labels': ['Consequence'],
                    'properties': {'text': consequent}
                })
                
                # 含意関係のエッジ
                pattern['edges'].append({
                    'from': 'antecedent',
                    'to': 'consequent',
                    'type': 'IMPLIES',
                    'properties': {}
                })
            
            # 他の論理関係の解析も同様に実装可能
            
        except Exception as e:
            self.logger.error(f"Error converting rule to graph pattern: {str(e)}")
        
        return pattern
    
    def graph_pattern_to_rule(self, pattern: Dict[str, Any]) -> str:
        """
        グラフパターンから論理ルールへの変換
        
        Args:
            pattern (Dict[str, Any]): グラフパターン表現
            
        Returns:
            str: 論理ルール表現
        """
        # グラフパターンからルールへの逆変換
        rule = ""
        
        try:
            # パターンタイプが'rule'か確認
            if pattern.get('type') != 'rule':
                return rule
            
            # ルールテキストが直接格納されている場合はそれを使用
            if 'rule_text' in pattern:
                return pattern['rule_text']
            
            # ノードとエッジから論理ルールを再構築
            antecedent_node = next((n for n in pattern.get('nodes', []) if n.get('id') == 'antecedent'), None)
            consequent_node = next((n for n in pattern.get('nodes', []) if n.get('id') == 'consequent'), None)
            
            if antecedent_node and consequent_node:
                antecedent = antecedent_node.get('properties', {}).get('text', '')
                consequent = consequent_node.get('properties', {}).get('text', '')
                
                if antecedent and consequent:
                    rule = f"{antecedent} → {consequent}"
            
            # 他の論理関係の再構築も同様に実装可能
            
        except Exception as e:
            self.logger.error(f"Error converting graph pattern to rule: {str(e)}")
        
        return rule
    
    def logic_to_graph_query(self, logical_query: str) -> Dict[str, Any]:
        """
        論理クエリからグラフクエリへの変換
        
        Args:
            logical_query (str): 論理形式のクエリ
            
        Returns:
            Dict[str, Any]: グラフデータベース用クエリ
        """
        # Neo4jのCypherクエリなどへの変換
        # 実際にはより複雑なパース/変換処理が必要
        
        # 簡易的な実装例
        graph_query = {
            'type': 'query',
            'source': 'logic',
            'original': logical_query,
            'cypher': ''
        }
        
        try:
            # ここではシンプルなケースのみ対応
            # 実際には構文解析が必要
            
            # 述語論理の基本検索クエリへの変換
            if '(' in logical_query and ')' in logical_query:
                # 述語名と引数の抽出
                match = logical_query.split('(', 1)
                if len(match) == 2:
                    predicate = match[0].strip()
                    args = match[1].split(')', 1)[0].strip()
                    
                    # 単純なCypherクエリ例 (述語をノードラベルとして扱う簡易実装)
                    cypher = f"MATCH (n:{predicate}) WHERE "
                    
                    # 引数の処理
                    arg_list = [a.strip() for a in args.split(',')]
                    conditions = []
                    
                    for i, arg in enumerate(arg_list):
                        # 変数か定数かの判断（簡易実装）
                        is_variable = arg.islower() and arg.isalpha()
                        
                        if not is_variable:
                            # 定数の場合、属性として検索
                            conditions.append(f"n.arg{i+1} = '{arg}'")
                    
                    if conditions:
                        cypher += " AND ".join(conditions)
                    else:
                        cypher += "true"
                        
                    cypher += " RETURN n"
                    graph_query['cypher'] = cypher
            
            # 他の論理形式の変換も同様に実装可能
            
        except Exception as e:
            self.logger.error(f"Error converting logic to graph query: {str(e)}")
            graph_query['cypher'] = "MATCH (n) RETURN n LIMIT 0"  # 安全なフォールバック
        
        return graph_query
    
    def results_to_logical_form(self, results: List[Dict[str, Any]]) -> str:
        """
        検索結果を論理形式に変換
        
        Args:
            results (List[Dict[str, Any]]): グラフ検索結果
            
        Returns:
            str: 論理形式の結果表現
        """
        if not results:
            return "False"  # 空の結果は偽として表現
            
        # 結果の論理形式への変換
        logical_results = []
        
        try:
            for item in results:
                # ノード情報の抽出
                if 'n' in item:
                    node = item['n']
                    
                    # ノードの種類（ラベル）を述語名として使用
                    if 'labels' in node and node['labels']:
                        predicate = node['labels'][0]
                        
                        # 属性を引数として使用
                        args = []
                        properties = node.get('properties', {})
                        
                        # arg1, arg2, ...の形式で属性を探索
                        i = 1
                        while f'arg{i}' in properties:
                            args.append(str(properties[f'arg{i}']))
                            i += 1
                            
                        # 他の重要な属性も含める
                        for key, value in properties.items():
                            if not key.startswith('arg'):
                                args.append(f"{key}={value}")
                        
                        # 論理形式の構築
                        arg_str = ', '.join(args)
                        logical_form = f"{predicate}({arg_str})"
                        logical_results.append(logical_form)
            
            # 結果の連言 (AND) で結合
            if logical_results:
                return ' ∧ '.join(logical_results)
            else:
                return "True"  # 結果はあるが内容が変換できない場合
                
        except Exception as e:
            self.logger.error(f"Error converting results to logical form: {str(e)}")
            return "Error"
        
        
class SymbolicKnowledgeInterface:
    """シンボリックAIの知識ベース連携インターフェースクラス"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        インターフェースの初期化
        
        Args:
            knowledge_graph (KnowledgeGraph): 知識グラフエンジン
        """
        self.logger = Logger(__name__)
        self.kg = knowledge_graph
        self.rule_converter = RuleConverter()
        
        # ルールストレージの初期化
        self.rules_collection = "symbolic_rules"
        self._ensure_rule_storage()
        
        self.logger.info("SymbolicKnowledgeInterface initialized")
    
    def _ensure_rule_storage(self):
        """ルールストレージの存在確認と初期化"""
        # 知識グラフにルール用コレクションが存在するか確認
        # 存在しない場合は初期化
        try:
            # Neo4jベースの実装例
            cypher_query = f"""
            MERGE (n:Collection {{name: '{self.rules_collection}'}})
            ON CREATE SET n.created = timestamp()
            RETURN n
            """
            self.kg.execute_query(cypher_query)
            
            # ルール関係のインデックス作成
            index_query = f"""
            CREATE INDEX IF NOT EXISTS FOR (n:LogicRule) ON (n.rule_id)
            """
            self.kg.execute_query(index_query)
            
            self.logger.info(f"Rule storage initialized: {self.rules_collection}")
            
        except Exception as e:
            self.logger.error(f"Error initializing rule storage: {str(e)}")
    
    def store_logical_rule(self, rule: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        論理ルールを知識ベースに保存
        
        Args:
            rule (str): 論理ルール
            metadata (Dict[str, Any], optional): ルールに関連するメタデータ
            
        Returns:
            str: 保存されたルールのID
        """
        if not rule:
            self.logger.warning("Attempted to store empty rule")
            return ""
            
        # ルールIDの生成
        rule_id = str(uuid.uuid4())
        
        # ルールのグラフパターンへの変換
        graph_pattern = self.rule_converter.rule_to_graph_pattern(rule)
        
        # メタデータの設定
        if metadata is None:
            metadata = {}
            
        metadata['rule_id'] = rule_id
        metadata['rule_text'] = rule
        metadata['timestamp'] = self.kg.get_current_timestamp()
        
        try:
            # グラフ内にルールノードを作成
            cypher_query = f"""
            MATCH (c:Collection {{name: '{self.rules_collection}'}})
            CREATE (r:LogicRule $metadata)
            CREATE (c)-[:CONTAINS]->(r)
            RETURN r.rule_id
            """
            
            result = self.kg.execute_query(cypher_query, parameters={'metadata': metadata})
            
            # ルールの構造をグラフに保存
            # 簡易実装では構造化されていないルールテキストのみ保存
            # 実際の実装ではより構造化された表現が必要
            
            self.logger.info(f"Stored logical rule with ID: {rule_id}")
            return rule_id
            
        except Exception as e:
            self.logger.error(f"Error storing logical rule: {str(e)}")
            return ""
    
    def get_logical_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        IDによる論理ルールの取得
        
        Args:
            rule_id (str): ルールID
            
        Returns:
            Optional[Dict[str, Any]]: ルール情報（見つからない場合はNone）
        """
        if not rule_id:
            return None
            
        try:
            cypher_query = f"""
            MATCH (r:LogicRule {{rule_id: $rule_id}})
            RETURN r
            """
            
            result = self.kg.execute_query(cypher_query, parameters={'rule_id': rule_id})
            
            if result and len(result) > 0 and 'r' in result[0]:
                rule_node = result[0]['r']
                
                # ルール情報の構築
                rule_info = {
                    'id': rule_node.get('rule_id', ''),
                    'text': rule_node.get('rule_text', ''),
                    'metadata': {}
                }
                
                # メタデータの抽出
                for key, value in rule_node.items():
                    if key not in ['rule_id', 'rule_text']:
                        rule_info['metadata'][key] = value
                
                return rule_info
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving logical rule: {str(e)}")
            return None
    
    def query_with_logic(self, logical_query: str) -> List[Dict[str, Any]]:
        """
        論理クエリを使用して知識ベースを検索
        
        Args:
            logical_query (str): 論理形式のクエリ
            
        Returns:
            List[Dict[str, Any]]: 検索結果
        """
        if not logical_query:
            return []
            
        # 論理クエリからグラフクエリへの変換
        graph_query = self.rule_converter.logic_to_graph_query(logical_query)
        
        if not graph_query.get('cypher'):
            self.logger.warning(f"Failed to convert logical query to graph query: {logical_query}")
            return []
            
        try:
            # グラフクエリの実行
            cypher = graph_query['cypher']
            results = self.kg.execute_query(cypher)
            
            self.logger.info(f"Executed query: {cypher}, found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing logical query: {str(e)}")
            return []
    
    def get_relevant_knowledge(self, context: Union[str, List[str]], max_results: int = 10) -> List[Dict[str, Any]]:
        """
        コンテキストに関連する知識の取得
        
        Args:
            context (Union[str, List[str]]): 検索コンテキスト（単一またはリストのテキスト）
            max_results (int, optional): 最大結果件数
            
        Returns:
            List[Dict[str, Any]]: 関連する知識のリスト
        """
        if not context:
            return []
            
        # コンテキストの標準化
        if isinstance(context, list):
            context_str = ' '.join(context)
        else:
            context_str = context
            
        # キーワード抽出（単純な実装）
        keywords = context_str.split()
        keywords = [k for k in keywords if len(k) > 3]  # 短すぎる単語は除外
        
        if not keywords:
            return []
            
        try:
            # キーワードベースの関連ルール検索
            # 実際には意味的類似性など高度な方法が必要
            conditions = []
            for keyword in keywords[:5]:  # 最大5キーワードを使用
                conditions.append(f"r.rule_text CONTAINS '{keyword}'")
                
            cypher_query = f"""
            MATCH (r:LogicRule)
            WHERE {' OR '.join(conditions)}
            RETURN r
            ORDER BY r.timestamp DESC
            LIMIT {max_results}
            """
            
            results = self.kg.execute_query(cypher_query)
            
            # 結果の整形
            knowledge_items = []
            for item in results:
                if 'r' in item:
                    rule_node = item['r']
                    knowledge_items.append({
                        'id': rule_node.get('rule_id', ''),
                        'rule': rule_node.get('rule_text', ''),
                        'relevance': 0.5,  # ダミーの関連性スコア
                        'metadata': {k: v for k, v in rule_node.items() 
                                   if k not in ['rule_id', 'rule_text']}
                    })
            
            self.logger.info(f"Found {len(knowledge_items)} relevant knowledge items")
            return knowledge_items
            
        except Exception as e:
            self.logger.error(f"Error retrieving relevant knowledge: {str(e)}")
            return []
    
    def add_rule_from_natural_language(self, text: str, logic_transformer: LogicTransformer) -> str:
        """
        自然言語テキストから論理ルールを抽出して追加
        
        Args:
            text (str): 自然言語テキスト
            logic_transformer (LogicTransformer): 論理変換モジュール
            
        Returns:
            str: 追加されたルールID（失敗時は空文字）
        """
        if not text or not logic_transformer:
            return ""
            
        try:
            # テキストから論理表現への変換
            logic_expr = logic_transformer.natural_to_logic(text)
            
            if not logic_expr:
                self.logger.warning(f"Failed to extract logic from text: {text}")
                return ""
                
            # メタデータの準備
            metadata = {
                'source_text': text,
                'extraction_method': 'llm_transformer',
                'timestamp': self.kg.get_current_timestamp()
            }
            
            # ルールの保存
            rule_id = self.store_logical_rule(logic_expr, metadata)
            
            self.logger.info(f"Added rule from natural language: {rule_id}")
            return rule_id
            
        except Exception as e:
            self.logger.error(f"Error adding rule from natural language: {str(e)}")
            return ""
    
    def search_rules(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        ルールの検索
        
        Args:
            query (str): 検索クエリ
            limit (int, optional): 最大結果件数
            
        Returns:
            List[Dict[str, Any]]: ルールのリスト
        """
        if not query:
            return []
            
        try:
            # ルールテキスト内の部分一致検索
            cypher_query = f"""
            MATCH (r:LogicRule)
            WHERE r.rule_text CONTAINS $query
            RETURN r
            LIMIT {limit}
            """
            
            results = self.kg.execute_query(cypher_query, parameters={'query': query})
            
            # 結果の整形
            rules = []
            for item in results:
                if 'r' in item:
                    rule_node = item['r']
                    rules.append({
                        'id': rule_node.get('rule_id', ''),
                        'text': rule_node.get('rule_text', ''),
                        'metadata': {k: v for k, v in rule_node.items() 
                                   if k not in ['rule_id', 'rule_text']}
                    })
            
            self.logger.info(f"Found {len(rules)} rules matching query: {query}")
            return rules
            
        except Exception as e:
            self.logger.error(f"Error searching rules: {str(e)}")
            return []


class KnowledgeBaseInterface:
    def __init__(self):
        pass

    def query(self, query_str):
        raise NotImplementedError("query() must be implemented by subclass")

    def add_fact(self, fact):
        raise NotImplementedError("add_fact() must be implemented by subclass")

    def remove_fact(self, fact):
        raise NotImplementedError("remove_fact() must be implemented by subclass")
