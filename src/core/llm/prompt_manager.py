"""
プロンプト管理システム

このモジュールは、LLMとの対話に使用するプロンプトの管理、最適化、
カスタマイズを担当します。目的に応じたプロンプトテンプレートの選択、
コンテキストに基づく動的なプロンプト生成、プロンプトの履歴管理などを行います。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import re
import string


class PromptTemplate:
    """プロンプトテンプレートを表すクラス"""
    
    def __init__(self, template: str, metadata: Dict[str, Any] = None):
        """
        プロンプトテンプレートを初期化
        
        Args:
            template: テンプレート文字列（変数は{variable_name}形式）
            metadata: テンプレートに関するメタデータ
        """
        self.template = template
        self.metadata = metadata or {}
        self._validate_template()
    
    def _validate_template(self):
        """テンプレートが有効か検証"""
        # 単純な検証: 括弧の対応チェック
        if self.template.count('{') != self.template.count('}'):
            raise ValueError("Invalid template: mismatched braces")
        
        # 変数名の形式チェック
        var_pattern = r'{([^{}]*)}'
        for var_name in re.findall(var_pattern, self.template):
            if not var_name.isidentifier() and not var_name in ['', ' ']:
                raise ValueError(f"Invalid variable name in template: '{var_name}'")
    
    def format(self, **kwargs) -> str:
        """
        テンプレートを変数で埋めて文字列を生成
        
        Args:
            **kwargs: テンプレート変数に対応する値
            
        Returns:
            生成された文字列
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing required variable in template: {e}")


class PromptManager:
    """プロンプト管理を担当するクラス"""
    
    def __init__(self, templates_dir: str = None):
        """
        プロンプトマネージャーを初期化
        
        Args:
            templates_dir: テンプレートファイルのディレクトリ
        """
        self.templates: Dict[str, PromptTemplate] = {}
        self.templates_dir = templates_dir or os.path.join("config", "prompts")
        self.logger = logging.getLogger(__name__)
        
        # 基本テンプレートをロード
        self._load_default_templates()
        
        # テンプレートディレクトリからロード
        if os.path.exists(self.templates_dir):
            self._load_templates_from_dir()
    
    def _load_default_templates(self):
        """基本的なプロンプトテンプレートをロード"""
        
        # 基本的な会話テンプレート
        self.templates["basic_chat"] = PromptTemplate(
            template=(
                "以下はJarvieeとユーザーとの会話です。\n"
                "Jarvieeはアイアンマンに登場するAIアシスタント、ジャーヴィスの性格を持ちます。\n"
                "ユーザーを「てゅん」と呼び、親しみを持ちながらも効率的に対応します。\n\n"
                "{conversation}\n"
                "Jarviee: "
            ),
            metadata={
                "name": "Basic Chat Template",
                "purpose": "General conversation",
                "version": "1.0"
            }
        )
        
        # 知識獲得テンプレート
        self.templates["knowledge_acquisition"] = PromptTemplate(
            template=(
                "以下のトピックについて重要な情報を整理・構造化してください。\n"
                "トピック: {topic}\n"
                "コンテキスト: {context}\n\n"
                "以下の形式で回答してください:\n"
                "- 主要概念と定義\n"
                "- 重要な関係性\n"
                "- 基本原則と応用\n"
                "- 関連する技術や領域\n"
            ),
            metadata={
                "name": "Knowledge Acquisition Template",
                "purpose": "Structured knowledge extraction",
                "version": "1.0"
            }
        )
        
        # コード生成テンプレート
        self.templates["code_generation"] = PromptTemplate(
            template=(
                "次の要件に基づいて{language}のコードを作成してください。\n"
                "要件: {requirements}\n\n"
                "コンテキスト情報:\n{context}\n\n"
                "以下の点に注意してください:\n"
                "- コードは{language}の最新のベストプラクティスに従う\n"
                "- エラー処理を適切に実装する\n"
                "- コードは効率的かつ読みやすく\n"
                "- 必要に応じてコメントを含める\n\n"
                "コード:"
            ),
            metadata={
                "name": "Code Generation Template",
                "purpose": "Generate code in specified language",
                "version": "1.0"
            }
        )
        
        # 自己思考テンプレート
        self.templates["self_reflection"] = PromptTemplate(
            template=(
                "以下の問題について、段階的に考えてください。\n"
                "問題: {problem}\n\n"
                "考慮すべき情報:\n{context}\n\n"
                "次のステップで思考を進めてください:\n"
                "1. 問題の理解と分析\n"
                "2. 重要な事実と仮定の整理\n"
                "3. 可能な解決アプローチの検討\n"
                "4. アプローチの評価と選択\n"
                "5. 詳細な解決策の導出\n"
                "6. 解決策の検証\n"
            ),
            metadata={
                "name": "Self-reflection Template",
                "purpose": "Structured thinking process",
                "version": "1.0"
            }
        )
    
    def _load_templates_from_dir(self):
        """テンプレートディレクトリからテンプレートをロード"""
        try:
            for file in os.listdir(self.templates_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(self.templates_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            template_data = json.load(f)
                            
                            # 必須フィールドの検証
                            if 'name' not in template_data or 'template' not in template_data:
                                self.logger.warning(f"Skipping invalid template file: {file}")
                                continue
                                
                            template_name = template_data.pop('name')
                            template_str = template_data.pop('template')
                            
                            # テンプレートの作成
                            self.templates[template_name] = PromptTemplate(
                                template=template_str,
                                metadata=template_data
                            )
                            
                            self.logger.info(f"Loaded template: {template_name}")
                    except Exception as e:
                        self.logger.error(f"Error loading template from {file}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading templates from directory: {e}")
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """
        名前からテンプレートを取得
        
        Args:
            template_name: テンプレート名
            
        Returns:
            テンプレートインスタンス、存在しない場合はNone
        """
        return self.templates.get(template_name)
    
    def create_prompt(self, template_name: str, **kwargs) -> str:
        """
        テンプレートを使用してプロンプトを生成
        
        Args:
            template_name: 使用するテンプレート名
            **kwargs: テンプレート変数の値
            
        Returns:
            生成されたプロンプト
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return template.format(**kwargs)
    
    def add_template(self, name: str, template: str, metadata: Dict[str, Any] = None) -> None:
        """
        新しいテンプレートを追加
        
        Args:
            name: テンプレート名
            template: テンプレート文字列
            metadata: メタデータ
        """
        self.templates[name] = PromptTemplate(template=template, metadata=metadata)
        self.logger.info(f"Added new template: {name}")
    
    def save_template(self, name: str, file_path: Optional[str] = None) -> None:
        """
        テンプレートをファイルに保存
        
        Args:
            name: 保存するテンプレート名
            file_path: 保存先パス（省略時はテンプレートディレクトリ）
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        if not file_path:
            os.makedirs(self.templates_dir, exist_ok=True)
            file_path = os.path.join(self.templates_dir, f"{name}.json")
        
        data = {
            "name": name,
            "template": template.template,
            **template.metadata
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Saved template '{name}' to {file_path}")
