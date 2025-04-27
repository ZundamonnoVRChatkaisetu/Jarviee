from .base import ProgrammingLanguage, LanguageFeature, SyntaxElement

class JavaScriptLanguage(ProgrammingLanguage):
    def __init__(self):
        super().__init__(
            name="JavaScript",
            file_extensions=[".js", ".mjs", ".cjs"]
        )
        self.features.update([
            LanguageFeature.DYNAMIC_TYPING,
            LanguageFeature.OBJECT_ORIENTED,
            LanguageFeature.FUNCTIONAL,
            LanguageFeature.PROCEDURAL,
            LanguageFeature.FIRST_CLASS_FUNCTIONS,
            LanguageFeature.CLOSURES,
            LanguageFeature.MODULE_SYSTEM,
            LanguageFeature.PACKAGE_MANAGER,
        ])
        # 必要に応じてsyntax_patterns等も追加
