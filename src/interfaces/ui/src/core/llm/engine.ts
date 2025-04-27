/**
 * LLMエンジン - 言語モデル処理のTypeScriptインターフェース
 * 
 * このモジュールは、バックエンドのLLMエンジンとのインターフェースを提供します。
 * UIコンポーネントから使用され、APIを通じてバックエンドのLLMエンジンと通信します。
 */

interface LLMProviderConfig {
  [key: string]: any;
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ChatResponse {
  content: string;
  [key: string]: any;
}

class LLMEngine {
  private config: any;
  private defaultProvider: string = 'openai';
  private apiBasePath: string = '/api';

  /**
   * LLMエンジンインターフェースを初期化
   * 
   * @param config 設定オブジェクトまたはパス
   */
  constructor(config: any) {
    this.config = config;
    
    // 設定からAPIパスとデフォルトプロバイダーを取得
    if (config && config.api && config.api.basePath) {
      this.apiBasePath = config.api.basePath;
    }
    
    if (config && config.llm && config.llm.default_provider) {
      this.defaultProvider = config.llm.default_provider;
    }
    
    console.log(`LLMEngine initialized with default provider: ${this.defaultProvider}`);
  }

  /**
   * テキスト生成を実行
   * 
   * @param prompt 入力プロンプト
   * @param provider 使用するプロバイダー（省略時はデフォルト）
   * @param options その他のオプション
   * @returns 生成されたテキストを含むPromise
   */
  async generate(prompt: string, provider?: string, options: any = {}): Promise<string> {
    const providerName = provider || this.defaultProvider;
    
    try {
      const response = await fetch(`${this.apiBasePath}/llm/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          provider: providerName,
          ...options
        }),
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      return data.text;
    } catch (error) {
      console.error('LLM generation error:', error);
      throw error;
    }
  }

  /**
   * チャット形式での生成を実行
   * 
   * @param messages 会話履歴
   * @param provider 使用するプロバイダー（省略時はデフォルト）
   * @param options その他のオプション
   * @returns 生成結果を含むPromise
   */
  async chat(messages: Message[], provider?: string, options: any = {}): Promise<ChatResponse> {
    const providerName = provider || this.defaultProvider;
    
    try {
      const response = await fetch(`${this.apiBasePath}/llm/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages,
          provider: providerName,
          ...options
        }),
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('LLM chat error:', error);
      throw error;
    }
  }

  /**
   * テキストの埋め込みベクトルを生成
   * 
   * @param text 入力テキスト
   * @param provider 使用するプロバイダー（省略時はデフォルト）
   * @returns 埋め込みベクトルを含むPromise
   */
  async embed(text: string, provider?: string): Promise<number[]> {
    const providerName = provider || this.defaultProvider;
    
    try {
      const response = await fetch(`${this.apiBasePath}/llm/embed`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          provider: providerName
        }),
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      return data.embedding;
    } catch (error) {
      console.error('LLM embedding error:', error);
      throw error;
    }
  }
}

export { LLMEngine, Message, ChatResponse };
