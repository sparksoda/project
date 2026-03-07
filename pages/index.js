// WebLLMをインポート
const webllm = await import("https://esm.run/@mlc-ai/web-llm");

// DOM要素
const statusEl = document.getElementById('status');
const loadBtn = document.getElementById('load-btn');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const streamBtn = document.getElementById('stream-btn');
const chatLog = document.getElementById('chat-log');

// グローバル変数
let engine = null;
let isModelLoaded = false;

// モデルをロードする関数
async function loadModel() {
    statusEl.textContent = "モデルをロード中...";
    loadBtn.disabled = true;
    
    try {
        // 使用するモデル
        const selectedModel = "Qwen2.5-1.5B-Instruct-q4f32_1-MLC";
        
        // モデル設定
        const appConfig = {
            model_list: [
                {
                    model: "https://huggingface.co/mlc-ai/Qwen2.5-1.5B-Instruct-q4f32_1-MLC",
                    model_id: selectedModel,
                    model_lib:
                        webllm.modelLibURLPrefix +
                        webllm.modelVersion +
                        "/Qwen2-1.5B-Instruct-q4f32_1-MLC-ctx4k_cs1k-webgpu.wasm",
                },
            ],
        };

        // モデルのロード
        engine = await webllm.CreateMLCEngine(
            selectedModel,
            {
                // ログ出力
                initProgressCallback: (initProgress) => {
                    statusEl.textContent = `ロード中... ${Math.round(initProgress.progress * 100)}% - ${initProgress.text || ""}`;
                    console.log(initProgress);
                },
                logLevel: "INFO",
            },
            appConfig
        );
        
        // ロード完了
        statusEl.textContent = `モデル「${selectedModel}」が正常にロードされました`;
        isModelLoaded = true;
        
        // 入力を有効化
        userInput.disabled = false;
        sendBtn.disabled = false;
        streamBtn.disabled = false;
        
        // システムメッセージ
        addSystemMessage("モデルのロードが完了しました。メッセージを入力してください。");
        
    } catch (error) {
        console.error("モデルのロードに失敗:", error);
        statusEl.textContent = `モデルのロードに失敗しました: ${error.message}`;
        loadBtn.disabled = false;
    }
}

// 通常のテキスト生成（非ストリーミング）
async function generateText() {
    if (!isModelLoaded) return;
    
    const message = userInput.value.trim();
    if (message === "") return;
    
    // ユーザーメッセージの表示
    addUserMessage(message);
    userInput.value = "";
    
    // 入力を無効化
    userInput.disabled = true;
    sendBtn.disabled = true;
    streamBtn.disabled = true;
    
    try {
        // 生成開始
        const startTime = performance.now();
        
        // AIモデルに問い合わせ
        const reply = await engine.chat.completions.create({
            messages: [
                { role: "system", content: "You are a helpful AI assistant." },
                { role: "user", content: message }
            ],
        });
        
        // 応答時間の計算
        const endTime = performance.now();
        const timeTaken = ((endTime - startTime) / 1000).toFixed(2);
        
        // 結果の表示
        const content = reply.choices[0].message.content;
        addAIMessage(content);
        
        // システムメッセージで時間を表示
        addSystemMessage(`応答時間: ${timeTaken}秒`);
        
    } catch (error) {
        console.error("テキスト生成に失敗:", error);
        addSystemMessage(`エラー: ${error.message}`);
    } finally {
        // 入力を再度有効化
        userInput.disabled = false;
        sendBtn.disabled = false;
        streamBtn.disabled = false;
    }
}

// ストリーミングテキスト生成
async function generateStreamingText() {
    if (!isModelLoaded) return;
    
    const message = userInput.value.trim();
    if (message === "") return;
    
    // ユーザーメッセージの表示
    addUserMessage(message);
    userInput.value = "";
    
    // 入力を無効化
    userInput.disabled = true;
    sendBtn.disabled = true;
    streamBtn.disabled = true;
    
    try {
        // AIメッセージの追加（空）
        const aiMessageEl = document.createElement('div');
        aiMessageEl.className = 'ai-message';
        const aiSpan = document.createElement('span');
        aiMessageEl.appendChild(aiSpan);
        chatLog.appendChild(aiMessageEl);
        
        // 生成開始
        const startTime = performance.now();
        
        // ストリーミング生成
        const chunks = await engine.chat.completions.create({
            messages: [
                { role: "system", content: "You are a helpful AI assistant." },
                { role: "user", content: message }
            ],
            temperature: 1,
            stream: true,
            stream_options: { include_usage: true },
        });
        
        let fullResponse = '';
        
        // ストリーミングで返されるチャンクを処理
        for await (const chunk of chunks) {
            const content = chunk.choices[0] ? chunk.choices[0].delta.content : "";
            fullResponse += content;
            aiSpan.textContent = fullResponse;
            
            // 自動スクロール
            chatLog.scrollTop = chatLog.scrollHeight;
        }
        
        // 応答時間の計算
        const endTime = performance.now();
        const timeTaken = ((endTime - startTime) / 1000).toFixed(2);
        
        // システムメッセージで時間を表示
        addSystemMessage(`応答時間: ${timeTaken}秒`);
        
    } catch (error) {
        console.error("ストリーミング生成に失敗:", error);
        addSystemMessage(`エラー: ${error.message}`);
    } finally {
        // 入力を再度有効化
        userInput.disabled = false;
        sendBtn.disabled = false;
        streamBtn.disabled = false;
    }
}

// ユーザーメッセージを追加
function addUserMessage(text) {
    const messageEl = document.createElement('div');
    messageEl.className = 'user-message';
    
    const span = document.createElement('span');
    span.textContent = text;
    
    messageEl.appendChild(span);
    chatLog.appendChild(messageEl);
    chatLog.scrollTop = chatLog.scrollHeight;
}

// AIメッセージを追加
function addAIMessage(text) {
    const messageEl = document.createElement('div');
    messageEl.className = 'ai-message';
    
    const span = document.createElement('span');
    span.textContent = text;
    
    messageEl.appendChild(span);
    chatLog.appendChild(messageEl);
    chatLog.scrollTop = chatLog.scrollHeight;
}

// システムメッセージを追加
function addSystemMessage(text) {
    const messageEl = document.createElement('div');
    messageEl.className = 'system-message';
    messageEl.textContent = text;
    
    chatLog.appendChild(messageEl);
    chatLog.scrollTop = chatLog.scrollHeight;
}

// イベントリスナーの設定
loadBtn.addEventListener('click', loadModel);
sendBtn.addEventListener('click', generateText);
streamBtn.addEventListener('click', generateStreamingText);

// Enterキーでの送信（日本語入力の確定を考慮）
let isComposing = false;

// IME入力開始時
userInput.addEventListener('compositionstart', () => {
    isComposing = true;
});

// IME入力完了時
userInput.addEventListener('compositionend', () => {
    isComposing = false;
});

// キー入力時
userInput.addEventListener('keydown', (e) => {
    // isComposingがtrueの場合は日本語変換中なので処理をスキップ
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
        e.preventDefault();
        generateText();
    }
});
