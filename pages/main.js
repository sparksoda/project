// ========================================
// GGUF Model Viewer with WebGPU Support
// ========================================

class GGUFModelViewer {
    constructor() {
        this.modelData = null;
        this.gpuContext = null;
        this.modelMetadata = null;
        this.isRunning = false;
        this.tensorCache = new Map();

        this.initializeElements();
        this.checkWebGPUSupport();
        this.attachEventListeners();
    }

    initializeElements() {
        this.elements = {
            modelFile: document.getElementById('modelFile'),
            prompt: document.getElementById('prompt'),
            contextSize: document.getElementById('contextSize'),
            temperature: document.getElementById('temperature'),
            runBtn: document.getElementById('runBtn'),
            resetBtn: document.getElementById('resetBtn'),
            statusBox: document.getElementById('statusBox'),
            outputBox: document.getElementById('outputBox'),
            modelSize: document.getElementById('modelSize'),
            layerCount: document.getElementById('layerCount'),
            inferenceTime: document.getElementById('inferenceTime'),
            tokenSpeed: document.getElementById('tokenSpeed'),
            gpuMemory: document.getElementById('gpuMemory'),
            status: document.getElementById('status'),
            gpuCanvas: document.getElementById('gpuCanvas'),
            canvasPlaceholder: document.getElementById('canvasPlaceholder'),
            gpuStatus: document.getElementById('gpuStatus'),
            gpuStatusText: document.getElementById('gpuStatusText'),
        };
    }

    attachEventListeners() {
        this.elements.modelFile.addEventListener('change', (e) => this.handleModelUpload(e));
        this.elements.runBtn.addEventListener('click', () => this.runInference());
        this.elements.resetBtn.addEventListener('click', () => this.reset());
    }

    // ==================== WebGPU Support ====================
    async checkWebGPUSupport() {
        try {
            if (!navigator.gpu) {
                this.updateGPUStatus('WebGPU is not supported', false);
                return;
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                this.updateGPUStatus('No GPU adapter found', false);
                return;
            }

            this.gpuContext = await adapter.requestDevice();
            this.updateGPUStatus('WebGPU Ready', true);
            console.log('✅ WebGPU initialized');
        } catch (error) {
            this.updateGPUStatus(`WebGPU Error: ${error.message}`, false);
            console.error('❌ WebGPU initialization failed:', error);
        }
    }

    updateGPUStatus(message, supported) {
        const statusEl = this.elements.gpuStatus;
        const textEl = this.elements.gpuStatusText;

        statusEl.className = `webgpu-status ${supported ? 'supported' : 'unsupported'}`;
        textEl.textContent = message;

        if (supported) {
            statusEl.innerHTML = `<span style="color: #4caf50; margin-right: 8px;">✓</span>${message}`;
        } else {
            statusEl.innerHTML = `<span style="color: #f44336; margin-right: 8px;">✕</span>${message}`;
        }
    }

    // ==================== GGUF Parser ====================
    async handleModelUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            this.updateStatus('info', `ファイルを読み込み中: ${file.name}`);
            this.elements.status.textContent = '読み込み中';

            const arrayBuffer = await file.arrayBuffer();
            this.modelData = new Uint8Array(arrayBuffer);

            await this.parseGGUFHeader();
            this.elements.modelSize.textContent = this.formatBytes(this.modelData.byteLength);
            this.elements.runBtn.disabled = false;

            this.updateStatus('success', `✓ モデル読み込み完了: ${file.name} (${this.formatBytes(file.size)})`);
            this.elements.status.textContent = '準備完了';
        } catch (error) {
            this.updateStatus('error', `❌ エラー: ${error.message}`);
            this.elements.status.textContent = 'エラー';
            console.error('Model loading error:', error);
        }
    }

    async parseGGUFHeader() {
        // GGUF Format Header Parser
        // Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

        const view = new DataView(this.modelData.buffer);
        let offset = 0;

        // Magic number (4 bytes)
        const magic = view.getUint32(offset, true);
        offset += 4;

        if (magic !== 0x46554747) { // "GGUF"
            throw new Error('Invalid GGUF magic number');
        }

        // Version (4 bytes)
        const version = view.getUint32(offset, true);
        offset += 4;

        // Tensor count (8 bytes)
        const tensorCount = this.readUint64(view, offset);
        offset += 8;

        // Metadata KV count (8 bytes)
        const kvCount = this.readUint64(view, offset);
        offset += 8;

        // Parse metadata
        this.modelMetadata = {};
        for (let i = 0; i < Number(kvCount); i++) {
            const { key, value, newOffset } = this.readMetadataKV(offset);
            offset = newOffset;
            this.modelMetadata[key] = value;
        }

        console.log('GGUF Metadata:', this.modelMetadata);

        // Update UI
        this.elements.layerCount.textContent = 
            this.modelMetadata['llama.block_count'] || 
            this.modelMetadata['gpt_neox.block_count'] || 
            '?';
    }

    readUint64(view, offset) {
        const low = view.getUint32(offset, true);
        const high = view.getUint32(offset + 4, true);
        return BigInt(high) * (1n << 32n) + BigInt(low);
    }

    readMetadataKV(offset) {
        const view = new DataView(this.modelData.buffer);

        // Key length (4 bytes)
        const keyLen = view.getUint32(offset, true);
        offset += 4;

        // Key (UTF-8)
        const keyBytes = this.modelData.slice(offset, offset + keyLen);
        const key = new TextDecoder().decode(keyBytes);
        offset += keyLen;

        // Value type (4 bytes)
        const valueType = view.getUint32(offset, true);
        offset += 4;

        let value;
        const result = this.parseMetadataValue(valueType, offset);
        value = result.value;
        offset = result.offset;

        return { key, value, newOffset: offset };
    }

    parseMetadataValue(type, offset) {
        const view = new DataView(this.modelData.buffer);

        // GGUF Value Types
        const GGUF_TYPE = {
            0: 'uint8',
            1: 'int8',
            2: 'uint16',
            3: 'int16',
            4: 'uint32',
            5: 'int32',
            6: 'float32',
            7: 'bool',
            8: 'string',
            9: 'array',
            10: 'uint64',
            11: 'int64',
            12: 'float64',
        };

        switch (type) {
            case 8: // string
                const strLen = view.getUint32(offset, true);
                offset += 4;
                const strBytes = this.modelData.slice(offset, offset + strLen);
                const str = new TextDecoder().decode(strBytes);
                offset += strLen;
                return { value: str, offset };

            case 4: // uint32
                const val32 = view.getUint32(offset, true);
                offset += 4;
                return { value: val32, offset };

            case 5: // int32
                const valInt32 = view.getInt32(offset, true);
                offset += 4;
                return { value: valInt32, offset };

            case 6: // float32
                const valFloat = view.getFloat32(offset, true);
                offset += 4;
                return { value: valFloat, offset };

            default:
                return { value: null, offset: offset + 8 };
        }
    }

    // ==================== WebGPU Inference ====================
    async runInference() {
        if (!this.modelData || !this.gpuContext) {
            this.updateStatus('error', '❌ モデルが読み込まれていないか、WebGPUが利用できません');
            return;
        }

        if (this.isRunning) {
            this.updateStatus('error', '❌ 既に実行中です');
            return;
        }

        this.isRunning = true;
        this.elements.runBtn.disabled = true;

        try {
            const prompt = this.elements.prompt.value || 'Hello';
            const contextSize = parseInt(this.elements.contextSize.value);
            const temperature = parseFloat(this.elements.temperature.value);

            this.elements.status.textContent = '推論中';
            this.updateStatus('info', `🔄 推論を開始: "${prompt}"`);

            const startTime = performance.now();

            // Setup WebGPU computation
            await this.setupWebGPUCompute();

            // Simulate inference
            const output = await this.simulateModelInference(prompt, contextSize, temperature);

            const endTime = performance.now();
            const inferenceTime = endTime - startTime;

            this.elements.inferenceTime.textContent = `${inferenceTime.toFixed(2)}ms`;
            this.elements.tokenSpeed.textContent = `${(contextSize / (inferenceTime / 1000)).toFixed(0)} tok/s`;

            this.elements.outputBox.textContent = output;
            this.updateStatus('success', `✓ 推論完了 (${inferenceTime.toFixed(2)}ms)`);
            this.elements.status.textContent = '完了';

            // Render visualization
            await this.renderWebGPUVisualization();

        } catch (error) {
            this.updateStatus('error', `❌ 推論エラー: ${error.message}`);
            this.elements.status.textContent = 'エラー';
            console.error('Inference error:', error);
        } finally {
            this.isRunning = false;
            this.elements.runBtn.disabled = false;
        }
    }

    async setupWebGPUCompute() {
        if (!this.gpuContext) return;

        // Create a simple compute shader
        const shaderCode = `
            @group(0) @binding(0)
            var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx < arrayLength(&data)) {
                    data[idx] = data[idx] * 2.0;
                }
            }
        `;

        try {
            const shaderModule = this.gpuContext.createShaderModule({ code: shaderCode });
            this.computePipeline = this.gpuContext.createComputePipeline({
                layout: 'auto',
                compute: { module: shaderModule, entryPoint: 'main' },
            });

            console.log('✓ WebGPU compute pipeline created');
        } catch (error) {
            console.warn('Compute shader compilation warning:', error);
        }
    }

    async simulateModelInference(prompt, contextSize, temperature) {
        // Simulate token generation
        const tokens = this.tokenize(prompt);
        let output = prompt;

        // Simulate generating contextSize tokens
        const generationSteps = Math.min(10, contextSize / 50);
        for (let i = 0; i < generationSteps; i++) {
            // Simulate sampling with temperature
            const nextToken = this.sampleToken(temperature);
            output += nextToken;
        }

        return output;
    }

    tokenize(text) {
        // Simple tokenization (in real implementation, use actual tokenizer)
        return text.split(/\s+/).filter(t => t.length > 0);
    }

    sampleToken(temperature) {
        // Simulate token sampling with temperature
        const tokens = [' the', ' model', ' is', ' working', ' well', ' now', ' here', ' there'];
        const idx = Math.floor(Math.random() * tokens.length);
        return tokens[idx];
    }

    async renderWebGPUVisualization() {
        const canvas = this.elements.gpuCanvas;
        const placeholder = this.elements.canvasPlaceholder;

        if (!navigator.gpu) {
            placeholder.textContent = 'WebGPU not available';
            return;
        }

        const context = canvas.getContext('webgpu');
        if (!context) {
            placeholder.textContent = 'WebGPU canvas context failed';
            return;
        }

        // Setup canvas
        canvas.width = 400;
        canvas.height = 300;

        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        context.configure({
            device: this.gpuContext,
            format: canvasFormat,
        });

        // Create shader
        const shaderCode = `
            @vertex
            fn vs(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
                var pos = array<vec2<f32>, 3>(
                    vec2<f32>(0.0, 0.5),
                    vec2<f32>(-0.5, -0.5),
                    vec2<f32>(0.5, -0.5)
                );
                return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
            }

            @fragment
            fn fs() -> @location(0) vec4<f32> {
                return vec4<f32>(0.4, 0.6, 1.0, 1.0);
            }
        `;

        const shaderModule = this.gpuContext.createShaderModule({ code: shaderCode });

        const pipeline = this.gpuContext.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vs',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs',
                targets: [{ format: canvasFormat }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        // Render
        const commandEncoder = this.gpuContext.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: textureView,
                    clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        });

        renderPass.setPipeline(pipeline);
        renderPass.draw(3);
        renderPass.end();

        this.gpuContext.queue.submit([commandEncoder.finish()]);

        // Show canvas
        canvas.classList.add('visible');
        placeholder.style.display = 'none';

        this.elements.gpuMemory.textContent = 'GPU Rendering ✓';
    }

    // ==================== Utility Methods ====================
    updateStatus(type, message) {
        const box = this.elements.statusBox;
        box.textContent = message;
        box.className = `status-box ${type}`;
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    reset() {
        this.modelData = null;
        this.modelMetadata = null;
        this.tensorCache.clear();
        this.elements.modelFile.value = '';
        this.elements.prompt.value = '';
        this.elements.outputBox.textContent = '出力がここに表示されます';
        this.elements.statusBox.textContent = 'モデルを選択してください';
        this.elements.statusBox.className = 'status-box info';
        this.elements.modelSize.textContent = '-';
        this.elements.layerCount.textContent = '-';
        this.elements.inferenceTime.textContent = '-';
        this.elements.tokenSpeed.textContent = '-';
        this.elements.gpuMemory.textContent = '-';
        this.elements.status.textContent = '待機中';
        this.elements.runBtn.disabled = true;
        this.elements.gpuCanvas.classList.remove('visible');
        this.elements.canvasPlaceholder.style.display = 'flex';
    }
}

// ==================== Application Startup ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 GGUF Model Viewer initializing...');
    new GGUFModelViewer();
    console.log('✓ Application ready');
});
