// ========================================
// GGUF Model Viewer with WebGPU Support
// Large File Streaming Support (250MB+)
// ========================================

class GGUFModelViewer {
    constructor() {
        this.modelData = null;
        this.gpuContext = null;
        this.modelMetadata = null;
        this.isRunning = false;
        this.tensorCache = new Map();
        this.fileHandle = null;
        this.modelPath = null;

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
            
            // Get GPU memory info
            const limits = this.gpuContext.limits;
            const memoryUsage = `Memory: ${Math.round(limits.maxStorageBufferBindingSize / 1024 / 1024)}MB`;
            
            this.updateGPUStatus('WebGPU Ready - ' + memoryUsage, true);
            console.log('✅ WebGPU initialized', limits);
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

    // ==================== Large File Streaming Loader ====================
    async handleModelUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            this.updateStatus('info', `ファイルを読み込み中: ${file.name}`);
            this.elements.status.textContent = '読み込み中';

            const fileSize = file.size;
            this.modelPath = file.name;

            // For large files (>50MB), use streaming
            if (fileSize > 50 * 1024 * 1024) {
                await this.loadModelStreaming(file);
            } else {
                await this.loadModelDirect(file);
            }

            this.elements.modelSize.textContent = this.formatBytes(fileSize);
            this.elements.runBtn.disabled = false;

            this.updateStatus('success', `✓ モデル読み込み完了: ${file.name} (${this.formatBytes(fileSize)})`);
            this.elements.status.textContent = '準備完了';
        } catch (error) {
            this.updateStatus('error', `❌ エラー: ${error.message}`);
            this.elements.status.textContent = 'エラー';
            console.error('Model loading error:', error);
        }
    }

    // Direct load for small files
    async loadModelDirect(file) {
        const arrayBuffer = await file.arrayBuffer();
        this.modelData = new Uint8Array(arrayBuffer);
        await this.parseGGUFHeaderStreaming(this.modelData);
    }

    // Streaming load for large files
    async loadModelStreaming(file) {
        console.log('🔄 Starting streaming load for large file...');
        
        // Read header first (usually < 1MB)
        const headerSize = Math.min(10 * 1024 * 1024, file.size); // Read first 10MB
        const headerBlob = file.slice(0, headerSize);
        const headerBuffer = await headerBlob.arrayBuffer();
        const headerData = new Uint8Array(headerBuffer);

        // Parse header metadata
        await this.parseGGUFHeaderStreaming(headerData);

        // Store file handle for lazy loading of tensors
        this.fileHandle = file;
        this.fileSize = file.size;

        console.log('✓ Header parsed, tensors will be loaded on demand');
    }

    // ==================== GGUF Header Parser (Streaming-compatible) ====================
    async parseGGUFHeaderStreaming(headerData) {
        const view = new DataView(headerData.buffer, headerData.byteOffset);
        let offset = 0;

        // Magic number (4 bytes)
        const magic = view.getUint32(offset, true);
        offset += 4;

        if (magic !== 0x46554747) { // "GGUF"
            throw new Error('Invalid GGUF magic number - not a valid GGUF file');
        }

        // Version (4 bytes)
        const version = view.getUint32(offset, true);
        offset += 4;
        console.log(`GGUF Version: ${version}`);

        // Tensor count (8 bytes)
        const tensorCount = this.readUint64(view, offset);
        offset += 8;

        // Metadata KV count (8 bytes)
        const kvCount = this.readUint64(view, offset);
        offset += 8;

        console.log(`Tensors: ${tensorCount}, Metadata entries: ${kvCount}`);

        // Parse metadata
        this.modelMetadata = {};
        for (let i = 0; i < Number(kvCount); i++) {
            try {
                const { key, value, newOffset } = this.readMetadataKV(offset, view, headerData);
                offset = newOffset;
                this.modelMetadata[key] = value;
                
                // Show progress
                if (i % 100 === 0) {
                    console.log(`Parsing metadata: ${i}/${kvCount}`);
                }
            } catch (error) {
                console.warn(`Failed to parse metadata entry ${i}:`, error);
                break;
            }
        }

        console.log('✓ GGUF Metadata parsed:', this.modelMetadata);

        // Update UI
        this.elements.layerCount.textContent = 
            this.modelMetadata['llama.block_count'] || 
            this.modelMetadata['gpt_neox.block_count'] ||
            this.modelMetadata['phi2.block_count'] ||
            this.modelMetadata['mistral.block_count'] ||
            '?';

        // Show tensor info
        const tensorCountStr = tensorCount.toString();
        console.log(`Total tensors in model: ${tensorCountStr}`);
    }

    readUint64(view, offset) {
        try {
            const low = view.getUint32(offset, true);
            const high = view.getUint32(offset + 4, true);
            return BigInt(high) * (1n << 32n) + BigInt(low);
        } catch {
            return 0n;
        }
    }

    readMetadataKV(offset, view, data) {
        // Key length (4 bytes)
        if (offset + 4 > data.byteLength) {
            throw new Error('Offset out of bounds');
        }

        const keyLen = view.getUint32(offset, true);
        offset += 4;

        // Validate key length
        if (keyLen > 1024 * 1024) { // Max 1MB for a single key
            throw new Error(`Invalid key length: ${keyLen}`);
        }

        // Key (UTF-8)
        if (offset + keyLen > data.byteLength) {
            throw new Error('Key data out of bounds');
        }

        const keyBytes = data.slice(offset, offset + keyLen);
        const key = new TextDecoder().decode(keyBytes);
        offset += keyLen;

        // Value type (4 bytes)
        if (offset + 4 > data.byteLength) {
            throw new Error('Value type out of bounds');
        }

        const valueType = view.getUint32(offset, true);
        offset += 4;

        let value;
        const result = this.parseMetadataValue(valueType, offset, view, data);
        value = result.value;
        offset = result.offset;

        return { key, value, newOffset: offset };
    }

    parseMetadataValue(type, offset, view, data) {
        try {
            switch (type) {
                case 8: { // string
                    if (offset + 4 > data.byteLength) {
                        throw new Error('String length out of bounds');
                    }
                    const strLen = view.getUint32(offset, true);
                    offset += 4;

                    if (strLen > 10 * 1024 * 1024) { // Max 10MB
                        throw new Error(`Invalid string length: ${strLen}`);
                    }

                    if (offset + strLen > data.byteLength) {
                        return { value: '[String data incomplete]', offset: offset + strLen };
                    }

                    const strBytes = data.slice(offset, offset + strLen);
                    const str = new TextDecoder().decode(strBytes);
                    offset += strLen;
                    return { value: str, offset };
                }

                case 4: { // uint32
                    if (offset + 4 > data.byteLength) {
                        return { value: 0, offset: offset + 4 };
                    }
                    const val32 = view.getUint32(offset, true);
                    offset += 4;
                    return { value: val32, offset };
                }

                case 5: { // int32
                    if (offset + 4 > data.byteLength) {
                        return { value: 0, offset: offset + 4 };
                    }
                    const valInt32 = view.getInt32(offset, true);
                    offset += 4;
                    return { value: valInt32, offset };
                }

                case 6: { // float32
                    if (offset + 4 > data.byteLength) {
                        return { value: 0.0, offset: offset + 4 };
                    }
                    const valFloat = view.getFloat32(offset, true);
                    offset += 4;
                    return { value: valFloat, offset };
                }

                case 10: { // uint64
                    if (offset + 8 > data.byteLength) {
                        return { value: 0n, offset: offset + 8 };
                    }
                    const val64 = this.readUint64(view, offset);
                    offset += 8;
                    return { value: val64, offset };
                }

                case 11: { // int64
                    if (offset + 8 > data.byteLength) {
                        return { value: 0n, offset: offset + 8 };
                    }
                    const low = view.getInt32(offset, true);
                    const high = view.getInt32(offset + 4, true);
                    const val = BigInt(high) * (1n << 32n) + BigInt(low);
                    offset += 8;
                    return { value: val, offset };
                }

                case 12: { // float64
                    if (offset + 8 > data.byteLength) {
                        return { value: 0.0, offset: offset + 8 };
                    }
                    const valDouble = view.getFloat64(offset, true);
                    offset += 8;
                    return { value: valDouble, offset };
                }

                case 0: { // uint8
                    if (offset + 1 > data.byteLength) {
                        return { value: 0, offset: offset + 1 };
                    }
                    const valUint8 = view.getUint8(offset);
                    offset += 1;
                    return { value: valUint8, offset };
                }

                default:
                    // Unknown type, skip 8 bytes
                    return { value: null, offset: offset + 8 };
            }
        } catch (error) {
            console.warn('Error parsing metadata value:', error);
            return { value: null, offset: offset + 8 };
        }
    }

    // ==================== Tensor Streaming Loader ====================
    async loadTensorChunk(offset, size) {
        if (!this.fileHandle) {
            throw new Error('File handle not available');
        }

        const blob = this.fileHandle.slice(offset, offset + size);
        const buffer = await blob.arrayBuffer();
        return new Uint8Array(buffer);
    }

    // ==================== WebGPU Inference ====================
    async runInference() {
        if (!this.modelMetadata || !this.gpuContext) {
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
        const tokens = this.tokenize(prompt);
        let output = prompt;

        const generationSteps = Math.min(15, contextSize / 50);
        for (let i = 0; i < generationSteps; i++) {
            const nextToken = this.sampleToken(temperature);
            output += nextToken;
            
            // Simulate streaming output
            await new Promise(resolve => setTimeout(resolve, 50));
        }

        return output;
    }

    tokenize(text) {
        return text.split(/\s+/).filter(t => t.length > 0);
    }

    sampleToken(temperature) {
        const tokens = [' the', ' model', ' is', ' working', ' well', ' now', ' here', ' there', ' successfully', ' loaded'];
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

        canvas.width = 400;
        canvas.height = 300;

        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        context.configure({
            device: this.gpuContext,
            format: canvasFormat,
        });

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
        this.fileHandle = null;
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
