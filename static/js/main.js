/**
 * LLM Inference Visualizer - 主逻辑模块
 */

// 应用状态
const AppState = {
    modelLoaded: false,
    generationResult: null,
    selectedTokenIndex: null,
    currentTab: 'attention',
    numLayers: 36,
    numHeads: 32
};

// DOM 元素引用
const Elements = {
    // 状态
    modelStatus: null,
    loadingOverlay: null,
    
    // 输入
    promptInput: null,
    generateBtn: null,
    
    // 参数
    temperature: null,
    temperatureValue: null,
    topK: null,
    topKValue: null,
    topP: null,
    topPValue: null,
    maxTokens: null,
    maxTokensValue: null,
    
    // 可视化参数
    topNProbs: null,
    topNProbsValue: null,
    attentionLayer: null,
    attentionLayerValue: null,
    attentionHead: null,
    
    // 统计
    statTokens: null,
    statEntropy: null,
    statPerplexity: null,
    statConfidence: null,
    
    // Token 展示
    tokenDisplay: null,
    selectedTokenInfo: null,
    selectedToken: null,
    selectedPosition: null,
    selectedEntropy: null,
    
    // 标签页
    tabBtns: null,
    tabContents: null,
    
    // 布局
    mainContent: null,
    resizeHandle: null,
    rightPanel: null
};

/**
 * 初始化 DOM 引用
 */
function initElements() {
    Elements.modelStatus = document.getElementById('modelStatus');
    Elements.loadingOverlay = document.getElementById('loadingOverlay');
    
    Elements.promptInput = document.getElementById('promptInput');
    Elements.generateBtn = document.getElementById('generateBtn');
    
    Elements.temperature = document.getElementById('temperature');
    Elements.temperatureValue = document.getElementById('temperatureValue');
    Elements.topK = document.getElementById('topK');
    Elements.topKValue = document.getElementById('topKValue');
    Elements.topP = document.getElementById('topP');
    Elements.topPValue = document.getElementById('topPValue');
    Elements.maxTokens = document.getElementById('maxTokens');
    Elements.maxTokensValue = document.getElementById('maxTokensValue');
    
    Elements.topNProbs = document.getElementById('topNProbs');
    Elements.topNProbsValue = document.getElementById('topNProbsValue');
    Elements.attentionLayer = document.getElementById('attentionLayer');
    Elements.attentionLayerValue = document.getElementById('attentionLayerValue');
    Elements.attentionHead = document.getElementById('attentionHead');
    
    Elements.statTokens = document.getElementById('statTokens');
    Elements.statEntropy = document.getElementById('statEntropy');
    Elements.statPerplexity = document.getElementById('statPerplexity');
    Elements.statConfidence = document.getElementById('statConfidence');
    
    Elements.tokenDisplay = document.getElementById('tokenDisplay');
    Elements.selectedTokenInfo = document.getElementById('selectedTokenInfo');
    Elements.selectedToken = document.getElementById('selectedToken');
    Elements.selectedPosition = document.getElementById('selectedPosition');
    Elements.selectedEntropy = document.getElementById('selectedEntropy');
    
    Elements.tabBtns = document.querySelectorAll('.tab-btn');
    Elements.tabContents = document.querySelectorAll('.tab-content');
    
    Elements.mainContent = document.querySelector('.main-content');
    Elements.resizeHandle = document.getElementById('resizeHandle');
    Elements.rightPanel = document.getElementById('rightPanel');
}

/**
 * 设置事件监听
 */
function setupEventListeners() {
    // 生成按钮
    Elements.generateBtn.addEventListener('click', handleGenerate);
    
    // 参数滑块
    setupSlider(Elements.temperature, Elements.temperatureValue);
    setupSlider(Elements.topK, Elements.topKValue);
    setupSlider(Elements.topP, Elements.topPValue);
    setupSlider(Elements.maxTokens, Elements.maxTokensValue);
    setupSlider(Elements.topNProbs, Elements.topNProbsValue);
    setupSlider(Elements.attentionLayer, Elements.attentionLayerValue, handleAttentionLayerChange);
    
    // 注意力头选择
    Elements.attentionHead.addEventListener('change', handleAttentionHeadChange);
    
    // 标签页切换
    Elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
    
    // Enter 键发送
    Elements.promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleGenerate();
        }
    });
    
    // 右侧面板拖动调整宽度
    setupResizeHandle();
}

/**
 * 设置拖动调整宽度
 */
function setupResizeHandle() {
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;
    
    const handle = Elements.resizeHandle;
    const panel = Elements.rightPanel;
    const mainContent = Elements.mainContent;
    
    if (!handle || !panel) return;
    
    handle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = panel.offsetWidth;
        handle.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        const deltaX = startX - e.clientX;
        // 最大宽度为窗口宽度减去左侧面板和最小中间区域
        const maxWidth = window.innerWidth - 280 - 100 - 16; // 左侧280 + 最小中间区100 + 间隙
        const newWidth = Math.max(300, Math.min(maxWidth, startWidth + deltaX));
        
        // 更新 grid 布局
        mainContent.style.gridTemplateColumns = `280px 1fr 8px ${newWidth}px`;
        
        // 调整图表大小
        Charts.resizeCharts();
    });
    
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            handle.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            
            // 最终调整图表大小
            setTimeout(() => Charts.resizeCharts(), 100);
        }
    });
}

/**
 * 设置滑块控件
 */
function setupSlider(slider, valueDisplay, onChange = null) {
    const updateValue = () => {
        valueDisplay.textContent = slider.value;
        if (onChange) onChange(slider.value);
    };
    
    slider.addEventListener('input', updateValue);
    updateValue();
}

/**
 * 切换标签页
 */
function switchTab(tabName) {
    AppState.currentTab = tabName;
    
    Elements.tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    Elements.tabContents.forEach(content => {
        const isActive = content.id === `tab${tabName.charAt(0).toUpperCase() + tabName.slice(1)}`;
        content.classList.toggle('active', isActive);
    });
    
    // 切换后调整图表大小（需要多次延迟确保渲染完成）
    setTimeout(() => {
        Charts.resizeCharts();
        // 再次延迟确保图表完全渲染
        setTimeout(() => Charts.resizeCharts(), 200);
    }, 50);
}

/**
 * 显示/隐藏加载遮罩
 */
function showLoading(show, text = '正在推理中...') {
    Elements.loadingOverlay.style.display = show ? 'flex' : 'none';
    Elements.loadingOverlay.querySelector('.loading-text').textContent = text;
}

/**
 * 更新模型状态显示
 */
function updateModelStatus(status, text) {
    Elements.modelStatus.className = `status-badge status-${status}`;
    Elements.modelStatus.textContent = text;
}

/**
 * 检查并加载模型
 */
async function checkAndLoadModel() {
    try {
        updateModelStatus('loading', '检查模型...');
        
        // 先检查模型是否已加载
        const infoResponse = await API.getModelInfo();
        
        if (infoResponse.loaded) {
            AppState.modelLoaded = true;
            AppState.numLayers = infoResponse.info.num_layers;
            AppState.numHeads = infoResponse.info.num_attention_heads;
            updateModelStatus('ready', '模型就绪');
            updateLayerSliderMax();
            return;
        }
        
        // 未加载，尝试加载
        updateModelStatus('loading', '加载模型中...');
        showLoading(true, '正在加载模型，这可能需要几分钟...');
        
        const loadResponse = await API.loadModel();
        
        if (loadResponse.success) {
            AppState.modelLoaded = true;
            AppState.numLayers = loadResponse.model_info.num_layers;
            AppState.numHeads = loadResponse.model_info.num_attention_heads;
            updateModelStatus('ready', '模型就绪');
            updateLayerSliderMax();
        } else {
            throw new Error(loadResponse.message);
        }
        
    } catch (error) {
        console.error('Model loading error:', error);
        updateModelStatus('error', '模型加载失败');
    } finally {
        showLoading(false);
    }
}

/**
 * 更新层数滑块最大值
 */
function updateLayerSliderMax() {
    Elements.attentionLayer.max = AppState.numLayers - 1;
}

/**
 * 处理生成请求
 */
async function handleGenerate() {
    if (!AppState.modelLoaded) {
        alert('请等待模型加载完成');
        return;
    }
    
    const prompt = Elements.promptInput.value.trim();
    if (!prompt) {
        alert('请输入内容');
        return;
    }
    
    try {
        showLoading(true, '正在推理中...');
        Elements.generateBtn.disabled = true;
        
        const params = {
            prompt: prompt,
            temperature: parseFloat(Elements.temperature.value),
            top_k: parseInt(Elements.topK.value),
            top_p: parseFloat(Elements.topP.value),
            max_tokens: parseInt(Elements.maxTokens.value)
        };
        
        const result = await API.generate(params);
        
        if (result.success) {
            AppState.generationResult = result;
            AppState.selectedTokenIndex = null;
            
            // 更新显示
            renderTokens(result);
            updateStats(result);
            
            // 更新图表
            await updateAllCharts();
        } else {
            throw new Error(result.message);
        }
        
    } catch (error) {
        console.error('Generation error:', error);
        alert('生成失败: ' + error.message);
    } finally {
        showLoading(false);
        Elements.generateBtn.disabled = false;
    }
}

/**
 * 渲染 Token 展示
 */
function renderTokens(result) {
    const { input_tokens, generated_tokens, analysis } = result;
    const { entropies, max_entropy, min_entropy } = analysis;
    
    let html = '';
    
    // 输入 tokens（折叠显示）
    html += '<div class="input-tokens-section">';
    html += '<span class="section-label">输入:</span> ';
    const displayInputTokens = input_tokens.slice(-20); // 只显示最后20个
    if (input_tokens.length > 20) {
        html += `<span class="token input-token">... (${input_tokens.length - 20} more)</span>`;
    }
    displayInputTokens.forEach(token => {
        const displayToken = formatToken(token);
        html += `<span class="token input-token">${displayToken}</span>`;
    });
    html += '</div>';
    
    // 分隔线
    html += '<div class="token-separator">▼ 生成输出 ▼</div>';
    
    // 生成的 tokens
    html += '<div class="generated-tokens-section">';
    generated_tokens.forEach((token, idx) => {
        const entropy = entropies[idx];
        const normalizedEntropy = (entropy - min_entropy) / (max_entropy - min_entropy + 0.001);
        
        // 根据熵值设置颜色类
        let entropyClass = '';
        if (normalizedEntropy > 0.7) {
            entropyClass = 'high-entropy';
        } else if (normalizedEntropy < 0.3) {
            entropyClass = 'low-entropy';
        }
        
        const displayToken = formatToken(token);
        
        html += `<span class="token generated-token ${entropyClass}" 
                       data-index="${idx}" 
                       data-entropy="${entropy.toFixed(3)}"
                       title="熵: ${entropy.toFixed(3)}">${displayToken}</span>`;
    });
    html += '</div>';
    
    Elements.tokenDisplay.innerHTML = html;
    
    // 添加样式
    addTokenStyles();
    
    // 绑定点击事件
    const generatedTokens = Elements.tokenDisplay.querySelectorAll('.generated-token');
    generatedTokens.forEach(tokenEl => {
        tokenEl.addEventListener('click', () => {
            const index = parseInt(tokenEl.dataset.index);
            selectToken(index);
        });
    });
}

/**
 * 添加动态样式
 */
function addTokenStyles() {
    if (!document.getElementById('token-dynamic-styles')) {
        const style = document.createElement('style');
        style.id = 'token-dynamic-styles';
        style.textContent = `
            .input-tokens-section {
                margin-bottom: 16px;
                opacity: 0.7;
            }
            .section-label {
                font-size: 11px;
                color: var(--text-dim);
                text-transform: uppercase;
                margin-right: 8px;
            }
            .token-separator {
                text-align: center;
                color: var(--color-primary);
                font-size: 11px;
                margin: 16px 0;
                opacity: 0.6;
            }
            .generated-tokens-section {
                line-height: 2.4;
            }
        `;
        document.head.appendChild(style);
    }
}

/**
 * 格式化 token 显示
 */
function formatToken(token) {
    if (!token) return '';
    return token
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '↵')
        .replace(/\t/g, '→')
        .replace(/ /g, '·');
}

/**
 * 选择 token
 */
async function selectToken(index) {
    AppState.selectedTokenIndex = index;
    
    // 更新选中状态
    const tokens = Elements.tokenDisplay.querySelectorAll('.generated-token');
    tokens.forEach((t, i) => {
        t.classList.toggle('selected', i === index);
    });
    
    // 显示详情
    const result = AppState.generationResult;
    const token = result.generated_tokens[index];
    const entropy = result.analysis.entropies[index];
    
    Elements.selectedTokenInfo.style.display = 'block';
    Elements.selectedToken.textContent = `"${token}"`;
    Elements.selectedPosition.textContent = index;
    Elements.selectedEntropy.textContent = entropy.toFixed(4);
    
    // 获取并显示概率分布
    await updateProbsChart(index);
    
    // 切换到概率分布标签页
    switchTab('probs');
}

/**
 * 更新统计信息
 */
function updateStats(result) {
    const { analysis, num_generated } = result;
    
    Elements.statTokens.textContent = num_generated;
    Elements.statEntropy.textContent = analysis.mean_entropy.toFixed(3);
    Elements.statPerplexity.textContent = analysis.perplexity.toFixed(2);
    Elements.statConfidence.textContent = (analysis.mean_max_prob * 100).toFixed(1) + '%';
}

/**
 * 更新所有图表
 */
async function updateAllCharts() {
    await updateAttentionChart();
    updateEntropyChart();
    updateConfidenceChart();
    
    // 如果有选中的token，更新概率分布
    if (AppState.selectedTokenIndex !== null) {
        await updateProbsChart(AppState.selectedTokenIndex);
    }
}

/**
 * 更新注意力热力图
 */
async function handleAttentionLayerChange(value) {
    if (AppState.generationResult) {
        await updateAttentionChart();
    }
}

/**
 * 处理注意力头变化
 */
async function handleAttentionHeadChange() {
    if (AppState.generationResult) {
        await updateAttentionChart();
    }
}

/**
 * 更新注意力热力图
 */
async function updateAttentionChart() {
    if (!AppState.generationResult) return;
    
    try {
        const layer = parseInt(Elements.attentionLayer.value);
        const aggregation = Elements.attentionHead.value;
        
        const data = await API.getAttention(layer, null, aggregation);
        
        if (data.success) {
            Charts.renderAttentionHeatmap(data);
        }
    } catch (error) {
        console.error('Attention chart error:', error);
    }
}

/**
 * 更新概率分布图
 */
async function updateProbsChart(position) {
    if (!AppState.generationResult) return;
    
    try {
        const topN = parseInt(Elements.topNProbs.value);
        const data = await API.getTokenProbs(position, topN);
        
        if (data.success) {
            Charts.renderProbsChart(data);
        }
    } catch (error) {
        console.error('Probs chart error:', error);
    }
}

/**
 * 更新熵分布图
 */
function updateEntropyChart() {
    if (!AppState.generationResult) return;
    
    const { analysis, generated_tokens } = AppState.generationResult;
    
    Charts.renderEntropyChart({
        entropies_bits: analysis.entropies_bits,
        generated_tokens: generated_tokens,
        mean_entropy: analysis.mean_entropy
    });
}

/**
 * 更新置信度曲线
 */
function updateConfidenceChart() {
    if (!AppState.generationResult) return;
    
    const { analysis, generated_tokens } = AppState.generationResult;
    
    Charts.renderConfidenceChart({
        confidence_curve: analysis.confidence_curve,
        generated_tokens: generated_tokens
    });
}

/**
 * 应用初始化
 */
async function init() {
    console.log('Initializing LLM Inference Visualizer...');
    
    // 初始化 DOM 引用
    initElements();
    
    // 初始化图表
    Charts.init();
    
    // 设置事件监听
    setupEventListeners();
    
    // 检查并加载模型
    await checkAndLoadModel();
    
    console.log('Initialization complete.');
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', init);

