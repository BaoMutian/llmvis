/**
 * LLM Inference Visualizer - API 调用模块
 */

const API_BASE = '';

/**
 * 基础请求函数
 */
async function request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        }
    };
    
    try {
        const response = await fetch(url, config);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || `HTTP Error: ${response.status}`);
        }
        
        return data;
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        throw error;
    }
}

/**
 * API 接口
 */
const API = {
    /**
     * 加载模型
     * @param {string} modelPath - 可选的模型路径
     */
    async loadModel(modelPath = null) {
        const body = modelPath ? { model_path: modelPath } : {};
        return await request('/api/model/load', {
            method: 'POST',
            body: JSON.stringify(body)
        });
    },
    
    /**
     * 获取模型信息
     */
    async getModelInfo() {
        return await request('/api/model/info');
    },
    
    /**
     * 执行推理
     * @param {Object} params - 推理参数
     * @param {string} params.prompt - 输入提示
     * @param {number} params.temperature - 温度
     * @param {number} params.top_k - Top-K
     * @param {number} params.top_p - Top-P
     * @param {number} params.max_tokens - 最大token数
     */
    async generate(params) {
        return await request('/api/generate', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    },
    
    /**
     * 获取注意力矩阵
     * @param {number} layer - 层索引
     * @param {number|null} head - 头索引（null表示聚合）
     * @param {string} aggregation - 聚合方式 ('mean' 或 'max')
     */
    async getAttention(layer, head = null, aggregation = 'mean') {
        let url = `/api/attention?layer=${layer}&aggregation=${aggregation}`;
        if (head !== null) {
            url += `&head=${head}`;
        }
        return await request(url);
    },
    
    /**
     * 获取指定位置的token概率分布
     * @param {number} position - token位置
     * @param {number} topN - 返回的top-N数量
     */
    async getTokenProbs(position, topN = 10) {
        return await request(`/api/token_probs?position=${position}&top_n=${topN}`);
    },
    
    /**
     * 获取层间相似度矩阵
     * @param {number} step - 步骤索引
     */
    async getLayerSimilarity(step) {
        return await request(`/api/layer_similarity?step=${step}`);
    },
    
    /**
     * 获取多头注意力对比数据
     * @param {number} layer - 层索引
     * @param {number} step - 生成步骤索引
     * @param {string} heads - 头索引，逗号分隔
     */
    async getMultiheadAttention(layer, step, heads = "0,1,2,3") {
        return await request(`/api/attention/multihead?layer=${layer}&step=${step}&heads=${heads}`);
    },
    
    /**
     * 获取注意力头熵分析
     * @param {number} layer - 层索引
     * @param {number|null} step - 步骤索引，null表示所有步骤平均
     */
    async getAttentionHeadEntropy(layer, step = null) {
        let url = `/api/attention/head_entropy?layer=${layer}`;
        if (step !== null) url += `&step=${step}`;
        return await request(url);
    },
    
    /**
     * 获取 Hidden States 层间相似度
     * @param {number} step - 步骤索引
     */
    async getHiddenStatesSimilarity(step) {
        return await request(`/api/analyze/hidden_states?step=${step}`);
    },
    
    /**
     * 获取残差流分析
     * @param {number} step - 步骤索引
     */
    async getResidualAnalysis(step) {
        return await request(`/api/analyze/residual?step=${step}`);
    },
    
    /**
     * 获取 Logits Lens 分析
     * @param {number} step - 步骤索引
     */
    async getLogitsLens(step) {
        return await request(`/api/analyze/logits_lens?step=${step}`);
    },
    
    /**
     * 获取 Token Embedding 投影
     */
    async getEmbeddingProjection() {
        return await request(`/api/analyze/embeddings`);
    },
    
    /**
     * 获取激活值分布
     * @param {number} layer - 层索引
     */
    async getActivationStats(layer) {
        return await request(`/api/analyze/activations?layer=${layer}`);
    },
    
    /**
     * 获取输入归因
     * @param {number} outputIdx - 输出token索引
     */
    async getInputAttribution(outputIdx) {
        return await request(`/api/analyze/attribution?output_idx=${outputIdx}`);
    }
};

// 导出到全局
window.API = API;

