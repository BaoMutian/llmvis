/**
 * LLM Inference Visualizer - ECharts 图表模块
 * 封装各类可视化图表
 */

// 颜色主题
const CHART_COLORS = {
    primary: '#00d4aa',
    primaryDim: '#00a080',
    accent: '#a855f7',
    warning: '#f59e0b',
    danger: '#ef4444',
    bgBase: '#0a0e14',
    bgSurface: '#111820',
    bgElevated: '#1a2230',
    textPrimary: '#e4e8f0',
    textSecondary: '#8892a0',
    textDim: '#5a6470',
    border: '#2a3545'
};

// 图表实例管理
const chartInstances = {
    // 注意力分析组
    attention: null,
    multihead: null,
    headEntropy: null,
    // 隐藏状态分析组
    hiddenSimilarity: null,
    residual: null,
    logitsLens: null,
    // Token分析组
    probs: null,
    embedding: null,
    activation: null,
    // 归因分析组
    entropy: null,
    confidence: null,
    attribution: null
};

/**
 * 初始化所有图表
 */
function initCharts() {
    // 注意力分析组
    const attentionContainer = document.getElementById('attentionChart');
    const multiheadContainer = document.getElementById('multiheadChart');
    const headEntropyContainer = document.getElementById('headEntropyChart');
    
    // 隐藏状态分析组
    const hiddenSimilarityContainer = document.getElementById('hiddenSimilarityChart');
    const residualContainer = document.getElementById('residualChart');
    const logitsLensContainer = document.getElementById('logitsLensChart');
    
    // Token分析组
    const probsContainer = document.getElementById('probsChart');
    const embeddingContainer = document.getElementById('embeddingChart');
    const activationContainer = document.getElementById('activationChart');
    
    // 归因分析组
    const entropyContainer = document.getElementById('entropyChart');
    const confidenceContainer = document.getElementById('confidenceChart');
    const attributionContainer = document.getElementById('attributionChart');
    
    // 初始化图表实例
    if (attentionContainer) chartInstances.attention = echarts.init(attentionContainer, null, { renderer: 'canvas' });
    if (multiheadContainer) chartInstances.multihead = echarts.init(multiheadContainer, null, { renderer: 'canvas' });
    if (headEntropyContainer) chartInstances.headEntropy = echarts.init(headEntropyContainer, null, { renderer: 'canvas' });
    
    if (hiddenSimilarityContainer) chartInstances.hiddenSimilarity = echarts.init(hiddenSimilarityContainer, null, { renderer: 'canvas' });
    if (residualContainer) chartInstances.residual = echarts.init(residualContainer, null, { renderer: 'canvas' });
    if (logitsLensContainer) chartInstances.logitsLens = echarts.init(logitsLensContainer, null, { renderer: 'canvas' });
    
    if (probsContainer) chartInstances.probs = echarts.init(probsContainer, null, { renderer: 'canvas' });
    if (embeddingContainer) chartInstances.embedding = echarts.init(embeddingContainer, null, { renderer: 'canvas' });
    if (activationContainer) chartInstances.activation = echarts.init(activationContainer, null, { renderer: 'canvas' });
    
    if (entropyContainer) chartInstances.entropy = echarts.init(entropyContainer, null, { renderer: 'canvas' });
    if (confidenceContainer) chartInstances.confidence = echarts.init(confidenceContainer, null, { renderer: 'canvas' });
    if (attributionContainer) chartInstances.attribution = echarts.init(attributionContainer, null, { renderer: 'canvas' });
    
    // 窗口大小变化时自动调整
    window.addEventListener('resize', () => {
        Object.values(chartInstances).forEach(chart => {
            if (chart) chart.resize();
        });
    });
}

/**
 * 渲染注意力热力图
 * @param {Object} data - 注意力数据
 */
function renderAttentionHeatmap(data) {
    const chart = chartInstances.attention;
    if (!chart) return;
    
    const { attention_matrix, x_labels, y_labels, layer, head, statistics } = data;
    
    if (!attention_matrix || attention_matrix.length === 0) {
        chart.clear();
        return;
    }
    
    // 转换数据格式为 ECharts 需要的 [x, y, value]
    const heatmapData = [];
    for (let y = 0; y < attention_matrix.length; y++) {
        for (let x = 0; x < attention_matrix[y].length; x++) {
            heatmapData.push([x, y, attention_matrix[y][x]]);
        }
    }
    
    // 清理token标签（移除特殊字符）
    const cleanLabels = (labels) => labels.map(label => {
        if (!label) return '';
        return label.replace(/[<>]/g, '').substring(0, 10);
    });
    
    const xLabels = cleanLabels(x_labels || []);
    const yLabels = cleanLabels(y_labels || []);
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: `注意力热力图 - Layer ${layer}`,
            subtext: head !== null ? `Head ${head}` : '平均值',
            left: 'center',
            textStyle: {
                color: CHART_COLORS.textPrimary,
                fontSize: 14,
                fontWeight: 'normal'
            },
            subtextStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            }
        },
        tooltip: {
            position: 'top',
            formatter: function(params) {
                const xLabel = xLabels[params.data[0]] || params.data[0];
                const yLabel = yLabels[params.data[1]] || params.data[1];
                return `
                    <div style="font-family: monospace;">
                        <div>生成: <b style="color:${CHART_COLORS.primary}">${yLabel}</b></div>
                        <div>关注: <b>${xLabel}</b></div>
                        <div>权重: <b>${params.data[2].toFixed(4)}</b></div>
                    </div>
                `;
            },
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: {
                color: CHART_COLORS.textPrimary
            }
        },
        grid: {
            left: 80,
            right: 80,
            top: 60,
            bottom: 60
        },
        xAxis: {
            type: 'category',
            data: xLabels,
            splitArea: { show: false },
            axisLabel: {
                color: CHART_COLORS.textDim,
                fontSize: 9,
                rotate: 45,
                interval: Math.max(0, Math.floor(xLabels.length / 20) - 1)
            },
            axisLine: {
                lineStyle: { color: CHART_COLORS.border }
            }
        },
        yAxis: {
            type: 'category',
            data: yLabels,
            splitArea: { show: false },
            axisLabel: {
                color: CHART_COLORS.textDim,
                fontSize: 9
            },
            axisLine: {
                lineStyle: { color: CHART_COLORS.border }
            }
        },
        visualMap: {
            min: 0,
            max: statistics ? statistics.max : 1,
            calculable: true,
            orient: 'vertical',
            right: 10,
            top: 'center',
            inRange: {
                color: [CHART_COLORS.bgSurface, CHART_COLORS.primaryDim, CHART_COLORS.primary, CHART_COLORS.warning]
            },
            textStyle: {
                color: CHART_COLORS.textSecondary
            }
        },
        series: [{
            name: 'Attention',
            type: 'heatmap',
            data: heatmapData,
            emphasis: {
                itemStyle: {
                    shadowBlur: 10,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
            }
        }]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染概率分布柱状图
 * @param {Object} data - 概率分布数据
 */
function renderProbsChart(data) {
    const chart = chartInstances.probs;
    if (!chart) return;
    
    const { position, actual_token, top_k_tokens, distribution_analysis } = data;
    
    if (!top_k_tokens || top_k_tokens.length === 0) {
        chart.clear();
        return;
    }
    
    const tokens = top_k_tokens.map(t => t.token.replace(/\n/g, '\\n').substring(0, 12));
    const probs = top_k_tokens.map(t => t.probability);
    
    // 标记实际选中的token
    const colors = top_k_tokens.map(t => {
        if (t.token === actual_token) {
            return CHART_COLORS.primary;
        }
        return CHART_COLORS.accent;
    });
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: `Token 概率分布 - 位置 ${position}`,
            subtext: `实际生成: "${actual_token}" | 熵: ${distribution_analysis?.entropy_bits?.toFixed(2) || '-'} bits`,
            left: 'center',
            textStyle: {
                color: CHART_COLORS.textPrimary,
                fontSize: 14,
                fontWeight: 'normal'
            },
            subtextStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: function(params) {
                const item = params[0];
                const tokenData = top_k_tokens[item.dataIndex];
                const isActual = tokenData.token === actual_token;
                return `
                    <div style="font-family: monospace;">
                        <div>${isActual ? '✓ ' : ''}Token: <b style="color:${isActual ? CHART_COLORS.primary : CHART_COLORS.accent}">${tokenData.token}</b></div>
                        <div>概率: <b>${(tokenData.probability * 100).toFixed(2)}%</b></div>
                        <div>Logit: <b>${tokenData.logit.toFixed(3)}</b></div>
                    </div>
                `;
            },
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        grid: {
            left: 60,
            right: 20,
            top: 80,
            bottom: 80
        },
        xAxis: {
            type: 'category',
            data: tokens,
            axisLabel: {
                color: CHART_COLORS.textSecondary,
                fontSize: 10,
                rotate: 30,
                interval: 0
            },
            axisLine: {
                lineStyle: { color: CHART_COLORS.border }
            }
        },
        yAxis: {
            type: 'value',
            name: '概率',
            nameTextStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            },
            axisLabel: {
                color: CHART_COLORS.textSecondary,
                formatter: (val) => (val * 100).toFixed(0) + '%'
            },
            axisLine: {
                lineStyle: { color: CHART_COLORS.border }
            },
            splitLine: {
                lineStyle: { color: CHART_COLORS.border, opacity: 0.3 }
            }
        },
        series: [{
            name: '概率',
            type: 'bar',
            data: probs.map((p, i) => ({
                value: p,
                itemStyle: { color: colors[i] }
            })),
            barWidth: '60%',
            label: {
                show: true,
                position: 'top',
                color: CHART_COLORS.textSecondary,
                fontSize: 9,
                formatter: (params) => (params.value * 100).toFixed(1) + '%'
            }
        }]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染熵分布折线图
 * @param {Object} data - 熵数据
 */
function renderEntropyChart(data) {
    const chart = chartInstances.entropy;
    if (!chart) return;
    
    const { entropies_bits, generated_tokens, mean_entropy, std_entropy } = data;
    
    if (!entropies_bits || entropies_bits.length === 0) {
        chart.clear();
        return;
    }
    
    const xData = generated_tokens || entropies_bits.map((_, i) => i);
    const meanValue = mean_entropy || entropies_bits.reduce((a, b) => a + b, 0) / entropies_bits.length;
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: '熵分布曲线',
            subtext: `平均熵: ${meanValue.toFixed(3)} nats`,
            left: 'center',
            textStyle: {
                color: CHART_COLORS.textPrimary,
                fontSize: 14,
                fontWeight: 'normal'
            },
            subtextStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            }
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const item = params[0];
                const token = generated_tokens ? generated_tokens[item.dataIndex] : item.dataIndex;
                return `
                    <div style="font-family: monospace;">
                        <div>位置: ${item.dataIndex}</div>
                        <div>Token: <b style="color:${CHART_COLORS.primary}">${token}</b></div>
                        <div>熵: <b>${item.value.toFixed(3)} bits</b></div>
                    </div>
                `;
            },
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        grid: {
            left: 60,
            right: 20,
            top: 70,
            bottom: 50
        },
        xAxis: {
            type: 'category',
            data: xData.map((_, i) => i),
            name: 'Token 位置',
            nameLocation: 'middle',
            nameGap: 30,
            nameTextStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            },
            axisLabel: {
                color: CHART_COLORS.textSecondary,
                fontSize: 10
            },
            axisLine: {
                lineStyle: { color: CHART_COLORS.border }
            }
        },
        yAxis: {
            type: 'value',
            name: '熵 (bits)',
            nameTextStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            },
            axisLabel: {
                color: CHART_COLORS.textSecondary,
                formatter: (val) => val.toFixed(1)
            },
            axisLine: {
                lineStyle: { color: CHART_COLORS.border }
            },
            splitLine: {
                lineStyle: { color: CHART_COLORS.border, opacity: 0.3 }
            }
        },
        series: [
            {
                name: '熵',
                type: 'line',
                data: entropies_bits,
                smooth: true,
                symbol: 'circle',
                symbolSize: 6,
                lineStyle: {
                    color: CHART_COLORS.primary,
                    width: 2
                },
                itemStyle: {
                    color: CHART_COLORS.primary
                },
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [
                            { offset: 0, color: 'rgba(0, 212, 170, 0.3)' },
                            { offset: 1, color: 'rgba(0, 212, 170, 0.02)' }
                        ]
                    }
                }
            },
            {
                name: '平均熵',
                type: 'line',
                data: entropies_bits.map(() => meanValue),
                lineStyle: {
                    color: CHART_COLORS.warning,
                    type: 'dashed',
                    width: 1
                },
                symbol: 'none'
            }
        ]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染置信度曲线
 * @param {Object} data - 置信度数据
 */
function renderConfidenceChart(data) {
    const chart = chartInstances.confidence;
    if (!chart) return;
    
    const { confidence_curve, generated_tokens } = data;
    
    if (!confidence_curve || confidence_curve.length === 0) {
        chart.clear();
        return;
    }
    
    const positions = confidence_curve.map(c => c.position);
    const maxProbs = confidence_curve.map(c => c.max_prob);
    const probGaps = confidence_curve.map(c => c.prob_gap);
    const top5Sums = confidence_curve.map(c => c.top_5_sum);
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: '生成置信度曲线',
            left: 'center',
            textStyle: {
                color: CHART_COLORS.textPrimary,
                fontSize: 14,
                fontWeight: 'normal'
            }
        },
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const idx = params[0].dataIndex;
                const token = generated_tokens ? generated_tokens[idx] : idx;
                return `
                    <div style="font-family: monospace;">
                        <div>Token: <b style="color:${CHART_COLORS.primary}">${token}</b></div>
                        <div>最高概率: <b>${(maxProbs[idx] * 100).toFixed(2)}%</b></div>
                        <div>概率差距: <b>${(probGaps[idx] * 100).toFixed(2)}%</b></div>
                        <div>Top-5 累计: <b>${(top5Sums[idx] * 100).toFixed(2)}%</b></div>
                    </div>
                `;
            },
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        legend: {
            data: ['最高概率', '概率差距', 'Top-5 累计'],
            top: 30,
            textStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            }
        },
        grid: {
            left: 60,
            right: 20,
            top: 80,
            bottom: 50
        },
        xAxis: {
            type: 'category',
            data: positions,
            name: 'Token 位置',
            nameLocation: 'middle',
            nameGap: 30,
            nameTextStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            },
            axisLabel: {
                color: CHART_COLORS.textSecondary,
                fontSize: 10
            },
            axisLine: {
                lineStyle: { color: CHART_COLORS.border }
            }
        },
        yAxis: {
            type: 'value',
            name: '概率',
            max: 1,
            nameTextStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            },
            axisLabel: {
                color: CHART_COLORS.textSecondary,
                formatter: (val) => (val * 100).toFixed(0) + '%'
            },
            axisLine: {
                lineStyle: { color: CHART_COLORS.border }
            },
            splitLine: {
                lineStyle: { color: CHART_COLORS.border, opacity: 0.3 }
            }
        },
        series: [
            {
                name: '最高概率',
                type: 'line',
                data: maxProbs,
                smooth: true,
                symbol: 'circle',
                symbolSize: 4,
                lineStyle: { color: CHART_COLORS.primary, width: 2 },
                itemStyle: { color: CHART_COLORS.primary }
            },
            {
                name: '概率差距',
                type: 'line',
                data: probGaps,
                smooth: true,
                symbol: 'circle',
                symbolSize: 4,
                lineStyle: { color: CHART_COLORS.accent, width: 2 },
                itemStyle: { color: CHART_COLORS.accent }
            },
            {
                name: 'Top-5 累计',
                type: 'line',
                data: top5Sums,
                smooth: true,
                symbol: 'circle',
                symbolSize: 4,
                lineStyle: { color: CHART_COLORS.warning, width: 2, type: 'dashed' },
                itemStyle: { color: CHART_COLORS.warning }
            }
        ]
    };
    
    chart.setOption(option, true);
}

/**
 * 清空所有图表
 */
function clearAllCharts() {
    Object.values(chartInstances).forEach(chart => {
        if (chart) chart.clear();
    });
}

/**
 * 调整图表大小
 */
function resizeCharts() {
    Object.values(chartInstances).forEach(chart => {
        if (chart) chart.resize();
    });
}

// 导出到全局
window.Charts = {
    init: initCharts,
    renderAttentionHeatmap,
    renderProbsChart,
    renderEntropyChart,
    renderConfidenceChart,
    clearAllCharts,
    resizeCharts
};

