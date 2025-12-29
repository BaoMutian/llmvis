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
 * 渲染熵分布图（支持排序和百分位数高亮）
 * @param {Object} data - 熵数据
 */
function renderEntropyChart(data) {
    const chart = chartInstances.entropy;
    if (!chart) return;
    
    const { entropies_bits, generated_tokens, mean_entropy, sortMode = 'position', percentile = 75 } = data;
    
    if (!entropies_bits || entropies_bits.length === 0) {
        chart.clear();
        return;
    }
    
    const meanValue = mean_entropy || entropies_bits.reduce((a, b) => a + b, 0) / entropies_bits.length;
    
    // 创建索引数据对，包含原始位置
    let indexedData = entropies_bits.map((entropy, idx) => ({
        entropy,
        token: generated_tokens ? generated_tokens[idx] : idx,
        originalIndex: idx
    }));
    
    // 根据排序模式排序
    let xAxisName = 'Token 位置';
    let titleSuffix = '';
    if (sortMode === 'desc') {
        indexedData.sort((a, b) => b.entropy - a.entropy);
        xAxisName = 'Token (高熵→低熵)';
        titleSuffix = ' [高→低]';
    } else if (sortMode === 'asc') {
        indexedData.sort((a, b) => a.entropy - b.entropy);
        xAxisName = 'Token (低熵→高熵)';
        titleSuffix = ' [低→高]';
    }
    
    // 计算百分位数阈值
    const sortedEntropies = [...entropies_bits].sort((a, b) => a - b);
    const percentileIdx = Math.floor((percentile / 100) * sortedEntropies.length);
    const thresholdValue = sortedEntropies[Math.min(percentileIdx, sortedEntropies.length - 1)];
    
    // 根据阈值着色
    const chartData = indexedData.map(item => ({
        value: item.entropy,
        itemStyle: {
            color: item.entropy >= thresholdValue ? CHART_COLORS.danger : CHART_COLORS.primary
        },
        originalIndex: item.originalIndex,
        token: item.token
    }));
    
    const xLabels = indexedData.map(item => {
        const label = typeof item.token === 'string' ? item.token.substring(0, 4) : item.originalIndex;
        return sortMode === 'position' ? item.originalIndex : label;
    });
    
    const isLinear = sortMode === 'position';
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: '熵分布' + titleSuffix,
            subtext: `平均: ${meanValue.toFixed(3)} bits | 高熵阈值(${percentile}%): ${thresholdValue.toFixed(3)}`,
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
                if (!item || item.value === undefined) return '';
                const dataItem = chartData[item.dataIndex];
                const isHigh = item.value >= thresholdValue;
                return `
                    <div style="font-family: monospace;">
                        <div>位置: ${dataItem.originalIndex}</div>
                        <div>Token: <b style="color:${CHART_COLORS.primary}">${dataItem.token}</b></div>
                        <div>熵: <b>${item.value.toFixed(3)} bits</b></div>
                        <div>${isHigh ? '<span style="color:#ef4444">⚠ 高熵</span>' : '<span style="color:#00d4aa">✓ 正常</span>'}</div>
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
            bottom: isLinear ? 50 : 70
        },
        xAxis: {
            type: 'category',
            data: xLabels,
            name: xAxisName,
            nameLocation: 'middle',
            nameGap: isLinear ? 30 : 50,
            nameTextStyle: {
                color: CHART_COLORS.textSecondary,
                fontSize: 11
            },
            axisLabel: {
                color: CHART_COLORS.textSecondary,
                fontSize: isLinear ? 10 : 8,
                rotate: isLinear ? 0 : 45,
                interval: isLinear ? 'auto' : Math.max(0, Math.floor(xLabels.length / 15) - 1)
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
                type: isLinear ? 'line' : 'bar',
                data: chartData,
                smooth: isLinear,
                symbol: isLinear ? 'circle' : 'none',
                symbolSize: 6,
                lineStyle: isLinear ? {
                    color: CHART_COLORS.primary,
                    width: 2
                } : undefined,
                areaStyle: isLinear ? {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [
                            { offset: 0, color: 'rgba(0, 212, 170, 0.3)' },
                            { offset: 1, color: 'rgba(0, 212, 170, 0.02)' }
                        ]
                    }
                } : undefined,
                barWidth: '60%'
            },
            {
                name: '高熵阈值',
                type: 'line',
                data: chartData.map(() => thresholdValue),
                lineStyle: {
                    color: CHART_COLORS.danger,
                    type: 'dashed',
                    width: 1.5
                },
                symbol: 'none',
                z: 10
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
 * 渲染多头注意力对比图
 * @param {Object} data - 多头注意力数据
 */
function renderMultiheadChart(data) {
    const chart = chartInstances.multihead;
    if (!chart) return;
    
    const { layer, step, generated_token, heads, x_labels } = data;
    
    if (!heads || heads.length === 0) {
        chart.clear();
        return;
    }
    
    // 为每个头生成一个系列
    const series = heads.map((headData, idx) => ({
        name: `Head ${headData.head_idx}`,
        type: 'line',
        data: headData.attention,
        smooth: true,
        symbol: 'circle',
        symbolSize: 3,
        lineStyle: { width: 2 }
    }));
    
    const cleanLabels = x_labels.map(l => l ? l.substring(0, 8) : '');
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: `多头注意力对比 - Layer ${layer}`,
            subtext: `生成 Token: "${generated_token}"`,
            left: 'center',
            textStyle: { color: CHART_COLORS.textPrimary, fontSize: 14, fontWeight: 'normal' },
            subtextStyle: { color: CHART_COLORS.textSecondary, fontSize: 11 }
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        legend: {
            data: heads.map(h => `Head ${h.head_idx}`),
            top: 50,
            textStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 }
        },
        grid: { left: 50, right: 20, top: 90, bottom: 60 },
        xAxis: {
            type: 'category',
            data: cleanLabels,
            axisLabel: { color: CHART_COLORS.textDim, fontSize: 8, rotate: 45, interval: Math.floor(cleanLabels.length / 15) },
            axisLine: { lineStyle: { color: CHART_COLORS.border } }
        },
        yAxis: {
            type: 'value',
            name: '注意力权重',
            nameTextStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 },
            axisLabel: { color: CHART_COLORS.textSecondary, fontSize: 10 },
            axisLine: { lineStyle: { color: CHART_COLORS.border } },
            splitLine: { lineStyle: { color: CHART_COLORS.border, opacity: 0.3 } }
        },
        series: series
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染注意力头熵分析图
 * @param {Object} data - 注意力头熵数据
 */
function renderHeadEntropyChart(data) {
    const chart = chartInstances.headEntropy;
    if (!chart) return;
    
    const { layer, head_entropies, num_heads } = data;
    
    if (!head_entropies || head_entropies.length === 0) {
        chart.clear();
        return;
    }
    
    const headLabels = head_entropies.map(h => `H${h.head_idx}`);
    const entropies = head_entropies.map(h => h.mean_entropy);
    
    // 根据熵值计算颜色（低熵=绿色/聚焦，高熵=橙色/分散）
    const maxEntropy = Math.max(...entropies);
    const minEntropy = Math.min(...entropies);
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: `注意力头熵分析 - Layer ${layer}`,
            subtext: '低熵=聚焦型 | 高熵=分散型',
            left: 'center',
            textStyle: { color: CHART_COLORS.textPrimary, fontSize: 14, fontWeight: 'normal' },
            subtextStyle: { color: CHART_COLORS.textSecondary, fontSize: 11 }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: function(params) {
                const item = params[0];
                const headData = head_entropies[item.dataIndex];
                return `
                    <div style="font-family: monospace;">
                        <div>Head ${headData.head_idx}</div>
                        <div>平均熵: <b>${headData.mean_entropy.toFixed(4)}</b></div>
                    </div>
                `;
            },
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        grid: { left: 60, right: 20, top: 70, bottom: 50 },
        xAxis: {
            type: 'category',
            data: headLabels,
            axisLabel: { color: CHART_COLORS.textSecondary, fontSize: 9 },
            axisLine: { lineStyle: { color: CHART_COLORS.border } }
        },
        yAxis: {
            type: 'value',
            name: '熵',
            nameTextStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 },
            axisLabel: { color: CHART_COLORS.textSecondary },
            axisLine: { lineStyle: { color: CHART_COLORS.border } },
            splitLine: { lineStyle: { color: CHART_COLORS.border, opacity: 0.3 } }
        },
        series: [{
            type: 'bar',
            data: entropies.map((e, i) => {
                const ratio = (e - minEntropy) / (maxEntropy - minEntropy + 0.001);
                return {
                    value: e,
                    itemStyle: {
                        color: ratio < 0.3 ? CHART_COLORS.primary : 
                               ratio > 0.7 ? CHART_COLORS.warning : CHART_COLORS.accent
                    }
                };
            }),
            barWidth: '60%'
        }]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染 Hidden States 层间相似度热力图
 * @param {Object} data - 相似度数据
 */
function renderHiddenSimilarityChart(data) {
    const chart = chartInstances.hiddenSimilarity;
    if (!chart) return;
    
    const { similarity_matrix, num_layers, step } = data;
    
    if (!similarity_matrix || similarity_matrix.length === 0) {
        chart.clear();
        return;
    }
    
    // 转换为热力图数据
    const heatmapData = [];
    for (let y = 0; y < similarity_matrix.length; y++) {
        for (let x = 0; x < similarity_matrix[y].length; x++) {
            heatmapData.push([x, y, similarity_matrix[y][x]]);
        }
    }
    
    const layerLabels = Array.from({length: num_layers}, (_, i) => `L${i}`);
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: '层间隐藏状态相似度',
            left: 'center',
            textStyle: { color: CHART_COLORS.textPrimary, fontSize: 14, fontWeight: 'normal' }
        },
        tooltip: {
            formatter: (params) => `Layer ${params.data[0]} vs Layer ${params.data[1]}: ${params.data[2].toFixed(4)}`,
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        grid: { left: 50, right: 80, top: 50, bottom: 50 },
        xAxis: {
            type: 'category',
            data: layerLabels,
            axisLabel: { color: CHART_COLORS.textDim, fontSize: 8 },
            axisLine: { lineStyle: { color: CHART_COLORS.border } }
        },
        yAxis: {
            type: 'category',
            data: layerLabels,
            axisLabel: { color: CHART_COLORS.textDim, fontSize: 8 },
            axisLine: { lineStyle: { color: CHART_COLORS.border } }
        },
        visualMap: {
            min: 0, max: 1,
            calculable: true, orient: 'vertical', right: 10, top: 'center',
            inRange: { color: [CHART_COLORS.bgSurface, CHART_COLORS.primary] },
            textStyle: { color: CHART_COLORS.textSecondary }
        },
        series: [{ type: 'heatmap', data: heatmapData }]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染 Logits Lens 图
 * @param {Object} data - Logits Lens 数据
 */
function renderLogitsLensChart(data) {
    const chart = chartInstances.logitsLens;
    if (!chart) return;
    
    const { layer_predictions, num_layers, step, actual_token } = data;
    
    if (!layer_predictions || layer_predictions.length === 0) {
        chart.clear();
        return;
    }
    
    // 构建热力图：层 x Top-K tokens
    const topK = layer_predictions[0]?.top_tokens?.length || 5;
    const heatmapData = [];
    const allTokens = new Set();
    
    layer_predictions.forEach((lp, layerIdx) => {
        lp.top_tokens?.forEach((t, rank) => {
            allTokens.add(t.token);
            heatmapData.push([rank, layerIdx, t.probability]);
        });
    });
    
    const layerLabels = layer_predictions.map((_, i) => `L${i}`);
    const rankLabels = Array.from({length: topK}, (_, i) => `#${i+1}`);
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: 'Logits Lens - 层间预测演变',
            subtext: `实际生成: "${actual_token}"`,
            left: 'center',
            textStyle: { color: CHART_COLORS.textPrimary, fontSize: 14, fontWeight: 'normal' },
            subtextStyle: { color: CHART_COLORS.textSecondary, fontSize: 11 }
        },
        tooltip: {
            formatter: (params) => {
                const layer = params.data[1];
                const rank = params.data[0];
                const lp = layer_predictions[layer];
                const token = lp?.top_tokens?.[rank]?.token || '';
                return `Layer ${layer} #${rank+1}: "${token}" (${(params.data[2]*100).toFixed(1)}%)`;
            },
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        grid: { left: 50, right: 80, top: 70, bottom: 50 },
        xAxis: {
            type: 'category', data: rankLabels,
            axisLabel: { color: CHART_COLORS.textSecondary, fontSize: 10 },
            axisLine: { lineStyle: { color: CHART_COLORS.border } }
        },
        yAxis: {
            type: 'category', data: layerLabels, inverse: true,
            axisLabel: { color: CHART_COLORS.textDim, fontSize: 8 },
            axisLine: { lineStyle: { color: CHART_COLORS.border } }
        },
        visualMap: {
            min: 0, max: 1, calculable: true, orient: 'vertical', right: 10, top: 'center',
            inRange: { color: [CHART_COLORS.bgSurface, CHART_COLORS.primaryDim, CHART_COLORS.primary] },
            textStyle: { color: CHART_COLORS.textSecondary }
        },
        series: [{ type: 'heatmap', data: heatmapData }]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染残差流分析图
 * @param {Object} data - 残差流数据
 */
function renderResidualChart(data) {
    const chart = chartInstances.residual;
    if (!chart) return;
    
    const { layer_norms, residual_contributions, num_layers } = data;
    
    if (!layer_norms || layer_norms.length === 0) {
        chart.clear();
        return;
    }
    
    const layerLabels = Array.from({length: num_layers}, (_, i) => `L${i}`);
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: '残差流分析',
            subtext: '每层对残差流的贡献',
            left: 'center',
            textStyle: { color: CHART_COLORS.textPrimary, fontSize: 14, fontWeight: 'normal' },
            subtextStyle: { color: CHART_COLORS.textSecondary, fontSize: 11 }
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        legend: {
            data: ['隐藏状态范数', '残差贡献'],
            top: 50,
            textStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 }
        },
        grid: { left: 60, right: 20, top: 90, bottom: 50 },
        xAxis: {
            type: 'category', data: layerLabels,
            axisLabel: { color: CHART_COLORS.textDim, fontSize: 8 },
            axisLine: { lineStyle: { color: CHART_COLORS.border } }
        },
        yAxis: {
            type: 'value', name: '范数',
            nameTextStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 },
            axisLabel: { color: CHART_COLORS.textSecondary },
            axisLine: { lineStyle: { color: CHART_COLORS.border } },
            splitLine: { lineStyle: { color: CHART_COLORS.border, opacity: 0.3 } }
        },
        series: [
            {
                name: '隐藏状态范数', type: 'line', data: layer_norms,
                smooth: true, lineStyle: { color: CHART_COLORS.primary, width: 2 },
                itemStyle: { color: CHART_COLORS.primary }
            },
            {
                name: '残差贡献', type: 'bar', data: residual_contributions,
                itemStyle: { color: CHART_COLORS.accent, opacity: 0.7 }
            }
        ]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染 Token Embedding 投影图 (2D)
 * @param {Object} data - Embedding 数据
 */
function renderEmbeddingChart2D(data) {
    const chart = chartInstances.embedding;
    if (!chart) return;
    
    const { embeddings, tokens, token_types, explained_variance } = data;
    
    if (!embeddings || embeddings.length === 0) {
        chart.clear();
        return;
    }
    
    // 按类型分组
    const inputPoints = [], outputPoints = [];
    embeddings.forEach((emb, i) => {
        const point = { value: [emb[0], emb[1]], name: tokens[i] };
        if (token_types[i] === 'input') inputPoints.push(point);
        else outputPoints.push(point);
    });
    
    const varianceText = explained_variance 
        ? `方差解释: ${(explained_variance[0]*100).toFixed(1)}% + ${(explained_variance[1]*100).toFixed(1)}%`
        : 'PCA 降维';
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: 'Token Embedding 投影 (2D)',
            subtext: varianceText,
            left: 'center',
            textStyle: { color: CHART_COLORS.textPrimary, fontSize: 14, fontWeight: 'normal' },
            subtextStyle: { color: CHART_COLORS.textSecondary, fontSize: 11 }
        },
        tooltip: {
            formatter: (params) => `"${params.name}"`,
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        legend: {
            data: ['输入 Token', '生成 Token'],
            top: 50,
            textStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 }
        },
        grid: { left: 50, right: 20, top: 80, bottom: 50 },
        xAxis: { type: 'value', name: 'PC1', axisLine: { lineStyle: { color: CHART_COLORS.border } }, splitLine: { show: false } },
        yAxis: { type: 'value', name: 'PC2', axisLine: { lineStyle: { color: CHART_COLORS.border } }, splitLine: { show: false } },
        series: [
            {
                name: '输入 Token', type: 'scatter', data: inputPoints,
                symbolSize: 8, itemStyle: { color: CHART_COLORS.textDim }
            },
            {
                name: '生成 Token', type: 'scatter', data: outputPoints,
                symbolSize: 10, itemStyle: { color: CHART_COLORS.primary }
            }
        ]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染 Token Embedding 投影图 (3D)
 * @param {Object} data - Embedding 数据
 */
function renderEmbeddingChart3D(data) {
    const chart = chartInstances.embedding;
    if (!chart) return;
    
    const { embeddings, tokens, token_types, explained_variance } = data;
    
    if (!embeddings || embeddings.length === 0) {
        chart.clear();
        return;
    }
    
    // 按类型分组
    const inputPoints = [], outputPoints = [];
    embeddings.forEach((emb, i) => {
        const point = { 
            value: [emb[0], emb[1], emb[2] || 0], 
            name: tokens[i],
            itemStyle: { color: token_types[i] === 'input' ? CHART_COLORS.textDim : CHART_COLORS.primary }
        };
        if (token_types[i] === 'input') inputPoints.push(point);
        else outputPoints.push(point);
    });
    
    const varianceText = explained_variance 
        ? `方差解释: ${explained_variance.map(v => (v*100).toFixed(1) + '%').join(' + ')}`
        : 'PCA 降维';
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: 'Token Embedding 投影 (3D)',
            subtext: varianceText,
            left: 'center',
            textStyle: { color: CHART_COLORS.textPrimary, fontSize: 14, fontWeight: 'normal' },
            subtextStyle: { color: CHART_COLORS.textSecondary, fontSize: 11 }
        },
        tooltip: {
            formatter: (params) => `"${params.name}"`,
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        legend: {
            data: ['输入 Token', '生成 Token'],
            top: 50,
            textStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 }
        },
        grid3D: {
            viewControl: {
                autoRotate: false,
                rotateSensitivity: 2,
                zoomSensitivity: 1
            },
            axisLine: { lineStyle: { color: CHART_COLORS.border } },
            axisPointer: { lineStyle: { color: CHART_COLORS.primary } },
            light: {
                main: { intensity: 1.2 },
                ambient: { intensity: 0.3 }
            }
        },
        xAxis3D: { type: 'value', name: 'PC1' },
        yAxis3D: { type: 'value', name: 'PC2' },
        zAxis3D: { type: 'value', name: 'PC3' },
        series: [
            {
                name: '输入 Token', 
                type: 'scatter3D', 
                data: inputPoints,
                symbolSize: 6,
                itemStyle: { color: CHART_COLORS.textDim, opacity: 0.8 }
            },
            {
                name: '生成 Token', 
                type: 'scatter3D', 
                data: outputPoints,
                symbolSize: 8,
                itemStyle: { color: CHART_COLORS.primary, opacity: 0.9 }
            }
        ]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染 Token Embedding 投影图（兼容旧接口）
 * @param {Object} data - Embedding 数据
 */
function renderEmbeddingChart(data) {
    if (data.n_components === 3) {
        renderEmbeddingChart3D(data);
    } else {
        renderEmbeddingChart2D(data);
    }
}

/**
 * 渲染激活值分布图
 * @param {Object} data - 激活值数据
 */
function renderActivationChart(data) {
    const chart = chartInstances.activation;
    if (!chart) return;
    
    const { layer, mlp_stats, attention_stats, histogram } = data;
    
    if (!histogram) {
        chart.clear();
        return;
    }
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: `激活值分布 - Layer ${layer}`,
            left: 'center',
            textStyle: { color: CHART_COLORS.textPrimary, fontSize: 14, fontWeight: 'normal' }
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        legend: {
            data: ['MLP', 'Attention'],
            top: 30,
            textStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 }
        },
        grid: { left: 60, right: 20, top: 70, bottom: 50 },
        xAxis: {
            type: 'category', data: histogram.bins,
            axisLabel: { color: CHART_COLORS.textSecondary, fontSize: 9 },
            axisLine: { lineStyle: { color: CHART_COLORS.border } }
        },
        yAxis: {
            type: 'value', name: '频数',
            nameTextStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 },
            axisLabel: { color: CHART_COLORS.textSecondary },
            axisLine: { lineStyle: { color: CHART_COLORS.border } },
            splitLine: { lineStyle: { color: CHART_COLORS.border, opacity: 0.3 } }
        },
        series: [
            { name: 'MLP', type: 'bar', data: histogram.mlp_counts, itemStyle: { color: CHART_COLORS.primary, opacity: 0.7 } },
            { name: 'Attention', type: 'bar', data: histogram.attn_counts, itemStyle: { color: CHART_COLORS.accent, opacity: 0.7 } }
        ]
    };
    
    chart.setOption(option, true);
}

/**
 * 渲染输入归因热力图
 * @param {Object} data - 归因数据
 */
function renderAttributionChart(data) {
    const chart = chartInstances.attribution;
    if (!chart) return;
    
    const { attributions, input_tokens, output_token, output_idx } = data;
    
    if (!attributions || attributions.length === 0) {
        chart.clear();
        return;
    }
    
    const cleanTokens = input_tokens.map(t => t ? t.substring(0, 10) : '');
    
    const option = {
        backgroundColor: 'transparent',
        title: {
            text: '输入归因分析',
            subtext: `输出 Token: "${output_token}"`,
            left: 'center',
            textStyle: { color: CHART_COLORS.textPrimary, fontSize: 14, fontWeight: 'normal' },
            subtextStyle: { color: CHART_COLORS.textSecondary, fontSize: 11 }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: (params) => {
                const idx = params[0].dataIndex;
                return `"${input_tokens[idx]}": ${attributions[idx].toFixed(4)}`;
            },
            backgroundColor: CHART_COLORS.bgElevated,
            borderColor: CHART_COLORS.border,
            textStyle: { color: CHART_COLORS.textPrimary }
        },
        grid: { left: 60, right: 20, top: 70, bottom: 80 },
        xAxis: {
            type: 'category', data: cleanTokens,
            axisLabel: { color: CHART_COLORS.textDim, fontSize: 9, rotate: 45 },
            axisLine: { lineStyle: { color: CHART_COLORS.border } }
        },
        yAxis: {
            type: 'value', name: '归因分数',
            nameTextStyle: { color: CHART_COLORS.textSecondary, fontSize: 10 },
            axisLabel: { color: CHART_COLORS.textSecondary },
            axisLine: { lineStyle: { color: CHART_COLORS.border } },
            splitLine: { lineStyle: { color: CHART_COLORS.border, opacity: 0.3 } }
        },
        series: [{
            type: 'bar',
            data: attributions.map(a => ({
                value: a,
                itemStyle: { color: a > 0 ? CHART_COLORS.primary : CHART_COLORS.danger }
            })),
            barWidth: '60%'
        }]
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
    renderMultiheadChart,
    renderHeadEntropyChart,
    renderHiddenSimilarityChart,
    renderLogitsLensChart,
    renderResidualChart,
    renderEmbeddingChart,
    renderActivationChart,
    renderAttributionChart,
    clearAllCharts,
    resizeCharts
};

