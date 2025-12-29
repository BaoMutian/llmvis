"""
分析模块
负责熵计算、概率分布分析等统计功能
"""

import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    计算softmax
    
    Args:
        x: 输入数组
        axis: 计算轴
        
    Returns:
        softmax概率分布
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_entropy(logits: np.ndarray) -> float:
    """
    计算给定logits的信息熵
    
    H = -sum(p * log(p))
    
    Args:
        logits: logits数组 [vocab_size]
        
    Returns:
        熵值（以自然对数为底）
    """
    probs = softmax(logits)
    # 避免log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def compute_entropy_bits(logits: np.ndarray) -> float:
    """
    计算给定logits的信息熵（以比特为单位）
    
    H = -sum(p * log2(p))
    
    Args:
        logits: logits数组 [vocab_size]
        
    Returns:
        熵值（以2为底的对数，单位：比特）
    """
    probs = softmax(logits)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def analyze_generation(logits_list: List[np.ndarray]) -> Dict[str, Any]:
    """
    分析整个生成过程的统计信息
    
    Args:
        logits_list: 每个生成步骤的logits列表
        
    Returns:
        分析结果字典
    """
    if not logits_list:
        return {
            "entropies": [],
            "entropies_bits": [],
            "mean_entropy": 0,
            "std_entropy": 0,
            "max_entropy": 0,
            "min_entropy": 0,
            "max_probs": [],
            "mean_max_prob": 0
        }
    
    # 计算每个token位置的熵
    entropies = [compute_entropy(logits) for logits in logits_list]
    entropies_bits = [compute_entropy_bits(logits) for logits in logits_list]
    
    # 计算每个位置的最大概率（置信度）
    max_probs = []
    for logits in logits_list:
        probs = softmax(logits)
        max_probs.append(float(np.max(probs)))
    
    return {
        "entropies": entropies,
        "entropies_bits": entropies_bits,
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "max_entropy": float(np.max(entropies)),
        "min_entropy": float(np.min(entropies)),
        "max_probs": max_probs,
        "mean_max_prob": float(np.mean(max_probs))
    }


def get_top_k_probabilities(
    logits: np.ndarray,
    k: int = 10,
    tokenizer=None
) -> List[Dict[str, Any]]:
    """
    获取概率最高的k个token及其概率
    
    Args:
        logits: logits数组 [vocab_size]
        k: 返回的token数量
        tokenizer: 分词器（用于解码token）
        
    Returns:
        top-k token信息列表
    """
    probs = softmax(logits)
    top_k_indices = np.argsort(probs)[-k:][::-1]
    
    results = []
    for idx in top_k_indices:
        token_info = {
            "token_id": int(idx),
            "probability": float(probs[idx]),
            "logit": float(logits[idx])
        }
        
        if tokenizer is not None:
            token_info["token"] = tokenizer.decode([idx])
        
        results.append(token_info)
    
    return results


def compute_perplexity(logits_list: List[np.ndarray], token_ids: List[int]) -> float:
    """
    计算困惑度
    
    PPL = exp(mean(-log(p(token))))
    
    Args:
        logits_list: 每个位置的logits
        token_ids: 实际生成的token id列表
        
    Returns:
        困惑度
    """
    if len(logits_list) != len(token_ids):
        raise ValueError("logits_list and token_ids must have the same length")
    
    if not logits_list:
        return 1.0
    
    log_probs = []
    for logits, token_id in zip(logits_list, token_ids):
        probs = softmax(logits)
        prob = probs[token_id]
        # 避免log(0)
        prob = max(prob, 1e-10)
        log_probs.append(np.log(prob))
    
    mean_log_prob = np.mean(log_probs)
    perplexity = np.exp(-mean_log_prob)
    
    return float(perplexity)


def compute_attention_statistics(
    attention_matrix: np.ndarray
) -> Dict[str, Any]:
    """
    计算注意力矩阵的统计信息
    
    Args:
        attention_matrix: 注意力矩阵 [query_tokens, key_tokens]
        
    Returns:
        统计信息字典
    """
    if attention_matrix.size == 0:
        return {
            "mean": 0,
            "std": 0,
            "max": 0,
            "min": 0,
            "sparsity": 1.0
        }
    
    # 计算稀疏度（接近0的值的比例）
    threshold = 0.01
    sparsity = np.mean(attention_matrix < threshold)
    
    return {
        "mean": float(np.mean(attention_matrix)),
        "std": float(np.std(attention_matrix)),
        "max": float(np.max(attention_matrix)),
        "min": float(np.min(attention_matrix)),
        "sparsity": float(sparsity)
    }


def compute_attention_entropy(attention_weights: np.ndarray) -> float:
    """
    计算注意力分布的熵
    
    Args:
        attention_weights: 注意力权重 [seq_len]（已归一化）
        
    Returns:
        注意力熵
    """
    weights = np.clip(attention_weights, 1e-10, 1.0)
    # 确保归一化
    weights = weights / weights.sum()
    entropy = -np.sum(weights * np.log(weights))
    return float(entropy)


def analyze_token_probability_distribution(
    logits: np.ndarray
) -> Dict[str, Any]:
    """
    分析token概率分布的特征
    
    Args:
        logits: logits数组 [vocab_size]
        
    Returns:
        分布特征字典
    """
    probs = softmax(logits)
    
    # 计算各种统计量
    entropy = compute_entropy(logits)
    entropy_bits = compute_entropy_bits(logits)
    
    # 排序后的概率
    sorted_probs = np.sort(probs)[::-1]
    
    # 累积概率（用于查看多少token覆盖了大部分概率）
    cumsum = np.cumsum(sorted_probs)
    
    # 找到覆盖50%、90%、99%概率所需的token数
    tokens_for_50 = int(np.searchsorted(cumsum, 0.5) + 1)
    tokens_for_90 = int(np.searchsorted(cumsum, 0.9) + 1)
    tokens_for_99 = int(np.searchsorted(cumsum, 0.99) + 1)
    
    # 峰度和偏度
    kurtosis = float(stats.kurtosis(probs))
    skewness = float(stats.skew(probs))
    
    return {
        "entropy": entropy,
        "entropy_bits": entropy_bits,
        "max_probability": float(sorted_probs[0]),
        "top_5_probability_sum": float(sorted_probs[:5].sum()),
        "top_10_probability_sum": float(sorted_probs[:10].sum()),
        "tokens_for_50_percent": tokens_for_50,
        "tokens_for_90_percent": tokens_for_90,
        "tokens_for_99_percent": tokens_for_99,
        "kurtosis": kurtosis,
        "skewness": skewness
    }


def compute_layer_similarity(
    attentions: List[List[np.ndarray]],
    step_idx: int
) -> np.ndarray:
    """
    计算各层注意力模式的相似度矩阵
    
    Args:
        attentions: 注意力数据 [step][layer][heads, seq]
        step_idx: 步骤索引
        
    Returns:
        层间相似度矩阵 [num_layers, num_layers]
    """
    if not attentions or step_idx >= len(attentions):
        return np.array([])
    
    step_attentions = attentions[step_idx]
    num_layers = len(step_attentions)
    
    similarity_matrix = np.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            # 展平注意力模式并计算余弦相似度
            attn_i = step_attentions[i].flatten()
            attn_j = step_attentions[j].flatten()
            
            # 余弦相似度
            norm_i = np.linalg.norm(attn_i)
            norm_j = np.linalg.norm(attn_j)
            
            if norm_i > 0 and norm_j > 0:
                similarity = np.dot(attn_i, attn_j) / (norm_i * norm_j)
            else:
                similarity = 0
            
            similarity_matrix[i, j] = similarity
    
    return similarity_matrix


def get_confidence_curve(logits_list: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    获取生成过程的置信度曲线
    
    Args:
        logits_list: 每个位置的logits列表
        
    Returns:
        置信度数据列表
    """
    results = []
    
    for idx, logits in enumerate(logits_list):
        probs = softmax(logits)
        sorted_probs = np.sort(probs)[::-1]
        
        results.append({
            "position": idx,
            "max_prob": float(sorted_probs[0]),
            "second_prob": float(sorted_probs[1]) if len(sorted_probs) > 1 else 0,
            "prob_gap": float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0]),
            "top_3_sum": float(sorted_probs[:3].sum()),
            "top_5_sum": float(sorted_probs[:5].sum())
        })
    
    return results

