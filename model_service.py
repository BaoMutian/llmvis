"""
LLM 模型服务模块
负责模型加载、推理、注意力和logits提取
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import threading


class ModelService:
    """LLM 模型服务类"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式确保模型只加载一次"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_path = None
        self._initialized = True
    
    def load_model(self, model_path: str) -> bool:
        """
        加载模型和分词器
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否加载成功
        """
        try:
            print(f"Loading model from {model_path}...")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 加载模型，启用注意力输出
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"  # 使用eager模式以获取注意力权重
            )
            
            self.model.eval()
            self.model_path = model_path
            
            print(f"Model loaded successfully!")
            print(f"Model config: {self.model.config.num_hidden_layers} layers, {self.model.config.num_attention_heads} attention heads")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_loaded():
            return {}
        
        config = self.model.config
        return {
            "model_path": self.model_path,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "device": str(self.device)
        }
    
    def generate_with_details(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        max_tokens: int = 128
    ) -> Dict[str, Any]:
        """
        执行推理并返回详细信息（注意力、logits等）
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
            top_k: top-k采样参数
            top_p: top-p采样参数
            max_tokens: 最大生成token数
            
        Returns:
            包含生成结果和详细分析数据的字典
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        # 构建对话格式
        messages = [{"role": "user", "content": prompt}]
        
        # 使用 chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]
        
        # 存储每一步的详细信息
        all_attentions = []  # [step][layer][batch, heads, seq, seq]
        all_logits = []  # [step][vocab_size]
        generated_tokens = []
        generated_token_ids = []
        
        # 逐token生成以获取每一步的注意力
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for step in range(max_tokens):
                # 前向传播，获取注意力
                outputs = self.model(
                    current_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # 获取最后一个位置的logits
                logits = outputs.logits[:, -1, :]  # [batch, vocab]
                
                # 应用温度
                if temperature > 0:
                    logits = logits / temperature
                
                # 计算概率分布
                probs = torch.softmax(logits, dim=-1)
                
                # Top-k 过滤
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs_filtered = torch.zeros_like(probs)
                    probs_filtered.scatter_(1, top_k_indices, top_k_probs)
                    probs = probs_filtered
                
                # Top-p (nucleus) 过滤
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    probs[indices_to_remove] = 0
                
                # 重新归一化
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # 采样
                if temperature > 0:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                next_token_id = next_token.item()
                
                # 检查是否是结束token
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                # 保存当前步骤的数据
                # 注意力：只保存最后一个token位置相关的注意力
                step_attentions = []
                for layer_attn in outputs.attentions:
                    # layer_attn: [batch, heads, seq, seq]
                    # 取最后一行（最后一个token对所有token的注意力）
                    attn_weights = layer_attn[0, :, -1, :].cpu().numpy()  # [heads, seq]
                    step_attentions.append(attn_weights)
                all_attentions.append(step_attentions)
                
                # 保存原始logits（应用温度后）
                all_logits.append(outputs.logits[:, -1, :].cpu().numpy()[0])
                
                # 保存生成的token
                token_str = self.tokenizer.decode([next_token_id])
                generated_tokens.append(token_str)
                generated_token_ids.append(next_token_id)
                
                # 更新输入
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        # 获取输入tokens（逐个解码以正确显示中文等字符）
        input_token_ids = input_ids[0].tolist()
        input_tokens = [self.tokenizer.decode([tid]) for tid in input_token_ids]
        
        # 构建完整的token序列（用于注意力可视化）
        all_tokens = input_tokens + generated_tokens
        
        # 生成完整回复
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "generated_token_ids": generated_token_ids,
            "all_tokens": all_tokens,
            "attentions": all_attentions,  # [step][layer][heads, seq]
            "logits": all_logits,  # [step][vocab]
            "input_length": input_length,
            "num_layers": self.model.config.num_hidden_layers,
            "num_heads": self.model.config.num_attention_heads,
            "parameters": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_tokens": max_tokens
            }
        }
    
    def get_attention_matrix(
        self,
        attentions: List[List[np.ndarray]],
        layer_idx: int,
        head_idx: Optional[int] = None,
        aggregation: str = "mean"
    ) -> np.ndarray:
        """
        获取指定层的注意力矩阵
        
        Args:
            attentions: 注意力数据 [step][layer][heads, seq]
            layer_idx: 层索引
            head_idx: 头索引，None表示聚合所有头
            aggregation: 聚合方式 "mean" 或 "max"
            
        Returns:
            注意力矩阵 [generated_tokens, all_tokens]
        """
        if not attentions:
            return np.array([])
        
        num_steps = len(attentions)
        
        # 构建注意力矩阵
        # 每一步的注意力形状是 [heads, current_seq_len]
        # 我们需要构建 [generated_tokens, input_tokens + generated_tokens] 的矩阵
        
        attention_rows = []
        for step_idx, step_attn in enumerate(attentions):
            layer_attn = step_attn[layer_idx]  # [heads, seq_len]
            
            if head_idx is not None:
                # 使用单个头
                attn_row = layer_attn[head_idx, :]
            else:
                # 聚合所有头
                if aggregation == "mean":
                    attn_row = np.mean(layer_attn, axis=0)
                else:  # max
                    attn_row = np.max(layer_attn, axis=0)
            
            attention_rows.append(attn_row)
        
        # 将所有行填充到相同长度
        max_len = max(len(row) for row in attention_rows)
        padded_rows = []
        for row in attention_rows:
            if len(row) < max_len:
                padded = np.pad(row, (0, max_len - len(row)), mode='constant', constant_values=0)
            else:
                padded = row
            padded_rows.append(padded)
        
        return np.array(padded_rows)
    
    def get_top_k_tokens(
        self,
        logits: np.ndarray,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取概率最高的k个token
        
        Args:
            logits: logits数组 [vocab_size]
            k: 返回的token数量
            
        Returns:
            top-k token列表，包含token、概率和logit值
        """
        # 计算概率
        probs = self._softmax(logits)
        
        # 获取top-k
        top_k_indices = np.argsort(probs)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            token = self.tokenizer.decode([idx])
            results.append({
                "token_id": int(idx),
                "token": token,
                "probability": float(probs[idx]),
                "logit": float(logits[idx])
            })
        
        return results
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """计算softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


# 全局模型服务实例
model_service = ModelService()

