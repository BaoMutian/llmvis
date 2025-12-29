"""
LLM 推理可视化 - Flask 应用
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from model_service import model_service
import analysis

app = Flask(__name__)
CORS(app)

# 模型路径
MODEL_PATH = os.path.expanduser("~/evo/models/qwen3-4b-instruct-2507")

# 存储最近一次推理的结果（用于后续查询）
latest_result = None


@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/api/model/load", methods=["POST"])
def load_model():
    """加载模型"""
    global MODEL_PATH

    data = request.json or {}
    model_path = data.get("model_path", MODEL_PATH)

    if model_service.is_loaded():
        return jsonify({
            "success": True,
            "message": "Model already loaded",
            "model_info": model_service.get_model_info()
        })

    success = model_service.load_model(model_path)

    if success:
        return jsonify({
            "success": True,
            "message": "Model loaded successfully",
            "model_info": model_service.get_model_info()
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to load model"
        }), 500


@app.route("/api/model/info", methods=["GET"])
def get_model_info():
    """获取模型信息"""
    if not model_service.is_loaded():
        return jsonify({
            "loaded": False,
            "message": "Model not loaded"
        })

    return jsonify({
        "loaded": True,
        "info": model_service.get_model_info()
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    """执行推理并返回详细分析数据"""
    global latest_result

    if not model_service.is_loaded():
        return jsonify({
            "success": False,
            "message": "Model not loaded. Please load the model first."
        }), 400

    data = request.json
    if not data or "prompt" not in data:
        return jsonify({
            "success": False,
            "message": "Missing 'prompt' in request body"
        }), 400

    prompt = data["prompt"]
    temperature = float(data.get("temperature", 0.7))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.9))
    max_tokens = int(data.get("max_tokens", 128))

    try:
        # 执行推理
        result = model_service.generate_with_details(
            prompt=prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens
        )

        # 分析生成结果
        logits_list = result["logits"]
        analysis_result = analysis.analyze_generation(logits_list)

        # 计算困惑度
        perplexity = analysis.compute_perplexity(
            logits_list,
            result["generated_token_ids"]
        )

        # 获取置信度曲线
        confidence_curve = analysis.get_confidence_curve(logits_list)

        # 存储结果供后续查询
        latest_result = result

        # 构建响应（不包含大型数组，按需获取）
        response = {
            "success": True,
            "prompt": result["prompt"],
            "generated_text": result["generated_text"],
            "input_tokens": result["input_tokens"],
            "generated_tokens": result["generated_tokens"],
            "num_generated": len(result["generated_tokens"]),
            "input_length": result["input_length"],
            "num_layers": result["num_layers"],
            "num_heads": result["num_heads"],
            "parameters": result["parameters"],
            "analysis": {
                "entropies": analysis_result["entropies"],
                "entropies_bits": analysis_result["entropies_bits"],
                "mean_entropy": analysis_result["mean_entropy"],
                "std_entropy": analysis_result["std_entropy"],
                "max_entropy": analysis_result["max_entropy"],
                "min_entropy": analysis_result["min_entropy"],
                "max_probs": analysis_result["max_probs"],
                "mean_max_prob": analysis_result["mean_max_prob"],
                "perplexity": perplexity,
                "confidence_curve": confidence_curve
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route("/api/attention", methods=["GET"])
def get_attention():
    """获取指定层的注意力矩阵"""
    global latest_result

    if latest_result is None:
        return jsonify({
            "success": False,
            "message": "No generation result available. Please generate first."
        }), 400

    layer_idx = int(request.args.get("layer", 0))
    head_idx = request.args.get("head", None)
    aggregation = request.args.get("aggregation", "mean")

    if head_idx is not None:
        head_idx = int(head_idx)

    num_layers = latest_result["num_layers"]
    if layer_idx < 0 or layer_idx >= num_layers:
        return jsonify({
            "success": False,
            "message": f"Invalid layer index. Must be between 0 and {num_layers - 1}"
        }), 400

    try:
        attention_matrix = model_service.get_attention_matrix(
            latest_result["attentions"],
            layer_idx=layer_idx,
            head_idx=head_idx,
            aggregation=aggregation
        )

        # 计算注意力统计信息
        attn_stats = analysis.compute_attention_statistics(attention_matrix)

        # 获取token标签
        all_tokens = latest_result["all_tokens"]

        return jsonify({
            "success": True,
            "layer": layer_idx,
            "head": head_idx,
            "aggregation": aggregation,
            "attention_matrix": attention_matrix.tolist(),
            "y_labels": latest_result["generated_tokens"],  # 生成的token
            "x_labels": all_tokens[:attention_matrix.shape[1]] if attention_matrix.size > 0 else [],
            "statistics": attn_stats
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route("/api/token_probs", methods=["GET"])
def get_token_probs():
    """获取指定位置的token概率分布"""
    global latest_result

    if latest_result is None:
        return jsonify({
            "success": False,
            "message": "No generation result available. Please generate first."
        }), 400

    position = int(request.args.get("position", 0))
    top_n = int(request.args.get("top_n", 10))

    num_generated = len(latest_result["generated_tokens"])
    if position < 0 or position >= num_generated:
        return jsonify({
            "success": False,
            "message": f"Invalid position. Must be between 0 and {num_generated - 1}"
        }), 400

    try:
        logits = latest_result["logits"][position]

        # 获取top-k概率
        top_k_probs = model_service.get_top_k_tokens(logits, k=top_n)

        # 分析概率分布特征
        dist_analysis = analysis.analyze_token_probability_distribution(logits)

        # 获取该位置实际生成的token
        actual_token = latest_result["generated_tokens"][position]
        actual_token_id = latest_result["generated_token_ids"][position]

        return jsonify({
            "success": True,
            "position": position,
            "actual_token": actual_token,
            "actual_token_id": actual_token_id,
            "top_k_tokens": top_k_probs,
            "distribution_analysis": dist_analysis
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route("/api/layer_similarity", methods=["GET"])
def get_layer_similarity():
    """获取层间注意力相似度矩阵"""
    global latest_result

    if latest_result is None:
        return jsonify({
            "success": False,
            "message": "No generation result available. Please generate first."
        }), 400

    step = int(request.args.get("step", 0))

    num_generated = len(latest_result["generated_tokens"])
    if step < 0 or step >= num_generated:
        return jsonify({
            "success": False,
            "message": f"Invalid step. Must be between 0 and {num_generated - 1}"
        }), 400

    try:
        similarity_matrix = analysis.compute_layer_similarity(
            latest_result["attentions"],
            step
        )

        return jsonify({
            "success": True,
            "step": step,
            "similarity_matrix": similarity_matrix.tolist(),
            "num_layers": latest_result["num_layers"]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Inference Visualization Server")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print("Loading model on startup...")

    # 启动时加载模型
    if model_service.load_model(MODEL_PATH):
        print("Model loaded successfully!")
        print(f"Model info: {model_service.get_model_info()}")
    else:
        print("Warning: Failed to load model. You can load it later via API.")

    print("=" * 60)
    print("Starting Flask server...")
    print("Open http://localhost:5050 in your browser")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5050, debug=False)
