import json
import os
import requests
import numpy as np

# ----------------------- 配置区 -----------------------
# 远程嵌入服务地址
API_URL = "https://api.siliconflow.cn/v1/embeddings"
# 访问密钥（如有需要，请自行替换为你的环境变量或明文字符串）
API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-qmyqlcevlelaxuxuvwhkpdqsyhoadeaudrawwylzhntpuknv")

# 自动检测路径配置
def get_data_paths():
    """根据当前工作目录自动调整数据文件路径"""
    current_dir = os.path.basename(os.getcwd())
    if current_dir == "new-code":
        # 从 new-code 目录内运行
        data_prefix = "../new-data/"
    else:
        # 从项目根目录运行
        data_prefix = "new-data/"

    return {
        "data_path": f"{data_prefix}sentence.json",
        "output_path": f"{data_prefix}Qwen3-Embedding-8B-output.json"
    }

# 获取路径配置
paths = get_data_paths()
DATA_PATH = paths["data_path"]
SIM_MATRIX_PATH = paths["output_path"]
# ------------------------------------------------------

# 想要测试的 sentence_id 列表，直接在此修改即可。例如 [0] 或 [0, 1]
TARGET_IDS = [0]


def load_sentences(json_path: str):
    """读取双语句子 JSON 文件并返回对象列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embeddings(text_list):
    """调用 嵌入向量模型 Qwen3-Embedding-8B 的 远程接口，返回与输入文本顺序对应的嵌入矩阵 (n, d)。"""
    if not text_list:
        # 返回空矩阵以保持类型一致
        return np.empty((0, 0), dtype=np.float32)

    payload = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": text_list,
        "encoding_format": "float",
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    try:
        resp_json = response.json()
        embeddings = [item.get("embedding") for item in resp_json.get("data", [])]
    except ValueError:
        raise RuntimeError(f"接口返回非 JSON 内容: {response.text[:200]}")

    if len(embeddings) != len(text_list):
        raise RuntimeError("返回的嵌入数量与输入文本数量不一致")

    return np.asarray(embeddings, dtype=np.float32)


def compute_similarity_matrix(emb_en: np.ndarray, emb_zh: np.ndarray):
    """已知各向量已做 L2 归一化, 直接点积即可得到余弦相似度矩阵."""
    # emb_en: (m, d), emb_zh: (n, d) -> (m, n)
    return np.matmul(emb_en, emb_zh.T)


def main():
    target_ids = set(TARGET_IDS)

    sentences_data = load_sentences(DATA_PATH)

    # 过滤出用户指定的 sentence_id
    filtered_items = [item for item in sentences_data if item.get("sentence_id") in target_ids]

    if not filtered_items:
        print("未找到任何匹配的 sentence_id，请检查输入！")
        return

    for item in filtered_items:
        sid = item.get("sentence_id")
        en_sentences = item.get("english_sentence_text", [])
        zh_sentences = item.get("chinese_sentence_text", [])

        # 将英/中文子句拼接, 一次请求减少网络开销
        combined_texts = en_sentences + zh_sentences
        embeddings = get_embeddings(combined_texts)
        emb_en = embeddings[: len(en_sentences)]
        emb_zh = embeddings[len(en_sentences) :]

        sim_matrix = compute_similarity_matrix(emb_en, emb_zh)

        # 打印结果
        print(f"\n=== sentence_id: {sid} ===")
        print(f"英文子句 (行, 共 {len(en_sentences)} 条):", en_sentences)
        print(f"中文子句 (列, 共 {len(zh_sentences)} 条):", zh_sentences)
        print(f"相似度矩阵 (点积) 形状: {sim_matrix.shape}")
        # 使用 numpy 打印，保留 3 位小数便于阅读
        np.set_printoptions(precision=3, suppress=True)
        print(sim_matrix)

        # ---------------- 保存到 JSON 文件 ----------------
        result_entry = {
            "sentence_id": sid,
            "semantic_similarity_matrix": sim_matrix.tolist(),
        }

        # 如果文件已存在，先读取并更新/追加
        existing_data = []
        if os.path.exists(SIM_MATRIX_PATH):
            with open(SIM_MATRIX_PATH, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    # 若文件内容损坏则重置
                    existing_data = []

        # 更新已有条目或追加
        updated = False
        for idx, entry in enumerate(existing_data):
            if entry.get("sentence_id") == sid:
                existing_data[idx] = result_entry
                updated = True
                break
        if not updated:
            existing_data.append(result_entry)

        # 写回文件
        with open(SIM_MATRIX_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        # --------------------------------------------------


if __name__ == "__main__":
    main()

