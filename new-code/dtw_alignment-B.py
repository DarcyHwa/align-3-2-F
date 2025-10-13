# -*- coding:utf8 -*-
"""
DTW对齐算法模块 - 基于已划分区间的动态时间规整对齐

本模块实现了跨语言句子对齐系统的核心DTW算法部分：
1. 读取已划分完成的对齐区间数据
2. 实现完整的DTW动态时间规整算法
3. 支持多种句子组合模式（1-1、1-多、多-1、2-2等）
4. 计算最优对齐路径和相似度得分
5. 输出对齐结果到多种格式

核心算法流程：
1. 加载区间划分数据和相似度矩阵
2. 初始化DTW算法参数和数据结构
3. 在每个区间内执行DTW搜索
4. 计算距离函数和路径优化
5. 生成最终对齐结果

技术特点：
- 完全基于您已实现的数据结构
- 复现参考代码的DTW核心算法
- 支持灵活的句子组合配置
- 高效的缓存和优化机制

作者：Augment-2
"""

import os
import sys
import json
import math
import time
import numpy as np
import requests
from typing import Dict, List, Tuple, Any
from collections import defaultdict


# 全局常量
INFINITE = float('inf')

# 低层算法参数（启用所有功能的优化配置）
coeff_sent_len = 0.33  # 平衡路径距离中"句长惩罚"的权重（值越大越依赖句长）
coeff_neighbour_sim = 0.6  # 邻近句相似度惩罚系数，用于抑制对语境相似但非对应句子的错误对齐
only_one_2_one_pairing = False  # 启用多种对齐模式，不限制仅使用 1-1 / 1-0 / 0-1 三种配对形式


class SimpleOOBTree:
    """
    简单的有序字典实现，替代BTrees.OOBTree
    """
    def __init__(self):
        self._data = {}

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def maxKey(self):
        if not self._data:
            raise ValueError("Empty tree")
        return max(self._data.keys())

    def keys(self):
        return sorted(self._data.keys())

    def items(self):
        return [(k, self._data[k]) for k in sorted(self._data.keys())]

# DTW算法参数（启用所有功能的优化配置）
COEFF_SENT_LEN = 0.33  # 平衡路径距离中"句长惩罚"的权重
COEFF_NEIGHBOUR_SIM = 0.6  # 邻近句相似度惩罚系数
ONLY_ONE_2_ONE_PAIRING = False  # 启用多种对齐模式，不限制仅使用 1-1 / 1-0 / 0-1 三种配对形式


def load_interval_data(interval_file_path: str, sentence_id: int) -> Dict[str, Any]:
    """
    从JSON文件中加载指定sentence_id的区间划分数据
    
    参数:
        interval_file_path (str): 区间数据文件路径
        sentence_id (int): 目标句子ID
        
    返回值:
        Dict[str, Any]: 包含区间、锚点、句子等完整数据的字典
    """
    print(f"\n=== 加载区间划分数据 ===")
    print(f"文件路径: {interval_file_path}")
    print(f"目标句子ID: {sentence_id}")
    
    try:
        with open(interval_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data.get('sentence_id') != sentence_id:
            raise ValueError(f"文件中的sentence_id ({data.get('sentence_id')}) 与目标ID ({sentence_id}) 不匹配")
        
        print(f"✓ 数据加载成功")
        print(f"  最终锚点数量: {data['final_anchors']['count']}")
        print(f"  可对齐区间数量: {data['alignable_intervals']['count']}")
        print(f"  英文句子数: {data['source_data']['document_size']['english_sentences']}")
        print(f"  中文句子数: {data['source_data']['document_size']['chinese_sentences']}")
        
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"区间数据文件未找到: {interval_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件格式错误: {e}")


def load_similarity_matrix(matrix_file_path: str, sentence_id: int) -> np.ndarray:
    """
    从JSON文件中加载指定sentence_id的相似度矩阵
    
    参数:
        matrix_file_path (str): 相似度矩阵文件路径
        sentence_id (int): 目标句子ID
        
    返回值:
        np.ndarray: 相似度矩阵
    """
    print(f"\n=== 加载相似度矩阵 ===")
    print(f"文件路径: {matrix_file_path}")
    
    try:
        with open(matrix_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if item.get('sentence_id') == sentence_id:
                matrix = np.array(item['semantic_similarity_matrix'], dtype=np.float64)
                print(f"✓ 相似度矩阵加载成功，维度: {matrix.shape}")
                return matrix
        
        raise ValueError(f"未找到sentence_id为{sentence_id}的相似度矩阵")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"相似度矩阵文件未找到: {matrix_file_path}")


def load_sentence_data(sentence_file_path: str, sentence_id: int) -> Tuple[List[str], List[str]]:
    """
    从JSON文件中加载指定sentence_id的句子数据
    
    参数:
        sentence_file_path (str): 句子数据文件路径
        sentence_id (int): 目标句子ID
        
    返回值:
        Tuple[List[str], List[str]]: (英文句子列表, 中文句子列表)
    """
    print(f"\n=== 加载句子数据 ===")
    print(f"文件路径: {sentence_file_path}")
    
    try:
        with open(sentence_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if item.get('sentence_id') == sentence_id:
                english_sentences = item.get('english_sentence_text', [])
                chinese_sentences = item.get('chinese_sentence_text', [])
                print(f"✓ 句子数据加载成功")
                print(f"  英文句子数: {len(english_sentences)}")
                print(f"  中文句子数: {len(chinese_sentences)}")
                return english_sentences, chinese_sentences
        
        raise ValueError(f"未找到sentence_id为{sentence_id}的句子数据")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"句子数据文件未找到: {sentence_file_path}")


def call_qwen3_embedding(text_list: List[str]) -> List[np.ndarray]:
    """
    调用远程Qwen3-Embedding-8B模型计算文本列表的向量表示
    支持批量处理，提高效率

    参数:
        text_list (List[str]): 需要向量化的文本列表

    返回值:
        List[np.ndarray]: 对应的嵌入向量列表
    """
    if not text_list:
        print("⚠️ 输入文本列表为空，返回空列表")
        return []

    try:
        # 调用远程Qwen3-Embedding-8B API
        api_url = "https://api.siliconflow.cn/v1/embeddings"
        api_key = "sk-qmyqlcevlelaxuxuvwhkpdqsyhoadeaudrawwylzhntpuknv"

        payload = {
            "model": "Qwen/Qwen3-Embedding-8B",
            "input": text_list,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        print(f"正在调用Qwen3-Embedding-8B模型，处理 {len(text_list)} 个文本...")
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        resp_json = response.json()
        embeddings_data = resp_json.get("data", [])

        if len(embeddings_data) != len(text_list):
            raise RuntimeError(f"返回的嵌入数量({len(embeddings_data)})与输入文本数量({len(text_list)})不一致")

        # 处理返回的嵌入向量
        embeddings = []
        for item in embeddings_data:
            embedding = np.array(item.get("embedding"), dtype=np.float64)

            # Qwen3-Embedding-8B已经L2归一化，但为了安全起见再次检查
            norm = np.linalg.norm(embedding)
            if norm > 0 and abs(norm - 1.0) > 1e-6:
                print(f"警告：远程向量未完全归一化，norm={norm:.6f}，重新归一化")
                embedding = embedding / norm

            embeddings.append(embedding)

        print(f"✓ Qwen3-Embedding-8B调用成功，返回 {len(embeddings)} 个向量")

        # 记录API调用统计
        if not hasattr(call_qwen3_embedding, 'stats'):
            call_qwen3_embedding.stats = {'total_calls': 0, 'total_texts': 0}
        call_qwen3_embedding.stats['total_calls'] += 1
        call_qwen3_embedding.stats['total_texts'] += len(text_list)

        return embeddings

    except requests.exceptions.RequestException as e:
        print(f"❌ Qwen3-Embedding-8B调用失败: {e}")
        raise e
    except Exception as e:
        print(f"❌ 嵌入向量计算异常: {e}")
        raise e





def call_single_qwen3_embedding(text: str) -> np.ndarray:
    """
    调用单个文本的Qwen3-Embedding-8B嵌入向量计算
    这是对批量函数的简单包装

    参数:
        text (str): 需要向量化的文本

    返回值:
        np.ndarray: 嵌入向量
    """
    results = call_qwen3_embedding([text])
    return results[0]


def print_qwen3_stats():
    """
    打印Qwen3-Embedding-8B API调用统计信息
    """
    if hasattr(call_qwen3_embedding, 'stats'):
        stats = call_qwen3_embedding.stats
        print(f"\n📈 Qwen3-Embedding-8B API调用统计:")
        print(f"  - 总调用次数: {stats['total_calls']}")
        print(f"  - 总处理文本数: {stats['total_texts']}")
        print(f"  - 平均每次调用文本数: {stats['total_texts'] / max(1, stats['total_calls']):.1f}")
    else:
        print("📈 未进行Qwen3-Embedding-8B API调用")


def safe_vector_norm(vector: np.ndarray) -> float:
    """
    安全计算向量的模长，包含异常处理

    参数:
        vector (np.ndarray): 输入向量

    返回值:
        float: 向量的模长
    """
    try:
        norm = np.linalg.norm(vector)
        return norm
    except Exception as e:
        print(f"linalg.norm 计算失败: {e}，使用手动计算")
        # 手动计算向量模长
        norm = 0.0
        for k in range(len(vector)):
            norm += vector[k] ** 2
        norm = math.sqrt(norm)
        print(f"手动计算向量模长: {norm}")
        return norm


def calculate_max_group_size(sents1: List[str], sents2: List[str]) -> int:
    """
    动态计算最大组合大小

    根据双语句子数量动态计算最大组合大小：
    max_group_size = ceil(max(len(sents1), len(sents2)) / 2)

    参数:
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表

    返回值:
        int: 计算得到的最大组合大小
    """
    max_sentences = max(len(sents1), len(sents2))
    max_group_size = math.ceil(max_sentences / 2)

    # 设置合理的最小值和最大值
    max_group_size = max(2, max_group_size)  # 最小为2
    max_group_size = min(max_sentences - 1, max_group_size)  # 最大为双语最大子句数减1

    print(f"📊 动态计算最大组合大小:")
    print(f"  - 源语言句子数: {len(sents1)}")
    print(f"  - 目标语言句子数: {len(sents2)}")
    print(f"  - 最大句子数: {max_sentences}")
    print(f"  - 计算公式: ceil({max_sentences} / 2) = {math.ceil(max_sentences / 2)}")
    print(f"  - 范围限制: 最小为2，最大为{max_sentences - 1}")
    print(f"  - 最终最大组合大小: {max_group_size}")

    return max_group_size


def safe_vector_normalize(vector: np.ndarray) -> np.ndarray:
    """
    安全归一化向量，包含异常处理

    参数:
        vector (np.ndarray): 输入向量

    返回值:
        np.ndarray: 归一化后的向量
    """
    norm = safe_vector_norm(vector)
    if norm > 0:
        return vector / norm
    else:
        print("警告：零向量无法归一化")
        return vector


def generate_allowed_groups(params: Dict[str, Any], sents1: List[str], sents2: List[str]) -> List[Tuple[int, int]]:
    """
    生成允许的句子组合模式

    参数:
        params (Dict[str, Any]): 参数字典
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表

    返回值:
        List[Tuple[int, int]]: 允许的组合列表，如[(0,1), (1,0), (1,1), (1,2), (2,1), (2,2)]
    """
    print(f"\n=== 生成允许的句子组合 ===")

    # 动态计算最大组合大小
    max_group_size = calculate_max_group_size(sents1, sents2)

    allowed_groups = [(0, 1), (1, 0), (1, 1)]  # 基础组合：空对1，1对空，1对1

    if not only_one_2_one_pairing:
        for i in range(2, max_group_size + 1):
            allowed_groups.append((1, i))  # 1对多
            allowed_groups.append((i, 1))  # 多对1

        if params.get('noEmptyPair', False):
            if (1, 0) in allowed_groups:
                allowed_groups.remove((1, 0))  # 移除空对齐
            if (0, 1) in allowed_groups:
                allowed_groups.remove((0, 1))

        if not params.get('no2_2Group', False):
            allowed_groups.append((2, 2))  # 2对2组合

    # 将结果存储到params中（与参考代码一致）
    params['allowedGroups'] = allowed_groups

    print(f"*** 允许的句子组合: {allowed_groups}")
    print(f"*** 使用动态计算的最大组合大小: {max_group_size}")
    return allowed_groups


def lenPenalty(len1: float, len2: float) -> float:
    """
    根据句长比例计算长度惩罚，值域 [0,1]；长度越接近惩罚越小。
    参考 Bertalign 算法。

    参数:
        len1 (float): 第一个句子的长度
        len2 (float): 第二个句子的长度

    返回值:
        float: 长度惩罚值，范围[0,1]
    """
    min_len = min(len1, len2)  # 较短长度
    max_len = max(len1, len2)  # 较长长度
    if max_len == 0:
        return 0
    return 1 - np.log2(1 + min_len / max_len)  # 对数惩罚函数


def len_penalty(len1: float, len2: float) -> float:
    """
    根据句长比例计算长度惩罚，值域 [0,1]；长度越接近惩罚越小
    参考 Bertalign 算法
    
    参数:
        len1 (float): 第一个句子的长度
        len2 (float): 第二个句子的长度
    
    返回值:
        float: 长度惩罚值，范围[0,1]
    """
    min_len = min(len1, len2)
    max_len = max(len1, len2)
    if max_len == 0:
        return 0
    return 1 - np.log2(1 + min_len / max_len)


def distance_dtw(sents1: List[str], sents2: List[str], sim_mat: np.ndarray,
                embeds1: List[np.ndarray], embeds2: List[np.ndarray],
                inf_i: int, i: int, inf_j: int, j: int, char_ratio: float,
                dist_hash: Dict[str, float], params: Dict[str, Any], use_coeff: bool = True) -> float:
    """
    计算源区间 (inf_i,i] 与目标区间 (inf_j,j] 的加权距离
    完整复现参考代码的distance_dtw函数逻辑

    参数:
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表
        sim_mat (np.ndarray): 相似度矩阵
        embeds1 (List[np.ndarray]): 源语言嵌入向量列表
        embeds2 (List[np.ndarray]): 目标语言嵌入向量列表
        inf_i (int): 源区间起始坐标
        i (int): 源区间结束坐标
        inf_j (int): 目标区间起始坐标
        j (int): 目标区间结束坐标
        char_ratio (float): 字符比例
        dist_hash (Dict[str, float]): 距离缓存
        params (Dict[str, Any]): 参数字典
        use_coeff (bool): 是否使用系数

    返回值:
        float: 计算得到的加权距离
    """
    # 如果距离已存储在dist_hash中，直接返回
    key = f"{inf_i}-{i};{inf_j}-{j}"
    if key in dist_hash:
        return dist_hash[key]

    # coeff表示对齐中涉及的总段数（两种语言）
    coeff = 1
    penalty = params.get('penalty_n_n', 0.06)  # 多对多对齐惩罚

    # 1-0和0-1关系的情况（空对齐）
    if inf_i == i or inf_j == j:
        dist_null = params.get('distNull', 1.0) * coeff
        dist_hash[key] = dist_null
        return dist_null

    if i < 0 or j < 0 or inf_i < -2 or inf_j < -2:
        dist_hash[key] = INFINITE
        return INFINITE

    coeff = 2

    # 计算相似度
    if params.get('useEncoder', True) and embeds1 is not None and embeds2 is not None:
        # 使用嵌入向量计算相似度
        # 1-1关系的情况
        if inf_i == i - 1 and inf_j == j - 1:
            sim = sim_mat[i, j]
            if use_coeff:
                penalty = 0
        # n-n关系的情况
        else:
            # 计算embed_i（源语言嵌入向量）
            if inf_i == i - 1:
                # 单个句子，直接使用预计算的嵌入向量
                embed_i = embeds1[i].copy()
                len_i = len(sents1[inf_i + 1])
            else:
                # 多个句子，需要拼接文本并动态调用Qwen3-Embedding-8B计算嵌入向量
                source_sentences = []
                sent_i = sents1[inf_i + 1]
                source_sentences.append(f"[{inf_i + 1}] {sent_i}")
                for coord_i in range(inf_i + 2, i + 1):
                    sentence = sents1[coord_i]
                    source_sentences.append(f"[{coord_i}] {sentence}")
                    sent_i += " " + sentence  # 使用空格作为拼接分隔符
                    if use_coeff:
                        coeff += 1
                len_i = len(sent_i)

                # 动态调用Qwen3-Embedding-8B模型计算组合句子的嵌入向量
                embed_i = call_single_qwen3_embedding(sent_i)
                if params.get('veryVerbose', False):
                    print(f"    🔗 组合源文本 (句子 {inf_i + 1}-{i}):")
                    for sent_info in source_sentences:
                        print(f"      {sent_info}")
                    print(f"    📝 组合源文本完整内容: '{sent_i}'")
                    print(f"    📊 组合源文本向量维度: {embed_i.shape}")

            # 计算embed_j（目标语言嵌入向量）
            if inf_j == j - 1:
                # 单个句子，直接使用预计算的嵌入向量
                embed_j = embeds2[j].copy()
                len_j = len(sents2[inf_j + 1])
            else:
                # 多个句子，需要拼接文本并动态调用Qwen3-Embedding-8B计算嵌入向量
                target_sentences = []
                sent_j = sents2[inf_j + 1]
                target_sentences.append(f"[{inf_j + 1}] {sent_j}")
                for coord_j in range(inf_j + 2, j + 1):
                    sentence = sents2[coord_j]
                    target_sentences.append(f"[{coord_j}] {sentence}")
                    sent_j += " " + sentence  # 使用空格作为拼接分隔符
                    if use_coeff:
                        coeff += 1
                len_j = len(sent_j)

                # 动态调用Qwen3-Embedding-8B模型计算组合句子的嵌入向量
                embed_j = call_single_qwen3_embedding(sent_j)
                if params.get('veryVerbose', False):
                    print(f"    🔗 组合目标文本 (句子 {inf_j + 1}-{j}):")
                    for sent_info in target_sentences:
                        print(f"      {sent_info}")
                    print(f"    📝 组合目标文本完整内容: '{sent_j}'")
                    print(f"    📊 组合目标文本向量维度: {embed_j.shape}")

            sim = float(np.matmul(embed_i, np.transpose(embed_j)))
    else:
        # 使用预计算的相似度矩阵
        # 1-1关系的情况：无惩罚
        if inf_i == i - 1 and inf_j == j - 1:
            sim = sim_mat[i, j]
            penalty = 0
        else:
            # n-n关系的情况 - 计算平均相似度
            sim_sum = 0
            count = 0
            for x in range(inf_i + 1, i + 1):
                for y in range(inf_j + 1, j + 1):
                    if 0 <= x < len(sents1) and 0 <= y < len(sents2):
                        sim_sum += sim_mat[x, y]
                        count += 1
                        if use_coeff:
                            coeff += 1
            sim = sim_sum / count if count > 0 else 0

        # 计算句长
        len_i = sum(len(sents1[x]) for x in range(max(0, inf_i + 1), min(i + 1, len(sents1))))
        len_j = sum(len(sents2[y]) for y in range(max(0, inf_j + 1), min(j + 1, len(sents2))))

    # 计算句长（如果还没有计算）
    if 'len_i' not in locals():
        len_i = sum(len(sents1[x]) for x in range(max(0, inf_i + 1), min(i + 1, len(sents1))))
    if 'len_j' not in locals():
        len_j = sum(len(sents2[y]) for y in range(max(0, inf_j + 1), min(j + 1, len(sents2))))

    # 计算与邻近句子的相似度并从全局相似度中减去（邻近句惩罚）
    if (not params.get('noMarginPenalty', False) and embeds1 is not None and embeds2 is not None and
        'embed_i' in locals() and 'embed_j' in locals()):
        nb = 0  # 邻居计数器
        nn = 0  # 有效邻居组数

        # 计算目标语言邻近句相似度
        if inf_j >= 0 and inf_j < len(embeds2):
            left_embed_j = embeds2[inf_j].copy()  # 左侧邻近句嵌入向量
            left_sim_j = np.matmul(embed_i, np.transpose(left_embed_j))
            nb += 1
        else:
            left_sim_j = 0

        if j + 1 < len(embeds2):
            right_embed_j = embeds2[j + 1].copy()  # 右侧邻近句嵌入向量
            right_sim_j = np.matmul(embed_i, np.transpose(right_embed_j))
            nb += 1
        else:
            right_sim_j = 0

        neighbour_sim_j = 0  # 目标语言邻近句相似度
        if nb > 0:
            neighbour_sim_j = (left_sim_j + right_sim_j) / nb
            nn += 1

        # 计算源语言邻近句相似度
        nb = 0
        if inf_i >= 0 and inf_i < len(embeds1):
            left_embed_i = embeds1[inf_i].copy()  # 左侧邻近句嵌入向量
            left_sim_i = np.matmul(left_embed_i, np.transpose(embed_j))
            nb += 1
        else:
            left_sim_i = 0

        if i + 1 < len(embeds1):
            right_embed_i = embeds1[i + 1].copy()  # 右侧邻近句嵌入向量
            right_sim_i = np.matmul(right_embed_i, np.transpose(embed_j))
            nb += 1
        else:
            right_sim_i = 0

        neighbour_sim_i = 0  # 源语言邻近句相似度
        if nb > 0:
            neighbour_sim_i = (left_sim_i + right_sim_i) / nb
            nn += 1

        # 计算平均邻近句相似度并应用惩罚
        average_neighbour_sim = 0  # 平均邻近句相似度
        if nn > 0:
            average_neighbour_sim = (neighbour_sim_i + neighbour_sim_j) / nn
        sim -= coeff_neighbour_sim * average_neighbour_sim

    # 处理空句子的情况
    if len_i * len_j == 0:
        dist_null = params.get('distNull', 1.0) * coeff
        dist_hash[key] = dist_null
        return dist_null

    # 计算最终距离
    dist = 1 - sim  # 距离 = 1 - 相似度
    if use_coeff:
        dist += penalty * coeff  # 添加惩罚项

    # 结合句长惩罚和相似度距离
    dist = (1 - coeff_sent_len) * dist + coeff_sent_len * lenPenalty(len_i * char_ratio, len_j)

    dist *= coeff  # 乘以系数
    dist_hash[key] = dist  # 缓存结果
    return dist


def dtw(sents1: List[str], sents2: List[str], sim_mat: np.ndarray,
        embeds1: List[np.ndarray], embeds2: List[np.ndarray],
        path_hash: Dict[Tuple[int, int], List], dist_hash: Dict[str, float],
        x_2_y: Dict[int, int], y_2_x: Dict[int, int],
        x_begin: int, y_begin: int, x_end: int, y_end: int,
        char_ratio: float, allowed_groups: List[Tuple[int, int]],
        params: Dict[str, Any]) -> Tuple[List[List[int]], float]:
    """
    在指定区间内递推计算局部最优路径
    完整复现参考代码的dtw函数逻辑

    参数:
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表
        sim_mat (np.ndarray): 相似度矩阵
        embeds1 (List[np.ndarray]): 源语言嵌入向量列表
        embeds2 (List[np.ndarray]): 目标语言嵌入向量列表
        path_hash (Dict): 路径哈希表
        dist_hash (Dict): 距离哈希表
        x_2_y (Dict): x到y的锚点映射
        y_2_x (Dict): y到x的锚点映射
        x_begin (int): x起始坐标
        y_begin (int): y起始坐标
        x_end (int): x结束坐标
        y_end (int): y结束坐标
        char_ratio (float): 字符比例
        allowed_groups (List): 允许的组合列表
        params (Dict): 参数字典

    返回值:
        Tuple[List[List[int]], float]: (最优路径, 累积距离)
    """
    if params.get('veryVerbose', False):
        print(f"    DTW计算区间: ({x_begin},{y_begin}) -> ({x_end},{y_end})")

    for i in range(x_begin, x_end + 1):
        for j in range(y_begin, y_end + 1):
            # 路径哈希表记录已计算路径的结果，以减少递归
            dtw_key = (i, j)

            # 如果已计算过则跳过
            if dtw_key in path_hash:
                continue

            path_by_group = {}  # 各组的路径记录
            dist_by_group = {}  # 各组的距离记录

            # 检查每个允许的组合
            for group in params['allowedGroups']:
                previous_i = i - group[0]  # 前一个i坐标
                previous_j = j - group[1]  # 前一个j坐标
                previous_key = (previous_i, previous_j)  # 前一个点的键

                # 原则上previous_key应该能找到
                if previous_key in path_hash:
                    (path_by_group[group], dist_by_group[group]) = path_hash[previous_key]
                else:
                    (path_by_group[group], dist_by_group[group]) = ([], INFINITE)

                # 为当前组增加距离
                dist_by_group[group] += distance_dtw(sents1, sents2, sim_mat, embeds1, embeds2,
                                                   previous_i, i, previous_j, j,
                                                   char_ratio, dist_hash, params)

            best_group = None  # 最佳组合
            min_dist = INFINITE  # 最小距离
            for group in params['allowedGroups']:
                if dist_by_group[group] < min_dist:
                    min_dist = dist_by_group[group]
                    best_group = group

            if best_group is not None:
                path = path_by_group[best_group][:]  # 注意：这里创建副本！
                path.append([i, j])
                path_hash[dtw_key] = [path, min_dist]
            else:
                path_hash[dtw_key] = [[], INFINITE]

    return path_hash[(x_end, y_end)]


def run_dtw(interval_data: Dict[str, Any], sim_mat: np.ndarray,
           sents1: List[str], sents2: List[str], embeds1: List[np.ndarray], embeds2: List[np.ndarray],
           params: Dict[str, Any]) -> Tuple[List[List[int]], List[List[int]], float]:
    """
    在锚点/区间约束下执行动态时间规整（DTW）算法
    完整复现参考代码的run_dtw函数逻辑，包括束搜索、锚点约束等

    参数:
        interval_data (Dict): 区间划分数据
        sim_mat (np.ndarray): 相似度矩阵
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表
        embeds1 (List[np.ndarray]): 源语言嵌入向量列表
        embeds2 (List[np.ndarray]): 目标语言嵌入向量列表
        params (Dict): 参数字典

    返回值:
        Tuple[List[List[int]], List[List[int]], float]: (x路径, y路径, 总得分)
    """
    print(f"\n=== 开始执行DTW算法 ===")

    # 获取区间和锚点信息
    intervals = interval_data['alignable_intervals']['intervals']
    final_anchors = interval_data['final_anchors']['anchors']

    print(f"可对齐区间数量: {len(intervals)}")
    print(f"最终锚点数量: {len(final_anchors)}")

    # 获取已生成的允许组合（避免重复计算）
    allowed_groups = params.get('allowedGroups', [])

    # 构建全局锚点映射（从区间数据中提取）
    filtered_x = []
    filtered_y = []
    # 注意：已去除预定义锚点功能，只使用动态提取的锚点

    for anchor in final_anchors:
        x, y = anchor['coordinates']
        filtered_x.append(x)
        filtered_y.append(y)

    # 计算字符比例
    nb_chars1 = sum(len(sent) for sent in sents1)
    nb_chars2 = sum(len(sent) for sent in sents2)
    char_ratio = nb_chars2 / nb_chars1 if nb_chars1 > 0 else 1.0

    print(f"字符比例: {char_ratio:.3f}")

    # 初始化DTW数据结构
    path_hash = {}  # 路径哈希表，记录到达每个点的最佳路径
    dist_hash = {"-2--1;-2--1": 0.0}  # 对于点(-1,-1)的下界，距离为0

    # 初始化空路径 - 这是关键步骤！
    x_first = intervals[0]['coordinates']['start'][0]  # 第一个区间的起始x坐标
    y_first = intervals[0]['coordinates']['start'][1]  # 第一个区间的起始y坐标

    path_hash[(x_first, y_first)] = [[[-1, -1]], 0]  # 初始路径和得分
    print(f"首个对齐点: {x_first}-{y_first}")
    print(f"初始化 DTW: 起点 {intervals[0]['coordinates']['start']} 终点 {intervals[-1]['coordinates']['end']}")

    # 转换区间格式以匹配参考代码
    converted_intervals = []
    for interval in intervals:
        start_coords = interval['coordinates']['start']
        end_coords = interval['coordinates']['end']
        converted_intervals.append([start_coords, end_coords])

    lastBestPath = [[x_first - 1, y_first - 1]]  # 上一个最佳路径
    lastBestScore = 0  # 上一个最佳得分

    t8 = time.time()

    # 处理每个可对齐区间
    for interval in converted_intervals:
        (x_begin, y_begin) = interval[0]  # 区间起始坐标
        (x_end, y_end) = interval[1]      # 区间结束坐标
        print(f"正在处理区间 {interval}")
        key_xy = (x_begin, y_begin)
        coeff_y_per_x = (y_end - y_begin) / (x_end - x_begin) if (x_end - x_begin) != 0 else 1  # Y轴相对X轴的斜率

        # 这些字典用于引导路径靠近位于区间内的锚点
        x_2_y = {}  # X坐标到Y坐标的映射
        y_2_x = {}  # Y坐标到X坐标的映射
        anchor_count = 0
        for i in range(len(filtered_x)):
            x = filtered_x[i]
            y = filtered_y[i]
            if x < x_begin:
                continue
            if x > x_end:
                break
            if y < y_begin or y > y_end:
                continue
            x_2_y[x] = y
            y_2_x[y] = x
            anchor_count += 1

        print(f"*** 区间内锚点数量: {anchor_count}")

        # 连接可对齐区间之间的空隙
        # 如果路径中最后一个点与当前区间第一个点之间有空隙，在路径中添加空点()
        if key_xy not in path_hash:
            (lastI, lastJ) = lastBestPath[-1]
            if params.get('verbose', False):
                print(f"在路径中插入空隙: ({lastI},{lastJ}) → ({x_begin},{y_begin})")
            lastBestPath.append(())  # 空点表示路径中的断点
            lastBestPath.append((x_begin - 1, y_begin - 1))
            path_hash[key_xy] = [lastBestPath, lastBestScore]

        # 现在在区间内每个锚点之间运行DTW搜索
        # 路径是递归计算的，但为了最小化递归深度，
        # DTW哈希通过逐点调用函数逐步填充
        previous_x = x_begin  # 前一个处理的x坐标
        previous_y = y_begin  # 前一个处理的y坐标
        processed_anchors = 0

        for x in range(x_begin, x_end + 1):
            localBeam = params.get('dtwBeam', 7)  # 局部搜索束宽（优化后默认值）

            # 如果(x,y)是锚点，从x开始运行DTW！
            if x in x_2_y:
                y = x_2_y[x]
                if params.get('verbose', False):
                    print(f"处理锚点 ({x},{y})")

                # 注意：已去除预定义锚点功能，所有锚点都使用相同的局部束宽
                # 计算偏差和束宽
                # 如果(x,y)距离区间对角线太远，将被丢弃
                deviation = 0
                if y >= y_begin and (x_end - x_begin) * (y_end - y_begin) != 0:
                    deviation = abs((y - y_begin) / (y_end - y_begin) - (x - x_begin) / (x_end - x_begin))
                else:
                    continue

                # 第一个条件：偏差 > localDiagBeam
                if (deviation > params.get('localDiagBeam', 0.35) and
                    deviation * (y_end - y_begin) > params.get('dtwBeam', 7)):
                    del x_2_y[x]
                    if y in y_2_x:
                        del y_2_x[y]
                    if params.get('verbose', False):
                        print(f"*** 偏差过大：{deviation * (y_end - y_begin):.3f} - 锚点 ({x},{y}) 距离区间对角线太远，已丢弃")
                        continue

                    # 第二个条件：deltaX和deltaY之间的比例超过4（最大允许1-4或4-1分组）
                    if (params.get('noEmptyPair', False) and (
                            min(y - previous_y, x - previous_x) == 0 or
                            max(y - previous_y, x - previous_x) / min(y - previous_y, x - previous_x) > 4)):
                        del x_2_y[x]
                        if y in y_2_x:
                            del y_2_x[y]
                        if params.get('verbose', False):
                            print(f"*** 偏差锚点 ({x},{y}) 距离前一个点太近，已丢弃")
                        continue

                    # 处理空隙（考虑非单调性）：
                    # 如果 y < previous_y，扩大区域：将y设为previous_y，减少previous_x，
                    # 对应到x_2_y[prev_x] < y的最后一个点

                    if y < previous_y:
                        print(f"*** 单调性偏差：y={y} < previous_y={previous_y}，重新计算 previous_x")
                        prev_x = previous_x
                        # 根据y查找前一个点
                        found = False
                        while prev_x > x_begin:
                            prev_x -= 1
                            if prev_x in x_2_y:
                                prev_y = x_2_y[prev_x]
                                if prev_y < y:
                                    y = previous_y
                                    previous_x = prev_x
                                    previous_y = prev_y
                                    found = True
                                    break
                        if not found:
                            y = previous_y
                            previous_x = x_begin
                            previous_y = y_begin

                # 计算下界值以给定区间来切断递归：x_inf,y_inf之前的点不予考虑
                x_inf = previous_x - localBeam
                y_inf = previous_y - localBeam

                if params.get('veryVerbose', False):
                    print(f"启动 DTW 子搜索: ({max(x_begin, x_inf)},{max(y_begin, y_inf)}) → ({x},{y})")

                print(f"启动 DTW 子搜索: ({max(x_begin, x_inf)},{max(y_begin, y_inf)}) → ({x},{y})")
                (path, dist) = dtw(sents1, sents2, sim_mat, embeds1, embeds2, path_hash, dist_hash, x_2_y, y_2_x,
                                   max(x_begin, x_inf), max(y_begin, y_inf), x, y, char_ratio, allowed_groups, params)

                if dist == INFINITE and params.get('verbose', False):
                    print(f"从点 ({x},{y}) 起路径不可达，距离为 ∞")
                    # 从x,y开始初始化新区间
                    x_begin = x
                    y_begin = y
                    key_xy = (x_begin, y_begin)
                    # 在此创建lastBestPath的副本，并添加断点
                    lastBestPath = lastBestPath[:]
                    lastBestPath.append(())  # 空点表示路径中的断点
                    lastBestPath.append((x_begin - 1, y_begin - 1))
                    path_hash[key_xy] = [lastBestPath, lastBestScore]
                else:
                    lastBestPath = path
                    lastBestScore = dist

                if params.get('veryVerbose', False):
                    print(f"当前路径距离 = {dist}")
                previous_x = x
                previous_y = y

        (lastBestPath, lastBestScore) = path_hash[(previous_x, previous_y)]

    # 与文本末尾进行连接
    last_x = len(sents1) - 1
    last_y = len(sents2) - 1
    if (last_x - previous_x) + (last_y - previous_y) < 200:
        if params.get('verbose', False):
            print(f"末尾对齐点 ({last_x},{last_y})")
        dtw(sents1, sents2, sim_mat, embeds1, embeds2, path_hash, dist_hash, x_2_y, y_2_x,
            previous_x, previous_y, last_x, last_y, char_ratio, allowed_groups, params)

    # 如果最后一个点未被丢弃
    score = INFINITE
    if (last_x, last_y) in path_hash:
        (best_path, score) = path_hash[(last_x, last_y)]
    # 否则使用最后一个区间
    if score == INFINITE:
        (best_path, score) = path_hash[(previous_x, previous_y)]

    t9 = time.time()
    if params.get('verbose', False):
        print(f"\n9. Elapsed time for complete DTW-->", t9 - t8, "s.\n")

    print(f"\n✓ DTW算法完成，总得分: {score:.4f}")
    print(f"完整路径: {best_path}")

    # 转换路径格式为分组对齐
    # 参考代码的路径格式：每个点表示一个区间的上界
    # 例如 [[-1,-1],[0,1],[3,2]] 表示对齐：(0:0,1), (1,2,3:2)
    x_dtw = []
    y_dtw = []

    # 处理路径，将连续的坐标转换为分组
    prev_x = -1
    prev_y = -1

    for i, point in enumerate(best_path):
        if i == 0 and point == [-1, -1]:
            # 跳过起始虚拟点
            prev_x = -1
            prev_y = -1
            continue
        elif point == ():
            # 遇到断点，跳过
            continue
        elif len(point) == 2 and point[0] >= 0 and point[1] >= 0:
            # 有效对齐点，创建从前一个点到当前点的分组
            curr_x = point[0]
            curr_y = point[1]

            # 创建x分组：从prev_x+1到curr_x
            x_group = []
            for x in range(prev_x + 1, curr_x + 1):
                if 0 <= x < len(sents1):
                    x_group.append(x)

            # 创建y分组：从prev_y+1到curr_y
            y_group = []
            for y in range(prev_y + 1, curr_y + 1):
                if 0 <= y < len(sents2):
                    y_group.append(y)

            # 只有当两个组都非空时才添加
            if x_group and y_group:
                x_dtw.append(x_group)
                y_dtw.append(y_group)

            prev_x = curr_x
            prev_y = curr_y

    # 如果没有有效对齐，创建默认1-1对齐
    if not x_dtw:
        print("未找到有效对齐点，创建默认1-1对齐")
        min_len = min(len(sents1), len(sents2))
        for i in range(min_len):
            x_dtw.append([i])
            y_dtw.append([i])
        score = 1.0

    print(f"最终对齐组数: {len(x_dtw)}")
    return x_dtw, y_dtw, score


def calc_int(group_x: List[int], group_y: List[int]) -> Tuple[int, int, int, int]:
    """
    计算组的边界坐标

    参数:
        group_x (List[int]): X坐标组
        group_y (List[int]): Y坐标组

    返回值:
        Tuple[int, int, int, int]: (x_inf, x_sup, y_inf, y_sup)
    """
    if len(group_x) == 0:
        x_inf = x_sup = -1
    else:
        x_inf = group_x[0] - 1  # 组的下界是第一个元素减1
        x_sup = group_x[-1]     # 组的上界是最后一个元素

    if len(group_y) == 0:
        y_inf = y_sup = -1
    else:
        y_inf = group_y[0] - 1  # 组的下界是第一个元素减1
        y_sup = group_y[-1]     # 组的上界是最后一个元素

    return (x_inf, x_sup, y_inf, y_sup)


def prev(groups: List[Dict], i: int) -> int:
    """
    找到前一个有效组的索引

    参数:
        groups (List[Dict]): 组列表
        i (int): 当前组索引

    返回值:
        int: 前一个有效组的索引，如果没有则返回-1
    """
    for j in range(i - 1, -1, -1):
        if not groups[j].get('deleted', False):
            return j
    return -1


def next(groups: List[Dict], i: int) -> int:
    """
    找到下一个有效组的索引

    参数:
        groups (List[Dict]): 组列表
        i (int): 当前组索引

    返回值:
        int: 下一个有效组的索引，如果没有则返回-1
    """
    for j in range(i + 1, len(groups)):
        if not groups[j].get('deleted', False):
            return j
    return -1


def compute_gain(gains: SimpleOOBTree, groups: List[Dict], i: int, sents1: List[str], sents2: List[str],
                sim_mat: np.ndarray, embeds1: List[np.ndarray], embeds2: List[np.ndarray],
                char_ratio: float, params: Dict[str, Any]) -> None:
    """
    计算将第 i 个组与左右相邻组合并所带来的相似度增益

    参数:
        gains (SimpleOOBTree): 增益排序B树
        groups (List[Dict]): 组信息列表
        i (int): 当前组索引
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表
        sim_mat (np.ndarray): 相似度矩阵
        embeds1 (List[np.ndarray]): 源语言嵌入向量列表
        embeds2 (List[np.ndarray]): 目标语言嵌入向量列表
        char_ratio (float): 字符比例
        params (Dict[str, Any]): 参数字典
    """
    if groups[i].get('deleted', False):
        return

    group_x = groups[i]['x']  # 当前组的X坐标
    group_y = groups[i]['y']  # 当前组的Y坐标
    dist = groups[i]['dist']  # 当前组的距离

    # 清除之前的增益记录
    if 'gain' in groups[i]:
        old_gain = groups[i]['gain']
        if old_gain in gains:
            if isinstance(gains[old_gain], list) and i in gains[old_gain]:
                gains[old_gain].remove(i)
                if len(gains[old_gain]) == 0:
                    del gains[old_gain]

    # 计算与前一组合并的增益
    prev_i = prev(groups, i)
    prev_gain = 0
    if prev_i != -1:
        prev_group_x = groups[prev_i]['x']
        prev_group_y = groups[prev_i]['y']
        no_empty = len(prev_group_x) > 0 and len(prev_group_y) > 0  # 检查是否为非空组
        new_group_x1 = prev_group_x + group_x  # 合并后的X组
        new_group_y1 = prev_group_y + group_y  # 合并后的Y组
        (inf_x, sup_x, inf_y, sup_y) = calc_int(new_group_x1, new_group_y1)
        prev_dist = distance_dtw(sents1, sents2, sim_mat, embeds1, embeds2, inf_x, sup_x,
                                inf_y, sup_y, char_ratio, {}, params, False)
        prev_gain = dist - prev_dist  # 增益 = 原距离 - 新距离
        if no_empty:
            prev_gain -= params.get('penalty_n_n', 0.06)  # 非空组合并惩罚
        else:
            prev_gain += params.get('penalty_0_n', 0.1)  # 空组合并奖励

    # 计算与下一组合并的增益
    next_i = next(groups, i)
    next_gain = 0
    if next_i != -1:
        next_group_x = groups[next_i]['x']
        next_group_y = groups[next_i]['y']
        no_empty = len(next_group_x) > 0 and len(next_group_y) > 0  # 检查是否为非空组
        new_group_x2 = group_x + next_group_x  # 合并后的X组
        new_group_y2 = group_y + next_group_y  # 合并后的Y组
        (inf_x, sup_x, inf_y, sup_y) = calc_int(new_group_x2, new_group_y2)
        next_dist = distance_dtw(sents1, sents2, sim_mat, embeds1, embeds2, inf_x, sup_x,
                                inf_y, sup_y, char_ratio, {}, params, False)
        next_gain = dist - next_dist  # 增益 = 原距离 - 新距离
        if no_empty:
            next_gain -= params.get('penalty_n_n', 0.06)  # 非空组合并惩罚
        else:
            next_gain += params.get('penalty_0_n', 0.1)  # 空组合并奖励

    # 选择最佳增益方向
    if next_gain > prev_gain and next_gain > 0:
        groups[i]['gain'] = next_gain
        groups[i]['direction'] = 1  # 向右合并
        groups[i]['newX'] = group_x + next_group_x
        groups[i]['newY'] = group_y + next_group_y
        groups[i]['newDist'] = next_dist
        groups[i]['mergeWith'] = next_i

        # 添加到增益B树
        if next_gain not in gains:
            gains[next_gain] = []
        gains[next_gain].append(i)

    elif prev_gain > 0:
        groups[i]['gain'] = prev_gain
        groups[i]['direction'] = -1  # 向左合并
        groups[i]['newX'] = prev_group_x + group_x
        groups[i]['newY'] = prev_group_y + group_y
        groups[i]['newDist'] = prev_dist
        groups[i]['mergeWith'] = prev_i

        # 添加到增益B树
        if prev_gain not in gains:
            gains[prev_gain] = []
        gains[prev_gain].append(i)
    else:
        # 没有正增益
        groups[i]['gain'] = 0
        groups[i]['direction'] = 0


def late_grouping(x_dtw: List[List[int]], y_dtw: List[List[int]], sents1: List[str], sents2: List[str],
                 sim_mat: np.ndarray, embeds1: List[np.ndarray], embeds2: List[np.ndarray],
                 char_ratio: float, params: Dict[str, Any]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    在获得初步 DTW 句子级对齐结果后，通过贪心方式尝试将相邻组合并，
    以提升整体相似度得分并减少空对齐。

    使用Qwen3-Embedding-8B模型进行高质量的组合句子向量化。

    参数:
        x_dtw (List[List[int]]): X侧对齐分组列表
        y_dtw (List[List[int]]): Y侧对齐分组列表
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表
        sim_mat (np.ndarray): 相似度矩阵
        embeds1 (List[np.ndarray]): 源语言句向量列表
        embeds2 (List[np.ndarray]): 目标语言句向量列表
        char_ratio (float): 字符长度比例
        params (Dict[str, Any]): 参数字典

    返回值:
        Tuple[List[List[int]], List[List[int]]]: 更新后的 x_dtw, y_dtw 路径
    """
    print("*** 开始执行后期分组优化（使用Qwen3-Embedding-8B）...")

    # 这个B树记录按增益排序的每个组的索引
    gains = SimpleOOBTree()  # 增益排序B树
    groups = []  # 组信息列表

    # 预收集所有需要计算的组合文本，进行批量向量化
    print("*** 预收集组合文本进行批量向量化...")
    all_combined_texts = []
    text_to_group_mapping = {}

    # 初始化组数据结构：对每个组，记录x,y和对应的距离
    print("*** 初始化后期分组数据结构...")
    for (group_x, group_y) in zip(x_dtw, y_dtw):
        (inf_x, sup_x, inf_y, sup_y) = calc_int(group_x, group_y)

        # 收集可能需要的组合文本
        if inf_x != sup_x - 1:  # 多个源句子需要组合
            source_sentences = [sents1[idx] for idx in range(inf_x + 1, sup_x + 1) if 0 <= idx < len(sents1)]
            combined_text_x = " ".join(source_sentences)
            if combined_text_x not in all_combined_texts:
                all_combined_texts.append(combined_text_x)
                text_to_group_mapping[combined_text_x] = 'source'
                print(f"    📝 收集组合源文本 (句子 {inf_x+1}-{sup_x}):")
                for i, sent in enumerate(source_sentences, inf_x+1):
                    print(f"      [{i}] {sent}")
                print(f"    🔗 组合后完整源文本: '{combined_text_x}'")

        if inf_y != sup_y - 1:  # 多个目标句子需要组合
            target_sentences = [sents2[idx] for idx in range(inf_y + 1, sup_y + 1) if 0 <= idx < len(sents2)]
            combined_text_y = " ".join(target_sentences)
            if combined_text_y not in all_combined_texts:
                all_combined_texts.append(combined_text_y)
                text_to_group_mapping[combined_text_y] = 'target'
                print(f"    📝 收集组合目标文本 (句子 {inf_y+1}-{sup_y}):")
                for i, sent in enumerate(target_sentences, inf_y+1):
                    print(f"      [{i}] {sent}")
                print(f"    🔗 组合后完整目标文本: '{combined_text_y}'")

        # 计算初始距离
        dist = distance_dtw(sents1, sents2, sim_mat, embeds1, embeds2, inf_x, sup_x, inf_y,
                           sup_y, char_ratio, {}, params, False)
        groups.append({
            'x': group_x,
            'y': group_y,
            'original_x': group_x.copy(),  # 保存原始分组
            'original_y': group_y.copy(),  # 保存原始分组
            "dist": dist,
            'deleted': False
        })

    # 批量计算组合文本的嵌入向量
    if all_combined_texts:
        print(f"*** 批量计算 {len(all_combined_texts)} 个组合文本的嵌入向量...")
        batch_embeddings = call_qwen3_embedding(all_combined_texts)
        print(f"*** 批量向量化完成")

        # 详细显示所有组合文本
        print(f"\n📋 **完整组合文本列表** ({len(all_combined_texts)} 个):")
        source_count = 0
        target_count = 0
        for i, text in enumerate(all_combined_texts, 1):
            text_type = text_to_group_mapping.get(text, 'unknown')
            if text_type == 'source':
                source_count += 1
                print(f"  🇨🇳 [{i}] 组合源文本: '{text}'")
            elif text_type == 'target':
                target_count += 1
                print(f"  🇺🇸 [{i}] 组合目标文本: '{text}'")
            else:
                print(f"  ❓ [{i}] 未知类型文本: '{text}'")

        print(f"\n📊 组合文本统计:")
        print(f"  - 组合源文本数量: {source_count}")
        print(f"  - 组合目标文本数量: {target_count}")
        print(f"  - 总计: {len(all_combined_texts)}")
    else:
        print("*** 无需要组合的文本")

    print(f"*** 共有 {len(groups)} 个初始分组")

    # 第一轮迭代：对每个组，计算向左或向右分组的相似度增益
    print("*** 计算各组合并增益...")
    for i in range(len(groups)):
        compute_gain(gains, groups, i, sents1, sents2, sim_mat, embeds1, embeds2, char_ratio, params)

    if len(gains) > 0:
        best_gain = gains.maxKey()  # 获取最大增益
        print(f"*** 最佳初始增益: {best_gain:.6f}")
    else:
        best_gain = 0
        print("*** 没有正增益，跳过分组优化")

    # 当最佳分组能产生正增益时，继续合并
    merge_count = 0
    while best_gain > 0:
        # 获取具有最佳增益的组列表
        best_groups = gains[best_gain]
        i = best_groups[0]  # 选择第一个组进行合并

        # 从增益B树中移除
        best_groups.remove(i)
        if len(best_groups) == 0:
            del gains[best_gain]

        # 执行合并
        merge_with = groups[i]['mergeWith']
        direction = groups[i]['direction']

        print(f"*** 合并组 {i} 与组 {merge_with}，增益: {best_gain:.6f}")

        # 标记被合并的组为已删除
        groups[merge_with]['deleted'] = True

        # 更新当前组
        groups[i]['x'] = groups[i]['newX']
        groups[i]['y'] = groups[i]['newY']
        groups[i]['dist'] = groups[i]['newDist']

        # 更新增益，在左右两侧
        compute_gain(gains, groups, i, sents1, sents2, sim_mat, embeds1, embeds2, char_ratio, params)

        # 更新左右两侧的增益
        prev_i = prev(groups, i)
        if prev_i != -1:
            compute_gain(gains, groups, prev_i, sents1, sents2, sim_mat, embeds1, embeds2, char_ratio, params)

        next_i = next(groups, i)
        if next_i != -1:
            compute_gain(gains, groups, next_i, sents1, sents2, sim_mat, embeds1, embeds2, char_ratio, params)

        # 为下一次迭代计算最佳增益
        if len(gains) > 0:
            best_gain = gains.maxKey()
        else:
            best_gain = 0

        merge_count += 1

    print(f"*** 后期分组完成，共执行 {merge_count} 次合并")

    # 重建x_dtw和y_dtw
    x_dtw = []
    y_dtw = []
    for group in groups:
        if not group.get('deleted', False):
            x_dtw.append(group['x'])
            y_dtw.append(group['y'])

    print(f"*** 优化后分组数: {len(x_dtw)}")

    # 如果所有组都被删除了，恢复原始分组
    if len(x_dtw) == 0:
        print("⚠️ 所有组都被合并删除，恢复原始分组")
        x_dtw = []
        y_dtw = []
        for group in groups:
            if 'original_x' in group and 'original_y' in group:
                x_dtw.append(group['original_x'])
                y_dtw.append(group['original_y'])
            else:
                x_dtw.append(group['x'])
                y_dtw.append(group['y'])
        print(f"*** 恢复后分组数: {len(x_dtw)}")

    return (x_dtw, y_dtw)


def save_alignment_results(output_file_path: str, sentence_id: int, x_dtw: List[List[int]], y_dtw: List[List[int]],
                          sents1: List[str], sents2: List[str], score: float, params: Dict[str, Any]) -> None:
    """
    保存DTW对齐结果到JSON文件

    参数:
        output_file_path (str): 输出文件路径
        sentence_id (int): 句子ID
        x_dtw (List[List[int]]): x轴对齐路径
        y_dtw (List[List[int]]): y轴对齐路径
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表
        score (float): 对齐得分
        params (Dict): 参数字典
    """
    print(f"\n=== 保存对齐结果 ===")
    print(f"输出文件: {output_file_path}")
    print(f"输入数据检查:")
    print(f"  x_dtw: {x_dtw}")
    print(f"  y_dtw: {y_dtw}")
    print(f"  score: {score}")

    # 构建对齐结果数据
    alignment_pairs = []
    for i, (x_group, y_group) in enumerate(zip(x_dtw, y_dtw)):
        # 获取对应的句子文本
        english_texts = [sents1[x] for x in x_group if 0 <= x < len(sents1)]
        chinese_texts = [sents2[y] for y in y_group if 0 <= y < len(sents2)]

        alignment_pairs.append({
            'pair_id': i + 1,
            'english_indices': x_group,
            'chinese_indices': y_group,
            'english_texts': english_texts,
            'chinese_texts': chinese_texts,
            'alignment_type': f"{len(x_group)}-{len(y_group)}"
        })

        print(f"  对齐对 {i+1}: {x_group} -> {y_group} ({len(x_group)}-{len(y_group)})")

    # 构建完整的结果数据
    result_data = {
        'sentence_id': sentence_id,
        'dtw_alignment_result': {
            'total_pairs': len(alignment_pairs),
            'alignment_score': float(score),
            'alignment_pairs': alignment_pairs
        },
        'statistics': {
            'english_sentence_count': len(sents1),
            'chinese_sentence_count': len(sents2),
            'alignment_types': {}
        },
        'parameters': params,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # 统计对齐类型
    for pair in alignment_pairs:
        alignment_type = pair['alignment_type']
        if alignment_type not in result_data['statistics']['alignment_types']:
            result_data['statistics']['alignment_types'][alignment_type] = 0
        result_data['statistics']['alignment_types'][alignment_type] += 1

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 对齐结果已保存")
        print(f"  对齐对数: {len(alignment_pairs)}")
        print(f"  对齐得分: {score:.4f}")
        print(f"  对齐类型分布: {result_data['statistics']['alignment_types']}")

    except Exception as e:
        print(f"✗ 保存失败: {e}")
        raise


def print_alignment_summary(x_dtw: List[List[int]], y_dtw: List[List[int]],
                           sents1: List[str], sents2: List[str], score: float) -> None:
    """
    打印对齐结果摘要

    参数:
        x_dtw (List[List[int]]): x轴对齐路径
        y_dtw (List[List[int]]): y轴对齐路径
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表
        score (float): 对齐得分
    """
    print(f"\n=== 对齐结果摘要 ===")
    print(f"总对齐对数: {len(x_dtw)}")
    print(f"对齐得分: {score:.4f}")
    print(f"平均每对得分: {score/len(x_dtw):.4f}" if len(x_dtw) > 0 else "N/A")

    # 统计对齐类型
    alignment_types = {}
    for x_group, y_group in zip(x_dtw, y_dtw):
        alignment_type = f"{len(x_group)}-{len(y_group)}"
        alignment_types[alignment_type] = alignment_types.get(alignment_type, 0) + 1

    print(f"对齐类型分布:")
    for alignment_type, count in sorted(alignment_types.items()):
        print(f"  {alignment_type}: {count} 对")

    print(f"\n详细对齐结果:")
    for i, (x_group, y_group) in enumerate(zip(x_dtw, y_dtw)):
        english_texts = [sents1[x] for x in x_group if 0 <= x < len(sents1)]
        chinese_texts = [sents2[y] for y in y_group if 0 <= y < len(sents2)]

        print(f"  对齐对 {i+1} ({len(x_group)}-{len(y_group)}):")
        print(f"    英文[{','.join(map(str, x_group))}]: {' | '.join(english_texts)}")
        print(f"    中文[{','.join(map(str, y_group))}]: {' | '.join(chinese_texts)}")
        print()


def load_precomputed_embeddings(sents1: List[str], sents2: List[str],
                              sim_mat: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    使用Qwen3-Embedding-8B模型计算单句嵌入向量
    这样可以确保单句和组合句子使用相同的向量空间

    参数:
        sents1 (List[str]): 源语言句子列表
        sents2 (List[str]): 目标语言句子列表
        sim_mat (np.ndarray): 预计算的相似度矩阵（用于验证）

    返回值:
        Tuple[List[np.ndarray], List[np.ndarray]]: (源语言嵌入向量列表, 目标语言嵌入向量列表)
    """
    print("*** 使用Qwen3-Embedding-8B计算单句嵌入向量...")

    try:
        # 使用Qwen3-Embedding-8B计算所有单句的嵌入向量
        print(f"  正在计算 {len(sents1)} 个源语言句子的嵌入向量...")
        embeds1 = call_qwen3_embedding(sents1)

        print(f"  正在计算 {len(sents2)} 个目标语言句子的嵌入向量...")
        embeds2 = call_qwen3_embedding(sents2)

        # 验证向量维度
        if embeds1 and embeds2:
            dim1 = embeds1[0].shape[0] if len(embeds1[0].shape) > 0 else len(embeds1[0])
            dim2 = embeds2[0].shape[0] if len(embeds2[0].shape) > 0 else len(embeds2[0])
            print(f"✓ 嵌入向量计算完成")
            print(f"  源语言: {len(embeds1)} 个句子，维度: {dim1}")
            print(f"  目标语言: {len(embeds2)} 个句子，维度: {dim2}")

            # 验证相似度矩阵的一致性
            if len(embeds1) > 0 and len(embeds2) > 0:
                computed_sim = np.matmul(embeds1[0], np.transpose(embeds2[0]))
                original_sim = sim_mat[0, 0]
                print(f"  相似度验证: 计算值={computed_sim:.4f}, 原始值={original_sim:.4f}")

        return embeds1, embeds2

    except Exception as e:
        print(f"❌ Qwen3-Embedding-8B计算失败: {e}")
        raise e





def main():
    """
    主函数 - 执行完整的DTW对齐流程
    集成Qwen3-Embedding-8B模型进行高质量句子对齐
    """
    print("🚀 开始执行DTW对齐算法（集成Qwen3-Embedding-8B）...")

    # 直接使用设定的密钥，不检查环境变量
    print("✓ 使用预设的API密钥进行远程嵌入向量计算")

    # 针对双语子句数1-6优化的配置参数
    params = {
        'verbose': True,
        'veryVerbose': True,   # 启用详细日志查看DTW决策过程
        'distNull': 1.0,
        'penalty_n_n': 0.01,  # 多对多对齐惩罚（1-6句子中多对多常见，保持低惩罚）
        'penalty_0_n': 0.05,  # 空组合并奖励
        'noEmptyPair': False,  # 允许空对齐
        'no2_2Group': False,   # 允许2-2组合
        'dtwBeam': 7,          # 束搜索宽度（针对1-6句子优化：20→7，减少65%计算成本）
        'localDiagBeam': 0.35, # 锚点偏差阈值（针对1-6句子优化：允许约2.1格子偏差，更精确）
        'useEncoder': True,    # 启用Qwen3-Embedding-8B嵌入向量计算
        'noMarginPenalty': False,  # 启用邻近句惩罚（提高对齐质量）
        'lateGrouping': True   # 启用后期分组优化，提升对齐质量
    }

    print("📋 算法配置（针对双语子句数1-6优化）:")
    print(f"  - 嵌入向量模型: Qwen3-Embedding-8B")
    print(f"  - 束搜索宽度: {params['dtwBeam']} (优化：20→7，减少65%计算成本)")
    print(f"  - 锚点偏差阈值: {params['localDiagBeam']} (优化：0.5→0.35，允许2.1格子偏差，更精确)")
    print(f"  - 邻近句惩罚: {'启用' if not params['noMarginPenalty'] else '禁用'}")
    print(f"  - 后期分组优化: {'启用' if params['lateGrouping'] else '禁用'}")
    print(f"  - 数据规模: 完美适配1-6句子双语数据，最大搜索空间36格子")

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
            "interval_file_path": f"{data_prefix}complete_interval_extraction-output.json",
            "matrix_file_path": f"{data_prefix}Qwen3-Embedding-8B-output.json",
            "sentence_file_path": f"{data_prefix}sentence.json",
            "output_file_path": f"{data_prefix}dtw_alignment-B-output.json"
        }

    # 获取路径配置
    paths = get_data_paths()
    sentence_id = 0
    interval_file_path = paths["interval_file_path"]
    matrix_file_path = paths["matrix_file_path"]
    sentence_file_path = paths["sentence_file_path"]
    output_file_path = paths["output_file_path"]

    try:
        # 步骤1: 加载所有必要数据
        print(f"\n📁 步骤1: 数据加载阶段")
        interval_data = load_interval_data(interval_file_path, sentence_id)
        sim_mat = load_similarity_matrix(matrix_file_path, sentence_id)
        sents1, sents2 = load_sentence_data(sentence_file_path, sentence_id)

        # 构建单句嵌入向量（用于邻近句惩罚计算和单句情况）
        print("🔧 构建单句嵌入向量...")
        embeds1, embeds2 = load_precomputed_embeddings(sents1, sents2, sim_mat)

        # 计算字符比例
        nb_chars1 = sum(len(sent) for sent in sents1)
        nb_chars2 = sum(len(sent) for sent in sents2)
        char_ratio = nb_chars2 / nb_chars1 if nb_chars1 > 0 else 1.0

        print(f"📊 文本统计:")
        print(f"  - 源语言字符数: {nb_chars1}")
        print(f"  - 目标语言字符数: {nb_chars2}")
        print(f"  - 字符比例: {char_ratio:.3f}")

        # 步骤2: 执行DTW算法
        print(f"\n🔄 步骤2: DTW算法执行")

        # 生成允许的句子组合并添加到参数中
        allowed_groups = generate_allowed_groups(params, sents1, sents2)
        params['allowedGroups'] = allowed_groups

        x_dtw, y_dtw, score = run_dtw(interval_data, sim_mat, sents1, sents2, embeds1, embeds2, params)

        if len(x_dtw) == 0:
            print(f"❌ DTW算法未产生有效对齐结果")
            return

        # 步骤3: 后期分组优化（使用Qwen3-Embedding-8B）
        if params.get('lateGrouping', False):
            print(f"\n🔧 步骤3: 后期分组优化（使用Qwen3-Embedding-8B）")
            x_dtw, y_dtw = late_grouping(x_dtw, y_dtw, sents1, sents2, sim_mat, embeds1, embeds2, char_ratio, params)
            print(f"✓ 后期分组优化完成")

        # 步骤4: 输出结果
        print(f"\n📊 步骤4: 结果输出")
        print_alignment_summary(x_dtw, y_dtw, sents1, sents2, score)
        save_alignment_results(output_file_path, sentence_id, x_dtw, y_dtw, sents1, sents2, score, params)

        print(f"\n🎉 DTW对齐算法执行完成！")
        print(f"📁 结果已保存到: {output_file_path}")
        print(f"🔍 使用Qwen3-Embedding-8B模型进行了高质量的句子向量化")

        # 显示API调用统计
        print_qwen3_stats()


    except Exception as e:
        print(f"\n❌ 执行过程中发生错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {str(e)}")
        raise


if __name__ == "__main__":
    main()
