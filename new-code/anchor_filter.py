# -*- coding:utf8 -*-
"""
锚点密度过滤模块 - 基于相似度矩阵的智能锚点提取（无预定义锚点）

本模块实现了跨语言句子对齐系统的第一阶段：锚点密度过滤
主要功能：
1. 从相似度矩阵中智能提取候选锚点（完全自动化，无需预定义锚点）
2. 计算每个锚点的局部密度和质量评分
3. 过滤低质量锚点，保留高质量锚点
4. 解决锚点冲突，确保锚点分布合理
5. 输出过滤后的锚点数据供后续区间划分使用

核心算法流程：
1. 加载句子数据和相似度矩阵
2. 基于阈值自动提取候选锚点（纯数据驱动）
3. 计算局部密度和质量评分
4. 执行质量过滤和冲突解决
5. 保存过滤后的锚点数据

技术特点：
- 完全去除预定义锚点依赖
- 纯基于相似度矩阵的自动锚点提取
- 数据驱动的质量评估机制

作者：AI Assistant
日期：2025-01-29
"""

import os
import sys
import json
import math
import time
import numpy as np




########################################################################### 虚拟边界处理安全函数

def safe_character_count(sentences, start_idx, end_idx):
    """
    安全的字符计数函数，避免越界访问句子列表
    
    参数：
    sentences (list[str]): 句子列表
    start_idx (int): 起始索引
    end_idx (int): 结束索引（包含）
    
    返回值：
    int: 字符总数
    """
    char_count = 0
    safe_start = max(0, start_idx)
    safe_end = min(end_idx, len(sentences) - 1)
    
    # 确保范围有效
    if safe_start <= safe_end and safe_start < len(sentences):
        for n in range(safe_start, safe_end + 1):
            if 0 <= n < len(sentences):  # 双重检查
                char_count += len(sentences[n])
    
    return char_count


def is_valid_anchor(x, y, len_sents1, len_sents2):
    """
    检查锚点是否为有效的真实锚点（非虚拟边界）
    
    参数：
    x (int): 英文句子索引
    y (int): 中文句子索引  
    len_sents1 (int): 英文句子总数
    len_sents2 (int): 中文句子总数
    
    返回值：
    bool: True如果是有效锚点，False如果是虚拟边界
    """
    # 检查下界虚拟边界
    if x < 0 or y < 0:
        return False
    # 检查上界虚拟边界  
    if x >= len_sents1 or y >= len_sents2:
        return False
    # 检查特殊虚拟边界
    if (x == -1 and y == -1):
        return False
    return True


def safe_diagonal_range(j, i, X, coeff, delta_y, max_j):
    """
    安全的对角线Y范围计算，避免极端负数索引
    
    参数：
    j (int): 当前点的y坐标
    i (int): 当前点的x坐标
    X (int): 扫描的x坐标
    coeff (float): 句子长度比例系数
    delta_y (int): y方向窗口半径
    max_j (int): y坐标的最大索引
    
    返回值：
    range: 安全的y坐标范围
    """
    diagonal_offset = (i - X) * coeff
    y_center = j - diagonal_offset
    
    # 防止极端负数或过大的数值
    if y_center < -1000 or y_center > max_j + 1000:
        return range(0, 0)  # 返回空范围
    
    y_range_start = int(max(0, y_center - delta_y))
    y_range_end = int(min(y_center + delta_y + 1, max_j + 1))
    
    # 确保范围有效
    if y_range_start >= y_range_end:
        return range(0, 0)
        
    return range(y_range_start, y_range_end)


########################################################################### 点过滤函数

def compute_local_density(params, i, j, points, max_i, max_j, sim_mat):
    """
    计算沿对角线的局部密度 - 这是锚点质量评估的核心算法
    
    算法原理：
    - 在当前点(i,j)周围定义一个局部窗口
    - 计算三种不同位置的局部空间密度：前向、后向、居中
    - 返回三种密度中的最大值作为该点的局部密度
    - 高密度区域的点更可能是真正的对齐锚点
    
    参数：
    params (dict): 全局参数字典，包含deltaX、deltaY等窗口大小参数
    i (int): 当前点的x坐标（英文句子索引）
    j (int): 当前点的y坐标（中文句子索引）
    points (dict): 所有候选锚点的字典 {(x,y): 1}
    max_i (int): 英文句子的最大索引
    max_j (int): 中文句子的最大索引
    sim_mat (numpy.ndarray): 相似度矩阵
    
    返回值：
    float: 局部锚点密度值（0-1之间，越高表示该点周围锚点越密集）
    """
    
    # === 自适应窗口大小计算 ===
    len_sents1 = max_i + 1
    len_sents2 = max_j + 1
    
    adaptive_delta_x = max(1, len_sents1 // 2)
    adaptive_delta_y = max(1, len_sents2 // 2)
    
    if 'deltaX' in params and params['deltaX'] is not None and params['deltaX'] > 0:
        delta_x = params['deltaX']
        use_adaptive_x = False
    else:
        delta_x = adaptive_delta_x
        use_adaptive_x = True
        
    if 'deltaY' in params and params['deltaY'] is not None and params['deltaY'] > 0:
        delta_y = params['deltaY'] 
        use_adaptive_y = False
    else:
        delta_y = adaptive_delta_y
        use_adaptive_y = True
    
    if params.get('veryVerbose', False):
        print(f"  === 窗口大小计算 ===")
        print(f"  句子数量: 英文{len_sents1}句, 中文{len_sents2}句")
        print(f"  自适应计算: deltaX={adaptive_delta_x} (英文{len_sents1}÷2), deltaY={adaptive_delta_y} (中文{len_sents2}÷2)")
        print(f"  实际使用: deltaX={delta_x}{'(自适应)' if use_adaptive_x else '(用户设置)'}, deltaY={delta_y}{'(自适应)' if use_adaptive_y else '(用户设置)'}")
        print(f"  计算点({i},{j})的局部密度...")
        
    coeff = len_sents2 / len_sents1 if params['sentRatio'] == 0 else params['sentRatio']
    
    if params.get('veryVerbose', False):
        print(f"  句子长度比例系数: {coeff:.3f} (中文{len_sents2}句 / 英文{len_sents1}句)")
    
    local_space_size_before = 0
    nb_points_in_local_space_size_before = 0
    local_space_size_centered = 0
    nb_points_in_local_space_size_centered = 0
    local_space_size_after = 0
    nb_points_in_local_space_size_after = 0

    x_range_start = max(0, i - 2 * delta_x)
    x_range_end = min(i + 2 * delta_x + 1, max_i + 1)
    
    if params.get('veryVerbose', False):
        print(f"  扫描X范围: [{x_range_start}, {x_range_end}) (窗口半径2×{delta_x}={2*delta_x})")
    
    scanned_points = 0
    
    for X in range(x_range_start, x_range_end):
        y_range = safe_diagonal_range(j, i, X, coeff, delta_y, max_j)
        
        for Y in y_range:
            scanned_points += 1
            
            if X <= i:
                local_space_size_before += 1
                if (X, Y) in points.keys():
                    # 使用锚点的相似度权重，如果是新格式则从anchor_info中获取
                    anchor_value = points[(X, Y)]
                    if isinstance(anchor_value, dict):
                        weight = anchor_value.get('similarity', sim_mat[X, Y])
                    else:
                        weight = sim_mat[X, Y]  # 兼容旧格式
                    nb_points_in_local_space_size_before += weight
                    
            if X >= i:
                local_space_size_after += 1
                if (X, Y) in points.keys():
                    # 使用锚点的相似度权重
                    anchor_value = points[(X, Y)]
                    if isinstance(anchor_value, dict):
                        weight = anchor_value.get('similarity', sim_mat[X, Y])
                    else:
                        weight = sim_mat[X, Y]  # 兼容旧格式
                    nb_points_in_local_space_size_after += weight
                    
            if max(0, i - delta_x) <= X < min(i + delta_x + 1, max_i + 1):
                local_space_size_centered += 1
                if (X, Y) in points.keys():
                    # 使用锚点的相似度权重
                    anchor_value = points[(X, Y)]
                    if isinstance(anchor_value, dict):
                        weight = anchor_value.get('similarity', sim_mat[X, Y])
                    else:
                        weight = sim_mat[X, Y]  # 兼容旧格式
                    nb_points_in_local_space_size_centered += weight

    if params.get('veryVerbose', False):
        print(f"  总共扫描了 {scanned_points} 个网格点")
        print(f"  空间大小统计: 前向={local_space_size_before}, 居中={local_space_size_centered}, 后向={local_space_size_after}")
        print(f"  相似度权重总和: 前向={nb_points_in_local_space_size_before:.3f}, 居中={nb_points_in_local_space_size_centered:.3f}, 后向={nb_points_in_local_space_size_after:.3f}")

    densityBefore = 0
    densityAfter = 0   
    densityCentered = 0
    
    if local_space_size_before > 0:
        densityBefore = nb_points_in_local_space_size_before / local_space_size_before
        
    if local_space_size_after > 0:
        densityAfter = nb_points_in_local_space_size_after / local_space_size_after
        
    if local_space_size_centered > 0:
        densityCentered = nb_points_in_local_space_size_centered / local_space_size_centered

    final_density = max(densityBefore, densityAfter, densityCentered)
    
    if params.get('veryVerbose', False):
        print(f"  密度计算结果: 前向={densityBefore:.4f}, 居中={densityCentered:.4f}, 后向={densityAfter:.4f}")
        print(f"  最终密度: {final_density:.4f}")
        print()
    
    return final_density


def filter_points(params, points, max_i, max_j, average_density, sim_mat):
    """
    锚点质量过滤器 - 基于局部密度分析移除低质量候选锚点
    """
    
    print(f"\n=== 开始锚点质量过滤 ===")
    print(f"初始候选锚点数量: {len(points)}")
    print(f"全局平均密度基准: {average_density:.4f}")
    print(f"密度比率阈值: {params.get('minDensityRatio', 'N/A')}")
    
    filtered_x = []
    filtered_y = []
    nbDeleted = 0
    nbPreserved = 0

    if params['veryVerbose']:
        print(f"开始逐一评估 {len(points)} 个候选锚点的质量...")

    points_key = sorted(list(points.keys()), key=lambda point: point[0])
    print(f"候选锚点列表（按x坐标排序）: {len(points_key)} 个点")

    for idx, point in enumerate(points_key):
        (i, j) = point

        localDensity = compute_local_density(params, i, j, points, max_i, max_j, sim_mat)
        
        density_ratio = localDensity / average_density if average_density > 0 else 0
        
        # 获取当前锚点信息
        anchor_info = points[(i, j)]
        anchor_id = anchor_info.get('id', 'N/A') if isinstance(anchor_info, dict) else 'legacy'
        similarity = anchor_info.get('similarity', sim_mat[i, j]) if isinstance(anchor_info, dict) else sim_mat[i, j]
        
        # 计算综合质量评分（相似度权重0.4 + 密度比率权重0.6）
        quality_score = 0.4 * similarity + 0.6 * min(density_ratio, 1.0)
        
        if params['veryVerbose']:
            print(f"  锚点#{anchor_id} {idx+1}/{len(points_key)}: ({i},{j})")
            print(f"    相似度: {similarity:.4f}")
            print(f"    局部密度: {localDensity:.4f}")
            print(f"    密度比率: {density_ratio:.4f} (阈值: {params['minDensityRatio']})")
            print(f"    质量评分: {quality_score:.4f}")

        should_keep = False
        reason = ""
        
        if average_density <= 0:
            should_keep = True
            reason = "全局密度异常，保守保留"
        elif density_ratio >= params['minDensityRatio']:
            should_keep = True
            reason = f"密度合格 ({density_ratio:.4f} >= {params['minDensityRatio']})"
        else:
            should_keep = False
            reason = f"密度过低 ({density_ratio:.4f} < {params['minDensityRatio']})"

        if should_keep:
            # 更新锚点信息，添加质量评估结果
            if isinstance(anchor_info, dict):
                anchor_info.update({
                    'local_density': float(localDensity),
                    'density_ratio': float(density_ratio),
                    'quality_score': float(quality_score),
                    'filter_status': 'passed',
                    'filter_reason': reason
                })
                points[(i, j)] = anchor_info
            
            filtered_x.append(i)
            filtered_y.append(j)
            nbPreserved += 1
            if params['veryVerbose']:
                print(f"    → 保留: {reason}")
        else:
            del points[(i, j)]
            nbDeleted += 1
            if params['veryVerbose']:
                print(f"    → 移除: {reason}")
        
        if params['veryVerbose']:
            print()

    print(f"=== 锚点过滤完成 ===")
    print(f"初始锚点: {len(points_key)} 个")
    print(f"保留锚点: {nbPreserved} 个")
    print(f"移除锚点: {nbDeleted} 个")
    print(f"保留率: {(nbPreserved/(nbPreserved+nbDeleted)*100):.1f}%")
    print()

    return (points, filtered_x, filtered_y)


def resolving_conflicts(params, points, max_i, max_j, sim_mat):
    """
    移除在同一坐标轴上冲突的锚点：优先保留质量评分更高的锚点
    """    
    print(f"\n=== 开始解决锚点冲突 ===")
    
    x2y = {}
    y2x = {}
    filtered_x = []
    filtered_y = []
    nbDeleted = 0
    conflict_details = []
    
    points_key = list(points.keys())
    
    print(f"检查 {len(points_key)} 个锚点的坐标冲突...")
    
    for point in points_key:
        (i, j) = point
        current_anchor = points[point]
        
        # X坐标冲突检查
        if i in x2y.keys():
            if x2y[i] != j:
                existing_j = x2y[i]
                existing_point = (i, existing_j)
                
                if existing_point in points:
                    existing_anchor = points[existing_point]
                    
                    # 比较锚点质量
                    current_quality = current_anchor.get('quality_score', 0) if isinstance(current_anchor, dict) else 0
                    existing_quality = existing_anchor.get('quality_score', 0) if isinstance(existing_anchor, dict) else 0
                    
                    current_id = current_anchor.get('id', 'N/A') if isinstance(current_anchor, dict) else 'legacy'
                    existing_id = existing_anchor.get('id', 'N/A') if isinstance(existing_anchor, dict) else 'legacy'
                    
                    nbDeleted += 1
                    
                    if current_quality > existing_quality:
                        del points[existing_point]
                        x2y[i] = j
                        conflict_details.append(f"X冲突: 保留锚点#{current_id}({i},{j})[质量:{current_quality:.3f}], 移除锚点#{existing_id}({i},{existing_j})[质量:{existing_quality:.3f}]")
                    else:
                        del points[point]
                        conflict_details.append(f"X冲突: 保留锚点#{existing_id}({i},{existing_j})[质量:{existing_quality:.3f}], 移除锚点#{current_id}({i},{j})[质量:{current_quality:.3f}]")
                        continue
        else:
            x2y[i] = j

        # Y坐标冲突检查
        if j in y2x.keys():
            if y2x[j] != i:
                existing_i = y2x[j]
                existing_point = (existing_i, j)
                
                if existing_point in points:
                    existing_anchor = points[existing_point]
                    
                    # 比较锚点质量
                    current_quality = current_anchor.get('quality_score', 0) if isinstance(current_anchor, dict) else 0
                    existing_quality = existing_anchor.get('quality_score', 0) if isinstance(existing_anchor, dict) else 0
                    
                    current_id = current_anchor.get('id', 'N/A') if isinstance(current_anchor, dict) else 'legacy'
                    existing_id = existing_anchor.get('id', 'N/A') if isinstance(existing_anchor, dict) else 'legacy'
                    
                    nbDeleted += 1
                    
                    if current_quality > existing_quality:
                        del points[existing_point]
                        y2x[j] = i
                        conflict_details.append(f"Y冲突: 保留锚点#{current_id}({i},{j})[质量:{current_quality:.3f}], 移除锚点#{existing_id}({existing_i},{j})[质量:{existing_quality:.3f}]")
                    else:
                        del points[point]
                        conflict_details.append(f"Y冲突: 保留锚点#{existing_id}({existing_i},{j})[质量:{existing_quality:.3f}], 移除锚点#{current_id}({i},{j})[质量:{current_quality:.3f}]")
        else:
            y2x[j] = i

    if params.get('verbose', False) or params.get('veryVerbose', False):
        print(f"✅ 冲突解决完成:")
        print(f"  移除冲突锚点: {nbDeleted} 个")
        if conflict_details and params.get('veryVerbose', False):
            print(f"  冲突解决详情:")
            for detail in conflict_details:
                print(f"    {detail}")

    # 生成最终的锚点列表
    points_key = list(points.keys())
    for point in points_key:
        (i, j) = point
        filtered_x.append(i)
        filtered_y.append(j)
        
    print(f"  最终保留锚点: {len(filtered_x)} 个")
    print()
    
    return (points, filtered_x, filtered_y)


########################################################################### 相似度矩阵读取函数

def load_sentences_data(sentences_file_path, sentence_id):
    """
    从JSON文件中读取指定sentence_id的英文和中文句子数据
    """
    
    print(f"\n=== 开始加载句子数据 ===")
    print(f"文件路径: {sentences_file_path}")
    print(f"目标句子ID: {sentence_id}")
    
    try:
        print("正在读取JSON文件...")
        with open(sentences_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件读取成功，包含 {len(data)} 个句子对")
        
        print(f"正在搜索sentence_id={sentence_id}的数据...")
        for idx, item in enumerate(data):
            current_id = item.get('sentence_id')
            print(f"  检查第 {idx+1} 项: sentence_id = {current_id}")
            
            if current_id == sentence_id:
                english_sentences = item.get('english_sentence_text', [])
                chinese_sentences = item.get('chinese_sentence_text', [])
                
                print(f"✓ 找到目标数据!")
                print(f"  英文句子数量: {len(english_sentences)}")
                print(f"  中文句子数量: {len(chinese_sentences)}")
                
                if english_sentences:
                    print(f"  英文句子详情:")
                    for i, sent in enumerate(english_sentences):
                        print(f"    [{i}] {sent}")
                        
                if chinese_sentences:
                    print(f"  中文句子详情:")
                    for i, sent in enumerate(chinese_sentences):
                        print(f"    [{i}] {sent}")
                
                print("句子数据加载完成\n")
                return english_sentences, chinese_sentences
        
        available_ids = [item.get('sentence_id', 'N/A') for item in data]
        error_msg = f"在文件中未找到sentence_id为{sentence_id}的数据。可用的sentence_id: {available_ids}"
        print(f"✗ 错误: {error_msg}")
        raise ValueError(error_msg)
        
    except FileNotFoundError:
        error_msg = f"句子数据文件未找到: {sentences_file_path}"
        print(f"✗ 文件错误: {error_msg}")
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"JSON文件格式错误: {sentences_file_path}, 错误详情: {str(e)}"
        print(f"✗ JSON解析错误: {error_msg}")
        raise ValueError(error_msg)


def load_similarity_matrix(matrix_file_path, sentence_id):
    """
    从JSON文件中读取指定sentence_id的相似度矩阵
    """
    
    print(f"\n=== 开始加载相似度矩阵 ===")
    print(f"矩阵文件路径: {matrix_file_path}")
    print(f"目标句子ID: {sentence_id}")
    
    try:
        print("正在读取相似度矩阵文件...")
        with open(matrix_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件读取成功，包含 {len(data)} 个矩阵数据")
        
        print(f"正在搜索sentence_id={sentence_id}的相似度矩阵...")
        for idx, item in enumerate(data):
            current_id = item.get('sentence_id')
            print(f"  检查第 {idx+1} 项: sentence_id = {current_id}")
            
            if current_id == sentence_id:
                print(f"✓ 找到目标相似度矩阵!")
                
                matrix_data = item.get('semantic_similarity_matrix')
                if matrix_data is None:
                    raise ValueError(f"sentence_id={sentence_id}的数据中缺少'semantic_similarity_matrix'字段")
                
                print("正在转换为numpy数组...")
                matrix = np.array(matrix_data, dtype=np.float64)
                
                if len(matrix.shape) != 2:
                    raise ValueError(f"相似度矩阵必须是二维的，当前维度: {matrix.shape}")
                
                rows, cols = matrix.shape
                print(f"  矩阵维度: {rows} x {cols} (英文句子 x 中文句子)")
                print(f"  数据类型: {matrix.dtype}")
                
                print(f"  完整相似度矩阵 ({rows}x{cols}):")
                for i in range(rows):
                    row_values = [f"{matrix[i,j]:.4f}" for j in range(cols)]
                    print(f"    [{i}] {' '.join(row_values)}")
                
                max_pos = np.unravel_index(np.argmax(matrix), matrix.shape)
                print(f"  最高相似度位置: ({max_pos[0]}, {max_pos[1]}) = {matrix[max_pos]:.4f}")
                
                print("相似度矩阵加载完成\n")
                return matrix
        
        available_ids = [item.get('sentence_id', 'N/A') for item in data]
        error_msg = f"在文件中未找到sentence_id为{sentence_id}的数据。可用的sentence_id: {available_ids}"
        print(f"✗ 错误: {error_msg}")
        raise ValueError(error_msg)
        
    except FileNotFoundError:
        error_msg = f"相似度矩阵文件未找到: {matrix_file_path}"
        print(f"✗ 文件错误: {error_msg}")
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"JSON文件格式错误: {matrix_file_path}, 错误详情: {str(e)}"
        print(f"✗ JSON解析错误: {error_msg}")
        raise ValueError(error_msg)


def extract_candidate_points_from_matrix(params, sim_mat):
    """
    基于相似度矩阵的智能候选锚点提取器 - 增强版本支持锚点编号和详细信息
    """
    
    print(f"\n=== 开始提取候选锚点 ===")
    
    len_sents1, len_sents2 = sim_mat.shape
    print(f"相似度矩阵维度: {len_sents1} × {len_sents2} (英文 × 中文)")
    
    points = {}
    x = []
    y = []
    
    threshold = params.get('cosThreshold', 0.5)
    margin = params.get('margin', 0.1)
    
    print(f"算法参数:")
    print(f"  相似度阈值: {threshold}")
    print(f"  边距阈值: {margin}")
    
    max_dimension = max(len_sents1, len_sents2)
    k_best = math.ceil(max_dimension / 2)
    
    print(f"  最大的文档长度或句子数: {max_dimension}")
    print(f"  自适应k值: {k_best} (每行最多选择{k_best}个锚点)")
    print()
    
    anchor_id = 1  # 锚点编号从1开始自动递增
    total_candidates = 0
    threshold_filtered = 0
    margin_filtered = 0
    final_selected = 0
    
    print("开始逐行扫描相似度矩阵...")
    
    for i in range(len_sents1):
        row_similarities = sim_mat[i, :]
        
        if params.get('veryVerbose', False):
            print(f"\n处理英文句子 {i}:")
            print(f"  相似度分布: min={np.min(row_similarities):.4f}, max={np.max(row_similarities):.4f}, mean={np.mean(row_similarities):.4f}")
        
        candidates = []
        for j in range(len_sents2):
            if row_similarities[j] > threshold:
                candidates.append((row_similarities[j], j))
            else:
                threshold_filtered += 1
        
        total_candidates += len(candidates)
        
        if not candidates:
            if params.get('veryVerbose', False):
                print(f"  → 无候选点（所有相似度均低于阈值{threshold}）")
            continue
        
        if params.get('veryVerbose', False):
            print(f"  通过阈值过滤的候选点: {len(candidates)} 个")
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        if params.get('veryVerbose', False):
            print(f"  最佳候选详情: {[(f'{score:.4f}', idx) for score, idx in candidates]}")
        
        if len(candidates) > 1:
            best_score = candidates[0][0]
            second_score = candidates[1][0]
            score_gap = best_score - second_score
            
            if score_gap < margin:
                if params.get('veryVerbose', False):
                    print(f"  → 边距过滤: 最佳分数差距 {score_gap:.4f} < {margin} → 去除该行所有候选点")
                margin_filtered += len(candidates)
                continue
            else:
                if params.get('veryVerbose', False):
                    print(f"  → 边距过滤: 最佳分数差距 {score_gap:.4f} >= {margin} → 保留候选点")
        
        selected_candidates = candidates[:min(k_best, len(candidates))]
        
        if params.get('veryVerbose', False):
            print(f"  最终选择: {len(selected_candidates)} 个锚点")
        
        for sim_score, j in selected_candidates:
            # 创建增强的锚点信息
            anchor_info = {
                'id': anchor_id,
                'type': 'auto_detected',
                'similarity': float(sim_score),
                'source': 'threshold_filter',
                'rank_in_row': len([c for c in candidates if c[0] > sim_score]) + 1,
                'total_candidates_in_row': len(candidates)
            }
            points[(i, j)] = anchor_info
            x.append(i)
            y.append(j)
            final_selected += 1
            anchor_id += 1
            
            if params.get('veryVerbose', False):
                print(f"    锚点#{anchor_info['id']} ({i},{j}): 相似度={sim_score:.4f}, 行内排名={anchor_info['rank_in_row']}/{anchor_info['total_candidates_in_row']}")
    
    print(f"\n=== 候选锚点提取完成 ===")
    print(f"扫描统计:")
    print(f"  总扫描点数: {len_sents1 * len_sents2}")
    print(f"  阈值过滤掉: {threshold_filtered} 个")
    print(f"  边距过滤掉: {margin_filtered} 个")
    print(f"  最终选中: {final_selected} 个候选锚点（编号1-{anchor_id-1}）")
    
    if final_selected > 0 and params.get('verbose', False):
        print(f"完整锚点分布详情:")
        for idx in range(len(x)):
            anchor_info = points[(x[idx], y[idx])]
            print(f"  锚点#{anchor_info['id']} ({x[idx]:2d}, {y[idx]:2d}): 相似度={anchor_info['similarity']:.4f}, 类型={anchor_info['type']}")
    
    print()
    return points, x, y


def save_filtered_data_to_json(output_file_path, sentence_id, filtered_x, filtered_y, anchor_points, 
                              english_sentences, chinese_sentences, len_sents1, len_sents2, 
                              sim_mat, params, average_density):
    """
    将第一部分过滤后的数据保存到JSON文件，供第二部分使用
    
    参数：
    output_file_path (str): 输出JSON文件路径
    sentence_id (int): 句子ID
    filtered_x (list): 过滤后的x坐标列表
    filtered_y (list): 过滤后的y坐标列表
    anchor_points (dict): 锚点字典
    english_sentences (list): 英文句子列表
    chinese_sentences (list): 中文句子列表
    len_sents1 (int): 英文句子总数
    len_sents2 (int): 中文句子总数
    sim_mat (numpy.ndarray): 相似度矩阵
    params (dict): 参数字典
    average_density (float): 平均密度
    """
    
    print(f"\n=== 保存过滤后数据到JSON文件 ===")
    print(f"输出文件: {output_file_path}")
    
    # 重新分配连续的锚点编号（从1开始）
    print(f"重新分配锚点编号，确保连续性...")
    final_anchor_list = []
    
    for idx, (x, y) in enumerate(zip(filtered_x, filtered_y)):
        anchor_info = anchor_points.get((x, y), {})
        anchor_id = idx + 1  # 连续编号从1开始
        
        if isinstance(anchor_info, dict):
            # 从anchor_info中移除id，将其提升到顶级
            updated_anchor_info = dict(anchor_info)
            updated_anchor_info.pop('id', None)  # 移除内部的id
            updated_anchor_info['final_rank'] = idx + 1  # 添加最终排名
        else:
            # 兼容旧格式
            updated_anchor_info = {
                'type': 'legacy',
                'similarity': float(sim_mat[x, y]),
                'final_rank': idx + 1
            }
        
        # 将编号提升到锚点对象的顶级
        final_anchor_list.append({
            'id': anchor_id,
            'coordinates': [int(x), int(y)],
            'anchor_info': updated_anchor_info
        })
        
        print(f"  锚点#{anchor_id}: ({x},{y}) 相似度={updated_anchor_info.get('similarity', 'N/A'):.4f}")
    
    # 构建优化的数据结构
    data = {
        "sentence_id": sentence_id,
        "filtered_anchors": {
            "count": len(filtered_x),
            "anchors": final_anchor_list
        },
        "sentence_data": {
            "english_sentences": english_sentences,
            "chinese_sentences": chinese_sentences,
            "len_sents1": len_sents1,
            "len_sents2": len_sents2
        },
        "similarity_matrix": sim_mat.tolist(),
        "parameters": params,
        "statistics": {
            "average_density": average_density,
            "matrix_shape": list(sim_mat.shape),
            "final_anchor_count": len(filtered_x)
        }
    }
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 数据已成功保存到: {output_file_path}")
        print(f"  过滤后锚点数量: {len(filtered_x)}")
        print(f"  锚点字典大小: {len(anchor_points)}")
        print(f"  英文句子数: {len_sents1}")
        print(f"  中文句子数: {len_sents2}")
        print(f"  相似度矩阵维度: {sim_mat.shape}")
        print()
        
    except Exception as e:
        error_msg = f"保存JSON文件时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)


def run_anchor_filtering_phase(params, sentences_file_path, matrix_file_path, sentence_id, 
                              output_json_path):
    """
    执行锚点过滤阶段（第一部分）的主函数
    
    参数：
    params (dict): 参数字典
    sentences_file_path (str): 句子文件路径
    matrix_file_path (str): 相似度矩阵文件路径
    sentence_id (int): 句子ID
    output_json_path (str): 输出JSON文件路径
    
    返回值：
    tuple: (filtered_x, filtered_y, len_sents1, len_sents2)
    """
    
    print(f"\n" + "="*60)
    print(f"锚点过滤阶段（第一部分）")
    print(f"="*60)
    
    try:
        # 阶段1：数据加载
        print(f"📁 阶段1：数据加载")
        english_sentences, chinese_sentences = load_sentences_data(sentences_file_path, sentence_id)
        sim_mat = load_similarity_matrix(matrix_file_path, sentence_id)
        
        len_sents1, len_sents2 = sim_mat.shape
        
        if len_sents1 != len(english_sentences):
            raise ValueError(f"英文数据不匹配：矩阵行数{len_sents1} ≠ 句子数{len(english_sentences)}")
        if len_sents2 != len(chinese_sentences):
            raise ValueError(f"中文数据不匹配：矩阵列数{len_sents2} ≠ 句子数{len(chinese_sentences)}")
        
        # 阶段2：候选锚点提取和编号
        print(f"\n🎯 阶段2：智能候选锚点提取")
        points, x, y = extract_candidate_points_from_matrix(params, sim_mat)
        
        if len(points) == 0:
            print(f"⚠️  警告：未找到任何候选锚点")
            return [], [], len_sents1, len_sents2
        
        # 阶段3：锚点质量过滤
        print(f"\n🔍 阶段3：锚点质量过滤")
        
        # 创建锚点工作副本
        anchor_points = dict.copy(points)
        points_key = list(anchor_points.keys())
        
        # 计算全局平均密度
        print(f"开始计算 {len(points_key)} 个候选锚点的局部密度...")
        
        tot_density = 0
        density_list = []
        
        for idx, point in enumerate(points_key):
            (x2, y2) = point
            local_density = compute_local_density(params, x2, y2, anchor_points, len_sents1 - 1, len_sents2 - 1, sim_mat)
            tot_density += local_density
            density_list.append(local_density)
            
            if params.get('veryVerbose', False):
                print(f"  点 {idx+1}/{len(points_key)}: ({x2},{y2}) 密度={local_density:.4f}")

        average_density = tot_density / float(len(points_key))
        
        max_density = max(density_list)
        min_density = min(density_list)
        density_std = np.std(density_list)
        
        print(f"✅ 全局密度计算完成:")
        print(f"  平均密度: {average_density:.4f}")
        print(f"  密度范围: [{min_density:.4f}, {max_density:.4f}]")
        print(f"  密度标准差: {density_std:.4f}")

        # 第一轮过滤：移除低密度点
        print("第一轮过滤：移除局部密度过低的锚点...")
        (anchor_points, filtered_x, filtered_y) = filter_points(params, anchor_points, 
                                                                len_sents1 - 1, len_sents2 - 1, average_density, sim_mat)
        
        # 解决冲突点
        print("冲突解决：处理同一坐标的多个候选锚点...")
        (anchor_points, filtered_x, filtered_y) = resolving_conflicts(params, anchor_points, 
                                                                      len_sents1 - 1, len_sents2 - 1, sim_mat)

        # 可选的第二轮严格过滤
        if params.get('reiterateFiltering', False):
            print("第二轮过滤：应用更严格的密度标准...")
            strict_threshold = average_density * 2
            print(f"使用严格阈值: {strict_threshold:.4f} (2倍平均密度)")
            (anchor_points, filtered_x, filtered_y) = filter_points(params, anchor_points, 
                                                                    len_sents1 - 1, len_sents2 - 1,
                                                                    strict_threshold, sim_mat)

        print(f"✅ 锚点质量过滤完成:")
        print(f"  最终保留锚点: {len(filtered_x)} 个")
        
        # 显示最终锚点列表详情
        if len(filtered_x) > 0:
            print(f"  📍 最终锚点详情列表:")
            for idx, (x, y) in enumerate(zip(filtered_x, filtered_y)):
                anchor_info = anchor_points.get((x, y), {})
                anchor_id = anchor_info.get('id', 'N/A')
                similarity = anchor_info.get('similarity', sim_mat[x, y])
                quality_score = anchor_info.get('quality_score', 'N/A')
                anchor_type = anchor_info.get('type', 'unknown')
                print(f"    锚点{idx+1}: #{anchor_id} ({x},{y}) 相似度={similarity:.4f} 质量={quality_score:.4f} 类型={anchor_type}")
        
        # 保存数据到JSON文件
        save_filtered_data_to_json(output_json_path, sentence_id, filtered_x, filtered_y, anchor_points,
                                  english_sentences, chinese_sentences, len_sents1, len_sents2,
                                  sim_mat, params, average_density)
        
        print(f"✅ 第一部分处理完成！")
        print(f"="*60)
        
        return filtered_x, filtered_y, len_sents1, len_sents2
        
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {str(e)}")
        raise


# 使用示例
if __name__ == "__main__":
    # 参数配置
    params = {
        'verbose': True,
        'veryVerbose': True,
        'cosThreshold': 0.5,
        'margin': 0.1,
        'minDensityRatio': 0.8,
        'detectIntervals': True,
        'sentRatio': 0,
        'maxDistToTheDiagonal': 10,
        'maxGapSize': 20,
        'minHorizontalDensity': 0.1,
        'localDiagBeam': 0.3,
        'reiterateFiltering': False,
        'kBest': 3,
    }
    
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
            "sentences_file_path": f"{data_prefix}sentence.json",
            "matrix_file_path": f"{data_prefix}Qwen3-Embedding-8B-output.json",
            "output_json_path": f"{data_prefix}anchor_filter-output.json"
        }

    # 获取路径配置
    paths = get_data_paths()
    sentences_file_path = paths["sentences_file_path"]
    matrix_file_path = paths["matrix_file_path"]
    output_json_path = paths["output_json_path"]
    sentence_id = 0
    
    print("🚀 开始执行锚点过滤阶段（第一部分）...")
    
    try:
        filtered_x, filtered_y, len_sents1, len_sents2 = run_anchor_filtering_phase(
            params, sentences_file_path, matrix_file_path, sentence_id, output_json_path
        )
        
        print(f"\n🎉 第一部分执行成功！")
        print(f"  过滤后锚点数量: {len(filtered_x)}")
        print(f"  数据已保存至: {output_json_path}")
        print(f"  请运行第二部分代码继续处理...")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}") 