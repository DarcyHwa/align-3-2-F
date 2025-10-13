# -*- coding:utf8 -*-
"""
完整的锚点提取和区间划分模块 - 完全复现参考代码逻辑

本模块完整复现 reference-code/anchor_points.py 中的 extract_anchor_points 函数，
包括锚点密度过滤后到区间划分完成之间的所有处理逻辑。

主要功能：
1. 完整复现参考代码的区间划分算法
2. 包含单调性检查、密度比率检查、前瞻性检查
3. 实现最终的对角线过滤步骤
4. 保持与参考代码完全一致的处理逻辑

作者：AI Assistant
日期：2025-01-29
"""

import os
import sys
import json
import math
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def compute_local_density(params: Dict[str, Any], x: int, y: int, anchor_points: Dict[Tuple[int, int], int], 
                         len_sents1: int, len_sents2: int, sim_mat: np.ndarray) -> float:
    """
    计算局部密度 - 复现参考代码逻辑
    
    参数：
    params: 参数字典
    x, y: 锚点坐标
    anchor_points: 锚点字典
    len_sents1, len_sents2: 文档大小
    sim_mat: 相似度矩阵
    
    返回值：
    float: 局部密度值
    """
    
    # 使用参考代码中的密度计算逻辑
    delta_x = params.get('deltaX', 10)
    delta_y = params.get('deltaY', 10)
    
    count = 0
    total_similarity = 0
    
    # 计算窗口内的锚点数量和相似度
    for (ax, ay) in anchor_points.keys():
        if abs(ax - x) <= delta_x and abs(ay - y) <= delta_y:
            count += 1
            if 0 <= ax < sim_mat.shape[0] and 0 <= ay < sim_mat.shape[1]:
                total_similarity += sim_mat[ax, ay]
    
    # 计算窗口面积
    window_area = (2 * delta_x + 1) * (2 * delta_y + 1)

    # 返回密度值（结合数量和相似度）
    if window_area > 0 and count > 0:
        return (count / window_area) * (total_similarity / count)
    return 0.0


def safe_character_count(sentences: List[str], start_idx: int, end_idx: int) -> int:
    """
    安全的字符计数函数，避免越界访问句子列表
    """
    char_count = 0
    safe_start = max(0, start_idx)
    safe_end = min(end_idx, len(sentences) - 1)
    
    if safe_start <= safe_end and safe_start < len(sentences):
        for n in range(safe_start, safe_end + 1):
            if 0 <= n < len(sentences):
                char_count += len(sentences[n])
    
    return char_count


def extract_anchor_points_complete(params: Dict[str, Any], filtered_x: List[int], filtered_y: List[int],
                                  anchor_points: Dict[Tuple[int, int], int], average_density: float,
                                  sents1: List[str], sents2: List[str], len_sents1: int, len_sents2: int,
                                  sim_mat: np.ndarray) -> Tuple[List[int], List[int], List[Tuple[Tuple[int, int], Tuple[int, int]]],
                                                               int, int, int, int]:
    """
    完整复现参考代码的区间划分和结果输出逻辑（第3、4步骤）

    严格按照 reference-code/anchor_points.py 第649-875行的逻辑实现：
    - STEP 8: 寻找可对齐区间 (第649-838行)
    - 最终对角线过滤 (第843-872行)
    - 不包含预定义锚点逻辑 (跳过第662-673行)
    - 添加文档结束点时检查重复 (改进第653-654行)

    参数：
    params: 参数字典，包含所有算法参数
    filtered_x, filtered_y: 密度过滤后的锚点坐标列表
    anchor_points: 锚点字典，用于密度计算
    average_density: 全局平均密度
    sents1, sents2: 英文和中文句子列表
    len_sents1, len_sents2: 文档句子数量
    sim_mat: 相似度矩阵

    返回值：
    Tuple: (final_filtered_x, final_filtered_y, intervals, interval_length_sent1, interval_length_sent2, interval_length_char1, interval_length_char2)
    - final_filtered_x, final_filtered_y: 最终过滤后的锚点坐标
    - intervals: 可对齐区间列表
    - interval_length_sent1, interval_length_sent2: 区间内句子总数
    - interval_length_char1, interval_length_char2: 区间内字符总数
    """

    print(f"\n" + "="*80)
    print(f"🚀 开始完整的锚点提取和区间划分（完全复现参考代码）")
    print(f"="*80)
    print(f"📊 输入数据概览:")
    print(f"   • 输入锚点数量: {len(filtered_x)}")
    print(f"   • 文档大小: 英文{len_sents1}句, 中文{len_sents2}句")
    print(f"   • 平均密度: {average_density:.4f}")
    print(f"   • 相似度矩阵维度: {sim_mat.shape}")
    print(f"   • 锚点字典大小: {len(anchor_points)}")

    if len(filtered_x) > 0:
        print(f"📍 输入锚点详情:")
        for i, (x, y) in enumerate(zip(filtered_x, filtered_y)):
            print(f"   锚点{i+1}: ({x},{y})")

    # 特殊情况：如果没有锚点，创建覆盖整个文档的区间
    if len(filtered_x) == 0:
        print(f"\n⚠️  特殊情况处理：没有输入锚点")
        print(f"   → 将创建覆盖整个文档的单一区间")

        begin_int = (-1, -1)
        last_i = len_sents1 - 1
        last_j = len_sents2 - 1

        print(f"   → 计算区间统计信息...")
        interval_length_sent1 = last_i - begin_int[0]
        interval_length_sent2 = last_j - begin_int[1]
        interval_length_char1 = safe_character_count(sents1, 0, last_i)
        interval_length_char2 = safe_character_count(sents2, 0, last_j)
        intervals = [(begin_int, (last_i, last_j))]

        print(f"   → 英文句子数: {interval_length_sent1}")
        print(f"   → 中文句子数: {interval_length_sent2}")
        print(f"   → 英文字符数: {interval_length_char1}")
        print(f"   → 中文字符数: {interval_length_char2}")
        print(f"✅ 创建单一区间: {intervals[0]}")
        print(f"   → 返回空锚点列表和单一区间")

        return [], [], intervals, interval_length_sent1, interval_length_sent2, interval_length_char1, interval_length_char2
    
    # =====> STEP 8 : finding aligned intervals (严格复现参考代码第649-838行)

    print(f"\n" + "="*60)
    print(f"📐 STEP 8: 寻找可对齐区间")
    print(f"="*60)
    print(f"🔧 步骤8.1: 初始化区间划分变量")

    # 复现参考代码第651行：初始化区间起点
    begin_int = (-1, -1)
    print(f"   → 初始化区间起点: {begin_int}")

    # 复现参考代码第652-654行：添加文档末尾作为锚点，但要检查重复
    print(f"🔧 步骤8.2: 准备工作锚点列表")
    work_x = filtered_x.copy()
    work_y = filtered_y.copy()
    print(f"   → 复制输入锚点列表: {len(work_x)}个锚点")

    last_doc_x = len_sents1 - 1
    last_doc_y = len_sents2 - 1
    print(f"   → 文档末尾坐标: ({last_doc_x},{last_doc_y})")

    # 改进：检查是否需要添加文档末尾锚点（避免重复）
    if not work_x or work_x[-1] != last_doc_x or work_y[-1] != last_doc_y:
        work_x.append(last_doc_x)
        work_y.append(last_doc_y)
        print(f"   ✅ 添加文档末尾锚点: ({last_doc_x},{last_doc_y})")
    else:
        print(f"   ℹ️  文档末尾锚点已存在，无需重复添加: ({last_doc_x},{last_doc_y})")

    # 复现参考代码第655-660行：初始化变量
    print(f"🔧 步骤8.3: 初始化区间划分状态变量")
    last_i = 0
    last_j = 0
    intervals = []  # 区间对列表 (beginInt, endInt)
    nb_in_interval = 0

    print(f"   → last_i (上一个有效锚点X): {last_i}")
    print(f"   → last_j (上一个有效锚点Y): {last_j}")
    print(f"   → intervals (区间列表): 空列表")
    print(f"   → nb_in_interval (当前区间内锚点数): {nb_in_interval}")

    # 初始化区间长度统计
    interval_length_sent1 = 0
    interval_length_sent2 = 0
    interval_length_char1 = 0
    interval_length_char2 = 0

    print(f"🔧 步骤8.4: 显示最终工作锚点列表")
    print(f"   → 工作锚点总数: {len(work_x)}")
    for i, (x, y) in enumerate(zip(work_x, work_y)):
        print(f"   锚点{i+1}: ({x},{y})")

    # 跳过预定义锚点逻辑（第662-673行），因为我们不使用预定义锚点
    print(f"🔧 步骤8.5: 跳过预定义锚点逻辑（我们不使用预定义锚点）")
    print(f"   → 直接进入 detectIntervals 分支")
    # 直接进入 detectIntervals 分支（严格复现参考代码第674-838行）

    if params.get('detectIntervals', True):
        print(f"\n" + "="*60)
        print(f"🔍 开始检测可对齐区间（detectIntervals=True）")
        print(f"="*60)

        # 复现参考代码第675行：计算句子长度比例系数
        print(f"🔧 步骤8.6: 计算句子长度比例系数")
        coeff = 1 if params.get('sentRatio', 0) == 0 else params.get('sentRatio', 1)
        print(f"   → sentRatio参数: {params.get('sentRatio', 0)}")
        print(f"   → 计算得到的比例系数: {coeff}")
        print(f"   → 用途: 计算期望的Y坐标位置")

        # 复现参考代码第676行：遍历所有锚点
        print(f"\n🔧 步骤8.7: 遍历所有工作锚点进行区间划分")
        print(f"   → 总锚点数: {len(work_x)}")
        print(f"   → 开始逐个处理每个锚点...")

        for num in range(0, len(work_x)):
            i, j = work_x[num], work_y[num]

            print(f"\n" + "-"*50)
            print(f"🔍 处理锚点 {num+1}/{len(work_x)}: ({i},{j})")
            print(f"-"*50)

            # 复现参考代码第679-682行：计算局部密度和密度比率
            print(f"   📊 步骤8.7.1: 计算局部密度和密度比率")
            local_density = compute_local_density(params, i, j, anchor_points, len_sents1, len_sents2, sim_mat)
            density_ratio = 0
            if average_density > 0:
                density_ratio = local_density / average_density

            print(f"      → 局部密度: {local_density:.4f}")
            print(f"      → 全局平均密度: {average_density:.4f}")
            print(f"      → 密度比率: {density_ratio:.4f}")

            # 复现参考代码第684-687行：计算与对角线的偏差
            print(f"   📏 步骤8.7.2: 计算与对角线的偏差")
            expected_j = last_j + (i - last_i) * coeff
            vertical_deviation = abs(j - expected_j)
            new_interval = False

            print(f"      → 上一个有效锚点: ({last_i},{last_j})")
            print(f"      → 当前锚点: ({i},{j})")
            print(f"      → 期望Y坐标: {expected_j:.1f}")
            print(f"      → 实际Y坐标: {j}")
            print(f"      → 垂直偏差: {vertical_deviation:.1f}")

            if params.get('veryVerbose', False):
                print(f"      → 详细计算: {last_j} + ({i} - {last_i}) * {coeff} = {expected_j:.1f}")
                print(f"      → 偏差计算: |{j} - {expected_j:.1f}| = {vertical_deviation:.1f}")

            # 复现参考代码第689-701行：单调性约束检查
            print(f"   🔍 步骤8.7.3: 单调性约束检查")
            if num > 1 and num < len(work_x) - 2:
                print(f"      → 检查位置: 锚点{num+1}在序列中间，可以进行单调性检查")
                print(f"      → 检查范围: 锚点{num-1}到锚点{num+3}")

                x_monotonic = (work_x[num - 2] <= work_x[num - 1] <= work_x[num + 1] <= work_x[num + 2])
                y_monotonic = (work_y[num - 2] <= work_y[num - 1] <= work_y[num + 1] <= work_y[num + 2])
                x_in_range = (work_x[num - 1] <= i <= work_x[num + 1])
                y_in_range = (work_y[num - 1] <= j <= work_y[num + 1])

                print(f"      → X序列单调性: {x_monotonic} ({work_x[num-2]} <= {work_x[num-1]} <= {work_x[num+1]} <= {work_x[num+2]})")
                print(f"      → Y序列单调性: {y_monotonic} ({work_y[num-2]} <= {work_y[num-1]} <= {work_y[num+1]} <= {work_y[num+2]})")
                print(f"      → 当前点X在范围内: {x_in_range} ({work_x[num-1]} <= {i} <= {work_x[num+1]})")
                print(f"      → 当前点Y在范围内: {y_in_range} ({work_y[num-1]} <= {j} <= {work_y[num+1]})")

                if x_monotonic and y_monotonic and (not x_in_range or not y_in_range):
                    print(f"      ❌ 单调性检查失败：锚点({i},{j})破坏单调性，被忽略")
                    # 当前点被跳过
                    work_x[num] = last_i
                    work_y[num] = last_j
                    continue
                else:
                    print(f"      ✅ 单调性检查通过")
            else:
                print(f"      → 跳过单调性检查（锚点位置：{num+1}，不在可检查范围内）")

            # 复现参考代码第703-713行：偏离且低密度点检查
            print(f"   🔍 步骤8.7.4: 偏离且低密度点检查")
            max_dist_half = params.get('maxDistToTheDiagonal', 10) / 2
            min_density_ratio = params.get('minDensityRatio', 0.8)

            is_deviating = vertical_deviation > max_dist_half
            is_backward = i < last_i or j < last_j
            is_low_density = density_ratio < min_density_ratio

            print(f"      → 偏离检查: 垂直偏差{vertical_deviation:.1f} > {max_dist_half} = {is_deviating}")
            print(f"      → 后退检查: ({i},{j}) < ({last_i},{last_j}) = {is_backward}")
            print(f"      → 低密度检查: 密度比率{density_ratio:.4f} < {min_density_ratio} = {is_low_density}")

            if (is_deviating or is_backward) and is_low_density:
                print(f"      ❌ 偏离且低密度检查失败：锚点({i},{j})被忽略")
                print(f"         原因: {'偏离对角线' if is_deviating else ''}{'且' if is_deviating and is_backward else ''}{'位置后退' if is_backward else ''}且密度过低")
                # 当前点被跳过
                work_x[num] = last_i
                work_y[num] = last_j
                continue
            else:
                print(f"      ✅ 偏离且低密度检查通过")

            # 复现参考代码第715-720行：检查是否在对角线附近
            print(f"   🔍 步骤8.7.5: 对角线附近检查")
            max_dist_to_diagonal = params.get('maxDistToTheDiagonal', 10)
            print(f"      → 最大允许距离: {max_dist_to_diagonal}")
            print(f"      → 当前垂直偏差: {vertical_deviation:.1f}")

            if vertical_deviation <= max_dist_to_diagonal:
                nb_in_interval += 1
                last_i = i
                last_j = j
                print(f"      ✅ 锚点({i},{j})在对角线附近，被接受")
                print(f"      → 更新last_i: {last_i}, last_j: {last_j}")
                print(f"      → 当前区间内锚点数: {nb_in_interval}")
            else:
                # 复现参考代码第722-723行：偏离点的详细日志
                print(f"      ⚠️  锚点({i},{j})偏离对角线")
                print(f"      → 偏差{vertical_deviation:.1f} > 阈值{max_dist_to_diagonal}")
                print(f"      → 进入偏离点处理流程...")

                # 复现参考代码第725-754行：前瞻性检查
                print(f"   🔮 步骤8.7.6: 前瞻性检查（预测后续锚点行为）")
                preview_scope = 2
                print(f"      → 前瞻范围: {preview_scope}个锚点")

                if num + preview_scope < len(work_x):
                    next_i, next_j = work_x[num + preview_scope], work_y[num + preview_scope]
                    print(f"      → 前瞻锚点: ({next_i},{next_j})")

                    # 检查前瞻锚点与前一个点的对齐情况
                    next_expected_j = last_j + (next_i - last_i) * params.get('sentRatio', 1)
                    next_vertical_deviation = abs(next_j - next_expected_j)

                    print(f"      → 前瞻锚点相对于前一个点的期望Y: {next_expected_j:.1f}")
                    print(f"      → 前瞻锚点的垂直偏差: {next_vertical_deviation:.1f}")

                    # 下一个点与前一个点对齐
                    if next_vertical_deviation <= params.get('maxDistToTheDiagonal', 10):
                        print(f"      ❌ 前瞻锚点与前一个点对齐，忽略当前点")
                        print(f"         → 当前点({i},{j})被跳过")
                        # 当前点被跳过
                        work_x[num] = last_i
                        work_y[num] = last_j
                        continue
                    else:
                        # 检查下一个点是否与当前点对齐
                        print(f"      → 检查前瞻锚点是否与当前点对齐...")
                        next_expected_j = j + (next_i - i) * params.get('sentRatio', 1)
                        next_vertical_deviation = abs(next_j - next_expected_j)

                        print(f"      → 前瞻锚点相对于当前点的期望Y: {next_expected_j:.1f}")
                        print(f"      → 前瞻锚点相对于当前点的偏差: {next_vertical_deviation:.1f}")

                        # 如果下一个点与当前点对齐，则应该创建新区间
                        if next_vertical_deviation <= params.get('maxDistToTheDiagonal', 10) and density_ratio > params.get('minDensityRatio', 0.8):
                            print(f"      ✅ 前瞻锚点与当前点对齐且密度足够，保留当前点用于新区间")
                            new_interval = True
                        else:
                            print(f"      ❌ 前瞻锚点不与当前点对齐或密度不足，忽略当前点")
                            print(f"         → 偏差: {next_vertical_deviation:.1f}, 密度比率: {density_ratio:.4f}")
                            # 当前点被跳过
                            work_x[num] = last_i
                            work_y[num] = last_j
                            continue
                else:
                    print(f"      → 无法进行前瞻检查（剩余锚点不足{preview_scope}个）")

                # 复现参考代码第767-770行：计算距离，检查是否需要创建新区间
                print(f"   📏 步骤8.7.7: 距离间隙检查")
                d = math.sqrt((i - last_i) ** 2 + (j - last_j) ** 2)
                max_gap_size = params.get('maxGapSize', 20)
                high_density_threshold = 1.5

                print(f"      → 当前点到上一个有效点的距离: {d:.2f}")
                print(f"      → 最大允许间隙: {max_gap_size}")
                print(f"      → 高密度阈值: {high_density_threshold}")
                print(f"      → 当前密度比率: {density_ratio:.4f}")

                # 如果有间隙，前一个区间被关闭，新区间将开始
                if d > max_gap_size and density_ratio > high_density_threshold:
                    print(f"      ✅ 距离过大且密度高，需要创建新区间")
                    print(f"         → 距离{d:.2f} > 阈值{max_gap_size} 且 密度{density_ratio:.4f} > {high_density_threshold}")
                    new_interval = True
                else:
                    print(f"      → 距离间隙检查通过，继续当前区间")

            # 复现参考代码第774-820行：创建新区间（如果需要）
            if new_interval:
                print(f"\n   🔄 步骤8.7.8: 创建新区间")
                end_int = (last_i, last_j)
                print(f"      → 关闭当前区间: {begin_int} → {end_int}")
                print(f"      → 触发点: ({i},{j})")

                if begin_int[0] < last_i and begin_int[1] < last_j:
                    # 为了保存区间，我们根据水平宽度计算选定点的密度
                    print(f"      → 验证区间有效性...")
                    horizontal_width = last_i - begin_int[0]
                    horizontal_density = nb_in_interval / horizontal_width if horizontal_width > 0 else 0
                    min_horizontal_density = params.get('minHorizontalDensity', 0.1)

                    print(f"         • 区间水平宽度: {horizontal_width}")
                    print(f"         • 区间内锚点数: {nb_in_interval}")
                    print(f"         • 水平密度: {horizontal_density:.4f}")
                    print(f"         • 最小密度要求: {min_horizontal_density}")
                    print(f"         • 最小锚点数要求: 2")

                    if horizontal_density >= min_horizontal_density and nb_in_interval > 1:
                        print(f"      ✅ 区间有效，保存到区间列表")
                        intervals.append((begin_int, end_int))

                        # 计算区间统计
                        sent_count_1 = last_i - begin_int[0] + 1
                        sent_count_2 = last_j - begin_int[1] + 1
                        interval_length_sent1 += sent_count_1
                        interval_length_sent2 += sent_count_2

                        print(f"         • 英文句子数: {sent_count_1}")
                        print(f"         • 中文句子数: {sent_count_2}")

                        # 计算字符数
                        char_count_1 = 0
                        char_count_2 = 0
                        for n in range(max(0, begin_int[0]), last_i + 1):
                            if n < len(sents1):
                                char_count_1 += len(sents1[n])
                        for n in range(max(0, begin_int[1]), last_j + 1):
                            if n < len(sents2):
                                char_count_2 += len(sents2[n])

                        interval_length_char1 += char_count_1
                        interval_length_char2 += char_count_2

                        print(f"         • 英文字符数: {char_count_1}")
                        print(f"         • 中文字符数: {char_count_2}")
                    else:
                        print(f"      ❌ 区间无效，被丢弃（密度过低或锚点不足）")
                        print(f"         原因: 密度{horizontal_density:.4f} < {min_horizontal_density} 或 锚点数{nb_in_interval} <= 1")
                else:
                    print(f"      → 区间无效（起点不小于终点），跳过保存")

                print(f"      → 开始新区间: ({i},{j})")
                begin_int = (i, j)
                nb_in_interval = 0

                # 复现参考代码第817-820行：更新lastI和lastJ
                print(f"      → 更新有效锚点位置: ({i},{j})")
                last_i = i
                last_j = j
    else:
        # 复现参考代码第821-823行：如果不检测区间，直接使用整个文档
        print(f"\n" + "="*60)
        print(f"⚠️  detectIntervals=False，使用整个文档作为单一区间")
        print(f"="*60)
        last_i = len_sents1 - 1
        last_j = len_sents2 - 1
        print(f"   → 设置最后有效位置为文档末尾: ({last_i},{last_j})")

    # 复现参考代码第829-838行：关闭最后一个区间
    print(f"\n" + "="*60)
    print(f"🔚 关闭最后一个区间")
    print(f"="*60)
    print(f"   → 检查是否需要关闭最后区间...")
    print(f"   → 当前区间起点: {begin_int}")
    print(f"   → 最后有效位置: ({last_i},{last_j})")

    if last_i != begin_int[0]:
        print(f"   ✅ 需要关闭最后区间")

        # closing last interval
        sent_count_1 = last_i - begin_int[0] + 1
        sent_count_2 = last_j - begin_int[1] + 1
        interval_length_sent1 += sent_count_1
        interval_length_sent2 += sent_count_2

        print(f"   → 计算最后区间统计...")
        print(f"      • 英文句子数: {sent_count_1}")
        print(f"      • 中文句子数: {sent_count_2}")

        # 计算字符数
        char_count_1 = 0
        char_count_2 = 0
        for n in range(max(0, begin_int[0]), last_i + 1):
            if n < len(sents1):
                char_count_1 += len(sents1[n])
        for n in range(max(0, begin_int[1]), last_j + 1):
            if n < len(sents2):
                char_count_2 += len(sents2[n])

        interval_length_char1 += char_count_1
        interval_length_char2 += char_count_2

        print(f"      • 英文字符数: {char_count_1}")
        print(f"      • 中文字符数: {char_count_2}")

        intervals.append((begin_int, (last_i, last_j)))
        print(f"   ✅ 最后区间已保存: {begin_int} → ({last_i},{last_j})")
    else:
        print(f"   → 无需关闭最后区间（起点与终点相同）")

    print(f"\n📊 区间划分完成统计:")
    print(f"   → 总区间数: {len(intervals)}")
    print(f"   → 总英文句子数: {interval_length_sent1}")
    print(f"   → 总中文句子数: {interval_length_sent2}")
    print(f"   → 总英文字符数: {interval_length_char1}")
    print(f"   → 总中文字符数: {interval_length_char2}")

    if intervals:
        print(f"\n📋 所有区间详情:")
        for idx, (begin, end) in enumerate(intervals):
            print(f"   区间{idx+1}: {begin} → {end}")
    
    print(f"\n=== 区间划分完成 ===")
    print(f"总区间数: {len(intervals)}")
    print(f"区间覆盖统计:")
    print(f"  英文句子: {interval_length_sent1}/{len_sents1}")
    print(f"  中文句子: {interval_length_sent2}/{len_sents2}")
    print(f"  英文字符: {interval_length_char1}")
    print(f"  中文字符: {interval_length_char2}")
    
    # 显示所有区间详情
    if intervals:
        print(f"\n📋 区间详情列表:")
        for idx, (begin, end) in enumerate(intervals):
            print(f"  区间{idx+1}: {begin} → {end}")
    
    # 复现参考代码第843-875行：最终过滤步骤 - 对于每个区间，距离对角线太远的点被丢弃
    print(f"\n" + "="*80)
    print(f"🔍 最终对角线过滤步骤（第4步骤的一部分）")
    print(f"="*80)
    print(f"📊 过滤前状态:")
    print(f"   → 锚点总数: {len(work_x)}")
    print(f"   → 区间总数: {len(intervals)}")

    if work_x:
        print(f"   → 过滤前锚点列表:")
        for idx, (x, y) in enumerate(zip(work_x, work_y)):
            print(f"      锚点{idx+1}: ({x},{y})")

    removed_count = 0
    i = 0

    for interval_idx, (begin, end) in enumerate(intervals):
        x_begin, y_begin = begin
        x_end, y_end = end

        print(f"\n🔍 处理区间{interval_idx+1}: {begin} → {end}")

        if (x_end - x_begin) * (y_end - y_begin) == 0:
            print(f"   → 跳过无效区间（面积为0）")
            continue

        print(f"   → 区间有效，开始过滤区间内的锚点...")

        # 寻找区间内的锚点
        print(f"   → 寻找区间内的锚点（从索引{i}开始）...")
        while i < len(work_x) and work_x[i] < x_begin:
            print(f"      跳过区间外锚点: ({work_x[i]},{work_y[i]})")
            i += 1

        # 如果点i落在x区间内
        points_in_interval = 0
        while i < len(work_x) and work_x[i] >= x_begin and work_x[i] <= x_end:
            points_in_interval += 1
            delete = False
            current_x, current_y = work_x[i], work_y[i]

            print(f"\n   📍 检查锚点{i+1}: ({current_x},{current_y})")

            # 如果点i不在y区间内，删除点
            if current_y < y_begin or current_y > y_end:
                delete = True
                print(f"      ❌ Y坐标超出区间范围: {current_y} ∉ [{y_begin},{y_end}]")

            # 计算期望的Y坐标（在区间对角线上）
            expected_y = y_begin + (current_x - x_begin) / (x_end - x_begin) * (y_end - y_begin)
            print(f"      → 期望Y坐标（对角线上）: {expected_y:.2f}")

            # 计算偏差
            relative_deviation = abs((current_y - expected_y) / (y_end - y_begin)) if (y_end - y_begin) > 0 else 0
            absolute_deviation = abs(current_y - expected_y)

            local_diag_beam = params.get('localDiagBeam', 0.3)
            max_dist_to_diagonal = params.get('maxDistToTheDiagonal', 10)

            print(f"      → 相对偏差: {relative_deviation:.4f} (阈值: {local_diag_beam})")
            print(f"      → 绝对偏差: {absolute_deviation:.2f} (阈值: {max_dist_to_diagonal})")

            # 如果点i距离对角线太远，删除点
            if relative_deviation > local_diag_beam or absolute_deviation > max_dist_to_diagonal:
                delete = True
                print(f"      ❌ 距离对角线太远")
                if relative_deviation > local_diag_beam:
                    print(f"         相对偏差 {relative_deviation:.4f} > {local_diag_beam}")
                if absolute_deviation > max_dist_to_diagonal:
                    print(f"         绝对偏差 {absolute_deviation:.2f} > {max_dist_to_diagonal}")

            if delete:
                print(f"      🗑️  删除锚点({current_x},{current_y})")
                del work_x[i]
                del work_y[i]
                removed_count += 1
                if i >= len(work_x):
                    break
            else:
                print(f"      ✅ 保留锚点({current_x},{current_y})")
                i += 1

        print(f"   → 区间{interval_idx+1}处理完成，检查了{points_in_interval}个锚点")

    print(f"\n📊 最终过滤完成统计:")
    print(f"   → 过滤后锚点数量: {len(work_x)}")
    print(f"   → 移除锚点数量: {removed_count}")

    if work_x:
        print(f"   → 最终保留的锚点:")
        for idx, (x, y) in enumerate(zip(work_x, work_y)):
            print(f"      锚点{idx+1}: ({x},{y})")
    else:
        print(f"   → 所有锚点都被过滤掉了")

    return work_x, work_y, intervals, interval_length_sent1, interval_length_sent2, interval_length_char1, interval_length_char2


def load_filtered_anchor_data_complete(json_file_path: str) -> Dict[str, Any]:
    """
    从JSON文件中加载密度过滤后的锚点数据（完整版本）


    1. 详细的数据验证和完整性检查
    2. 丰富的锚点信息提取和展示
    3. 结构化的数据组织和错误处理
    4. 支持多锚点的详细信息管理

    参数：
    json_file_path (str): 过滤后锚点数据的JSON文件路径

    返回值：
    Dict[str, Any]: 包含所有必要数据的字典，支持多锚点处理
    """

    print(f"\n" + "="*80)
    print(f"📂 加载过滤后的锚点数据（完整增强版本）")
    print(f"="*80)
    print(f"📁 数据文件路径: {json_file_path}")

    try:
        print(f"🔄 正在读取JSON文件...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ 文件读取成功")

        # 提取基本信息
        print(f"\n🔍 步骤1: 提取基本信息")
        sentence_id = data.get('sentence_id')
        filtered_anchors = data.get('filtered_anchors', {})
        sentence_data = data.get('sentence_data', {})
        similarity_matrix = data.get('similarity_matrix', [])
        parameters = data.get('parameters', {})
        statistics = data.get('statistics', {})

        print(f"   → 句子ID: {sentence_id}")
        print(f"   → 数据结构完整性: {'✅' if all([filtered_anchors, sentence_data, similarity_matrix]) else '⚠️'}")

        # 验证数据完整性
        print(f"\n🔍 步骤2: 验证数据完整性")
        anchor_count = filtered_anchors.get('count', 0)
        anchors = filtered_anchors.get('anchors', [])

        if anchor_count != len(anchors):
            print(f"⚠️  警告：锚点数量不匹配 - 声明{anchor_count}个，实际{len(anchors)}个")
        else:
            print(f"✅ 锚点数量验证通过: {anchor_count}个")

        # 详细的数据概览
        print(f"\n📊 数据概览:")
        print(f"   • 句子ID: {sentence_id}")
        print(f"   • 过滤后锚点数量: {anchor_count}")
        print(f"   • 英文句子数: {sentence_data.get('len_sents1', 0)}")
        print(f"   • 中文句子数: {sentence_data.get('len_sents2', 0)}")
        print(f"   • 相似度矩阵维度: {statistics.get('matrix_shape', 'N/A')}")
        print(f"   • 平均密度: {statistics.get('average_density', 'N/A'):.4f}")

        # 提取锚点坐标和构建增强的锚点字典
        print(f"\n🔍 步骤3: 构建增强的锚点数据结构")
        filtered_x = []
        filtered_y = []
        anchor_points = {}  # 简单锚点字典（用于密度计算）
        anchor_details = {}  # 详细锚点信息字典
        anchor_list = []     # 锚点列表（保持顺序）

        print(f"📍 锚点详情:")
        for anchor in anchors:
            anchor_id = anchor.get('id')
            coordinates = anchor.get('coordinates', [])
            anchor_info = anchor.get('anchor_info', {})

            if len(coordinates) == 2:
                x, y = coordinates
                filtered_x.append(x)
                filtered_y.append(y)

                # 简单锚点字典（兼容原有逻辑）
                anchor_points[(x, y)] = 1

                # 详细锚点信息字典
                anchor_details[(x, y)] = {
                    'id': anchor_id,
                    'coordinates': [x, y],
                    'type': anchor_info.get('type', 'unknown'),
                    'similarity': anchor_info.get('similarity', 0.0),
                    'source': anchor_info.get('source', 'unknown'),
                    'rank_in_row': anchor_info.get('rank_in_row', 0),
                    'total_candidates_in_row': anchor_info.get('total_candidates_in_row', 0),
                    'local_density': anchor_info.get('local_density', 0.0),
                    'density_ratio': anchor_info.get('density_ratio', 0.0),
                    'quality_score': anchor_info.get('quality_score', 0.0),
                    'filter_status': anchor_info.get('filter_status', 'unknown'),
                    'filter_reason': anchor_info.get('filter_reason', ''),
                    'final_rank': anchor_info.get('final_rank', 0)
                }

                # 锚点列表（保持顺序）
                anchor_list.append({
                    'id': anchor_id,
                    'coordinates': [x, y],
                    'info': anchor_details[(x, y)]
                })

                # 显示锚点详情
                similarity = anchor_info.get('similarity', 0.0)
                quality_score = anchor_info.get('quality_score', 0.0)
                density_ratio = anchor_info.get('density_ratio', 0.0)
                print(f"   锚点#{anchor_id}: ({x},{y}) 相似度={similarity:.4f} 质量={quality_score:.4f} 密度比={density_ratio:.4f}")

        # 转换相似度矩阵
        print(f"\n🔍 步骤4: 处理相似度矩阵")
        sim_mat = np.array(similarity_matrix, dtype=np.float64)
        print(f"   → 相似度矩阵形状: {sim_mat.shape}")
        print(f"   → 矩阵数据类型: {sim_mat.dtype}")
        print(f"   → 矩阵值范围: [{sim_mat.min():.4f}, {sim_mat.max():.4f}]")

        # 构建增强的返回结果
        result = {
            # 基本信息
            'sentence_id': sentence_id,
            'anchor_count': anchor_count,

            # 锚点数据（多种格式支持）
            'filtered_x': filtered_x,
            'filtered_y': filtered_y,
            'anchor_points': anchor_points,      # 简单字典（兼容性）
            'anchor_details': anchor_details,    # 详细信息字典
            'anchor_list': anchor_list,          # 有序列表

            # 统计信息
            'average_density': statistics.get('average_density', 0.0),
            'matrix_shape': statistics.get('matrix_shape', sim_mat.shape),

            # 句子数据
            'english_sentences': sentence_data.get('english_sentences', []),
            'chinese_sentences': sentence_data.get('chinese_sentences', []),
            'len_sents1': sentence_data.get('len_sents1', 0),
            'len_sents2': sentence_data.get('len_sents2', 0),

            # 矩阵数据
            'sim_mat': sim_mat,
            'similarity_matrix': similarity_matrix,  # 原始格式

            # 元数据
            'parameters': parameters,
            'statistics': statistics,
            'source_file': json_file_path,
            'load_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"\n✅ 数据加载完成")
        print(f"   → 成功加载 {len(filtered_x)} 个锚点")
        print(f"   → 文档大小: 英文{result['len_sents1']}句, 中文{result['len_sents2']}句")
        print(f"   → 数据结构: 3种锚点格式 + 完整元数据")
        print()

        return result

    except FileNotFoundError:
        error_msg = f"文件未找到: {json_file_path}"
        print(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"JSON解析错误: {str(e)}"
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"加载数据时发生未知错误: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)


def save_complete_results_to_json(output_file_path: str, sentence_id: int,
                                 final_x: List[int], final_y: List[int],
                                 intervals: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                                 interval_stats: Dict[str, int],
                                 anchor_data: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    将完整的区间划分结果保存到JSON文件（第4步骤：结果输出）

    学习 alignable_intervals_part2.py 的优点：
    1. 详细的锚点信息保留和传递
    2. 丰富的统计信息和质量评估
    3. 完整的处理历史和元数据记录
    4. 支持多锚点的详细信息管理

    参数：
    output_file_path (str): 输出JSON文件路径
    sentence_id (int): 句子ID
    final_x, final_y: 最终过滤后的锚点坐标
    intervals: 可对齐区间列表
    interval_stats: 区间统计信息
    anchor_data: 原始锚点数据（包含详细信息）
    params: 参数字典
    """

    print(f"\n" + "="*80)
    print(f"💾 保存完整区间划分结果到JSON文件（增强版本）")
    print(f"="*80)
    print(f"📁 输出文件: {output_file_path}")

    # 构建增强的最终锚点列表
    print(f"\n🔧 步骤1: 构建最终锚点数据结构")
    final_anchors = []
    anchor_processing_history = []

    for idx, (x, y) in enumerate(zip(final_x, final_y)):
        # 获取原始锚点的详细信息
        original_info = anchor_data.get('anchor_details', {}).get((x, y), {})

        # 直接构建最终锚点，优先保留高层键值
        final_anchor = {
            'id': idx + 1,
            'coordinates': [int(x), int(y)],
            'processing_status': 'final_kept',
            'final_rank': idx + 1
        }

        # 如果有原始信息，添加关键指标到顶层
        if original_info:
            final_anchor.update({
                'original_id': original_info.get('id', 'unknown'),
                'similarity': original_info.get('similarity', 0.0),
                'quality_score': original_info.get('quality_score', 0.0),
                'density_ratio': original_info.get('density_ratio', 0.0),
                'source': original_info.get('source', 'unknown')
            })
            
            # 构建original_info时直接排除已在顶层的属性
            final_anchor['original_info'] = {
                'type': original_info.get('type'),
                'rank_in_row': original_info.get('rank_in_row'),
                'total_candidates_in_row': original_info.get('total_candidates_in_row'),
                'local_density': original_info.get('local_density'),
                'filter_status': original_info.get('filter_status'),
                'filter_reason': original_info.get('filter_reason')
            }

            print(f"   锚点#{idx+1}: ({x},{y}) 原始ID={original_info.get('id', 'N/A')} 质量={original_info.get('quality_score', 0.0):.4f}")
        else:
            print(f"   锚点#{idx+1}: ({x},{y}) [新增锚点]")

        final_anchors.append(final_anchor)

    # 记录锚点处理历史
    input_anchor_count = len(anchor_data.get('filtered_x', []))
    final_anchor_count = len(final_anchors)
    filtered_count = input_anchor_count - final_anchor_count

    anchor_processing_history = {
        'input_count': input_anchor_count,
        'final_count': final_anchor_count,
        'filtered_count': filtered_count,
        'filter_rate': (filtered_count / input_anchor_count * 100) if input_anchor_count > 0 else 0,
        'processing_steps': [
            'density_filtering_completed',
            'interval_division_applied',
            'final_diagonal_filtering_applied'
        ]
    }

    # 构建增强的区间详情列表
    print(f"\n🔧 步骤2: 构建区间数据结构")
    interval_details = []
    total_interval_area = 0

    for idx, (begin, end) in enumerate(intervals):
        # 计算区间统计
        english_sentences = int(end[0] - begin[0] + 1)
        chinese_sentences = int(end[1] - begin[1] + 1)
        interval_area = english_sentences * chinese_sentences
        total_interval_area += interval_area

        # 计算区间内的字符数
        english_chars = 0
        chinese_chars = 0

        english_sents = anchor_data.get('english_sentences', [])
        chinese_sents = anchor_data.get('chinese_sentences', [])

        for i in range(max(0, begin[0]), min(end[0] + 1, len(english_sents))):
            english_chars += len(english_sents[i])
        for i in range(max(0, begin[1]), min(end[1] + 1, len(chinese_sents))):
            chinese_chars += len(chinese_sents[i])

        interval_detail = {
            'interval_id': idx + 1,
            'coordinates': {
                'start': [int(begin[0]), int(begin[1])],
                'end': [int(end[0]), int(end[1])]
            },
            'sentence_counts': {
                'english': english_sentences,
                'chinese': chinese_sentences
            },
            'character_counts': {
                'english': english_chars,
                'chinese': chinese_chars
            },
            'statistics': {
                'area': interval_area,
                'density': interval_area / (anchor_data.get('len_sents1', 1) * anchor_data.get('len_sents2', 1)),
                'english_coverage': english_sentences / anchor_data.get('len_sents1', 1),
                'chinese_coverage': chinese_sentences / anchor_data.get('len_sents2', 1)
            }
        }
        interval_details.append(interval_detail)

        print(f"   区间{idx+1}: {begin} → {end} 面积={interval_area} 英文={english_sentences}句 中文={chinese_sentences}句")

    # 计算质量评估指标
    print(f"\n🔧 步骤3: 计算质量评估指标")
    total_doc_area = anchor_data.get('len_sents1', 1) * anchor_data.get('len_sents2', 1)
    coverage_ratio = total_interval_area / total_doc_area if total_doc_area > 0 else 0

    quality_metrics = {
        'overall_quality': min(1.0, coverage_ratio + (final_anchor_count / max(1, input_anchor_count)) * 0.3),
        'coverage_ratio': coverage_ratio,
        'anchor_retention_rate': (final_anchor_count / max(1, input_anchor_count)),
        'interval_efficiency': len(intervals) / max(1, final_anchor_count),
        'average_interval_size': total_interval_area / max(1, len(intervals))
    }

    print(f"   → 整体质量评分: {quality_metrics['overall_quality']:.4f}")
    print(f"   → 覆盖率: {quality_metrics['coverage_ratio']:.4f}")
    print(f"   → 锚点保留率: {quality_metrics['anchor_retention_rate']:.4f}")

    # 构建完整的数据结构
    print(f"\n🔧 步骤4: 构建完整的输出数据结构")
    data = {
        "sentence_id": sentence_id,
        "final_anchors": {
            "count": len(final_anchors),
            "anchors": final_anchors,
            "processing_history": anchor_processing_history
        },
        "alignable_intervals": {
            "count": len(intervals),
            "intervals": interval_details,
            "total_area": total_interval_area
        },
        "interval_statistics": {
            "total_english_sentences": interval_stats.get('interval_length_sent1', 0),
            "total_chinese_sentences": interval_stats.get('interval_length_sent2', 0),
            "total_english_characters": interval_stats.get('interval_length_char1', 0),
            "total_chinese_characters": interval_stats.get('interval_length_char2', 0),
            "coverage_english_percent": (interval_stats.get('interval_length_sent1', 0) / anchor_data.get('len_sents1', 1)) * 100,
            "coverage_chinese_percent": (interval_stats.get('interval_length_sent2', 0) / anchor_data.get('len_sents2', 1)) * 100
        },
        "quality_metrics": quality_metrics,
        "source_data": {
            "input_anchor_count": input_anchor_count,
            "final_anchor_count": final_anchor_count,
            "filtered_anchor_count": filtered_count,
            "document_size": {
                "english_sentences": anchor_data.get('len_sents1', 0),
                "chinese_sentences": anchor_data.get('len_sents2', 0)
            },
            "average_density": anchor_data.get('average_density', 0.0),
            "source_file": anchor_data.get('source_file', 'unknown')
        },
        "processing_metadata": {
            "parameters": params,
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_note": "完整复现参考代码的区间划分和结果输出逻辑（增强版本）",
            "algorithm_version": "complete_anchor_extraction_v2.0",
            "features": [
                "enhanced_anchor_tracking",
                "detailed_interval_statistics",
                "quality_metrics_calculation",
                "processing_history_recording"
            ]
        }
    }

    try:
        print(f"\n💾 保存JSON文件...")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 完整结果已成功保存到: {output_file_path}")
        print(f"\n📊 保存结果统计:")
        print(f"   • 最终锚点数量: {len(final_anchors)}")
        print(f"   • 区间数量: {len(intervals)}")
        print(f"   • 英文覆盖率: {data['interval_statistics']['coverage_english_percent']:.1f}%")
        print(f"   • 中文覆盖率: {data['interval_statistics']['coverage_chinese_percent']:.1f}%")
        print(f"   • 整体质量评分: {quality_metrics['overall_quality']:.4f}")
        print(f"   • 锚点保留率: {quality_metrics['anchor_retention_rate']:.1%}")
        print()

    except Exception as e:
        error_msg = f"保存JSON文件时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)


# 使用示例和主程序入口
if __name__ == "__main__":
    print("="*100)
    print("🚀 完整的锚点提取和区间划分程序启动")
    print("="*100)
    print("📋 程序功能说明:")
    print("   • 第3步骤: 区间划分 - 基于过滤后的锚点划分可对齐区间")
    print("   • 第4步骤: 结果输出 - 最终对角线过滤并保存结果")
    print("   • 完全复现参考代码 anchor_points.py 第649-875行的逻辑")
    print("   • 为后续DTW对齐算法提供标准化输入数据")
    print()

    # 参数配置（与参考代码保持一致）
    print("🔧 步骤1: 配置算法参数")
    params = {
        'verbose': True,           # 显示详细处理信息
        'veryVerbose': True,       # 显示超详细调试信息
        'detectIntervals': True,   # 启用区间检测
        'sentRatio': 0,           # 句子比例（0表示自动计算）
        'maxDistToTheDiagonal': 10,    # 最大对角线距离
        'maxGapSize': 20,         # 最大间隙大小
        'minHorizontalDensity': 0.1,   # 最小水平密度
        'localDiagBeam': 0.3,     # 局部对角线束宽
        'minDensityRatio': 0.8,   # 最小密度比率
        'deltaX': 10,             # X方向密度计算窗口
        'deltaY': 10,             # Y方向密度计算窗口
    }

    print("   ✅ 参数配置完成:")
    for key, value in params.items():
        print(f"      • {key}: {value}")
    print()

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
            "input_json_path": f"{data_prefix}anchor_filter-output.json",
            "output_json_path": f"{data_prefix}complete_interval_extraction-output.json"
        }

    # 获取路径配置
    paths = get_data_paths()
    input_json_path = paths["input_json_path"]
    output_json_path = paths["output_json_path"]

    print("🔧 步骤2: 设置输入输出文件路径")

    print(f"   → 输入文件: {input_json_path}")
    print(f"   → 输出文件: {output_json_path}")
    print()

    print("� 步骤3: 开始执行完整的区间划分和结果输出...")
    print()

    try:
        # 加载数据
        print("📂 步骤3.1: 加载过滤后的锚点数据")
        anchor_data = load_filtered_anchor_data_complete(input_json_path)

        # 显示加载的数据结构详情
        print("🔍 步骤3.1.1: 验证加载的数据结构")
        print(f"   → 数据结构类型: {type(anchor_data)}")
        print(f"   → 主要键值: {list(anchor_data.keys())}")
        print(f"   → 锚点数据格式:")
        print(f"      • filtered_x: {len(anchor_data.get('filtered_x', []))} 个X坐标")
        print(f"      • filtered_y: {len(anchor_data.get('filtered_y', []))} 个Y坐标")
        print(f"      • anchor_points: {len(anchor_data.get('anchor_points', {}))} 个简单锚点")
        print(f"      • anchor_details: {len(anchor_data.get('anchor_details', {}))} 个详细锚点")
        print(f"      • anchor_list: {len(anchor_data.get('anchor_list', []))} 个有序锚点")

        if anchor_data.get('anchor_details'):
            print(f"   → 锚点详细信息示例:")
            for coord, details in list(anchor_data['anchor_details'].items())[:3]:  # 显示前3个
                print(f"      • {coord}: ID={details.get('id')}, 质量={details.get('quality_score', 0):.4f}")
        print()

        # 执行完整的锚点提取和区间划分
        print("⚙️  步骤3.2: 执行完整的锚点提取和区间划分")
        print("🔧 步骤3.2.1: 准备算法输入参数")
        print(f"   → 输入锚点坐标: X={anchor_data['filtered_x']}, Y={anchor_data['filtered_y']}")
        print(f"   → 锚点字典大小: {len(anchor_data['anchor_points'])}")
        print(f"   → 平均密度: {anchor_data['average_density']:.4f}")
        print(f"   → 文档大小: 英文{anchor_data['len_sents1']}句, 中文{anchor_data['len_sents2']}句")
        print(f"   → 相似度矩阵: {anchor_data['sim_mat'].shape}")

        final_x, final_y, intervals, sent1, sent2, char1, char2 = extract_anchor_points_complete(
            params,
            anchor_data['filtered_x'],
            anchor_data['filtered_y'],
            anchor_data['anchor_points'],
            anchor_data['average_density'],
            anchor_data['english_sentences'],
            anchor_data['chinese_sentences'],
            anchor_data['len_sents1'],
            anchor_data['len_sents2'],
            anchor_data['sim_mat']
        )

        print("🔍 步骤3.2.2: 验证处理结果")
        print(f"   → 最终锚点数量: {len(final_x)}")
        print(f"   → 输出区间数量: {len(intervals)}")
        print(f"   → 锚点保留率: {len(final_x)/len(anchor_data['filtered_x'])*100:.1f}%")
        if final_x:
            print(f"   → 最终锚点坐标: {list(zip(final_x, final_y))}")
        if intervals:
            print(f"   → 区间概览: {intervals}")
        print()

        # 保存结果
        print("💾 步骤3.3: 保存处理结果到JSON文件")
        print("🔧 步骤3.3.1: 构建区间统计数据")
        interval_stats = {
            'interval_length_sent1': sent1,
            'interval_length_sent2': sent2,
            'interval_length_char1': char1,
            'interval_length_char2': char2
        }

        print(f"   → 区间统计:")
        print(f"      • 英文句子总数: {sent1}")
        print(f"      • 中文句子总数: {sent2}")
        print(f"      • 英文字符总数: {char1}")
        print(f"      • 中文字符总数: {char2}")

        print("🔧 步骤3.3.2: 执行增强版JSON保存")
        save_complete_results_to_json(
            output_json_path,
            anchor_data['sentence_id'],
            final_x, final_y, intervals, interval_stats,
            anchor_data, params
        )

        print("="*100)
        print("🎉 第3、4步骤完整处理成功！")
        print("="*100)
        print("📊 处理结果总结:")
        print(f"   • 输入锚点数量: {len(anchor_data['filtered_x'])}")
        print(f"   • 最终锚点数量: {len(final_x)}")
        print(f"   • 输出区间数量: {len(intervals)}")
        print(f"   • 英文句子覆盖: {sent1}/{anchor_data['len_sents1']} ({sent1/anchor_data['len_sents1']*100:.1f}%)")
        print(f"   • 中文句子覆盖: {sent2}/{anchor_data['len_sents2']} ({sent2/anchor_data['len_sents2']*100:.1f}%)")
        print(f"   • 英文字符覆盖: {char1}")
        print(f"   • 中文字符覆盖: {char2}")
        print()
        print("📁 输出文件:")
        print(f"   • 结果文件: {output_json_path}")
        print()
        print("🚀 下一步:")
        print("   ✅ 数据已准备就绪，可继续进行DTW对齐算法！")
        print("   → 使用生成的区间数据进行动态时间规整对齐")
        print("="*100)

    except Exception as e:
        print("="*100)
        print("❌ 执行失败")
        print("="*100)
        print(f"错误信息: {e}")
        print("请检查:")
        print("   • 输入文件是否存在且格式正确")
        print("   • 输出目录是否有写入权限")
        print("   • 参数配置是否合理")
        print("="*100)
