import json
import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

def load_json(filepath):
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def map_clusters_to_labels(pred_clusters, true_labels):
    """
    将聚类编号映射到真实标签
    找到每个聚类对应的主要类别
    """
    cluster_to_label = {}
    
    # 统计每个聚类中各类别的数量
    cluster_stats = defaultdict(lambda: defaultdict(int))
    for img, cluster in pred_clusters.items():
        if img in true_labels:
            label = true_labels[img]
            cluster_stats[cluster][label] += 1
    
    # 为每个聚类分配出现最多的类别
    for cluster, label_counts in cluster_stats.items():
        main_label = max(label_counts.items(), key=lambda x: x[1])[0]
        cluster_to_label[cluster] = main_label
    
    return cluster_to_label

def compare_results(pred_file, true_file):
    """对比预测结果和标准答案"""
    
    # 加载数据
    pred_clusters = load_json(pred_file)
    true_labels = load_json(true_file)
    
    print("=" * 70)
    print("聚类结果对比分析")
    print("=" * 70)
    
    # 基本信息
    print(f"\n预测文件: {os.path.basename(pred_file)}")
    print(f"标准答案: {os.path.basename(true_file)}")
    print(f"总图片数: {len(pred_clusters)}")
    
    # 获取聚类映射
    cluster_to_label = map_clusters_to_labels(pred_clusters, true_labels)
    
    print("\n聚类编号到类别的映射:")
    for cluster_id in sorted(cluster_to_label.keys()):
        print(f"  Cluster {cluster_id} -> {cluster_to_label[cluster_id]}")
    
    # 准备数据用于计算指标
    true_label_list = []
    pred_cluster_list = []
    
    for img in sorted(pred_clusters.keys()):
        if img in true_labels:
            true_label_list.append(true_labels[img])
            pred_cluster_list.append(pred_clusters[img])
    
    # 计算评估指标
    ari = adjusted_rand_score(true_label_list, pred_cluster_list)
    nmi = normalized_mutual_info_score(true_label_list, pred_cluster_list)
    
    print("\n" + "=" * 70)
    print("评估指标")
    print("=" * 70)
    print(f"Adjusted Rand Index (ARI):           {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    
    # 计算准确率（基于映射后的标签）
    correct = 0
    total = len(pred_cluster_list)
    
    for i, (true_label, pred_cluster) in enumerate(zip(true_label_list, pred_cluster_list)):
        mapped_label = cluster_to_label.get(pred_cluster, "unknown")
        if mapped_label == true_label:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"映射后准确率 (Accuracy):              {accuracy:.4f} ({correct}/{total})")
    
    # 混淆矩阵
    print("\n" + "=" * 70)
    print("混淆矩阵 (真实标签 vs 聚类)")
    print("=" * 70)
    
    df = pd.DataFrame({
        'True Label': true_label_list,
        'Predicted Cluster': pred_cluster_list
    })
    
    cross_tab = pd.crosstab(
        df['True Label'], 
        df['Predicted Cluster'],
        rownames=['True Label'],
        colnames=['Cluster']
    )
    print(cross_tab)
    
    # 每个类别的统计
    print("\n" + "=" * 70)
    print("每个类别的聚类分布")
    print("=" * 70)
    
    label_stats = defaultdict(lambda: defaultdict(int))
    for true_label, pred_cluster in zip(true_label_list, pred_cluster_list):
        label_stats[true_label][pred_cluster] += 1
    
    for label in sorted(label_stats.keys()):
        total_count = sum(label_stats[label].values())
        print(f"\n{label} (共 {total_count} 张):")
        for cluster in sorted(label_stats[label].keys()):
            count = label_stats[label][cluster]
            percentage = (count / total_count) * 100
            print(f"  Cluster {cluster}: {count:3d} 张 ({percentage:5.1f}%)")
    
    # 错误分类详情
    print("\n" + "=" * 70)
    print("错误分类详情")
    print("=" * 70)
    
    errors = []
    for img in sorted(pred_clusters.keys()):
        if img in true_labels:
            true_label = true_labels[img]
            pred_cluster = pred_clusters[img]
            mapped_label = cluster_to_label.get(pred_cluster, "unknown")
            
            if mapped_label != true_label:
                errors.append({
                    'image': img,
                    'true_label': true_label,
                    'pred_cluster': pred_cluster,
                    'mapped_label': mapped_label
                })
    
    if errors:
        print(f"\n发现 {len(errors)} 个错误分类:")
        for i, error in enumerate(errors[:20], 1):  # 只显示前20个
            print(f"  {i}. {error['image']}: "
                  f"真实={error['true_label']}, "
                  f"预测聚类={error['pred_cluster']} (映射为 {error['mapped_label']})")
        
        if len(errors) > 20:
            print(f"  ... 还有 {len(errors) - 20} 个错误未显示")
    else:
        print("\n✓ 完美！没有错误分类！")
    
    print("\n" + "=" * 70)
    
    # 保存详细对比结果
    output_file = os.path.join(
        os.path.dirname(pred_file),
        'comparison_report.json'
    )
    
    report = {
        'metrics': {
            'ARI': float(ari),
            'NMI': float(nmi),
            'accuracy': float(accuracy),
            'correct': correct,
            'total': total
        },
        'cluster_mapping': cluster_to_label,
        'errors': errors
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"详细报告已保存到: {output_file}")
    
    # 返回数据用于可视化
    return {
        'true_label_list': true_label_list,
        'pred_cluster_list': pred_cluster_list,
        'cluster_to_label': cluster_to_label,
        'cross_tab': cross_tab,
        'metrics': {'ARI': ari, 'NMI': nmi, 'accuracy': accuracy}
    }

def visualize_results(data, output_dir):
    """生成可视化图表"""
    
    true_label_list = data['true_label_list']
    pred_cluster_list = data['pred_cluster_list']
    cluster_to_label = data['cluster_to_label']
    cross_tab = data['cross_tab']
    metrics = data['metrics']
    
    # 创建一个大图，包含多个子图
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 混淆矩阵热力图
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': '样本数量'}, ax=ax1)
    ax1.set_title('混淆矩阵 (真实标签 vs 聚类)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('聚类编号', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    
    # 2. 每个类别的聚类分布（堆叠条形图）
    ax2 = plt.subplot(2, 3, 2)
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
    cross_tab_pct.T.plot(kind='bar', stacked=True, ax=ax2, 
                         colormap='Set3', width=0.7)
    ax2.set_title('每个聚类的类别分布 (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('聚类编号', fontsize=12)
    ax2.set_ylabel('百分比 (%)', fontsize=12)
    ax2.legend(title='真实标签', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    
    # 3. 评估指标柱状图
    ax3 = plt.subplot(2, 3, 3)
    metric_names = ['ARI', 'NMI', 'Accuracy']
    metric_values = [metrics['ARI'], metrics['NMI'], metrics['accuracy']]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax3.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylim([0, 1.1])
    ax3.set_title('评估指标', fontsize=14, fontweight='bold')
    ax3.set_ylabel('分数', fontsize=12)
    ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='完美分数')
    
    # 在柱子上显示数值
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 每个类别的样本数量
    ax4 = plt.subplot(2, 3, 4)
    label_counts = pd.Series(true_label_list).value_counts().sort_index()
    colors_cat = plt.cm.Set2(np.linspace(0, 1, len(label_counts)))
    bars = ax4.barh(label_counts.index, label_counts.values, color=colors_cat, 
                    edgecolor='black', alpha=0.8)
    ax4.set_title('每个类别的样本数量', fontsize=14, fontweight='bold')
    ax4.set_xlabel('样本数量', fontsize=12)
    ax4.set_ylabel('类别', fontsize=12)
    
    # 在条形上显示数值
    for i, (bar, value) in enumerate(zip(bars, label_counts.values)):
        ax4.text(value + 2, bar.get_y() + bar.get_height()/2,
                f'{value}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. 聚类映射关系
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # 创建映射表格
    mapping_data = []
    for cluster_id in sorted(cluster_to_label.keys()):
        mapping_data.append([f'Cluster {cluster_id}', cluster_to_label[cluster_id]])
    
    table = ax5.table(cellText=mapping_data,
                     colLabels=['聚类编号', '对应类别'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置单元格样式
    for i in range(1, len(mapping_data) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    ax5.set_title('聚类到类别的映射关系', fontsize=14, fontweight='bold', pad=20)
    
    # 6. 聚类纯度饼图
    ax6 = plt.subplot(2, 3, 6)
    
    # 计算每个聚类的纯度
    cluster_purity = []
    cluster_labels = []
    for cluster_id in sorted(cluster_to_label.keys()):
        cluster_mask = [pred == cluster_id for pred in pred_cluster_list]
        cluster_true_labels = [true_label_list[i] for i, mask in enumerate(cluster_mask) if mask]
        
        if cluster_true_labels:
            main_label = cluster_to_label[cluster_id]
            purity = cluster_true_labels.count(main_label) / len(cluster_true_labels)
            cluster_purity.append(purity * 100)
            cluster_labels.append(f'C{cluster_id}\n{main_label}')
    
    colors_pie = plt.cm.Pastel1(np.linspace(0, 1, len(cluster_purity)))
    wedges, texts, autotexts = ax6.pie(cluster_purity, labels=cluster_labels,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors_pie, textprops={'fontsize': 10})
    
    # 加粗百分比文字
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax6.set_title('各聚类的纯度分布', fontsize=14, fontweight='bold')
    
    # 总标题
    fig.suptitle('聚类结果可视化分析报告', fontsize=18, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    output_path = os.path.join(output_dir, 'clustering_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存到: {output_path}")
    
    # 显示图片（可选）
    # plt.show()
    plt.close()
    
    # 生成单独的混淆矩阵高清图
    fig2, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': '样本数量'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('混淆矩阵 - 聚类结果 vs 真实标签', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('预测聚类编号', fontsize=13)
    ax.set_ylabel('真实类别标签', fontsize=13)
    
    plt.tight_layout()
    confusion_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {confusion_path}")
    plt.close()

def main():
    # 文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_file = os.path.join(base_dir, 'clustering_results.json')
    true_file = os.path.abspath(os.path.join(base_dir, '../Cluster/cluster_labels.json'))
    
    # 检查文件是否存在
    if not os.path.exists(pred_file):
        print(f"错误: 预测文件不存在: {pred_file}")
        return
    
    if not os.path.exists(true_file):
        print(f"错误: 标准答案文件不存在: {true_file}")
        return
    
    # 执行对比
    data = compare_results(pred_file, true_file)
    
    # 生成可视化
    print("\n" + "=" * 70)
    print("生成可视化图表...")
    print("=" * 70)
    visualize_results(data, base_dir)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
