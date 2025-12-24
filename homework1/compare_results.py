import json
import os
from collections import defaultdict
import numpy as np
import argparse
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

def resolve_cluster_labels_path(base_dir, labels_path=None):
    if labels_path is not None:
        labels_path = os.path.abspath(labels_path)
        if os.path.isfile(labels_path):
            return labels_path
        return labels_path

    probe_dir = os.path.abspath(base_dir)
    for _ in range(10):
        candidate_labels = os.path.join(probe_dir, 'Cluster', 'cluster_labels.json')
        if os.path.isfile(candidate_labels):
            return os.path.abspath(candidate_labels)

        parent = os.path.dirname(probe_dir)
        if parent == probe_dir:
            break
        probe_dir = parent

    return None

def load_json(filepath):
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_feature_files(base_dir):
    features_pca_path = os.path.join(base_dir, 'features_pca.npy')
    features_2d_path = os.path.join(base_dir, 'features_2d.npy')
    image_files_path = os.path.join(base_dir, 'image_files.json')

    if not (os.path.exists(features_pca_path) and os.path.exists(features_2d_path) and os.path.exists(image_files_path)):
        return None

    features_pca = np.load(features_pca_path)
    features_2d = np.load(features_2d_path)
    image_files = load_json(image_files_path)

    if len(image_files) != features_pca.shape[0] or len(image_files) != features_2d.shape[0]:
        raise ValueError('Feature files and image_files.json have inconsistent lengths.')

    return {
        'features_pca': features_pca,
        'features_2d': features_2d,
        'image_files': image_files,
    }

def align_features_to_pred_order(feature_data, pred_img_order):
    img_to_idx = {img: i for i, img in enumerate(feature_data['image_files'])}
    idxs = []
    for img in pred_img_order:
        if img not in img_to_idx:
            raise KeyError(f'Missing image in feature index: {img}')
        idxs.append(img_to_idx[img])

    return {
        'features_pca': feature_data['features_pca'][idxs],
        'features_2d': feature_data['features_2d'][idxs],
    }

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
    
    pred_img_order = []
    for img in sorted(pred_clusters.keys()):
        if img in true_labels:
            true_label_list.append(true_labels[img])
            pred_cluster_list.append(pred_clusters[img])
            pred_img_order.append(img)
    
    # 计算评估指标
    ari = adjusted_rand_score(true_label_list, pred_cluster_list)
    nmi = normalized_mutual_info_score(true_label_list, pred_cluster_list)

    homogeneity = homogeneity_score(true_label_list, pred_cluster_list)
    completeness = completeness_score(true_label_list, pred_cluster_list)
    v_measure = v_measure_score(true_label_list, pred_cluster_list)
    fmi = fowlkes_mallows_score(true_label_list, pred_cluster_list)
    
    print("\n" + "=" * 70)
    print("评估指标")
    print("=" * 70)
    print(f"Adjusted Rand Index (ARI):           {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Homogeneity:                         {homogeneity:.4f}")
    print(f"Completeness:                        {completeness:.4f}")
    print(f"V-measure:                           {v_measure:.4f}")
    print(f"Fowlkes-Mallows Index (FMI):         {fmi:.4f}")
    
    # 计算准确率（基于映射后的标签）
    correct = 0
    total = len(pred_cluster_list)
    
    for i, (true_label, pred_cluster) in enumerate(zip(true_label_list, pred_cluster_list)):
        mapped_label = cluster_to_label.get(pred_cluster, "unknown")
        if mapped_label == true_label:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"映射后准确率 (Accuracy):              {accuracy:.4f} ({correct}/{total})")

    internal_metrics = {}
    base_dir = os.path.dirname(os.path.abspath(pred_file))
    feature_data = load_feature_files(base_dir)
    if feature_data is not None:
        aligned = align_features_to_pred_order(feature_data, pred_img_order)

        X = aligned['features_pca']
        y = np.array(pred_cluster_list)

        internal_metrics = {
            'silhouette': float(silhouette_score(X, y)),
            'calinski_harabasz': float(calinski_harabasz_score(X, y)),
            'davies_bouldin': float(davies_bouldin_score(X, y)),
        }

        print("\n" + "=" * 70)
        print("内部指标（仅使用特征与聚类结果）")
        print("=" * 70)
        print(f"Silhouette Coefficient:              {internal_metrics['silhouette']:.4f}")
        print(f"Calinski-Harabasz Index:             {internal_metrics['calinski_harabasz']:.4f}")
        print(f"Davies-Bouldin Index:                {internal_metrics['davies_bouldin']:.4f}")
    else:
        print("\n提示：未找到 features_pca.npy / features_2d.npy / image_files.json，将跳过内部指标与二维散点图。")
    
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
            'homogeneity': float(homogeneity),
            'completeness': float(completeness),
            'v_measure': float(v_measure),
            'fmi': float(fmi),
            'accuracy': float(accuracy),
            'correct': correct,
            'total': total
        },
        'internal_metrics': internal_metrics,
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
        'pred_img_order': pred_img_order,
        'cluster_to_label': cluster_to_label,
        'cross_tab': cross_tab,
        'metrics': {
            'ARI': ari,
            'NMI': nmi,
            'accuracy': accuracy,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure,
            'fmi': fmi,
        },
        'internal_metrics': internal_metrics,
        'feature_data': feature_data,
    }

def visualize_results(data, output_dir):
    """生成可视化图表"""
    
    true_label_list = data['true_label_list']
    pred_cluster_list = data['pred_cluster_list']
    cluster_to_label = data['cluster_to_label']
    cross_tab = data['cross_tab']
    metrics = data['metrics']
    internal_metrics = data.get('internal_metrics', {})
    pred_img_order = data.get('pred_img_order', [])

    feature_data = data.get('feature_data')
    aligned_2d = None
    if feature_data is not None and pred_img_order:
        aligned_2d = align_features_to_pred_order(feature_data, pred_img_order).get('features_2d')

    df_scatter = None
    if aligned_2d is not None:
        df_scatter = pd.DataFrame({
            'x': aligned_2d[:, 0],
            'y': aligned_2d[:, 1],
            'true_label': true_label_list,
            'cluster': pred_cluster_list,
        })
        df_scatter['cluster'] = df_scatter['cluster'].astype(str)

    fig_dist, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax_tl, ax_tr = axes[0, 0], axes[0, 1]
    ax_bl, ax_br = axes[1, 0], axes[1, 1]

    if df_scatter is not None:
        sns.scatterplot(
            data=df_scatter,
            x='x',
            y='y',
            hue='true_label',
            palette='tab10',
            s=24,
            alpha=0.85,
            edgecolor=None,
            ax=ax_tl,
        )
        ax_tl.set_title('真实标签分布图', fontsize=14, fontweight='bold')
        ax_tl.set_xlabel('PC1')
        ax_tl.set_ylabel('PC2')
        ax_tl.grid(alpha=0.2)
        ax_tl.legend(title='真实类别', bbox_to_anchor=(1.02, 1), loc='upper left')

        sns.scatterplot(
            data=df_scatter,
            x='x',
            y='y',
            hue='cluster',
            palette='tab10',
            s=24,
            alpha=0.85,
            edgecolor=None,
            ax=ax_tr,
        )
        ax_tr.set_title('聚类结果分布图', fontsize=14, fontweight='bold')
        ax_tr.set_xlabel('PC1')
        ax_tr.set_ylabel('PC2')
        ax_tr.grid(alpha=0.2)
        ax_tr.legend(title='聚类编号', bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        ax_tl.axis('off')
        ax_tr.axis('off')
        ax_tl.text(0.5, 0.5, '未找到 features_2d.npy\n跳过二维散点图', ha='center', va='center', fontsize=14)

    labels_sorted = sorted(pd.Series(true_label_list).unique())
    true_counts = pd.Series(true_label_list).value_counts().reindex(labels_sorted).fillna(0).astype(int)
    mapped_pred_labels = [cluster_to_label.get(c, 'unknown') for c in pred_cluster_list]
    pred_counts = pd.Series(mapped_pred_labels).value_counts().reindex(labels_sorted).fillna(0).astype(int)

    x = np.arange(len(labels_sorted))
    width = 0.38
    ax_bl.bar(x - width / 2, true_counts.values, width=width, label='真实数量', color='#1f77b4', alpha=0.85)
    ax_bl.bar(x + width / 2, pred_counts.values, width=width, label='聚类分配数量', color='#ff7f0e', alpha=0.85)
    ax_bl.set_title('类别分布对比', fontsize=14, fontweight='bold')
    ax_bl.set_xlabel('类别')
    ax_bl.set_ylabel('样本数量')
    ax_bl.set_xticks(x)
    ax_bl.set_xticklabels(labels_sorted, rotation=20)
    ax_bl.legend()
    ax_bl.grid(axis='y', alpha=0.25)

    cross_tab_for_plot = cross_tab.copy()
    cross_tab_for_plot = cross_tab_for_plot.reindex(index=labels_sorted)
    cross_tab_for_plot = cross_tab_for_plot.reindex(columns=sorted(cross_tab_for_plot.columns))
    sns.heatmap(
        cross_tab_for_plot,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': '样本数量'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax_br,
    )
    ax_br.set_title('混淆矩阵', fontsize=14, fontweight='bold')
    ax_br.set_xlabel('聚类编号')
    ax_br.set_ylabel('真实标签')

    fig_dist.suptitle('聚类分布图', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dist_path = os.path.join(output_dir, 'clustering_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"聚类分布图已保存到: {dist_path}")
    plt.close(fig_dist)
    
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

    if df_scatter is not None:
        features_2d = df_scatter[['x', 'y']].to_numpy()

        fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.scatterplot(
            data=df_scatter,
            x='x',
            y='y',
            hue='true_label',
            palette='tab10',
            s=30,
            alpha=0.85,
            edgecolor=None,
            ax=axes[0],
        )
        axes[0].set_title('PCA 2D 分布（按真实类别着色）', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].grid(alpha=0.2)
        axes[0].legend(title='真实类别', bbox_to_anchor=(1.02, 1), loc='upper left')

        sns.scatterplot(
            data=df_scatter,
            x='x',
            y='y',
            hue='cluster',
            palette='tab10',
            s=30,
            alpha=0.85,
            edgecolor=None,
            ax=axes[1],
        )
        axes[1].set_title('PCA 2D 分布（按聚类编号着色）', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].grid(alpha=0.2)
        axes[1].legend(title='聚类编号', bbox_to_anchor=(1.02, 1), loc='upper left')

        subtitle = ''
        if internal_metrics:
            subtitle = (
                f"内部指标：Silhouette={internal_metrics.get('silhouette', float('nan')):.4f}  "
                f"CH={internal_metrics.get('calinski_harabasz', float('nan')):.1f}  "
                f"DBI={internal_metrics.get('davies_bouldin', float('nan')):.4f}"
            )

        if subtitle:
            fig3.suptitle(subtitle, fontsize=12, y=1.02)

        plt.tight_layout()
        scatter_path = os.path.join(output_dir, 'embedding_distribution.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"二维散点分布图已保存到: {scatter_path}")
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, default=None)
    parser.add_argument('--labels', type=str, default=None)
    args = parser.parse_args()

    pred_file = args.pred or os.path.join(base_dir, 'clustering_results.json')
    if not os.path.isabs(pred_file):
        pred_file = os.path.abspath(os.path.join(base_dir, pred_file))

    true_file = resolve_cluster_labels_path(base_dir=base_dir, labels_path=args.labels)
    
    # 检查文件是否存在
    if not os.path.exists(pred_file):
        print(f"错误: 预测文件不存在: {pred_file}")
        return
    
    if not true_file or not os.path.exists(true_file):
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
