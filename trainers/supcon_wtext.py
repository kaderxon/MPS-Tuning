from __future__ import print_function

import torch
import torch.nn as nn

class SupConLossWithText(nn.Module):
    """
    监督对比损失，增加了与类别文本特征的对比。
    一个图像样本不仅会与同类别的其他图像样本拉近，还会与其对应类别的文本特征拉近。
    同时，它会与不同类别的图像样本以及不同类别的文本特征推远。
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossWithText, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, text_features=None, text_labels=None):
        """
        计算模型的损失。
        如果提供了 text_features 和 text_labels，则会计算图像-文本混合的监督对比损失。
        如果 text_features 为 None，则退化为原始的监督对比损失。
        
        Args:
            features: 图像的隐藏向量，形状为 [bsz, n_views, ...]。
            labels: 图像的真实标签，形状为 [bsz]。
            text_features: (可选) 每个类别的文本特征，形状为 [num_classes, feature_dim]。
                           这些特征是固定的，不参与训练。
            text_labels: (可选) 文本特征对应的标签，形状为 [num_classes]。
                         例如 tensor([0, 1, 2, ..., num_classes-1])。

        Returns:
            一个表示损失的标量。
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = torch.device('cuda')

        # --- 1. 统一和准备图像特征 ---
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        num_views = features.shape[1]

        if labels is None:
            raise ValueError('`labels` must be provided for supervised learning.')
        
        # 将 labels 变形以进行广播计算
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # 将来自不同 view 的特征拼接在一起
        # 形状从 [bsz, n_views, dim] -> [bsz * n_views, dim]
        image_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # 根据 contrast_mode 选择 anchor
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = image_features
            anchor_count = num_views
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # 同样，为 anchor 准备标签
        # 形状从 [bsz, 1] -> [bsz * anchor_count, 1]
        anchor_labels = labels.repeat(anchor_count, 1)

        # --- 2. 构建对比集合和标签 ---
        if text_features is not None:
            # 如果提供了文本特征，则将它们加入对比集合
            if text_labels is None:
                raise ValueError('`text_labels` must be provided with `text_features`.')
            
            text_labels = text_labels.contiguous().view(-1, 1)
            
            # 对比集合 = 图像特征 + 文本特征
            contrast_feature = torch.cat([image_features, text_features], dim=0)
            
            # 对比集合的标签 = 图像标签 + 文本标签
            # 图像标签需要为每个 view 重复
            image_labels_for_contrast = labels.repeat(num_views, 1)
            contrast_labels = torch.cat([image_labels_for_contrast, text_labels], dim=0)
        else:
            # 如果没有文本特征，则行为和原始 SupConLoss 一致
            contrast_feature = image_features
            contrast_labels = labels.repeat(num_views, 1)

        # --- 3. 构建正样本掩码 (Positives Mask) ---
        # `mask`[i, j] = 1 表示样本 i 和 j 是正样本对
        mask = torch.eq(anchor_labels, contrast_labels.T).float().to(device)

        # --- 4. 计算相似度矩阵 (Logits) ---
        # anchor_feature:      [bsz * anchor_count, dim]
        # contrast_feature.T:  [dim, bsz * num_views (+ num_classes)]
        # anchor_dot_contrast: [bsz * anchor_count, bsz * num_views (+ num_classes)]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # --- 5. 掩码掉自对比情况 ---
        # 对于 anchor i，它不能与自己形成正样本对
        # 在图像-文本模式下，这个逻辑依然成立，因为 anchor 都是图像，
        # 而在 contrast_feature 中，与 anchor 自身对应的位置是确定的。
        # 例如，第 k 个 anchor 对应 contrast_feature 的第 k 个位置。
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # --- 6. 计算损失 ---
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算每个 anchor 的正样本对的 log-likelihood 均值
        # 处理没有正样本对的边缘情况
        mask_pos_pairs = mask.sum(1)
        # 避免除以零
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # 最终损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# --- 使用示例 ---
if __name__ == '__main__':
    # 参数设置
    batch_size = 128
    num_views = 2
    feature_dim = 128
    num_classes = 10

    # 1. 创建假的图像特征和标签
    # 形状: [bsz, n_views, dim]
    dummy_features = torch.randn(batch_size, num_views, feature_dim).cuda()
    # 形状: [bsz]
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).cuda()

    # 2. 创建假的、固定的文本特征和标签
    # 形状: [num_classes, dim]
    dummy_text_features = torch.randn(num_classes, feature_dim).cuda()
    # 形状: [num_classes]
    dummy_text_labels = torch.arange(num_classes).cuda()

    # 初始化损失函数
    loss_fn = SupConLossWithText(temperature=0.1).cuda()

    # --- 场景一：同时使用图像和文本特征 ---
    print("--- 计算图像-文本混合监督对比损失 ---")
    loss_with_text = loss_fn(dummy_features, dummy_labels, 
                             text_features=dummy_text_features, 
                             text_labels=dummy_text_labels)
    print("混合损失:", loss_with_text.item())

    # --- 场景二：仅使用图像特征（退化为原始 SupConLoss） ---
    print("\n--- 计算仅图像的监督对比损失（作为对比） ---")
    loss_without_text = loss_fn(dummy_features, dummy_labels)
    print("仅图像损失:", loss_without_text.item())
    
    # 验证与原始代码行为是否一致
    from supcon_original import SupConLoss # 假设你将原始代码保存在 supcon_original.py
    print("\n--- 与原始 SupConLoss 代码对比 ---")
    original_loss_fn = SupConLoss(temperature=0.1).cuda()
    original_loss = original_loss_fn(dummy_features, dummy_labels)
    print("原始 SupConLoss:", original_loss.item())
    print("与新实现（仅图像模式）的差异:", abs(loss_without_text.item() - original_loss.item()))