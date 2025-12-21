"""
Training Loss Functions

Implements various loss functions for multimodal cognitive training.

Losses:
1. Language Modeling: Cross-entropy on next token prediction
2. Contrastive: InfoNCE for image-text alignment (CLIP-style)
3. Multimodal Alignment: L2 distance in shared embedding space
4. Field Regularization: Entropy penalties on field evolution
5. Band Regularization: Soft penalty for A, B, Θ generator norms outside healthy range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


def language_modeling_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Standard language modeling loss (cross-entropy).

    Args:
        logits: (batch, seq_len, vocab_size) model predictions
        labels: (batch, seq_len) target token IDs
        attention_mask: (batch, seq_len) mask (1 = valid, 0 = padding)
        label_smoothing: Label smoothing factor (0.0 - 0.1 typical)

    Returns:
        loss: Scalar loss value
    """
    batch_size, seq_len, vocab_size = logits.shape
    if labels.shape[:2] != (batch_size, seq_len):
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match logits batch/seq {(batch_size, seq_len)}"
        )

    # Standard causal LM: predict token t+1 from logits at position t
    # logits: (B, T, V) -> (B, T-1, V)
    # labels: (B, T)    -> (B, T-1)
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Flatten logits and labels
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # Create mask
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous().view(-1).float()
    else:
        shift_mask = torch.ones_like(shift_labels, dtype=torch.float32)

    # Cross-entropy loss
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        reduction='none',
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )

    # Apply mask and ensure ignored positions do not affect denominator
    valid = (shift_labels != ignore_index).float()
    weight = shift_mask * valid
    denom = weight.sum().clamp_min(1.0)
    loss = (loss * weight).sum() / denom

    return loss


def contrastive_loss(
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Contrastive loss for text-image alignment (CLIP-style).

    Uses InfoNCE (NT-Xent) loss:
    - Maximize similarity between matched pairs
    - Minimize similarity between unmatched pairs

    Args:
        text_embeddings: (batch, d_model) text representations
        image_embeddings: (batch, d_model) image representations
        temperature: Softmax temperature (0.07 typical for CLIP)

    Returns:
        loss: Scalar contrastive loss
    """
    batch_size = text_embeddings.shape[0]
    device = text_embeddings.device

    # Normalize embeddings
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    image_embeddings = F.normalize(image_embeddings, dim=-1)

    # Compute similarity matrix: (batch, batch)
    # similarity[i, j] = text[i] · image[j]
    similarity = torch.matmul(text_embeddings, image_embeddings.T) / temperature

    # Labels: positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=device)

    # Loss in both directions
    loss_text_to_image = F.cross_entropy(similarity, labels)
    loss_image_to_text = F.cross_entropy(similarity.T, labels)

    # Average both directions
    loss = (loss_text_to_image + loss_image_to_text) / 2

    return loss


def multimodal_alignment_loss(
    embeddings: Dict[str, torch.Tensor],
    margin: float = 1.0
) -> torch.Tensor:
    """
    Multimodal alignment loss.

    Encourages embeddings from different modalities to align
    in shared d_model space.

    Uses triplet-style loss:
    - Anchor: text embedding
    - Positive: matched image/video
    - Negative: random other image/video

    Args:
        embeddings: Dict with keys ['text', 'image', 'video']
                   Each is (batch, d_model)
        margin: Triplet margin

    Returns:
        loss: Scalar alignment loss
    """
    if 'text' not in embeddings:
        return torch.tensor(0.0, device=list(embeddings.values())[0].device)

    text_emb = embeddings['text']
    batch_size = text_emb.shape[0]
    device = text_emb.device

    total_loss = 0.0
    n_pairs = 0

    # Text-Image alignment
    if 'image' in embeddings:
        image_emb = embeddings['image']

        # Positive pairs: (text[i], image[i])
        pos_dist = F.pairwise_distance(text_emb, image_emb)

        # Negative pairs: (text[i], image[j]) where j != i
        # Create negative samples by shifting
        neg_image = torch.roll(image_emb, shifts=1, dims=0)
        neg_dist = F.pairwise_distance(text_emb, neg_image)

        # Triplet margin loss
        loss_ti = F.relu(pos_dist - neg_dist + margin).mean()
        total_loss += loss_ti
        n_pairs += 1

    # Text-Video alignment
    if 'video' in embeddings:
        video_emb = embeddings['video']

        pos_dist = F.pairwise_distance(text_emb, video_emb)
        neg_video = torch.roll(video_emb, shifts=1, dims=0)
        neg_dist = F.pairwise_distance(text_emb, neg_video)

        loss_tv = F.relu(pos_dist - neg_dist + margin).mean()
        total_loss += loss_tv
        n_pairs += 1

    if n_pairs > 0:
        total_loss = total_loss / n_pairs

    return total_loss


def band_regularizer(
    vec: torch.Tensor,
    low: float = 0.7,
    high: float = 1.4
) -> torch.Tensor:
    """
    Soft penalty for vector norms outside healthy band [low, high].

    Used for A, B, Θ generators in the exponential phase extractor.
    Unlike unit norm regularization, this allows norms to vary within
    a healthy range rather than forcing exactly 1.0.

    Args:
        vec: Input tensor (..., dim) - computes norm over last dimension
        low: Lower bound of healthy band (default 0.7)
        high: Upper bound of healthy band (default 1.4)

    Returns:
        Scalar penalty (0 if all norms in band, positive otherwise)

    Example:
        A_gen, B_gen, Theta_gen = extractor.get_generators(x)
        reg_loss = (band_regularizer(A_gen) +
                    band_regularizer(B_gen) +
                    band_regularizer(Theta_gen))
    """
    norm = vec.norm(dim=-1)  # (...,)
    too_small = F.relu(low - norm)   # Penalty if norm < low
    too_big = F.relu(norm - high)    # Penalty if norm > high
    return (too_small + too_big).mean()


def generator_band_regularizer(
    A_gen: torch.Tensor,
    B_gen: torch.Tensor,
    Theta_gen: torch.Tensor,
    low: float = 0.7,
    high: float = 1.4,
    theta_clamp: float = 5.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined band regularization for all three generators.

    Also applies soft clamping penalty to Θ to discourage values
    that would cause exp(Θ) overflow even before the hard clamp.

    Args:
        A_gen: Real exponential generator (..., 8)
        B_gen: Phase/imaginary generator (..., 8)
        Theta_gen: Rotation-boost generator (..., 8)
        low: Lower norm bound
        high: Upper norm bound
        theta_clamp: Soft penalty threshold for |Θ| components

    Returns:
        total_reg: Combined regularization loss
        reg_dict: Individual components for logging
    """
    reg_A = band_regularizer(A_gen, low, high)
    reg_B = band_regularizer(B_gen, low, high)
    reg_Theta = band_regularizer(Theta_gen, low, high)

    # Additional: penalize Θ components approaching clamp boundary
    # This encourages staying well within [-8, 8] hard clamp
    theta_excess = F.relu(Theta_gen.abs() - theta_clamp).mean()

    total_reg = reg_A + reg_B + reg_Theta + theta_excess

    reg_dict = {
        'band_reg_A': reg_A.item(),
        'band_reg_B': reg_B.item(),
        'band_reg_Theta': reg_Theta.item(),
        'theta_excess': theta_excess.item(),
        'band_reg_total': total_reg.item()
    }

    return total_reg, reg_dict


def field_entropy_regularization(
    field: torch.Tensor,
    target_entropy: Optional[float] = None,
    weight: float = 0.01
) -> torch.Tensor:
    """
    Regularize field entropy during training.

    Encourages field to maintain reasonable entropy levels
    (not too collapsed, not too diffuse).

    Args:
        field: (N_x, N_y, D, D) complex field state
        target_entropy: Target entropy value (if None, penalize extremes)
        weight: Regularization weight

    Returns:
        loss: Scalar regularization term
    """
    # Compute field magnitude distribution
    field_mag = torch.abs(field)  # (N_x, N_y, D, D)

    # Flatten spatial
    field_mag = field_mag.flatten(0, 1)  # (N_x * N_y, D, D)

    # Compute entropy per spatial location
    # H = -Σ p log p where p = |T_ij|² / Σ|T_ij|²
    field_mag_sq = field_mag ** 2
    probs = field_mag_sq / (field_mag_sq.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    probs = probs.flatten(1)  # (N_x * N_y, D*D)

    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    if target_entropy is not None:
        # Penalize deviation from target
        loss = weight * (entropy - target_entropy) ** 2
    else:
        # Penalize very low or very high entropy
        # Encourage moderate entropy around log(D*D) / 2
        D = field.shape[-1]
        mid_entropy = entropy.new_tensor(D * D).log() / 2
        loss = weight * (entropy - mid_entropy) ** 2

    return loss


def combined_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: Optional[Dict[str, float]] = None,
    field_state: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Combined loss for multimodal training.

    Computes weighted sum of:
    - Language modeling loss
    - Contrastive loss (if multimodal)
    - Alignment loss (if multimodal)
    - Field regularization (if provided)

    Args:
        outputs: Dict with model outputs
                 - 'logits': (batch, seq_len, vocab_size)
                 - 'text_embedding': (batch, d_model)
                 - 'image_embedding': (batch, d_model) [optional]
                 - 'video_embedding': (batch, d_model) [optional]
        batch: Dict with batch data
               - 'labels': (batch, seq_len)
               - 'attention_mask': (batch, seq_len)
               - 'modality': str
        loss_weights: Dict of loss component weights
        field_state: Optional field state for regularization

    Returns:
        total_loss: Combined scalar loss
        loss_dict: Dict of individual loss components
    """
    if loss_weights is None:
        loss_weights = {
            'lm': 1.0,
            'contrastive': 0.5,
            'alignment': 0.3,
            'field_reg': 0.01
        }

    loss_dict: Dict[str, torch.Tensor] = {}
    total_loss: Optional[torch.Tensor] = None

    # Language modeling loss (always computed)
    if 'logits' in outputs and 'labels' in batch:
        lm_loss = language_modeling_loss(
            outputs['logits'],
            batch['labels'],
            batch.get('attention_mask')
        )
        loss_dict['lm_loss'] = lm_loss.detach()
        w_lm = lm_loss.new_tensor(loss_weights['lm'])
        total_loss = (total_loss if total_loss is not None else lm_loss.new_zeros(())) + (w_lm * lm_loss)

    # Contrastive loss (for multimodal)
    if 'text_embedding' in outputs:
        if 'image_embedding' in outputs:
            contrast_loss = contrastive_loss(
                outputs['text_embedding'],
                outputs['image_embedding']
            )
            loss_dict['contrastive_loss'] = contrast_loss.detach()
            w_con = contrast_loss.new_tensor(loss_weights['contrastive'])
            total_loss = (total_loss if total_loss is not None else contrast_loss.new_zeros(())) + (w_con * contrast_loss)

        if 'video_embedding' in outputs:
            contrast_loss = contrastive_loss(
                outputs['text_embedding'],
                outputs['video_embedding']
            )
            loss_dict['contrastive_loss'] = contrast_loss.detach()
            w_con = contrast_loss.new_tensor(loss_weights['contrastive'])
            total_loss = (total_loss if total_loss is not None else contrast_loss.new_zeros(())) + (w_con * contrast_loss)

    # Alignment loss (multimodal embeddings)
    embeddings = {}
    if 'text_embedding' in outputs:
        embeddings['text'] = outputs['text_embedding']
    if 'image_embedding' in outputs:
        embeddings['image'] = outputs['image_embedding']
    if 'video_embedding' in outputs:
        embeddings['video'] = outputs['video_embedding']

    if len(embeddings) > 1:
        align_loss = multimodal_alignment_loss(embeddings)
        loss_dict['alignment_loss'] = align_loss.detach()
        w_align = align_loss.new_tensor(loss_weights['alignment'])
        total_loss = (total_loss if total_loss is not None else align_loss.new_zeros(())) + (w_align * align_loss)

    # Field regularization
    if field_state is not None:
        # Field regularization is a monitor/stabilizer by default; avoid building graphs through field ops.
        with torch.no_grad():
            field_reg = field_entropy_regularization(field_state)
        field_reg = field_reg.detach()
        loss_dict['field_entropy'] = field_reg
        if total_loss is None:
            total_loss = field_reg.new_zeros(())
        w_field = total_loss.new_tensor(loss_weights['field_reg'])
        total_loss = total_loss + (w_field * field_reg.to(device=total_loss.device, dtype=total_loss.dtype))

    # Store total loss
    if total_loss is None:
        device = next(iter(outputs.values())).device if outputs else torch.device("cpu")
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
    loss_dict['total_loss'] = total_loss.detach()

    return total_loss, loss_dict
