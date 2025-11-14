
from diffusers.models.attention_processor import Attention


def _get_projections(attn: Attention, hidden_states, encoder_hidden_states):
    img_q = attn.to_q(hidden_states)
    img_k = attn.to_k(hidden_states)
    img_v = attn.to_v(hidden_states)

    txt_q = attn.add_q_proj(encoder_hidden_states)
    txt_k = attn.add_k_proj(encoder_hidden_states)
    txt_v = attn.add_v_proj(encoder_hidden_states)

    return img_q, img_k, img_v, txt_q, txt_k, txt_v


def _get_fused_projections(attn: Attention, hidden_states, encoder_hidden_states):
    img_q, img_k, img_v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    txt_q, txt_k, txt_v = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
    return img_q, img_k, img_v, txt_q, txt_k, txt_v


def attn_get_qkv_projections(attn: Attention, hidden_states, encoder_hidden_states):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)