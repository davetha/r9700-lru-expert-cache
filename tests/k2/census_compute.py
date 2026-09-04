#!/usr/bin/env python3
import json, re, sys
from collections import defaultdict

d = json.load(open('$REPO_ROOT/artifacts/tensor_headers.json'))
TP = 2

def rank_bytes(name, full_bytes, split):
    # split: 'col' (dim0/N split), 'row' (dim1/K split), 'rep' (replicated), 'half' (generic /TP)
    if split in ('col', 'row', 'half'):
        return full_bytes / TP
    return full_bytes

CAT_BYTES = defaultdict(float)
CAT_COUNT = defaultdict(int)
CAT_EXAMPLES = defaultdict(list)
UNCLASSIFIED = []

def classify(name):
    # MTP branch first (checked separately so it doesn't fall into main-layer buckets)
    if name.startswith('mtp.'):
        if '.mlp.experts.' in name:
            return 'mtp_experts', 'col_row_moe'   # gate/up=col(N split), down=row(K split) -> treat generically as half
        if '.self_attn.' in name:
            return 'mtp_module_attn', 'attn'
        if '.mlp.shared_expert.' in name or 'shared_expert_gate' in name:
            return 'mtp_module_shared', 'shared'
        if '.mlp.gate.weight' in name:
            return 'mtp_module_router', 'rep'
        if 'hyper_connection' in name:
            return 'mtp_module_hc', 'rep'
        if name in ('mtp.fc_embedding.weight', 'mtp.fc_hidden.weight'):
            return 'mtp_module_fc', 'col'
        if 'pre_fc_norm' in name:
            return 'mtp_module_norm', 'rep'
        return 'mtp_module_other', 'rep'

    if name.startswith('model.visual.') or name == 'model.visual.pos_embed.weight':
        if '.attn.qkv.' in name or '.mlp.linear_fc1.' in name:
            return 'vision_tower', 'col'
        if '.attn.proj.' in name or '.mlp.linear_fc2.' in name:
            return 'vision_tower', 'row'
        return 'vision_tower', 'rep'

    if name == 'lm_head.weight':
        return 'lm_head', 'col'  # VocabParallel split on vocab dim

    if name == 'model.language_model.embed_tokens.weight':
        return 'embed_tokens', 'col'

    if name.startswith('model.language_model.layers.'):
        if '.mlp.experts.' in name:
            return 'routed_experts_mxfp4', 'col_row_moe'
        if '.mlp.shared_expert.' in name or 'shared_expert_gate' in name:
            return 'shared_experts', 'shared'
        if '.mlp.gate.weight' in name:
            return 'moe_router', 'rep'
        if 'hyper_connection' in name:
            return 'hyper_connection', 'rep'
        if '.self_attn.indexer.' in name:
            return 'indexer', 'rep'
        if '.self_attn.' in name:
            if '.q_norm.' in name or '.k_norm.' in name:
                return 'self_attn_norm', 'rep'
            return 'self_attn_fp8', 'attn'
        if '.linear_attn.' in name:
            return 'linear_attn', 'gdn'
        if '.ple.' in name:
            return 'ple_host_offload', 'rep'
        return 'main_layer_other', 'rep'

    if name.startswith('model.language_model.hyper_connection_mixer'):
        return 'hyper_connection', 'rep'

    return 'UNCLASSIFIED', 'rep'


for name, info in d.items():
    cat, split = classify(name)
    if cat == 'UNCLASSIFIED':
        UNCLASSIFIED.append(name)
        continue
    full_bytes = info['bytes']
    if split == 'attn':
        # q_proj/k_proj/v_proj -> col (N split); o_proj -> row (K split); scale tensors follow weight's split
        base = name.replace('.weight_scale_inv', '.weight')
        if '.o_proj.' in base:
            rb = full_bytes / TP
        else:
            rb = full_bytes / TP
    elif split == 'gdn':
        # in_proj_qkv/in_proj_z/conv1d/A_log/dt_bias -> col-ish split by TP; out_proj -> row split; norm uncertain -> treat as split too (per-head)
        rb = full_bytes / TP
    elif split == 'shared':
        if 'shared_expert_gate' in name:
            rb = full_bytes  # tiny ReplicatedLinear
        else:
            rb = full_bytes / TP
    elif split in ('col_row_moe',):
        rb = full_bytes / TP
    else:
        rb = rank_bytes(name, full_bytes, split)
    CAT_BYTES[cat] += rb
    CAT_COUNT[cat] += 1
    if len(CAT_EXAMPLES[cat]) < 3:
        CAT_EXAMPLES[cat].append((name, info['dtype'], info['shape']))

GIB = 1024**3
print(f"{'category':28s} {'tensors':>8s} {'GiB/rank':>10s}")
total = 0.0
for cat in sorted(CAT_BYTES, key=lambda c: -CAT_BYTES[c]):
    gib = CAT_BYTES[cat] / GIB
    total += CAT_BYTES[cat]
    print(f"{cat:28s} {CAT_COUNT[cat]:8d} {gib:10.4f}")
print(f"{'TOTAL':28s} {sum(CAT_COUNT.values()):8d} {total/GIB:10.4f}")
print()
print('unclassified tensor count:', len(UNCLASSIFIED))
for u in UNCLASSIFIED[:20]:
    print('  ', u)
print()
for cat, exs in CAT_EXAMPLES.items():
    print(cat, '::')
    for e in exs:
        print('   ', e)
