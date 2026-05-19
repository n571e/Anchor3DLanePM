_base_ = ['./bundlelane_ablation_frame_anchor_warm_r18_10k_bs8.py']

model = dict(
    bundle_cfg=dict(
        use_inject_gate=True,
        inject_gate_init_bias=-2.1972245773,
        inject_gate_min=0.0,
        inject_gate_max=1.0,
    ),
)

work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_gate01_bs8'
