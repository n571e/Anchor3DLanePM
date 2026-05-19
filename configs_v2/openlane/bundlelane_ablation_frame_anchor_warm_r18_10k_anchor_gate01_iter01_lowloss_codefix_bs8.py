_base_ = ['./bundlelane_ablation_frame_anchor_warm_r18_10k_lowloss_codefix_bs8.py']

model = dict(
    bundle_cfg=dict(
        inject_iters=[0, 1],
        inject_pre_prior=False,
        use_anchor_gate=True,
        anchor_gate_init_bias=-2.1972245773362196,
        anchor_gate_min=0.0,
        anchor_gate_max=1.0,
    ),
)

work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_anchor_gate01_iter01_lowloss_codefix_bs8'
