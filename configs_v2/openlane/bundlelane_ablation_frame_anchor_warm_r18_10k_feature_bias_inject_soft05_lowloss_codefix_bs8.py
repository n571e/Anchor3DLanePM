_base_ = ['./bundlelane_ablation_frame_anchor_warm_r18_10k_lowloss_codefix_bs8.py']

model = dict(
    bundle_cfg=dict(
        inject_strength=0.5,
        use_feature_bias=True,
        feature_bias_scale=0.1,
    ),
)

work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_feature_bias_inject_soft05_lowloss_codefix_bs8'
