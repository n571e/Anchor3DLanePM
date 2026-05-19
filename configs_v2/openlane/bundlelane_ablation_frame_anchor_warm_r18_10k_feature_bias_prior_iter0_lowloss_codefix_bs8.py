_base_ = ['./bundlelane_ablation_frame_anchor_warm_r18_10k_feature_bias_inject_lowloss_codefix_bs8.py']

model = dict(
    bundle_cfg=dict(
        feature_bias_iters=[0],
        feature_bias_feat_indices=[0],
    ),
)

work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_feature_bias_prior_iter0_lowloss_codefix_bs8'
