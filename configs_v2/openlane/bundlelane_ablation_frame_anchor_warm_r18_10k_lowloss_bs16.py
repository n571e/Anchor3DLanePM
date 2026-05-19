_base_ = ['./bundlelane_ablation_frame_anchor_warm_r18_10k_bs16.py']

model = dict(
    bundle_cfg=dict(
        frame_x_loss_weight=0.05,
        frame_h_loss_weight=0.05,
        frame_bank_loss_weight=0.02,
        frame_smooth_loss_weight=0.02,
    ),
)

work_dir = 'output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_lowloss_bs16'
