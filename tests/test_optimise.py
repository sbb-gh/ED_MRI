from optimise import optimise_experiment

base_dir = "/Users/scmps8/Data/wand/WAND/sub-01187/ses-02/dwi"

result = optimise_experiment(
    input_data=f"{base_dir}/sub-01187_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.nii.gz",
    target_data=f"{base_dir}/tensor/dt.nii.gz",
    superdesign=f"{base_dir}/sub-01187_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.grad",
    mask=f"{base_dir}/mask_test.nii.gz",
    opt_protocol_size=0.5,
    output_dir=f"{base_dir}/edmri_output",
)


