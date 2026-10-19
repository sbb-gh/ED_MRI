from apply import apply_trained_model
from helpers import extract_optimised_measurements

base_dir = "/Users/scmps8/Data/wand/WAND/sub-01187/ses-02/dwi"

subset, indices = extract_optimised_measurements(
    input_data=f"{base_dir}/sub-01187_ses-02_acq-CHARMED_dir-AP_part-mag_dwi.nii.gz",
    optimised_indices=f"{base_dir}/edmri_output/optimised_indices.txt",
    output_file=f"{base_dir}/edmri_output/optimised_dwi.nii.gz",
)


output = apply_trained_model(
    input_data=f"{base_dir}/edmri_output/optimised_dwi.nii.gz",
    trained_model=f"{base_dir}/edmri_output/2026-09-23_16-53-16/results/def_all_trained_task_network.pt",
    mask=f"{base_dir}/brain_mask.nii.gz",
    output_file=f"{base_dir}/tensor/edmri_predicted_dt.nii.gz",
)