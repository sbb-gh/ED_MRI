
# TODO Modify the below
data_fil=""
out_base=""
proj_name="JOFSTO_noddi_simulations"

# TODO Activate environment


total_epochs=10000; epochs_fix_sigma=25; epochs_decay_sigma=10; epochs_decay=10
C_i_values='3612 1806 903 452 226'; C_i_eval='1806 903 452 226'
learning_rate=1E-4

SEED=0
name="$proj_name"

PYTHONHASHSEED="$SEED" python ./jofsto_code/jofsto_main.py \
--data_fil "$data_fil" \
--data_train_subjs train \
--data_val_subjs val \
--data_test_subjs "test" \
--out_base "$scratch_output_dir" \
--total_epochs $total_epochs \
--epochs_decay $epochs_decay \
--epochs_fix_sigma $epochs_fix_sigma \
--epochs_decay_sigma $epochs_decay_sigma \
--learning_rate $learning_rate \
--batch_size 1500 \
--random_seed_value $SEED \
--proj_name $proj_name \
--run_name $name \
--C_i_values $C_i_values \
--C_i_eval $C_i_eval \
--num_units_score $num_units_score \
--num_units_task $num_units_task

date
echo "EOF""$0"
