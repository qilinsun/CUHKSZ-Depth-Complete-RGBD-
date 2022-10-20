name="EMDC-validation"
metric="visual"
data="data0929"
mkdir -p ./results/${name}/${metric}
mkdir ./results/${name}/${metric}/${data}
python3 -u StoDense_plt.py --arch EMDC \
--txt_path ./validation_input/${data}/data.list \
--out_path ${name}/${metric}/${data} \
--visualization 1 \
--seemap 0 \
--output_num 3 \
--readme_path ${name}/${metric} \
--ckp_path ../checkpoints/milestone.pth.tar \
--gpu 0

