DIR="data_1219/dormitory/"

mkdir ${DIR}/aligned_raw_dep
mkdir ${DIR}/aligned_rgb
mkdir ${DIR}/cropped_interpo_dep
mkdir ${DIR}/cropped_rgb
mkdir ${DIR}/infrared
mkdir ${DIR}/infrared_param
mkdir ${DIR}/interpo_depth
mkdir ${DIR}/origin_raw_dep
mkdir ${DIR}/origin_raw_dep_param
mkdir ${DIR}/origin_raw_dep_pgm
mkdir ${DIR}/origin_rgb
mkdir ${DIR}/origin_rgb_param
mkdir ${DIR}/resize_interpo_dep
mkdir ${DIR}/resize_rgb
mkdir ${DIR}/undistorted_raw_dep

mv ${DIR}/*Dep*.pgm ${DIR}/origin_raw_dep_pgm
mv ${DIR}/*Dep*.txt ${DIR}/origin_raw_dep_param
mv ${DIR}/*RGB*.png ${DIR}/origin_rgb
mv ${DIR}/*RGB*.txt ${DIR}/origin_rgb_param
mv ${DIR}/*infrared*.pgm ${DIR}/infrared
mv ${DIR}/*infrared*.txt ${DIR}/infrared_param

python data_preparation/pgm2png.py --batch --input_dir ${DIR}/origin_raw_dep_pgm --output_dir ${DIR}/origin_raw_dep
python data_preparation/undistort.py --batch --input_dir ${DIR}/origin_raw_dep --output_dir ${DIR}/undistorted_raw_dep
python data_preparation/depth_rgb_align.py --batch --input_dir ${DIR}/undistorted_raw_dep --output_dir ${DIR}/aligned_raw_dep
python data_preparation/interpolation_np.py --batch --input_dir ${DIR}/aligned_raw_dep --output_dir ${DIR}/interpo_depth
python data_preparation/crop_img.py --batch --rgb_input_dir ${DIR}/origin_rgb --rgb_output_dir ${DIR}/cropped_rgb --dep_input_dir ${DIR}/interpo_depth --dep_output_dir ${DIR}/cropped_interpo_dep
python data_preparation/resize_img.py --batch --rgb_input_dir ${DIR}/cropped_rgb --rgb_output_dir ${DIR}/resize_rgb --dep_input_dir ${DIR}/cropped_interpo_dep --dep_output_dir ${DIR}/resize_interpo_dep