## Part 5
The test accuracy of the facial segmentation network is reported in [segmentation](./part%204/segmentation/). Different dataset than the dataset used in [part 4](./part%204/) is not considered for getting face matching scores, since we use a pretrained ArcFace network without applying any training procedure.
[Slides](https://github.com/kaganozturk/Investigation-of-possible-gender-bias-in-face-recognition/raw/main/project_slides.pptx)

### Installation
[Download](https://drive.google.com/file/d/1KSa9_g_cL047Z0B2hEn8cRPsDP7sjPXC/view?usp=share_link) and save the facial hair segmentation network to 'part 4/segmentation'.
```
conda create -n kagan python=3.8
conda activate kagan
pip install arcface astropy
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install matplotlib tqdm

# predict facial hair masks
cd "part 4/segmentation/"
python predict_facial_hair_mask.py

# download ArcFace weights and calculate matching scores
cd ..
python calculate_matching_score.py

# plot pairs
python plot_score_vs_no_hair_pixels.py
```
