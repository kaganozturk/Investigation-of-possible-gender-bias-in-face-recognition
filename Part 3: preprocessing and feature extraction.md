# Feature Extraction

In this part, a pretrained convolutional neural network (CNN) is used to extract features from facial images. Also, a CNN is trained for segmentation of facial hair (beard and mustache).

### ArcFace
Convolutional neural networks have achieved great performance on face recognition in recent years. They use a big collection of facial images obtained from the web. Several manual and automatic data collection protocols are utilized to remove duplicates and label the identities. A couple of thousands identities, which have hundreds of images per identity, are included in training data.  Once the network is trained to recognize people in training data, it can be utilized to extract features not only for training data but also for unseen dataset, since it has been shown in many works in recent years that CNNs can learn generalizable discriminative features when trained with big data. Then a distance metric is applied to measure the similarity between features to make a verficion decision.

[ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf) is one of the most successful CNN for face verification proposed in recent years. It is trained using MS1MV2 dataset that has 58k identities and 5.8M images. They employ ResNet-100 acrhitecture. The input image size 112 x 112 is selected to produce 512-D embedding features. They propose a new loss function "Additive Angular Margin Loss" to increase discriminative power of features and help stabilize training. In this project, [this publicly-available network](https://github.com/deepinsight/insightface) is used for feature extraction. 

In this phase any face detection and alignment algorithm is not employed since the face images in CelebA-HQ dataset are created using the facial landmark annotations in CelebA dataset.

# Facial Hair Segmentation
A semantic segmentation network is trained to segment beard and mustache on face images in this project.

### BiSeNet
[The Bilateral Segmentation Network (BiSeNet)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf) is proposed to both increase inference speed and preserve the spatial information to produce high resolution features for segmentation. A [publicly-available implementation](https://github.com/zllrunning/face-parsing.PyTorch) is utilized to train a neural network for facial hair segmentation in this work. The network is trained from scratch using hand-annotated facial hair masks on CelebA-HQ dataset.

##### Next steps
Facial hair annotations will be updated with one extra label for near clean shaved face images (5 o'clock shadow) since it is observed that most of the errors produced by the facial hair segmentation network come from these kinds of images. By doing so, the noise in training data, because of the loosy definition of beard and mustache, is expected to be reduced; as it is difficult to make a decision whether to consider these areas as beard and/or mustache or annotate them as no-facial hair region. Also, 5 o'clock shadow is expected to have less effect on face recognition systems compared to longer facial hair and having an additional label for them can help analyze these comparisons.

Once the facial hair segmentation for the test set is obtained using the network, an input perturbation protocol will be applied to measure the effect of facial hair.

# Code

[Download the ArcFace network](https://drive.google.com/file/d/1Hc5zUfBATaXUgcU2haUNa7dcaZSw95h2/view?usp=sharing).
Extract the downloaded file in the 'models' folder.

Run **feature_extraction.py** to extract features for the images in the 'examples' folder. The produced features will be saved in the 'results' folder as numpy arrays.

Then, run **feature_matching.py** to print cosine similarity score between two features. Use "--im1" and "--im2" to set the feature paths.
