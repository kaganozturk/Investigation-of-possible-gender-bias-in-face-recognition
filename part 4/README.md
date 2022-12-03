In this part,

+ Codes for feature extraction and macthing score calculation are updated.
+ [MinPlus](https://colab.research.google.com/drive/1AL2aEEyZOWJTyTaspFQcry_1g0E4b4x5?usp=sharing#scrollTo=3m46JmcMKX-b) visualization technique for face verification is implemented.
+ [Facial Hair Segmentation](./segmentation/) network is improved using more training data.
+ 50 frontal face images are selected from the [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset and using [FaceApp](https://www.faceapp.com) 7 different synthetic facial hair styles are obtained to investigate effects of hair style variation for face matching.

### Visualization

In this work, we use MinPlus saliency map method, proposed in [True Black-Box Explanation in Facial Analysis],(https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Mery_True_Black-Box_Explanation_in_Facial_Analysis_CVPRW_2022_paper.pdf) to find most relevant image regions for [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf) feature descriptor. Heatmaps produced by this method, are obtained using a removal strategy, an aggregation strategy and then combination of these two steps. 

<p align="center">
  <img src="heatmap_MinPlus.png" width="750" title="heatmap_MinPlus", alt"Title">
</p>

