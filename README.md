<h2>TensorFlow-FlexUNet-Image-Segmentation-CoBrA-Hippocampal-Subfield-T2W (2026/04/09)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>CoBrA-Hippocampus-Subfield-T2W</b>
 based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>), and a 512x512 pixels PNG
 <a href="https://drive.google.com/file/d/1vHh9VDAIwcrbrr3P2skNLXQWQ7cckVH5/view?usp=sharing">
CoBrA-Hippocampus-Subfield-T2W-ImageMask-Dataset.zip</a> 
(<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
CC BY-NC-SA 4.0</a>), which was derived by us from <br><br>
<a href="https://cobralab.net/files/brains_t2.tar.bz2">Hippocampus-subfields(T2-Weighted)</a></b>
in <a href="http://cobralab.ca/atlases/Hippocampus-subfields/">MRI-Based Digital Brain Atlases</a> and 
<br> 
<a href="https://github.com/CoBrALab/atlases/tree/master/hippocampus-subfields"><b>hippocampus-subfields/labels</b></a> in 
<a href="https://github.com/CoBrALab/atlases">
<b>Expertly segmented structures on high resolution T1 and T2 human brain scans</b>
</a>
<br><br>
<hr>
<b>Actual Image Segmentation for CoBrA-Hippocampus-Subfield-T2W Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the 
ground truth masks.
<br><br>
<a href="#color-class-mapping-table">Hippocampus Subfield class-color mapS</a><br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/images/10001_265.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/masks/10001_265.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test_output/10001_265.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/images/10002_302.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/masks/10002_302.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test_output/10002_302.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/images/10003_352.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/masks/10003_352.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test_output/10003_352.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<br>
<h3>1. Dataset Citation</h3>
The dataset used here was taken from <br><br>
<a href="https://cobralab.net/files/brains_t2.tar.bz2">Hippocampus-subfields(T2-Weighted)</a></b>
in <a href="http://cobralab.ca/atlases/Hippocampus-subfields/">MRI-Based Digital Brain Atlases</a> and 
<br>
<a href="https://github.com/CoBrALab/atlases/tree/master/hippocampus-subfields"><b>hippocampus-subfields/labels</b></a> in 
<a href="https://github.com/CoBrALab/atlases">
<b>Expertly segmented structures on high resolution T1 and T2 human brain scans</b>
</a>
<br><br>
The following explanation was taken from <a href="https://github.com/CoBrALab/atlases/blob/master/hippocampus-subfields/README.md">
Whole Hippocampus Atlas
</a>
<br><br>
These atlases, and the manual segmentation protocol used to produce them are described in:
<br>
<pre>
Winterburn JL, Pruessner JC, Chavez S, et al. A novel in vivo atlas of
human hippocampal subfields using high-resolution 3 T magnetic resonance
imaging.  Neuroimage. 2013;74:254-65.
</pre>
<br>
To obtain the subject T1 and T2 images for these labels, or for more information, please visit:
<a href="http://cobralab.ca/atlases/Hippocampus-subfields/">http://cobralab.ca/atlases/Hippocampus-subfields/</a>
<br><br>
<b>Labels</b>
<pre>
Label: 1   - right CA1
Label: 2   - right subiculum
Label: 4   - right CA4/dentate gyrus
Label: 5   - right CA2/CA3
Label: 6   - right stratum radiatum/stratum lacunosum/stratum moleculare
Label: 101 - left CA1
Label: 102 - left subiculum
Label: 104 - left CA4/dentate gyrus
Label: 105 - left CA2/CA3
Label: 106 - left stratum radiatum/stratum lacunosum/stratum moleculare
</pre>
<b>License</b><br>
(<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0</a>)
<br>
<br>
<h3>
<a id="2">
2 CoBrA-Hippocampus-Subfield-T2W ImageMask Dataset
</a>
</h3>
<h3>2.1 Download ImageMask Dataset</h3>
 If you would like to train this CoBrA-Hippocampus-Subfield-T2W Segmentation model by yourself,
 please download the dataset from the google drive  
 <a href="https://drive.google.com/file/d/1vHh9VDAIwcrbrr3P2skNLXQWQ7cckVH5/view?usp=sharing">
CoBrA-Hippocampus-Subfield-T2W-ImageMask-Dataset.zip</a> 
(<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
CC BY-NC-SA 4.0</a>)
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<br>
<pre>
./dataset
└─CoBrA-Hippocampus-Subfield-T2W
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>CoBrA-Hippocampus-Subfield-T2W Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/CoBrA-Hippocampus-Subfield-T2W_Statistics.png" width="512" height="auto"><br>
<br><br>
As shown above, the number of images of train and valid datasets is not so large to use for the
 training set of our segmentation model.
<br><br>
<h3>2.2 Derivation of CoBrA-Hippocampus-Subfield-T2W </h3>
Firstly, please clone the orginal github repository <a href="https://github.com/CoBrALab/atlases">CoBrALab atlases</a> 
to your local folder. 
Secondly, please download <a href="https://cobralab.net/files/brains_t2.tar.bz2">brain_t2.tar.bz2</a>, expand the downloaded,
and put it under <b>/atlases/hippocampus-subfields</b> of your local repository.
The folder structure of your local <b>atlases</b> folder will become the following.<br> 
<pre>
./atlases
└─hippocampus-subfields
     ├─brains_t2
     │   ├─brain1_t2.mnc
     │   ├─brain2_t2.mnc
     │   ├─brain3_t2.mnc
     │   ├─brain4_t2.mnc
     │   └─brain5_t2.mnc
     └─labels
          ├─brain1_labels.mnc
          ├─brain2_labels.mnc
          ├─brain3_labels.mnc
          ├─brain4_labels.mnc
          └─brain5_labels.mnc
...
</pre>
We used a Python script and the following class-color mapping table to generate our PNG dataset with colorized masks
from <b>brain*_t2.mnc </b> and  <b>brain*_labels.mnc</b> files.
<br><br>
<a id="color-class-mapping-table"><b>Hippocampus Subfield class-color map</b></a>
<br><br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr><th>Indexed Color</th><th>Color</th><th>RGB</th><th>Class</th></tr>
<tr><td>1</td><td with='80' height='auto'><img src='./color_class_mapping/CA1.png' widith='40' height='25'></td><td>(0, 0, 255)</td><td>CA1</td></tr>
<tr><td>2</td><td with='80' height='auto'><img src='./color_class_mapping/Subiculum.png' widith='40' height='25'></td><td>(0, 255, 0)</td><td>Subiculum</td></tr>
<tr><td>4</td><td with='80' height='auto'><img src='./color_class_mapping/CA4-DG.png' widith='40' height='25'></td><td>(255, 0, 0)</td><td>CA4-DG</td></tr>
<tr><td>5</td><td with='80' height='auto'><img src='./color_class_mapping/CA2-CA3.png' widith='40' height='25'></td><td>(255, 255, 0)</td><td>CA2-CA3</td></tr>
<tr><td>6</td><td with='80' height='auto'><img src='./color_class_mapping/Stratum.png' widith='40' height='25'></td><td>(0, 255, 255)</td><td>Stratum</td></tr>
</table>
<br>
<br>
<h3>2.3 Train Image and SmoothedMask Saｍples</h3>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained CoBrA-Hippocampus-Subfield-T2W TensorFlowFlexUNet Model by using the 
<a href="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 6
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specifed rgb color map dict for CoBrA-Hippocampus-Subfield-T2W 1+5 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;CoBrA-Hippocampus-Subfield-T2W rgb color map dict for 1+5 classes.
;                        blue,     green,    red           yellow,      cyan,        
rgb_map = {(0,0,0):0,(0,0,255):1,(0,255,0):2,(255,0,0):3, (255,255,0):4,(0,255,255):5,}
</pre>
<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 46,47,48)</b><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 94,95,96)</b><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was terminated at epoch 50.<br><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/train_console_output_at_epoch50.png" width="1024" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W</b> folder, 
and run the following bat file to evaluate TensorFlowUNet model for CoBrA-Hippocampus-Subfield-T2W.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/evaluate_console_output_at_epoch50.png" width="1024" height="auto">
<br><br>Image-Segmentation-CoBrA-Hippocampus-Subfield-T2W

<a href="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this <b>CoBrA-Hippocampus-Subfield-T2W/test</b> was very low, and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0057
dice_coef_multiclass,0.9975
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W</b> folder, and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for CoBrA-Hippocampus-Subfield-T2W.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of CoBrA-Hippocampus-Subfield-T2W Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the
ground truth masks.
<br><br>
<a href="#color-class-mapping-table">Hippocampus Subfield class-color mapS</a><br><br>
<br><br>
<table>
<tr>
<th>Input:Image</th>
<th>Mask (ground_truth)</th>
<th>Prediction:Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/images/10001_269.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/masks/10001_269.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test_output/10001_269.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/images/10001_296.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/masks/10001_296.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test_output/10001_296.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/images/10001_348.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/masks/10001_348.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test_output/10001_348.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/images/10003_343.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/masks/10003_343.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test_output/10003_343.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/images/10003_260.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/masks/10003_260.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test_output/10003_260.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/images/10003_365.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test/masks/10003_365.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/CoBrA-Hippocampus-Subfield-T2W/mini_test_output/10003_365.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Multi-contrast submillimetric 3 Tesla hippocampal subfield segmentation protocol and dataset</b><br>
Jessie Kulaga-Yoskovitz, Boris C. Bernhardt, Seok-Jun Hong, Tommaso Mansi, Kevin E. Liang, <br>
Andre J.W. van der Kouwe, Jonathan Smallwood, Andrea Bernasconi & Neda Bernasconi<br>
<a href="https://www.nature.com/articles/sdata201559">https://www.nature.com/articles/sdata201559</a>
<br><br>
<b>2. A paired dataset of multi-modal MRI at 3 Tesla and 7 Tesla with manual hippocampal subfield segmentations</b><br>
Lei Chu, Baoqiang Ma, Xiaoxi Dong, Yirong He, Tongtong Che, Debin Zeng, Zihao Zhang & Shuyu Li<br>
<a href="https://www.nature.com/articles/s41597-025-04586-9?fromPaywallRec=false">
https://www.nature.com/articles/s41597-025-04586-9?fromPaywallRec=false</a>
<br><br>
<b>3. A novel deep learning based hippocampus subfield segmentation method</b><br>
JoséV. Manjón, José E. Romero & Pierrick Coupe<br>
<a href="https://www.nature.com/articles/s41598-022-05287-8.pdf">https://www.nature.com/articles/s41598-022-05287-8.pdf</a>
<br><br>
<b>4. DSnet: a new dual‑branch network for hippocampus subfield segmentation</b><br>
Hancan Zhu, Wangang Cheng, Keli Hu & Guanghua He<br>
<a href="https://www.researchgate.net/publication/381961150_DSnet_a_new_dual-branch_network_for_hippocampus_subfield_segmentation">
https://www.researchgate.net/publication/381961150_DSnet_a_new_dual-branch_network_for_hippocampus_subfield_segmentation</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
