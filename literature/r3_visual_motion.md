# R3 literature review: optical flow and learned visual motion

The camera branch asks whether visual observations contain information about
the error of a two-second inertial integration window. That is a regression
problem on 1,312 overlapping windows from two flights. It is not supervised
optical-flow estimation or a complete visual-odometry system. The literature
motivates representations; it does not supply an accuracy claim for this
small cross-flight experiment.

Farnebäck, **“Two-Frame Motion Estimation Based on Polynomial Expansion”**
(SCIA 2003), derives an image displacement field by comparing local polynomial
approximations across frames. This motivates R3's inexpensive dense-flow
baseline. OpenCV's implementation runs offline between adjacent resized
frames, so no learned vision weights or flow labels are required. Magnitude,
direction and dispersion summaries can expose scene motion that the IMU's
hand-made statistics miss. The translation model is local, however: lighting
changes, textureless regions, large displacement, depth and rotation all
complicate its interpretation. Our pixel/second features are not metric
velocities. [Author's paper page](https://www.ida.liu.se/ext/WITAS-ev/Computer_Vision_Technologies/PaperInfo/farneback03.html),
[DOI](https://doi.org/10.1007/3-540-45103-X_50).

Shi and Tomasi, **“Good Features to Track”** (CVPR 1994), connect feature
selection to the conditioning of a local image-tracking problem. R3 counts
corners returned by `goodFeaturesToTrack` as a visual-texture indicator.
This is not a claim that those corners survive tracking: no multi-frame
track is maintained. Combined with Laplacian-variance sharpness statistics,
the count helps describe windows where a visual motion estimate may be less
informative. Both statistics can respond to scene content as well as image
quality, so they remain ordinary candidate predictors in the ablation.
[Institutional record](https://publications.ri.cmu.edu/good-features-to-track),
[DOI](https://doi.org/10.1109/CVPR.1994.323794).

Dosovitskiy and colleagues, **“FlowNet: Learning Optical Flow with
Convolutional Networks”** (ICCV 2015), show that CNNs can learn a motion
representation from image pairs with supervised flow targets. Their use of
synthetic training data also illustrates the data demand of learned motion.
R3 borrows the motivation for jointly processing temporal images, not the
architecture, weights or loss. Our three-channel input means start, middle
and end grayscale frames; a 23,808-parameter encoder emits a 64-dimensional
embedding supervised only by drift magnitude. It neither predicts a dense
flow field nor proves that its embedding measures motion rather than scene
appearance. E5's held-out-flight results and the later E6 fusion comparison
are necessary to test usefulness. [Primary paper](https://openaccess.thecvf.com/content_iccv_2015/html/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.html).

Wu and He, **“Group Normalization”** (ECCV 2018), replace batch-wide
normalisation with statistics computed over channel groups within each
example. R3 uses four groups after each convolution. With only 450 or 553
eligible training windows, small batches and possible flight differences,
this avoids relying on running batch statistics during evaluation. It does
not remove overfitting; dropout, weight decay, validation-based stopping and
three random seeds remain necessary. This choice belongs only to the new
camera encoder and does not change R2's BatchNorm-based IMU model.
[Primary paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.php).

Tier 1 tests an explicit, low-dimensional image representation by appending
ten predictors to the existing 104-feature Lasso with identical tuning.
Tier 2 tests a compact learned image representation with the frozen R1
splits. A camera-only improvement does not establish a fusion improvement,
and optimisation spread across three seeds is not uncertainty over new
flights. These distinctions keep the experiment claims narrower than the
capabilities of modern optical-flow and visual-odometry systems.
