---
title: "Machine Learning for MRI Image Reconstruction"
date: 2021-12-22T13:19:56-05:00
draft: false
math: true
_build:
 list: false

---
Magnetic resonance imaging (MRI) has long scan times, sometimes close to an hour for an exam. This sucks because long scan times makes MRI exams more expensive, less accessible, and unpleasant. {{<hide prompt="How does it feel like to be in an MRI?" uniqueNum="74" no-markdown="True">}}
Imagine hearing this for an hour. <br><br>

{{<youtube "9GZvd_4ot04?start=203">}}

{{</hide>}}


Here, I review the latest methods in machine learning that aim to reduce the scan time through new ways of image reconstruction. Smarter image reconstructions allows us to acquire way less data, which means shorter scan times. These techniques are pretty general and can be applied to other image reconstruction problems.


## MRI Image Reconstruction
In most medical imaging methods, what you see on the screen isn’t just a raw feed of what the device’s sensors are picking up.

In MRI, this is what the sensors pick up:

{{< figure src="/ml-for-mri/mri-sensor-data.png" width="50%">}}

How in the world is this data useful? Image reconstruction is this incredible procedure that can turn this mess of sensor data into an actual image. After doing image reconstruction on the sensor data above, we get:

{{< figure src="/ml-for-mri/mri-knee.png" width="50%">}}

Now that's much better! (this is an MRI of the knee)

So how does this magical procedure of turning sensor data into images work? 

A nice way to frame this problem is to consider the signals the sensors pick up as a mathematical transformation of the image. In this framing, creating an image is inverting this mathematical transformation. This might seem backward, but it’ll become handy soon.

In MRI, the transformation from image to sensor data is a [2D or 3D Fourier transform](https://youtu.be/spUNpyF58BY). This is super wacky! It means the sensors somehow measure the spatial frequencies in the image[^1]! We can write this as:

[^1]: This comes from two cool tricks in MRI, known as frequency encoding and phase encoding -- maybe I will write a blog post on this.

$$
    \mathbf{y} = \mathcal{F} (\mathbf{x}^*)
$$

where $ \mathbf{y} $ is the (noiseless) sensor data, $\mathbf{x}^* $ is the ground-truth image, and $\mathcal{F}$ is the Fourier transform.

Reconstructing the image from the frequency-domain (sensor) data is simple: we just apply an inverse Fourier transform.
$$
    \mathbf{\hat{x}} = \mathcal{F}^{-1}(\mathbf{y})
$$

For simplicity, let's assume we're recording from a single coil, but these methods can be extended to multi-coil imaging (also called parallel imaging).


## Using Less Data
The MRI Gods tell us that if we want to reconstruct an image with $n$ pixels (or voxels), we need at least $n$ frequencies. {{<hide prompt="Why?" uniqueNum="5">}}
This can be seen using a bit of linear algebra. Since the Fourier transform is linear, we can represent it by an $n\times n$ matrix, say $\mathbf{F}$, with each column of $\mathbf{F}$ corresponding to a different frequency. If we only use a subset of the frequencies, this amounts to removing some of the columns of $\mathbf{F}$. But then the new version of $\mathbf{F}$ has less than $n$ columns, which means the problem of finding an $\mathbf{x}$ such that $\mathbf{F} \mathbf{x}=\mathbf{y}$ is underdetermined. Therefore, there are infinitely many images $\mathbf{x}$ that are consistent with the sensor data.
{{</hide>}}

But the problem with acquiring $n$ frequencies is that it takes a lot of time. This is because MRI scan time scales linearly with the number of frequencies you acquire[^2]. A typical MRI image has on the order of 10 million frequencies, which -- even with many hardware tricks to cut acquisition time -- means an MRI exam typically takes ~40 minutes and can sometimes take as long as an hour. If we could acquire only 1/4th of the frequencies, we can reduce acquisition time by 4x (and therefore MRIs could cost 4x less).

[^2]: To be precise, MRI scan time scales linearly in 2 of the 3 spatial dimensions. We actually get one dimension of frequencies for free. This is from a neat trick known as frequency encoding which allows us to parallelize the acquisition process.

So suppose we drink a bit too much and forget about the linear algebra result, only acquiring a subset of the frequencies. Let's set the data at the frequencies that we didn't acquire to be 0. We can write this as

\begin{equation}
    \mathbf{\tilde{y}} = \mathcal{M} \odot \mathbf{y} = \mathcal{M} \odot \mathcal{F} (\mathbf{x}^*)
\end{equation}
where $\mathcal{M}$ is a masking matrix filled with 0s and 1s, and $\odot$ denotes element-wise multiplication. If we try to reconstruct the same knee MRI data as above with less frequencies, we get (aliasing) artifacts:

{{<figure src="/ml-for-mri/simple-compressed-recon.png" width="75%">}}

{{<hide prompt="Why is the mask composed of horizontal lines? And why is the mask more densely sampled near the middle?" uniqueNum="9">}}
For most MRI acquisition methods, there is no time savings to keeping only part of a horizontal line. It's all or nothing.

More of the information in the signal is contained in the low frequencies, so we sample the middle (where the lower frequncies are) more than the rest.

{{</hide>}}

So our dreams of using less frequencies are over, right?

What if we add more information to the image reconstruction process that is not from the current measurement $\mathbf{\tilde{y}} $? For example, in compressed sensing, we can assume that the desired image $\mathbf{x}$ doesn't have many edges. Here's a knee MRI along with its edge map, which we see is very sparse:



How do we incorporate the fact that we know that MRI images aren't supposed to have many edges? First, we need some way of counting how many edges are in an MRI image. A decent way to do this is by summing the spatial derivatives in the image (this is called the total variation we can write this mathematically as $R_{TV}(\mathbf{x}) = ||\nabla \mathbf{x}||_1$).



Next, we create an objective function that minimizes 





We can write this as:

$$R(\mathbf{x}) = ||\nabla \mathbf{x}||_1$$













































For example, in compressed sensing, we assume that the desired image $\mathbf{x}$ is compressible. How do we know if an image is compressible? Can we measure compressibility?

One way to measure compressibility is to count how many pixels (or voxels in 3D) in the image are zero. If the image is mostly zeros, then it can be compressed by remembering only the non-zero elements (and where they go)! The images you get when imaging blood vessels (from [Magnetic Resonance Angiography](https://en.wikipedia.org/wiki/Magnetic_resonance_angiography)) are highly compressible in this way! 

Unfortunately, this doesn't work for non-blood vessel MRIs. Another way to measure compressibility is to see if to count how many edges there are in the image. This tends to work decently well for many types of MRI images!

We want our measure of compressibility to be differentiable.

<!-- {{<hide prompt="What do you mean by sparse?" uniqueNum="64">}} A sparse signal/image/vector means it has many zeros. A consequence of this is that you can compress it (hence the name "compressed sensing"). 
{{</hide>}} -->

We can then solve the following optimization problem

\begin{equation}
    \argmin_{\mathbf{x}} || \mathcal{M} \odot \mathcal{F}(\mathbf{x}) - \mathbf{\tilde{y}} ||_2^2 + R(\mathbf{x})
\end{equation}

The left term says: "If $\mathbf{x}$ were the real image, how would the sensor data we'd capture from $\mathbf{x}$ compare with our real sensor data $\mathbf{\tilde{y}}$?" In other words, it tells us how much our reconstruction $\mathbf{x}$ agrees with our measurements $\mathbf{\tilde{y}}$. The right term, $R(\mathbf{x})$, penalizes images if they are not sparse. $R(\mathbf{x})$ is called a regularizer. The challenge is finding an image that both agrees with our measurements and is sparse.

One way to choose $R(\mathbf{x})$ is as the sum of all the elements of $\mathbf{x}$.

Some examples of $R(\mathbf{x})$ for MRI image reconstruction include:

* $R_{L^1}(\mathbf{x}) = \lambda ||\mathbf{x}||_1$, where $||\mathbf{x}||_1 = \sum_i |\mathbf{x}_i|$ is the $L^1$ norm, and $\lambda \in (0,\infty)$ is a constant that is selected (in machine learning language, we'd call $\lambda$ a hyperparameter). The L1 norm encourages sparsity.
* $R_{\text{wavelet}}(\mathbf{x}) = \lambda ||\mathbf{\Psi} \mathbf{x}||_1$ where $\mathbf{\Psi}$ denotes a wavelet transform, and $\lambda \in (0,\infty)$ is a constant. The wavelet regularizer encourages sparsity in the wavelet basis.
* $R_{TV}(\mathbf{x}) = \lambda ||\nabla \mathbf{x} ||_2$ where $\nabla$ is the spatial gradient (estimated numerically), and $\lambda \in (0,\infty)$ is a constant. The total variation regularizer removes excessive details but keeps edges.

{{<hide prompt="How do you interpret R(x) probabilistically?" uniqueNum="1">}}

$R(\mathbf{x})$ is a measure of how many bits you need to encode your image $\mathbf{x}$. If you use maximum _a posteriori_ estimation, you can show that $R(\mathbf{x}) \propto -\log p(\mathbf{x})$! We call $p(\mathbf{x})$ the prior distribution. So $R(\mathbf{x})$ really is measuring how likely an image is under your prior!

{{</hide >}}

Algorithms like gradient descent allow one to solve (4). Though compressed sensing can improve the image quality relative to a vanilla inverse Fourier transform, it still suffers from artifacts. Below is another knee MRI with 4x subsampled data:

{{< figure src="/ml-for-mri/CS.png" width="75%">}}

The difficulty with classical compressed sensing is that humans must manually design the regularizers $R(\mathbf{x})$. We can come up with basic heuristics like the examples above, but ultimately deciding whether an image looks like it could have come from an MRI is a complicated process.

Enter machine learning... Over the past decade-ish, machine learning has had great success in learning functions that humans have difficulty hard coding. It has revolutionized the fields of computer vision, natural language processing, among others. Instead of hard coding functions, machine learning algorithms learn functions from data. In the next section, we will explore a few recent machine learning approaches to MRI image reconstruction.

{{<hide prompt="Did you know Terence Tao was one of the pioneers of compressed sensing?" uniqueNum="12">}}
It turns out [Terence Tao](https://en.wikipedia.org/wiki/Terence_Tao)'s most cited paper is from his work on compressed sensing!
{{</hide>}}

## Machine Learning Comes to the Rescue
### fastMRI Baseline
The [fastMRI baseline with U-Net](http://arxiv.org/abs/1811.08839) is one of the simpler machine learning methods for MRI image reconstruction. Its approach has two steps. First, the sensor data is turned into an image via the inverse Fourier transform. Then, this image is "cleaned up" by a [U-Net](http://arxiv.org/abs/1505.04597), producing a new image that is supposed to look like the real image. We can formally write the operations performed by this network as
\begin{equation}
    \mathbf{\hat{x}} = f_{{\boldsymbol{\theta}}}(\mathbf{\tilde{y}}) = \text{UNET}_{{\boldsymbol{\theta}}}(\mathcal{F}^{-1}(\mathbf{\tilde{y}}))
\end{equation}

where $\mathbf{\tilde{y}}$ is the subsampled sensor data, and $\text{UNET}_{\boldsymbol{\theta}}$ is the U-Net parameterized by a vector of parameters ${\boldsymbol{\theta}}$.

A U-Net is a type of convolutional neural network and is a popular model for biomedical applications. {{< hide prompt="What is a convolutional neural network?" uniqueNum="2">}}
A convolutional neural network (CNN) -- like other neural networks -- learns a function from data. CNNs, in particular, are used for learning functions whose input is an image. The output of a CNN could be a class (for example, given an image, the CNN could say which animal is in it), another image (for example, the CNN could denoise an image).

A CNN is composed of a series of convolutions composed with simple nonlinear functions between the convolutions.

Why convolutions? Convolutions apply the same operation to every region in its input, so it is robust to translations of the image. Robustness to translations is a great property to have for image processing!

Why nonlinear functions? Composing convolutions with more convolutions results in just another convolution. But if we add nonlinear functions in between the convolutions, our network becomes much more expressive! In fact, there is a theorem that states that neural networks made up of linear functions intertwined with a set of simple nonlinear functions can approximate any continuous function!

CNNs have achieved incredible success on a variety of computer vision problems. See this [great course](https://cs231n.github.io/convolutional-networks/) to learn more about CNNs.
{{< /hide >}}

The parameters ${\boldsymbol{\theta}}$ of the U-Net are optimized in order to minimize the following loss function.

\begin{equation}
    \mathcal{L}({\boldsymbol{\theta}}) = \sum_{(\mathbf{\tilde{y}},{\mathbf{x}}^*) \in \mathcal{D}} ||\text{UNET}_{\boldsymbol{\theta}}(\mathcal{F}^{-1}(\mathbf{\tilde{y}})) - \mathbf{x^{\*}} ||_1
\end{equation}

where $\mathbf{\tilde{y}}$ and $\mathbf{x}^\*$ are subsampled sensor, image reconstruction pairs from the dataset $\mathcal{D}$. In words, our neural network takes as its input a subsampled sensor representation $\mathbf{\tilde{y}}$ and tries to product an output $\text{UNET}_{\boldsymbol{\theta}}(\mathcal{F}^{-1}(\mathbf{\tilde{y}}))$ that is as close to the real image ${\mathbf{x}}^*$ as possible.

The parameters ${\boldsymbol{\theta}}$ are optimized via gradient descent. {{< hide prompt="What is gradient descent?" uniqueNum="16" >}}

Gradient descent is an iterative algorithm to minimze some function $\mathcal{L}(\boldsymbol{\theta})$. It starts at some initial parameters ${\boldsymbol{\theta}}^{(0)}$ and updates its parameters in the direction of the gradient, $\nabla L({\boldsymbol{\theta}})$, so as to locally reduce the loss function as much as possible. In the $t$-th iteration, ${\boldsymbol{\theta}}^t$ is updated to ${\boldsymbol{\theta}}^{t+1}$ via
$$
    {\boldsymbol{\theta}}^{t+1} = {\boldsymbol{\theta}}^{t} - \alpha^{t} \nabla \mathcal{L}({\boldsymbol{\theta}})
$$
where $t$ is the iteration number, $\alpha^{t}$ is called the learning rate, ${\boldsymbol{\theta}}^{t}$ and ${\boldsymbol{\theta}}^{{t+1}}$ are the parameters from the previous and current iterations, respectively.


You might worry that gradient descent gets stuck in local minima, but in practice, for neural networks with a huge amount of parameters, the minima found by gradient descent turn out to be really good ones! To my knowledge, we still don't fully understand why this is.

{{< /hide >}}

{{<figure src="/ml-for-mri/unet-diagram.png" width="75%" caption=
    `**The process of fastMRI U-Net.** First, the subsampled sensor data is inverted to an aliased image via the inverse Fourier transform. Then, the U-Net cleans up the image. To train the U-Net, the cleaned up image is compared with the ground truth image via a loss function $\mathcal{L}$, and gradient descent is applied to the parameters of the U-Net.`>}}

In the figure below, we see a significant qualitative improvement in the reconstructions from the U-Net relative to traditional compressed sensing with total variation regularization.

{{<figure src="/ml-for-mri/unet.png" width="75%" caption=`**Knee MRI reconstructions comparison between compressed sensing with total variation regularization and the fastMRI U-Net baseline.** The data is acquired using multiple coils at 4x and 8x subsampling. Reproduced from [Zbontar 2018](http://arxiv.org/abs/1811.08839).`>}}

{{<hide prompt="Wait, but where does the training data come from?" uniqueNum="17">}}
Any machine learning model needs data to learn from. In fact, much of the improvements in machine learning over the past decade has come from expanding the size of datasets. Open source datasets are crucial in machine learning for comparing methods: it's hard to compare methods when they use different proprietary data.

In 2019, Facebook AI released an MRI dataset called fastMRI (Zbontar 2018). The dataset contains 8344 brain and knee MRI scans. The scans contain raw fully sampled sensor data as well as the corresponding image reconstructions. The scans were done with a variety of MRI parameters (different pulse sequences, field strengths, repetition times, and echo times). The diversity of parameters is important: we want image reconstruction methods to work for all relevant clinical parameters. The dataset also includes 20,000 brain and knee MRI scans that only contain the reconstructed images and not the sensor data (it is also not straightforward to get the raw frequency-domain data from the images as there are multiple coils and postprocessing).

The dataset consists of both a training set and a test set. The training set is used to set the parameters of the model, and the test set is used to evaluate the model.
{{</hide>}}

### VarNet
Recall that in classical compressed sensing, we solve (4). If we write the forward operator $\mathbf{A}=\mathcal{M} \odot \mathcal{F}$, the optimization problem becomes

\begin{equation}
    \argmin_{\mathbf{x}} || \mathbf{A}\mathbf{x} - \mathbf{\tilde{y}} ||_2^2 + R(\mathbf{x})
\end{equation}

If we solve this via gradient descent, we get the following update equation for the $t$-th iteration of the image, ${\mathbf{x}}^t$.

\begin{equation}
    {\mathbf{x}}^{t+1} = {\mathbf{x}}^t - \alpha^t (\mathbf{A}^*(\mathbf{A}{\mathbf{x}}^t - \mathbf{\tilde{y}}) + \nabla R({\mathbf{x}}^t))
\end{equation}

where $\mathbf{A}^*$ is the adjoint of $\mathbf{A}$. Note that gradient descent in (8) is done on the image $\mathbf{x}$, as opposed to ${\boldsymbol{\theta}}$. Instead of hard coding the regularizer $R(\mathbf{x}^t)$, we can replace it with a neural network. We do this by replacing $\nabla R(\mathbf{x}^t)$ with a CNN. We get a new update equation:

\begin{equation}
    {\mathbf{x}}^{t+1} = {\mathbf{x}}^t - \alpha^t \mathbf{A}^*(\mathbf{A}{\mathbf{x}}^t - \mathbf{\tilde{y}}) + \text{CNN}_{\boldsymbol{\theta}} ({\mathbf{x}}^t)
\end{equation}

The VarNet architecture ([Sriram 2020](http://arxiv.org/abs/2004.06688) & [Hammernik 2018](http://arxiv.org/abs/1704.00447)) consists of multiple layers. Each layer takes the output of the previous layer, ${\mathbf{x}}^t$, as its input, and outputs ${\mathbf{x}}^{t+1}$ according to (9). This style of architecture is called unrolled optimization. In practice, VarNet has about 8 layers, and the CNN is a U-Net. The parameters of the U-Net are updated via gradient descent on $\boldsymbol{\theta}$, but the loss function, $\mathcal{L}({\boldsymbol{\theta}})$, is taken to be the structural similarity index measure (SSIM). {{<hide prompt="What is SSIM?" uniqueNum="3">}}
The SSIM is a measure of similarity for images that is more aligned with the human perceptual system than the mean-squared error. It compares two images across three dimensions: luminosity, contrast, and structural similarity. A great explanation of SSIM can be found in [this blog post](https://bluesky314.github.io/ssim/).
{{</hide>}}

{{<figure src="/ml-for-mri/varnet-diagram.png" width="75%" caption=`**The process of VarNet.** An image starts off as blank and is updated iteratively via (9), producing a better image at each step. To train VarNet, the image at the final $T$th step is compared with the ground truth via a loss function, $\mathcal{L}(\boldsymbol{\theta})$, and the parameters of VarNet, $\boldsymbol{\theta}$, are updated via gradient descent.`>}}

Technically, the approach above isn't quite the [latest version of VarNet](http://arxiv.org/abs/2004.06688): there were a few changes that improve things a tiny bit. {{<hide prompt="What things?">}}
* Updating in sensor-space instead of in image space. The sensor-domain update can be obtained by taking the Fourier transform of both sides of (9):

$$
    \mathbf{y}^{t+1} = \mathbf{y}^t - \alpha^t \mathcal{M} \odot (\mathbf{y}^t - \mathbf{\tilde{y}}) + \mathcal{F}(\text{CNN}_{\boldsymbol{\theta}} (\mathcal{F}(\mathbf{y}^t)))
$$

* Learning a 2nd CNN to estimate the sensitivity maps of each coil in the case of multi-coil (parallel) imaging
{{</hide>}}

Recently, an [interchangeability study of VarNet](https://www.ajronline.org/doi/10.2214/AJR.20.23313) was done. It found that using $1/4$th of the data with VarNet was diagnostically interchangeable with the ground truth reconstruction. In other words, radiologists made the same diagnoses with both methods. {{<hide prompt="Tell me a funny story about this study" uniqueNum="14">}}

At first the physicians thought the sVarNet images didn't look great because the images were too smooth. So the authors added some random Gaussian noise, and then the physicians loved the images! In fact, the authors give a fancy name to their process of adding random noise; they call it "adaptive image dithering."
{{</hide>}}

Below is a sample reconstruction from their study, compared with the ground truth. I can't tell the difference.

{{<figure src="/ml-for-mri/varnet.png" width="75%" caption=`Knee MRI comparison between VarNet and the ground truth at 4x acceleration. Figure reproduced from [Recht 2020](https://www.ajronline.org/doi/10.2214/AJR.20.23313).`>}}

<!-- add figure -->

### Deep Generative Priors
All methods above required access to a dataset that had both MRI images paired with raw sensor data. However, to my understanding, the raw sensor data is not typically saved. Constructing a dataset with only the MRI images and without the raw sensor data might be easier. Fortunately, there are machine learning methods that only require MRI images as training data. 

One approach is to train what is called a generative model. Generative models are very popular in the computer vision community for generating realistic human faces or scenes (that it has never seen before!). Similarly, we can train a generative model to generate new MRI-like images.

Formally, a generative model is a mapping $G_{\boldsymbol{\theta}}: \mathbb{R}^m \rightarrow \mathbb{R}^n$, often with $m \ll n$ (i.e. the input space is often much smaller than the output space). The generative model learns to turn any random vector $\mathbf{z} \in \mathbb{R}^m$ into a realistic image $\mathbf{x} \in \mathbb{R}^n$.

Image reconstruction with generative models is done by solving the optimization problem:
\begin{equation}
    \argmin_{\mathbf{z}} ||\mathbf{A} G_{\boldsymbol{\theta}}(\mathbf{z}) - \mathbf{\tilde{y}}||_2^2
\end{equation}

Instead of optimizing over all images $x \in \mathbb{R}^n$, we optimize only over the images produced by the generator, $G_{\boldsymbol{\theta}}(\mathbb{R}^m)$. Since $m \ll n$, the range of the generator, $G_{\boldsymbol{\theta}}(\mathbb{R}^m)$, is much smaller than $\mathbb{R}^n$. {{<hide prompt="What if m=n?" uniqueNum="19">}}It turns out it can still work if we use early stopping! This says something deep about the optimization landscape. Early stopping still implicitly restricts the range of the generator.{{</hide>}}

An important question is how well do these models generalize outside of their training set. This is especially important for diagnosing rare conditions that might not appear in the training set. [Jalal et al.](http://arxiv.org/abs/2108.01368) recently showed that you can get pretty extraordinary generalization using a type of generative model called a [score-based generative model](https://yang-song.github.io/blog/2021/score/). As seen in the results below, they train their model on brain data and test it on a completely different anatomy -- in this case the abdomen! Their model performs much better in this case than other approaches.

{{<figure src="/ml-for-mri/dgp.png" width="75%" caption=`**Reconstructions of 2D abdominal scans at 4x acceleration for methods trained on brain MRI data.** The red arrows points to artifacts in the images. The deep generative prior method from [Jalal 2021](http://arxiv.org/abs/2108.01368) does not have the artifacts from the other methods. Results from [Jalal 2021](http://arxiv.org/abs/2108.01368).`>}}

Why generative models generalize, I don't fully understand yet, but the authors do [give some theoretical justification](http://arxiv.org/abs/2108.01368). A limitation to image reconstruction using deep generative priors is that the reconstruction time is typically longer than methods like VarNet (it can be more than 15 minutes on a modern GPU).
<!-- you're not learning the forward operator -->

### Untrained Neural Networks
Imagine we get drunk again and forget to feed our machine learning model any data. We should get nonsense right...? Well, recently, it's been [shown](http://arxiv.org/abs/2007.02471) that even with no data at all, the models in machine learning can be competitive with fully trained machine learning methods for MRI image reconstruction.

How do you explain this? First, let's see how these models work. These no-data methods start with the deep generative priors approach in the previous section. But instead of using data to train the generator $G_{\boldsymbol{\theta}}(\mathbf{z})$, we set the parameters ${\boldsymbol{\theta}}$ randomly. The structure of these ML models -- the fact that they're made of convolutions, for example -- make it such that without any data, realistic images are more likely to be generated than random noise.

This is remarkable! And confusing! We started off by saying that machine learning removes the need to manually engineer regularizers for compressed sensing. But instead, we are manually engineering the architectures of machine learning models! How much are these machine learning models really learning?

It turns out, such untrained models have been applied to other inverse problems like region inpainting, denoising, and super resolution, and [achieved remarkable results](http://arxiv.org/abs/1711.10925).

{{<figure src="/ml-for-mri/convdecoder.png" width="75%" caption=`**Comparison of the untrained [ConvDecoder](http://arxiv.org/abs/2007.02471) with the [fastMRI U-net baseline](http://arxiv.org/abs/1811.08839), and total-variation regularized compressed sensing.** Reconstructions of knee-MRI at 4x acceleration. The second row is a zoomed in version of the first row. We see that even though ConvDecoder is untrained, it produces better reconstructions than U-Net and TV-regularized compressed sensing. Figure reproduced from [Darestani 2020](http://arxiv.org/abs/2007.02471).`>}}

## Concluding Thoughts
Machine learning methods have made significant progress in reducing the scan time of MRI. Not only have ML methods for compressed sensing produced strong results on quantitative metrics like SSIM, but they have started to be [validated by clinicians](https://www.ajronline.org/doi/10.2214/AJR.20.23313). Validation by clinicians is essential in image reconstruction because a fine detail can be essential in a diagnosis but might not make it's way into a metric like the mean-squared-error.

A limitation to deep learning for healthcare is that we still don't have a good understanding of *why* deep learning works. This makes it hard to predict when and how deep learning methods will fail (there are no theoretical guarantees that deep learning will work). One tool to help in this regard is uncertainty quantification. Stochastic methods like deep generative priors can estimate the uncertainty in their reconstruction by creating many reconstructions with different random seeds and computing the standard deviation. For non-generative methods, works like [Edupuganti 2019](http://arxiv.org/abs/1901.11228) make use of Stein's unbiased risk estimate (SURE) to estimate uncertainty.

In addition to MRI, machine learning methods have also been used for other forms of image reconstruction. A great review can be found [here](http://arxiv.org/abs/2005.06001).

_**A big thank you** to [Milan Cvitkovic](https://milan.cvitkovic.net/), Hannah Le, and [Marley Xiong](https://marleyx.com) for reviewing drafts of this._

<!-- ## References
Akçakaya, Mehmet, Steen Moeller, Sebastian Weingärtner, and Kâmil Uğurbil. 2019. “Scan-Specific Robust Artificial-Neural-Networks for K-Space Interpolation (RAKI) Reconstruction: Database-Free Deep Learning for Fast Imaging.” Magnetic Resonance in Medicine: Official Journal of the Society of Magnetic Resonance in Medicine / Society of Magnetic Resonance in Medicine 81 (1): 439–53.

Alemi, Alex A., Francois Chollet, Niklas Een, Geoffrey Irving, Christian Szegedy, and Josef Urban. 2016. “DeepMath - Deep Sequence Models for Premise Selection.” arXiv [cs.AI]. arXiv. http://arxiv.org/abs/1606.04442.

Barbano, Riccardo, Željko Kereta, Chen Zhang, Andreas Hauptmann, Simon Arridge, and Bangti Jin. 2020. “Quantifying Sources of Uncertainty in Deep Learning-Based Image Reconstruction.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2011.08413.

Chaudhuri, Rishidev, Berk Gerçek, Biraj Pandey, Adrien Peyrache, and Ila Fiete. 2019. “The Intrinsic Attractor Manifold and Population Dynamics of a Canonical Cognitive Circuit across Waking and Sleep.” Nature Neuroscience 22 (9): 1512–20.

Cole, Elizabeth K., John M. Pauly, Shreyas S. Vasanawala, and Frank Ong. 2020. “Unsupervised MRI Reconstruction with Generative Adversarial Networks.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2008.13065.

“CS231n Convolutional Neural Networks for Visual Recognition.” n.d. Accessed November 27, 2021. https://cs231n.github.io/convolutional-networks/.

Cueto, Carlos, Oscar Bates, George Strong, Javier Cudeiro, Fabio Luporini, Oscar Calderon Agudo, Gerard Gorman, Lluis Guasch, and Meng-Xing Tang. 2021. “Stride: A Flexible Platform for High-Performance Ultrasound Computed Tomography.” arXiv [physics.med-Ph]. arXiv. http://arxiv.org/abs/2110.03345.

Daras, Giannis, Joseph Dean, Ajil Jalal, and Alexandros G. Dimakis. 2021. “Intermediate Layer Optimization for Inverse Problems Using Deep Generative Models.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2102.07364.

Darestani, Mohammad Zalbagi, and Reinhard Heckel. 2020a. “Accelerated MRI with Un-Trained Neural Networks.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2007.02471.

———. 2020b. “Accelerated MRI with Un-Trained Neural Networks.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2007.02471.

Defazio, Aaron, Tullie Murrell, and Michael P. Recht. 2020. “MRI Banding Removal via Adversarial Training.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2001.08699.

Deffieux, Thomas, Charlie Demene, Mathieu Pernot, and Mickael Tanter. 2018. “Functional Ultrasound Neuroimaging: A Review of the Preclinical and Clinical State of the Art.” Current Opinion in Neurobiology 50 (June): 128–35.

Deora, Rahul. n.d. “A Brief Introduction to SSIM: Structural Similarity Index.” Accessed November 27, 2021. https://bluesky314.github.io/ssim/.

Edupuganti, Vineet, Morteza Mardani, Shreyas Vasanawala, and John Pauly. 2019. “Uncertainty Quantification in Deep MRI Reconstruction.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1901.11228.

Hammernik, Kerstin, Teresa Klatzer, Erich Kobler, Michael P. Recht, Daniel K. Sodickson, Thomas Pock, and Florian Knoll. 2017. “Learning a Variational Network for Reconstruction of Accelerated MRI Data.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1704.00447.

He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. “Deep Residual Learning for Image Recognition.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1512.03385.

Huang, Chin-Wei, Jae Hyun Lim, and Aaron Courville. 2021. “A Variational Perspective on Diffusion-Based Generative Models and Score Matching.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2106.02808.

Jalal, Ajil, Marius Arvinte, Giannis Daras, Eric Price, Alexandros G. Dimakis, and Jonathan I. Tamir. 2021. “Robust Compressed Sensing MRI with Deep Generative Priors.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2108.01368.

Jalal, Ajil, Liu Liu, Alexandros G. Dimakis, and Constantine Caramanis. 2020. “Robust Compressed Sensing Using Generative Models.” Advances in Neural Information Processing Systems. https://github.com/ajiljalal/csgm-robust-neurips.

Kaliszyk, Cezary, Josef Urban, Henryk Michalewski, and Mirek Olšák. 2018. “Reinforcement Learning of Theorem Proving.” arXiv [cs.AI]. arXiv. http://arxiv.org/abs/1805.07563.

Liu, Fang, Alexey Samsonov, Lihua Chen, Richard Kijowski, and Li Feng. 2019. “SANTIS: Sampling-Augmented Neural neTwork with Incoherent Structure for MR Image Reconstruction.” Magnetic Resonance in Medicine: Official Journal of the Society of Magnetic Resonance in Medicine / Society of Magnetic Resonance in Medicine 82 (5): 1890–1904.

Macé, Emilie, Gabriel Montaldo, Ivan Cohen, Michel Baulac, Mathias Fink, and Mickael Tanter. 2011. “Functional Ultrasound Imaging of the Brain.” Nature Methods 8 (8): 662–64.

“Machine Learning for Image Reconstruction.” 2020. In Handbook of Medical Image Computing and Computer Assisted Intervention, 25–64. Academic Press.

Oksuz, Ilkay, James Clough, Aurelien Bustin, Gastao Cruz, Claudia Prieto, Rene Botnar, Daniel Rueckert, Julia A. Schnabel, and Andrew P. King. 2018. “Cardiac MR Motion Artefact Correction from K-Space Using Deep Learning-Based Reconstruction.” In Machine Learning for Medical Image Reconstruction, 21–29. Springer International Publishing.

Ongie, Gregory, Ajil Jalal, Christopher A. Metzler, Richard G. Baraniuk, Alexandros G. Dimakis, and Rebecca Willett. 2020. “Deep Learning Techniques for Inverse Problems in Imaging.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2005.06001.

Pal, Arghya, and Yogesh Rathi. 2021. “A Review of Deep Learning Methods for MRI Reconstruction.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2109.08618.

Paliwal, Aditya, Sarah Loos, Markus Rabe, Kshitij Bansal, and Christian Szegedy. 2020. “Graph Representations for Higher-Order Logic and Theorem Proving.” Proceedings of the AAAI Conference on Artificial Intelligence. https://doi.org/10.1609/aaai.v34i03.5689.

Polu, Stanislas, and Ilya Sutskever. 2020. “Generative Language Modeling for Automated Theorem Proving.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2009.03393.

Rabe, Markus N., Dennis Lee, Kshitij Bansal, and Christian Szegedy. 2020. “Mathematical Reasoning via Self-Supervised Skip-Tree Training.” arXiv Preprint arXiv:2006. 04757. https://openreview.net/pdf?id=xhKm6VAQmm8.

Radford, Alec, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, et al. 2021. “Learning Transferable Visual Models From Natural Language Supervision.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2103.00020.

Recht, Michael P., Jure Zbontar, Daniel K. Sodickson, Florian Knoll, Nafissa Yakubova, Anuroop Sriram, Tullie Murrell, et al. 2020. “Using Deep Learning to Accelerate Knee MRI at 3 T: Results of an Interchangeability Study.” AJR. American Journal of Roentgenology 215 (6): 1421–29.

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. 2015. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1505.04597.

Song, Yang, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020. “Score-Based Generative Modeling through Stochastic Differential Equations.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2011.13456.

Sriram, Anuroop, Jure Zbontar, Tullie Murrell, Aaron Defazio, C. Lawrence Zitnick, Nafissa Yakubova, Florian Knoll, and Patricia Johnson. 2020. “End-to-End Variational Networks for Accelerated MRI Reconstruction.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2004.06688.

Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. 2017. “Deep Image Prior.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1711.10925.

Wagner, Adam Zsolt. 2021. “Constructions in Combinatorics via Neural Networks.” arXiv [math.CO]. arXiv. http://arxiv.org/abs/2104.14516.

Williams, Alex, Erin Kunz, Simon Kornblith, and Scott Linderman. 2021. “Generalized Shape Metrics on Neural Representations.” Advances in Neural Information Processing Systems 34. https://proceedings.neurips.cc/paper/2021/file/252a3dbaeb32e7690242ad3b556e626b-Paper.pdf.

Zbontar, Jure, Florian Knoll, Anuroop Sriram, Tullie Murrell, Zhengnan Huang, Matthew J. Muckley, Aaron Defazio, et al. 2018. “fastMRI: An Open Dataset and Benchmarks for Accelerated MRI.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1811.08839. -->