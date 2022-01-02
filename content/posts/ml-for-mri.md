---
title: "Machine Learning for MRI Image Reconstruction"
date: 2022-01-01T13:19:56-05:00
draft: false
math: true
---
Magnetic resonance imaging (MRI) has long scan times, sometimes close to an hour for an exam. This sucks because long scan times makes MRI exams more expensive, less accessible, and unpleasant. {{<hide prompt="How does it feel like to be in an MRI?" uniqueNum="74" no-markdown="True">}}
Imagine hearing this for an hour. <br><br>

{{<youtube "9GZvd_4ot04?start=203">}}

{{</hide>}}

Here, I review some methods in machine learning that aim to reduce the scan time through new ways of image reconstruction. Smarter image reconstruction allows us to acquire way less data, which means shorter scan times. These techniques are pretty general and can be applied to other image reconstruction problems.

*Disclaimer: This is not meant to be a comprehensive review. Rather, it is just a sample of some methods I found cool.*


## MRI Image Reconstruction
In most medical imaging methods, what you see on the screen isn’t just a raw feed of what the device’s sensors are picking up.

In MRI, this is what the sensors pick up:

{{< figure src="/ml-for-mri/mri-sensor-data.png" width="50%">}}

How in the world is this data useful? Image reconstruction is this incredible procedure that can turn this mess of sensor data into an actual image. After doing image reconstruction on the sensor data above, we get:

{{< figure src="/ml-for-mri/mri-knee.png" width="50%">}}

Now that's much better! (this is an MRI of the knee.) So how does this magical procedure of turning sensor data into images work? 

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

(Note, we're assuming that we're recording from a single MRI coil, but these methods can be extended to [multi-coil imaging](https://mriquestions.com/what-is-pi.html).)


## Using Less Data
The MRI Gods (linear algebra) tell us that if we want to reconstruct an image with $n$ pixels (or voxels), we need at least $n$ frequencies. {{<hide prompt="Why?" uniqueNum="5">}}
This can be seen using a bit of linear algebra. Since the Fourier transform is linear, we can represent it by an $n\times n$ matrix, say $\mathbf{F}$, with each column of $\mathbf{F}$ corresponding to a different frequency. If we only use a subset of the frequencies, this amounts to removing some of the columns of $\mathbf{F}$. But then the new version of $\mathbf{F}$ has less than $n$ columns, which means the problem of finding an $\mathbf{x}$ such that $\mathbf{F} \mathbf{x}=\mathbf{y}$ is underdetermined. Therefore, there are infinitely many images $\mathbf{x}$ that are consistent with the sensor data.
{{</hide>}}

But the problem with acquiring $n$ frequencies is that it takes a lot of time. This is because MRI scan time scales linearly with the number of frequencies you acquire[^2]. A typical MRI image has on the order of 10 million frequencies, which -- even with many hardware tricks to cut acquisition time -- means an MRI exam typically takes ~40 minutes and can sometimes take as long as an hour. If we could acquire only 1/4th of the frequencies, we can reduce acquisition time by 4x (and therefore MRIs could cost 4x less).

[^2]: To be precise, MRI scan time scales linearly in 2 of the 3 spatial dimensions. We actually get one dimension of frequencies for free. This is from a neat trick known as frequency encoding which allows us to parallelize the acquisition process.

So suppose we drink a bit too much and forget about the linear algebra result, only acquiring a subset of the frequencies. Let's set the data at the frequencies that we didn't acquire to be 0. We can write this as

$$\mathbf{\tilde{y}} = \mathcal{M} \odot \mathbf{y} = \mathcal{M} \odot \mathcal{F} (\mathbf{x}^*)$$
where $\mathcal{M}$ is a masking matrix filled with 0s and 1s, and $\odot$ denotes element-wise multiplication. If we try to reconstruct the same knee MRI data as above with less frequencies, we get (aliasing) artifacts:

{{<figure src="/ml-for-mri/simple-compressed-recon.png" width="75%">}}

{{<hide prompt="Why is the mask composed of horizontal lines? And why is the mask more densely sampled near the middle?" uniqueNum="9">}}
For most MRI acquisition methods, there is no time savings to keeping only part of a horizontal line. It's all or nothing.

More of the information in the signal is contained in the low frequencies, so we sample the middle (where the lower frequncies are) more than the rest.

{{</hide>}}

So our dreams of using less frequencies are over, right?

What if we add more information to the image reconstruction process that is not from the current measurement $\mathbf{\tilde{y}} $? For example, in compressed sensing, we can assume that the desired image $\mathbf{x}$ doesn't have many edges (i.e., that we can "compress" the edges). Here's a knee MRI along with its edge map, which we see is very sparse:

{{<figure src="/ml-for-mri/knee-edges.png" width="75%">}}

How do we incorporate the fact that we know that MRI images aren't supposed to have many edges? First, we need some way of counting how many edges are in an MRI image. Edges are places in the image with high spatial derivatives, so a  decent way to count edges is by summing the spatial derivatives (this is called the total variation, and we can write this mathematically as $R_{TV}(\mathbf{x}) = ||\nabla \mathbf{x}||_1$, where $\nabla$ is the spatial gradient and $||.||_1$ is the [L1 norm](https://en.wikipedia.org/wiki/Norm_(mathematics)#Taxicab_norm_or_Manhattan_norm)).



It isn't enough to just look for images that are not so edgy though; we still need our images to match the measurements that we collect (otherwise, we can just make our image blank). We can combine these two components into the following optimization problem:
$$
\argmin_{\mathbf{x}} || \mathcal{M} \odot \mathcal{F}(\mathbf{x}) - \mathbf{\tilde{y}} ||_2^2 + R _{TV}(\mathbf{x})
$$

where $ || . ||_2 $ is the [L2 norm](https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) (i.e., $||z||_2^2 = \sum_i |z_i|^2 $). The left term says: "If $\mathbf{x}$ were the real image, how would the sensor data we'd capture from $\mathbf{x}$ compare with our real sensor data $\mathbf{\tilde{y}}$?" In other words, it tells us how much our reconstruction $\mathbf{x}$ agrees with our measurements $\mathbf{\tilde{y}}$. The right term $R _ {TV} (\mathbf{x})$ penalizes images if they are too edgy. The challenge is finding an image that both agrees with our measurements and isn't too edgy. Algorithms like gradient descent allows us to solve the optimization problem above. {{<hide prompt="What is gradient descent?" uniqueNum="16" >}}

Gradient descent is an iterative algorithm to minimze some function $\mathcal{L}(\boldsymbol{\theta})$. It starts at some initial parameters ${\boldsymbol{\theta}}^{(0)}$ and updates its parameters in the direction of the gradient, $\nabla L({\boldsymbol{\theta}})$, so as to locally reduce the loss function as much as possible. In the $t$-th iteration, ${\boldsymbol{\theta}}^t$ is updated to ${\boldsymbol{\theta}}^{t+1}$ via
$$
    {\boldsymbol{\theta}}^{t+1} = {\boldsymbol{\theta}}^{t} - \alpha^{t} \nabla \mathcal{L}({\boldsymbol{\theta}})
$$
where $t$ is the iteration number, $\alpha^{t}$ is called the learning rate, ${\boldsymbol{\theta}}^{t}$ and ${\boldsymbol{\theta}}^{{t+1}}$ are the parameters from the previous and current iterations, respectively.


You might worry that gradient descent gets stuck in local minima, but in practice, for neural networks with a huge amount of parameters, the minima found by gradient descent turn out to be really good ones! To my knowledge, we still don't fully understand why this is.

{{</hide >}}

Though compressed sensing can improve the image quality relative to a vanilla inverse Fourier transform, it still suffers from artifacts. We see below on a 4x subsampled knee MRI that TV regularization makes some improvements over the inverse Fourier transform ([source](http://arxiv.org/abs/1811.08839)):

{{< figure src="/ml-for-mri/CS.png" width="75%">}}

Maybe just saying "MRI images shouldn't be very edgy" isn't enough information to cut the sensor data by a factor of 4. So other methods of compressed sensing might say "MRI images should be [sparse](https://en.wikipedia.org/wiki/Sparse_matrix)" or "MRI images should be sparse in a [wavelet basis](https://en.wikipedia.org/wiki/Wavelet_transform)." These methods do this by replacing $R_{TV}(\mathbf{x})$ with a more general $R(\mathbf{x})$, which we call a regularizer. The difficulty with classical compressed sensing is that humans must manually encode what an MRI should look like through the regularizer $R(\mathbf{x})$​. We can come up with basic heuristics like the examples above, but ultimately deciding whether an image looks like it could have come from an MRI is a complicated process. {{<hide prompt="How do you interpret R(x) using information theory?" uniqueNum="1">}}

If you use maximum _a posteriori_ estimation, you can show that $R(\mathbf{x}) \propto -\log p(\mathbf{x})$! We call $p(\mathbf{x})$ the prior distribution. So $ R(\mathbf{x})$ is really a measure of how many bits you need to encode your image $\mathbf{x}$ with your prior.

{{</hide >}}

Enter machine learning... Over the past decade-ish, machine learning has had great success in learning functions that humans have difficulty hard coding. It has revolutionized the fields of computer vision, natural language processing, among others. Instead of hard coding functions, machine learning algorithms learn functions from data. In the next section, we will explore a few recent machine learning approaches to MRI image reconstruction.

{{<hide prompt="Did you know Terence Tao was one of the pioneers of compressed sensing?" uniqueNum="12">}}
It turns out [Terence Tao](https://en.wikipedia.org/wiki/Terence_Tao)'s [most cited paper](https://ieeexplore.ieee.org/abstract/document/1580791) is from his work on compressed sensing!
{{</hide>}}

## Machine Learning Comes to the Rescue

### A Naive Approach

We can throw a simple convolutional neural network (CNN) at the problem. {{<hide prompt="What is a convolutional neural network?" uniqueNum="2">}}

A convolutional neural network -- like other neural networks -- learns a function from data. CNNs, in particular, are used for learning functions whose input is an image. The output of a CNN could be a class (for example, given an image, the CNN could say which animal is in it), another image (for example, the CNN could denoise an image).

A CNN is composed of a series of convolutions composed with simple nonlinear functions between the convolutions.

Why convolutions? Convolutions apply the same operation to every region in its input, so it is robust to translations of the image. Robustness to translations is a great property to have for image processing!

Why nonlinear functions? Composing convolutions with more convolutions results in just another convolution. But if we add nonlinear functions in between the convolutions, our network becomes much more expressive! In fact, there is a theorem that states that neural networks made up of linear functions intertwined with a set of simple nonlinear functions can approximate any continuous function!

CNNs have achieved incredible success on a variety of computer vision problems. See this [great course](https://cs231n.github.io/convolutional-networks/) to learn more about CNNs.
{{</hide >}}



The CNN could take in the sensor measurements $\mathbf{\tilde{y}}$ and output its predicted image $\mathbf{\hat{x}}$[^3]. After collecting a dataset that includes both measurements $\mathbf{\tilde{y}}$ and properly reconstructed images $\mathbf{{x}^*}$, you can train the neural network to make its predicted images as close to the ground truth images.

[^3]: Typically in machine learning, we use $\mathbf{x}$ to represent the input, and $\mathbf{y}$ as the output. But since image reconstruction is an "inverse problem," we use the opposite notation.

The problem with this approach is that we don't tell the neural network anything about the physics of MRI. This means it has to learn to do MRI image reconstruction from scratch, which will require a lot of training data.

### U-Net MRI
How can we tell our machine learning method about the physics of MRI image reconstruction? One idea is to first turn the sensor data into an image via an inverse Fourier transform before feeding it into a CNN. Now, the CNN would just have to "clean up" what was missed by the inverse Fourier transform. This is the approach taken by [U-Net MRI](http://arxiv.org/abs/1811.08839), where the CNN is chosen to be a U-Net. [U-Nets](https://arxiv.org/abs/1505.04597) are a popular image-to-image model for biomedical applications. 

{{<figure src="/ml-for-mri/unet-diagram.png" width="100%">}}

We can formally write the operations performed by this network as
$$ \mathbf{\hat{x}} = \text{UNET}_{{\boldsymbol{\theta}}}(\mathcal{F}^{-1}(\mathbf{\tilde{y}}))$$

where $\mathbf{\tilde{y}}$ is the subsampled sensor data, and $\text{UNET}_{\boldsymbol{\theta}}$ is the U-Net parameterized by a vector of parameters ${\boldsymbol{\theta}}$. The parameters ${\boldsymbol{\theta}}$ of the U-Net are optimized in order to minimize the following loss function.

$$    \mathcal{L}({\boldsymbol{\theta}}) = \sum_{(\mathbf{\tilde{y}},{\mathbf{x}}^*) \in \mathcal{D}} ||\text{UNET}_{\boldsymbol{\theta}}(\mathcal{F}^{-1}(\mathbf{\tilde{y}})) - \mathbf{x^{\*}} ||_1$$

where $\mathbf{\tilde{y}}$ and $\mathbf{x}^\*$ are subsampled sensor data and ground truth images, respectively, sampled from the dataset $\mathcal{D}$. In words, our neural network takes as its input subsampled sensor data $\mathbf{\tilde{y}}$ and tries to output $\text{UNET}_{\boldsymbol{\theta}}(\mathcal{F}^{-1}(\mathbf{\tilde{y}}))$ that is as close to the real image ${\mathbf{x}}^*$ as possible. The parameters ${\boldsymbol{\theta}}$ are optimized via gradient descent (or slightly cooler versions of gradient descent).

In the figure below, we see a significant qualitative improvement in the reconstructions from the U-Net, in comparison with traditional compressed sensing with total variation regularization.

{{<figure src="/ml-for-mri/unet.png" width="75%" caption=`**Knee MRI reconstructions comparison between compressed sensing with total variation regularization and the fastMRI U-Net baseline.** The data is acquired using multiple coils at 4x and 8x subsampling. Reproduced from [Zbontar et al. 2018](http://arxiv.org/abs/1811.08839).`>}}

{{<hide prompt="Wait, but where does the training data come from?" uniqueNum="17">}}
In 2019, Facebook AI released an MRI dataset called [fastMRI](http://arxiv.org/abs/1811.08839). The dataset contains 8344 brain and knee MRI scans. The scans contain raw fully sampled sensor data as well as the corresponding image reconstructions. The scans were done with a variety of MRI parameters (different pulse sequences, field strengths, repetition times, and echo times). The diversity of parameters is important: we want image reconstruction methods to work for all relevant clinical parameters. The dataset also includes 20,000 brain and knee MRI scans that only contain the reconstructed images and not the sensor data (it is also not straightforward to get the raw frequency-domain data from the images as there are multiple coils and postprocessing).

The dataset consists of both a training set and a test set. The training set is used to set the parameters of the model, and the test set is used to evaluate the model.
{{</hide>}}

### VarNet

Instead of just feeding a physics-informed guess to a U-Net, VarNet uses the physics of MRI at multiple steps in its network ([Sriram et al. 2020](http://arxiv.org/abs/2004.06688) & [Hammernik et al. 2018](http://arxiv.org/abs/1704.00447)). Recently, an [interchangeability study of VarNet](https://www.ajronline.org/doi/10.2214/AJR.20.23313) was done. It found that 1/4-th of the data with VarNet was diagnostically interchangeable with the ground truth reconstructions. In other words, radiologists made the same diagnoses with both methods! {{<hide prompt="Tell me a funny story about this study" uniqueNum="14">}}

At first, the physicians thought the VarNet images didn't look great because the images were too smooth. So the authors added some random Gaussian noise, and then the physicians loved the images! In fact, the authors give a fancy name to their process of adding random noise; they call it "adaptive image dithering."
{{</hide>}}

Below is a sample reconstruction from their study, compared with the ground truth. I can't tell the difference.

{{<figure src="/ml-for-mri/varnet.png" width="75%" caption=`**Knee MRI comparison between VarNet and the ground truth at 4x acceleration.** Figure reproduced from [Recht et al. 2020](https://www.ajronline.org/doi/10.2214/AJR.20.23313).`>}}

So how does VarNet work? It starts with a blank image, and consists of a series of refinement steps, progressively turning the blank image into a better and better version.

{{<figure src="/ml-for-mri/varnet-diagram.png" width="100%">}}

<!-- caption=`**The process of VarNet.** An image starts off as blank and is updated iteratively via (9), producing a better image at each step. To train VarNet, the image at the final $T$th step is compared with the ground truth via a loss function, $\mathcal{L}(\boldsymbol{\theta})$, and the parameters of VarNet, $\boldsymbol{\theta}$, are updated via gradient descent.`-->

Let's take a look at where the refinement step comes from. Recall that in classical compressed sensing, we solve the optimization problem above. Writing the forward operator $\mathbf{A}=\mathcal{M} \odot \mathcal{F}$, the optimization problem for compressed sensing becomes:

$$ \argmin_{\mathbf{x}} || \mathbf{A}\mathbf{x} - \mathbf{\tilde{y}} ||_2^2 + R(\mathbf{x})$$

This is not the optimization problem for VarNet, but we will use a cool trick, called unrolled optimization:

If we solve the compressed sensing objective function via gradient descent, we get the following update equation for the $t$-th iteration of the image, ${\mathbf{x}}^t$.

$${\mathbf{x}}^{t+1} = {\mathbf{x}}^t - \alpha^t (\mathbf{A}^*(\mathbf{A}{\mathbf{x}}^t - \mathbf{\tilde{y}}) + \nabla R({\mathbf{x}}^t))$$

where $\mathbf{A}^*$ is the [adjoint](https://en.wikipedia.org/wiki/Conjugate_transpose) of $\mathbf{A}$. Note that gradient descent in the above equation is done on the image $\mathbf{x}$, as opposed to ${\boldsymbol{\theta}}$ . Now here's the trick! Instead of hard coding the regularizer $R(\mathbf{x}^t)$, we can replace it with a neural network. We do this by replacing $\nabla R(\mathbf{x}^t)$ with a CNN. We get a new update equation:

$${\mathbf{x}}^{t+1} = {\mathbf{x}}^t - \alpha^t \mathbf{A}^*(\mathbf{A}{\mathbf{x}}^t - \mathbf{\tilde{y}}) + \text{CNN}_{\boldsymbol{\theta}} ({\mathbf{x}}^t)$$

The VarNet architecture consists of multiple layers. Each layer takes the output of the previous layer, ${\mathbf{x}}^t$, as its input, and outputs ${\mathbf{x}}^{t+1}$ according to the above equation. In practice, VarNet has about 8 layers, and the CNN is a U-Net. The parameters of the U-Net are updated via gradient descent on $\boldsymbol{\theta}$, and the loss function $\mathcal{L}({\boldsymbol{\theta}})$ is taken to be the structural similarity index measure (SSIM). {{<hide prompt="What is SSIM?" uniqueNum="3">}}
The SSIM is a measure of similarity for images that is more aligned with the human perceptual system than the mean-squared error. It compares two images across three dimensions: luminosity, contrast, and structural similarity. A great explanation of SSIM can be found in [this blog post](https://bluesky314.github.io/ssim/).
{{</hide>}}



Technically, the approach above isn't quite the [latest version of VarNet](http://arxiv.org/abs/2004.06688): there were a few changes that improve things a tiny bit. {{<hide prompt="What things?">}}
* Updating in sensor-space instead of in image space. The sensor-domain update can be obtained by taking the Fourier transform of both sides of the update equation:

$$
    \mathbf{y}^{t+1} = \mathbf{y}^t - \alpha^t \mathcal{M} \odot (\mathbf{y}^t - \mathbf{\tilde{y}}) + \mathcal{F}(\text{CNN}_{\boldsymbol{\theta}} (\mathcal{F}(\mathbf{y}^t)))
$$

* Learning a 2nd CNN to estimate the sensitivity maps of each coil in the case of multi-coil (parallel) imaging
{{</hide>}}



### Deep Generative Priors
All methods above required access to a dataset that had both MRI images and the raw sensor data. However, to my understanding, the raw sensor data is not typically saved on clinical MRIs. Constructing a dataset with only the MRI images and without the raw sensor data might be easier. Fortunately, there are machine learning methods that only require MRI images as training data (i.e., [unsupervised models](https://en.wikipedia.org/wiki/Unsupervised_learning)). 

One approach is to train what is called a [generative model](https://en.wikipedia.org/wiki/Generative_model#Deep_generative_models). Generative models are very popular in the computer vision community for generating realistic human faces or scenes (that it has never seen before!). Similarly, we can train a generative model to generate new MRI-like images.

A generative MRI model is a function $G_{\boldsymbol{\theta}}$ that tries to turn any random vector $\mathbf{z} \in \mathbb{R}^m$ into a realistic image $\mathbf{x} \in \mathbb{R}^n$. Typicaly,  $m \ll n$, i.e., the input space is often much smaller than the output space. 

Image reconstruction with generative models is done by solving the optimization problem:
\begin{equation}
    \argmin_{\mathbf{z}} ||\mathbf{A} G_{\boldsymbol{\theta}}(\mathbf{z}) - \mathbf{\tilde{y}}||_2^2
\end{equation}

Instead of optimizing over all images $x \in \mathbb{R}^n$, we optimize only over the images produced by the generator, $G_{\boldsymbol{\theta}}(\mathbb{R}^m)$. Since $m \ll n$, the range of the generator $G_{\boldsymbol{\theta}}(\mathbb{R}^m)$ is much smaller than $\mathbb{R}^n$. {{<hide prompt="What if m=n?" uniqueNum="19">}}It turns out it can still work if we use early stopping! This says something deep about the optimization landscape. Early stopping still implicitly restricts the range of the generator.{{</hide>}}

An important question is how well these models generalize outside of their training set. This is especially important for diagnosing rare conditions that might not appear in the training set. [Jalal et al. 2021](http://arxiv.org/abs/2108.01368) recently showed that you can get pretty extraordinary generalization using a type of generative model called a [score-based generative model](https://yang-song.github.io/blog/2021/score/). As seen in the results below, they train their model on brain data and test it on a completely different anatomy -- in this case the abdomen! Their model performs much better in this case than other approaches.

{{<figure src="/ml-for-mri/dgp.png" width="75%" caption=`**Reconstructions of 2D abdominal scans at 4x acceleration for methods trained on brain MRI data.** The red arrows points to artifacts in the images. The deep generative prior method from [Jalal 2021](http://arxiv.org/abs/2108.01368) does not have the artifacts from the other methods. Results from [Jalal 2021](http://arxiv.org/abs/2108.01368).`>}}

Why generative models generalize so well, I don't fully understand yet, but the authors do [give some theoretical justification](http://arxiv.org/abs/2108.01368). A limitation to image reconstruction using deep generative priors is that the reconstruction time is typically longer than methods like VarNet (it can be more than 15 minutes on a modern GPU). This is because the optimization process needs to be run at test time.
<!-- you're not learning the forward operator -->

### Untrained Neural Networks
Imagine we get drunk again and forget to feed our machine learning model any data. We should get nonsense right...? Well, recently, it's been [shown](http://arxiv.org/abs/2007.02471) that even with no data at all, the models in machine learning can be competitive with fully trained machine learning methods for MRI image reconstruction.

How do you explain this? First, let's see how these models work. These no-data methods start with the deep generative priors approach in the previous section. But instead of using data to train the generator $G_{\boldsymbol{\theta}}(\mathbf{z})$, we set the parameters ${\boldsymbol{\theta}}$ randomly. The structure of these ML models -- the fact that they're made of convolutions, for example -- make it such that without any data, realistic images are more likely to be generated than random noise.

This is remarkable! And confusing! We started off by saying that machine learning removes the need to manually engineer regularizers for compressed sensing. But instead, we are manually engineering the architectures of machine learning models! How much are these machine learning models *really* learning?

It turns out such untrained models have been applied to other inverse problems like region inpainting, denoising, and super resolution, and they have achieved [remarkable results](http://arxiv.org/abs/1711.10925). Below are some results of an untrained model, [ConvDecoder](http://arxiv.org/abs/2007.02471), on 4x subsampled data in MRI. We see that even though ConvDecoder is untrained, it produces better reconstructions than U-Net and TV-regularized compressed sensing.

{{<figure src="/ml-for-mri/convdecoder.png" width="75%" caption=`**Comparison of the untrained [ConvDecoder](http://arxiv.org/abs/2007.02471) with the [U-Net MRI](http://arxiv.org/abs/1811.08839), and total-variation regularized compressed sensing.** Reconstructions of knee-MRI at 4x acceleration. The second row is a zoomed in version of the first row. Figure reproduced from [Darestani et al. 2020](http://arxiv.org/abs/2007.02471).`>}}

## Concluding Thoughts
Machine learning methods have made significant progress in reducing the scan time of MRI. Not only have ML methods for compressed sensing produced strong results on quantitative metrics like SSIM, but they have started to be [validated by clinicians](https://www.ajronline.org/doi/10.2214/AJR.20.23313). Validation by clinicians is essential in image reconstruction because a fine detail can be essential in a diagnosis but might not make its way into a metric like the mean-squared-error.

A limitation to deep learning for healthcare is that we still don't have a good understanding of *why* deep learning works. This makes it hard to predict when and how deep learning methods will fail (there are no theoretical guarantees that deep learning will work). One tool to help in this regard is uncertainty quantification: instead of only outputting a reconstructed image, you'd also output how much confidence you have in this image. Stochastic methods like deep generative priors can estimate the uncertainty in their reconstruction by creating many reconstructions with different random seeds and computing the standard deviation. For non-generative methods, works like [Edupuganti 2019](http://arxiv.org/abs/1901.11228) make use of Stein's unbiased risk estimate (SURE) to estimate uncertainty.

In addition to MRI, machine learning methods have also been used for other forms of image reconstruction. A great review can be found [here](http://arxiv.org/abs/2005.06001).

_**A big thank you** to [Milan Cvitkovic](https://milan.cvitkovic.net/), [Stephen Fay](https://stephenfay.xyz/), Jonathan Kalinowski, [Hannah Le](https://twitter.com/hannah_1e), and [Marley Xiong](https://marleyx.com) for reviewing drafts of this._

<!-- ## References
Akçakaya, Mehmet, Steen Moeller, Sebastian Weingärtner, and Kâmil Uğurbil. 2019. “Scan-Specific Robust Artificial-Neural-Networks for K-Space Interpolation (RAKI) Reconstruction: Database-Free Deep Learning for Fast Imaging.” Magnetic Resonance in Medicine: Official Journal of the Society of Magnetic Resonance in Medicine / Society of Magnetic Resonance in Medicine 81 (1): 439–53.

Barbano, Riccardo, Željko Kereta, Chen Zhang, Andreas Hauptmann, Simon Arridge, and Bangti Jin. 2020. “Quantifying Sources of Uncertainty in Deep Learning-Based Image Reconstruction.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2011.08413.

Cole, Elizabeth K., John M. Pauly, Shreyas S. Vasanawala, and Frank Ong. 2020. “Unsupervised MRI Reconstruction with Generative Adversarial Networks.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2008.13065.

“CS231n Convolutional Neural Networks for Visual Recognition.” n.d. Accessed November 27, 2021. https://cs231n.github.io/convolutional-networks/.

Daras, Giannis, Joseph Dean, Ajil Jalal, and Alexandros G. Dimakis. 2021. “Intermediate Layer Optimization for Inverse Problems Using Deep Generative Models.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2102.07364.

Darestani, Mohammad Zalbagi, and Reinhard Heckel. 2020a. “Accelerated MRI with Un-Trained Neural Networks.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2007.02471.

Defazio, Aaron, Tullie Murrell, and Michael P. Recht. 2020. “MRI Banding Removal via Adversarial Training.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2001.08699.

Deora, Rahul. n.d. “A Brief Introduction to SSIM: Structural Similarity Index.” Accessed November 27, 2021. https://bluesky314.github.io/ssim/.

Edupuganti, Vineet, Morteza Mardani, Shreyas Vasanawala, and John Pauly. 2019. “Uncertainty Quantification in Deep MRI Reconstruction.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1901.11228.

Hammernik, Kerstin, Teresa Klatzer, Erich Kobler, Michael P. Recht, Daniel K. Sodickson, Thomas Pock, and Florian Knoll. 2017. “Learning a Variational Network for Reconstruction of Accelerated MRI Data.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1704.00447.

He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. “Deep Residual Learning for Image Recognition.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1512.03385.

Huang, Chin-Wei, Jae Hyun Lim, and Aaron Courville. 2021. “A Variational Perspective on Diffusion-Based Generative Models and Score Matching.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2106.02808.

Jalal, Ajil, Marius Arvinte, Giannis Daras, Eric Price, Alexandros G. Dimakis, and Jonathan I. Tamir. 2021. “Robust Compressed Sensing MRI with Deep Generative Priors.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2108.01368.

Jalal, Ajil, Liu Liu, Alexandros G. Dimakis, and Constantine Caramanis. 2020. “Robust Compressed Sensing Using Generative Models.” Advances in Neural Information Processing Systems. https://github.com/ajiljalal/csgm-robust-neurips.

Liu, Fang, Alexey Samsonov, Lihua Chen, Richard Kijowski, and Li Feng. 2019. “SANTIS: Sampling-Augmented Neural neTwork with Incoherent Structure for MR Image Reconstruction.” Magnetic Resonance in Medicine: Official Journal of the Society of Magnetic Resonance in Medicine / Society of Magnetic Resonance in Medicine 82 (5): 1890–1904.

“Machine Learning for Image Reconstruction.” 2020. In Handbook of Medical Image Computing and Computer Assisted Intervention, 25–64. Academic Press.

Oksuz, Ilkay, James Clough, Aurelien Bustin, Gastao Cruz, Claudia Prieto, Rene Botnar, Daniel Rueckert, Julia A. Schnabel, and Andrew P. King. 2018. “Cardiac MR Motion Artefact Correction from K-Space Using Deep Learning-Based Reconstruction.” In Machine Learning for Medical Image Reconstruction, 21–29. Springer International Publishing.

Ongie, Gregory, Ajil Jalal, Christopher A. Metzler, Richard G. Baraniuk, Alexandros G. Dimakis, and Rebecca Willett. 2020. “Deep Learning Techniques for Inverse Problems in Imaging.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2005.06001.

Pal, Arghya, and Yogesh Rathi. 2021. “A Review of Deep Learning Methods for MRI Reconstruction.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2109.08618.

Recht, Michael P., Jure Zbontar, Daniel K. Sodickson, Florian Knoll, Nafissa Yakubova, Anuroop Sriram, Tullie Murrell, et al. 2020. “Using Deep Learning to Accelerate Knee MRI at 3 T: Results of an Interchangeability Study.” AJR. American Journal of Roentgenology 215 (6): 1421–29.

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. 2015. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1505.04597.

Song, Yang, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020. “Score-Based Generative Modeling through Stochastic Differential Equations.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2011.13456.

Sriram, Anuroop, Jure Zbontar, Tullie Murrell, Aaron Defazio, C. Lawrence Zitnick, Nafissa Yakubova, Florian Knoll, and Patricia Johnson. 2020. “End-to-End Variational Networks for Accelerated MRI Reconstruction.” arXiv [eess.IV]. arXiv. http://arxiv.org/abs/2004.06688.

Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. 2017. “Deep Image Prior.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1711.10925.

Zbontar, Jure, Florian Knoll, Anuroop Sriram, Tullie Murrell, Zhengnan Huang, Matthew J. Muckley, Aaron Defazio, et al. 2018. “fastMRI: An Open Dataset and Benchmarks for Accelerated MRI.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1811.08839. -->
