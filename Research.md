# Research: Mathematical Foundations of Drone Video Enhancement

This document provides a comprehensive theoretical foundation for understanding the image processing and signal processing concepts used in this project. The goal is to explain the mathematics behind deblurring, denoising, and video enhancement without diving into implementation details.

## Table of Contents

1. [Introduction: The Image Degradation Problem](#1-introduction-the-image-degradation-problem)
2. [Mathematical Model of Image Degradation](#2-mathematical-model-of-image-degradation)
3. [Fourier Transform Fundamentals](#3-fourier-transform-fundamentals)
4. [Point Spread Function (PSF)](#4-point-spread-function-psf)
5. [Deconvolution: The Inverse Problem](#5-deconvolution-the-inverse-problem)
6. [Wiener Deconvolution](#6-wiener-deconvolution)
7. [Richardson-Lucy Algorithm](#7-richardson-lucy-algorithm)
8. [Tikhonov Regularization](#8-tikhonov-regularization)
9. [Denoising Techniques](#9-denoising-techniques)
10. [Video Stabilization with Optical Flow](#10-video-stabilization-with-optical-flow)
11. [Contrast Enhancement: CLAHE](#11-contrast-enhancement-clahe)
12. [References](#12-references)

---

## 1. Introduction: The Image Degradation Problem

When capturing video from drones, several factors cause image quality degradation:

- **Motion blur**: Caused by camera movement during exposure
- **Defocus blur**: Caused by incorrect focus distance
- **Atmospheric turbulence**: Air movement causing light path distortion
- **Sensor noise**: Electronic noise from the image sensor

The fundamental challenge is to recover the original sharp image from the observed degraded version. This is known as an **inverse problem** in mathematics and signal processing.

### Why Classical Methods?

This project uses classical digital signal processing (DSP) methods rather than AI/ML approaches because:

1. **Mathematical Guarantees**: Classical methods have well-understood theoretical properties
2. **Computational Efficiency**: Real-time processing on GPU without large model inference
3. **Interpretability**: Parameters have clear physical meaning
4. **No Training Data Required**: Works directly on any input without pre-training

---

## 2. Mathematical Model of Image Degradation

### The Convolution Model

Image degradation can be mathematically modeled as:

$$g(x, y) = h(x, y) * f(x, y) + n(x, y)$$

Where:
- $f(x, y)$ — Original (sharp) image we want to recover
- $h(x, y)$ — Point Spread Function (PSF) — the blur kernel
- $g(x, y)$ — Observed (blurred) image
- $n(x, y)$ — Additive noise
- $*$ — Convolution operator

The **convolution operation** is defined as:

$$(h * f)(x, y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} h(x - x', y - y') \cdot f(x', y') \, dx' \, dy'$$

For discrete images, this becomes:

$$(h * f)[m, n] = \sum_{i} \sum_{j} h[i, j] \cdot f[m - i, n - j]$$

### Physical Interpretation

The convolution model has a clear physical interpretation:
- Each pixel in the observed image is a **weighted sum** of nearby pixels in the original image
- The weights are determined by the PSF
- This "spreads" point sources across neighboring pixels, causing blur

---

## 3. Fourier Transform Fundamentals

### Why Fourier Transform?

The Fourier Transform is essential for efficient image processing because of the **Convolution Theorem**:

> Convolution in the spatial domain equals multiplication in the frequency domain.

This transforms the computationally expensive convolution operation into simple element-wise multiplication.

### 2D Discrete Fourier Transform

For an image $f[m, n]$ of size $M \times N$, the 2D DFT is:

$$F[u, v] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} f[m, n] \cdot e^{-j2\pi(\frac{um}{M} + \frac{vn}{N})}$$

The inverse transform recovers the original:

$$f[m, n] = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F[u, v] \cdot e^{j2\pi(\frac{um}{M} + \frac{vn}{N})}$$

Where:
- $F[u, v]$ — Complex Fourier coefficients (frequency domain representation)
- $(u, v)$ — Spatial frequencies
- $j = \sqrt{-1}$ — Imaginary unit

### Properties Relevant to Deconvolution

1. **Linearity**: $\mathcal{F}\{af + bg\} = a\mathcal{F}\{f\} + b\mathcal{F}\{g\}$

2. **Convolution Theorem**: $\mathcal{F}\{h * f\} = H \cdot F$
   - Where $H = \mathcal{F}\{h\}$ and $F = \mathcal{F}\{f\}$

3. **Parseval's Theorem**: Energy is preserved:
   $$\sum_{m,n} |f[m,n]|^2 = \frac{1}{MN} \sum_{u,v} |F[u,v]|^2$$

### Frequency Domain Representation

Using the convolution theorem, our degradation model in frequency domain becomes:

$$G(u, v) = H(u, v) \cdot F(u, v) + N(u, v)$$

Where:
- $G$ — Fourier transform of observed image
- $H$ — Optical Transfer Function (OTF) — Fourier transform of PSF
- $F$ — Fourier transform of original image
- $N$ — Fourier transform of noise

---

## 4. Point Spread Function (PSF)

The PSF describes how a single point of light is spread (blurred) by the imaging system. Understanding PSF is crucial for deblurring.

### Motion Blur PSF

Motion blur occurs when the camera moves during exposure. For linear motion:

$$h_{motion}(x, y) = \begin{cases} \frac{1}{L} & \text{if } \sqrt{x^2 + y^2} \leq L/2 \text{ along motion direction} \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $L$ — Length of motion blur (in pixels)
- Motion direction is parameterized by angle $\theta$

For motion at angle $\theta$:
$$x' = x \cos\theta + y \sin\theta$$
$$y' = -x \sin\theta + y \cos\theta$$

The PSF is a line segment of length $L$ oriented at angle $\theta$.

### Gaussian Blur PSF

Gaussian blur (defocus blur approximation):

$$h_{gaussian}(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

Where:
- $\sigma$ — Standard deviation (blur strength)
- The PSF is symmetric and has infinite extent (in theory)
- In practice, truncated to finite kernel size

### PSF to OTF Conversion

To use PSF in frequency domain computations, we convert it to the Optical Transfer Function (OTF):

1. **Pad** the PSF to match image size
2. **Center** the PSF at the origin (shift so center is at [0,0])
3. **Apply FFT** to get the OTF

The OTF $H(u,v)$ is complex-valued, containing:
- **Modulation Transfer Function (MTF)**: $|H(u,v)|$ — amplitude attenuation
- **Phase Transfer Function (PTF)**: $\angle H(u,v)$ — phase shift

---

## 5. Deconvolution: The Inverse Problem

### Naive Inverse Filtering

The simplest approach to deconvolution is **inverse filtering**:

$$\hat{F}(u, v) = \frac{G(u, v)}{H(u, v)}$$

Then recover $\hat{f}$ by inverse FFT.

### The Problem with Inverse Filtering

Inverse filtering fails catastrophically because:

1. **Division by small values**: Where $|H(u, v)| \approx 0$, the result explodes
2. **Noise amplification**: Even small noise $N$ becomes $N/H$, which can be huge
3. **Zeros in OTF**: Motion blur OTF has zeros (sinc function), causing infinite amplification

**Example**: If $|H| = 0.001$ and $|N| = 0.01$, then $|N/H| = 10$!

### Why Regularization is Necessary

To combat these issues, we need **regularization** — adding constraints or prior knowledge to make the inverse problem well-posed.

The general regularized solution minimizes:

$$\hat{f} = \arg\min_f \left[ \|Hf - g\|^2 + \lambda \cdot R(f) \right]$$

Where:
- $\|Hf - g\|^2$ — Data fidelity term (match observed data)
- $R(f)$ — Regularization term (enforce prior knowledge)
- $\lambda$ — Regularization parameter (balance between terms)

---

## 6. Wiener Deconvolution

### Derivation from Statistical Principles

Wiener filtering is the optimal linear filter in the **minimum mean squared error (MMSE)** sense. It assumes:

- The original image and noise are uncorrelated
- We have statistical knowledge of signal and noise power spectra

### The Wiener Filter

The optimal filter in frequency domain is:

$$W(u, v) = \frac{H^*(u, v)}{|H(u, v)|^2 + K}$$

Where:
- $H^*$ — Complex conjugate of OTF
- $|H|^2 = H \cdot H^*$ — Power spectrum of OTF
- $K$ — Regularization constant (noise-to-signal power ratio)

The restored image estimate:

$$\hat{F}(u, v) = W(u, v) \cdot G(u, v) = \frac{H^*(u, v) \cdot G(u, v)}{|H(u, v)|^2 + K}$$

### Mathematical Derivation

Starting from the degradation model $G = HF + N$, we want to find filter $W$ that minimizes:

$$E\left[|F - WG|^2\right]$$

Using the orthogonality principle and assuming uncorrelated signal and noise:

$$W = \frac{H^* S_f}{|H|^2 S_f + S_n}$$

Where $S_f$ and $S_n$ are the power spectral densities of signal and noise.

If we assume $S_f / S_n \approx 1/K$ (constant), we get the simplified form shown above.

### Interpretation

1. **When $|H|^2 \gg K$**: Filter behaves like inverse filter $\approx 1/H$
2. **When $|H|^2 \ll K$**: Filter suppresses the frequency $\approx H^*/K$
3. **The parameter $K$** controls the trade-off:
   - Small $K$ → more sharpening, more noise
   - Large $K$ → less noise, less sharpening

### Choosing the K Parameter

- **Theoretically**: $K = S_n/S_f$ (noise-to-signal ratio)
- **Practically**: Typical values are $0.001$ to $0.1$
- **Heuristic**: Start with $K = 0.01$ and adjust based on results

---

## 7. Richardson-Lucy Algorithm

### Bayesian Motivation

Richardson-Lucy (RL) is an iterative algorithm based on **Bayesian statistics** and **maximum likelihood estimation** for Poisson noise models.

### Assumptions

1. **Poisson noise model**: Photon counting follows Poisson distribution
   $$P(g|f) = \prod_{i} \frac{(Hf)_i^{g_i}}{g_i!} e^{-(Hf)_i}$$

2. **Non-negativity**: Image values are non-negative (physical constraint)

### The Algorithm

The RL update formula is:

$$f^{(k+1)} = f^{(k)} \cdot \left( h^T * \frac{g}{h * f^{(k)}} \right)$$

Where:
- $f^{(k)}$ — Current estimate at iteration $k$
- $h^T$ — Transposed (flipped) PSF
- $*$ — Convolution operation
- Division is element-wise

### Step-by-Step Process

1. **Initialize**: $f^{(0)} = g$ (start with observed image)

2. **For each iteration $k$**:
   - Compute predicted observation: $\hat{g} = h * f^{(k)}$
   - Compute ratio: $r = g / \hat{g}$
   - Compute correction: $c = h^T * r$
   - Update estimate: $f^{(k+1)} = f^{(k)} \cdot c$

3. **Repeat** until convergence or fixed number of iterations

### FFT-Based Implementation

For efficiency, convolutions are performed in frequency domain:

$$h * f = \mathcal{F}^{-1}\{H \cdot F\}$$
$$h^T * r = \mathcal{F}^{-1}\{H^* \cdot R\}$$

Where $H^*$ is used because the flipped kernel's FFT is the complex conjugate of $H$.

### Properties

**Advantages**:
- Preserves non-negativity (essential for images)
- Good for Poisson noise (common in photon-limited imaging)
- Converges to maximum likelihood estimate

**Disadvantages**:
- Iterative (slower than Wiener)
- Can amplify noise with too many iterations
- Requires stopping criterion (regularization by early stopping)

### Convergence Behavior

- **Early iterations**: Recover low frequencies (large-scale structure)
- **Later iterations**: Recover high frequencies (fine details, but also noise)
- **Optimal stopping**: Balance between detail recovery and noise amplification

---

## 8. Tikhonov Regularization

### Classical Form

Tikhonov regularization adds an $L_2$ penalty on the solution:

$$\hat{f} = \arg\min_f \left[ \|h * f - g\|^2 + \alpha \|f\|^2 \right]$$

Where $\alpha > 0$ is the regularization parameter.

### Solution in Frequency Domain

The closed-form solution is:

$$\hat{F}(u, v) = \frac{H^*(u, v) \cdot G(u, v)}{|H(u, v)|^2 + \alpha}$$

This is **identical to Wiener filtering** when $K = \alpha$ and we assume uniform power spectra.

### Gradient-Based Tikhonov

A more sophisticated version penalizes gradients instead of pixel values:

$$\hat{f} = \arg\min_f \left[ \|h * f - g\|^2 + \alpha \|\nabla f\|^2 \right]$$

Where $\nabla f$ is the gradient of the image.

### Frequency Domain Solution with Gradient Regularization

$$\hat{F}(u, v) = \frac{H^*(u, v) \cdot G(u, v)}{|H(u, v)|^2 + \alpha |L(u, v)|^2}$$

Where $L$ is the Laplacian operator in frequency domain:

$$L(u, v) = \mathcal{F}\left\{\begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}\right\}$$

### Interpretation

- **Zero-order Tikhonov** ($\|\cdot\|^2$): Penalizes large values, tends to smooth everything
- **First-order Tikhonov** ($\|\nabla \cdot\|^2$): Penalizes gradients, preserves constant regions
- **Second-order**: Penalizes curvature, preserves linear ramps

### Choosing α

- Too small: Under-regularization, noise amplification
- Too large: Over-regularization, excessive smoothing
- Methods for selection: L-curve, generalized cross-validation (GCV)

---

## 9. Denoising Techniques

Denoising is often applied as a preprocessing step before deblurring.

### Gaussian Filter

The simplest smoothing filter:

$$g(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right) * f(x, y)$$

**Properties**:
- Removes high-frequency noise
- Also removes high-frequency details (edges become blurry)
- Isotropic (same in all directions)

### Bilateral Filter

Edge-preserving smoothing that considers both **spatial** and **intensity** proximity:

$$g(p) = \frac{1}{W_p} \sum_{q \in \mathcal{N}(p)} G_{\sigma_s}(\|p - q\|) \cdot G_{\sigma_r}(|f(p) - f(q)|) \cdot f(q)$$

Where:
- $G_{\sigma_s}$ — Spatial Gaussian (nearby pixels have more weight)
- $G_{\sigma_r}$ — Range/intensity Gaussian (similar intensity pixels have more weight)
- $W_p$ — Normalization factor

**Why it preserves edges**:
- Near edges, pixels across the edge have different intensities
- Range Gaussian gives them low weight
- So edge pixels don't mix, preserving sharpness

### Non-Local Means (NLM)

A powerful denoising method that exploits **self-similarity** in images:

$$g(p) = \frac{1}{Z(p)} \sum_{q \in \mathcal{S}} w(p, q) \cdot f(q)$$

Where the weight $w(p, q)$ depends on **patch similarity**:

$$w(p, q) = \exp\left(-\frac{\|P(p) - P(q)\|^2}{h^2}\right)$$

- $P(p)$ — Patch centered at pixel $p$
- $h$ — Filtering parameter
- $\mathcal{S}$ — Search window

**Key insight**: Similar patches anywhere in the image contribute to denoising, not just nearby pixels.

**Properties**:
- Excellent at preserving fine details and textures
- Computationally expensive (many patch comparisons)
- Parameter $h$ controls smoothing strength

---

## 10. Video Stabilization with Optical Flow

### The Stabilization Problem

Drone video often has unwanted camera motion (jitter, vibration). Stabilization compensates for this motion to produce smoother video.

### Optical Flow

Optical flow estimates the **apparent motion** of brightness patterns between frames.

For two consecutive frames $I_1$ and $I_2$, we find displacement $(u, v)$ for each pixel such that:

$$I_1(x, y) \approx I_2(x + u, y + v)$$

### Horn-Schunck Method

Based on the **brightness constancy assumption**:

$$I(x, y, t) = I(x + u\delta t, y + v\delta t, t + \delta t)$$

Taylor expansion gives the **optical flow constraint equation**:

$$I_x u + I_y v + I_t = 0$$

Where $I_x, I_y, I_t$ are partial derivatives of intensity.

### Farneback Dense Optical Flow

This project uses Farneback's method, which:

1. **Approximates** each neighborhood with a polynomial:
   $$f(x) \approx x^T A x + b^T x + c$$

2. **Computes** displacement from polynomial coefficient changes between frames

3. **Uses pyramids** for multi-scale estimation (coarse-to-fine)

### Motion Compensation

Once optical flow $(u, v)$ is computed:

1. **Estimate global motion** (e.g., median of flow vectors)
2. **Compute compensation transform** (translation, affine, or homography)
3. **Warp frame** to cancel unwanted motion

For simple translation compensation:
$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix} - \begin{bmatrix} \bar{u} \\ \bar{v} \end{bmatrix}$$

Where $(\bar{u}, \bar{v})$ is the average (or median) motion.

---

## 11. Contrast Enhancement: CLAHE

### Histogram Equalization

Classic histogram equalization transforms intensities to achieve a uniform histogram:

$$T(r) = \text{CDF}(r) = \sum_{j=0}^{r} p(j)$$

Where $p(j)$ is the probability of intensity $j$.

**Problem**: Global equalization can over-amplify noise in uniform regions.

### Contrast Limited Adaptive Histogram Equalization (CLAHE)

CLAHE addresses this by:

1. **Dividing** the image into tiles (e.g., 8×8 grid)
2. **Computing** histogram for each tile
3. **Clipping** the histogram at a threshold (limiting contrast amplification)
4. **Redistributing** clipped pixels uniformly
5. **Applying** histogram equalization per tile
6. **Interpolating** at tile boundaries for smooth transitions

### The Clip Limit

The clip limit $C$ controls maximum contrast amplification:

- If histogram bin count exceeds $C$, excess is redistributed
- Lower $C$ → less contrast amplification, less noise enhancement
- Higher $C$ → more contrast but potential noise amplification

### Mathematical Formulation

For each tile, the clipped histogram $h'$ is:

$$h'(k) = \min(h(k), C)$$

Excess pixels are redistributed:
$$h''(k) = h'(k) + \frac{\sum_j (h(j) - h'(j))}{L}$$

Where $L$ is the number of gray levels.

The transformation function for the tile:
$$T(r) = \sum_{j=0}^{r} \frac{h''(j)}{N}$$

Where $N$ is the number of pixels in the tile.

---

## 12. References

### Core Textbooks

1. **Gonzalez, R.C. & Woods, R.E.** *Digital Image Processing*, 4th Ed. Pearson, 2018.

2. **Jain, A.K.** *Fundamentals of Digital Image Processing*. Prentice Hall, 1989.

3. **Oppenheim, A.V. & Schafer, R.W.** *Discrete-Time Signal Processing*, 3rd Ed. Pearson, 2009.

### Wiener Filtering

4. **Wiener, N.** *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*. MIT Press, 1949.

5. **Hunt, B.R.** "The Application of Constrained Least Squares Estimation to Image Restoration by Digital Computer." *IEEE Trans. Computers*, C-22(9), 1973.

### Richardson-Lucy Algorithm

6. **Richardson, W.H.** "Bayesian-Based Iterative Method of Image Restoration." *Journal of the Optical Society of America*, 62(1), 1972.

7. **Lucy, L.B.** "An iterative technique for the rectification of observed distributions." *The Astronomical Journal*, 79, 1974.

### Tikhonov Regularization

8. **Tikhonov, A.N. & Arsenin, V.Y.** *Solutions of Ill-Posed Problems*. Winston & Sons, 1977.

9. **Hansen, P.C.** *Rank-Deficient and Discrete Ill-Posed Problems*. SIAM, 1998.

### Non-Local Means

10. **Buades, A., Coll, B., & Morel, J.M.** "A Non-Local Algorithm for Image Denoising." *CVPR*, 2005.

### Bilateral Filter

11. **Tomasi, C. & Manduchi, R.** "Bilateral Filtering for Gray and Color Images." *ICCV*, 1998.

### CLAHE

12. **Zuiderveld, K.** "Contrast Limited Adaptive Histogram Equalization." *Graphics Gems IV*, Academic Press, 1994.

### Optical Flow

13. **Horn, B.K. & Schunck, B.G.** "Determining Optical Flow." *Artificial Intelligence*, 17(1-3), 1981.

14. **Farnebäck, G.** "Two-Frame Motion Estimation Based on Polynomial Expansion." *SCIA*, 2003.

### Course Material

15. **Aizenberg, I.** CMPG 767 - Image Processing and Analysis, Lecture 11: Restoration of Blurred Images.
    [Course Link](https://www.igoraizenberg.com/my-classes/cmpg-767-image-processing-and-analysis)

---

## Summary: The Complete Pipeline

The drone video enhancement pipeline can be summarized mathematically:

1. **Input**: Degraded video frames $g[m,n,t]$

2. **Preprocessing**:
   - Denoising: $g_{denoised} = \text{NLM}(g)$ or $\text{Bilateral}(g)$
   - Stabilization: $g_{stable} = T^{-1}(g_{denoised})$ where $T$ is estimated motion

3. **PSF Estimation or Model**:
   - Motion: $h = h_{motion}(L, \theta)$
   - Gaussian: $h = h_{gaussian}(\sigma)$

4. **Deconvolution** (one of):
   - Wiener: $\hat{f} = \mathcal{F}^{-1}\left\{\frac{H^* G}{|H|^2 + K}\right\}$
   - Richardson-Lucy: Iterate $f^{k+1} = f^k \cdot (h^T * (g / (h * f^k)))$
   - Tikhonov: $\hat{f} = \mathcal{F}^{-1}\left\{\frac{H^* G}{|H|^2 + \alpha |L|^2}\right\}$

5. **Post-processing**:
   - CLAHE enhancement for improved contrast

6. **Output**: Enhanced video frames $\hat{f}[m,n,t]$

Each step has a solid mathematical foundation that ensures predictable, reliable results without the "black box" nature of deep learning approaches.
