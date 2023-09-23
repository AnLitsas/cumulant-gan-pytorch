# cumulant-gan-pytorch
## PyTorch implementation of Cumulant GAN

This repository contains a PyTorch implementation of the Cumulant GAN model, as described in the original paper. The Cumulant GAN is an advanced Generative Adversarial Network designed to tackle a variety of challenges, including instability, mode collapse, and sample quality.

---

## This implementation includes three toy examples:

- GMM8 Dataset
- TMM6 Dataset
- Swiss roll dataset

---

## Results for Swiss roll dataset

<div align="center">
  <table>
    <tr>
      <th>Wasserstein Distance</th>
      <th>Reverse KLD Distance</th>
      <th>Hellinger Distance</th>
    </tr>
    <tr>
      <td><img src="./Results/swiss_roll_2d_with_labels_0_0.gif" alt="Wasserstein Distance" width="200"/></td>
      <td><img src="./Results/swiss_roll_2d_with_labels_1_0.gif" alt="Reverse KLD Distance" width="200"/></td>
      <td><img src="./Results/swiss_roll_2d_with_labels_0.5_0.5.gif" alt="Hellinger Distance" width="200"/></td>
    </tr>
  </table>
</div>

---

## Results for GMM8 dataset

### Hellinger Distance

<div align="center">
  <img src="./Results/toy_example_gmm8_0.5_0.5.gif" alt="GMM8" width="400"/>
</div>

---

## Acknowledgments

- The paper can be found [here](https://arxiv.org/pdf/2006.06625.pdf).
- Dip's implementation in TensorFlow 2 (Dip is one of the authors) can be found [here](https://github.com/dipjyoti92/CumulantGAN/tree/main/).
- For a Tensorflow 2.x implementation (not by me), visit [this link](https://github.com/andrewkof/Cumulant-GAN).
