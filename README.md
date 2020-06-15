# Gradient Parameter Estimate

## Motivation

In ***Deep Leakage in Gradients*** shows the possibility of obtaining private training data from the corresponding gradient. Furthermore, in ***Improved Deep Leakage in Gradients*** proved that gradients definitely leaks the ground-truth labels and presented an accurate and reliable method (iDLG) to extract accurate data from its gradients. Both paper claimed its method is independent to the deep neural network parameters. Thus, we are curious about the relationship between the difficulty and efficiency of extracting data from its gradients corresponding to the model parameters amount and architecture.

## Tasks

- [ ] Model architecture design with different parameters amount magnitude.
- [ ] Randomly pick 3 images from each category of MNIST.
- [ ] Pick top three high gradients magnitude of each model corresponding to each category of MNIST
- [ ] Train, validate, and test models on MNIST.
- [ ] Run DLG and iDLG on models with the chosen images and record the MSE corresponding to iterations.



## Expetiments Design

### Dataset Information



### Model Summary

| No.  | Model        | ConvNet Channels    | Trainable Params # | Train Acc. | Val. Acc. | Test. Acc. |
| :--- | :----------- | :------------------ | -----------------: | ---------: | --------: | ---------: |
| 1    | CNN_L2D1     | [32, 64]            |             34,026 |        63% |           |        63% |
| 2    | CNN_L2D2     | [64, 128]           |            133,578 |        74% |           |        74% |
| 3    | CNN_L4D1     | [32, 64, 128, 256]  |            691,690 |        95% |           |        95% |
| 4    | CNN_L4D2     | [64, 128, 256, 512] |          2,759,626 |        94% |           |        94% |
| 5    | **CNN_Base** | x                   |          1,199,882 |            |           |            |



### Randomly Picked Images



### iDLG MSE w.r.t Iterations Curves