# Decomposed Fusion with Soft Prompt (DFSP)
DFSP is a model which decomposes the prompt language feature into state feature and object feature, then fuses them with image feature to improve the response for state and object respectively.


## Setup
```
conda create --name clip python=3.7
conda activate clip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install git+https://github.com/openai/CLIP.git
```
Alternatively, you can use `pip install -r requirements.txt` to install all the dependencies.

## Download Dataset
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.
```
sh download_data.sh
```

## Challenge
The goal of this paper is to conduct experiments on the CZSL dataset to predict the attributes of an object given its known situation. By integrating the expert1 and expert2 models, we aim to reduce the Uncertainty Calibration Error (UCE) of the model and improve its accuracy.

In this study, we also used different evaluation metrics to test the performance of the model and demonstrated that integrating expert1 and expert2 can achieve better performance on the CZSL dataset by comparing different model configurations with different integration methods.

<!-- ## Experimental results
This document introduces three model configurations: ut-zappos(baseline), ut-zappos_ours, and ut-zappos_ours_0.05. The main purpose of these configurations is to compare the accuracy and uncalibrated expected penalty (UCE) of the model under different loss functions.
| Model Configuration | Lattr | Lobj | Lattr_ours | 
| -------- | -------- | -------- | -------- |
| ut-zappos(baseline)| 0.01 | 0.01 | 0 |
| ut-zappos_ours | 0 | 0.01 | 0.01 |
| ut-zappos_ours_0.005 | 0 | 0.005 | 0.005 |

| Model Configuration | ep1 attr acc(↑) | ep2 attr acc(↑) | ep3 attr acc(↑) | ep1 uce(↓) | ep2 uce(↓) | ep3 uce(↓)| table_acc_ep12(↑) | table_acc_ep123_ys(↑) | table_acc_yfs_ep23_nfs_ep1(↑) |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |  -------- | -------- |  -------- |
| ut-zappos(baseline)            |  0.4191 | 0.4184 | 0.4131 | 0.1579 | 0.3663 | 0.1861 | 0.4234▲ | 0.4153 |  0.4181 |
| ut-zappos_ours                 |  0.4013 | 0.4184 | 0.3867 | 0.1024 | 0.3663 | 0.1726 | 0.4159  | 0.4032 |  0.3879 |
| ut-zappos_ours_0.005           |  0.4253 | 0.4184 | 0.4119 | 0.1263 | 0.3663 | 0.1973 | 0.4293▲ | 0.4265 |  0.4290▲|



| Model Configuration | Lattr | Lobj | Lattr_ours | 
| -------- | -------- | -------- | -------- |
| mit-state(baseline)  | 0.01 | 0.01 | 0 |
| mit-states_ours      | 0 | 0.01 | 0.01 |
| mit-states_ours_0.05 | 0 | 0.005 | 0.005 |

| Model Configuration | ep1 attr acc(↑) | ep2 attr acc(↑) | ep3 attr acc(↑) | ep1 uce(↓) | ep2 uce(↓) | ep3 uce(↓)| table_acc_ep12(↑) | table_acc_ep123_ys(↑) | table_acc_yfs_ep23_nfs_ep1(↑) |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |  -------- | -------- |  -------- |
| mit-state(baseline)  |  0.2723 | 0.3047 | 0.2837 | 0.1196 | 0.5593 | 0.4132 | 0.3065▲ | 0.2973 |  0.3197▲ |
| mit-states_ours      |  0.2754 | 0.3047 | 0.2832 | 0.1120 | 0.5593 | 0.4128 | 0.3084▲ | 0.2968 |  0.3213▲ |
| mit-states_ours_0.005|  0.2726 | 0.3047 | 0.2838 | 0.1176 | 0.5593 | 0.4127 | 0.3063▲ | 0.2960 |  0.3197▲ |




| Model Configuration | val ep1 attr acc(↑) | val ep2 attr acc(↑) | val ep1 uce(↓) | 
| -------- | -------- | -------- | -------- | 
| mit-state(baseline)  | 0.2723 | 0.3049 | 0.1196 | 
| mit-states_ours      | 0.2754 | 0.2964 | 0.1120 | 
| mit-states_ours_0.005| 0.2726 | 0.2983 | 0.1176 | 

This document introduces three model configurations: ut-zappos_ours_MC_dropout1, ut-zappos_ours_MC_dropout2, and ut-zappos_ours_MC_dropout(1+2). The main purpose of these configurations is to compare the performance of different MC dropout applications in the model in terms of accuracy and uncalibrated expected penalty (UCE).

ut-zappos_ours_MC_dropout1 is a model configuration that applies MC dropout after the image encoder, while ut-zappos_ours_MC_dropout2 applies MC dropout between the self-attention and cross-attention mechanisms. ut-zappos_ours_MC_dropout(1+2) integrates both of them.

The four model configurations all use the ut-zappos_ours_0.05 loss setting, which impacts the performance and stability of the model during training and testing.

| Model Configuration | Dropout rate | validation Monte-Carlo Dropout steps | 
| -------- | -------- | -------- |
| ut-zappos_ours_MC_dropout1| 0.3 | 3 |
| ut-zappos_ours_MC_dropout2| 0.3 | 3 |
| ut-zappos_ours_MC_dropout(1+2)| 0.3 | 3 |

in val set

| Model Configuration | ep1 attr acc(↑) | ep2 attr acc(↑) | ep3 attr acc(↑) | ep1 uce(↓) | ep2 uce(↓) | ep3 uce(↓)| table_acc_ep12(↑) | table_acc_ep13(↑) | table_acc_ep23(↑) |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |  -------- | -------- |  -------- |
| ut-zappos_ours_MC_dropout1| 0.4038 | 0.4253 | 0.0906 | 0.4211 | 0.4175 | 0.4262▲ |
| ut-zappos_ours_MC_dropout2| 0.3998 | 0.3998 | 0.1536 | 0.4764 | 0.3979 | 0.3991 |
| ut-zappos_ours_MC_dropout(1+2)| 0.3998 | 0.3727 | 0.060 | 0.4540 | 0.3991 | 0.3727 |




| Model Configuration | Dropout rate | (Model selection) validation MC Dropout steps | 
| -------- | -------- | -------- |
| ut-zappos_ours_0.05_MC_dropout(1+2)_step20 | 0.3 | 20 |

| Model Configuration | val ep1 attr acc(↑) | val ep2 attr acc(↑) | val ep1 uce(↓) | val ep2 uce(↓) | table_acc_ep12(↑) | softmax_acc(↑) |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| ut-zappos_ours_0.05_MC_dropout(1+2)_step20| 0.4135 | 0.4153 | 0.075 | 0.4124 | 0.4131 | 0.4169▲ | -->
## block diagram
![image](https://github.com/SerenityOuO/DFSP_project/blob/main/readme_/繪圖.png)


If you already have setup the datasets, you can use symlink and ensure the following paths exist:
`data/<dataset>` where `<datasets> = {'mit-states', 'ut-zappos', 'cgqa'}`.

## Training
```
python -u train.py --dataset <dataset>
```
## Evaluation
We evaluate our models in two settings: closed-world and open-world.

### Closed-World Evaluation
```
python -u test.py --dataset <dataset>
```
You can replace `--dataset` with `{mit-states, ut-zappos}`.


### Open-World Evaluation
For our open-world evaluation, we compute the feasbility calibration and then evaluate on the dataset.

### Feasibility Calibration
We use GloVe embeddings to compute the similarities between objects and attributes.
Download the GloVe embeddings in the `data` directory:

```
cd data
wget https://nlp.stanford.edu/data/glove.6B.zip
```
Move `glove.6B.300d.txt` into `data/glove.6B.300d.txt`.

To compute feasibility calibration for each dataset, run the following command:
```
python -u feasibility.py --dataset mit-states
```
The feasibility similarities are saved at `data/feasibility_<dataset>.pt`.
To run, just edit the open-world parameter in `config/<dataset>.yml`


### MOM and ERV-SoP
```
python -u test_v2.py --dataset config1_<dataset>
```
To run, just edit the open-world parameter in `config1_<dataset>.yml`

    
## References
If you use this code, please cite
```
@article{lu2022decomposed,
  title={Decomposed Soft Prompt Guided Fusion Enhancing for Compositional Zero-Shot Learning},
  author={Lu, Xiaocheng and Liu, Ziming and Guo, Song and Guo, Jingcai},
  journal={arXiv preprint arXiv:2211.10681},
  year={2022}
}
```