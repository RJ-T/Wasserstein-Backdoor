# Wasserstein-Backdoor
Unofficial  implementation of NeurIPS 2021 paper Backdoor Attack with Imperceptible Input and Latent Modification.


# Implementations of Backdoor Attacks


## Setup the environment
Solve the enviroments referring to the original repo https://github.com/khoadoan106/backdoor_attacks.
Python 3.9 + torch 1.10+ should work well.

## Implementations

### Backdoor attack with imperceptible input and latent modification (NeurIPS 2021).  
* [Paper](https://proceedings.neurips.cc/paper/2021/file/9d99197e2ebf03fc388d09f1e94af89b-Paper.pdf)
* Stage 1: Trigger Generation - LIRA learns to generate the trigger in Stage 1. Examples:
    * MNIST
        ```
        python lira_trigger_generation.py --dataset mnist --clsmodel mnist_cnn --path experiments/ --epochs 10  --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.1 --avoid_cls_reinit 2>&1 >experiments/logs/mnist_trigger_generation.log &
        ```
    * CIFAR10
        ```
        python lira_trigger_generation.py --dataset cifar10 --clsmodel vgg11 --path experiments/ --epochs 50 --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.1 --avoid_cls_reinit 2>&1 >experiments/logs/cifar10_trigger_generation.log &	
        ```
* Stage 2: Backdoor Injection. After the trigger is learned, LIRA poison and fine-tune the classifier in Stage 2. Examples:
    * MNIST
        ```
        . etc/setup_env
         python lira_backdoor_injection.py --dataset mnist --clsmodel mnist_cnn --path experiments/ --epochs 50 --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.1 --avoid_cls_reinit --test_eps 0.01 --test_alpha 0.5 --test_epochs 50 --test_lr 0.01 --schedulerC_lambda 0.1 --schedulerC_milestones 10,20,30,40 2>&1 >experiments/logs/mnist_backdoor_injection.log &	
        ```
    * CIFAR10
        ```
        . etc/setup_env
        python lira_backdoor_injection.py --dataset cifar10 --clsmodel vgg11 --path experiments/ --epochs 50 --train-epoch 1 --mode all2one --target_label 0 --epochs_per_external_eval 10 --cls_test_epochs 5 --verbose 2 --batch-size 128 --alpha 0.5 --eps 0.1 --avoid_cls_reinit --test_eps 0.01 --test_alpha 0.5 --test_epochs 500 --test_lr 0.01 --schedulerC_lambda 0.1 --schedulerC_milestones 100,200,300,400 2>&1 >experiments/logs/cifar10_backdoor_injection.log &		
        ```
        
Please cite the paper, as below, when using this repository:
```
@inproceedings{doan2021lira,
  title={Lira: Learnable, imperceptible and robust backdoor attacks},
  author={Doan, Khoa and Lao, Yingjie and Zhao, Weijie and Li, Ping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11966--11976},
  year={2021}
}
@article{doan2021backdoor,
  title={Backdoor attack with imperceptible input and latent modification},
  author={Doan, Khoa and Lao, Yingjie and Li, Ping},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={18944--18957},
  year={2021}
}
```
