python main.py --cfg ./config/CIFAR10_LT/softmax_imba100.yaml 
python main.py --cfg ./config/CIFAR10_LT/softmax_imba100.yaml --test --save_feat 'train'
python main.py --cfg ./config/CIFAR10_LT/softmax_imba100.yaml --test --save_feat 'val'
python main.py --cfg ./config/CIFAR10_LT/balanced_softmax_imba100.yaml
python main.py --cfg ./config/CIFAR10_LT/balanced_softmax_imba100.yaml --test --save_feat 'train'
python main.py --cfg ./config/CIFAR10_LT/balanced_softmax_imba100.yaml --test --save_feat 'val'
python main.py --cfg ./config/CIFAR10_LT/balms_imba100.yaml
python main.py --cfg ./config/CIFAR10_LT/balms_imba100.yaml --test --save_feat 'train'
python main.py --cfg ./config/CIFAR10_LT/balms_imba100.yaml --test --save_feat 'val'
python main.py --cfg ./config/CIFAR10_LT/decouple_balanced_softmax_imba100.yaml
python main.py --cfg ./config/CIFAR10_LT/decouple_balanced_softmax_imba100.yaml --test --save_feat 'train'
python main.py --cfg ./config/CIFAR10_LT/decouple_balanced_softmax_imba100.yaml --test --save_feat 'val'

