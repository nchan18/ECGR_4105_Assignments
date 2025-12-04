#!
source ../venv/bin/activate
#Final metrics: {'best_val_acc': 0.7819, 'final_val_acc': 0.7819, 'final_val_loss': 0.6728869713783264, 'params': 1070794, 'model_size_MB': 4.084754943847656, 'total_time_s': 739.3834583759308}
#Final metrics: {'best_val_acc': 0.7585, 'final_val_acc': 0.7559, 'final_val_loss': 0.7351767257690429, 'params': 620362, 'model_size_MB': 2.3664932250976562, 'total_time_s': 850.2416291236877}
#Final metrics: {'best_val_acc': 0.7236, 'final_val_acc': 0.1, 'final_val_loss': 2.302585416793823, 'params': 1290602, 'model_size_MB': 4.923255920410156, 'total_time_s': 1669.1335031986237}
python assignment_7.py --model cnn --epochs 300 --batch-size 128 --lr 0.1 --out-dir runs/problem1a

python assignment_7.py --model cnn_extra --epochs 300 --batch-size 128 --lr 0.1 --out-dir runs/problem1b

python assignment_7.py --model resnet10 --epochs 300 --batch-size 128 --lr 0.1 --out-dir runs/resnet10_baseline

python assignment_7.py --model resnet10 --epochs 300 --batch-size 128 --lr 0.1 --weight-decay 0.001 --out-dir runs/resnet10_wd

python assignment_7.py --model resnet10 --epochs 300 --batch-size 128 --lr 0.1 --dropout-p 0.3 --out-dir runs/resnet10_dropout

python assignment_7.py --model resnet10 --epochs 300 --batch-size 128 --lr 0.1 --batchnorm --out-dir runs/resnet10_bn
