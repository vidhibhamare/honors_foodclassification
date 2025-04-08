# # import multiprocessing
# import multiprocessing
# import torch
# import math
# from torch.cuda.amp import GradScaler
# from torch.distributed.elastic.multiprocessing import errors
# from sklearn.metrics import precision_recall_fscore_support
# import matplotlib.pyplot as plt
# import pandas as pd
# import os

# from utils import logger
# from options.opts import get_training_arguments
# from utils.common_utils import device_setup, create_directories
# from utils.ddp_utils import is_master, distributed_init
# from cvnets import get_model, EMA
# from loss_fn import build_loss_fn
# from optim import build_optimizer
# from optim.scheduler import build_scheduler
# from data import create_train_val_loader
# from utils.checkpoint_utils import load_checkpoint, load_model_state
# from engine import Trainer
# from common import (
#     DEFAULT_EPOCHS,
#     DEFAULT_ITERATIONS,
#     DEFAULT_MAX_ITERATIONS,
#     DEFAULT_MAX_EPOCHS,
# )

# import warnings
# import experiments_config


# @errors.record
# def main(opts, **kwargs):
#     num_gpus = getattr(opts, "dev.num_gpus", 0)
#     dev_id = getattr(opts, "dev.device_id", torch.device("cpu"))
#     device = getattr(opts, "dev.device", torch.device("cpu"))
#     is_distributed = getattr(opts, "ddp.use_distributed", False)
#     is_master_node = is_master(opts)

#     # Metric tracking setup
#     training_metrics = {
#         'iterations': [],
#         'precisions': [],
#         'recalls': [],
#         'f1_scores': []
#     }

#     def log_metrics(current_iteration, y_true, y_pred):
#         p, r, f1, _ = precision_recall_fscore_support(
#             y_true.cpu(), y_pred.cpu(), average='weighted', zero_division=0
#         )
#         training_metrics['iterations'].append(current_iteration)
#         training_metrics['precisions'].append(p)
#         training_metrics['recalls'].append(r)
#         training_metrics['f1_scores'].append(f1)
        
#         if current_iteration % 500 == 0 and is_master_node:
#             save_metrics_plot(opts, training_metrics)

#     def save_metrics_plot(opts, metrics):
#         plt.figure(figsize=(12, 6))
#         plt.plot(metrics['iterations'], metrics['precisions'], 'b-', label='Precision')
#         plt.plot(metrics['iterations'], metrics['recalls'], 'g-', label='Recall')
#         plt.plot(metrics['iterations'], metrics['f1_scores'], 'r-', label='F1-Score')
        
#         # Add smoothed lines
#         window_size = max(1, len(metrics['iterations']) // 20)
#         if window_size > 1:
#             plt.plot(metrics['iterations'], 
#                     pd.Series(metrics['precisions']).rolling(window_size).mean(),
#                     'b--', alpha=0.5)
#             plt.plot(metrics['iterations'], 
#                     pd.Series(metrics['recalls']).rolling(window_size).mean(),
#                     'g--', alpha=0.5)
#             plt.plot(metrics['iterations'], 
#                     pd.Series(metrics['f1_scores']).rolling(window_size).mean(),
#                     'r--', alpha=0.5)
        
#         plt.xlabel('Iterations')
#         plt.ylabel('Score')
#         plt.title('Training Metrics Over Time')
#         plt.legend()
#         plt.grid(True)
        
#         plot_path = os.path.join(getattr(opts, "common.exp_loc"), "training_metrics.png")
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         plt.close()

#     # set-up data loaders
#     train_loader, val_loader, train_sampler = create_train_val_loader(opts)

#     # compute max iterations based on max epochs
#     is_iteration_based = getattr(opts, "scheduler.is_iteration_based", False)
#     if is_iteration_based:
#         max_iter = getattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
#         if max_iter is None or max_iter <= 0:
#             logger.log("Setting max. iterations to {}".format(DEFAULT_ITERATIONS))
#             setattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
#             max_iter = DEFAULT_ITERATIONS
#         setattr(opts, "scheduler.max_epochs", DEFAULT_MAX_EPOCHS)
#         if is_master_node:
#             logger.log("Max. iteration for training: {}".format(max_iter))
#     else:
#         max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
#         if max_epochs is None or max_epochs <= 0:
#             logger.log("Setting max. epochs to {}".format(DEFAULT_EPOCHS))
#             setattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
#         setattr(opts, "scheduler.max_iterations", DEFAULT_MAX_ITERATIONS)
#         max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
#         if is_master_node:
#             logger.log("Max. epochs for training: {}".format(max_epochs))

#     # set-up the model
#     model = get_model(opts)

#     # memory format
#     memory_format = (
#         torch.channels_last
#         if getattr(opts, "common.channels_last", False)
#         else torch.contiguous_format
#     )

#     if num_gpus == 0:
#         logger.warning(
#             "No GPUs are available, so training on CPU. Consider training on GPU for faster training"
#         )
#         model = model.to(device=device, memory_format=memory_format)
#     elif num_gpus == 1:
#         model = model.to(device=device, memory_format=memory_format)
#     elif is_distributed:
#         model = model.to(device=device, memory_format=memory_format)
#         model = torch.nn.parallel.DistributedDataParallel(
#             model,
#             device_ids=[dev_id],
#             output_device=dev_id,
#             find_unused_parameters=getattr(opts, "ddp.find_unused_params", False),
#         )
#         if is_master_node:
#             logger.log("Using DistributedDataParallel for training")
#     else:
#         model = model.to(memory_format=memory_format)
#         model = torch.nn.DataParallel(model)
#         model = model.to(device=device)
#         if is_master_node:
#             logger.log("Using DataParallel for training")

#     # setup criteria
#     criteria = build_loss_fn(opts)
#     criteria = criteria.to(device=device)

#     # create the optimizer
#     optimizer = build_optimizer(model, opts=opts)

#     # create the gradient scalar
#     gradient_scalar = GradScaler(enabled=getattr(opts, "common.mixed_precision", False))

#     # LR scheduler
#     scheduler = build_scheduler(opts=opts)

#     model_ema = None
#     use_ema = getattr(opts, "ema.enable", False)

#     if use_ema:
#         ema_momentum = getattr(opts, "ema.momentum", 0.0001)
#         model_ema = EMA(model=model, ema_momentum=ema_momentum, device=device)
#         if is_master_node:
#             logger.log("Using EMA")

#     best_metric = (
#         0.0 if getattr(opts, "stats.checkpoint_metric_max", False) else math.inf
#     )

#     start_epoch = 0
#     start_iteration = 0
#     resume_loc = getattr(opts, "common.resume", None)
#     finetune_loc = getattr(opts, "common.finetune_imagenet1k", None)
#     auto_resume = getattr(opts, "common.auto_resume", False)
#     if resume_loc is not None or auto_resume:
#         (
#             model,
#             optimizer,
#             gradient_scalar,
#             start_epoch,
#             start_iteration,
#             best_metric,
#             model_ema,
#         ) = load_checkpoint(
#             opts=opts,
#             model=model,
#             optimizer=optimizer,
#             model_ema=model_ema,
#             gradient_scalar=gradient_scalar,
#         )
#     elif finetune_loc is not None:
#         model, model_ema = load_model_state(opts=opts, model=model, model_ema=model_ema)
#         if is_master_node:
#             logger.log("Finetuning model from checkpoint {}".format(finetune_loc))

#     # Custom Trainer Class with Metrics Tracking
#     class MetricsTrackingTrainer(Trainer):
#         def training_iteration(self, *args, **kwargs):
#             output = super().training_iteration(*args, **kwargs)
            
#             if is_master_node:
#                 current_iter = self.epoch * len(self.train_loader) + self.batch_idx
                
#                 if current_iter % 100 == 0:
#                     with torch.no_grad():
#                         samples, targets = args[0], args[1]
#                         outputs = self.model(samples)
#                         _, preds = torch.max(outputs, 1)
#                         log_metrics(current_iter, targets, preds)
            
#             return output

#     training_engine = MetricsTrackingTrainer(
#         opts=opts,
#         model=model,
#         validation_loader=val_loader,
#         training_loader=train_loader,
#         optimizer=optimizer,
#         criterion=criteria,
#         scheduler=scheduler,
#         start_epoch=start_epoch,
#         start_iteration=start_iteration,
#         best_metric=best_metric,
#         model_ema=model_ema,
#         gradient_scalar=gradient_scalar,
#     )

#     training_engine.run(train_sampler=train_sampler)

#     # Final metrics save
#     if is_master_node:
#         save_metrics_plot(opts, training_metrics)
#         metrics_path = os.path.join(getattr(opts, "common.exp_loc"), "training_metrics.csv")
#         pd.DataFrame(training_metrics).to_csv(metrics_path, index=False)
#         logger.log(f"Training metrics saved to {metrics_path}")


# def distributed_worker(i, main, opts, kwargs):
#     setattr(opts, "dev.device_id", i)
#     torch.cuda.set_device(i)
#     setattr(opts, "dev.device", torch.device(f"cuda:{i}"))

#     ddp_rank = getattr(opts, "ddp.rank", None)
#     if ddp_rank is None:  # torch.multiprocessing.spawn
#         ddp_rank = kwargs.get("start_rank", 0) + i
#         setattr(opts, "ddp.rank", ddp_rank)

#     node_rank = distributed_init(opts)
#     setattr(opts, "ddp.rank", node_rank)
#     main(opts, **kwargs)


# def main_worker(**kwargs):
#     warnings.filterwarnings("ignore")
#     experiments_config.ehfr_net_config_forward(dataset_name="food256", width_multiplier=1.75)
#     opts = get_training_arguments()
    
#     # device set-up
#     opts = device_setup(opts)

#     node_rank = getattr(opts, "ddp.rank", 0)
#     if node_rank < 0:
#         logger.error("--rank should be >=0. Got {}".format(node_rank))

#     is_master_node = is_master(opts)

#     # create the directory for saving results
#     save_dir = getattr(opts, "common.results_loc", "results")
#     run_label = getattr(opts, "common.run_label", "run_1")
#     exp_dir = "{}/{}".format(save_dir, run_label)
#     setattr(opts, "common.exp_loc", exp_dir)
#     create_directories(dir_path=exp_dir, is_master_node=is_master_node)

#     num_gpus = getattr(opts, "dev.num_gpus", 1)
#     world_size = getattr(opts, "ddp.world_size", -1)
#     use_distributed = not getattr(opts, "ddp.disable", False)
#     if num_gpus <= 1:
#         use_distributed = False
#     setattr(opts, "ddp.use_distributed", use_distributed)

#     # No of data workers = no of CPUs (if not specified or -1)
#     n_cpus = multiprocessing.cpu_count()
#     dataset_workers = getattr(opts, "dataset.workers", -1)

#     norm_name = getattr(opts, "model.normalization.name", "batch_norm")
#     ddp_spawn = not getattr(opts, "ddp.no_spawn", False)
#     if use_distributed and ddp_spawn and torch.cuda.is_available():
#         # get device id
#         dev_id = getattr(opts, "ddp.device_id", None)
#         setattr(opts, "dev.device_id", dev_id)

#         if world_size == -1:
#             logger.log(
#                 "Setting --ddp.world-size the same as the number of available gpus"
#             )
#             world_size = num_gpus
#             setattr(opts, "ddp.world_size", world_size)

#         if dataset_workers == -1 or dataset_workers is None:
#             setattr(opts, "dataset.workers", n_cpus // num_gpus)

#         start_rank = getattr(opts, "ddp.rank", 0)
#         setattr(opts, "ddp.rank", None)
#         kwargs["start_rank"] = start_rank
#         setattr(opts, "ddp.start_rank", start_rank)
#         torch.multiprocessing.spawn(
#             fn=distributed_worker,
#             args=(main, opts, kwargs),
#             nprocs=num_gpus,
#         )
#     else:
#         if dataset_workers == -1:
#             setattr(opts, "dataset.workers", n_cpus)

#         if norm_name in ["sync_batch_norm", "sbn"]:
#             setattr(opts, "model.normalization.name", "batch_norm")

#         # adjust the batch size
#         train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
#         val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
#         setattr(opts, "dataset.train_batch_size0", train_bsize)
#         setattr(opts, "dataset.val_batch_size0", val_bsize)
#         setattr(opts, "dev.device_id", None)
#         main(opts=opts, **kwargs)


# if __name__ == "__main__":
#     main_worker()


###latestttt
# import glob
# import torch
# import sys
# import os
# import json
# import argparse
# from PIL import Image

# from cvnets import get_model
# from options.opts import get_training_arguments
# from torchvision import transforms as T


# def create_image_classes_dict(data_path):
#     assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)

#     image_class = [cla for cla in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cla))]
#     image_class.sort()

#     class_indices = dict((k, v) for v, k in enumerate(image_class))
#     json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
#     with open('./image_classes_dict/class_indices.json', 'w') as json_file:
#         json_file.write(json_str)


# def set_model_argument():
#     sys.argv.append('--common.config-file')
#     sys.argv.append('config/classification/food_image/ehfr_net_food101.yaml')

#     sys.argv.append('--model.classification.n-classes')
#     sys.argv.append('101')  # Change this to 256 if you're using food256 dataset


# def set_args(image_path: str = None):
#     # set the device
#     sys.argv.append('--use-cuda')

#     # set the path that is used to analysis
#     if image_path:
#         sys.argv.append('--image-path')
#         sys.argv.append(image_path)
#     else:
#         sys.argv.append('--image-path')
#         sys.argv.append('/kaggle/input/food-101-split/kaggle/working/split_dataset/val/*/*.jpg')  # Changed to your validation path

#     # set the weights path
#     sys.argv.append('--weights_path')
#     sys.argv.append(r'./cam_relative_file/food101/ehfr_net/checkpoint_ema_best.pt')


# def get_args_other():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--use-cuda', action='store_true', default=False,
#                       help='Use NVIDIA GPU acceleration')
#     parser.add_argument(
#         '--image-path',
#         type=str,
#         default='./examples/both.png',
#         help='Input image path')

#     parser.add_argument('--weights_path', type=str, default=None, help='Input weights path')
#     parser.add_argument('--common.config-file', type=str, default=None, help='Test')
#     parser.add_argument('--model.classification.n-classes', type=int, default=None, help='the number of classification')

#     args = parser.parse_args()
#     return args


# def get_image_name(path_org):
#     name = os.path.basename(path_org)
#     return name


# def predict(image_path, model, data_transform, device):
#     img = Image.open(image_path)

#     img = data_transform(img)
#     # expand batch dimension
#     img = torch.unsqueeze(img, dim=0)

#     json_path = './image_classes_dict/class_indices.json'
#     assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

#     with open(json_path, "r") as f:
#         class_indict = json.load(f)

#     model.eval()
#     with torch.no_grad():
#         # predict class
#         output = torch.squeeze(model(img.to(device))).cpu()
#         predict = torch.softmax(output, dim=0)

#     max_prob = 0
#     class_name = 0
#     for i in range(len(predict)):
#         if predict[i].numpy() > max_prob:
#             class_name = class_indict[str(i)]
#             max_prob = predict[i].numpy()

#     image_name = get_image_name(image_path)
#     print("image: {} The most likely species: {:10}   it's prob: {:.3}".format(image_name, class_name, max_prob))


# def predict_run(model):
#     set_args()
#     opts = get_args_other()

#     img_size = 256
#     data_transform = T.Compose(
#         [T.Resize(size=288, interpolation=Image.BICUBIC),
#          T.CenterCrop(img_size),
#          T.ToTensor()])

#     if opts.use_cuda and torch.cuda.is_available():
#         device = "cuda:0"
#     else:
#         device = "cpu"

#     model = model.to(device)

#     model.load_state_dict(torch.load(opts.weights_path, map_location=device))
#     for image_name in glob.glob(opts.image_path):
#         predict(image_path=image_name, model=model, data_transform=data_transform, device=device)


# def setup_model():
#     set_model_argument()
#     opts = get_training_arguments()

#     # set-up the model
#     model = get_model(opts)

#     set_args()
#     opts = get_args_other()

#     if opts.use_cuda and torch.cuda.is_available():
#         device = "cuda:0"
#     else:
#         device = "cpu"

#     model = model.to(device)
#     model.load_state_dict(torch.load(opts.weights_path, map_location=device))

#     return model, device


# def main():
#     # Create class indices JSON if it doesn't exist
#     data_path = '/kaggle/input/food-101-split/kaggle/working/split_dataset/train'  # Changed to your train path
#     classes_json_path = './image_classes_dict/class_indices.json'
    
#     # Create directory if it doesn't exist
#     os.makedirs('./image_classes_dict', exist_ok=True)
    
#     if not os.path.exists(classes_json_path):
#         create_image_classes_dict(data_path)
    
#     # Initialize model and run prediction
#     set_model_argument()
#     opts = get_training_arguments()
#     model = get_model(opts)
#     predict_run(model=model)


# if __name__ == '__main__':
#     main()
import multiprocessing
import torch
import math
from torch.cuda.amp import GradScaler
from torch.distributed.elastic.multiprocessing import errors
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import zipfile

from utils import logger
from options.opts import get_training_arguments
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master, distributed_init
from cvnets import get_model, EMA
from loss_fn import build_loss_fn
from optim import build_optimizer
from optim.scheduler import build_scheduler
from data import create_train_val_loader
from utils.checkpoint_utils import load_checkpoint, load_model_state
from engine import Trainer
from common import (
    DEFAULT_EPOCHS,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_EPOCHS,
)

import warnings
import experiments_config

def package_model_for_download(exp_dir, model_name="ehfr_net_food101"):
    """
    Package the model and related files into a zip for download
    """
    # Create necessary directories
    os.makedirs(os.path.join(exp_dir, "model_package"), exist_ok=True)
    
    # Create a zip file with all necessary components
    zip_path = os.path.join(exp_dir, f"{model_name}_package.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add model files
        model_files = [
            "checkpoint_ema_best.pt",
            "trained_model.pth",
            "model_state_dict.pth"
        ]
        
        for file in model_files:
            file_path = os.path.join(exp_dir, file)
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.join("model_package", file))
        
        # Add configuration files
        config_files = [
            "config/classification/food_image/ehfr_net_food101.yaml",
            "training_metrics.csv",
            "training_metrics.png"
        ]
        
        for file in config_files:
            if os.path.exists(file):
                zipf.write(file, os.path.join("model_package", os.path.basename(file)))
        
        # Add class indices
        class_idx_path = './image_classes_dict/class_indices.json'
        if os.path.exists(class_idx_path):
            zipf.write(class_idx_path, os.path.join("model_package", "class_indices.json"))
    
    return zip_path

@errors.record
def main(opts, **kwargs):
    num_gpus = getattr(opts, "dev.num_gpus", 0)
    dev_id = getattr(opts, "dev.device_id", torch.device("cpu"))
    device = getattr(opts, "dev.device", torch.device("cpu"))
    is_distributed = getattr(opts, "ddp.use_distributed", False)
    is_master_node = is_master(opts)

    # Metric tracking setup
    training_metrics = {
        'iterations': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': []
    }

    def log_metrics(current_iteration, y_true, y_pred):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true.cpu(), y_pred.cpu(), average='weighted', zero_division=0
        )
        training_metrics['iterations'].append(current_iteration)
        training_metrics['precisions'].append(p)
        training_metrics['recalls'].append(r)
        training_metrics['f1_scores'].append(f1)
        
        if current_iteration % 500 == 0 and is_master_node:
            save_metrics_plot(opts, training_metrics)

    def save_metrics_plot(opts, metrics):
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['iterations'], metrics['precisions'], 'b-', label='Precision')
        plt.plot(metrics['iterations'], metrics['recalls'], 'g-', label='Recall')
        plt.plot(metrics['iterations'], metrics['f1_scores'], 'r-', label='F1-Score')
        
        window_size = max(1, len(metrics['iterations']) // 20)
        if window_size > 1:
            plt.plot(metrics['iterations'], 
                    pd.Series(metrics['precisions']).rolling(window_size).mean(),
                    'b--', alpha=0.5)
            plt.plot(metrics['iterations'], 
                    pd.Series(metrics['recalls']).rolling(window_size).mean(),
                    'g--', alpha=0.5)
            plt.plot(metrics['iterations'], 
                    pd.Series(metrics['f1_scores']).rolling(window_size).mean(),
                    'r--', alpha=0.5)
        
        plt.xlabel('Iterations')
        plt.ylabel('Score')
        plt.title('Training Metrics Over Time')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(getattr(opts, "common.exp_loc"), "training_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    # set-up data loaders
    train_loader, val_loader, train_sampler = create_train_val_loader(opts)

    # compute max iterations based on max epochs
    is_iteration_based = getattr(opts, "scheduler.is_iteration_based", False)
    if is_iteration_based:
        max_iter = getattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
        if max_iter is None or max_iter <= 0:
            logger.log("Setting max. iterations to {}".format(DEFAULT_ITERATIONS))
            setattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
            max_iter = DEFAULT_ITERATIONS
        setattr(opts, "scheduler.max_epochs", DEFAULT_MAX_EPOCHS)
        if is_master_node:
            logger.log("Max. iteration for training: {}".format(max_iter))
    else:
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if max_epochs is None or max_epochs <= 0:
            logger.log("Setting max. epochs to {}".format(DEFAULT_EPOCHS))
            setattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        setattr(opts, "scheduler.max_iterations", DEFAULT_MAX_ITERATIONS)
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if is_master_node:
            logger.log("Max. epochs for training: {}".format(max_epochs))

    # set-up the model
    model = get_model(opts)

    # memory format
    memory_format = (
        torch.channels_last
        if getattr(opts, "common.channels_last", False)
        else torch.contiguous_format
    )

    if num_gpus == 0:
        logger.warning(
            "No GPUs are available, so training on CPU. Consider training on GPU for faster training"
        )
        model = model.to(device=device, memory_format=memory_format)
    elif num_gpus == 1:
        model = model.to(device=device, memory_format=memory_format)
    elif is_distributed:
        model = model.to(device=device, memory_format=memory_format)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dev_id],
            output_device=dev_id,
            find_unused_parameters=getattr(opts, "ddp.find_unused_params", False),
        )
        if is_master_node:
            logger.log("Using DistributedDataParallel for training")
    else:
        model = model.to(memory_format=memory_format)
        model = torch.nn.DataParallel(model)
        model = model.to(device=device)
        if is_master_node:
            logger.log("Using DataParallel for training")

    # setup criteria
    criteria = build_loss_fn(opts)
    criteria = criteria.to(device=device)

    # create the optimizer
    optimizer = build_optimizer(model, opts=opts)

    # create the gradient scalar
    gradient_scalar = GradScaler(enabled=getattr(opts, "common.mixed_precision", False))

    # LR scheduler
    scheduler = build_scheduler(opts=opts)

    model_ema = None
    use_ema = getattr(opts, "ema.enable", False)

    if use_ema:
        ema_momentum = getattr(opts, "ema.momentum", 0.0001)
        model_ema = EMA(model=model, ema_momentum=ema_momentum, device=device)
        if is_master_node:
            logger.log("Using EMA")

    best_metric = (
        0.0 if getattr(opts, "stats.checkpoint_metric_max", False) else math.inf
    )

    start_epoch = 0
    start_iteration = 0
    resume_loc = getattr(opts, "common.resume", None)
    finetune_loc = getattr(opts, "common.finetune_imagenet1k", None)
    auto_resume = getattr(opts, "common.auto_resume", False)
    if resume_loc is not None or auto_resume:
        (
            model,
            optimizer,
            gradient_scalar,
            start_epoch,
            start_iteration,
            best_metric,
            model_ema,
        ) = load_checkpoint(
            opts=opts,
            model=model,
            optimizer=optimizer,
            model_ema=model_ema,
            gradient_scalar=gradient_scalar,
        )
    elif finetune_loc is not None:
        model, model_ema = load_model_state(opts=opts, model=model, model_ema=model_ema)
        if is_master_node:
            logger.log("Finetuning model from checkpoint {}".format(finetune_loc))

    class MetricsTrackingTrainer(Trainer):
        def training_iteration(self, *args, **kwargs):
            output = super().training_iteration(*args, **kwargs)
            
            if is_master_node:
                current_iter = self.epoch * len(self.train_loader) + self.batch_idx
                
                if current_iter % 100 == 0:
                    with torch.no_grad():
                        samples, targets = args[0], args[1]
                        outputs = self.model(samples)
                        _, preds = torch.max(outputs, 1)
                        log_metrics(current_iter, targets, preds)
            
            return output

        def after_epoch(self, epoch):
            super().after_epoch(epoch)
            
            if is_master_node:
                # Save checkpoint after each epoch
                self.save_checkpoint(epoch)
                
                # Save best model separately
                if self.is_best_epoch:
                    self.save_checkpoint(epoch, is_best=True)

        def save_checkpoint(self, epoch, is_best=False):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric': self.best_metric,
                'opts': vars(self.opts),
                'training_metrics': training_metrics
            }
            
            # Regular checkpoint
            torch.save(checkpoint, os.path.join(self.opts.common.exp_loc, f"checkpoint_epoch_{epoch}.pth"))
            
            # Best checkpoint
            if is_best:
                best_path = os.path.join(self.opts.common.exp_loc, "checkpoint_best.pth")
                torch.save(checkpoint, best_path)
                
                # Also save complete model
                model_path = os.path.join(self.opts.common.exp_loc, "trained_model.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'opts': vars(self.opts),
                    'class_indices': self.get_class_indices()  # Implement this method
                }, model_path)
                
                # Save state_dict separately
                state_dict_path = os.path.join(self.opts.common.exp_loc, "model_state_dict.pth")
                torch.save(self.model.state_dict(), state_dict_path)

    training_engine = MetricsTrackingTrainer(
        opts=opts,
        model=model,
        validation_loader=val_loader,
        training_loader=train_loader,
        optimizer=optimizer,
        criterion=criteria,
        scheduler=scheduler,
        start_epoch=start_epoch,
        start_iteration=start_iteration,
        best_metric=best_metric,
        model_ema=model_ema,
        gradient_scalar=gradient_scalar,
    )

    training_engine.run(train_sampler=train_sampler)

    # Final metrics save
    if is_master_node:
        save_metrics_plot(opts, training_metrics)
        metrics_path = os.path.join(getattr(opts, "common.exp_loc"), "training_metrics.csv")
        pd.DataFrame(training_metrics).to_csv(metrics_path, index=False)
        logger.log(f"Training metrics saved to {metrics_path}")
        
        # Package model for download
        zip_path = package_model_for_download(getattr(opts, "common.exp_loc"))
        logger.log(f"Model package ready at: {zip_path}")
        logger.log("In Kaggle, this file will be available in the Output tab")

def distributed_worker(i, main, opts, kwargs):
    setattr(opts, "dev.device_id", i)
    torch.cuda.set_device(i)
    setattr(opts, "dev.device", torch.device(f"cuda:{i}"))

    ddp_rank = getattr(opts, "ddp.rank", None)
    if ddp_rank is None:  # torch.multiprocessing.spawn
        ddp_rank = kwargs.get("start_rank", 0) + i
        setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts, **kwargs)

def main_worker(**kwargs):
    warnings.filterwarnings("ignore")
    experiments_config.ehfr_net_config_forward(dataset_name="food101", width_multiplier=1.75)
    opts = get_training_arguments()
    
    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error("--rank should be >=0. Got {}".format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "/kaggle/working/results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    world_size = getattr(opts, "ddp.world_size", -1)
    use_distributed = not getattr(opts, "ddp.disable", False)
    if num_gpus <= 1:
        use_distributed = False
    setattr(opts, "ddp.use_distributed", use_distributed)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    norm_name = getattr(opts, "model.normalization.name", "batch_norm")
    ddp_spawn = not getattr(opts, "ddp.no_spawn", False)
    if use_distributed and ddp_spawn and torch.cuda.is_available():
        # get device id
        dev_id = getattr(opts, "ddp.device_id", None)
        setattr(opts, "dev.device_id", dev_id)

        if world_size == -1:
            logger.log(
                "Setting --ddp.world-size the same as the number of available gpus"
            )
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)

        if dataset_workers == -1 or dataset_workers is None:
            setattr(opts, "dataset.workers", n_cpus // num_gpus)

        start_rank = getattr(opts, "ddp.rank", 0)
        setattr(opts, "ddp.rank", None)
        kwargs["start_rank"] = start_rank
        setattr(opts, "ddp.start_rank", start_rank)
        torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts, kwargs),
            nprocs=num_gpus,
        )
    else:
        if dataset_workers == -1:
            setattr(opts, "dataset.workers", n_cpus)

        if norm_name in ["sync_batch_norm", "sbn"]:
            setattr(opts, "model.normalization.name", "batch_norm")

        # adjust the batch size
        train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
        val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
        setattr(opts, "dataset.train_batch_size0", train_bsize)
        setattr(opts, "dataset.val_batch_size0", val_bsize)
        setattr(opts, "dev.device_id", None)
        main(opts=opts, **kwargs)

if __name__ == "__main__":
    main_worker()