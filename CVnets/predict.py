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
#     sys.argv.append('101')


# def set_args(image_path: str = None):
#     # set the device
#     sys.argv.append('--use-cuda')

#     # set the path that is used to analysis
#     if image_path:
#         sys.argv.append('--image-path')
#         sys.argv.append(image_path)
#     else:
#         sys.argv.append('--image-path')
#         sys.argv.append(r'.\cam_relative_file\food101\origin\*.jpg')

#     # set the weights path
#     sys.argv.append('--weights_path')
#     sys.argv.append(r'.\cam_relative_file\food101\ehfr_net\checkpoint_ema_best.pt')


# def get_args_other():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--use-cuda', action='store_true', default=False,
#                         help='Use NVIDIA GPU acceleration')
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
#         if i == 0:
#             class_name = class_indict[str(i)]
#             max_prob = predict[i].numpy()
#         elif predict[i].numpy() > max_prob:
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
#     # This needs to be replaced with the path where the data set is located
#     data_path = r'/ai35/Food/Food-101'
#     classes_json_path = './image_classes_dict/class_indices.json'
#     if os.path.exists(classes_json_path):
#         pass
#     else:
#         create_image_classes_dict(data_path)
#     # set_model_argument()
#     set_model_argument()
#     opts = get_training_arguments()

#     # set-up the model
#     model = get_model(opts)
#     # print(model)
#     predict_run(model=model)


# if __name__ == '__main__':
#     main()
import glob
import torch
import sys
import os
import json
import argparse
from PIL import Image
from torchvision import transforms as T

from cvnets import get_model
from options.opts import get_training_arguments

def load_saved_model(model_path, device='cuda'):
    """
    Load a saved model from checkpoint
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # If it's a full model checkpoint
    if 'opts' in checkpoint:
        # Recreate the model architecture
        opts = argparse.Namespace(**checkpoint['opts'])
        model = get_model(opts)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If it's just a state_dict, we need the original model configuration
        # You'll need to set up the model architecture first
        raise ValueError("Full model checkpoint with 'opts' is required for prediction")
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint.get('class_indices', None)

def create_image_classes_dict(data_path, save_dir='./image_classes_dict'):
    assert os.path.exists(data_path), f"Dataset root: {data_path} does not exist."
    
    os.makedirs(save_dir, exist_ok=True)
    image_class = [cla for cla in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cla))]
    image_class.sort()
    
    class_indices = dict((k, v) for v, k in enumerate(image_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    
    json_path = os.path.join(save_dir, 'class_indices.json')
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)
    
    return class_indices

def predict(image_path, model, data_transform, device, class_indices=None):
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
    
    max_prob, pred_class = torch.max(predict, dim=0)
    pred_class = pred_class.item()
    max_prob = max_prob.item()
    
    if class_indices:
        class_name = class_indices.get(str(pred_class), f"Class_{pred_class}")
    else:
        class_name = f"Class_{pred_class}"
    
    image_name = os.path.basename(image_path)
    print(f"image: {image_name} | Predicted: {class_name} | Probability: {max_prob:.3f}")
    
    return pred_class, max_prob

def main():
    # Set up paths
    train_data_path = '/kaggle/input/food-101-split/kaggle/working/split_dataset/train'
    model_path = '/kaggle/working/results/run_1/trained_model.pth'  # Update this path
    
    # Create/Load class indices
    classes_json_path = './image_classes_dict/class_indices.json'
    if not os.path.exists(classes_json_path):
        class_indices = create_image_classes_dict(train_data_path)
    else:
        with open(classes_json_path, 'r') as f:
            class_indices = json.load(f)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, loaded_class_indices = load_saved_model(model_path, device)
    
    # Use loaded class indices if available
    if loaded_class_indices:
        class_indices = loaded_class_indices
    
    # Image transformations
    img_size = 256
    data_transform = T.Compose([
        T.Resize(size=288, interpolation=Image.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor()
    ])
    
    # Prediction on validation set
    val_images_path = '/kaggle/input/food-101-split/kaggle/working/split_dataset/val/*/*.jpg'
    for image_path in glob.glob(val_images_path):
        predict(image_path, model, data_transform, device, class_indices)

if __name__ == '__main__':
    main()