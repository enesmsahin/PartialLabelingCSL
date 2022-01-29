import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os


parser = argparse.ArgumentParser(description='Class Selective Loss for Partial Multi Label Classification.')

parser.add_argument('--model_path', type=str, default='./models/mtresnet_opim_86.72.pth')
parser.add_argument('--pic_path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dataset_type', type=str, default='OpenImagess')
parser.add_argument('--class_description_path', type=str, default='./data/oidv6-class-descriptions.csv')
parser.add_argument('--th', type=float, default=0.97)
parser.add_argument('--top_k', type=float, default=30)
parser.add_argument('--out-file', default="partial_csl_tag_predictions.tsv", type=str)
parser.add_argument('--img_id_file', default="/home/enes/Desktop/VisionAndLanguageExperimental/Dataset/Flickr30k/image_features_vinvl/flickr_vinvl_img_list.tsv", type=str)
parser.add_argument('--allowed_classes_file', type=str, required=True, help="Open Images CSV File indicating boxable classes")
parser.add_argument('--trainable_classes_file', type=str, default="/home/enes/Desktop/VisionAndLanguageExperimental/Dataset/open_images/oidv6-classes-trainable.txt")
parser.add_argument('--all_classes_file', type=str, default="/home/enes/Desktop/VisionAndLanguageExperimental/Dataset/open_images/oidv6-class-descriptions.csv")


def inference(im, model, classes_list_allowed, allowed_indices_in_classes_list, args):
    with open(args.out_file, "w") as out_file:
        with open(args.img_id_file, "r") as img_id_file:
            for img_id in img_id_file:
                img_id = img_id.strip("\n")
                im_path = os.path.join(args.pic_path, img_id + ".jpg")
                try:
                    im = Image.open(im_path)
                except FileNotFoundError:
                    print(f"Image not found: {im_path}")
                    continue

                im_resize = im.resize((args.input_size, args.input_size))
                np_img = np.array(im_resize, dtype=np.uint8)
                tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
                tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
                output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
                np_output = output.cpu().detach().numpy()

                np_output_allowed = np_output[allowed_indices_in_classes_list]
                idx_sort = np.argsort(-np_output_allowed)
                detected_classes = np.array(classes_list_allowed)[idx_sort]
                scores = np_output_allowed[idx_sort]
                idx_th = scores > args.th
                final_detected_classes = detected_classes[idx_th]

                # im.show()

                if len(final_detected_classes) == 0:
                    print("*" * 10)
                    print(f"Detected classes is zero for {im_path}.")
                    print(f"Max score: {scores.max()}")
                    final_detected_classes = [detected_classes[np.argmax(scores)]]
                    print(final_detected_classes)

                line = img_id + "\t" + "["
                line += ",".join(final_detected_classes)
                line += "]\n"
                out_file.write(line)

def display_image(im, tags, filename):

    path_dest = "./results"
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    # plt.rcParams["axes.titlesize"] = 10
    plt.title("Predicted classes: {}".format(tags))
    plt.savefig(os.path.join(path_dest, filename))


def get_allowed_image_tags_for_open_images(image_tag_file_path: str) -> list:
        """Read allowed image tags to a list from the csv file in the format of
        Open Images dataset.

        Args:
            image_tag_file_path (str): path to csv file containing allowed image tags

        Returns:
            list[str]: list of allowed tags
        """
        tags_list = []
        tag_ids_list = []
        with open(image_tag_file_path) as f:
            for line in f:
                tag_id, tag_name = line.split(",", maxsplit=1)
                tag_name = tag_name.strip()
                # tag_name = tag_name.split("(")[0].strip() # Remove tags with explanatory parentheses
                tags_list.append(tag_name)
                tag_ids_list.append(tag_id)

        return tags_list, tag_ids_list

def get_orig_trainable_class_names(trainable_class_file_path, all_classes_file_path, out_file=None):
    trainable_cls_ids = []
    with open(trainable_class_file_path, "r") as trainable_f:
        for line in trainable_f:
            cls_id = line.strip()
            trainable_cls_ids.append(cls_id)

    trainable_cls_names = []
    ordered_trainable_cls_ids = []
    
    if out_file is not None:
        with open(out_file, "w") as out_f:
            with open(all_classes_file_path, "r") as all_f:
                for line_all in all_f:
                    cls_id, cls_name = line_all.split(",", maxsplit=1)
                    if cls_id in trainable_cls_ids:
                        out_f.write(line_all)
                        trainable_cls_names.append(cls_name.strip().strip("\""))
                        ordered_trainable_cls_ids.append(cls_id)

    else:
        with open(all_classes_file_path, "r") as all_f:
            for line_all in all_f:
                cls_id, cls_name = line_all.split(",", maxsplit=1)
                if cls_id in trainable_cls_ids:
                    trainable_cls_names.append(cls_name.strip().strip("\""))
                    ordered_trainable_cls_ids.append(cls_id)

    return trainable_cls_names, ordered_trainable_cls_ids

    
def save_allowed_classes_list(save_path, class_list):
    with open(save_path, "w") as f:
        for cls in class_list:
            f.write(cls + "\n")

def main():
    print('Inference demo with CSL model')

    # Parsing args
    args = parse_args(parser)

    # Setup model
    print('Creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    print(f"Number of Classes: {args.num_classes}")
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    class_list = np.array(list(state['idx_to_class'].values()))
    print('Done\n')

    classes_list_orig, class_ids_list_orig = get_orig_trainable_class_names(args.trainable_classes_file, args.all_classes_file)
    allowed_image_tags, allowed_image_tag_ids = get_allowed_image_tags_for_open_images(args.allowed_classes_file)

    allowed_indices_in_classes_list = [class_ids_list_orig.index(allowed_cls_id) for allowed_cls_id in allowed_image_tag_ids if allowed_cls_id in class_ids_list_orig]

    # Convert class MID format to class description
    df_description = pd.read_csv(args.class_description_path)
    dict_desc = dict(zip(df_description.values[:, 0], df_description.values[:, 1]))
    class_list = [dict_desc[x] for x in class_list]

    classes_list_allowed = np.array(class_list)[allowed_indices_in_classes_list]

    # Inference
    print('Inference...')
    inference(args.pic_path, model, classes_list_allowed, allowed_indices_in_classes_list, args)

    # displaying image
    # display_image(im, tags, os.path.split(args.pic_path)[1])

    # example loss calculation
    # output = model(tensor_batch)
    # loss_func1 = AsymmetricLoss()
    # loss_func2 = AsymmetricLossOptimized()
    # target = output.clone()
    # target[output < 0] = 0  # mockup target
    # target[output >= 0] = 1
    # loss1 = loss_func1(output, target)
    # loss2 = loss_func2(output, target)
    # assert abs((loss1.item() - loss2.item())) < 1e-6

    # plt.show()
    print('done\n')


if __name__ == '__main__':
    main()
