import os
import torch
import yaml
import copy
import numpy as np
import timm

from model import VisionTransformer, create_ResNet, create_VGG

def save_state(config, output_dir):
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_list(value) for key, value in obj.items()}
        else:
            return obj

    state = convert_numpy_to_list(config)

    yaml_file = os.path.join(output_dir, "state.yaml")

    with open(yaml_file, "w") as file:
        yaml.dump(state, file, default_flow_style=False, allow_unicode=True)

    print("state.yaml has been saved in {}".format(yaml_file))


def get_pruned_config_vit(masks:dict):
    config = {}
    # embed_dim
    config["embed_dim"] = int(np.sum(masks["embed_dim"]))

    # num_heads
    config["num_heads"] = [int(np.sum(num_heads_per_layer)) for num_heads_per_layer in masks["heads"]]

    # hidden_dim
    config["hidden_dim"] = [int(np.sum(hidden_dim_per_layer)) for hidden_dim_per_layer in masks["hidden_dim"]]

    return config

def get_pruned_config_resnet(masks):

    config_layer_flatten = list()
    masks_layer_flatten = list()

    for index in range(len(masks)):
        conv_num = len(masks[index])
        current_layer_cfg = list()
        current_layer_mask = list()
        for i in range(conv_num):
            current_layer_cfg.append(np.sum(masks[index][i]))
            current_layer_mask.append(masks[index][i])

        if index > 0:
            current_layer_cfg.insert(0, config_layer_flatten[index-1][-1])
            current_layer_mask.insert(0, masks_layer_flatten[index-1][-1])

        config_layer_flatten.append(current_layer_cfg)
        masks_layer_flatten.append(current_layer_mask)

    config = dict()
    reshape_masks = dict()
    keys = ["conv1", "layer1", "layer2", "layer3", "layer4"]

    if len(config_layer_flatten) == 9:
        blocks = [1, 2, 2, 2, 2]
    elif len(config_layer_flatten) == 17:
        blocks = [1, 3, 4, 6, 3]

    index = 0
    for i in range(len(blocks)):
        block_num = blocks[i]
        config[keys[i]] = list()
        reshape_masks[keys[i]] = list()
        for j in range(block_num):
            config[keys[i]].append(config_layer_flatten[index + j])
            reshape_masks[keys[i]].append(masks_layer_flatten[index + j])
        index = index + block_num

    return config, reshape_masks

def get_pruned_config_vgg(masks):
    config = list()
    indicies = list()
    for mask in masks['conv']:
        config.append(np.sum(mask))
        indicies.append(np.argwhere(mask).flatten())

    return config, indicies


def get_pruned_model_vit(model, masks:dict, output_dir):
    config = get_pruned_config_vit(masks)
    old_model_dict = model.state_dict()

    # original model state
    embed_dim = masks["embed_dim"].shape[0]
    depth, head_num = masks["heads"].shape
    head_dim = embed_dim // head_num

    embed_index = np.argwhere(masks["embed_dim"]).reshape(-1)
    hidden_index = list()
    head_index = list()
    for i in range(depth):
        hidden_index.append(np.argwhere(masks["hidden_dim"][i] == 1).reshape(-1))
        head_index.append(np.argwhere(masks["heads"][i] == 1).reshape(-1))

    qkv = np.arange(embed_dim * 3).reshape(3, head_num, head_dim)

    pruned_model = VisionTransformer(pruned=True, config=config)
    new_state_dict = copy.deepcopy(pruned_model.state_dict())

    for key, value in new_state_dict.items():

        if key.startswith("block"):
            block_idx = int(key.split(".")[1])

            if 'attn.qkv.weight' in key:
                head = head_index[block_idx]
                qkv_index = np.asarray(qkv[:, head]).reshape(-1)
                value.data = old_model_dict[key][qkv_index][:, embed_index]

            elif 'attn.qkv.bias' in key:
                head = head_index[block_idx]
                qkv_index = np.asarray(qkv[:, head]).reshape(-1)
                value.data = old_model_dict[key][qkv_index]

            elif 'attn.proj.weight' in key:
                head = head_index[block_idx]
                qkv_index = np.asarray(qkv[0][head]).reshape(-1)
                value.data = old_model_dict[key][embed_index][:, qkv_index]

            elif 'attn.proj.bias' in key:
                value.data = old_model_dict[key][embed_index]


            elif 'mlp.fc1.weight' in key:
                hidden = hidden_index[block_idx]
                value.data = old_model_dict[key][hidden][:, embed_index]

            elif 'mlp.fc1.bias' in key:
                hidden = hidden_index[block_idx]
                value.data = old_model_dict[key][hidden]

            elif 'mlp.fc2.weight' in key:
                hidden = hidden_index[block_idx]
                value.data = old_model_dict[key][embed_index][:, hidden]

            elif 'mlp.fc2.bias' in key:
                value.data = old_model_dict[key][embed_index]

            else:
                dim_mismatch = np.squeeze(np.argwhere(np.array(value.shape) != np.array(old_model_dict[key].shape)))
                if not dim_mismatch.size:
                    value.data = old_model_dict[key]
                else:
                    value.data = torch.index_select(old_model_dict[key], int(dim_mismatch), torch.tensor(embed_index))


        else:
            dim_mismatch = np.squeeze(np.argwhere(np.array(value.shape) != np.array(old_model_dict[key].shape)))

            if not dim_mismatch.size:
                value.data = old_model_dict[key]

            else:
                value.data = torch.index_select(old_model_dict[key], int(dim_mismatch), torch.tensor(embed_index))


    pruned_model.load_state_dict(new_state_dict)

    save_state(config, output_dir)

    return pruned_model

def get_pruned_model_resnet(model, masks, output_dir, num_classes):
    old_model_dict = model.state_dict()
    config, reshape_masks = get_pruned_config_resnet(masks)
    depth = len(masks)
    conv_num= len(masks[-1])

    depth = conv_num * (depth - 1) + 2

    pruned_model = create_ResNet(depth=depth, num_classes=num_classes, pruned=True, pruned_cfg=config)
    new_state_dict = copy.deepcopy(pruned_model.state_dict())

    for key, value in new_state_dict.items():
        if key.startswith("layer"):
            parts = key.split(".")
            layer_name = parts[0]
            block_index = int(parts[1])

            if "conv" in key:
                conv_index = int(parts[2][-1])
                in_channel_indices = np.nonzero(np.asarray(reshape_masks[layer_name][block_index][conv_index-1]))[0]
                out_channel_indices = np.nonzero(np.asarray(reshape_masks[layer_name][block_index][conv_index]))[0]
                value.data = old_model_dict[key][out_channel_indices][:, in_channel_indices, :, :]
            elif "bn" in key:
                conv_index = int(parts[2][-1])
                if "num_batches_tracked" not in key:
                    out_channel_indices = np.nonzero(np.asarray(reshape_masks[layer_name][block_index][conv_index]))[0]
                    value.data = old_model_dict[key][out_channel_indices]
                else:
                    continue
            elif "downsample.0" in key:
                in_channel_indices = np.nonzero(np.asarray(reshape_masks[layer_name][block_index][0]))[0]
                out_channel_indices = np.nonzero(np.asarray(reshape_masks[layer_name][block_index][-1]))[0]
                value.data = old_model_dict[key][out_channel_indices][:, in_channel_indices, :, :]
            elif "downsample.1" in key:
                out_channel_indices = np.nonzero(np.asarray(reshape_masks[layer_name][block_index][-1]))[0]
                if "num_batches_tracked" not in key:
                    value.data = old_model_dict[key][out_channel_indices]

        else:
            out_channel_indices = np.nonzero(np.asarray(reshape_masks["conv1"][0][0]))[0]
            if "conv" in key:
                value.data = old_model_dict[key][out_channel_indices]
            elif "bn" in key:
                if "num_batches_tracked" not in key:
                    value.data = old_model_dict[key][out_channel_indices]
                else:
                    continue
            elif "fc.weight" in key:
                in_channel_indices = np.nonzero(np.asarray(reshape_masks["layer4"][-1][-1]))[0]
                value.data = old_model_dict[key][:, in_channel_indices]

    pruned_model.load_state_dict(new_state_dict)

    save_state(config, output_dir)

    return pruned_model

def get_pruned_model_vgg(model, masks, output_dir, numclass):
    old_model_dict = model.state_dict()
    config, indicies = get_pruned_config_vgg(masks)

    pruned_model = create_VGG("vgg16_bn", num_classes=100, pruned=True, pruned_cfg=config)
    print(pruned_model)
    new_state_dict = copy.deepcopy(pruned_model.state_dict())

    conv_index = 0

    in_indices = [0, 1, 2]
    out_indices = indicies[conv_index]

    for key, value in new_state_dict.items():

        if key.startswith("features"):
            dim = value.dim()
            if dim == 4:
                value.data = old_model_dict[key][out_indices][:, in_indices, :, :]

                conv_index += 1
                in_indices = out_indices
                if conv_index < len(indicies):
                    out_indices = indicies[conv_index]
            else:
                dim_mismatch = np.squeeze(np.argwhere(np.array(value.shape) != np.array(old_model_dict[key].shape)))
                if not dim_mismatch.size:
                    value.data = old_model_dict[key]
                else:
                    value.data = old_model_dict[key][in_indices]
        elif key.startswith("head"):
            if "weight" in key:
                value.data = old_model_dict[key][:][:, in_indices]

    pruned_model.load_state_dict(new_state_dict)
    save_state(config, output_dir)
    torch.save(pruned_model.state_dict(), os.path.join(output_dir, "checkpoint_pruned.pth"))
    return pruned_model


def get_pruned_model(model, masks, output_dir, num_classes):
    if isinstance(model, VisionTransformer):
        return get_pruned_model_vit(model, masks, output_dir)
    elif isinstance(model, timm.models.ResNet):
        return get_pruned_model_resnet(model, masks, output_dir, num_classes)
    elif isinstance(model, timm.models.VGG):
        return get_pruned_model_vgg(model, masks, output_dir, num_classes)