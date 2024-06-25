import os
import argparse
import numpy as np
import cv2
from scipy import ndimage
from multiprocessing import Process
import gc

def get_sub_path(path):
    sub_path = []
    if isinstance(path, list):
        for p in path:
            if os.path.isdir(p):
                for file in os.listdir(p):
                    sub_path.append(os.path.join(p, file))
            else:
                continue
    else:
        for file in os.listdir(path):
            sub_path.append(os.path.join(path, file))
    return sub_path

def resize(input):
    dimension = input.shape
    result = ndimage.zoom(input, (256 / dimension[0], 256 / dimension[1]), order=3)
    return result

def resize_cv2(input):
    output = cv2.resize(input, (256, 256), interpolation = cv2.INTER_AREA)
    return output

def std(input):
    if input.max() == 0:
        return input
    else:
        result = (input-input.min()) / (input.max()-input.min())
        return result

def save_npy(out_list, save_path, name):
    output = np.array(out_list)
    output = np.transpose(output, (1, 2, 0))
    np.save(os.path.join(save_path, name), output)

def divide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]

def pack_data(args, name_list, read_feature_list, read_label_list, save_path):
    os.makedirs(save_path, exist_ok=True)
    feature_save_path = os.path.join(args.save_path, args.task, 'feature')
    os.makedirs(feature_save_path, exist_ok=True)
    label_save_path = os.path.join(args.save_path, args.task, 'label')
    os.makedirs(label_save_path, exist_ok=True)

    for name in name_list:
        out_feature_list = []
        for feature_name in read_feature_list:
            name = os.path.basename(name)
            feature = np.load(os.path.join(args.data_path, feature_name, name))
            if args.task == 'congestion':
                feature = std(resize(feature))
                out_feature_list.append(feature)
            elif args.task == 'DRC':
                feature = std(resize(feature))
                out_feature_list.append(feature)
            elif args.task == 'IR_drop':
                if feature_name == 'IR_drop_features_decompressed/power_t':
                    for i in range(20):
                        slice = feature[i,:,:]
                        out_feature_list.append(std(resize_cv2(slice)))
                else:
                    feature = std(resize_cv2(feature.squeeze()))
                    out_feature_list.append(feature)
            else:
                raise ValueError('Task not implemented')

        save_npy(out_feature_list, feature_save_path, name)

        out_label_list = []
        congestion_temp = np.zeros((256,256))
        for label_name in read_label_list:
            name = os.path.basename(name)
            label = np.load(os.path.join(args.data_path, label_name, name))

            if args.task == 'congestion':
                congestion_temp += resize(label)
                
            elif args.task == 'DRC':
                label = np.clip(label, 0, 200)
                label = resize_cv2(label) / 200
                out_label_list.append(label)
            elif args.task == 'IR_drop':
                label = np.squeeze(label)
                label = np.clip(label, 1e-6, 50)
                label = (np.log10(resize_cv2(label)) +6) / (np.log10(50)+6)
                out_label_list.append(label)
            else:
                raise ValueError('Task not implemented')

        if args.task == 'congestion':
            out_label_list.append(std(congestion_temp))

        save_npy(out_label_list, label_save_path, name)
        
         # 清理内存
        del out_feature_list[:]
        del out_label_list[:]
        del feature
        del label
        if args.task == 'congestion':
            del congestion_temp
        # 手动调用垃圾回收
        gc.collect()
            
def parse_args():
    description = "you should add those parameter" 
    parser = argparse.ArgumentParser(description=description)
                                                             
    parser.add_argument("--task", default = 'congestion', type=str, help = 'congestion' )
    parser.add_argument("--data_path", default = "D:/", type=str, help = "D:/routability_features_decompressed")
    parser.add_argument("--save_path",  default = "D:/training_set_forfinal", type=str, help =  "D:/training_set_forfinal")

    args = parser.parse_args()                                       
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.task == 'congestion':
        feature_list = ['routability_features_decompressed/macro_region', 'routability_features_decompressed/RUDY/RUDY', 
        'routability_features_decompressed/RUDY/RUDY_pin']
        label_list = ['routability_features_decompressed/overflow_based/congestion_GR_horizontal_overflow', 
        'routability_features_decompressed/overflow_based/congestion_GR_vertical_overflow']

    elif args.task == 'DRC':
        feature_list = ['routability_features_decompressed/macro_region', 'routability_features_decompressed/cell_density', 
        'routability_features_decompressed/RUDY/RUDY_long', 'routability_features_decompressed/RUDY/RUDY_short',
        'routability_features_decompressed/RUDY/RUDY_pin_long', 
        'routability_features_decompressed/congestion/congestion_early_global_routing/overflow_based/congestion_eGR_horizontal_overflow', 
        'routability_features_decompressed/congestion/congestion_early_global_routing/overflow_based/congestion_eGR_vertical_overflow', 
        'routability_features_decompressed/congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow', 
        'routability_features_decompressed/congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow']
        label_list = ['routability_features_decompressed/DRC/DRC_all']

    elif args.task == 'IR_drop':
        feature_list = ['IR_drop_features_decompressed/power_i', 'IR_drop_features_decompressed/power_s', 
        'IR_drop_features_decompressed/power_sca', 'IR_drop_features_decompressed/power_all', 'IR_drop_features_decompressed/power_t']
        label_list = ['IR_drop_features_decompressed/IR_drop']
    else:
        raise ValueError('Please specify argument --task from congestion, DRC and IR_drop')

    name_list = get_sub_path(os.path.join(args.data_path, feature_list[-1]))
    print('processing %s files' % len(name_list))
    save_path = os.path.join(args.save_path, args.task)

    nlist = divide_list(name_list, 1000)###原来：1000
    process = []
    for list in nlist:
        p = Process(target=pack_data, args=(args, list, feature_list, label_list, save_path))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()
