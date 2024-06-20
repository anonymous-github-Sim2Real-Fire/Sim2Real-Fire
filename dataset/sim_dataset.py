import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import tifffile as tif

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class SimFireDataset(Dataset):

    def __init__(self, cfg, training=True):
        self.sample = cfg['input_length'] * 2
        self.root_dir = cfg['sim_root_dir']
        self.fire_dir = cfg['train_dir'] if training else cfg['val_dir']
        self.wxs_dir = cfg['trainwxs_dir'] if training else cfg['valwxs_dir']
        self.topo_dir = cfg['topo_dir']
        self.vege_dir = cfg['vege_dir']
        self.fuel_dir = cfg['fuel_dir']
        self.sate_dir = cfg['sate_dir']
        self.reverse = cfg['reverse']
        self.num_workers = cfg['num_workers']
        self.topo = {}
        self.vege = {}
        self.fuel = {}
        self.img_size = cfg['img_size']
        self.indices_of_degree_features = [2]
        self.vege_indices = [0, 1, 2, 3, 4]
        self.fuel_indices = [1]  
        self.vege_one_hot_matrices = {index: torch.eye(cfg['vege_classes'][index]) for index in range(0,len(self.vege_indices))}
        self.fuel_one_hot_matrices = {index: torch.eye(cfg['fuel_classes'][index]) for index in range(0,len(self.fuel_indices))}
        
        self.topo_means, self.topo_stds = self.compute_means_stds(self.topo_dir, self.indices_of_degree_features)
        self.vege_means, self.vege_stds = self.compute_means_stds(self.vege_dir, self.vege_indices)
        self.fuel_means, self.fuel_stds = self.compute_means_stds(self.fuel_dir, self.fuel_indices)

        for rd in self.root_dir:
            topo = self.get_landfire_modality(rd, self.topo_dir, self.topo_means, self.topo_stds, apply_sin_indices=self.indices_of_degree_features)
            vege = self.get_landfire_modality(rd, self.vege_dir, self.vege_means, self.vege_stds, indices=self.vege_indices, one_hot_matrices=self.vege_one_hot_matrices)
            fuel = self.get_landfire_modality(rd, self.fuel_dir, self.fuel_means, self.fuel_stds, indices=self.fuel_indices, one_hot_matrices=self.fuel_one_hot_matrices)
            self.topo[f'{rd}'] = topo
            self.vege[f'{rd}'] = vege
            self.fuel[f'{rd}'] = fuel

        self.fire_path = []
        self.target_path = []
        self.sate_path = []
        self.fire_name = []
        self.wxs_value = []
        self.time_steps = []

        wxs_dict = self.read_wxs(self.root_dir, self.wxs_dir)     
        
        for rd in self.root_dir:
            for root, dirs, files in os.walk(os.path.join(rd,self.fire_dir)):
                if root == os.path.join(rd,self.fire_dir):
                    continue
                else:
                    length = len(files)
                    i = os.path.basename(root)[3:][:-12]
                    sequence = self.extract_sequences(1, length, self.sample, reverse=self.reverse)
                    for s in sequence:
                        self.time_steps.append(self.normalize_timestamps(1, length, s))
                        ind = int(len(s) / 2)
                        input_sequence = s[:ind]
                        output_sequence = s[ind:]
                        input_path = []
                        input_wxs = []
                        out_path = []
                        sate_path = []
                        flag = -1
                        for ind, line_sequence in enumerate(input_sequence):
                            f = os.path.join(root, 'out' + str(line_sequence) + '.jpg')   
                            o = os.path.join(root, 'out' + str(output_sequence[ind]) + '.jpg') 
                            s_path = os.path.join(rd, self.sate_dir, os.path.basename(root), 'out' + str(line_sequence) + '.jpg')                                       
                            key = f'{rd}_{i}_{line_sequence}'
                            if wxs_dict[key][0] == '#':
                                flag += 1
                            input_wxs.append(wxs_dict[key])
                            input_path.append(f) 
                            out_path.append(o)
                            sate_path.append(s_path)

                        if flag == -1:
                            self.fire_path.append(input_path) 
                            self.target_path.append(out_path)
                            self.fire_name.append(f'{rd}')
                            self.wxs_value.append(input_wxs)
                            self.sate_path.append(sate_path)
                        flag = -1

    def __len__(self):
        return len(self.fire_path)

    def __getitem__(self, index): 
        fire_path = self.fire_path[index]
        target_path = self.target_path[index]
        sate_path = self.sate_path[index]
        fire_name = self.fire_name[index]
        time_steps = self.time_steps[index]
        topo = self.topo[fire_name]
        vege = self.vege[fire_name]
        fuel = self.fuel[fire_name]
        
        wxs_value = self.wxs_value[index]
        for i, wxs in enumerate(wxs_value):
            wxs_value[i] = wxs
            for w, j in enumerate(wxs_value[i]):
                wxs_value[i][w] = float(j)

        wxs = torch.tensor(wxs_value)
        
        input_squence = [self.process_image(file, gray=True) for file in fire_path]
        input_squence = torch.cat(input_squence, dim=0).float()

        output_squence = [self.process_image(file, gray=True) for file in target_path]
        output_squence = torch.cat(output_squence, dim=0).float()

        sate_squence = [self.process_image(file, gray=False) for file in sate_path]
        sate_squence = torch.cat(sate_squence, dim=0).float()

        return index, input_squence, output_squence, fuel, vege, topo, torch.rand(3,3,256, 256), wxs, torch.tensor(time_steps)

    def process_image(self, file_path, gray=True):
        img = cv2.imread(file_path)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img[np.where(img > 0)] = 1
            img = torch.from_numpy(cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        else:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)
            img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def get_landfire_modality(self, root_dir, landfire_dir, means, stds, apply_sin_indices=[], indices=[], one_hot_matrices={}):
        file_names = os.listdir(os.path.join(root_dir, landfire_dir))
        modalities = []
        
        for i, file in enumerate(file_names):
            apply_sin = i in apply_sin_indices
            modalities.append(self.read_and_process_landfire_file(os.path.join(root_dir, landfire_dir), file, means, stds, apply_sin))
        
        modalities = torch.cat(modalities, dim=0).float()

        if indices:
            modalities = self.process_land_cover(modalities, indices, one_hot_matrices)

        return modalities

    def read_and_process_landfire_file(self, path, file, means, stds, apply_sin=False):
        file_path = os.path.join(path, file)
        try:
            lf = tif.imread(file_path)
            lf = cv2.resize(lf, self.img_size, interpolation=cv2.INTER_NEAREST)
            lf = torch.from_numpy(lf).unsqueeze(0).float()

            if apply_sin:
                lf = torch.sin(torch.deg2rad(lf))

            for i in range(lf.shape[0]): 
                lf[i] = (lf[i] - means[i]) / stds[i]

            return lf
        except IOError:
            print(f"Open ERROR!: {file_path}")
            return torch.zeros((1, *self.img_size))

    def process_land_cover(self, lf, indices, one_hot_matrices):
        one_hot_encoded_tensors = []
        for i, index in enumerate(indices):
            new_shape = (lf.shape[1], lf.shape[2], one_hot_matrices[i].shape[0])
            landcover_classes_flattened = lf[index, ...].long().flatten() - 1
            landcover_encoding = one_hot_matrices[i][landcover_classes_flattened].reshape(new_shape).permute(2, 0, 1)
            one_hot_encoded_tensors.append(landcover_encoding)
        all_one_hot_encoded = torch.cat(one_hot_encoded_tensors, dim=0)
        remaining_features = lf[len(indices):, ...] if len(indices) < lf.shape[0] else None
        lf = torch.cat([all_one_hot_encoded, remaining_features], dim=0)
        return lf

    def extract_sequences(self, start, end, sample, u_interval=0.167, reverse=True):
        if end - start <= 0:
            return []
        sequence = []
        num_samples = max(1, int((end - 1) - np.floor(end * u_interval)))
        
        for _ in range(num_samples):
            seq = self.extract_sequences_interval_random(start, end, sample, reverse)
            if seq:
                sequence.append(seq)
        return sequence

    def extract_sequences_interval_random(self, start, end, sample, reverse):
        max_interval = int(np.floor((end - start) * 0.167))
        intervals = np.random.randint(1, max_interval + 1, size=sample - 1)
        intervals = np.insert(intervals, 0, 0).cumsum()
        if intervals[-1] > (end - start):
            return None
        if reverse:
            seq = [end - i for i in intervals[::-1]]
        else:
            seq = [start + i for i in intervals]

        return seq if all(1 <= x <= end for x in seq) else None

    def read_wxs(self, root_dir, wxs_path):
        file_lines_dict = {}
        for rd in root_dir:
            wxs_dir = os.path.join(rd , wxs_path)
            try:
                for filename in os.listdir(wxs_dir):
                    file_path = os.path.join(wxs_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for line_number, line_content in enumerate(lines, 1):
                            key = f"{rd}_{filename[:-4]}_{line_number}"
                            file_lines_dict[key] = line_content.strip().split(' ')
            except Exception as e:
                print(f"Error reading {wxs_dir}: {e}")
        return file_lines_dict

    def normalize_timestamps(self, start_time, end_time, timestamps):
        start_time = float(start_time)
        end_time = float(end_time)
        timestamps = np.array(timestamps, dtype=float)
        normalized_timestamps = (timestamps - start_time) / (end_time - start_time)
        normalized_timestamps = np.clip(normalized_timestamps, 0, 1)
        return normalized_timestamps[3:].tolist()


    def compute_means_stds(self, modality_dir, indices_to_exclude):
        all_data = []
        for rd in self.root_dir:
            for root, dirs, files in os.walk(os.path.join(rd, modality_dir)):
                for file in files:
                    if file.endswith(".tif"):
                        file_path = os.path.join(root, file)
                        try:
                            lf = tif.imread(file_path)
                            lf = cv2.resize(lf, self.img_size, interpolation=cv2.INTER_NEAREST)
                            all_data.append(lf)
                        except IOError:
                            print(f"Open ERROR!: {file_path}")

        all_data = np.stack(all_data, axis=0)
        means = np.mean(all_data, axis=(1, 2))
        stds = np.std(all_data, axis=(1, 2))
        means = np.asarray(means)
        stds = np.asarray(stds)
        means[indices_to_exclude] = 0
        stds[indices_to_exclude] = 1

        return torch.tensor(means), torch.tensor(stds)
