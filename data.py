import numpy
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import soundfile as sf
import numpy as np
import resampy
import pickle

class Music4AllDataset(Dataset):
    def __init__(self,
                 path="/import/c4dm-04/yz007/music4all/",
                 split="train",
                 sample_rate: int = 16000,
                 segment_length: int = 81920,):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.data_list = []
        # self.chunk_data_list = []
        self.file_names = os.listdir(f"{path}/audios_wav/")
        self.boundaries = [0]
        # split list by 8:1:1
        train_files, val_files, test_files = self.file_names[:int(len(self.file_names)*0.8)], \
            self.file_names[int(len(self.file_names)*0.8):int(len(self.file_names)*0.9)], \
            self.file_names[int(len(self.file_names)*0.9):]
        if split == "train":
            self.file_names = train_files
        elif split == "val":
            self.file_names = val_files
        elif split == "test":
            self.file_names = test_files
        else:
            raise ValueError(f'Invalid split: {split}')
        print("Preprocessing data...")
        for file in tqdm(self.file_names):
            # get metadata
            title = file.split(".")[0]
            wav_file = f"{path}/audios_wav/{title}.wav"
            info = sf.info(wav_file)
            sr = info.samplerate
            frames = info.frames
            # self.data_list.append((wav_file, sr, frames))
            # process audio and split to chunks
            if sample_rate is None:
                segment_length_in_time = segment_length / sr
            else:
                segment_length_in_time = segment_length / sample_rate
            num_chunks = int(frames / (segment_length_in_time * sr))
            self.boundaries.append(self.boundaries[-1] + num_chunks)
            self.data_list.append(
                (wav_file, sr, segment_length_in_time))

        print(
            f'total number of chunks: {self.boundaries[-1]}')
        self.boundaries = np.array(self.boundaries)

        self.audio_conditions = None

    def __len__(self) -> int:
        return self.boundaries[-1]

    def _get_file_idx_and_chunk_idx(self, index: int):
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        chunk_index = index - self.boundaries[bin_pos]
        return bin_pos, chunk_index

    def _get_waveforms(self, index: int, chunk_index: int):
        """Get waveform without resampling."""
        wav_file, sr, length_in_time = self.data_list[index]
        offset = int(chunk_index * length_in_time * sr)
        frames = int(length_in_time * sr)

        data, _ = sf.read(
            wav_file, start=offset, frames=frames, dtype='float32', always_2d=True)
        # data = data.mean(axis=1, keepdims=True)  # numpy array
        if data.shape[-1] == 1:
            data = np.concatenate([data, data], axis=-1)
        return data
    #
    # def _get_audio_condition(self, index: int):
    #     return self.audio_conditions[index]

    def __getitem__(self, index: int):
        file_idx, chunk_idx = self._get_file_idx_and_chunk_idx(index)
        data = self._get_waveforms(file_idx, chunk_idx)
        if data.shape[0] != self.segment_length:
            data = resampy.resample(data, data.shape[0], self.segment_length, axis=0, filter='kaiser_fast')[
                :self.segment_length]
            if data.shape[0] < self.segment_length:
                data = np.pad(
                    data, ((0, self.segment_length - data.shape[0]),), 'constant')
        data = torch.from_numpy(data)
        #
        # if self.audio_conditions is not None:
        #     audio_condition = self._get_audio_condition(index)
        #     return data, audio_condition

        return data


class Music4AllDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 num_workers: int = 64,
                 sample_rate: int = 16000,
                 segment_length: int = 81920,
                 condition=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.cache = True
        self.cache_folder = f"cache/{sample_rate}_{segment_length}"
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.condition = condition

    def setup(self, stage=None):
        if stage == "fit":
            if self.cache and os.path.exists(f"{self.cache_folder}/fit.pkl"):
                self.train_dataset = load_dataset(f"{self.cache_folder}/fit.pkl")
            else:
                self.train_dataset = Music4AllDataset(
                    split="train", sample_rate=self.sample_rate, segment_length=self.segment_length)
                if self.cache:
                    save_dataset(self.train_dataset, f"{self.cache_folder}/fit.pkl")
            # if self.condition:
            #     self.train_dataset.audio_conditions = pickle.load(open(
            #         "/import/c4dm-04/yz007/audio_conditions_train.pkl", "rb"))

        if stage == "validate" or stage == "fit":
            if self.cache and os.path.exists(f"{self.cache_folder}/validate.pkl"):
                self.val_dataset = load_dataset(f"{self.cache_folder}/validate.pkl")
            else:
                self.val_dataset = Music4AllDataset(
                    split="val", sample_rate=self.sample_rate, segment_length=self.segment_length)
                if self.cache:
                    save_dataset(self.val_dataset, f"{self.cache_folder}/validate.pkl")
            # if self.condition:
            #     self.val_dataset.audio_conditions = pickle.load(open(
            #         "/import/c4dm-04/yz007/audio_conditions_val.pkl", "rb"))

        if stage == "test":
            if self.cache and os.path.exists(f"{self.cache_folder}/test.pkl"):
                self.test_dataset = load_dataset(f"{self.cache_folder}/test.pkl")
            else:
                self.test_dataset = Music4AllDataset(
                    split="test", sample_rate=self.sample_rate, segment_length=self.segment_length)
                if self.cache:
                    save_dataset(self.test_dataset, f"{self.cache_folder}/test.pkl")


    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# def collate_fn(batch):
#     if len(batch[0]) == 2:
#         data, audio_condition = zip(*batch)
#         return torch.stack(data), torch.stack(audio_condition)
#     else:
#         return torch.stack(batch)



def load_dataset(path):
    return pickle.load(open(path, 'rb'))


def save_dataset(dataset, path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    pickle.dump(dataset, open(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)