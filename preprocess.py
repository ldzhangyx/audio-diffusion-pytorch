from data import Music4AllDataModule, Music4AllDataset
from mtr.contrastive.model import ContrastiveModel
from mtr.utils.demo_utils import get_model
from tqdm import tqdm

# GPU 3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

pretrained_model_ckpt = "/import/c4dm-04/yz007/best.pth"
condition_model, tokenizer, condition_model_config = get_model(ckpt=pretrained_model_ckpt)
for param in condition_model.parameters():
    param.requires_grad = False

data = Music4AllDataModule(batch_size=32, num_workers=64, sample_rate=16000, segment_length=2**17)
data.setup(stage="fit")
train_dataloader = data.train_dataloader(shuffle=False)
val_dataloader = data.val_dataloader()
test_dataloader = data.test_dataloader()

# calculate condition across all data
condition_model.eval()
condition_model.to("cuda")

audio_conditions = list()

for batch in tqdm(train_dataloader):
    batch = batch.permute(0, 2, 1).mean(dim=1)  # batch => (batch_size, segment_length)
    audio_embedding = condition_model.encode_audio(batch.cuda())
    audio_embedding = audio_embedding.detach().cpu()
    # split batch to segments
    audio_embedding_list = [audio_embedding[i] for i in range(audio_embedding.shape[0])]
    audio_conditions.extend(audio_embedding_list)

# save
import pickle
pickle.dump(audio_conditions, open("audio_conditions_train.pkl", "wb"))

for batch in tqdm(val_dataloader):
    batch = batch.permute(0, 2, 1).mean(dim=1)  # batch => (batch_size, segment_length)
    audio_embedding = condition_model.encode_audio(batch.cuda())
    audio_embedding = audio_embedding.detach().cpu()
    # split batch to segments
    audio_embedding_list = [audio_embedding[i] for i in range(audio_embedding.shape[0])]
    audio_conditions.extend(audio_embedding_list)

pickle.dump(audio_conditions, open("audio_conditions_val.pkl", "wb"))

for batch in tqdm(test_dataloader):
    batch = batch.permute(0, 2, 1).mean(dim=1)  # batch => (batch_size, segment_length)
    audio_embedding = condition_model.encode_audio(batch.cuda())
    audio_embedding = audio_embedding.detach().cpu()
    # split batch to segments
    audio_embedding_list = [audio_embedding[i] for i in range(audio_embedding.shape[0])]
    audio_conditions.extend(audio_embedding_list)

pickle.dump(audio_conditions, open("audio_conditions_test.pkl", "wb"))
