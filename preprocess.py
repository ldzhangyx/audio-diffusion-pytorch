from data import Music4AllDataModule, Music4AllDataset
from mtr.contrastive.model import ContrastiveModel
from mtr.utils.demo_utils import get_model
from tqdm import tqdm
from mtr.modules.audio_rep import TFRep
import torch
import pickle

# GPU 3

if __name__ == "__main__":

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    pretrained_model_ckpt = "/import/c4dm-04/yz007/best.pth"
    condition_model, tokenizer, condition_model_config = get_model(ckpt=pretrained_model_ckpt)
    for param in condition_model.parameters():
        param.requires_grad = False

def text_condition():

    query = ["fusion jazz with synth, bass, drums, saxophone",
             "beaufitul classical music with piano, violin, cello, flute, harp, and oboe",
             "exciting rock music with guitar, bass, drums",
             "pop music with woman vocal, with piano accompaniment",]


    text_input = [tokenizer(query[i], return_tensors="pt")['input_ids'] for i in range(len(query))]
    with torch.no_grad():
        text_embs = [condition_model.encode_bert_text(text_input[i], None) for i in range(len(query))]
        text_embs = torch.cat(text_embs, dim=0)
        pickle.dump(text_embs, open("text_embs.pkl", "wb"))
        print(text_embs.shape)



def get_audio_conditions():
    data = Music4AllDataModule(batch_size=64, num_workers=64, sample_rate=16000, segment_length=2**17)
    data.setup(stage="fit")
    train_dataloader = data.train_dataloader(shuffle=False)
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader()

    # calculate condition across all data
    condition_model.eval()
    condition_model.to("cuda")



    rep = TFRep(
            sample_rate= 16000,
            f_min=0,
            f_max= 8000,
            n_fft = 1024,
            win_length = 1024,
            hop_length = int(0.01 * 16000),
            n_mels = 128
    )

    import pickle
    for i, loader in enumerate([train_dataloader, val_dataloader, test_dataloader]):
        audio_conditions = list()
        for batch in tqdm(loader):
            batch = batch.permute(0, 2, 1).mean(dim=1)  # batch => (batch_size, segment_length)
            spec = rep.melspec(batch)
            audio_embedding = condition_model.encode_audio(spec.cuda())
            audio_embedding = audio_embedding.detach().cpu()
            # split batch to segments
            audio_embedding_list = [audio_embedding[i] for i in range(audio_embedding.shape[0])]
            audio_conditions.extend(audio_embedding_list)

        # save
        pickle.dump(audio_conditions, open(f"audio_conditions_{i}.pkl", "wb"))
