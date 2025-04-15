import argparse
import os
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from wavenet import WaveNet  # Your converted WaveNet model
from data_loader import AudioDataset  # You will need to define this

BATCH_SIZE = 1
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
SAMPLE_SIZE = 100000
MAX_TO_KEEP = 5

STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY)
    parser.add_argument('--logdir_root', type=str, default=LOGDIR_ROOT)
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY)
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS)
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE)
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP)
    return parser.parse_args()

def get_default_logdir(logdir_root):
    return os.path.join(logdir_root, 'train', STARTED_DATESTRING)

def main():
    args = get_arguments()
    logdir = get_default_logdir(args.logdir_root)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)

    with open(args.wavenet_params) as f:
        params = json.load(f)

    dataset = AudioDataset(args.data_dir, args.sample_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = WaveNet(
        in_channels=params["quantization_channels"],
        residual_channels=params["residual_channels"],
        dilation_channels=params["dilation_channels"],
        skip_channels=params["skip_channels"],
        end_channels=params["skip_channels"],
        kernel_size=params["filter_width"],
        dilations=params["dilations"],
        global_condition_channels=None,
        quantization_channels=params["quantization_channels"]
    ).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    step = 0
    while step < args.num_steps:
        for audio, target in dataloader:
            audio, target = audio.cuda(), target.cuda()
            model.train()
            optimizer.zero_grad()
            output = model(audio)
            output = output.transpose(1, 2).reshape(-1, output.size(-1))
            target = target.view(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), step)
            print(f"Step {step} | Loss: {loss.item():.4f}")

            if step % args.checkpoint_every == 0:
                ckpt_path = os.path.join(logdir, f"checkpoint_{step}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step
                }, ckpt_path)

            step += 1
            if step >= args.num_steps:
                break

    writer.close()

if __name__ == '__main__':
    main()
