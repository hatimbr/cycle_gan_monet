import argparse
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import idr_torch
import itertools
from utils.data import get_dataset
from utils.model import GeneratorResNet, Discriminator
from utils.train import train_loop, train_loop_amp
from mlflow import log_params, set_experiment, start_run  # mlflow !!!!!!!!!!!!!!!!!!
from datetime import datetime
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule # profiler !!!!!!!!!!!!!!
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.996)
    parser.add_argument("--n_epoches", type=int, default=10)
    parser.add_argument("--decay_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default='./datasets/monet_kaggle/')
    parser.add_argument("--tracking", action='store_true')
    parser.add_argument("--profiling", action='store_true')
    parser.add_argument("--track_name", type=str, default="exp")
    parser.add_argument("--evaluation", action='store_true')
    parser.add_argument("--model_path", type=str, default="model.pt")
    parser.add_argument("--amp", action='store_true')
    
    return parser.parse_args()
    

def train(args):
    # Define training parameters
    lr = args.lr
    b1 = args.b1
    b2 = args.b2
    n_epoches = args.n_epoches
    decay_epoch = args.decay_epoch
    batch_size = args.batch_size
    data_dir = args.data_dir 

    # Initialize parallelism
    dist.init_process_group(backend='nccl', init_method='env://', world_size=idr_torch.size, rank=idr_torch.rank)
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")


    # Define Loss
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    

    # Initialize models (Generators and Discriminators)
    G_AB = GeneratorResNet(3, num_residual_blocks=9)
    D_B = Discriminator(3)
    G_BA = GeneratorResNet(3, num_residual_blocks=9)
    D_A = Discriminator(3)


    # Prepare models to data parallelism
    G_AB_dist = DistributedDataParallel(G_AB.to(gpu))
    D_B_dist = DistributedDataParallel(D_B.to(gpu))
    G_BA_dist = DistributedDataParallel(G_BA.to(gpu))
    D_A_dist = DistributedDataParallel(D_A.to(gpu))


    # Configure Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB_dist.parameters(), G_BA_dist.parameters()), lr=lr, betas=(b1, b2))
    optimizer_D_A = torch.optim.Adam(D_A_dist.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B_dist.parameters(), lr=lr, betas=(b1, b2))


    # Configure Scheduler
    lambda_func = lambda epoch: 1 - max(0, epoch-decay_epoch)/(n_epoches-decay_epoch)

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_func)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_func)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_func)


    # Get Datasets
    trainloader = get_dataset(data_dir, batch_size=batch_size, mode="train")
    #testloader = get_dataset(data_dir, batch_size=1, mode="test")

    # Get the amp loop function or not
    if args.amp:
        train_loop_ = train_loop_amp
    else:
        train_loop_ = train_loop

    # Training loop
    if args.profiling:

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=schedule(wait=1, warmup=1, active=12),
            on_trace_ready=tensorboard_trace_handler('./profiler'), profile_memory=True, with_stack=False, record_shapes=False
            ) as prof:

                G_BA = train_loop_(G_AB_dist, G_BA_dist, D_A_dist, D_B_dist, gpu, trainloader, n_epoches, optimizer_G, lr_scheduler_G, 
                optimizer_D_A,  lr_scheduler_D_A, optimizer_D_B, lr_scheduler_D_B, criterion_identity, criterion_GAN, 
                criterion_cycle, tracking=args.tracking, profiling=args.profiling, prof=prof)

    else:

        G_BA = train_loop_(G_AB_dist, G_BA_dist, D_A_dist, D_B_dist, gpu, trainloader, n_epoches, optimizer_G, lr_scheduler_G, optimizer_D_A, 
        lr_scheduler_D_A, optimizer_D_B, lr_scheduler_D_B, criterion_identity, criterion_GAN, criterion_cycle, 
        tracking=args.tracking, profiling=args.profiling, prof=None)

    if idr_torch.rank == 0:
        torch.save(G_BA_dist.module.state_dict(), args.model_path)


        

def evaluate(args): # Transform every pics in the dir to "Monet style" for Kaggle evaluation

    # Load model
    G_BA = GeneratorResNet(3, num_residual_blocks=9)
    G_BA.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
    G_BA.eval()

    cuda = torch.cuda.is_available()
    print(f'cuda: {cuda}')
    if cuda:
        G_BA = G_BA.cuda()

    # Prepare dataset
    evalloader = get_dataset(args.data_dir, batch_size=args.batch_size, mode="eval")

    # Create dir
    if not os.path.isdir("./out_eval"):
        os.mkdir("./out_eval")
    os.mkdir(f"./out_eval/{timestamp}")
    out_dir = f"./out_eval/{timestamp}"

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    with torch.no_grad():
        for i, (real_B) in enumerate(evalloader):

            real_B = real_B.type(Tensor)
            fake_A = G_BA(real_B)

            for j in range(fake_A.size(0)):
                save_image(fake_A[j, :, :, :], f'{out_dir}/{i*args.batch_size+j}.png')

            print(f"eval [{i}/{len(evalloader)}]")


if __name__ == '__main__':
    args = parse_args()
    timestamp = int(datetime.timestamp(datetime.now()))

    if args.evaluation:
        evaluate(args)

    else:

        if args.tracking:
            set_experiment("Test CycleGAN") # mlflow !!!!!!!!!!!!!!!!!!

            with start_run(run_name=f"{args.track_name}-{timestamp}") as run: # mlflow !!!!!!!!!!!!!!!!!!
                log_params({
                    'lr' : args.lr,
                    'b1' : args.b1,
                    'b2' : args.b2,
                    'n_epoches' : args.n_epoches,
                    'decay_epoch' : args.decay_epoch,
                    'batch_size' : args.batch_size,
                    'data_dir' : args.data_dir
                })

                train(args)

        else:

            train(args)

