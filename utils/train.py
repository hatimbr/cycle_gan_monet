import torch
from torch.cuda.amp import autocast, GradScaler
from mlflow import log_metrics


def train_loop(G_AB, G_BA, D_A, D_B, gpu, trainloader, n_epoches, optimizer_G, lr_scheduler_G, optimizer_D_A, lr_scheduler_D_A, 
optimizer_D_B, lr_scheduler_D_B, criterion_identity, criterion_GAN, criterion_cycle, tracking=False, profiling=False, 
prof=None):

    step = 0
    for epoch in range(n_epoches):
        for i, (real_A, real_B) in enumerate(trainloader):
            real_A, real_B = real_A.to(gpu), real_B.to(gpu)
            
            # groud truth
            out_shape = [real_A.size(0), 1, real_A.size(2)//16, real_A.size(3)//16] #16 because scale factor of Discriminator (see Discriminator class)
            valid = torch.ones(out_shape).to(gpu)
            fake = torch.zeros(out_shape).to(gpu)
            
            """Train Generators"""
            # set to training mode in the begining, beacause sample_images will set it to eval mode
            G_AB.train()
            G_BA.train()
            
            optimizer_G.zero_grad()
            
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            
            # identity loss
            loss_id_A = criterion_identity(fake_B, real_A)
            loss_id_B = criterion_identity(fake_A, real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            
            # GAN loss, train G to make D think it's true
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid) 
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            
            # cycle loss
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            
            # G totol loss
            loss_G = 5.0*loss_identity + loss_GAN + 10.0*loss_cycle
            
            loss_G.backward()
            optimizer_G.step()
            
            """Train Discriminator A"""
            optimizer_D_A.zero_grad()
            
            loss_real_DA = criterion_GAN(D_A(real_A), valid)
            loss_fake_DA = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real_DA + loss_fake_DA) / 2
            
            loss_D_A.backward()
            optimizer_D_A.step()
            
            """Train Discriminator B"""
            optimizer_D_B.zero_grad()
            
            loss_real_DB = criterion_GAN(D_B(real_B), valid)
            loss_fake_DB = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real_DB + loss_fake_DB) / 2
            
            loss_D_B.backward()
            optimizer_D_B.step()

            if tracking: # mlflox !!!!!!!!!!!!!
                log_metrics({

                    'G_loss_id_A' : loss_id_A.data.tolist(),
                    'G_loss_id_B' : loss_id_B.data.tolist(),
                    'G_loss_id_total' : loss_identity.data.tolist(),
                    'G_loss_GAN_AB' : loss_GAN_AB.data.tolist(),
                    'G_loss_GAN_BA' : loss_GAN_BA.data.tolist(),
                    'G_loss_GAN_total' : loss_GAN.data.tolist(),
                    'G_loss_cycle_A' : loss_cycle_A.data.tolist(),
                    'G_loss_cycle_B' : loss_cycle_B.data.tolist(),
                    'G_loss_cycle_total' : loss_cycle.data.tolist(),
                    'G_loss_total' : loss_G.data.tolist(),

                    'DA_loss_real' : loss_real_DA.data.tolist(),
                    'DA_loss_fake' : loss_fake_DA.data.tolist(),
                    'DA_loss_total' : loss_D_A.data.tolist(),
                    'DB_loss_real' : loss_real_DB.data.tolist(),
                    'DB_loss_fake' : loss_fake_DB.data.tolist(),
                    'DB_loss_total' : loss_D_B.data.tolist(),

                }, step=step)

            print(f'[Epoch {epoch+1}/{n_epoches} |  Step_Epoch {i+1}/{len(trainloader)}]')
            step += 1

            if profiling: # profiler !!!!!!!!!!!!!!
                prof.step() 

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    return G_BA


def train_loop_amp(G_AB, G_BA, D_A, D_B, gpu, trainloader, n_epoches, optimizer_G, lr_scheduler_G, optimizer_D_A, lr_scheduler_D_A, 
optimizer_D_B, lr_scheduler_D_B, criterion_identity, criterion_GAN, criterion_cycle, tracking=False, profiling=False, 
prof=None):

    print('AMP ok')
    step = 0
    scaler = GradScaler()
    for epoch in range(n_epoches):
        for i, (real_A, real_B) in enumerate(trainloader):
            real_A, real_B = real_A.to(gpu), real_B.to(gpu)
            
            # groud truth
            out_shape = [real_A.size(0), 1, real_A.size(2)//16, real_A.size(3)//16] #16 because scale factor of Discriminator (see Discriminator class)
            valid = torch.ones(out_shape).to(gpu)
            fake = torch.zeros(out_shape).to(gpu)
            
            """Train Generators"""
            # set to training mode in the begining, beacause sample_images will set it to eval mode
            G_AB.train()
            G_BA.train()
            
            optimizer_G.zero_grad()
            with autocast():
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)
                
                # identity loss
                loss_id_A = criterion_identity(fake_B, real_A)
                loss_id_B = criterion_identity(fake_A, real_B)
                loss_identity = (loss_id_A + loss_id_B) / 2
                
                # GAN loss, train G to make D think it's true
                loss_GAN_AB = criterion_GAN(D_B(fake_B), valid) 
                loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
                
                # cycle loss
                recov_A = G_BA(fake_B)
                recov_B = G_AB(fake_A)
                loss_cycle_A = criterion_cycle(recov_A, real_A)
                loss_cycle_B = criterion_cycle(recov_B, real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
                
                # G totol loss
                loss_G = 5.0*loss_identity + loss_GAN + 10.0*loss_cycle
            
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            
            """Train Discriminator A"""
            optimizer_D_A.zero_grad()
            with autocast():
                loss_real_DA = criterion_GAN(D_A(real_A), valid)
                loss_fake_DA = criterion_GAN(D_A(fake_A.detach()), fake)
                loss_D_A = (loss_real_DA + loss_fake_DA) / 2
            
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizer_D_A)
            
            """Train Discriminator B"""
            optimizer_D_B.zero_grad()
            with autocast():
                loss_real_DB = criterion_GAN(D_B(real_B), valid)
                loss_fake_DB = criterion_GAN(D_B(fake_B.detach()), fake)
                loss_D_B = (loss_real_DB + loss_fake_DB) / 2
            
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)

            if tracking: # mlflow !!!!!!!!!!!!!
                log_metrics({

                    'G_loss_id_A' : loss_id_A.data.tolist(),
                    'G_loss_id_B' : loss_id_B.data.tolist(),
                    'G_loss_id_total' : loss_identity.data.tolist(),
                    'G_loss_GAN_AB' : loss_GAN_AB.data.tolist(),
                    'G_loss_GAN_BA' : loss_GAN_BA.data.tolist(),
                    'G_loss_GAN_total' : loss_GAN.data.tolist(),
                    'G_loss_cycle_A' : loss_cycle_A.data.tolist(),
                    'G_loss_cycle_B' : loss_cycle_B.data.tolist(),
                    'G_loss_cycle_total' : loss_cycle.data.tolist(),
                    'G_loss_total' : loss_G.data.tolist(),

                    'DA_loss_real' : loss_real_DA.data.tolist(),
                    'DA_loss_fake' : loss_fake_DA.data.tolist(),
                    'DA_loss_total' : loss_D_A.data.tolist(),
                    'DB_loss_real' : loss_real_DB.data.tolist(),
                    'DB_loss_fake' : loss_fake_DB.data.tolist(),
                    'DB_loss_total' : loss_D_B.data.tolist(),

                }, step=step)

            print(f'[Epoch {epoch+1}/{n_epoches} |  Step_Epoch {i+1}/{len(trainloader)}]')
            step += 1
            scaler.update()

            if profiling: # profiler !!!!!!!!!!!!!!
                prof.step() 

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    return G_BA