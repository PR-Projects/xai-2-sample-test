import torch
import numpy as np
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
import os

from model import *
from data_fintune import *
import argparse

def save_checkpoint(path, model, optimizer, epoch, best_val_loss, scheduler=None, extra=None):
    # If using DataParallel/DistributedDataParallel, save the underlying module
    payload = {
        "epoch": epoch,                       # 1-based epoch number
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
    }
    if extra: payload.update(extra)
    torch.save(payload, path)


def finetune(num_epochs=10, batch_size=32, args=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Loading data 
    root_dir = Path('/path/to/data')
    ckpt_dir = root_dir / "AdniGithub"/ "adni_results" / "ckps"
    if args.corrupted:
        print(f'upload corrupted images {args.deg}')
        train_dir = Path(root_dir) / "AdniGithub"/ "adni_results" / "split" / "train" / str(args.corrupted) / str(args.deg)
        print(f'data corrupted:{args.corrupted}, degree:{args.deg} is loaded!')
    else:
        print('upload non-corrupted images')
        if args.task=='hippo':
            # this is for hippocampus volume stratification task
            train_dir = Path(root_dir) / "AdniGithub"/ "adni_results" / "split" / "train" / str(args.corrupted)/ str(args.deg)
        else:
            # this is for CN-AD or MCI-AD tasks
            train_dir = Path(root_dir) / "AdniGithub"/ "adni_results" / "split" / "train" / str(args.corrupted)/ str(args.deg) / str(args.task) / 'float'
    loss_dir = root_dir / "AdniGithub"/ "adni_results" / "loss"
    data = np.load(train_dir / "train_val_splits.npz")
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"]

    # Making dataloaders
    if args.task=='hippo':
        # this is for hippocampus volume stratification task, no rubust float needed 
        train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size=32, num_workers=4)
    else:
        # this is for CN-AD or MCI-AD tasks, we need robust float
        # this handels the float images in a robust way
        print('Using robust float loader for CN-AD or MCI-AD task!')
        train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, batch_size=32, num_workers=4, robust_float_norm=True)


    # path where I saved pre-trained model on Diabetic Rethinopathy dataset => self-supervised SimCLR
    if args.pre=='selfsup':
        print('Using self-supervised pre-trained model')
        base_path = "/path/to/data/Retina_Codes"
        root_dir = Path(base_path)
        checkpoint_dir = root_dir / 'self_supervised' / 'simclr' / 'simclr_ckpts'
        pre_exp = 2
        sam_dir_last = os.path.join(checkpoint_dir, f'{pre_exp}_last_sclr.pt')
        state_dict = torch.load(sam_dir_last, weights_only=False, map_location=device) 
        # pre-trained resnet50 backbone which is used in SimCLR
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        net = SimCLR(backbone, hid_dim=2048, out_dim=128).to(device)
        net.load_state_dict(state_dict)
        # Best case: networksimclr exposes .encoder (pre-projection)
        # Encoder backbone: we remove projecttion head from SimCLR
        encoder = net.encoder

    elif args.pre=='sup':
        print('Using supervised pre-trained model')
        base_path = Path('../adni_results/ckps')
        checkpoint_dir = base_path / 'resnet50_ukb_age_predict_epoch13.pth'
        net = ResNet50Predictor(embed_dim=2048, dropout=0.5).to(device)
        weights = torch.load(checkpoint_dir, map_location=device)
        net.load_state_dict(weights)
        encoder = net
        print('supervised model loaded')


    model = finetune_net(encoder, num_classes=2).to(device)
    #print('#################################')
    #print(f'whole model:{model}')
    #print('#################################')
    encoder = model.feature_extractor
    #print('#################################')
    #print(f'encoder:{encoder}')
    #print('#################################')
    container = encoder[0] if isinstance(encoder[0], nn.Sequential) else encoder
    layer4 = container[7]   # final ResNet stage
    last_block = layer4[-1] # last Bottleneck in that stage (index 2)
    #print('#################################')
    #print(f'last_block:{last_block}')
    #print('#################################')
    last_conv3 = last_block.conv3
    last_bn3   = last_block.bn3



    # We freeze all layers except the final layer (linear probing)
    def freeze_all(m): 
        for p in m.parameters(): p.requires_grad = False

    # Unfreeze the last layers of the model if needed
    def unfreeze_module(m):
        for p in m.parameters(): p.requires_grad = True

    # Option 1: Linear probe only (recommended start)
    if args.freez=='all':
        freeze_all(encoder)
        print('All layers were freezed! except the linear layer on top of that!')
        # Linear probe only:
        opt = torch.optim.AdamW(model.linear.parameters(), lr=1e-3, weight_decay=1e-4)
    elif args.freez=='none':
        print('All layers are trainable!')
        # Option 2: Unfreeze all layers (after probe baseline)
        unfreeze_module(encoder)
        print('All layers were unfreezed!')
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif args.freez=='lastblk':
        print('All layers were freezed except the last block and linear layer on top of that!')
        # freeze all first
        freeze_all(encoder)
        # unfreeze last block
        for p in last_block.parameters():
            p.requires_grad = True

        param_groups = [
            {"params": model.linear.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
            {"params": last_block.parameters(),   "lr": 1e-4, "weight_decay": 1e-4},  # smaller LR for convs
            ]
        opt = torch.optim.AdamW(param_groups)

    elif args.freez=='lastconv':
        print('All layers were freezed except conv3+bn3 in the last block and the linear layer!')
        # 1. freeze encoder
        freeze_all(encoder)

        # 2. unfreeze conv3 and bn3 in last block
        for p in last_conv3.parameters():
            p.requires_grad = True
        for p in last_bn3.parameters():
            p.requires_grad = True

        # 3. optimizer over just those + head
        param_groups = [
            {"params": model.linear.parameters(),   "lr": 1e-3, "weight_decay": 1e-4},
            {"params": last_conv3.parameters(),     "lr": 1e-4, "weight_decay": 1e-4},
            {"params": last_bn3.parameters(),       "lr": 1e-4, "weight_decay": 1e-4},
        ]
        opt = torch.optim.AdamW(param_groups)


    # check list of trainable paameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", total_params)
    

    # Defining loss function
    # Compute the cross-entropy loss between the predicted logits and the true labels
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val = float('inf'); best_state=None

    loss_tr_list = []
    loss_val_list = []

    for epoch in range(num_epochs):
        model.train() 
        loss_epoch = 0
        correct_tr = 0
        n_train = 0
        # model.encoder.eval()
        for xb, yb in train_loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            # accumulate loss
            bs = xb.size(0)
            loss_epoch += loss.item()* bs 
            n_train += bs
            pred_tr = logits.argmax(1); correct_tr += (pred_tr==yb).sum().item()
            opt.zero_grad(); loss.backward(); opt.step()
        loss_epoch /= n_train; train_acc = correct_tr / n_train
        loss_tr_list.append(loss_epoch)
        # validate
        model.eval(); val_loss=0; correct=0; n=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device).float(), yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item()*xb.size(0)
                pred = logits.argmax(1); correct += (pred==yb).sum().item(); n+=xb.size(0)
        val_loss/=n; val_acc=correct/n
        loss_val_list.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
            ckpt_dir / f"model_finetun_best_{args.exp}_{args.corrupted}.pt",
            model=model,
            optimizer=opt,
            scheduler=None,  # or pass your scheduler if you have one
            epoch=epoch + 1,
            best_val_loss=best_val)

        # print train and val losses and val accuracy for each batch
        print(f'Epoch {epoch+1}/{num_epochs} | Train loss: {loss_epoch:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Train acc: {train_acc:.4f}')
        # save last model 
        # -------- always save "last" for this epoch --------
        save_checkpoint(
        ckpt_dir / f"model_finetun_last_{args.exp}_{args.corrupted}.pt",
        model=model,
        optimizer=opt,
        scheduler=None,  # or pass your scheduler if you have one
        epoch=epoch + 1,
        best_val_loss=best_val,
    )
        
    # save losses
    stem = loss_dir / f"loss_{args.exp}_{args.corrupted}"
    np.savez_compressed(
    stem.with_suffix(".npz"),
    loss_tr=np.asarray(loss_tr_list, dtype=float),
    loss_val=np.asarray(loss_val_list, dtype=float),)


    # load best
    # model.load_state_dict({k:v.to(device) for k,v in best_state.items()})

# here we can load test set and evaluate model on it
parser = argparse.ArgumentParser(description='Fine tuning pre-trained model on datasets!')
parser.add_argument('--exp', type=int, default= 24, help='Experiment number for fine-tuning')
parser.add_argument('--pre', type=str, default= 'sup', help='Type of pre-trained model: sup or selsup')
parser.add_argument('--corrupted', type=str, default=False, help='Use corrupted images for group 1')
parser.add_argument('--freez', type=str, default='all',choices=('all','none','lastblk','lastconv'), help='If we want to freeze all layers except the linear layer on top')
parser.add_argument('--deg', type=str, default=None, choices=('zer32','circ'), help='Degree of corruption: 4 or 8 or None, if we do not use corrupted images')      
parser.add_argument('--task', type=str, default='CN_AD', choices=('hippo','CN_AD', 'MCI_AD'), help='Set task for finetuning: based on stratification')
if __name__=="__main__":
    
    args = parser.parse_args()
    finetune(num_epochs=150, batch_size=32, args=args)







    
    



    

