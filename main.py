import copy
import csv
import os
import warnings
from argparse import ArgumentParser
from datetime import datetime
import zipfile

import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

# data_dir = 'D:/datasets/coco'
data_dir = 'D:/dataset_d/mscoco_yolo'


def train(args, params):
    # Model
    version = args.version
    if version == 'n':
        model = nn.yolo_v11_n(len(params['names']))
    elif version == 's':
        model = nn.yolo_v11_s(len(params['names']))
    elif version == 'm':
        model = nn.yolo_v11_m(len(params['names']))
    elif version == 'l':
        model = nn.yolo_v11_l(len(params['names']))
    elif version == 'x':
        model = nn.yolo_v11_x(len(params['names']))
    else:
        raise ValueError(f"Unsupported YOLOv11 variant: {version}. Choose from 'n', 's', 'm', 'l', 'x'.")
    # model = nn.yolo_v11_m(len(params['names']))
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open(f'{data_dir}/train2017.txt') as f:
        for filename in f.readlines():
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}/images/train2017/' + filename)
            # filenames.append(f'./images/train2017/' + filename)
        print("filename lists: ", len(filenames))

    # check if file exists
    existing_count = 0
    nonexisting_count = 0

    for filepath in filenames:
        if os.path.exists(filepath):
            existing_count += 1
        else:
            nonexisting_count += 1

    print(f"Number of existing files: {existing_count}")
    print(f"Number of non-existing files: {nonexisting_count}")

    sampler = None
    dataset = Dataset(filenames, args.input_size, params, augment=True)
    # dataset = Dataset(filenames, args.input_size, params, augment=True, data_dir=data_dir)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)
    
    # loading data
    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # Scheduler
    num_steps = len(loader)
    # print(args)
    # print(params)
    # print(num_steps)
    scheduler = util.LinearLR(args, params, num_steps)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)

    csv_file = f'weights/step_{version}_{args.epochs}.csv' 
    # Initialize lists to record mAP and epoch numbers
    mAP_list = []
    epoch_list = []
    # Write file
    with open(csv_file, 'w', newline='') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()


        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() / 255

                # Forward
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % accumulate == 0:
                    # amp_scale.unscale_(optimizer)  # unscale gradients
                    # util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)
                current_mAP = last[0]  # mAP computed from test()
                mAP_list.append(current_mAP)
                epoch_list.append(epoch + 1)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'mAP': str(f'{last[0]:.3f}'),
                                 'mAP@50': str(f'{last[1]:.3f}'),
                                 'Recall': str(f'{last[2]:.3f}'),
                                 'Precision': str(f'{last[3]:.3f}')})
                log.flush()

                # Update best mAP
                # if last[0] > best:
                #     best = last[0]
                if current_mAP > best:
                    best = current_mAP

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema),
                        # 'model': copy.deepcopy(ema.ema).state_dict()
                        }
                # print(save['model'])

                # Save last, best and delete
                torch.save(save, f=f'./weights/last_{version}_{args.epochs}.pt')
                # if best == last[0]:
                if best == current_mAP:
                    torch.save(save, f=f'./weights/best_{version}_{args.epochs}.pt')
                del save

    if args.local_rank == 0:
        # Finalize logging and close file.
        # log.close()
        util.strip_optimizer(f'./weights/last_{version}_{args.epochs}.pt')  # strip optimizers
        util.strip_optimizer(f'./weights/best_{version}_{args.epochs}.pt')  # strip optimizers

# CSV read and plot mAP function
def plot_mAP(args):
    mAP_list = []
    epoch_list = []
    with open(f"./weights/step_{args.version}_{args.epochs}.csv", "r") as file:
        reader = csv.DictReader(file)  # Reads as a dictionary
        for row in reader:
            epoch_list.append(int(row["epoch"]))  # Convert epoch to integer
            mAP_list.append(float(row["mAP"]))    # Convert mAP to float

    # Find the best mAP and corresponding epoch
    best_mAP = max(mAP_list)
    best_epoch = epoch_list[mAP_list.index(best_mAP)]
    last_mAP = mAP_list[-1]
    last_epoch = epoch_list[-1]
    
    # Plot mAP vs. epochs using Matplotlib
    import matplotlib.pyplot as plt
    # plt.figure()
    # # plt.plot(epoch_list, mAP_list, marker='o', label='mAP')
    # plt.plot(epoch_list, mAP_list, label=f'mAP (last: {last_mAP:.3f}, best: {best_mAP:.3f})')
    # # Highlight best mAP epoch
    # plt.scatter(best_epoch, best_mAP, color='red', label=f'Best mAP at epoch {best_epoch}', zorder=3)
    # plt.xlabel('Epoch')
    # plt.ylabel('mAP')
    # plt.title(f'mAP vs. Epochs (YOLOv11{version} at {args.epochs})')
    # plt.ylim(0,1)
    # plt.grid(True)
    # plt.legend()

    # Create subplots
    fig, ax = plt.subplots(1, 1, layout='constrained')
    # Plot mAP vs. epochs
    ax.plot(epoch_list, mAP_list, label=f'mAP (last: {last_mAP:.3f}, best: {best_mAP:.3f})')
    # Highlight best mAP epoch
    ax.scatter(best_epoch, best_mAP, color='red', label=f'Best mAP at epoch {best_epoch}', zorder=3)
    # Set axis limits
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)  # Set y-axis scale from 0 to 1
    ax.grid(True)
    # Set title and subtitle
    # version = "YOLOv11 version n"  # Change this dynamically based on actual version
    # epochs = last_epoch  # Total epochs
    fig.suptitle("mAP vs. Epochs")
    ax.set_title(f"YOLOv11 version {args.version} at {args.epochs} epochs")
    # Position legend above the graph
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    fig.legend(loc="outside lower center")

    plt.savefig(f"./weights/mAP_vs_epochs_{args.version}_{args.epochs}.png")
    plt.close()

@torch.no_grad()
def test(args, params, model=None):
    version = args.version
    epochs = args.epochs
    filenames = []
    with open(f'{data_dir}/val2017.txt') as f:
        for filename in f.readlines():
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}/images/val2017/' + filename)
            # filenames.append(f'./images/val2017/' + filename)

    dataset = Dataset(filenames, args.input_size, params, augment=False)
    # dataset = Dataset(filenames, args.input_size, params, augment=False, data_dir=data_dir)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    plot = False
    if not model:
        plot = True
        model = torch.load(f=f'./weights/best_{version}_{args.epochs}.pt', map_location='cuda')
        model = model['model'].float().fuse()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))
    
    # Computer mAP
    plot_mAP(args)

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(version, epochs, *metrics, plot=plot, names=params["names"])
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args, params):
    import thop
    shape = (1, 3, args.input_size, args.input_size)
    print(f"params amount: {len(params['names'])}")
    model = nn.yolo_v11_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')

def zip_weights_directory(args):
    weights_dir = "./weights/"
    files_to_zip = []

    # Ensure weights directory exists
    if not os.path.exists(weights_dir):
        print("Error: ./weights/ directory does not exist.")
        return

    # Collect matching files
    for filename in os.listdir(weights_dir):
        if f"_{args.version}_{args.epochs}." in filename or f"_{args.version}_{args.epochs}_state_dict." in filename:  # Match file_n_x.suffix format
            files_to_zip.append(os.path.join(weights_dir, filename))
    print(files_to_zip)

    if not files_to_zip:
        print("No matching files found to zip.")
        return

    # Create ZIP file
    output_zip = f"result_{args.version}_{args.epochs}.zip"
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            zipf.write(file, os.path.basename(file))

    print(f"Successfully created {output_zip} containing {len(files_to_zip)} files.")

def main():
    time_start = datetime.now()
    print("Started at Date and Time:", time_start.strftime("%Y-%m-%d %H:%M:%S"))

    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--version', default='m', type=str)
    parser.add_argument('--zip', action='store_true')

    args = parser.parse_args()
    print(args)

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
        # print(params)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    if args.train:
        # print(args)
        # print(params)
        train(args, params)
    if args.test:
        test(args, params)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()

    if args.zip:
        zip_weights_directory(args)

    time_end = datetime.now()
    print("Finished at Date and Time:", time_end.strftime("%Y-%m-%d %H:%M:%S"))
    time_duration = time_end - time_start
    # Format the duration as Days HH:MM:SS
    days = time_duration.days
    seconds = time_duration.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    formatted_duration = f"{days} Days {hours:02}:{minutes:02}:{seconds:02}"
    print(f"Code execution time: {formatted_duration}")


if __name__ == "__main__":
    main()
