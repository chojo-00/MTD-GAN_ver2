import os
import sys
import utils
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from metrics import compute_feat, compute_FID, compute_PL, compute_TML, compute_RMSE, compute_PSNR, compute_SSIM


# Setting
fn_denorm  = lambda x: (x * 0.5) + 0.5
fn_tonumpy = lambda x: x.cpu().detach().numpy().transpose(0, 2, 3, 1)


# ============================================================
#  MTD-GAN (Ours)
# ============================================================

def train_MTD_GAN_Ours(model, data_loader, optimizer_G, optimizer_D, device, epoch, print_freq, batch_size, method_D):
    model.Generator.train(True)
    model.Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):

        input_n_20  = batch_data['n_20'].to(device).float()
        input_n_100 = batch_data['n_100'].to(device).float()

        if method_D is not None:
            # Discriminator (multi-task weighting: PCGrad / CAGrad / MGDA / ...)
            optimizer_D.zero_grad()
            model.Discriminator.zero_grad()
            d_losses, d_loss_details = model.d_loss(input_n_20, input_n_100)
            actual_discriminator = model.Discriminator.module if hasattr(model.Discriminator, 'module') else model.Discriminator
            loss_D, extra_outputs_D = method_D.backward(
                losses=d_losses,
                shared_parameters=list(actual_discriminator.shared_parameters()),
                task_specific_parameters=list(actual_discriminator.task_specific_parameters()),
                last_shared_parameters=list(actual_discriminator.last_shared_parameters()),
            )
            optimizer_D.step()
            metric_logger.update(d_loss=sum(d_losses))
            metric_logger.update(**d_loss_details)

            # Generator
            optimizer_G.zero_grad()
            model.Generator.zero_grad()
            g_loss, g_loss_details = model.g_loss(input_n_20, input_n_100)
            g_loss.backward()
            optimizer_G.step()
            metric_logger.update(g_loss=g_loss)
            metric_logger.update(**g_loss_details)
            metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])

        else:
            # Discriminator (vanilla sum-of-losses)
            optimizer_D.zero_grad()
            model.Discriminator.zero_grad()
            d_loss, d_loss_details = model.d_loss(input_n_20, input_n_100)
            d_loss.backward()
            optimizer_D.step()
            metric_logger.update(d_loss=d_loss)
            metric_logger.update(**d_loss_details)

            # Generator
            optimizer_G.zero_grad()
            model.Generator.zero_grad()
            g_loss, g_loss_details = model.g_loss(input_n_20, input_n_100)
            g_loss.backward()
            optimizer_G.step()
            metric_logger.update(g_loss=g_loss)
            metric_logger.update(**g_loss_details)
            metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def valid_MTD_GAN_Ours(model, loss, data_loader, device, epoch, save_dir, print_freq):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'Valid: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20  = batch_data['n_20'].to(device).float()
        input_n_100 = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)

        L1_loss = loss(pred_n_100, input_n_100)
        metric_logger.update(L1_loss=L1_loss.item())

    # Denormalize (windowing input version)
    input_n_20  = fn_tonumpy(input_n_20)
    input_n_100 = fn_tonumpy(input_n_100)
    pred_n_100  = fn_tonumpy(pred_n_100)

    # PNG Save
    plt.imsave(save_dir + '/epoch_' + str(epoch) + '_input_n_20.png',  input_n_20[0].squeeze(),  cmap="gray")
    plt.imsave(save_dir + '/epoch_' + str(epoch) + '_gt_n_100.png',    input_n_100[0].squeeze(), cmap="gray")
    plt.imsave(save_dir + '/epoch_' + str(epoch) + '_pred_n_100.png',  pred_n_100[0].squeeze(),  cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_MTD_GAN_Ours(model, loss, data_loader, device, save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    input_features  = []
    target_features = []
    pred_features   = []

    # Per-sample metric records
    path_list = []
    pl_list   = []
    tml_list  = []
    rmse_list = []
    psnr_list = []
    ssim_list = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):

        input_n_20  = batch_data['n_20'].to(device).float()
        input_n_100 = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)

        L1_loss = loss(pred_n_100, input_n_100)
        metric_logger.update(L1_loss=L1_loss.item())

        # SAVE folders
        os.makedirs(save_dir.replace('/png/', '/dcm/'), mode=0o777, exist_ok=True)  # dicom save folder
        os.makedirs(save_dir,                            mode=0o777, exist_ok=True)  # png save folder

        # Metrics
        input_pl,   gt_pl,   pred_pl    = compute_PL(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_tml,  gt_tml,  pred_tml   = compute_TML(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_rmse, gt_rmse, pred_rmse  = compute_RMSE(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_psnr, gt_psnr, pred_psnr  = compute_PSNR(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_ssim, gt_ssim, pred_ssim  = compute_SSIM(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))

        input_feat, target_feat, pred_feat = compute_feat(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_features.append(input_feat); target_features.append(target_feat); pred_features.append(pred_feat)

        metric_logger.update(input_pl=input_pl, input_tml=input_tml, input_rmse=input_rmse, input_psnr=input_psnr, input_ssim=input_ssim)
        metric_logger.update(gt_pl=gt_pl,       gt_tml=gt_tml,       gt_rmse=gt_rmse,       gt_psnr=gt_psnr,       gt_ssim=gt_ssim)
        metric_logger.update(pred_pl=pred_pl,   pred_tml=pred_tml,   pred_rmse=pred_rmse,   pred_psnr=pred_psnr,   pred_ssim=pred_ssim)

        # Denormalize (windowing input version)
        input_n_20  = fn_tonumpy(input_n_20)
        input_n_100 = fn_tonumpy(input_n_100)
        pred_n_100  = fn_tonumpy(pred_n_100)

        # PNG Save
        plt.imsave(save_dir + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),    input_n_20.squeeze(),  cmap="gray")
        plt.imsave(save_dir + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'), input_n_100.squeeze(), cmap="gray")
        plt.imsave(save_dir + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),pred_n_100.squeeze(),  cmap="gray")

        # Per-sample records
        path_list.append(batch_data['path_n_20'][0])
        pl_list.append(pred_pl.item())
        tml_list.append(pred_tml.item())
        rmse_list.append(pred_rmse)
        psnr_list.append(pred_psnr)
        ssim_list.append(pred_ssim)

    # DataFrame dump
    df = pd.DataFrame()
    df['PATH'] = path_list
    df['PL']   = pl_list
    df['TML']  = tml_list
    df['RMSE'] = rmse_list
    df['PSNR'] = psnr_list
    df['SSIM'] = ssim_list
    df.to_csv(save_dir + '/pred_results.csv')

    # FID
    input_fid, gt_fid, pred_fid = compute_FID(
        torch.cat(input_features, dim=0),
        torch.cat(target_features, dim=0),
        torch.cat(pred_features, dim=0),
    )
    metric_logger.update(input_fid=input_fid, gt_fid=gt_fid, pred_fid=pred_fid)

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}