import os
import sys
import utils
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler
from metrics import compute_LPIPS, compute_RMSE, compute_PSNR, compute_SSIM
import lpips

# Setting
fn_denorm  = lambda x: (x * 0.5) + 0.5
fn_tonumpy = lambda x: x.cpu().detach().numpy().transpose(0, 2, 3, 1)

scaler_D = GradScaler()
scaler_G = GradScaler()


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
            
            # --- Forward에 autocast 적용 ---
            with autocast():
                d_losses, d_loss_details = model.d_loss(input_n_20, input_n_100)
                
            actual_discriminator = model.Discriminator.module if hasattr(model.Discriminator, 'module') else model.Discriminator
            
            # method_D.backward 내부 연산의 안정성을 위해 float32로 캐스팅하여 전달합니다.
            loss_D, extra_outputs_D = method_D.backward(
                losses=d_losses.float(),  # <--- 파이토치 텐서 유지
                shared_parameters=list(actual_discriminator.shared_parameters()),
                task_specific_parameters=list(actual_discriminator.task_specific_parameters()),
                last_shared_parameters=list(actual_discriminator.last_shared_parameters()),
            )
            optimizer_D.step()
            metric_logger.update(d_loss=d_losses.sum().item()) # <--- sum() 연산도 파이토치 내장 함수로 변경
            metric_logger.update(**d_loss_details)


            # Generator
            optimizer_G.zero_grad()
            model.Generator.zero_grad()
            
            # --- Forward에 autocast 적용 ---
            with autocast():
                g_loss, g_loss_details = model.g_loss(input_n_20, input_n_100)
                
            # --- Scaler를 통한 Backward 및 Optimizer Step ---
            scaler_G.scale(g_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            
            metric_logger.update(g_loss=g_loss.item())
            metric_logger.update(**g_loss_details)
            metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])

        else:
            # Discriminator (vanilla sum-of-losses)
            optimizer_D.zero_grad()
            model.Discriminator.zero_grad()
            
            # --- Forward에 autocast 적용 ---
            with autocast():
                d_loss, d_loss_details = model.d_loss(input_n_20, input_n_100)
                
            # --- Scaler를 통한 Backward 및 Optimizer Step ---
            scaler_D.scale(d_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            
            metric_logger.update(d_loss=d_loss.item())
            metric_logger.update(**d_loss_details)

            # Generator
            optimizer_G.zero_grad()
            model.Generator.zero_grad()
            
            # --- Forward에 autocast 적용 ---
            with autocast():
                g_loss, g_loss_details = model.g_loss(input_n_20, input_n_100)
                
            # --- Scaler를 통한 Backward 및 Optimizer Step ---
            scaler_G.scale(g_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            
            metric_logger.update(g_loss=g_loss.item())
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

    # Denormalize
    input_n_20  = fn_tonumpy(input_n_20)
    input_n_100 = fn_tonumpy(input_n_100)
    pred_n_100  = fn_tonumpy(pred_n_100)

    # PNG Save (epoch별 1장, 모니터링용)
    plt.imsave(save_dir + '/epoch_' + str(epoch) + '_input_n_20.png', input_n_20[0].squeeze(),  cmap="gray")
    plt.imsave(save_dir + '/epoch_' + str(epoch) + '_gt_n_100.png',   input_n_100[0].squeeze(), cmap="gray")
    plt.imsave(save_dir + '/epoch_' + str(epoch) + '_pred_n_100.png', pred_n_100[0].squeeze(),  cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_MTD_GAN_Ours(model, loss, data_loader, device, save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    # LPIPS 모델 1회 생성 후 재사용 (alex가 표준, 원하면 net='vgg')
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()


    # Per-sample records
    path_list  = []
    lpips_list = []
    rmse_list  = []
    psnr_list  = []
    ssim_list  = []

    header = 'TEST:'
    print_freq = 10
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):

        input_n_20  = batch_data['n_20'].to(device).float()
        input_n_100 = batch_data['n_100'].to(device).float()
        pred_n_100  = model.Generator(input_n_20)

        L1_loss = loss(pred_n_100, input_n_100)
        metric_logger.update(L1_loss=L1_loss.item())

        # SAVE folders
        in_path = batch_data['path_n_20'][0]
        parts   = in_path.split('/')
        
        # 1. WBCT_Chest 혹은 WBCT_Chest_add 추출
        src_idx = next((i for i, p in enumerate(parts) if p.startswith('WBCT_Chest')), None)
        if src_idx is not None:
            source_dir = parts[src_idx]        # WBCT_Chest / WBCT_Chest_add
            # 경로 구조: WBCT_Chest / Quarter(dose) / 환자ID / 커널 / 파일명
            patient_id = parts[src_idx + 2]    # ANONY_004 등
        else:
            source_dir = 'unknown'
            patient_id = 'unknown'

        patient_save_dir = os.path.join(save_dir, source_dir, patient_id)
        os.makedirs(patient_save_dir, mode=0o777, exist_ok=True)
        fname = parts[-1]   # 파일명 (예: 0316.dcm)

        # --- Metrics ---
        pred_clip = pred_n_100.clip(0, 1)
        input_lpips, gt_lpips, pred_lpips = compute_LPIPS(lpips_fn, input=input_n_20, target=input_n_100, pred=pred_clip)
        input_rmse,  gt_rmse,  pred_rmse  = compute_RMSE(input=input_n_20, target=input_n_100, pred=pred_clip)
        input_psnr,  gt_psnr,  pred_psnr  = compute_PSNR(input=input_n_20, target=input_n_100, pred=pred_clip)
        input_ssim,  gt_ssim,  pred_ssim  = compute_SSIM(input=input_n_20, target=input_n_100, pred=pred_clip)

        metric_logger.update(input_lpips=input_lpips, input_rmse=input_rmse, input_psnr=input_psnr, input_ssim=input_ssim)
        metric_logger.update(gt_lpips=gt_lpips,       gt_rmse=gt_rmse,       gt_psnr=gt_psnr,       gt_ssim=gt_ssim)
        metric_logger.update(pred_lpips=pred_lpips,   pred_rmse=pred_rmse,   pred_psnr=pred_psnr,   pred_ssim=pred_ssim)

        # Denormalize (windowing input version)
        input_n_20  = fn_tonumpy(input_n_20)
        input_n_100 = fn_tonumpy(input_n_100)
        pred_n_100  = fn_tonumpy(pred_n_100)

        # 3. 환자별로 생성된 폴더 내부에 각각 안전하게 저장
        plt.imsave(os.path.join(patient_save_dir, fname.replace('.dcm', '_gt_n_20.png')),    input_n_20.squeeze(),  cmap="gray")
        plt.imsave(os.path.join(patient_save_dir, fname.replace('.dcm', '_gt_n_100.png')),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(os.path.join(patient_save_dir, fname.replace('.dcm', '_pred_n_100.png')), pred_n_100.squeeze(),  cmap="gray")
        # Per-sample records
        path_list.append(batch_data['path_n_20'][0])
        lpips_list.append(pred_lpips)
        rmse_list.append(pred_rmse)
        psnr_list.append(pred_psnr)
        ssim_list.append(pred_ssim)

    # DataFrame dump
    df = pd.DataFrame()
    df['PATH']  = path_list
    df['LPIPS'] = lpips_list
    df['RMSE']  = rmse_list
    df['PSNR']  = psnr_list
    df['SSIM']  = ssim_list
    df.to_csv(save_dir + '/pred_results.csv')

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}