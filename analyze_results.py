"""
analyze_results.py
------------------
pred_results.csv 를 가지고
  (1) 각 평가지표(RMSE[HU], PSNR[dB], SSIM)의 평균/분산 + 히스토그램
  (2) best / worst / random 각 k(기본 10)장 선택  -> 총 30 슬라이스
        --unique-patient : best/worst 선택 시 한 환자당 1 슬라이스 (random 은 그대로)
  (3) 그리드(3행 = Input/Pred/GT) 저장
        - 카테고리 10장을 GRID_COLS(=5)장씩 2파트, 윈도 3종(nowin/lung/abdomen)
        - 3 카테고리 x 2 파트 x 3 윈도 = 18 장 PNG
        - 열 제목은 slice id (예: WBCT_Chest__ANONY_002__00001)
  (4) 개별 PNG : individual_png/<category>/  -> 30 x (input/gt/pred) x (nowin/lung/abdomen) = 270 장
  (5) 개별 DICOM : dicom/<category>/         -> 30 x (input/gt/pred) = 90 개 (raw HU)

파일명 형식:
  PNG  : WBCT_Chest__ANONY_002__00001__pred__lung.png
  DICOM: WBCT_Chest__ANONY_002__00001__pred.dcm

* test.py 와 동일한 docker 컨테이너(같은 repo 루트, CODE_DIR)에서 실행.

실행 예:
  python3 analyze_results.py \
      --csv      /workspace/.../pred_results.csv \
      --resume   /workspace/.../epoch_9_checkpoint.pth \
      --save-dir /workspace/.../analysis \
      --in-kernel B45f --gt-kernel B30f --k 10 --unique-patient
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import pydicom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast

from models import get_model
from create_datasets import Custom
from create_datasets.Mayo import get_pixels_hu, dicom_normalize
from utils import save_dicom


# ----------------------------------------------------------------------
# 설정값
# ----------------------------------------------------------------------
MIN_HU, MAX_HU = -1024.0, 3072.0
HU_RANGE = MAX_HU - MIN_HU                  # 4096  -> RMSE_norm * 4096 = RMSE_HU

WINDOWS = [("nowin",   None,   None),
           ("lung",   -750.0, 1500.0),
           ("abdomen",  30.0,  150.0)]      # (name, WL, WW). WL=None -> 윈도 미적용

GRID_COLS = 5                               # 그리드 1장당 슬라이스 수


def denorm_to_hu(x01):
    return x01 * HU_RANGE + MIN_HU


def apply_window(hu, wl, ww):
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def to_disp(hu, wl=None, ww=None):
    if wl is None:
        return np.clip(dicom_normalize(hu), 0.0, 1.0)
    return apply_window(hu, wl, ww)


def patient_key(path):
    """환자 식별 키 = ANONY_xxx 까지의 전체 경로 (커널/파일명 제거).
       .../WBCT_Chest/Full/ANONY_002/B45f/00001.dcm -> .../WBCT_Chest/Full/ANONY_002
       WBCT_Chest vs WBCT_Chest_add, share vs share_add 도 경로가 달라 구분됨."""
    return os.path.dirname(os.path.dirname(path))


def slice_id(path):
    """파일명용 slice id = <dataset>__<patient>__<slice>.
       .../WBCT_Chest/Full/ANONY_002/B45f/00001.dcm -> WBCT_Chest__ANONY_002__00001"""
    parts = path.split("/")
    ds, pid, stem = parts[-5], parts[-3], os.path.splitext(parts[-1])[0]
    return f"{ds}__{pid}__{stem}"


# ----------------------------------------------------------------------
# (1) 통계 + 히스토그램
# ----------------------------------------------------------------------
def stats_and_hist(df, save_dir):
    df = df.copy()
    df["RMSE_HU"] = df["RMSE"] * HU_RANGE

    print("\n========== Metric summary (N=%d) ==========" % len(df))
    rows = []
    for col, label in [("RMSE_HU", "RMSE (HU)"), ("PSNR", "PSNR (dB)"),
                   ("SSIM", "SSIM"), ("LPIPS", "LPIPS")]:   # PL, TML 제거 / LPIPS 추가
        s = df[col]
        print(f"{label:10s} | mean={s.mean():.4f}  var={s.var():.6f}  "
              f"std={s.std():.4f}  median={s.median():.4f}  "
              f"min={s.min():.4f}  max={s.max():.4f}")
        rows.append({"metric": label, "mean": s.mean(), "variance": s.var(),
                     "std": s.std(), "median": s.median(),
                     "min": s.min(), "max": s.max()})
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "metric_summary.csv"), index=False)

    metrics = [("RMSE_HU", "RMSE (HU)", "tab:blue"),
               ("PSNR", "PSNR (dB)", "tab:orange"),
               ("SSIM", "SSIM", "tab:green")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    fig.suptitle("Metric Distributions across Test Samples (N=%d)" % len(df),
                 fontweight="bold")
    for ax, (col, label, color) in zip(axes, metrics):
        s = df[col]
        ax.hist(s, bins=40, color=color, alpha=0.8)
        ax.axvline(s.mean(),   color="red",   ls="--", lw=1.5, label=f"Mean={s.mean():.3f}")
        ax.axvline(s.median(), color="black", ls=":",  lw=1.5, label=f"Median={s.median():.3f}")
        ax.set_title(label); ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(save_dir, "metric_distributions.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved:", out)


# ----------------------------------------------------------------------
# (2) best / worst / random 선택
# ----------------------------------------------------------------------
def _topk_unique_patient(ordered_df, k):
    """정렬된 df 를 위에서부터 훑어 환자 중복 없이 k 개 인덱스 수집."""
    seen, out = set(), []
    for idx in ordered_df.index:
        pk = patient_key(ordered_df.loc[idx, "PATH"])
        if pk in seen:
            continue
        seen.add(pk); out.append(idx)
        if len(out) == k:
            break
    return out


def select_indices(df, rank_metric, k, seed, unique_patient=False):
    higher_better = rank_metric in ("SSIM", "PSNR")     # RMSE/PL/TML 은 낮을수록 좋음
    ordered = df.sort_values(rank_metric, ascending=not higher_better)  # best -> worst

    if unique_patient:
        best  = _topk_unique_patient(ordered,            k)   # 위(=best)에서
        worst = _topk_unique_patient(ordered.iloc[::-1], k)   # 아래(=worst)에서
    else:
        best  = ordered.head(k).index.tolist()
        worst = ordered.tail(k).index.tolist()[::-1]

    pool = df.index.difference(best + worst)
    rng = np.random.default_rng(seed)
    rand = rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()

    tags = ["BEST"] * len(best) + ["WORST"] * len(worst) + ["RANDOM"] * len(rand)
    return best + worst + rand, tags


# ----------------------------------------------------------------------
# 선택 슬라이스 재추론
# ----------------------------------------------------------------------
@torch.no_grad()
def infer_selected(df, sel_idx, args, device):
    class A: pass
    a = A()
    a.task = args.task
    a.in_kernel = args.in_kernel
    a.gt_kernel = args.gt_kernel
    a.dose = "both"
    a.dataset_type_test = "full"
    dataset, _ = Custom.TEST_CUSTOM_Dataset_DCM(mode="test", type="full", args=a)
    pair = {d["path_n_20"]: d["path_n_100"] for d in dataset.data}

    model = get_model(name=args.model)
    ckpt = torch.load(args.resume, map_location="cpu")
    ckpt["model_state_dict"] = {k.replace(".module", ""): v
                                for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(ckpt["model_state_dict"])
    model.Generator.to(device).eval()

    results = []
    for idx in sel_idx:
        in_path = df.loc[idx, "PATH"]
        gt_path = pair.get(in_path, None)
        if gt_path is None:
            raise FileNotFoundError(f"GT 매핑 실패: {in_path} "
                                    f"(in/gt kernel 또는 dose 설정 확인)")

        inp_hu = get_pixels_hu(in_path).astype(np.float32)
        gt_hu  = get_pixels_hu(gt_path).astype(np.float32)

        inp01 = dicom_normalize(inp_hu)
        x = torch.from_numpy(inp01)[None, None].float().to(device)
        with autocast():
            pred01 = model.Generator(x)
        pred01 = pred01.float().clamp(0, 1).cpu().numpy()[0, 0]
        pred_hu = denorm_to_hu(pred01)

        results.append(dict(
            idx=idx, in_path=in_path, gt_path=gt_path,
            inp_hu=inp_hu, pred_hu=pred_hu, gt_hu=gt_hu,
            rmse_hu=df.loc[idx, "RMSE"] * HU_RANGE,
            psnr=df.loc[idx, "PSNR"], ssim=df.loc[idx, "SSIM"],
        ))
    return results


# ----------------------------------------------------------------------
# (3) 그리드 (Input / Pred / GT)
# ----------------------------------------------------------------------
def draw_grid(items, title, save_path, wl=None, ww=None):
    n = len(items)
    rows = ["Input (n_20)", "Pred", "GT (n_100)"]
    fig, axes = plt.subplots(3, n, figsize=(1.9 * n, 5.8))
    if n == 1:
        axes = axes[:, None]
    fig.suptitle(title, fontweight="bold", y=1.02)

    for j, r in enumerate(items):
        imgs = [to_disp(r["inp_hu"], wl, ww),
                to_disp(r["pred_hu"], wl, ww),
                to_disp(r["gt_hu"],  wl, ww)]
        for i in range(3):
            ax = axes[i, j]
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=10)
        axes[0, j].set_title(
            f"{slice_id(r['in_path'])}\nRMSE={r['rmse_hu']:.2f} HU\n"
            f"PSNR={r['psnr']:.2f}\nSSIM={r['ssim']:.4f}", fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved:", save_path)


# ----------------------------------------------------------------------
# (4)(5) 개별 PNG + DICOM
# ----------------------------------------------------------------------
def save_individual(groups, png_root, dcm_root):
    for cat, items in groups.items():
        png_cat = os.path.join(png_root, cat.lower())
        dcm_cat = os.path.join(dcm_root, cat.lower())
        os.makedirs(png_cat, exist_ok=True)
        os.makedirs(dcm_cat, exist_ok=True)

        for r in items:
            sid = slice_id(r["in_path"])
            triples = [("input", r["inp_hu"]),     # n_20
                       ("gt",    r["gt_hu"]),       # n_100
                       ("pred",  r["pred_hu"])]     # 예측

            # ----- PNG : type 3 x window 3 -----
            for name, hu in triples:
                for wname, wl, ww in WINDOWS:
                    disp = to_disp(hu) if wl is None else to_disp(hu, wl, ww)
                    plt.imsave(os.path.join(png_cat, f"{sid}__{name}__{wname}.png"),
                               disp, cmap="gray", vmin=0, vmax=1)

            # ----- DICOM : raw HU -----
            pydicom.dcmread(r["in_path"]).save_as(os.path.join(dcm_cat, f"{sid}__input.dcm"))
            pydicom.dcmread(r["gt_path"]).save_as(os.path.join(dcm_cat, f"{sid}__gt.dcm"))
            save_dicom(r["in_path"], r["pred_hu"].astype(np.float32),
                       os.path.join(dcm_cat, f"{sid}__pred.dcm"))
    print(f"saved individual PNG -> {png_root} , DICOM -> {dcm_root}")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--resume", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--model", default="MTD_GAN_Method")
    p.add_argument("--in-kernel", default="B45f")
    p.add_argument("--gt-kernel", default="B30f")
    p.add_argument("--rank-metric", default="SSIM",
               choices=["SSIM", "PSNR", "RMSE", "LPIPS"])   # PL, TML 제거 / LPIPS 추가
    p.add_argument("--k", default=10, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--unique-patient", action="store_true",
                   help="best/worst 선택 시 한 환자당 1 슬라이스만 (random 은 그대로)")
    p.add_argument("--task", default="kernel", choices=["denoising", "kernel"])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    grid_dir = os.path.join(args.save_dir, "grids")
    png_root = os.path.join(args.save_dir, "individual_png")
    dcm_root = os.path.join(args.save_dir, "dicom")
    for d in (args.save_dir, grid_dir, png_root, dcm_root):
        os.makedirs(d, exist_ok=True)

    device = torch.device(args.device)
    df = pd.read_csv(args.csv, index_col=0)

    # (1)
    stats_and_hist(df, args.save_dir)

    # (2)
    sel_idx, tags = select_indices(df, args.rank_metric, args.k, args.seed,
                                   unique_patient=args.unique_patient)
    results = infer_selected(df, sel_idx, args, device)

    groups = {"BEST": [], "WORST": [], "RANDOM": []}
    for r, t in zip(results, tags):
        groups[t].append(r)

    # (3) 그리드 18 장
    for cat, items in groups.items():
        for part_off in range(0, len(items), GRID_COLS):
            chunk = items[part_off:part_off + GRID_COLS]
            part_no = part_off // GRID_COLS + 1
            for wname, wl, ww in WINDOWS:
                wtitle = "No Windowing (full HU)" if wl is None \
                         else f"WL={wl:.0f} HU, WW={ww:.0f} HU ({wname})"
                draw_grid(chunk, f"{cat} part{part_no} | {wtitle}",
                          os.path.join(grid_dir,
                                       f"grid_{cat.lower()}_part{part_no}_{wname}.png"),
                          wl, ww)

    # (4)(5) 개별 PNG 270 + DICOM 90
    save_individual(groups, png_root, dcm_root)

    print("\nDone. ->", args.save_dir)


if __name__ == "__main__":
    main()