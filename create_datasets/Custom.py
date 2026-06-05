import os
import glob
from monai.data import Dataset, list_data_collate
from create_datasets.Mayo import get_transforms, list_sort_nicely, default_collate_fn


# ---------------------------------------------------------
# WBCT_Chest(B-series) -> WBCT_Chest_add(Br-series) 커널명 매핑
# ---------------------------------------------------------
KERNEL_MAP = {'B30f': 'Br36d', 'B45f': 'Br49d', 'B60f': 'Br59d'}

def _map_kernel(source, chest_kernel):
    # WBCT_Chest 는 B-series 그대로, WBCT_Chest_add 는 Br-series 로 변환
    if source == 'WBCT_Chest_add':
        return KERNEL_MAP.get(chest_kernel, chest_kernel)
    return chest_kernel


# ---------------------------------------------------------
# 환자 폴더 목록
# ---------------------------------------------------------
def _list_patients(mode_dir, source, dose):
    base = os.path.join(mode_dir, source, dose)
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


# ---------------------------------------------------------
# 위치(index) 기준 페어링
#  - 입력/정답 슬라이스를 각각 자연정렬한 뒤 N번째끼리 묶는다.
#  - WBCT_Chest_add 처럼 커널마다 번호가 이어져도(Br49d 0316~, Br36d 0001~)
#    N번째 슬라이스는 같은 해부학적 위치이므로 정확히 매칭된다.
#  - 개수가 다르면 경고를 찍고 짧은 쪽까지만 사용 → 밀림을 환자 단위로 가둔다.
# ---------------------------------------------------------
def _pair_by_position(in_glob, gt_glob, tag=""):
    in_list = list_sort_nicely(glob.glob(in_glob))
    gt_list = list_sort_nicely(glob.glob(gt_glob))
    if len(in_list) != len(gt_list):
        print(f"[WARN] slice count mismatch ({tag}): in={len(in_list)}, gt={len(gt_list)}")
    n = min(len(in_list), len(gt_list))
    return list(zip(in_list[:n], gt_list[:n]))


# ---------------------------------------------------------
# task 별 파일 쌍 구성
#   - denoising : 입력 = Quarter,   정답 = Full       (동일 커널)
#   - kernel    : 입력 = in_kernel, 정답 = gt_kernel  (동일 dose)
# ---------------------------------------------------------
def _build_files(mode_dir, args, with_path=False):
    task      = getattr(args, 'task',      'denoising')
    in_kernel = getattr(args, 'in_kernel', 'B30f')
    gt_kernel = getattr(args, 'gt_kernel', 'B30f')
    dose      = (getattr(args, 'dose', 'both') or 'both').lower()

    files = []
    for source in ['WBCT_Chest', 'WBCT_Chest_add']:

        if task == 'denoising':
            kernel = _map_kernel(source, in_kernel)
            pats_q = set(_list_patients(mode_dir, source, 'Quarter'))
            pats_f = set(_list_patients(mode_dir, source, 'Full'))
            for pid in sorted(pats_q & pats_f):
                in_glob = os.path.join(mode_dir, source, 'Quarter', pid, kernel, '*.dcm')
                gt_glob = os.path.join(mode_dir, source, 'Full',    pid, kernel, '*.dcm')
                for a, b in _pair_by_position(in_glob, gt_glob, f"{source}/{pid}/{kernel}"):
                    item = {"n_20": a, "n_100": b}
                    if with_path:
                        item["path_n_20"], item["path_n_100"] = a, b
                    files.append(item)

        elif task == 'kernel':
            in_k = _map_kernel(source, in_kernel)
            gt_k = _map_kernel(source, gt_kernel)
            dose_dirs = {'full': ['Full'], 'quarter': ['Quarter'], 'both': ['Full', 'Quarter']}[dose]
            for d in dose_dirs:
                for pid in _list_patients(mode_dir, source, d):
                    in_glob = os.path.join(mode_dir, source, d, pid, in_k, '*.dcm')
                    gt_glob = os.path.join(mode_dir, source, d, pid, gt_k, '*.dcm')
                    for a, b in _pair_by_position(in_glob, gt_glob, f"{source}/{d}/{pid}"):
                        item = {"n_20": a, "n_100": b}
                        if with_path:
                            item["path_n_20"], item["path_n_100"] = a, b
                        files.append(item)
        else:
            raise ValueError(f"Unknown task: {task} (use 'denoising' or 'kernel')")

    return files


# ---------------------------------------------------------
# Train / Valid
# ---------------------------------------------------------
def CUSTOM_Dataset_DCM(mode, type='window', args=None):
    base_dir = "/workspace/bc_cho/0_Project/2_LDCT2NDCT/dataset/nas69"
    if mode == 'train':
        mode_dir = os.path.join(base_dir, 'train')
    elif mode == 'valid':
        mode_dir = os.path.join(base_dir, 'valid')
    else:
        raise ValueError("mode must be 'train' or 'valid'")

    files = _build_files(mode_dir, args, with_path=False)
    print(f"[{mode}] task={getattr(args, 'task', 'denoising')} | Total pairs: {len(files)}")

    transforms = get_transforms(mode=mode, type=type)
    if mode == 'train' and (type == 'full_patch' or type == 'window_patch'):
        return Dataset(data=files, transform=transforms), list_data_collate
    else:
        return Dataset(data=files, transform=transforms), default_collate_fn


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------
def TEST_CUSTOM_Dataset_DCM(mode='test', type='window_patch', args=None):
    mode_dir = "/workspace/bc_cho/0_Project/2_LDCT2NDCT/dataset/nas206"

    files = _build_files(mode_dir, args, with_path=True)
    print(f"[{mode}] task={getattr(args, 'task', 'denoising')} | Total pairs: {len(files)}")

    transforms = get_transforms(mode=mode, type=type)
    if type == 'full_patch' or type == 'window_patch':
        return Dataset(data=files, transform=transforms), list_data_collate
    else:
        return Dataset(data=files, transform=transforms), default_collate_fn