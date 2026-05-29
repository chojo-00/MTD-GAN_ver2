import os
import glob
from monai.data import Dataset, list_data_collate
from create_datasets.Mayo import get_transforms, list_sort_nicely, default_collate_fn


# ---------------------------------------------------------
# 1. 커널 매핑 함수 추가
# args에서 B-series 이름을 받아오면, 그에 맞는 Br-series 이름을 찾아 반환합니다.
# ---------------------------------------------------------
def get_kernel_names(args):
    # args로 커널을 받지 않았을 경우 작동할 기본값
    in_chest = "B45f"
    gt_chest = "B30f"

    # args 객체가 존재하고, 안에 in_kernel / gt_kernel 값이 있다면 가져옵니다.
    if args is not None:
        if hasattr(args, 'in_kernel') and args.in_kernel:
            in_chest = args.in_kernel
        if hasattr(args, 'gt_kernel') and args.gt_kernel:
            gt_chest = args.gt_kernel

    # 사용자가 정한 매핑 규칙
    kernel_map = {
        'B30f': 'Br36d',
        'B45f': 'Br49d',
        'B60f': 'Br59d'
    }

    # 딕셔너리에서 매핑되는 Br-series를 찾고, 만약 딕셔너리에 정의되지 않은 커널이면 원래 이름을 그대로 사용합니다.
    in_add = kernel_map.get(in_chest, in_chest)
    gt_add = kernel_map.get(gt_chest, gt_chest)

    return in_chest, gt_chest, in_add, gt_add


# ---------------------------------------------------------
# 2. Train / Valid 데이터셋 함수 (args 파라미터 추가)
# ---------------------------------------------------------
def CUSTOM_Dataset_DCM(mode, type='window', args=None):
    # 실제 데이터가 있는 최상위 경로로 수정해주세요.
    base_dir = "/workspace/bc_cho/0_Project/2_LDCT2NDCT/dataset/nas69" 
    
    # Train / Valid 폴더 경로 설정
    if mode == 'train':
        mode_dir = os.path.join(base_dir, 'train')
    elif mode == 'valid':
        mode_dir = os.path.join(base_dir, 'valid')
    else:
        raise ValueError("mode must be 'train' or 'valid'")

    # 추가된 함수를 사용해 동적으로 폴더 이름 4가지를 가져옵니다.
    in_chest_k, gt_chest_k, in_add_k, gt_add_k = get_kernel_names(args)

    # 파라미터에서 데이터 유형 읽어오기 (기본값은 both)
    dose_type = 'both'
    if args is not None and hasattr(args, 'dose') and args.dose:
        dose_type = args.dose.lower()

    n_20_imgs = []
    n_100_imgs = []

    # Full 데이터 로드
    if dose_type in ['full', 'both']:
        n_20_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest/Full/*/{in_chest_k}/*.dcm')))
        n_100_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest/Full/*/{gt_chest_k}/*.dcm')))
        n_20_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest_add/Full/*/{in_add_k}/*.dcm')))
        n_100_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest_add/Full/*/{gt_add_k}/*.dcm')))

    # Quarter 데이터 로드
    if dose_type in ['quarter', 'both']:
        n_20_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest/Quarter/*/{in_chest_k}/*.dcm')))
        n_100_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest/Quarter/*/{gt_chest_k}/*.dcm')))
        n_20_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest_add/Quarter/*/{in_add_k}/*.dcm')))
        n_100_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest_add/Quarter/*/{gt_add_k}/*.dcm')))

    print(f"[{mode}] Dataset Dose Type: {dose_type.upper()}")
    print(f"[{mode}] Input count: {len(n_20_imgs)} (Chest: {in_chest_k}, Add: {in_add_k})")
    print(f"[{mode}] Target count: {len(n_100_imgs)} (Chest: {gt_chest_k}, Add: {gt_add_k})")
    
    if len(n_20_imgs) != len(n_100_imgs):
        print("경고: 입력 이미지와 정답 이미지의 개수가 다릅니다!")

    # 딕셔너리 형태로 묶기
    files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]
    
    # Mayo.py에 정의된 Data Augmentation & Normalization 가져오기
    transforms = get_transforms(mode=mode, type=type)

    # 데이터셋 반환
    if mode == 'train' and (type == 'full_patch' or type == 'window_patch'):
        return Dataset(data=files, transform=transforms), list_data_collate
    else:
        return Dataset(data=files, transform=transforms), default_collate_fn
    

# ---------------------------------------------------------
# 3. Test 데이터셋 함수 (args 파라미터 추가)
# ---------------------------------------------------------
def TEST_CUSTOM_Dataset_DCM(mode='test', type='window_patch', args=None):
    base_dir = "/workspace/bc_cho/0_Project/2_LDCT2NDCT/dataset/nas206" 
    mode_dir = os.path.join(base_dir)

    in_chest_k, gt_chest_k, in_add_k, gt_add_k = get_kernel_names(args)

    dose_type = 'both'
    if args is not None and hasattr(args, 'dose') and args.dose:
        dose_type = args.dose.lower()

    n_20_imgs = []
    n_100_imgs = []

    if dose_type in ['full', 'both']:
        n_20_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest/Full/*/{in_chest_k}/*.dcm')))
        n_100_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest/Full/*/{gt_chest_k}/*.dcm')))
        n_20_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest_add/Full/*/{in_add_k}/*.dcm')))
        n_100_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest_add/Full/*/{gt_add_k}/*.dcm')))

    if dose_type in ['quarter', 'both']:
        n_20_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest/Quarter/*/{in_chest_k}/*.dcm')))
        n_100_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest/Quarter/*/{gt_chest_k}/*.dcm')))
        n_20_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest_add/Quarter/*/{in_add_k}/*.dcm')))
        n_100_imgs += list_sort_nicely(glob.glob(os.path.join(mode_dir, f'WBCT_Chest_add/Quarter/*/{gt_add_k}/*.dcm')))

    print(f"[{mode}] Dataset Dose Type: {dose_type.upper()}")
    print(f"[{mode}] Input count: {len(n_20_imgs)} (Chest: {in_chest_k}, Add: {in_add_k})")
    print(f"[{mode}] Target count: {len(n_100_imgs)} (Chest: {gt_chest_k}, Add: {gt_add_k})")

    # 딕셔너리 형태로 묶기
    files = [{"n_20": n_20, "n_100": n_100, "path_n_20": n_20, "path_n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]
    # Mayo.py에 정의된 Data Augmentation & Normalization 가져오기
    transforms = get_transforms(mode=mode, type=type)

    # 데이터셋 반환 (Test는 주로 기본 collate_fn을 사용합니다)
    if type == 'full_patch' or type == 'window_patch':
        return Dataset(data=files, transform=transforms), list_data_collate
    else:
        return Dataset(data=files, transform=transforms), default_collate_fn