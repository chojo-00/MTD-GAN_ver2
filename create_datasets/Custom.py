import os
import glob
from monai.data import Dataset, list_data_collate
from create_datasets.Mayo import get_transforms, list_sort_nicely, default_collate_fn

def CUSTOM_Dataset_DCM(mode, type='window'):
    # 실제 데이터가 있는 최상위 경로로 수정해주세요.
    base_dir = "/workspace/bc_cho/0_Project/2_LDCT2NDCT/dataset/nas69" 
    
    # Train / Valid 폴더 경로 설정
    if mode == 'train':
        mode_dir = os.path.join(base_dir, 'train')
    elif mode == 'valid':
        mode_dir = os.path.join(base_dir, 'valid')
    else:
        raise ValueError("mode must be 'train' or 'valid'")

    # 1. WBCT_Chest 데이터 (입력: B45f -> 정답: B30f)
    in_chest_full = list_sort_nicely(glob.glob(os.path.join(mode_dir, 'WBCT_Chest/Full/*/B45f/*.dcm')))
    gt_chest_full = list_sort_nicely(glob.glob(os.path.join(mode_dir, 'WBCT_Chest/Full/*/B30f/*.dcm')))

    in_chest_quarter = list_sort_nicely(glob.glob(os.path.join(mode_dir, 'WBCT_Chest/Quarter/*/B45f/*.dcm')))
    gt_chest_quarter = list_sort_nicely(glob.glob(os.path.join(mode_dir, 'WBCT_Chest/Quarter/*/B30f/*.dcm')))

    # 2. WBCT_Chest_add 데이터 (입력: Br49d -> 정답: Br36d)
    in_add_full = list_sort_nicely(glob.glob(os.path.join(mode_dir, 'WBCT_Chest_add/Full/*/Br49d/*.dcm')))
    gt_add_full = list_sort_nicely(glob.glob(os.path.join(mode_dir, 'WBCT_Chest_add/Full/*/Br36d/*.dcm')))

    in_add_quarter = list_sort_nicely(glob.glob(os.path.join(mode_dir, 'WBCT_Chest_add/Quarter/*/Br49d/*.dcm')))
    gt_add_quarter = list_sort_nicely(glob.glob(os.path.join(mode_dir, 'WBCT_Chest_add/Quarter/*/Br36d/*.dcm')))

    # 3. 모든 경로 합치기
    # 기존 코드와의 호환성을 위해 변수명은 n_20(입력), n_100(정답)을 유지합니다.
    n_20_imgs = in_chest_full + in_chest_quarter + in_add_full + in_add_quarter
    n_100_imgs = gt_chest_full + gt_chest_quarter + gt_add_full + gt_add_quarter

    # 디버깅용: 파일 개수가 서로 일치하는지 확인
    print(f"[{mode}] Input count: {len(n_20_imgs)}, Target count: {len(n_100_imgs)}")
    if len(n_20_imgs) != len(n_100_imgs):
        print("경고: 입력 이미지와 정답 이미지의 개수가 다릅니다! 폴더 내 파일 짝을 확인해주세요.")

    # 딕셔너리 형태로 묶기
    files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]
    
    # Mayo.py에 정의된 Data Augmentation & Normalization 가져오기
    transforms = get_transforms(mode=mode, type=type)

    # 데이터셋 반환
    if mode == 'train' and (type == 'full_patch' or type == 'window_patch'):
        return Dataset(data=files, transform=transforms), list_data_collate
    else:
        return Dataset(data=files, transform=transforms), default_collate_fn