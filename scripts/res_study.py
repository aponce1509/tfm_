from utils_res_ensemble import metric_study_ensemble
from utils_study_res import metric_study

def main_study(run_id, case_of_study, transform=None, save=False,
               is_ensemble=False):
    if is_ensemble:
        metric_study_ensemble(run_id, case_of_study, transform, save)
    else:
        metric_study(run_id, transform, case_of_study, save)

if __name__ == "__main__":
    import torchvision.transforms as T
    # run_id = "1e4f392a00b3462e9109b68a5493b97f"

    run_id = "1f7b43573e0a4556945381b2e4933c07"  # mejor y
    # run_id = "0b49acce2a024bb09056b292126ad4d5"  # mejor z
    # run_id = "1e4f392a00b3462e9109b68a5493b97f"
    # run_id = "0f3e299973b54be7aa3c5dcae2fc496d"
    case_of_study = "test"
    is_ensemble = False
    transform = T.Compose([
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # transform = [T.Compose([
    #     T.Lambda(lambda x: x.repeat(3, 1, 1)),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ]), ] * 3
        
    main_study(run_id, case_of_study, transform, False, is_ensemble)