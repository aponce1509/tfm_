from utils_res_ensemble import metric_study_ensemble
from utils_study_res import metric_study

def main_study(run_id, case_of_study, save=False, is_ensemble=False):
    if is_ensemble:
        metric_study_ensemble(run_id, case_of_study, save)
    else:
        metric_study(run_id, case_of_study, save)

if __name__ == "__main__":
    run_id = "6f992dbc4d894711ac9841e7b9d3ed2a"
    case_of_study = "train"
    is_ensemble = False
    
    main_study(run_id, case_of_study, False, is_ensemble)