import numpy as np
import pandas as pd

if __name__ == '__main__':

    # IMPORTING WHOLE EXCEL FILE AS DATAFRAME
    sens_df = pd.read_excel(
        f"case_1_gPROMS_sens/sensitivities_manual.xlsx",
        sheet_name=None,
        # f"Experiment {exp}",
    )
    sens_df = pd.concat(sens_df, ignore_index=False, sort=True).reset_index()
    n_spt_max = sens_df["level_1"].max() + 1

    # SAVING SENSITIVITIES
    sens_matrix = np.full((10, n_spt_max, 1, 3), np.nan)
    for i, exp in enumerate(sens_df["level_0"].unique()):
        n_spt = sens_df[sens_df["level_0"] == exp].shape[0]
        sens_matrix[i, :n_spt, :, :] = sens_df.loc[:, ["Sens. W.r.t. a0", "Sens. W.r.t. CT", "Sens w.r.t. Ss"]][sens_df["level_0"] == exp].values[:, None, :]
        print(sens_matrix)

    # SAVING THE PREDICTED RESPONSES
    response_matrix = np.full((10, n_spt_max, 1), np.nan)
    for i, exp in enumerate(sens_df["level_0"].unique()):
        spt = sens_df[sens_df["level_0"] == exp]["Time"]
        n_spt = spt.shape[0]
        response_matrix[i, :n_spt, :] = sens_df[sens_df["level_0"] == exp]["Conc. At end"][:, None]

    # SAVING THE VARIABLE SAMPLING TIMES
    spt_candidates = np.empty((10, n_spt_max))
    # spt_candidates = spt_candidates.astype(str)
    for i, exp in enumerate(sens_df["level_0"].unique()):
        spt = sens_df[sens_df["level_0"] == exp]["Time"]
        n_spt = spt.shape[0]
        spt_candidates[i, :n_spt] = sens_df[sens_df["level_0"] == exp]["Time"]
    print(spt_candidates)

    # SAVING WHAT HAS BEEN IMPORTED INTO PICKLE
    import pickle
    with open("case_1_result/analytical_sensitivity_case1.pkl", "wb") as file:
        pickle.dump(sens_matrix, file)
    with open("case_1_result/variable_sampling_times.pkl", "wb") as file:
        pickle.dump(spt_candidates, file)
    with open("case_1_result/response_matrix.pkl", "wb") as file:
        pickle.dump(response_matrix, file)
