# pip install residual_utils

import pandas as pd
import xarray as xr
import gcsfs
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import datetime
import csv
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from . import residual_utils as supporting_functions
# import residual_utils as supporting_functions
# import ./residual_utils.py as supporting_functions


def apply_socat_mask_inplace(df, mask_dataset) -> None:
    """
    Fast in-place assignment of 'socat_mask' from an xarray.Dataset to a DataFrame with MultiIndex (time, ylat, xlon).

    Parameters:
        df (pd.DataFrame): DataFrame with MultiIndex (time, ylat, xlon)
        mask_dataset (xr.Dataset): xarray dataset containing 'socat_mask' with dims (time, ylat, xlon)

    Returns:
        None (modifies df in-place)
    """
    time = df.index.get_level_values("time").values
    ylat = df.index.get_level_values("ylat").astype(float).values
    xlon = df.index.get_level_values("xlon").astype(float).values

    mask_da = mask_dataset["socat_mask"]
    mask_vals = mask_da.sel(
        time=xr.DataArray(time, dims="points"),
        ylat=xr.DataArray(ylat, dims="points"),
        xlon=xr.DataArray(xlon, dims="points"),
        method="nearest",
    ).values

    if hasattr(mask_vals, "compute"):
        mask_vals = mask_vals.compute()

    df["socat_mask"] = np.nan_to_num(mask_vals, nan=0).astype(int)


###


def save_ngb_model_locally(model, dates, mask_name, ens, member, output_dir):
    """
    Saves the trained XGBoost model to a local directory.

    Parameters
    ----------
    model : xgboost.sklearn.XGBRegressor
        Trained XGBoost model.

    dates : pandas.DatetimeIndex
        List of dataset dates.

    mask_name : str
        Name of the socat mask used for training.

    ens : str
        Earth System Model name.

    member : str
        Member index (e.g., 'member_r1i1p1f1').
    username : str
        Username of the person running the code. Reviewer should also change this to their own username.
    """

    print("Starting local model saving process...")

    # Ensure the output directory exists
    Path("output/model_saved").mkdir(parents=True, exist_ok=True)

    # Format time information
    init_date = f"{dates[0].year}{dates[0].month:02d}"
    fin_date = f"{dates[-1].year}{dates[-1].month:02d}"

    # Define the local filename
    model_filename = f"{mask_name}_model_pCO2_2D_{ens}_{member.split('_')[-1]}_mon_1x1_{init_date}_{fin_date}.p"
    local_output_dir = "output/model_saved"
    model_path = Path(local_output_dir) / model_filename

    gcsfs_path = f"{output_dir}/models/{model_filename}"
    fs = gcsfs.GCSFileSystem()

    # Save the model
    try:
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        print(f"Model successfully saved locally at: {model_path}")

        with fs.open(gcsfs_path, "wb") as gcs_file:
            pickle.dump(model, gcs_file)
        print(f"Model successfully saved to GCS at: {gcsfs_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("Local model saving process complete.")


###


def run_ngboost_with_masks(
    mask_data_dict,
    selected_mems_dict,
    features_sel,
    target_sel,
    year_mon,
    test_year_mon,
    path_seeds,
    MLinputs_path,
    dates,
    init_date,
    fin_date,
    output_dir,
    params,
    runthiscell=-1,
):
    fs = gcsfs.GCSFileSystem()
    if runthiscell:
        random_seeds = np.load(fs.open(path_seeds))
    seed_loc_dict = defaultdict(dict)
    for ens, mem_list in selected_mems_dict.items():
        sub_dictt = {mem: no for no, mem in enumerate(mem_list)}
        seed_loc_dict[ens] = sub_dictt

    val_prop = 0.2
    test_prop = 0.0

    print(datetime.datetime.now())

    if runthiscell == -1:
        print(
            "Reviewing process: Running ML only for the first member of the first ESM. \n"
            "Running ML only for the first mask."
        )
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]
        run_selected_mems_dict = {first_ens: [first_mem]}
        first_mask = list(mask_data_dict.keys())[0]
        run_selected_mask_data_dict = {first_mask: mask_data_dict[first_mask]}
    else:
        print(
            "Running ML for all members of all ESMs and all masks. \n"
            "This may take a while."
        )
        run_selected_mems_dict = selected_mems_dict
        run_selected_mask_data_dict = mask_data_dict

    for mask_name, mask_data in run_selected_mask_data_dict.items():
        print(f"\nRunning NGBoost for mask: {mask_name}\n")

        for ens, mem_list in run_selected_mems_dict.items():
            for member in mem_list:
                print(ens, member)
                seed_loc = seed_loc_dict[ens][member]
                data_dir = f"{MLinputs_path}/{ens}/{member}"
                fname = f"MLinput_{ens}_{member.split('_')[-1]}_mon_1x1_{init_date}_{fin_date}.pkl"
                file_path = f"{data_dir}/{fname}"

                with fs.open(file_path, "rb") as filee:
                    df = pd.read_pickle(filee)
                    df["year"] = df.index.get_level_values("time").year
                    df["mon"] = df.index.get_level_values("time").month
                    df["year_month"] = (
                        df["year"].astype(str) + "-" + df["mon"].astype(str)
                    )

                    # print df['socat+mask'] stats before applying new mask
                    print(df["socat_mask"].describe())

                    # Replace socat_mask with the new mask
                    apply_socat_mask_inplace(df, mask_data)
                    print("Applied new SOCAT mask")
                    # print df['socat+mask'] stats after applying new mask
                    print(df["socat_mask"].describe())

                    recon_sel = (
                        ~df[features_sel + target_sel + ["net_mask"]].isna().any(axis=1)
                    ) & (
                        (df[target_sel] < 250) & (df[target_sel] > -250)
                    ).to_numpy().ravel()

                    sel = recon_sel & (df["socat_mask"] == 1)

                    train_sel = (
                        (sel & (pd.Series(df["year_month"]).isin(year_mon)))
                        .to_numpy()
                        .ravel()
                    )
                    test_sel = (
                        (sel & (pd.Series(df["year_month"]).isin(test_year_mon)))
                        .to_numpy()
                        .ravel()
                    )

                    X = df.loc[sel, features_sel].to_numpy()
                    y = df.loc[sel, target_sel].to_numpy().ravel()
                    Xtrain = df.loc[train_sel, features_sel].to_numpy()
                    ytrain = df.loc[train_sel, target_sel].to_numpy().ravel()
                    X_test = df.loc[test_sel, features_sel].to_numpy()
                    y_test = df.loc[test_sel, target_sel].to_numpy().ravel()

                    N = Xtrain.shape[0]
                    train_val_idx, train_idx, val_idx, test_idx = (
                        supporting_functions.train_val_test_split(
                            N, test_prop, val_prop, random_seeds, seed_loc
                        )
                    )

                    (
                        X_train_val,
                        X_train,
                        X_val,
                        X_test_tmp,
                        y_train_val,
                        y_train,
                        y_val,
                        y_test_tmp,
                    ) = supporting_functions.apply_splits(
                        Xtrain, ytrain, train_val_idx, train_idx, val_idx, test_idx
                    )

                model = NGBRegressor(
                    Dist=Normal, random_state=random_seeds[5, seed_loc], **params
                )
                model.fit(X_train_val, y_train_val, X_val=X_val, Y_val=y_val)

                # save ngboost model
                save_ngb_model_locally(model, dates, mask_name, ens, member, output_dir)

                y_pred_test = model.predict(X_test)
                pred_dist = model.pred_dist(X_test)

                pred_dist_dict = {
                    "mean_preds": pred_dist.loc,
                    "std_preds": pred_dist.scale,
                }

                test_performance = supporting_functions.evaluate_test(
                    y_test, y_pred_test
                )

                # Save prediction distribution
                rows = [
                    {
                        "ensemble": ens,
                        "member": member,
                        "index": i,
                        "mean_pred": pred_dist_dict["mean_preds"][i],
                        "std_pred": pred_dist_dict["std_preds"][i],
                        "mask": mask_name,
                    }
                    for i in range(len(pred_dist_dict["mean_preds"]))
                ]

                df_out = pd.DataFrame(rows)
                test_dist_fname = f"{output_dir}/metrics/ngb_test_dists_{mask_name}_{init_date}-{fin_date}.csv"
                print("Saving test distribution to", test_dist_fname)
                df_out.to_csv(test_dist_fname, index=False)

                # Save evaluation metrics
                test_row_dict = {"model": ens, "member": member, "mask": mask_name}
                test_row_dict.update(test_performance)

                test_perform_fname = f"{output_dir}/metrics/ngb_test_performance_{mask_name}_{init_date}-{fin_date}.csv"  # path for test performance metrics
                file_exists = fs.exists(test_perform_fname)
                with fs.open(test_perform_fname, "a") as f_object:
                    writer = csv.DictWriter(f_object, fieldnames=test_row_dict.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(test_row_dict)

                print(f"Test metrics for {mask_name}:", test_performance)

        print(f"Done with mask: {mask_name} at", datetime.datetime.now())


###

fs = gcsfs.GCSFileSystem()


def run_reconstruction_with_masks(
    mask_data_dict,
    selected_mems_dict,
    features_sel,
    target_sel,
    year_mon,
    test_year_mon,
    seed_loc_dict,
    MLinputs_path,
    init_date,
    fin_date,
    dates,
    output_dir,
    runthiscell=1,
    load_model_cloud=True,
):
    if runthiscell == -1:
        print(
            "Reviewing process: Running reconstruction only for the first member of the first ESM. \n"
            "Running reconstruction only for the first mask."
        )
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]
        run_selected_mems_dict = {first_ens: [first_mem]}
        first_mask = list(mask_data_dict.keys())[0]
        run_selected_mask_data_dict = {first_mask: mask_data_dict[first_mask]}
    else:
        print(
            "Running reconstruction for all members of all ESMs and all masks. \n"
            "This may take a while."
        )
        run_selected_mems_dict = selected_mems_dict
        run_selected_mask_data_dict = mask_data_dict

    # Save summary of mask coverage
    summary_rows = []
    for mask_name, mask_data in run_selected_mask_data_dict.items():
        print(f"\nRunning reconstruction for mask: {mask_name}\n")

        for ens, mem_list in run_selected_mems_dict.items():
            for member in mem_list:
                print(ens, member)
                seed_loc = seed_loc_dict[ens][member]
                data_dir = f"{MLinputs_path}/{ens}/{member}"
                fname = f"MLinput_{ens}_{member.split('_')[-1]}_mon_1x1_{init_date}_{fin_date}.pkl"
                file_path = f"{data_dir}/{fname}"

                # Load the model
                model_filename = f"{mask_name}_model_pCO2_2D_{ens}_{member.split('_')[-1]}_mon_1x1_{init_date}_{fin_date}.p"
                if load_model_cloud:
                    model_path = f"{output_dir}/models/{model_filename}"
                    with fs.open(model_path, "rb") as f:
                        model = pickle.load(f)
                    print(f"Model loaded from GCS at: {model_path}")

                else:
                    local_output_dir = "output/model_saved"
                    model_path = Path(local_output_dir) / model_filename

                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    print(f"Model loaded from local path: {model_path}")

                with fs.open(file_path, "rb") as filee:
                    df = pd.read_pickle(filee)
                    df["year"] = df.index.get_level_values("time").year
                    df["mon"] = df.index.get_level_values("time").month
                    df["year_month"] = (
                        df["year"].astype(str) + "-" + df["mon"].astype(str)
                    )

                    # print df['socat_mask'] stats before applying new mask
                    print(df["socat_mask"].describe())

                    # Replace socat_mask with the new mask
                    apply_socat_mask_inplace(df, mask_data)
                    print("Applied new SOCAT mask")

                    # print df['socat_mask'] stats after applying new mask
                    print(df["socat_mask"].describe())

                    recon_sel = (
                        ~df[features_sel + target_sel + ["net_mask"]].isna().any(axis=1)
                    ) & (
                        (df[target_sel] < 250) & (df[target_sel] > -250)
                    ).to_numpy().ravel()

                    sel = recon_sel & (df["socat_mask"] == 1)
                    train_sel = (
                        (sel & (pd.Series(df["year_month"]).isin(year_mon)))
                        .to_numpy()
                        .ravel()
                    )
                    test_sel = (
                        (sel & (pd.Series(df["year_month"]).isin(test_year_mon)))
                        .to_numpy()
                        .ravel()
                    )
                    unseen_sel = recon_sel & (df["socat_mask"] == 0)

                    print(
                        f"Total selected locations for training ({mask_name}):",
                        sel.sum(),
                    )
                    print(f"Total unseen locations ({mask_name}):", unseen_sel.sum())

                    X = df.loc[sel, features_sel].to_numpy()
                    y = df.loc[sel, target_sel].to_numpy().ravel()

                y_pred_unseen = model.predict(
                    df.loc[unseen_sel, features_sel].to_numpy()
                )
                y_dists_unseen = model.pred_dist(
                    df.loc[unseen_sel, features_sel].to_numpy()
                )
                y_unseen = df.loc[unseen_sel, target_sel].to_numpy().ravel()

                unseen_performance = defaultdict(dict)
                unseen_performance[ens][member] = supporting_functions.evaluate_test(
                    y_unseen, y_pred_unseen
                )

                fields = unseen_performance[ens][member].keys()
                unseen_row_dict = {"model": ens, "member": member, "mask": mask_name}
                for field in fields:
                    unseen_row_dict[field] = unseen_performance[ens][member][field]

                unseen_perform_fname = f"{output_dir}/metrics/ngb_unseen_performance_{mask_name}_{init_date}-{fin_date}.csv"  # path for unseen performance metrics

                file_exists = fs.exists(unseen_perform_fname)
                with fs.open(unseen_perform_fname, "a") as f_object:
                    writer = csv.DictWriter(f_object, fieldnames=unseen_row_dict.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(unseen_row_dict)

                print(
                    f"unseen performance metrics for {mask_name}:",
                    unseen_performance[ens][member],
                )

                y_pred_seen = model.predict(X)
                y_dists_seen = model.pred_dist(X)

                df["pCO2_recon_full"] = np.nan
                df.loc[unseen_sel, "pCO2_recon_full"] = y_pred_unseen
                df.loc[unseen_sel, "pCO2_recon_full_mean"] = y_dists_unseen.loc
                df.loc[unseen_sel, "pCO2_recon_full_std"] = y_dists_unseen.scale
                df.loc[sel, "pCO2_recon_full"] = y_pred_seen
                df.loc[sel, "pCO2_recon_full_mean"] = y_dists_seen.loc
                df.loc[sel, "pCO2_recon_full_std"] = y_dists_seen.scale

                df["pCO2_recon_unseen"] = np.nan
                df.loc[unseen_sel, "pCO2_recon_unseen"] = y_pred_unseen
                df.loc[unseen_sel, "pCO2_recon_unseen_mean"] = y_dists_unseen.loc
                df.loc[unseen_sel, "pCO2_recon_unseen_std"] = y_dists_unseen.scale
                df.loc[sel, "pCO2_recon_unseen"] = np.nan

                df["pCO2_truth"] = df.loc[:, target_sel]

                DS_recon = df[
                    [
                        "net_mask",
                        "socat_mask",
                        "pCO2_recon_full",
                        "pCO2_recon_unseen",
                        "pCO2_truth",
                        "pCO2_recon_full_mean",
                        "pCO2_recon_full_std",
                        "pCO2_recon_unseen_mean",
                        "pCO2_recon_unseen_std",
                    ]
                ].to_xarray()
                recon_output_dir = f"{output_dir}/reconstructions/{mask_name}"
                supporting_functions.save_recon(
                    DS_recon, dates, recon_output_dir, ens, member
                )

                summary_rows.append(
                    {
                        "mask": mask_name,
                        "member": member,
                        "seen_count": int(sel.sum()),
                        "unseen_count": int(unseen_sel.sum()),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(
        f"{output_dir}/metrics/mask_coverage_summary.csv", index=False
    )

    print("end of all members", datetime.datetime.now())


###


def calc_recon_pco2(
    regridded_members_dir,
    pco2_recon_dir,
    selected_mems_dict,
    mask_name,
    init_date,
    fin_date,
    owner_name=None,
):
    """
    Calculates reconstructed pco2 per member.

    Parameters
    ----------
    regridded_members_dir : str
        Path to regridded data from notebook 00, which contains pco2T.

    pco2_recon_dir : str
        Path to directory where ML reconstructions from notebook 02 are saved.
    """
    fs = gcsfs.GCSFileSystem()
    init_date_sel = pd.to_datetime(init_date, format="%Y%m")
    fin_date_sel = pd.to_datetime(fin_date, format="%Y%m")

    if owner_name:
        print(
            "Reviewing process: Running ML only for the first member of the first ESM, loading remaining reconstructed data from the notebook owner."
        )
        first_ens = list(selected_mems_dict.keys())[0]  # Get the first ensemble key
        first_mem = selected_mems_dict[first_ens][
            0
        ]  # Get the first member in that ensemble
        run_selected_mems_dict = {
            first_ens: [first_mem]
        }  # Create a dictionary with only the first ensemble and member

        grid_search_approach = "nmse"
        owener_output_dir = f"gs://leap-persistent/{owner_name}/{owner_name}/pco2_residual/{grid_search_approach}/post02_xgb"  # where to save machine learning results
        owener_recon_output_dir = f"{owener_output_dir}/reconstructions/{mask_name}"  # where owner save ML reconstructions

    else:
        run_selected_mems_dict = selected_mems_dict

    for ens, mem_list in run_selected_mems_dict.items():
        print(f"Current ESM: {ens}")

        for member in mem_list:
            print(f"On member {member}")

            ### File paths ###

            ### Path to regridded data from notebook 00, so we can get the pCO2-T we calculated in 00
            ### pCO2-T calculated from model pCO2 and SST
            pco2T_path = f"{regridded_members_dir}/{ens}/{member}/{ens}.{member.split('_')[-1]}.Omon.zarr"
            print("pco2T path:", pco2T_path)

            ### Path to reconstruction (ML output from notebook 02), where pCO2-residual was reconstructed
            pCO2R_path = f"{pco2_recon_dir}/{ens}/{member}/recon_pCO2residual_{ens}_{member}_mon_1x1_{init_date}_{fin_date}.zarr"
            print("pCO2R path:", pCO2R_path)

            ### Path to save calculated pCO2 (reconstructed pCO2-residual PLUS pCO2-T: Total pCO2 =  pCO2-residual + pCO2-T)
            file_out = f"{pco2_recon_dir}/{ens}/{member}/recon_pCO2_{ens}_{member}_mon_1x1_{init_date}_{fin_date}.zarr"  # change this to just pco2
            print("save path:", file_out)

            ### Loading pCO2-T and reconstructed pCO2-residual:
            pco2T_series = (
                xr.open_zarr(pco2T_path)
                .pco2_T.transpose("time", "ylat", "xlon")
                .sel(time=slice(init_date_sel, fin_date_sel))
            )
            pco2_ml_output = xr.open_zarr(
                pCO2R_path
            )  # , consolidated=False, storage_options={"token": "cloud"}, group=None)

            ### unseen reconstructed pCO2-Residual from XGB
            pCO2R_unseen_series = pco2_ml_output.pCO2_recon_unseen.transpose(
                "time", "ylat", "xlon"
            )

            pCO2R_unseen_means_series = pco2_ml_output.pCO2_recon_unseen_mean.transpose(
                "time", "ylat", "xlon"
            )
            pCO2R_unseen_stds_series = pco2_ml_output.pCO2_recon_unseen_std.transpose(
                "time", "ylat", "xlon"
            )

            ### Full (seen and unseen) reconstructed pCO2-Residual from XGB
            pCO2R_full_series = pco2_ml_output.pCO2_recon_full.transpose(
                "time", "ylat", "xlon"
            )

            pCO2_full_means_series = pco2_ml_output.pCO2_recon_full_mean.transpose(
                "time", "ylat", "xlon"
            )

            pCO2_full_stds_series = pco2_ml_output.pCO2_recon_full_std.transpose(
                "time", "ylat", "xlon"
            )

            # ### training set for pco2 residual
            # pCO2R_train_series = pco2_ml_output.pCO2_recon_train.transpose("time","ylat","xlon")

            # ### testing set for pco2 residual
            # pCO2R_test_series = pco2_ml_output.pCO2_recon_test.transpose("time","ylat","xlon")

            pCO2R_truth = pco2_ml_output.pCO2_truth.transpose("time", "ylat", "xlon")

            ### Get time coordinate correct
            pco2T_series = pco2T_series.assign_coords(
                {"time": ("time", pCO2R_unseen_series.time.data)}
            )

            ### Total pCO2 =  pCO2-residual + pCO2-T
            pco2_unseen = pco2T_series + pCO2R_unseen_series
            pco2_unseen_means = pco2T_series + pCO2R_unseen_means_series
            pco2_unseen_stds = pco2T_series + pCO2R_unseen_stds_series
            pco2_full = pco2T_series + pCO2R_full_series
            pco2_full_means = pco2T_series + pCO2_full_means_series
            pco2_full_stds = pCO2_full_stds_series
            # pco2_full_stds = pco2T_series + pCO2_full_stds_series
            # pco2_train =  pco2T_series + pCO2R_train_series
            # pco2_test =  pco2T_series + pCO2R_test_series
            pco2_truth = pco2T_series + pCO2R_truth

            ### Creating xarray of pco2 ML output, but with temperature added back
            comp = xr.Dataset(
                {
                    "pCO2_recon_unseen": (["time", "ylat", "xlon"], pco2_unseen.data),
                    "pCO2_recon_full": (["time", "ylat", "xlon"], pco2_full.data),
                    "pCO2_recon_unseen_mean": (
                        ["time", "ylat", "xlon"],
                        pco2_unseen_means.data,
                    ),
                    "pCO2_recon_unseen_std": (
                        ["time", "ylat", "xlon"],
                        pco2_unseen_stds.data,
                    ),
                    "pCO2_recon_full_mean": (
                        ["time", "ylat", "xlon"],
                        pco2_full_means.data,
                    ),
                    "pCO2_recon_full_std": (
                        ["time", "ylat", "xlon"],
                        pco2_full_stds.data,
                    ),
                    # 'pCO2_recon_train': (["time","ylat","xlon"], pco2_train.data),
                    # 'pCO2_recon_train':(["time","ylat","xlon"],pco2_train.data),
                    # 'pCO2_recon_test':(["time","ylat","xlon"],pco2_test.data),
                    "pCO2_truth": (["time", "ylat", "xlon"], pco2_truth.data),
                },
                coords={
                    "time": (["time"], pco2T_series.time.values),
                    "xlon": (["xlon"], pco2T_series.xlon.values),
                    "ylat": (["ylat"], pco2T_series.ylat.values),
                },
            )

            ### to overwrite file if it exists already
            if fs.exists(file_out):
                fs.rm(file_out, recursive=True)

            ### for saving:
            comp = comp.chunk({"time": 100, "ylat": 45, "xlon": 90})
            comp.to_zarr(file_out, mode="w", zarr_format=2)

            print(f"finished with {member}")

    if owner_name:
        print("Copying remaining members from owner’s directory...")
        for ens, mem_list in selected_mems_dict.items():
            print(f"On member {member}")
            if ens in run_selected_mems_dict:
                remaining_members = [
                    m for m in mem_list if m not in run_selected_mems_dict[ens]
                ]
            else:
                remaining_members = mem_list

            for member in remaining_members:
                owner_file_out = f"{owener_recon_output_dir}/{ens}/{member}/recon_pCO2_{ens}_{member}_mon_1x1_{init_date}_{fin_date}.zarr"
                target_file_out = f"{pco2_recon_dir}/{ens}/{member}/recon_pCO2_{ens}_{member}_mon_1x1_{init_date}_{fin_date}.zarr"

                if fs.exists(owner_file_out):
                    print(f"Copying {owner_file_out} → {target_file_out}")
                    fs.copy(owner_file_out, target_file_out)
                else:
                    print(f"Warning: {owner_file_out} not found. Skipping.")
                print(f"finished with {member}")
