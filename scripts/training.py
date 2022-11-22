from datetime import datetime, timedelta
import gc
import os

import dask
import numpy as np

from c4dllightning.features import batch, regions, transform
from c4dllightning.ml.models import models


def setup_batch_gen(file_dir, file_suffix="2020", primary="RZC",
    target="occurrence-8-10", batch_size=48, epoch=datetime(1970,1,1)):

    files = os.listdir(file_dir)
    files = [
        fn for fn in files if 
        fn.startswith("patches") and fn.endswith(file_suffix+".nc")
    ]
    files = {
        fn.split("_")[1]: os.path.join(file_dir,fn) for fn in files
    } # map variable name to file

    # raw data
    raw = {
        var_name: dask.delayed(regions.load_patches)(fn)
        for (var_name, fn) in files.items()
    }
    raw = dask.compute(raw, scheduler="processes")[0]

    raw_interp = ["CAPE-MU", "CIN-MU", "HZEROCL", "MCONV",
        "LCL-ML", "OMEGA", "SLI", "SOILTYP", "T-2M", "T-SO"]
    for var in raw_interp:
        if var in raw:
            raw[var]["interpolation"] = "linear"
            raw[var]["stride"] = 12

    static_vars = ["Altitude", "EW-deriv", "NS-deriv"]
    for var in static_vars:
        if var in raw:
            raw[var]["static"] = True

    # configure missing values (to use when data is missing)
    missing_values = {
        "CAPE-MU": 200.0,
        "CIN-MU": 21.0,
        "HZEROCL": 3300.0,
        "LCL-ML": 1000.0,
        "MCONV": 0.0,
        "OMEGA": 0.0,
        "SLI": 2.0,
        "SOILTYP": 5,
        "T-2M": 289.18,
        "T-SO": 289.63,
        "HRV": 38.9,
        "VIS006": 37.1,
        "VIS008": 57.0,
        "IR-016": 41.8,
        "IR-039": 274.2,
        "WV-062": 232.8,
        "WV-073": 247.3,
        "IR-087": 266.1,
        "IR-097": 247.5,
        "IR-108": 267.5,
        "IR-120": 266.1,
        "IR-134": 250.6,
        "ctth-tempe": 260.0,
        "ctth-alti": 5260.0,
        "cmic-phase": 4
    }
    for var in missing_values:
        raw[var]["missing_value"] = np.float32(missing_values[var])

    transform_CAPE = lambda: transform.normalize(std=200.0)
    transform_CIN = lambda: transform.normalize(std=21.0)
    transform_HZEROCL = lambda: transform.normalize_threshold(std=3300, 
        threshold=0.0, fill_value=0.0)
    transform_LCL = lambda: transform.normalize(std=1000.0)
    transform_MCONV = lambda: transform.normalize_threshold(std=3.8e-6, 
        threshold=-1.0, fill_value=0.0)
    transform_OMEGA = lambda: transform.normalize(std=4.2)
    transform_SLI = lambda: transform.normalize(std=3.5)
    transform_SOILTYP = lambda: transform.one_hot([1,3,4,5,6,7,9])
    transform_T = lambda: transform.normalize_threshold(
        mean=290.0, std=7.2, threshold=200, fill_value=290.0)
    transform_Altitude = lambda: transform.normalize(std=820.0)
    transform_deriv = lambda: transform.normalize(std=200.0)
    transform_HRV = lambda: transform.normalize(std=100.0, dtype=np.float16)
    transform_radiance = lambda: transform.normalize(std=100.0)
    transform_TB = lambda: transform.normalize(mean=250.0, std=10.0)

    # features and targets are defined by transforming the raw data
    transforms = {
        "RZC": {
            "source_vars": ["RZC"],
            "transform": transform.scale_log_norm(raw["RZC"]["scale"],
                threshold=0.1, fill_value=0.01, mean=-0.051, std=0.528,
                dtype=np.float16)
        },
        "CZC": {
            "source_vars": ["CZC"],
            "transform": transform.scale_norm(raw["CZC"]["scale"],
                threshold=5.0, fill_value=-5.0, mean=21.3, std=8.71,
                dtype=np.float16)
        },
        "EZC-20": {
            "source_vars": ["EZC-20"],
            "transform": transform.scale_norm(raw["EZC-20"]["scale"],
                std=1.97, dtype=np.float16)
        },
        "EZC-45": {
            "source_vars": ["EZC-45"],
            "transform": transform.scale_norm(raw["EZC-45"]["scale"],
                std=1.97, dtype=np.float16)
        },
        "HZC": {
            "source_vars": ["HZC"],
            "transform": transform.scale_norm(raw["HZC"]["scale"],
                std=1.97, dtype=np.float16)
        },
        "LZC": {
            "source_vars": ["LZC"],
            "transform": transform.scale_log_norm(raw["LZC"]["scale"],
                threshold=0.75, fill_value=0.5, mean=-0.274, std=0.135,
                dtype=np.float16)
        },
        "BZC": {
            "source_vars": ["BZC"],
            "transform": transform.scale_norm(raw["BZC"]["scale"],
                std=100.0, dtype=np.float16)
        },
        "AREA57": {
            "source_vars": ["AREA57"],
            "transform": transform.normalize(std=14.0, dtype=np.float16)
        },
        "occurrence-8-10-target": {
            "source_vars": ["occurrence-8-10"],
            "transform": transform.cast(np.uint8) #x.astype(np.float32),
        },
        "occurrence-8-10": {
            "source_vars": ["occurrence-8-10"],
            "transform": transform.cast(np.uint8)
        },
        "density": {
            "source_vars": ["density"],
            "transform": transform.scale_log_norm(raw["density"]["scale"],
               threshold=1e-3, fill_value=1e-4, mean=-0.593, std=0.640,
               dtype=np.float16)
        },
        "current": {
            "source_vars": ["current"],
            "transform": transform.scale_log_norm(raw["current"]["scale"],
                threshold=1e-7, fill_value=1e-8, mean=0.0718, std=0.731,
                dtype=np.float16)
        },
        "ctth-tempe": {
            "source_vars": ["ctth-tempe"],
            "transform": transform.scale_norm(raw["ctth-tempe"]["scale"],
                missing_value=65535, fill_value=330.0, mean=260.0, std=19.1)
        },
        "ctth-alti": {
            "source_vars": ["ctth-alti"],
            "transform": transform.scale_norm(raw["ctth-alti"]["scale"],
                missing_value=65535, fill_value=-1000, mean=5260.0, std=2810.0)
        },
        "cmic-phase": {
            "source_vars": ["cmic-phase"],
            "transform": transform.one_hot(values=[1,2,3,4,255])
        },
        "cmic-cot": {
            "source_vars": ["cmic-cot"],
            "transform": transform.scale_log_norm(raw["cmic-cot"]["scale"],
                missing_value=65535, fill_value=0.1, mean=0.94, std=0.588)
        },
        "HRV": {
            "source_vars": ["HRV"],
            "transform": transform_HRV()
        },
        "VIS006": {
            "source_vars": ["VIS006"],
            "transform": transform_radiance()
        },
        "VIS008": {
            "source_vars": ["VIS008"],
            "transform": transform_radiance()
        },
        "IR-016": {
            "source_vars": ["IR-016"],
            "transform": transform_radiance()
        },
        "IR-039": {
            "source_vars": ["IR-039"],
            "transform": transform.normalize(mean=274, std=17.5)
        },
        "WV-062": {
            "source_vars": ["WV-062"],
            "transform": transform_TB()
        },
        "WV-073": {
            "source_vars": ["WV-073"],
            "transform": transform_TB()
        },
        "IR-087": {
            "source_vars": ["IR-087"],
            "transform": transform_TB()
        },
        "IR-097": {
            "source_vars": ["IR-097"],
            "transform": transform_TB()
        },
        "IR-108": {
            "source_vars": ["IR-108"],
            "transform": transform_TB()
        },
        "IR-120": {
            "source_vars": ["IR-120"],
            "transform": transform_TB()
        },
        "IR-134": {
            "source_vars": ["IR-134"],
            "transform": transform_TB()
        },
        "CAPE-MU": {
            "source_vars": ["CAPE-MU"],
            "transform": transform_CAPE(),
        },
        "CAPE-MU-future": {
            "source_vars": ["CAPE-MU"],
            "transform": transform_CAPE(),
            "timeframe": "future"
        },
        "CIN-MU": {
            "source_vars": ["CIN-MU"],
            "transform": transform_CIN()
        },
        "CIN-MU-future": {
            "source_vars": ["CIN-MU"],
            "transform": transform_CIN(),
            "timeframe": "future"
        },
        "HZEROCL": {
            "source_vars": ["HZEROCL"],
            "transform": transform_HZEROCL()
        },
        "HZEROCL-future": {
            "source_vars": ["HZEROCL"],
            "transform": transform_HZEROCL(),
            "timeframe": "future"
        },
        "LCL-ML": {
            "source_vars": ["LCL-ML"],
            "transform": transform_LCL()
        },
        "LCL-ML-future": {
            "source_vars": ["LCL-ML"],
            "transform": transform_LCL(),
            "timeframe": "future"
        },
        "MCONV": {
            "source_vars": ["MCONV"],
            "transform": transform_MCONV()
        },
        "MCONV-future": {
            "source_vars": ["MCONV"],
            "transform": transform_MCONV(),
            "timeframe": "future"
        },
        "OMEGA": {
            "source_vars": ["OMEGA"],
            "transform": transform_OMEGA()
        },
        "OMEGA-future": {
            "source_vars": ["OMEGA"],
            "transform": transform_OMEGA(),
            "timeframe": "future"
        },
        "SLI": {
            "source_vars": ["SLI"],
            "transform": transform_SLI()
        },
        "SLI-future": {
            "source_vars": ["SLI"],
            "transform": transform_SLI(),
            "timeframe": "future"
        },
        "SOILTYP": {
            "source_vars": ["SOILTYP"],
            "transform": transform_SOILTYP(),
            "timeframe": "static"
        },
        "T-2M": {
            "source_vars": ["T-2M"],
            "transform": transform_T()
        },
        "T-2M-future": {
            "source_vars": ["T-2M"],
            "transform": transform_T(),
            "timeframe": "future"
        },
        "T-SO": {
            "source_vars": ["T-SO"],
            "transform": transform_T()
        },
        "T-SO-future": {
            "source_vars": ["T-SO"],
            "transform": transform_T(),
            "timeframe": "future"
        },
        "Altitude": {
            "source_vars": ["Altitude"],
            "transform": transform_Altitude(),
            "timeframe": "static"
        },
        "EW-deriv": {
            "source_vars": ["EW-deriv"],
            "transform": transform_deriv(),
            "timeframe": "static"
        },
        "NS-deriv": {
            "source_vars": ["NS-deriv"],
            "transform": transform_deriv(),
            "timeframe": "static"
        },
        "sun-z": {
            "source_vars": ["sun-z"],
            "transform": transform.normalize(std=127.0)
        },
    }

    # predictors
    pred_names = [
        "RZC", "CZC",
        "EZC-20", "EZC-45",
        "HZC", "LZC",
        "density", "current",
        "ctth-tempe", "ctth-alti", "cmic-phase", "cmic-cot",
        "HRV", "VIS006", "VIS008", "IR-016",
        "IR-016", "IR-039", "WV-062", "WV-073",
        "IR-087", "IR-097", "IR-108", "IR-120", "IR-134",
        "CAPE-MU-future", "CIN-MU-future",
        "HZEROCL-future", "LCL-ML-future",
        "MCONV-future", "OMEGA-future",
        "SLI-future", "SOILTYP",
        "T-2M-future", "T-SO-future",
        "Altitude", "EW-deriv", "NS-deriv",
        "sun-z"
    ]
    pred_names.append(target)

    predictors = {
        var_name: transforms[var_name]
        for var_name in pred_names
    }

    # targets
    target_names = [target+"-target"]
    targets = {var_name: transforms[var_name] for var_name in target_names}

    # we need one "primary" raw data variable
    # that determines the location of the data for all variables
    primary_patch_data = raw[primary]    
    (box_locs, t0) = regions.box_locations(
        primary_patch_data["patch_coords"],
        primary_patch_data["patch_times"],
        primary_patch_data["zero_patch_coords"],
        primary_patch_data["zero_patch_times"]
    )

    batch_gen = batch.BatchGenerator(predictors, targets, raw, box_locs,
        primary, valid_frac=0.1, test_frac=0.1, batch_size=batch_size,
        timesteps=(6,12), random_seed=1234)

    gc.collect()

    return batch_gen


def build_ensemble_model(batch_gen, dropout=True):
    def create_model(init_strategy=False):
        return models.init_model(batch_gen, 
            init_strategy=init_strategy,
            compile=False
        )
        
    (model1,strategy) = create_model(init_strategy=True)
    with strategy.scope():
        (model2, _) = create_model()
        (model3, _) = create_model()

    ind_models = [model1, model2, model3]
    if dropout:
        weight_files = [
            "../models/lightning_dropout_weightdecay_noclassweight.h5",
            "../models/lightning_dropout_weightdecay_noclassweight2.h5",
            "../models/lightning_dropout_weightdecay_noclassweight3.h5",
        ]
    else:
        weight_files = [
            "../models/lightning_noclassweight1.h5",
            "../models/lightning_noclassweight2.h5",
            "../models/lightning_noclassweight3.h5",
        ]

    for (m,w) in zip(ind_models, weight_files):
        m.load_weights(w)
    
    with strategy.scope():
        ens_model = models.ensemble_model(ind_models)
        models.compile_model(ens_model, event_occurrence=0.5, optimizer='sgd')

    return (ens_model, strategy)


def build_persistence_model(batch_gen):
    return models.init_model(batch_gen, 
        model_func=models.persistence_model)
