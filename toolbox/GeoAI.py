from typing import Optional, Dict, List, Any
import yaml
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, mean_squared_error
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
import pickle
from imblearn.under_sampling import RandomUnderSampler
import json
# from fastmcp import FastMCP


cfg_path = Path(__file__).parent.parent / "config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
OUTPUT_DIR = cfg["output_dir"]

def _get_out_path(name: str, base_dir: str = OUTPUT_DIR) -> str:
    p = os.path.join(base_dir, name)
    if os.path.exists(p):
        raise FileExistsError(f"Output file already exists: {p}")
    return p

# mcp = FastMCP()

# @mcp.tool()
def train_random_forest_model(
    X_train_path: str,
    output_name: str,
    y_train_path: Optional[str] = None,
    label_column: str = "label",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a Random Forest Classifier.
    
    Args:
        X_train_path: Path to training features (CSV or NPY).
        output_name: Prefix for output filenames.
        y_train_path: Path to training labels (CSV or NPY). Optional if labels are in X_train_path CSV.
        label_column: Name of the label column in CSV (default: "label").
        n_estimators: Number of trees in the forest (default: 100).
        max_depth: Maximum depth of the tree (default: None).
        random_state: Random seed for reproducibility (default: 42).
        
    Returns:
        model_path: Path to the saved model file.
        feature_importances_path: Path to the feature importances CSV (if available).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    feature_names = None
    if X_train_path.lower().endswith(".csv"):
        df_train = pd.read_csv(X_train_path)
        if label_column in df_train.columns and (y_train_path is None or not os.path.isfile(y_train_path)):
            y_train = df_train[label_column].values
            X_train = df_train.drop(columns=[label_column]).values
            feature_names = list(df_train.drop(columns=[label_column]).columns)
        else:
            X_train = df_train.values
            if y_train_path is None:
                raise ValueError("y_train_path must be provided when label_column not in training CSV")
            y_series = pd.read_csv(y_train_path)
            if label_column in y_series.columns:
                y_train = y_series[label_column].values
            else:
                # fallback: first column
                y_train = y_series.iloc[:, 0].values
            feature_names = list(df_train.columns)
    else:
        X_train = np.load(X_train_path)
        if y_train_path is None:
            raise ValueError("y_train_path must be provided for NPY inputs")
        y_train = np.load(y_train_path)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    
    model_path = _get_out_path(f"{output_name}.joblib")
    joblib.dump(model, model_path)
    result: Dict[str, Any] = {
        "model_path": model_path
    }

    if feature_names:
        try:
            importances = model.feature_importances_
            fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi_csv = _get_out_path(f"{output_name}_feature_importances.csv")
            fi_df.to_csv(fi_csv, index=False)
            result["feature_importances_path"] = fi_csv
        except Exception:
            pass
    return result

# @mcp.tool()
def tune_model_hyperparameters(
    X_train_path: str,
    output_name: str,
    y_train_path: Optional[str] = None,
    target_column: str = "TotalFlow",
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for a RandomForestRegressor using GridSearchCV and save the best model and results.

    Args:
        X_train_path: Path to the training features CSV.
        output_name: Base name for output files (model, results, importances).
        y_train_path: Path to the training target CSV. Optional if target is in X_train_path CSV.
        target_column: The name of the target column (default: "TotalFlow").

    Returns:
        model_path: The absolute path to the saved best model file.
        best_params: A dictionary of the best hyperparameters found.
        grid_results_path: The absolute path to the saved grid search results JSON file.
        feature_importances_path: The absolute path to the saved feature importances CSV file.
    """
    DEFAULT_SEP = ","
    DEFAULT_ENCODING = None
    X_df = pd.read_csv(os.path.abspath(X_train_path), sep=DEFAULT_SEP, encoding=DEFAULT_ENCODING)
    
    if target_column in X_df.columns and (y_train_path is None or not os.path.isfile(y_train_path)):
        y_train = X_df[target_column].values
        X_train = X_df.drop(columns=[target_column])
    else:
        X_train = X_df
        if y_train_path is None:
             raise ValueError("y_train_path must be provided if target_column is not in X_train_path")
        y_train = pd.read_csv(os.path.abspath(y_train_path), sep=DEFAULT_SEP, encoding=DEFAULT_ENCODING)[target_column].values

    PG = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    SCORING = "neg_mean_squared_error"
    CV = 5
    N_JOBS = -1
    VERBOSE = 0

    RF_CFG = {"random_state": 42}
    rf = RandomForestRegressor(**RF_CFG)
    gs = GridSearchCV(estimator=rf, param_grid=PG, cv=CV, scoring=SCORING, n_jobs=N_JOBS, verbose=VERBOSE)
    gs.fit(X_train, y_train)

    base_dir = OUTPUT_DIR
    os.makedirs(base_dir, exist_ok=True)
    model_path = _get_out_path(f"{output_name}_best_model.joblib", base_dir)
    joblib.dump(gs.best_estimator_, model_path)

    cv_path = _get_out_path(f"{output_name}_gridsearch_results.json", base_dir)
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump({"best_params": gs.best_params_, "best_score": float(gs.best_score_)}, f, ensure_ascii=False, indent=2)

    fi_path = _get_out_path(f"{output_name}_feature_importances.csv", base_dir)
    import pandas as pd
    pd.DataFrame({"feature": X_train.columns, "importance": gs.best_estimator_.feature_importances_}).to_csv(fi_path, index=False)

    return {
        "model_path": model_path,
        "best_params": gs.best_params_,
        "grid_results_path": cv_path,
        "feature_importances_path": fi_path,
    }

# @mcp.tool()
def generate_probability_raster(
    model_path: str,
    data_path: str,
    metadata_path: str,
    output_name: str
) -> Dict[str, Any]:
    """
    Generate probability rasters from a trained model.

    Args:
        model_path: Path to the saved model file (.joblib).
        data_path: Path to the input data NPY file.
        metadata_path: Path to the metadata pickle file.
        output_name: Base name for the output probability raster file.

    Returns:
        proba_raster_path: The absolute path to the saved probability raster TIFF file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = joblib.load(model_path)
    data = np.load(data_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    mask_band = data[0]
    nodata_mask = np.isnan(mask_band)
    X_pix = data.reshape((data.shape[0], data.shape[1] * data.shape[2])).T
    X = X_pix[np.invert(nodata_mask.flatten())]
    predictions = model.predict_proba(X)[:, 1]
    pred_ar = np.zeros(shape=nodata_mask.flatten().shape, dtype='float32')
    pred_ar[np.invert(nodata_mask.flatten())] = predictions
    pred_ar = pred_ar.reshape(nodata_mask.shape)
    pred_ar[nodata_mask] = np.nan
    out_tif = _get_out_path(f"{output_name}_proba.tif")
    transform = metadata.get("transform")
    crs = metadata.get("crs")
    with rasterio.open(
        out_tif,
        'w',
        driver='GTiff',
        height=pred_ar.shape[0],
        width=pred_ar.shape[1],
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(pred_ar, 1)
    return {"proba_raster_path": out_tif}

# @mcp.tool()
def predict_habitat_suitability(
    presence_points_path: str,
    background_points_path: str,
    prediction_area_path: str,
    raster_paths: Dict[str, str],
    output_name: str,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a Random Forest on presence vs. background points and predict habitat suitability.

    Args:
        presence_points_path: Path to presence points.
        background_points_path: Path to background (pseudo-absence) points.
        prediction_area_path: Polygon mask to clip predictions.
        raster_paths: Dict mapping feature name -> raster path.
        output_name: Base name for output files (raster, metrics, train features).
        n_estimators: Number of trees in the Random Forest.
        random_state: Random seed for reproducibility.

    Returns:
        output_path: Absolute path to the prediction raster.
        auc: AUC computed on the training points.
        metrics_path: Absolute path to the sensitivity table CSV.
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Internal Constants
    LABEL_FIELD = "label"
    PRESENCE_VAL = 1
    BACKGROUND_VAL = 0
    CLASS_WEIGHT = "balanced"
    THRESHOLD_STEP = 0.01

    # 1. 加载存在点（Presence Points）
    gdf_presence = gpd.read_file(presence_points_path)
    gdf_presence[LABEL_FIELD] = PRESENCE_VAL
    target_crs = gdf_presence.crs
    
    # 2. 加载背景点（Background Points）
    gdf_bg = gpd.read_file(background_points_path)
    if gdf_bg.crs != target_crs:
        gdf_bg = gdf_bg.to_crs(target_crs)
    gdf_bg[LABEL_FIELD] = BACKGROUND_VAL
    
    # 3. 合并训练数据（将存在点与背景点纵向拼接，形成完整的训练集）
    gdf_train = pd.concat([gdf_presence, gdf_bg], ignore_index=True)
    
    # 4. 从栅格中提取训练数据
    # Get transform from first raster
    first_raster = list(raster_paths.values())[0]
    with rasterio.open(first_raster) as src:
        transform = src.transform
        shape = src.shape
        profile = src.profile
        
    # 读取所有栅格数据并存储在字典中
    raster_data = {}
    for key, path in raster_paths.items():
        with rasterio.open(path) as src:
            raster_data[key] = src.read(1)
            
    # 从训练数据点的坐标提取栅格值
    rows, cols = rasterio.transform.rowcol(transform, gdf_train.geometry.x, gdf_train.geometry.y)
    rows = np.clip(rows, 0, shape[0]-1)
    cols = np.clip(cols, 0, shape[1]-1)
    
    for key in raster_paths:
        gdf_train[key] = raster_data[key][rows, cols]
        
    # 删除Nan值
    gdf_train = gdf_train.dropna(subset=list(raster_paths.keys()))
    
    # 5. 训练随机森林模型并评估
    # Ensure deterministic order
    keys_order = sorted(list(raster_paths.keys()))
    X = gdf_train[keys_order]
    y = gdf_train[LABEL_FIELD]
    # 使用RandomForestClassifier训练模型
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight=CLASS_WEIGHT)
    clf.fit(X, y)
    # 计算训练集上的准确率和AUC
    score = clf.score(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)

    # 保存训练样本（包含存在点和背景点）及其提取的特征到Shapefile
    train_features_path = _get_out_path(f"{output_name}_trainfeatures.shp")
    gdf_train.to_file(train_features_path)
    
    # 6. 对全局栅格进行预测
    flat_data = {k: v.flatten() for k, v in raster_data.items()}
    # keys_order is already defined above
    X_pred = np.column_stack([flat_data[k] for k in keys_order])
    
    # Prediction mask (valid pixels)对所有栅格像元值进行概率预测（输出0~1的连续值）
    valid_mask = np.ones(X_pred.shape[0], dtype=bool)
    for k in raster_paths:
        valid_mask &= ~np.isnan(flat_data[k])
        
    pred_flat = np.full(X_pred.shape[0], np.nan)
    if np.any(valid_mask):
        pred_flat[valid_mask] = clf.predict_proba(X_pred[valid_mask])[:, 1]
        
    pred_grid = pred_flat.reshape(shape)
    
    # Apply Prediction Area Mask 对预测结果进行裁剪，仅保留预测区域内的结果
    gdf_pred_area = gpd.read_file(prediction_area_path)
    if gdf_pred_area.crs != target_crs:
        gdf_pred_area = gdf_pred_area.to_crs(target_crs)
        
    mask = geometry_mask(gdf_pred_area.geometry, transform=transform, invert=True, out_shape=shape)
    pred_grid[~mask] = np.nan
    
    # Fill NaNs with nodata value from profile to ensure consistency
    if profile.get('nodata') is not None:
        pred_grid[np.isnan(pred_grid)] = profile['nodata']
    
    # 保存预测结果到GeoTIFF文件
    output_path = _get_out_path(f"{output_name}.tif")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pred_grid.astype(rasterio.float32), 1)
        
    # 生成并保存预测结果的（敏感度表）
    metrics_list = []
    for thresh in np.arange(0, 1.0 + THRESHOLD_STEP, THRESHOLD_STEP):
        y_pred_t = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred_t, labels=[0,1]).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        omis = fn/(tp+fn) if (tp+fn)>0 else 0
        metrics_list.append({
            "Threshold": thresh, 
            "Sensitivity": sens, 
            "Specificity": spec, 
            "Omission_Rate": omis,
            "AUC": auc
        })
    
    csv_path = _get_out_path(f"{output_name}_sensitivity_table.csv")
    pd.DataFrame(metrics_list).to_csv(csv_path, index=False)
    
    return {
        "output_path": output_path,
        "auc": auc,
        "metrics_path": csv_path
    }

# @mcp.tool()
def generate_model_predictions(
    model_path: str,
    X_test_path: str,
    output_name: str,
    output_type: str = "labels",
    pred_column: str = "y_pred",
) -> Dict[str, Any]:
    """
    Generate predictions on the test set using a trained model.

    Args:
        model_path: Path to the saved model file (.joblib).
        X_test_path: Path to the testing features CSV.
        output_name: Base name for the output files (will be prefixed to _predictions.csv or _y_proba.csv).
        output_type: Type of output to generate: "labels" for class/value predictions, "proba" for probabilities.
        pred_column: The column name for the predicted values (default: "y_pred").

    Returns:
        predictions_path: The absolute path to the saved predictions/probabilities CSV file.
        count: The number of predictions generated.
        n_classes: Number of classes (only returned if output_type="proba").
    """
    model = joblib.load(model_path)
    DEFAULT_SEP = ","
    DEFAULT_ENCODING = None
    if X_test_path.lower().endswith('.csv'):
        X_test = pd.read_csv(X_test_path, sep=DEFAULT_SEP, encoding=DEFAULT_ENCODING)
        X_arr = X_test.values
    else:
        X_arr = np.load(X_test_path)

    base_dir = OUTPUT_DIR
    os.makedirs(base_dir, exist_ok=True)
    INDEX = False
    if output_type == "proba":
        proba = model.predict_proba(X_arr)
        cols = [f"proba_class_{i}" for i in range(proba.shape[1])]
        df_out = pd.DataFrame(proba, columns=cols)
        out_path = _get_out_path(f"{output_name}_y_proba.csv", base_dir)
        df_out.to_csv(out_path, index=INDEX)
        return {"proba_path": out_path, "count": int(len(df_out)), "n_classes": int(proba.shape[1])}
    else:
        preds = model.predict(X_arr)
        out_path = _get_out_path(f"{output_name}_predictions.csv", base_dir)
        pd.DataFrame({pred_column: preds}).to_csv(out_path, index=INDEX)
        return {"predictions_path": out_path, "count": int(len(preds))}

# @mcp.tool()
def extract_raster_variables(
    raster_paths: List[str],
    output_name: str,  
    band: int = 1  
) -> Dict[str, Any]:
    """
    Read raster files (e.g., GeoTIFFs), stack them into a 3D array, and save the data and metadata.
    
    Args:
        raster_paths: List of paths to raster files.
        output_name: Prefix for output filenames.
        band: Band number to read from each raster (default: 1).
        
    Returns:
        output_path: Path to the saved 3D numpy array of raster data.
        metadata_path: Path to the saved metadata (transform, nodata, names).
        names: List of variable names.
        shape: Shape of the stacked data.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data = []
    names = []
    transform = None
    nodata = None
    crs = None
    
    # Ensure consistent order
    raster_paths.sort()
    
    for fn in raster_paths:
        with rasterio.open(fn, 'r') as src:
            # 检查指定波段是否存在
            if band > src.count:
                raise ValueError(f"Band {band} not available in {fn}. Max band: {src.count}")
                
            if transform is None:
                transform = src.transform
                nodata = src.nodata
                crs = src.crs
            
            # 使用指定的波段号读取数据
            d = src.read(band)
            nodata_mask = (d == src.nodata) if src.nodata is not None else np.isnan(d)
            d[nodata_mask] = np.nan
            
            data.append(d)
            # Remove extension for name
            name = os.path.basename(fn)
            if '.' in name:
                name = name.rsplit('.', 1)[0]
            names.append(name)

    data_arr = np.stack(data)
    
    # 使用用户指定的output_name生成输出文件名
    data_path = _get_out_path(f"{output_name}.npy")
    np.save(data_path, data_arr)
    
    metadata = {
        "transform": transform,
        "nodata": nodata,
        "names": names,
        "shape": data_arr.shape,
        "crs": crs,
        "band_used": band,  # 新增：记录使用的波段号到元数据
    }
    
    metadata_path = _get_out_path(f"{output_name}_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
        
    return {
        "output_path": data_path,
        "metadata_path": metadata_path,
        "names": names,
        "shape": data_arr.shape,
    }

# @mcp.tool()
def prepare_model_data(
    data_path: str,
    output_name: str,
    target_column: str = "TotalFlow",
    feature_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
    index: bool = False,
) -> Dict[str, Any]:
    """
    Prepare the feature matrix (x) and target vector (y) from the merged interactions dataset for machine learning.

    Args:
        data_path: Path to the merged interactions CSV file.
        output_name: Base name for output files (will generate {output_name}_x.csv and {output_name}_y.csv).
        target_column: The column name to be used as the target variable y (default: "TotalFlow").
        feature_columns: A list of specific column names to use as features. If provided, only these columns are used.
        drop_columns: A list of columns to exclude from features if feature_columns is not provided (default: None, implies ["Origin", "Destination"]).
        index: Whether to include the index in the saved CSVs (default: False).

    Returns:
        x_path: The absolute path to the saved feature matrix CSV.
        y_path: The absolute path to the saved target vector CSV.
        data_path: The absolute path to the saved combined cleaned CSV.
        feature_count: The number of feature columns in x.
        sample_count: The number of samples (rows) in x and y.
    """
    mp = os.path.abspath(data_path)
    DEFAULT_HEADER = 0
    DEFAULT_INDEX_COL = None
    DEFAULT_SEP = ","
    DEFAULT_ENCODING = None
    df = pd.read_csv(mp, index_col=DEFAULT_INDEX_COL, header=DEFAULT_HEADER, sep=DEFAULT_SEP, encoding=DEFAULT_ENCODING)
    target = target_column
    if target not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) > 0:
            target = num_cols[0]
        else:
            raise ValueError(f"Merged data missing target field '{target_column}', and no numeric columns found in data")

    if feature_columns is not None and len(feature_columns) > 0:
        X = df[feature_columns].copy()
    else:
        default_drop = ["Origin", "Destination"]
        drop_cols = drop_columns if drop_columns is not None else default_drop
        X = df.drop(columns=[target] + drop_cols, errors="ignore")
    X = X.select_dtypes(include=["number"])
    y = df[target]

    data = pd.concat([X, y], axis=1)
    data = data.dropna()
    X = data.drop(columns=[data.columns[-1]])
    y = data.iloc[:, -1]

    base_dir = OUTPUT_DIR
    os.makedirs(base_dir, exist_ok=True)
    x_path = _get_out_path(f"{output_name}_x.csv", base_dir)
    y_path = _get_out_path(f"{output_name}_y.csv", base_dir)
    data_path = _get_out_path(f"{output_name}_prepared.csv", base_dir)
    
    X.to_csv(x_path, index=index)
    pd.DataFrame({target: y}).to_csv(y_path, index=index)
    
    # Save combined cleaned data for compatibility with split_train_test
    # data variable already contains X and y combined and cleaned (dropna)
    data.to_csv(data_path, index=index)
    
    return {
        "x_path": x_path,
        "y_path": y_path,
        "data_path": data_path,
        "feature_count": int(X.shape[1]),
        "sample_count": int(X.shape[0]),
    }

# @mcp.tool()
def undersample_raster_data(
    data_path: str,
    labels_path: str,
    metadata_path: str,
    output_name: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Align data and labels, remove NaNs, and perform random undersampling.
    
    Args:
        data_path: Path to the stacked data numpy array.
        labels_path: Path to the labels numpy array.
        metadata_path: Path to metadata.
        output_name: Prefix for output filenames.
        random_state: Random seed for reproducibility (default: 42).
        
    Returns:
        output_path: Path to the saved undersampled CSV file.
        sample_count: Number of samples after resampling.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data = np.load(data_path) # (bands, h, w)
    labels = np.load(labels_path) # (h, w)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    mask_band = data[0]
    nodata_mask = np.isnan(mask_band)
    labels[nodata_mask] = np.nan
    
    X_pix = data.reshape((data.shape[0], data.shape[1] * data.shape[2])).T
    y_pix = labels.flatten()
    
    valid_mask = ~np.isnan(y_pix)
    X = X_pix[valid_mask]
    y = y_pix[valid_mask]
    
    # Undersampling
    rus = RandomUnderSampler(random_state=random_state)
    X_strat, y_strat = rus.fit_resample(X, y)
    
    names = metadata.get("names") or []
    if len(names) != X_strat.shape[1]:
        names = [f"band_{i}" for i in range(X_strat.shape[1])]
    df = pd.DataFrame(X_strat, columns=names)
    df["label"] = y_strat
    csv_path = _get_out_path(f"{output_name}_undersampled.csv")
    df.to_csv(csv_path, index=False)
    return {
        "output_path": csv_path,
        "sample_count": len(y_strat)
    }

# @mcp.tool()
def evaluate_classification_auc(
    labels_path: str,
    predictions_path: str,
    output_name: str,
    label_column: str = "label",
    class_index: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate classification performance using AUC-ROC.

    Args:
        labels_path: Path to the true labels (CSV or NPY).
        predictions_path: Path to the predicted probabilities CSV.
        output_name: Base name for the output metrics file.
        label_column: Name of the column containing true labels (if CSV input) (default: "label").
        class_index: Index of the positive class probability column (default: 1).

    Returns:
        metrics_path: Path to the saved JSON file containing metrics.
        roc_auc: The calculated Area Under the ROC Curve.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if labels_path.lower().endswith('.csv'):
        y_df = pd.read_csv(labels_path)
        if label_column in y_df.columns:
            y_test = y_df[label_column].values
        else:
            y_test = y_df.iloc[:, 0].values
    else:
        y_test = np.load(labels_path)
    proba_df = pd.read_csv(predictions_path)
    col = f"proba_class_{class_index}"
    if col not in proba_df.columns:
        col = proba_df.columns[-1]
    y_scores = proba_df[col].values
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    metrics = {"roc_auc": float(roc_auc), "n_test": int(len(y_test))}
    metrics_path = _get_out_path(f"{output_name}_rf_auc_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return {"metrics_path": metrics_path, "roc_auc": roc_auc}

# @mcp.tool()
def evaluate_regression_mse(
    labels_path: str,
    predictions_path: str,
    output_name: str,
    label_column: str = "TotalFlow",
    pred_column: str = "y_pred",
) -> Dict[str, Any]:
    """
    Evaluate the model predictions against the ground truth values and save the metrics.

    Args:
        labels_path: Path to the testing target CSV containing ground truth values.
        predictions_path: Path to the predictions CSV containing predicted values.
        output_name: Base name for output files (metrics, combined predictions).
        label_column: The column name of the ground truth values (default: "TotalFlow").
        pred_column: The column name of the predicted values (default: "y_pred").

    Returns:
        metrics_path: The absolute path to the saved metrics JSON file.
        mse: The Mean Squared Error (MSE) of the predictions.
    """
    DEFAULT_SEP = ","
    DEFAULT_ENCODING = None
    y_test = pd.read_csv(os.path.abspath(labels_path), sep=DEFAULT_SEP, encoding=DEFAULT_ENCODING)[label_column].values
    preds = pd.read_csv(os.path.abspath(predictions_path), sep=DEFAULT_SEP, encoding=DEFAULT_ENCODING)[pred_column].values
    mse = float(mean_squared_error(y_test, preds))

    base_dir = OUTPUT_DIR
    os.makedirs(base_dir, exist_ok=True)
    INDEX = False
    metrics_path = _get_out_path(f"{output_name}.json", base_dir)
    try:
        # Try to load grid search results using standard naming convention
        grid_results_path = os.path.join(base_dir, f"{output_name}_gridsearch_results.json")
        if os.path.exists(grid_results_path):
            with open(grid_results_path, "r", encoding="utf-8") as f:
                gs_info = json.load(f)
            best_params = gs_info.get("best_params")
        else:
             best_params = None
    except Exception:
        best_params = None
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"mse": mse, "best_params": best_params}, f, ensure_ascii=False, indent=2)

    try:
        combined = pd.DataFrame({"y_true": y_test, pred_column: preds})
        combined.to_csv(_get_out_path(f"{output_name}_predictions_comparison.csv", base_dir), index=INDEX)
    except Exception:
        pass

    return {
        "metrics_path": metrics_path,
        "mse": mse,
    }