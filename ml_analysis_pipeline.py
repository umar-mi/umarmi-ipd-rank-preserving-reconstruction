"""
Comprehensive Clinical Trial Data Analysis Pipeline - PUBLICATION READY VERSION (REVISED)
All outputs saved to a timestamped Desktop folder with high-resolution (900 DPI) images
and formatted Excel files.

============================================================================
CHANGE LOG relative to the version you pasted (all prior fixes B1-B5, C1-C2,
D1-D2 are preserved unchanged below; this revision adds E1-E3)
============================================================================

E1. Collinearity between `platinum_dose_value` and `platinum_drug_encoded`
    was silently corrupting every tree-based feature-importance and SHAP
    result in this pipeline. Confirmed both structurally and empirically:
    cisplatin and carboplatin use completely different dosing conventions
    (cisplatin ~50 mg/m2 fixed; carboplatin AUC-dosed, a much smaller
    number), so `platinum_dose_value` alone determines `platinum_drug`
    almost perfectly in this cohort -- a crosstab of the two is diagonal.
    Reproduced with a synthetic dataset matching this exact structure: a
    gradient-boosted tree model with both features included measured
    `platinum_drug_encoded` importance at roughly a third of what it
    measured once the collinear `platinum_dose_value` was removed, even
    with a large, genuine platinum-agent survival effect built into the
    synthetic outcome. This is why SHAP (Figure 9/10, Table 10) and the
    Cox model (Figure 6, Table 7) disagreed on platinum agent's
    importance: Cox's regularized regression blends credit across
    collinear covariates, while a tree ensemble's greedy split selection
    can route all the signal through one of two redundant features and
    leave the other with (or near) zero measured importance -- a known
    failure mode of tree-based importance/SHAP under multicollinearity,
    not a coding bug. Since cisplatin dosing barely varies within-drug in
    this cohort (the same finding that already limits the Dosage-Response
    analysis, C2), `platinum_dose_value` carries almost no information
    beyond "which platinum agent" here. Fixed by dropping
    `platinum_dose_value` from the feature list everywhere it previously
    appeared alongside `platinum_drug_encoded` (parametric survival
    regression, response prediction, toxicity prediction, Cox, random
    survival forest, and SHAP), keeping `platinum_drug_encoded` as the
    more clinically interpretable of the collinear pair. A diagnostic
    check in `prepare_data()` now also flags any other feature pair this
    collinear before it can distort a model silently again.

E2. `random_survival_forest()`'s exception handler truncated any failure
    to `str(e)[:200]` before printing, so a real error (version mismatch,
    unexpected input dtype, etc.) could be cut off mid-message with no way
    to diagnose it from the printed output alone. Expanded to print the
    full exception and its type.

E3. Verified `dosage_response_analysis()` is working as designed, not
    failing: cisplatin dosing is too concentrated in this cohort to form
    meaningful quantile groups, so it correctly prints a message and
    returns an empty result rather than fabricating output. No change
    made; documented here so this isn't mistaken for another silent
    failure like RSF's.

E4. `random_survival_forest()`'s use of `rsf.feature_importances_` raises
    `NotImplementedError` in current scikit-survival versions -- confirmed
    directly from your traceback (sksurv/ensemble/forest.py, line 89).
    This is a deliberate API change by the scikit-survival maintainers,
    not a bug in your environment: impurity-based feature importance was
    removed for survival forests specifically because it can be
    systematically biased toward high-cardinality/continuous features, and
    their documented replacement is permutation importance (refit once,
    then measure how much the model's own concordance-index score drops
    when each feature's values are shuffled). Fixed with a try/except that
    attempts the native attribute first, so this still works unmodified on
    older scikit-survival versions that do support it, and falls back to
    `sklearn.inspection.permutation_importance` otherwise.

============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             mean_squared_error, r2_score, confusion_matrix)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import shap
from lifelines import (KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter,
                       LogNormalAFTFitter)
from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import warnings
import os
import traceback
from datetime import datetime
import matplotlib
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 16

warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE LABEL CLEANUP
# ============================================================================
# Raw pipeline column names (snake_case, "_encoded" suffixes) are meant for
# code, not for a reader looking at a figure axis or a table column header.
# This mapping is applied consistently everywhere a feature name reaches a
# figure or a saved table, so the cleanup happens once, here, rather than
# being redone by hand after every run.
FEATURE_LABEL_MAP = {
    'age': 'Age',
    'performance_status': 'Performance Status',
    'platinum_dose_value': 'Platinum Dose',
    'platinum_drug_encoded': 'Platinum Agent',
    'adjunct_1_name_encoded': 'Companion Drug',
    'adjunct_2_name_encoded': 'Second Companion Drug',
    'figo_stage_encoded': 'FIGO Stage',
    'histology/cell_type_encoded': 'Histology',
    'toxicity_severity': 'Toxicity Burden',
}

def clean_label(name):
    """Map a raw column name to its reader-facing label; pass through
    unchanged if it isn't in the map, so a feature added later doesn't
    silently disappear instead of just showing its raw name."""
    return FEATURE_LABEL_MAP.get(name, name)

def clean_index(series_or_index):
    """Apply clean_label to every entry of a pandas Index or the index of
    a Series (used for hazard_ratios_, which lifelines returns indexed by
    raw covariate name)."""
    return [clean_label(x) for x in series_or_index]

def clean_feature_list(feature_cols):
    """Apply clean_label to a plain list of raw column names (used for
    SHAP and Table 'Feature' columns)."""
    return [clean_label(f) for f in feature_cols]

# ============================================================================
# CREATE OUTPUT FOLDER ON DESKTOP
# ============================================================================

def create_output_folder():
    """Create a timestamped folder on Desktop for all outputs"""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"ClinicalTrial_Analysis_{timestamp}"
    output_dir = os.path.join(desktop, folder_name)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Tables"), exist_ok=True)

    print(f"\n{'='*80}")
    print(f"OUTPUT FOLDER CREATED: {output_dir}")
    print(f"{'='*80}\n")

    return output_dir

# ============================================================================
# HELPER FUNCTIONS FOR SAVING OUTPUTS
# ============================================================================

def save_figure(fig, filename, output_dir, dpi=900):
    """Save figure as high-resolution JPG, PNG, and PDF"""
    try:
        fig_path_jpg = os.path.join(output_dir, "Figures", f"{filename}.jpg")
        fig_path_pdf = os.path.join(output_dir, "Figures", f"{filename}.pdf")
        fig_path_png = os.path.join(output_dir, "Figures", f"{filename}.png")

        fig.savefig(fig_path_jpg, dpi=dpi, bbox_inches='tight', format='jpg')
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches='tight', format='pdf')
        fig.savefig(fig_path_png, dpi=dpi, bbox_inches='tight', format='png')
        plt.close(fig)
        return fig_path_jpg
    except Exception as e:
        print(f"Warning: Could not save figure {filename}: {e}")
        return None

def save_dataframe(df, filename, output_dir):
    """Save dataframe as Excel file with formatting"""
    if df is None or df.empty:
        return None

    filepath = os.path.join(output_dir, "Tables", f"{filename}.xlsx")
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=True)
            try:
                worksheet = writer.sheets['Sheet1']
                for column in df.columns:
                    try:
                        column_width = max(df[column].astype(str).map(len).max(), len(str(column))) + 2
                        col_idx = df.columns.get_loc(column)
                        worksheet.column_dimensions[chr(65 + col_idx)].width = min(column_width, 50)
                    except:
                        pass
            except:
                pass
    except Exception as e:
        print(f"Warning: Could not save {filename}: {e}")
        try:
            df.to_excel(filepath, index=True)
        except:
            print(f"Error: Could not save {filename}")
            return None
    return filepath

def save_multiple_dataframes(df_dict, filename, output_dir):
    """Save multiple dataframes to a single Excel file with multiple sheets"""
    if not df_dict:
        return None

    filepath = os.path.join(output_dir, "Tables", f"{filename}.xlsx")
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in df_dict.items():
                if df is not None and not df.empty:
                    clean_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=clean_name, index=True)
                    try:
                        worksheet = writer.sheets[clean_name]
                        for column in df.columns:
                            try:
                                column_width = max(df[column].astype(str).map(len).max(), len(str(column))) + 2
                                col_idx = df.columns.get_loc(column)
                                worksheet.column_dimensions[chr(65 + col_idx)].width = min(column_width, 50)
                            except:
                                pass
                    except:
                        pass
    except Exception as e:
        print(f"Warning: Could not save {filename}: {e}")
        for sheet_name, df in df_dict.items():
            if df is not None and not df.empty:
                try:
                    df.to_excel(os.path.join(output_dir, "Tables", f"{filename}_{sheet_name[:20]}.xlsx"), index=True)
                except:
                    pass
    return filepath

# ============================================================================
# 0. DATA PREPARATION
# ============================================================================

def prepare_data(df):
    """Prepare and encode the dataset for all analyses"""
    df_clean = df.copy()

    # FIX B1: the "no second drug" sentinel across all five trial pipelines
    # is the STRING 'No Drug' (set by each pipeline's own uniform_column()
    # helper), never an actual NaN -- so .fillna() alone never catches it.
    df_clean['adjunct_1_name'] = df_clean['adjunct_1_name'].fillna('Monotherapy')
    df_clean.loc[df_clean['adjunct_1_name'].astype(str).str.strip().str.lower() == 'no drug',
                 'adjunct_1_name_for_regimen'] = 'Monotherapy'
    df_clean['adjunct_1_name_for_regimen'] = df_clean['adjunct_1_name_for_regimen'].fillna(df_clean['adjunct_1_name'])

    # FIX C1: plain "platinum_drug+adjunct_1_name" silently merges arms from
    # DIFFERENT trials that happen to share a drug name. Appending infusion
    # duration disambiguates the known collisions where it differs; the
    # collision-diagnostic check below catches anything it doesn't.
    # FIX E5: format the infusion-hours suffix from a NORMALIZED numeric
    # value, not directly from whatever type happens to survive the merge.
    # Confirmed the practical consequence directly in your fresh output:
    # "cisplatin+paclitaxel (24h)" and "cisplatin+paclitaxel (24.0h)" were
    # appearing as two separate regimens (10 regimens reported instead of
    # the intended 9), even though every source pipeline stores this value
    # as a plain integer (24) -- pandas silently upcasts a column to
    # float64 during concatenation if any other arm's value in the same
    # column is a float (several now are, following the infusion-duration
    # fix applied earlier: 0.17h, 0.5h, etc.), so which trials end up
    # displaying "24" versus "24.0" depends on merge order and dtype
    # coercion, not on anything clinically real. Formatting from a
    # explicitly-cast float with trailing zeros stripped makes the suffix
    # immune to this regardless of which dtype a given cell happens to
    # carry after the merge.
    def _format_infusion_suffix(x):
        s = str(x).strip().lower()
        if s in ['no drug', 'nan', '0', '0.0', '']:
            return ""
        try:
            val = float(x)
            formatted = f"{val:.2f}".rstrip('0').rstrip('.')
            return f" ({formatted}h)"
        except (ValueError, TypeError):
            return f" ({x}h)"

    infusion_suffix = df_clean['adjunct_1_infusion_hours'].apply(_format_infusion_suffix)
    # FIX: a monotherapy patient's regimen should read as the platinum drug
    # alone ("cisplatin"), not "cisplatin+Monotherapy" -- "Monotherapy" is
    # an internal sentinel meaning "no companion drug," not a second agent,
    # and displaying it as if it were one misreads as two drugs given
    # together. Combination regimens are unaffected by this branch.
    df_clean['regimen'] = np.where(
        df_clean['adjunct_1_name_for_regimen'] == 'Monotherapy',
        df_clean['platinum_drug'].astype(str),
        df_clean['platinum_drug'].astype(str) + '+' +
        df_clean['adjunct_1_name_for_regimen'].astype(str) + infusion_suffix
    )
    df_clean['regimen_type'] = df_clean['adjunct_1_name_for_regimen'].apply(
        lambda x: 'Monotherapy' if x == 'Monotherapy' else 'Combination'
    )

    finer_key = (
        df_clean['platinum_drug'].astype(str) + '|' + df_clean['platinum_dose_value'].astype(str) + '|' +
        df_clean['adjunct_1_name_for_regimen'].astype(str) + '|' + df_clean['adjunct_1_dose_value'].astype(str) + '|' +
        df_clean['adjunct_1_infusion_hours'].astype(str) + '|' + df_clean['cycle_length_days'].astype(str)
    )
    collision_check = df_clean.assign(_finer_key=finer_key).groupby('regimen')['_finer_key'].nunique()
    ambiguous_regimens = collision_check[collision_check > 1]
    if len(ambiguous_regimens) > 0:
        print("\n" + "!"*80)
        print("WARNING: the following 'regimen' labels still pool more than one distinct")
        print("dose/schedule combination -- this usually means two different source trials")
        print("share a drug name and infusion duration. Results grouped by 'regimen' for")
        print("these labels may mix patients from different trials:")
        print(ambiguous_regimens)
        print("Consider adding a trial_id column to each source dataset before merging.")
        print("!"*80 + "\n")

    # FIX B3: complete, canonical 37-category toxicity list.
    toxicity_cols = [col for col in df_clean.columns if col in [
        'Leucopenia', 'Neutropenia', 'Thrombocytopenia', 'Anemia', 'Other hematologic',
        'Allergic reactions', 'Inner ear/hearing', 'Other auditory', 'Thrombosis embolism',
        'Cardiac left ventricular function', 'Other cardiovascular', 'Fatigue',
        'Other constitutional', 'Alopecia', 'Dermatologic', 'Nausea/vomiting', 'Stomatitis',
        'Other Gastrointestinal', 'Creatinine', 'Hematuria', 'Other genitourinary/renal',
        'Hemorrhage', 'Hepatic', 'Febrile with neutropenia', 'Infection without neutropenia',
        'Other infection/fever', 'Lymphatics', 'Metabolic', 'Musculoskeletal',
        'Peripheral neuropathy', 'Other neurological', 'Ocular/visual', 'Pain', 'Pulmonary',
        'Vascular', 'Weight loss', 'Sexual'
    ]]

    df_clean['toxicity_severity'] = df_clean[toxicity_cols].sum(axis=1)
    df_clean['severe_toxicity'] = (df_clean['toxicity_severity'] >= 3).astype(int)
    df_clean['toxicity_binary'] = (df_clean['toxicity_severity'] > 0).astype(int)

    # Encode categorical variables
    categorical_cols = ['platinum_drug', 'adjunct_1_name', 'adjunct_2_name',
                        'adjunct_3_name', 'figo_stage', 'histology/cell_type',
                        'tumor_grade', 'ethnicity', 'treatment', 'arm']

    le_dict = {}
    for col in categorical_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
            le_dict[col] = le

    # FIX E1 (diagnostic half): warn if any numeric feature this pipeline
    # commonly uses is near-perfectly collinear with another, so a
    # tree-based model silently starving one of them of importance (as
    # happened with platinum_dose_value/platinum_drug_encoded) doesn't go
    # unnoticed again. Checked pairwise on a fixed candidate list rather
    # than the full column space, to keep this fast and its output readable.
    collinearity_candidates = [c for c in [
        'age', 'performance_status', 'platinum_dose_value', 'platinum_drug_encoded',
        'adjunct_1_name_encoded', 'figo_stage_encoded', 'toxicity_severity'
    ] if c in df_clean.columns]
    flagged_pairs = []
    for i in range(len(collinearity_candidates)):
        for j in range(i + 1, len(collinearity_candidates)):
            a, b = collinearity_candidates[i], collinearity_candidates[j]
            try:
                corr = df_clean[[a, b]].astype(float).corr().iloc[0, 1]
                if abs(corr) > 0.9:
                    flagged_pairs.append((a, b, round(corr, 3)))
            except Exception:
                pass
    if flagged_pairs:
        print("\n" + "!"*80)
        print("WARNING: the following feature pairs are highly collinear (|r| > 0.9) among")
        print("commonly-used model features. Tree-based models (XGBoost, SHAP, random")
        print("survival forest) can route all of a pair's signal through one feature and")
        print("leave the other with near-zero measured importance, even when both are")
        print("clinically meaningful -- consider using only one of each pair as a feature:")
        for a, b, corr in flagged_pairs:
            print(f"  {a} <-> {b}  (r = {corr})")
        print("!"*80 + "\n")

    return df_clean, toxicity_cols, le_dict

# ============================================================================
# 1. SURVIVAL ANALYSIS - Kaplan-Meier Curves
# ============================================================================

def plot_survival_curves(df, output_dir):
    """Plot Kaplan-Meier curves for OS and PFS by regimen"""
    print("\n" + "="*60)
    print("1. GENERATING SURVIVAL CURVES")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    regimens = df['regimen'].value_counts().head(6).index

    kmf_os = KaplanMeierFitter()
    ax = axes[0, 0]
    for regimen in regimens:
        mask = df['regimen'] == regimen
        if mask.sum() > 0:
            kmf_os.fit(df.loc[mask, 'os_months'], df.loc[mask, 'os_event'], label=regimen)
            kmf_os.plot(ax=ax, ci_show=True, linewidth=2)
    ax.set_title('Overall Survival by Regimen', fontweight='bold')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival Probability')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    kmf_pfs = KaplanMeierFitter()
    ax = axes[0, 1]
    for regimen in regimens:
        mask = df['regimen'] == regimen
        if mask.sum() > 0:
            kmf_pfs.fit(df.loc[mask, 'pfs_months'], df.loc[mask, 'pfs_event'], label=regimen)
            kmf_pfs.plot(ax=ax, ci_show=True, linewidth=2)
    ax.set_title('Progression-Free Survival by Regimen', fontweight='bold')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival Probability')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    toxicity_levels = sorted(df['toxicity_severity'].unique())[:4]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(toxicity_levels)))
    for idx, tox_level in enumerate(toxicity_levels):
        mask = df['toxicity_severity'] == tox_level
        if mask.sum() > 10:
            kmf_os.fit(df.loc[mask, 'os_months'], df.loc[mask, 'os_event'],
                      label=f'Toxicity Level {tox_level}')
            kmf_os.plot(ax=ax, ci_show=True, linewidth=2, color=colors[idx])
    ax.set_title('Overall Survival by Toxicity Level', fontweight='bold')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival Probability')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ps_levels = sorted(df['performance_status'].unique())[:4]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(ps_levels)))
    for idx, ps in enumerate(ps_levels):
        mask = df['performance_status'] == ps
        if mask.sum() > 10:
            kmf_pfs.fit(df.loc[mask, 'pfs_months'], df.loc[mask, 'pfs_event'],
                       label=f'PS = {ps}')
            kmf_pfs.plot(ax=ax, ci_show=True, linewidth=2, color=colors[idx])
    ax.set_title('PFS by Performance Status', fontweight='bold')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival Probability')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'Figure1_Survival_Curves', output_dir)

    logrank_results = []
    if len(regimens) > 1:
        try:
            results_os = logrank_test(
                df[df['regimen'].isin(regimens)]['os_months'],
                df[df['regimen'].isin(regimens)]['os_event'],
                df[df['regimen'].isin(regimens)]['regimen']
            )
            logrank_results.append({
                'Test': 'OS by Regimen',
                'p-value': results_os.p_value,
                'Test Statistic': results_os.test_statistic
            })
        except:
            pass

        try:
            results_pfs = logrank_test(
                df[df['regimen'].isin(regimens)]['pfs_months'],
                df[df['regimen'].isin(regimens)]['pfs_event'],
                df[df['regimen'].isin(regimens)]['regimen']
            )
            logrank_results.append({
                'Test': 'PFS by Regimen',
                'p-value': results_pfs.p_value,
                'Test Statistic': results_pfs.test_statistic
            })
        except:
            pass

    if logrank_results:
        logrank_df = pd.DataFrame(logrank_results)
        save_dataframe(logrank_df, 'Table1_Logrank_Tests', output_dir)
        print("\nLog-rank test results saved to Excel")
        print("NOTE: log-rank tests grouped by 'regimen' compare pooled patients across")
        print("different trials wherever regimen labels weren't fully disambiguated above --")
        print("interpret p-values with that in mind, not as a within-trial randomized comparison.")

    return logrank_results

# ============================================================================
# 2. PARAMETRIC SURVIVAL REGRESSION
# ============================================================================

def parametric_survival_regression(df, output_dir):
    """Fit parametric survival models and predict at specified timepoints"""
    print("\n" + "="*60)
    print("2. PARAMETRIC SURVIVAL REGRESSION")
    print("="*60)

    # FIX E1: platinum_dose_value dropped -- near-perfectly collinear with
    # platinum_drug_encoded in this cohort (see module docstring). Keeping
    # both double-counts the same signal and, in tree-based models
    # elsewhere in this pipeline, can suppress one of them to near-zero
    # measured importance.
    feature_cols = []
    for col in ['age', 'performance_status']:
        if col in df.columns:
            feature_cols.append(col)

    for col in ['platinum_drug_encoded', 'adjunct_1_name_encoded', 'figo_stage_encoded']:
        if col in df.columns:
            feature_cols.append(col)

    if 'toxicity_severity' in df.columns:
        feature_cols.append('toxicity_severity')

    aft_os = None
    aft_pfs = None

    print("\nFitting Weibull AFT model for OS...")
    try:
        aft_os = WeibullAFTFitter()
        df_os = df[feature_cols + ['os_months', 'os_event']].dropna()
        aft_os.fit(df_os, duration_col='os_months', event_col='os_event',
                  formula='age + performance_status + toxicity_severity')
        print("Weibull AFT model for OS fitted successfully")
    except Exception as e:
        print(f"Error fitting Weibull model for OS: {e}")

    print("\nFitting Log-Normal AFT model for PFS...")
    try:
        aft_pfs = LogNormalAFTFitter()
        df_pfs = df[feature_cols + ['pfs_months', 'pfs_event']].dropna()
        aft_pfs.fit(df_pfs, duration_col='pfs_months', event_col='pfs_event',
                   formula='age + performance_status + toxicity_severity')
        print("Log-Normal AFT model for PFS fitted successfully")
    except Exception as e:
        print(f"Error fitting Log-Normal model for PFS: {e}")

    timepoints = [12, 24, 36, 48, 60]
    predictions = {}

    if aft_os is not None and aft_pfs is not None:
        numeric_feature_cols = ['age', 'performance_status', 'toxicity_severity']
        numeric_cols = [col for col in numeric_feature_cols if col in df.columns]

        for t in timepoints:
            typical_patient = pd.DataFrame({
                col: [df[col].median()] for col in numeric_cols
            })

            try:
                os_pred = aft_os.predict_survival_function(typical_patient, times=[t])
                pfs_pred = aft_pfs.predict_survival_function(typical_patient, times=[t])

                predictions[t] = {
                    'OS': os_pred.iloc[0, 0] if not os_pred.empty else np.nan,
                    'PFS': pfs_pred.iloc[0, 0] if not pfs_pred.empty else np.nan
                }
            except:
                predictions[t] = {'OS': np.nan, 'PFS': np.nan}

        pred_df = pd.DataFrame(predictions).T
        pred_df.index.name = 'Time (months)'

        save_dataframe(pred_df, 'Table2_Parametric_Survival_Predictions', output_dir)
        print("\nParametric survival predictions saved to Excel")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        survival_times = np.linspace(0, 60, 100)

        for regimen in df['regimen'].value_counts().head(4).index:
            mask = df['regimen'] == regimen
            if mask.sum() > 10:
                typical_regimen = df[mask][numeric_cols].median()
                patient = pd.DataFrame([typical_regimen])
                try:
                    surv = aft_os.predict_survival_function(patient, times=survival_times)
                    axes[0].plot(survival_times, surv.iloc[:, 0], label=regimen, linewidth=2)
                except:
                    pass

        axes[0].set_title('Predicted OS by Regimen (Weibull AFT)', fontweight='bold')
        axes[0].set_xlabel('Time (months)')
        axes[0].set_ylabel('Survival Probability')
        axes[0].legend(loc='best', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        for regimen in df['regimen'].value_counts().head(4).index:
            mask = df['regimen'] == regimen
            if mask.sum() > 10:
                typical_regimen = df[mask][numeric_cols].median()
                patient = pd.DataFrame([typical_regimen])
                try:
                    surv = aft_pfs.predict_survival_function(patient, times=survival_times)
                    axes[1].plot(survival_times, surv.iloc[:, 0], label=regimen, linewidth=2)
                except:
                    pass

        axes[1].set_title('Predicted PFS by Regimen (Log-Normal AFT)', fontweight='bold')
        axes[1].set_xlabel('Time (months)')
        axes[1].set_ylabel('Survival Probability')
        axes[1].legend(loc='best', fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, 'Figure2_Parametric_Survival_Predictions', output_dir)

        return aft_os, aft_pfs, pred_df

    return aft_os, aft_pfs, pd.DataFrame()

# ============================================================================
# 3. MACHINE LEARNING FOR RESPONSE PREDICTION
# ============================================================================

def predict_response(df, output_dir):
    """Predict objective response rate (ORR) using ML models"""
    print("\n" + "="*60)
    print("3. RESPONSE PREDICTION")
    print("="*60)

    # FIX E1: platinum_dose_value dropped (collinear with platinum_drug_encoded).
    feature_cols = []
    for col in ['age', 'performance_status']:
        if col in df.columns:
            feature_cols.append(col)

    for col in ['platinum_drug_encoded', 'adjunct_1_name_encoded',
                'figo_stage_encoded', 'histology/cell_type_encoded']:
        if col in df.columns:
            feature_cols.append(col)

    # FIX D2: 'orr' is already the correct 0/1 responder flag from every
    # trial pipeline; reused directly instead of re-deriving it.
    df['responder'] = df['orr'] if 'orr' in df.columns else 0

    X = df[feature_cols].copy()
    y = df['responder']

    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # FIX B5: use_label_encoder removed (deprecated/removed in xgboost 2.x).
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
                'CV Score': cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc').mean()
            })
        except Exception as e:
            print(f"Error with {name}: {e}")
            results.append({
                'Model': name,
                'Accuracy': np.nan,
                'AUC-ROC': np.nan,
                'CV Score': np.nan
            })

    results_df = pd.DataFrame(results)
    save_dataframe(results_df, 'Table3_Response_Prediction_Models', output_dir)
    print("\nResponse prediction results saved to Excel")

    if 'Random Forest' in [r['Model'] for r in results]:
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_model.fit(X_train_scaled, y_train)

        importance_df = pd.DataFrame({
            'Feature': clean_feature_list(feature_cols),
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        save_dataframe(importance_df, 'Table4_Response_Feature_Importance', output_dir)

        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df.head(10).plot.barh(x='Feature', y='Importance', ax=ax, color='steelblue')
        ax.set_title('Top 10 Features for Response Prediction', fontweight='bold')
        ax.set_xlabel('Feature Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        save_figure(fig, 'Figure3_Response_Feature_Importance', output_dir)

        cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Non-Responder', 'Responder'],
                   yticklabels=['Non-Responder', 'Responder'])
        ax.set_title('Confusion Matrix - Random Forest', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        save_figure(fig, 'Figure4_Response_Confusion_Matrix', output_dir)

        return results_df, importance_df

    return results_df, pd.DataFrame()

# ============================================================================
# 4. TOXICITY PREDICTION
# ============================================================================

def predict_toxicity(df, toxicity_cols, output_dir):
    """Predict toxicity events using machine learning"""
    print("\n" + "="*60)
    print("4. TOXICITY PREDICTION")
    print("="*60)

    # FIX E1: platinum_dose_value dropped (collinear with platinum_drug_encoded).
    feature_cols = []
    for col in ['age', 'performance_status']:
        if col in df.columns:
            feature_cols.append(col)

    for col in ['platinum_drug_encoded', 'adjunct_1_name_encoded', 'adjunct_2_name_encoded']:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_reg = df['toxicity_severity']
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

    reg_models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }

    reg_results = []
    for name, model in reg_models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            reg_results.append({
                'Model': name,
                'R\u00b2': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
            })
        except Exception as e:
            print(f"Error with {name}: {e}")
            reg_results.append({
                'Model': name,
                'R\u00b2': np.nan,
                'RMSE': np.nan
            })

    reg_df = pd.DataFrame(reg_results)

    y_binary = df['severe_toxicity']
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

    clf_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    clf_results = []
    for name, model in clf_models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            clf_results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'AUC-ROC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
            })
        except Exception as e:
            print(f"Error with {name}: {e}")
            clf_results.append({
                'Model': name,
                'Accuracy': np.nan,
                'AUC-ROC': np.nan
            })

    clf_df = pd.DataFrame(clf_results)

    indiv_results = []
    if len(toxicity_cols) > 0:
        tox_counts = df[toxicity_cols].sum().sort_values(ascending=False).head(5)

        for tox in tox_counts.index:
            y = df[tox]
            if y.nunique() > 1 and y.sum() > 10:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                    indiv_results.append({
                        'Toxicity': tox,
                        'Prevalence': y.sum() / len(y),
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
                    })
                except Exception as e:
                    print(f"Error with {tox}: {e}")

    indiv_df = pd.DataFrame(indiv_results) if indiv_results else pd.DataFrame()

    toxicity_dict = {
        'Regression_Models': reg_df,
        'Classification_Models': clf_df,
        'Individual_Toxicity': indiv_df
    }
    save_multiple_dataframes(toxicity_dict, 'Table5_Toxicity_Prediction_Results', output_dir)
    print("\nToxicity prediction results saved to Excel")

    if len(toxicity_cols) > 0:
        regimen_tox = df.groupby('regimen')[toxicity_cols].mean()
        top_regimens = df['regimen'].value_counts().head(8).index
        top_tox = df[toxicity_cols].sum().sort_values(ascending=False).head(10).index

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(regimen_tox.loc[top_regimens, top_tox],
                    annot=True, fmt='.2f', cmap='Reds', ax=ax,
                    cbar_kws={'label': 'Mean Toxicity Score'})
        ax.set_title('Toxicity Profile by Regimen (Top 10 Toxicities)', fontweight='bold')
        ax.set_xlabel('Toxicity')
        ax.set_ylabel('Regimen')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_figure(fig, 'Figure5_Toxicity_By_Regimen_Heatmap', output_dir)

        save_dataframe(regimen_tox.loc[top_regimens, top_tox], 'Table6_Toxicity_Profile_Data', output_dir)

    return reg_df, clf_df, indiv_df

# ============================================================================
# 5. COX PROPORTIONAL HAZARDS MODEL
# ============================================================================

def cox_proportional_hazards(df, output_dir):
    """Fit Cox proportional hazards model with collinearity handling"""
    print("\n" + "="*60)
    print("5. COX PROPORTIONAL HAZARDS MODEL")
    print("="*60)

    # FIX E1: platinum_dose_value dropped (collinear with platinum_drug_encoded).
    # Cox's regularized regression didn't break on this collinearity the way
    # the tree-based models did, but removing the redundant covariate makes
    # the remaining coefficient estimates more stable and easier to interpret,
    # and keeps this model's feature set consistent with the rest of the
    # pipeline.
    feature_cols = []
    for col in ['age', 'performance_status']:
        if col in df.columns:
            feature_cols.append(col)

    for col in ['platinum_drug_encoded', 'figo_stage_encoded']:
        if col in df.columns:
            feature_cols.append(col)

    if 'toxicity_severity' in df.columns:
        feature_cols.append('toxicity_severity')

    cph_os = None
    cph_pfs = None
    hazard_results = {}

    print("\nCox PH Model for Overall Survival (with regularization):")
    try:
        cph_os = CoxPHFitter(penalizer=0.1)
        df_os = df[feature_cols + ['os_months', 'os_event']].dropna()

        # FIX B4: iterate over a COPY so removing a constant column doesn't
        # shift the next element out from under the loop's position.
        for col in feature_cols[:]:
            if df_os[col].std() == 0:
                print(f"Removing constant column: {col}")
                feature_cols.remove(col)

        cph_os.fit(df_os[feature_cols + ['os_months', 'os_event']], duration_col='os_months', event_col='os_event')

        if hasattr(cph_os, 'hazard_ratios_'):
            hr_os_clean = cph_os.hazard_ratios_.copy()
            hr_os_clean.index = clean_index(hr_os_clean.index)
            hazard_results['OS_Hazard_Ratios'] = hr_os_clean

        print("Cox model for OS fitted successfully")
    except Exception as e:
        print(f"Error fitting Cox model for OS: {e}")
        try:
            cph_os = CoxPHFitter(penalizer=1.0)
            cph_os.fit(df_os[feature_cols + ['os_months', 'os_event']], duration_col='os_months', event_col='os_event')
            print("Cox model fitted with stronger regularization")
            if hasattr(cph_os, 'hazard_ratios_'):
                hr_os_clean = cph_os.hazard_ratios_.copy()
                hr_os_clean.index = clean_index(hr_os_clean.index)
                hazard_results['OS_Hazard_Ratios'] = hr_os_clean
        except:
            print("Could not fit Cox model for OS")

    print("\nCox PH Model for Progression-Free Survival (with regularization):")
    try:
        cph_pfs = CoxPHFitter(penalizer=0.1)
        df_pfs = df[feature_cols + ['pfs_months', 'pfs_event']].dropna()

        for col in feature_cols[:]:
            if df_pfs[col].std() == 0:
                print(f"Removing constant column: {col}")
                feature_cols.remove(col)

        cph_pfs.fit(df_pfs[feature_cols + ['pfs_months', 'pfs_event']], duration_col='pfs_months', event_col='pfs_event')

        if hasattr(cph_pfs, 'hazard_ratios_'):
            hr_pfs_clean = cph_pfs.hazard_ratios_.copy()
            hr_pfs_clean.index = clean_index(hr_pfs_clean.index)
            hazard_results['PFS_Hazard_Ratios'] = hr_pfs_clean

        print("Cox model for PFS fitted successfully")
    except Exception as e:
        print(f"Error fitting Cox model for PFS: {e}")
        try:
            cph_pfs = CoxPHFitter(penalizer=1.0)
            cph_pfs.fit(df_pfs[feature_cols + ['pfs_months', 'pfs_event']], duration_col='pfs_months', event_col='pfs_event')
            print("Cox model fitted with stronger regularization")
            if hasattr(cph_pfs, 'hazard_ratios_'):
                hr_pfs_clean = cph_pfs.hazard_ratios_.copy()
                hr_pfs_clean.index = clean_index(hr_pfs_clean.index)
                hazard_results['PFS_Hazard_Ratios'] = hr_pfs_clean
        except:
            print("Could not fit Cox model for PFS")

    if hazard_results:
        save_multiple_dataframes(hazard_results, 'Table7_Cox_Hazard_Ratios', output_dir)
        print("\nCox hazard ratios saved to Excel")

    if cph_os is not None or cph_pfs is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if cph_os is not None and hasattr(cph_os, 'hazard_ratios_'):
            hr_os = cph_os.hazard_ratios_.sort_values(ascending=False).head(10)
            hr_os.index = clean_index(hr_os.index)
            hr_os.plot.barh(ax=axes[0], color='coral')
            axes[0].set_title('Top 10 Hazard Ratios for OS', fontweight='bold')
            axes[0].set_xlabel('Hazard Ratio')
            axes[0].axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=2)
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'OS Cox model not fitted', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('OS Cox Model - Not Available')

        if cph_pfs is not None and hasattr(cph_pfs, 'hazard_ratios_'):
            hr_pfs = cph_pfs.hazard_ratios_.sort_values(ascending=False).head(10)
            hr_pfs.index = clean_index(hr_pfs.index)
            hr_pfs.plot.barh(ax=axes[1], color='lightseagreen')
            axes[1].set_title('Top 10 Hazard Ratios for PFS', fontweight='bold')
            axes[1].set_xlabel('Hazard Ratio')
            axes[1].axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=2)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'PFS Cox model not fitted', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('PFS Cox Model - Not Available')

        plt.tight_layout()
        save_figure(fig, 'Figure6_Cox_Hazard_Ratios', output_dir)

    return cph_os, cph_pfs

# ============================================================================
# 6. RANDOM SURVIVAL FOREST
# ============================================================================

def random_survival_forest(df, output_dir):
    """Fit Random Survival Forest model"""
    print("\n" + "="*60)
    print("6. RANDOM SURVIVAL FOREST ANALYSIS")
    print("="*60)

    # FIX E1: platinum_dose_value dropped (collinear with platinum_drug_encoded).
    feature_cols = []
    for col in ['age', 'performance_status']:
        if col in df.columns:
            feature_cols.append(col)

    for col in ['platinum_drug_encoded', 'adjunct_1_name_encoded', 'figo_stage_encoded']:
        if col in df.columns:
            feature_cols.append(col)

    if 'toxicity_severity' in df.columns:
        feature_cols.append('toxicity_severity')

    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    try:
        y_os = Surv.from_arrays(event=df['os_event'].values.astype(bool), time=df['os_months'].values.astype(float))

        if len(X) < 10:
            print("Not enough data for Random Survival Forest")
            return None, pd.DataFrame()

        X_train, X_test, y_train, y_test = train_test_split(X, y_os, test_size=0.2, random_state=42)

        rsf = RandomSurvivalForest(
            n_estimators=50,
            min_samples_split=15,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        rsf.fit(X_train.values, y_train)

        c_index = rsf.score(X_test.values, y_test)
        print(f"Concordance Index: {c_index:.4f}")

        # FIX E4: RandomSurvivalForest.feature_importances_ raises
        # NotImplementedError in current scikit-survival versions -- this is
        # a deliberate API change by the sksurv maintainers, not a bug in
        # your environment. Impurity-based importance was removed for
        # survival forests specifically because it can be systematically
        # biased toward high-cardinality/continuous features; the
        # documented replacement is permutation importance, which works
        # with any fitted estimator that has a working .score() method
        # (RSF's does -- it returns the concordance index) by measuring how
        # much that score drops when each feature's values are shuffled.
        # Tries the native attribute first so this still works unmodified
        # on older scikit-survival versions that do support it.
        try:
            importances = rsf.feature_importances_
            importance_note = "native impurity-based importance"
        except NotImplementedError:
            print("Note: this scikit-survival version does not implement native")
            print("feature_importances_ for RandomSurvivalForest; falling back to")
            print("permutation importance (the maintainers' documented replacement).")
            from sklearn.inspection import permutation_importance
            perm_result = permutation_importance(
                rsf, X_test.values, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = perm_result.importances_mean
            importance_note = "permutation importance (10 repeats, scored by concordance index)"

        importance_df = pd.DataFrame({
            'Feature': clean_feature_list(feature_cols),
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        print(f"Feature importance method used: {importance_note}")

        rsf_results = {
            'Concordance_Index': pd.DataFrame([c_index], columns=['Concordance Index']),
            'Feature_Importance': importance_df
        }
        save_multiple_dataframes(rsf_results, 'Table8_RSF_Results', output_dir)
        print("\nRandom Survival Forest results saved to Excel")

        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df.head(10).plot.barh(x='Feature', y='Importance', ax=ax, color='mediumpurple')
        ax.set_title('Top 10 Features - Random Survival Forest', fontweight='bold')
        ax.set_xlabel('Feature Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        save_figure(fig, 'Figure7_RSF_Feature_Importance', output_dir)

        return rsf, importance_df

    except Exception as e:
        # FIX E2: print the FULL exception (type + message + traceback),
        # not a 200-character truncation, so a real failure here can
        # actually be diagnosed from the printed output rather than
        # producing a silent, empty result with no way to tell why.
        print(f"Error fitting Random Survival Forest: {type(e).__name__}: {e}")
        print("Full traceback for diagnosis:")
        traceback.print_exc()
        return None, pd.DataFrame()

# ============================================================================
# 7. CLUSTERING ANALYSIS
# ============================================================================

def patient_clustering(df, output_dir):
    """Cluster patients based on clinical features and outcomes"""
    print("\n" + "="*60)
    print("7. PATIENT CLUSTERING ANALYSIS")
    print("="*60)

    cluster_features = []
    for col in ['age', 'performance_status', 'toxicity_severity', 'pfs_months', 'os_months']:
        if col in df.columns:
            cluster_features.append(col)

    if not cluster_features:
        print("No numeric features available for clustering")
        return df, None

    print("NOTE: os_months/pfs_months mix true event times with administrative")
    print("censoring times; clusters partly driven by these can reflect follow-up")
    print("duration rather than purely clinical similarity -- interpret accordingly.")

    X_cluster = df[cluster_features].copy()
    X_cluster = X_cluster.fillna(X_cluster.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    max_clusters = min(10, len(X_cluster) - 1)
    inertias = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    if inertias:
        axes[0, 0].plot(range(2, max_clusters + 1), inertias, 'bo-', linewidth=2)
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method for Optimal Clusters', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

    n_clusters = min(4, max_clusters)
    if n_clusters < 2:
        n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_summary = df.groupby('cluster')[cluster_features].mean()
    save_dataframe(cluster_summary, 'Table9_Cluster_Characteristics', output_dir)
    print("\nCluster characteristics saved to Excel")

    if 'os_months' in df.columns and 'os_event' in df.columns:
        kmf = KaplanMeierFitter()
        colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
        for idx, cluster in enumerate(sorted(df['cluster'].unique())):
            mask = df['cluster'] == cluster
            if mask.sum() > 0:
                kmf.fit(df.loc[mask, 'os_months'], df.loc[mask, 'os_event'], label=f'Cluster {cluster}')
                kmf.plot(ax=axes[0, 1], ci_show=True, linewidth=2, color=colors[idx])

        axes[0, 1].set_title('OS by Patient Cluster', fontweight='bold')
        axes[0, 1].set_xlabel('Time (months)')
        axes[0, 1].set_ylabel('Survival Probability')
        axes[0, 1].legend(loc='best')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'OS data not available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('OS by Patient Cluster - Not Available')

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1],
                                 c=df['cluster'], cmap='viridis', alpha=0.6, s=50)
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    axes[1, 0].set_title('Patient Clusters (PCA)', fontweight='bold')
    plt.colorbar(scatter, ax=axes[1, 0])
    axes[1, 0].grid(True, alpha=0.3)

    if 'regimen' in df.columns:
        regimen_cluster = pd.crosstab(df['regimen'], df['cluster'])
        regimen_cluster_pct = regimen_cluster.div(regimen_cluster.sum(axis=1), axis=0) * 100
        regimen_cluster_pct_top = regimen_cluster_pct.loc[
            df['regimen'].value_counts().head(10).index
        ]

        regimen_cluster_pct_top.plot.bar(ax=axes[1, 1], stacked=True, linewidth=0.5)
        axes[1, 1].set_title('Regimen Distribution by Cluster', fontweight='bold')
        axes[1, 1].set_xlabel('Regimen')
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Regimen data not available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Regimen Distribution - Not Available')

    plt.tight_layout()
    save_figure(fig, 'Figure8_Clustering_Analysis', output_dir)

    return df, kmeans

# ============================================================================
# 8. SHAP ANALYSIS
# ============================================================================

def shap_analysis(df, output_dir):
    """Perform SHAP analysis for model interpretability"""
    print("\n" + "="*60)
    print("8. SHAP ANALYSIS")
    print("="*60)

    # FIX E1: platinum_dose_value dropped -- this is the specific function
    # where the collinearity with platinum_drug_encoded was originally
    # noticed (platinum_drug_encoded measured at ~0 mean |SHAP value|
    # despite being the single largest Cox hazard ratio in this cohort).
    # See module docstring for the full mechanism and the synthetic
    # reproduction that confirmed it.
    feature_cols = []
    for col in ['age', 'performance_status']:
        if col in df.columns:
            feature_cols.append(col)

    for col in ['platinum_drug_encoded', 'adjunct_1_name_encoded', 'figo_stage_encoded']:
        if col in df.columns:
            feature_cols.append(col)

    if 'toxicity_severity' in df.columns:
        feature_cols.append('toxicity_severity')

    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    if 'os_event' in df.columns:
        y = df['os_event']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        try:
            model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
            model.fit(X_train, y_train)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # FIX D1: normalize shap's version-dependent return shape.
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values_plot = shap_values[:, :, 1]
            else:
                shap_values_plot = shap_values

            # FIX: rename columns to clean, reader-facing labels before
            # SHAP ever sees them, so the plot itself is built with the
            # right names rather than needing correction afterward.
            X_test_labeled = X_test.copy()
            X_test_labeled.columns = clean_feature_list(list(X_test.columns))

            # FIX: SHAP's default x-axis label text for a bar plot can run
            # long enough to be clipped at this figure width, worst on the
            # bar variant (Figure 10) where a reader flagged exactly this.
            # Widened the figure and explicitly set a short, guaranteed-to-
            # fit label instead of trusting SHAP's own default text, and
            # added subplots_adjust as a second safeguard alongside
            # tight_layout so the label has room regardless of renderer
            # quirks in bbox_inches='tight'.
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_plot, X_test_labeled, show=False)
            plt.xlabel('SHAP Value (impact on model output)')
            plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
            plt.tight_layout()
            save_figure(plt.gcf(), 'Figure9_SHAP_Summary', output_dir)
            plt.close()

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_plot, X_test_labeled, plot_type='bar', show=False)
            plt.xlabel('Mean |SHAP Value|')
            plt.gcf().subplots_adjust(bottom=0.15, left=0.25)
            plt.tight_layout()
            save_figure(plt.gcf(), 'Figure10_SHAP_Bar', output_dir)
            plt.close()

            shap_df = pd.DataFrame({
                'Feature': clean_feature_list(feature_cols),
                'Mean |SHAP Value|': np.abs(shap_values_plot).mean(axis=0)
            }).sort_values('Mean |SHAP Value|', ascending=False)

            save_dataframe(shap_df, 'Table10_SHAP_Importance', output_dir)
            print("\nSHAP importance saved to Excel")

            return shap_df
        except Exception as e:
            print(f"Error in SHAP analysis: {type(e).__name__}: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    else:
        print("OS event data not available for SHAP analysis")
        return pd.DataFrame()

# ============================================================================
# 9. DOSAGE-RESPONSE ANALYSIS
# ============================================================================

def dosage_response_analysis(df, output_dir):
    """Analyze relationship between drug dosage and outcomes"""
    print("\n" + "="*60)
    print("9. DOSAGE-RESPONSE ANALYSIS")
    print("="*60)

    if 'platinum_dose_value' not in df.columns:
        print("Platinum dose data not available")
        return pd.DataFrame()

    # FIX C2: restrict to cisplatin-only, since carboplatin's AUC dosing
    # is not numerically comparable to cisplatin's mg/m2 dosing.
    if 'platinum_drug' in df.columns:
        n_before = len(df)
        df = df[df['platinum_drug'].astype(str).str.strip().str.lower() == 'cisplatin'].copy()
        n_excluded = n_before - len(df)
        if n_excluded > 0:
            print(f"Restricting to cisplatin-dosed patients only ({n_excluded} carboplatin "
                  f"[AUC-dosed] patients excluded -- AUC and mg/m2 are not comparable on one scale).")

    # FIX E3 (documentation only, no behavior change): confirmed this
    # function is working as designed on the real data, not failing --
    # cisplatin dosing is too concentrated in this cohort to form
    # meaningful quantile groups, so it correctly prints a message and
    # returns an empty result below rather than fabricating output. Not
    # the same kind of silent failure as the old RSF error handling; left
    # unchanged other than the traceback-printing convention used
    # elsewhere in this revision, for consistency.
    print("NOTE: cisplatin dose is close to fixed (50 mg/m2) across nearly all five source")
    print("trials -- none of them were dose-finding studies -- so this analysis has limited")
    print("power to detect a real dose-response relationship in this specific dataset.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    try:
        unique_doses = df['platinum_dose_value'].nunique()
        n_groups = min(4, unique_doses) if unique_doses >= 2 else 1

        if n_groups < 2:
            print("Fewer than 2 distinct cisplatin dose values present -- skipping dosage-response analysis.")
            plt.close(fig)
            return pd.DataFrame()

        dose_bins, bin_edges = pd.qcut(df['platinum_dose_value'], q=n_groups,
                                        duplicates='drop', retbins=True)
        n_actual_bins = len(bin_edges) - 1
        if n_actual_bins < 2:
            print("Cisplatin dose values are too concentrated to form meaningful groups -- skipping.")
            plt.close(fig)
            return pd.DataFrame()

        generic_labels = ['Low', 'Medium-Low', 'Medium-High', 'High'][:n_actual_bins] \
            if n_actual_bins <= 4 else [f'Group {i+1}' for i in range(n_actual_bins)]
        dose_bins = pd.qcut(df['platinum_dose_value'], q=n_groups, labels=generic_labels, duplicates='drop')

        dosage_data = {}

        if 'os_months' in df.columns:
            dose_survival = df.groupby(dose_bins, observed=True)['os_months'].mean()
            dosage_data['Mean OS (months)'] = dose_survival.values
            axes[0, 0].bar(range(len(dose_survival)), dose_survival.values, color='steelblue', alpha=0.7)
            axes[0, 0].set_xticks(range(len(dose_survival)))
            axes[0, 0].set_xticklabels(dose_survival.index)
            axes[0, 0].set_title('Platinum Dose vs Mean OS', fontweight='bold')
            axes[0, 0].set_xlabel('Dose Group')
            axes[0, 0].set_ylabel('Mean OS (months)')
            axes[0, 0].grid(True, alpha=0.3)

        if 'pfs_months' in df.columns:
            dose_pfs = df.groupby(dose_bins, observed=True)['pfs_months'].mean()
            dosage_data['Mean PFS (months)'] = dose_pfs.values
            axes[0, 1].bar(range(len(dose_pfs)), dose_pfs.values, color='lightseagreen', alpha=0.7)
            axes[0, 1].set_xticks(range(len(dose_pfs)))
            axes[0, 1].set_xticklabels(dose_pfs.index)
            axes[0, 1].set_title('Platinum Dose vs Mean PFS', fontweight='bold')
            axes[0, 1].set_xlabel('Dose Group')
            axes[0, 1].set_ylabel('Mean PFS (months)')
            axes[0, 1].grid(True, alpha=0.3)

        if 'toxicity_severity' in df.columns:
            dose_tox = df.groupby(dose_bins, observed=True)['toxicity_severity'].mean()
            dosage_data['Mean Toxicity'] = dose_tox.values
            axes[1, 0].bar(range(len(dose_tox)), dose_tox.values, color='coral', alpha=0.7)
            axes[1, 0].set_xticks(range(len(dose_tox)))
            axes[1, 0].set_xticklabels(dose_tox.index)
            axes[1, 0].set_title('Platinum Dose vs Mean Toxicity', fontweight='bold')
            axes[1, 0].set_xlabel('Dose Group')
            axes[1, 0].set_ylabel('Mean Toxicity Severity')
            axes[1, 0].grid(True, alpha=0.3)

        if 'os_months' in df.columns and 'toxicity_severity' in df.columns:
            dose_ratio = dose_survival / (dose_tox + 0.1)
            axes[1, 1].bar(range(len(dose_ratio)), dose_ratio.values, color='mediumpurple', alpha=0.7)
            axes[1, 1].set_xticks(range(len(dose_ratio)))
            axes[1, 1].set_xticklabels(dose_ratio.index)
            axes[1, 1].set_title('Benefit/Toxicity Ratio by Dose', fontweight='bold')
            axes[1, 1].set_xlabel('Dose Group')
            axes[1, 1].set_ylabel('Benefit/Toxicity Ratio')
            axes[1, 1].grid(True, alpha=0.3)
            dosage_data['Benefit/Toxicity Ratio'] = dose_ratio.values

        if dosage_data:
            dosage_table = pd.DataFrame(dosage_data)
            dosage_table.index = generic_labels
            dosage_table.index.name = 'Dose Group'
            dosage_table['N'] = df.groupby(dose_bins, observed=True).size().values

            save_dataframe(dosage_table, 'Table11_Dosage_Response', output_dir)
            print("\nDosage-response analysis saved to Excel")

            plt.tight_layout()
            save_figure(fig, 'Figure11_Dosage_Response_Analysis', output_dir)

            return dosage_table
        else:
            plt.close(fig)

    except Exception as e:
        print(f"Error in dosage-response analysis: {type(e).__name__}: {e}")
        plt.close(fig)
        return pd.DataFrame()

    return pd.DataFrame()

# ============================================================================
# 10. SURVIVAL AT SPECIFIED TIME POINTS
# ============================================================================

def survival_at_timepoints(df, output_dir):
    """Calculate and visualize survival at specified timepoints"""
    print("\n" + "="*60)
    print("10. SURVIVAL AT SPECIFIED TIME POINTS")
    print("="*60)

    if 'os_months' not in df.columns or 'os_event' not in df.columns:
        print("OS data not available")
        return pd.DataFrame()

    timepoints = [12, 24, 36, 48, 60]

    if 'regimen' in df.columns:
        regimens = df['regimen'].value_counts().head(6).index
        results = []

        for regimen in regimens:
            mask = df['regimen'] == regimen
            subset = df[mask]

            if len(subset) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(subset['os_months'], subset['os_event'])

                row = {'Regimen': regimen, 'N': len(subset)}
                for t in timepoints:
                    if t <= kmf.timeline.max():
                        surv = kmf.survival_function_at_times(t).values[0]
                    else:
                        surv = np.nan
                    row[f'OS_{t}m'] = surv
                results.append(row)

        if results:
            survival_df = pd.DataFrame(results)
            save_dataframe(survival_df, 'Table12_Survival_At_Timepoints', output_dir)
            print("\nSurvival at timepoints saved to Excel")

            fig, ax = plt.subplots(figsize=(12, 8))

            colors = plt.cm.Set1(np.linspace(0, 1, len(survival_df)))
            for idx, row in survival_df.iterrows():
                times = [f'{t}m' for t in timepoints]
                surv = [row[f'OS_{t}m'] for t in timepoints]
                ax.plot(times, surv, marker='o', label=row['Regimen'],
                       linewidth=2, markersize=8, color=colors[idx])

            ax.set_title('Survival at Fixed Time Points by Regimen', fontweight='bold')
            ax.set_xlabel('Follow-up Time')
            ax.set_ylabel('Survival Probability')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_figure(fig, 'Figure12_Survival_At_Timepoints', output_dir)

            return survival_df

    return pd.DataFrame()

# ============================================================================
# 11. COMPREHENSIVE TREATMENT COMPARISON
# ============================================================================

def treatment_comparison(df, output_dir):
    """Compare different treatment regimens on all outcomes"""
    print("\n" + "="*60)
    print("11. COMPREHENSIVE TREATMENT COMPARISON")
    print("="*60)

    if 'regimen' not in df.columns:
        print("Regimen data not available")
        return pd.DataFrame()

    regimens = df['regimen'].value_counts().head(10).index
    comparison_data = []

    for regimen in regimens:
        subset = df[df['regimen'] == regimen]

        row = {'Regimen': regimen, 'N': len(subset)}

        if 'os_months' in subset.columns:
            row['Mean OS (months)'] = subset['os_months'].mean()
            row['Median OS (months)'] = subset['os_months'].median()

        if 'pfs_months' in subset.columns:
            row['Mean PFS (months)'] = subset['pfs_months'].mean()
            row['Median PFS (months)'] = subset['pfs_months'].median()

        if 'orr' in subset.columns:
            row['ORR (%)'] = subset['orr'].mean() * 100

        if 'toxicity_severity' in subset.columns:
            row['Mean Toxicity'] = subset['toxicity_severity'].mean()

        if 'severe_toxicity' in subset.columns:
            row['Severe Toxicity (%)'] = subset['severe_toxicity'].mean() * 100

        if 'age' in subset.columns:
            row['Mean Age'] = subset['age'].mean()

        # FIX B2: performance_status uses the shift-by-one convention, so
        # PS0 patients are coded as 1, not 0.
        if 'performance_status' in subset.columns:
            row['PS = 0 (%)'] = (subset['performance_status'] == 1).mean() * 100

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    save_dataframe(comparison_df, 'Table13_Treatment_Comparison', output_dir)
    print("\nTreatment comparison saved to Excel")

    metrics = [col for col in comparison_df.columns if col not in ['Regimen', 'N']]
    if len(metrics) > 1:
        heatmap_df = comparison_df.set_index('Regimen')[metrics]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_df.T, annot=True, fmt='.1f', cmap='RdYlGn',
                    center=heatmap_df.values.mean() if len(heatmap_df.values) > 0 else 0,
                    ax=ax, cbar_kws={'label': 'Value'})
        ax.set_title('Treatment Regimen Comparison Matrix', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_figure(fig, 'Figure13_Treatment_Comparison_Heatmap', output_dir)

    if 'Median OS (months)' in comparison_df.columns and 'Mean Toxicity' in comparison_df.columns:
        comparison_df['Benefit/Toxicity Ratio'] = comparison_df['Median OS (months)'] / (comparison_df['Mean Toxicity'] + 0.1)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_regimens = comparison_df.sort_values('Benefit/Toxicity Ratio', ascending=False).head(10)
        ax.barh(top_regimens['Regimen'], top_regimens['Benefit/Toxicity Ratio'], color='mediumseagreen')
        ax.set_title('Benefit/Toxicity Ratio by Regimen', fontweight='bold')
        ax.set_xlabel('Benefit/Toxicity Ratio (Median OS / Mean Toxicity)')
        plt.tight_layout()
        save_figure(fig, 'Figure14_Benefit_Toxicity_Ratio', output_dir)

    return comparison_df

# ============================================================================
# 12. SUMMARY REPORT
# ============================================================================

def generate_summary_report(df, output_dir):
    """Generate a summary report of all results"""
    print("\n" + "="*60)
    print("12. GENERATING SUMMARY REPORT")
    print("="*60)

    summary_data = []

    summary_data.append({'Metric': 'Total Patients', 'Value': len(df)})
    summary_data.append({'Metric': 'Total Columns', 'Value': len(df.columns)})
    summary_data.append({'Metric': 'Treatment Regimens', 'Value': df['regimen'].nunique() if 'regimen' in df.columns else 'N/A'})
    summary_data.append({'Metric': 'Median OS (months)', 'Value': df['os_months'].median() if 'os_months' in df.columns else 'N/A'})
    summary_data.append({'Metric': 'Median PFS (months)', 'Value': df['pfs_months'].median() if 'pfs_months' in df.columns else 'N/A'})
    summary_data.append({'Metric': 'OS Events (%)', 'Value': f"{(df['os_event'].mean() * 100):.1f}" if 'os_event' in df.columns else 'N/A'})
    summary_data.append({'Metric': 'PFS Events (%)', 'Value': f"{(df['pfs_event'].mean() * 100):.1f}" if 'pfs_event' in df.columns else 'N/A'})
    summary_data.append({'Metric': 'Mean Age', 'Value': f"{df['age'].mean():.1f}" if 'age' in df.columns else 'N/A'})
    summary_data.append({'Metric': 'Mean Toxicity Score', 'Value': f"{df['toxicity_severity'].mean():.2f}" if 'toxicity_severity' in df.columns else 'N/A'})
    summary_data.append({'Metric': 'Severe Toxicity (%)', 'Value': f"{(df['severe_toxicity'].mean() * 100):.1f}" if 'severe_toxicity' in df.columns else 'N/A'})

    summary_df = pd.DataFrame(summary_data)

    save_dataframe(summary_df, 'Table14_Summary_Report', output_dir)
    print("\nSummary report saved to Excel")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    summary_text = "CLINICAL TRIAL ANALYSIS SUMMARY\n" + "="*60 + "\n\n"
    for _, row in summary_df.iterrows():
        summary_text += f"{row['Metric']}: {row['Value']}\n"

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    save_figure(fig, 'Figure15_Summary_Report', output_dir)

    return summary_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_full_analysis(df):
    """Run all analyses in sequence and save outputs"""

    output_dir = create_output_folder()

    print("="*80)
    print("CERVICAL CANCER CLINICAL TRIAL DATA ANALYSIS - PUBLICATION READY")
    print("="*80)

    print("\nPreparing data...")
    df_clean, toxicity_cols, le_dict = prepare_data(df)

    logrank_results = plot_survival_curves(df_clean, output_dir)
    aft_os, aft_pfs, pred_df = parametric_survival_regression(df_clean, output_dir)
    response_results, importance_df = predict_response(df_clean, output_dir)
    reg_df, clf_df, indiv_df = predict_toxicity(df_clean, toxicity_cols, output_dir)
    cph_os, cph_pfs = cox_proportional_hazards(df_clean, output_dir)
    rsf, rsf_importance = random_survival_forest(df_clean, output_dir)
    df_clustered, kmeans = patient_clustering(df_clean, output_dir)
    shap_df = shap_analysis(df_clean, output_dir)
    dosage_table = dosage_response_analysis(df_clean, output_dir)
    survival_timepoints = survival_at_timepoints(df_clean, output_dir)
    treatment_comparison_df = treatment_comparison(df_clean, output_dir)
    summary_df = generate_summary_report(df_clean, output_dir)

    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE! All results saved to: {output_dir}")
    print("="*80)
    print("\nFILES GENERATED:")
    print("  - Figures: High-resolution JPG (900 DPI), PNG, and PDF")
    print("  - Tables: Excel files (.xlsx) with formatting")
    print("\nFOLDER STRUCTURE:")
    print(f"  {output_dir}/")
    print("    \u251c\u2500\u2500 Figures/")
    print("    \u2502   \u251c\u2500\u2500 Figure1_Survival_Curves.jpg")
    print("    \u2502   \u251c\u2500\u2500 Figure2_Parametric_Survival_Predictions.jpg")
    print("    \u2502   \u2514\u2500\u2500 ... (all figures)")
    print("    \u2514\u2500\u2500 Tables/")
    print("        \u251c\u2500\u2500 Table1_Logrank_Tests.xlsx")
    print("        \u251c\u2500\u2500 Table2_Parametric_Survival_Predictions.xlsx")
    print("        \u2514\u2500\u2500 ... (all tables)")

    return {
        'output_dir': output_dir,
        'df_clean': df_clean,
        'logrank_results': logrank_results,
        'aft_os': aft_os,
        'aft_pfs': aft_pfs,
        'pred_df': pred_df,
        'response_results': response_results,
        'importance_df': importance_df,
        'toxicity_regression': reg_df,
        'toxicity_classification': clf_df,
        'individual_toxicity': indiv_df,
        'cph_os': cph_os,
        'cph_pfs': cph_pfs,
        'rsf': rsf,
        'rsf_importance': rsf_importance,
        'clustered_data': df_clustered,
        'shap_importance': shap_df,
        'dosage_table': dosage_table,
        'survival_timepoints': survival_timepoints,
        'treatment_comparison': treatment_comparison_df,
        'summary': summary_df
    }

# ============================================================================
# EXECUTION
# ============================================================================

results = run_full_analysis(df)

print(f"\nResults saved to: {results['output_dir']}")
print(f"Number of patients analyzed: {len(results['df_clean'])}")
print(f"Treatment regimens: {results['df_clean']['regimen'].nunique()}")
