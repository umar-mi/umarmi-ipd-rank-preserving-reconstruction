#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MONK 2009 -- SYNTHETIC IPD GENERATION PIPELINE (CORRECTED)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Source: Monk BJ, Sill MW, McMeekin DS, et al. Phase III Trial of Four
# Cisplatin-Containing Doublet Combinations in Stage IVB, Recurrent, or
# Persistent Cervical Carcinoma: A Gynecologic Oncology Group Study.
# J Clin Oncol. 2009;27(28):4649-4655.
#
# CHANGE LOG relative to the original script (see chat review for full detail)
# ------------------------------------------------------------------------
#  [Bug fixes -- unambiguous, evidence-based]
#  B1. PFS x histology adjustment was DIVIDING by the histology weight
#      (lengthened PFS for "worse" histologies). Now MULTIPLIES.
#  B2. `target_pfsr` (0.484/0.426/0.482/0.478) is actually Table 4's
#      "Stable Disease" percentage, not a PFS rate. Response category
#      (CR/PR/SD/PD) is now built directly and exactly from Table 4's
#      reported counts (cr_count, pr_count, new sd_count), instead of via
#      a (target_pfsr - responders) subtraction that under-counted SD by
#      roughly half.
#  B3. No target existed for median PFS in months. Added `target_pfs`
#      (5.82 / 3.98 / 4.70 / 4.57 months, Results/"Response and Survival"
#      section) and a rescale-to-target step mirroring the existing OS
#      calibration.
#  B4. PFS event count used a flat `int(0.92*n)` approximation. Replaced
#      with the trial's exact arm-specific "alive/progression-free" counts
#      from the Figure 2 caption table (7/5/8/9), analogous to how
#      `alive_count` is already handled for OS.
#  B5. `os_ps_weight`, `os_orr_weight`, `os_hazard_ratio`, and the PFI-based
#      `disease_hr` were extracted from trial_config but never applied to
#      the generated survival TIMES (only to categorical nudges). PS is
#      wired in now using the trial's own reported multivariate HR (1.799,
#      PS1 vs PS0; Fig 3A). `os_hazard_ratio` is intentionally NOT
#      reapplied multiplicatively (see note below) to avoid double-counting
#      the arm effect that is already encoded via the arm-specific digitized
#      KM curve + median-rescaling step; it is retained as a printed QC
#      cross-check only.
#  B6. The 4-way Cholesky/copula `correlated_binary()` draw produced mostly
#      discarded output (PFSR and PS_base were overwritten/unused). Replaced
#      with a transparent latent-propensity-index + exact-count selection
#      approach (same pattern already used for toxicity assignment
#      elsewhere in the script), which actually uses the self-proposed
#      correlation weights to decide WHICH patients respond / have stable
#      disease, while hitting Table 4's counts exactly.
#
#  [Data-mapping corrections]
#  C1. `disease_nature` / `figo_stage` were inconsistent: Table 1's mutually
#      exclusive "Stage" row (IVB / Persistent / Recurrent) had its IVB
#      patients folded into "Recurrent", while a SEPARATE figo_stage field
#      simultaneously marked ALL patients as IVB. Fixed to a single,
#      internally consistent 3-way split matching Table 1 exactly, with
#      figo_stage = "IVB" only for the Advanced subgroup and "Unknown" for
#      Persistent/Recurrent (their initial FIGO stage is not reported).
#  C2. `ethnicity_base` mixes GOG's "Race" categories with Hispanic/Filipino
#      placeholders fixed at 0, discarding the paper's real Hispanic-origin
#      counts. NOT restructured here (would add a new output column, which
#      you asked me to avoid for merge compatibility) -- flagged with a
#      prominent comment instead. See chat message for detail.
#  C3. Table 1's "Prior primary CISPLATIN AND radiation" (70/79/72/81) was
#      coded as "radiation_only". Recoded as "chemoradiotherapy" (already an
#      existing category), which also now correctly flags prior platinum
#      exposure in derive_prior_treatments().
#  C4. Paclitaxel infusion duration was hardcoded to 0h; the paper states
#      "paclitaxel 135 mg/m2 OVER 24 HOURS" explicitly. Fixed to 24 for the
#      PC arm.
#
#  [Hygiene / dead code removed]
#  D1. DRUG_monk_b/c/d now define explicit Adjunct_Drug_2 'No Drug'
#      placeholders (previously only DRUG_monk_a did; b/c/d relied on a
#      silent .get()-returns-None fallback).
#  D2. Arm-specific "latent nudges" (monk_a alopecia +1.0, monk_c
#      leucopenia/neutropenia -1.2/-1.5) removed: alopecia's grade>=3 target
#      count is genuinely 0 in this trial (the nudge could never surface),
#      and the GC arm's lower myelosuppression is already fully captured by
#      its (correct) target counts from the xlsx -- the nudge was redundant
#      and only obscured that.
#  D3. Removed additional dead trial_config parameters found while
#      implementing the fixes above: shape_k, initial_probabilities,
#      asian_weight, black_weight, american_indian_weight, age_latent_risk,
#      ps_vs_radiation, orr_vs_pfsr -- none of these were ever referenced
#      in generate_clinical_data() even in the original script.
#  D4. toxicity_base categories Hematuria / Vascular / Weight loss remain at
#      0 for Monk (absent from the appendix table) -- expected, not a bug;
#      kept for cross-trial harmonization with the other 4 trials.
#
#  [Evaluation table]
#  E.  Added rows for median PFS (months) and Tumor Response (CR/PR/SD/PD),
#      which were previously unchecked -- this is exactly what let B1-B3
#      go undetected before.
#
# A full table of every SELF-PROPOSED (uncited) weight the pipeline now
# uses is built and printed/exported at the very end of this script
# (`weight_reference_table`), so you can attach literature citations later.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Importing Libraries
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from collections import Counter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generating Trial Specific Dictionary
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MONK_A_NUMBER = 103
MONK_B_NUMBER = 108
MONK_C_NUMBER = 112
MONK_D_NUMBER = 111

SEED = 42

#=======================================================
# LITERATURE- AND TRIAL-DERIVED EFFECT SIZES (CITED)
#=======================================================
# Directly reported in Monk et al. 2009 -- no external citation needed.
PS_HAZARD_RATIO_OS = 1.799   # PS 1 vs PS 0, OS, multivariate Cox PH model, Fig 3A

# NOT reported in Monk et al. 2009 -- self-proposed placeholders that DO
# need an external literature citation. Listed again in the weight
# reference table generated at the end of this script.
ORR_OS_HR_ASSUMED = 0.70          # assumed OS benefit, responders (CR/PR) vs non-responders
PFS_RESPONSE_HR_ASSUMED = 0.65    # assumed PFS benefit, responders (CR/PR) vs non-responders


def hr_to_time_multiplier(hazard_ratio, weight=1.0):
    """
    Convert a Cox-model hazard ratio into an accelerated-failure-time (AFT)
    style multiplicative factor on survival TIME (as opposed to hazard).

    This equivalence (time_multiplier = HR^-1) is exact under a Weibull
    proportional-hazards model and is used here as a practical working
    approximation for other baseline hazard shapes -- a standard
    simplification in synthetic-IPD work, but worth stating explicitly
    since it is an approximation, not an identity.

    `weight` is a damping/tuning exponent: weight=1 applies the hazard
    ratio as reported; weight=0 applies no effect at all; weight>1
    exaggerates it. This lets a self-proposed "how strongly do I trust /
    want to express this effect" judgment sit on top of a cited base HR
    without changing the cited number itself.
    """
    hazard_ratio = np.asarray(hazard_ratio, dtype=float)
    return hazard_ratio ** (-float(weight))


#=======================================================
# DEFINING BASE LISTS
#=======================================================

histology_base = ["Squamous Cell Carcinoma", "Adenocarcinoma", "Adenosquamous", "Mucinous Adenocarcinoma",
                  "Clear Cell", "Endometrioid", "Villoglandular", "Undifferentiated Carcinoma", "Not Specified"]

# NOTE (C2): Table 1 in Monk et al. 2009 reports "Race" (White/Black/
# Asian-Pacific/American Indian/Unspecified) as a SEPARATE row from
# "Ethnicity" (Hispanic/Non-Hispanic/Unknown). The values populated into
# this list below are Table 1's RACE row -- "Hispanic"/"Filipino" are kept
# as placeholder categories (always 0 for Monk) so the column list is
# reusable across the 5 trials, but the paper's real Hispanic-origin counts
# (16/10/20/19) are NOT captured anywhere in the output, even though the
# paper's own multivariate model flags Hispanic origin as OS-prognostic
# (HR 0.677, Fig 3A). Left as-is deliberately, since adding it as a
# genuinely separate field would add a new output column -- flagging here
# per your instruction not to change the merged dataset's column count.
ethnicity_base = ["White", "Black", "Asian", "American Indian", "Hispanic", "Filipino", "Unspecified"]

# FIX C1: third category renamed from "Non-specified" to "Advanced (IVB)" to
# match Table 1's actual "Stage" row (IVB / Persistent / Recurrent is a
# mutually-exclusive 3-way split of the whole cohort, not a 2-way split with
# IVB folded into Recurrent). "Advanced (IVB)" is also literally the
# reference category used in Fig 3B ("All hazard ratios are relative to
# patients with advanced disease").
disease_nature = ["Persistent", "Recurrent", "Advanced (IVB)"]

figo_stage = ["I", "II", "III", "IVA", "IVB", "Unknown"]

grade_base = ["Grade 1", "Grade 2", "Grade 3", "Grade Unspecified"]

death_cause_base = ["Treatment", "Disease", "Other/Unknown"]

dead_alive = ["Alive", "Dead", "Unspecified"]

performance_status = [1, 2, 3, 4]

drug_effect_base = ["orr_mult", "pfs_mult", "ps_shift", "os_mult"]

prior_treatment_base = [
    "none",                        # No prior treatment
    "radiation_only",               # Radiation only
    "chemotherapy_only",            # Chemo only (prior)
    "surgery_only",                 # Surgery only
    "surgery_plus_radiation",       # Surgery + radiation
    "radiation_plus_chemotherapy",  # Radiation + chemo
    "surgery_radiation_chemotherapy",  # Surgery + radiation + chemo
    "prior_chemo_platinum",         # Prior platinum chemo (specific)
    "prior_cisplatin",              # Prior cisplatin
    "chemoradiotherapy",            # Concurrent chemoradiotherapy
    "surgery_intent_curative",      # Curative surgery
    "surgery_intent_palliative",    # Palliative surgery
    "unknown"                       # Unknown treatment info
]

toxicity_base = ['Leucopenia', 'Neutropenia', 'Thrombocytopenia', 'Anemia', 'Other hematologic',
                 'Allergic reactions', 'Inner ear/hearing', 'Other auditory', 'Thrombosis embolism',
                 'Cardiac left ventricular function', 'Other cardiovascular', 'Fatigue',
                 'Other constitutional', 'Alopecia', 'Dermatologic', 'Nausea/vomiting', 'Stomatitis',
                 'Other Gastrointestinal', 'Creatinine', 'Hematuria', 'Other genitourinary/renal',
                 'Hemorrhage', 'Hepatic', 'Febrile with neutropenia', 'Infection without neutropenia',
                 'Other infection/fever', 'Lymphatics', 'Metabolic', 'Musculoskeletal', 'Peripheral neuropathy',
                 'Other neurological', 'Ocular/visual', 'Pain', 'Pulmonary', 'Vascular', 'Weight loss', 'Sexual']
# NOTE (D4): Hematuria, Vascular, Weight loss will always come out as 0 for
# Monk -- they simply are not among the 40 adverse-event categories reported
# in the trial's appendix table. Retained here (not removed) purely so the
# column stays available for harmonization with the other 4 trials.

#===========================================
# DRUG COMBINATION
#===========================================

# ============================================================================
# MONK REGIMENS
# ============================================================================

DRUG_monk_a = {
    # Platinum agent
    "Platinum_Drug": "cisplatin",
    "Platinum_Dose_Value": 50,
    "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based",
    "Platinum_Infusion_Duration_Hours": 1,  # not explicitly stated for cisplatin in this paper; 1h with hydration is standard practice, not cited
    "Platinum_Days_Number": 1,
    "Platinum_Day": 2,

    # Adjunct drug 1
    "Adjunct_Drug_1": "paclitaxel",
    "Adjunct_Drug_1_Dose_Value": 135,
    "Adjunct_Drug_1_Dose_Unit": "mg/m2",
    "Adjunct_Drug_1_Dose_Method": "BSA-based",
    # FIX C4: paper explicitly states "paclitaxel 135 mg/m2 OVER 24 HOURS"
    # (Abstract and Treatment section) -- was hardcoded to 0.
    "Adjunct_Drug_1_Infusion_Duration_Hours": 24,
    "Adjunct_Drug_1_Days_Number": 1,
    "Adjunct_Drug_1_Days": 1,
    "Adjunct_Drug_1_Same_Day_As_Platinum": 0,  # paclitaxel day 1, cisplatin day 2

    # Adjunct drug 2 (none in this regimen)
    "Adjunct_Drug_2": 'No Drug',
    "Adjunct_Drug_2_Dose_Value": 0,
    "Adjunct_Drug_2_Dose_Unit": 'No Drug',
    "Adjunct_Drug_2_Dose_Method": 'No Drug',
    "Adjunct_Drug_2_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_2_Days_Number": 0,
    "Adjunct_Drug_2_Days": 0,
    "Adjunct_Drug_2_Same_Day_As_Platinum": 0,

    # Adjunct drug 3
    "Adjunct_Drug_3": 'No Drug',
    "Adjunct_Drug_3_Dose_Value": 0,
    "Adjunct_Drug_3_Dose_Unit": 'No Drug',
    "Adjunct_Drug_3_Dose_Method": 'No Drug',
    "Adjunct_Drug_3_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_3_Days_Number": 0,
    "Adjunct_Drug_3_Days": 0,
    "Adjunct_Drug_3_Same_Day_As_Platinum": 0,

    # Supportive / prophylactic drugs
    "Supportive_Drug_1": 'No Drug', "Supportive_Drug_1_Dose_Value": 0, "Supportive_Drug_1_Dose_Unit": 'No Drug',
    "Supportive_Drug_1_Route": 'No Drug', "Supportive_Drug_1_Start_Day": 0, "Supportive_Drug_1_Duration_Value": 0,
    "Supportive_Drug_1_Duration_Unit": 'No Drug', "Supportive_Drug_1_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_1_Frequency_Interval_Hours": 0, "Supportive_Drug_1_Frequency_Description": 'No Drug',
    "Supportive_Drug_1_Dose_Is_Range": 0, "Supportive_Drug_1_Dose_Min": 0, "Supportive_Drug_1_Dose_Max": 0,

    "Supportive_Drug_2": 'No Drug', "Supportive_Drug_2_Dose_Value": 0, "Supportive_Drug_2_Dose_Unit": 'No Drug',
    "Supportive_Drug_2_Route": 'No Drug', "Supportive_Drug_2_Start_Day": 0, "Supportive_Drug_2_Duration_Value": 0,
    "Supportive_Drug_2_Duration_Unit": 'No Drug', "Supportive_Drug_2_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_2_Frequency_Interval_Hours": 0, "Supportive_Drug_2_Frequency_Description": 'No Drug',
    "Supportive_Drug_2_Dose_Is_Range": 0, "Supportive_Drug_2_Dose_Min": 0, "Supportive_Drug_2_Dose_Max": 0,

    "Supportive_Drug_3": 'No Drug', "Supportive_Drug_3_Dose_Value": 0, "Supportive_Drug_3_Dose_Unit": 'No Drug',
    "Supportive_Drug_3_Route": 'No Drug', "Supportive_Drug_3_Start_Day": 0, "Supportive_Drug_3_Duration_Value": 0,
    "Supportive_Drug_3_Duration_Unit": 'No Drug', "Supportive_Drug_3_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_3_Frequency_Interval_Hours": 0, "Supportive_Drug_3_Frequency_Description": 'No Drug',
    "Supportive_Drug_3_Dose_Is_Range": 0, "Supportive_Drug_3_Dose_Min": 0, "Supportive_Drug_3_Dose_Max": 0,

    # Cycle length and metadata
    "Cycle_Length_Days": 21
}

DRUG_monk_b = {
    # Platinum agent
    "Platinum_Drug": "cisplatin",
    "Platinum_Dose_Value": 50,
    "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based",
    "Platinum_Infusion_Duration_Hours": 1,  # not explicitly stated; 1h with hydration is standard practice, not cited
    "Platinum_Days_Number": 1,
    "Platinum_Day": 1,

    # Adjunct drug 1
    "Adjunct_Drug_1": "vinorelbine",
    "Adjunct_Drug_1_Dose_Value": 30,
    "Adjunct_Drug_1_Dose_Unit": "mg/m2",
    "Adjunct_Drug_1_Dose_Method": "BSA-based",
    "Adjunct_Drug_1_Infusion_Duration_Hours": 0.17,  # not explicitly stated; ~10 min standard IV vinorelbine administration, not cited
    "Adjunct_Drug_1_Days_Number": 2,
    "Adjunct_Drug_1_Days": (1, 8),
    "Adjunct_Drug_1_Same_Day_As_Platinum": 1,

    # FIX D1: explicit Adjunct_Drug_2 placeholder (previously omitted; relied
    # on a silent .get() -> None -> 'No Drug' fallback in uniform_column()).
    "Adjunct_Drug_2": 'No Drug', "Adjunct_Drug_2_Dose_Value": 0, "Adjunct_Drug_2_Dose_Unit": 'No Drug',
    "Adjunct_Drug_2_Dose_Method": 'No Drug', "Adjunct_Drug_2_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_2_Days_Number": 0, "Adjunct_Drug_2_Days": 0, "Adjunct_Drug_2_Same_Day_As_Platinum": 0,

    # Adjunct drug 3
    "Adjunct_Drug_3": 'No Drug', "Adjunct_Drug_3_Dose_Value": 0, "Adjunct_Drug_3_Dose_Unit": 'No Drug',
    "Adjunct_Drug_3_Dose_Method": 'No Drug', "Adjunct_Drug_3_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_3_Days_Number": 0, "Adjunct_Drug_3_Days": 0, "Adjunct_Drug_3_Same_Day_As_Platinum": 0,

    # Supportive / prophylactic drugs
    "Supportive_Drug_1": 'No Drug', "Supportive_Drug_1_Dose_Value": 0, "Supportive_Drug_1_Dose_Unit": 'No Drug',
    "Supportive_Drug_1_Route": 'No Drug', "Supportive_Drug_1_Start_Day": 0, "Supportive_Drug_1_Duration_Value": 0,
    "Supportive_Drug_1_Duration_Unit": 'No Drug', "Supportive_Drug_1_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_1_Frequency_Interval_Hours": 0, "Supportive_Drug_1_Frequency_Description": 'No Drug',
    "Supportive_Drug_1_Dose_Is_Range": 0, "Supportive_Drug_1_Dose_Min": 0, "Supportive_Drug_1_Dose_Max": 0,

    "Supportive_Drug_2": 'No Drug', "Supportive_Drug_2_Dose_Value": 0, "Supportive_Drug_2_Dose_Unit": 'No Drug',
    "Supportive_Drug_2_Route": 'No Drug', "Supportive_Drug_2_Start_Day": 0, "Supportive_Drug_2_Duration_Value": 0,
    "Supportive_Drug_2_Duration_Unit": 'No Drug', "Supportive_Drug_2_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_2_Frequency_Interval_Hours": 0, "Supportive_Drug_2_Frequency_Description": 'No Drug',
    "Supportive_Drug_2_Dose_Is_Range": 0, "Supportive_Drug_2_Dose_Min": 0, "Supportive_Drug_2_Dose_Max": 0,

    "Supportive_Drug_3": 'No Drug', "Supportive_Drug_3_Dose_Value": 0, "Supportive_Drug_3_Dose_Unit": 'No Drug',
    "Supportive_Drug_3_Route": 'No Drug', "Supportive_Drug_3_Start_Day": 0, "Supportive_Drug_3_Duration_Value": 0,
    "Supportive_Drug_3_Duration_Unit": 'No Drug', "Supportive_Drug_3_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_3_Frequency_Interval_Hours": 0, "Supportive_Drug_3_Frequency_Description": 'No Drug',
    "Supportive_Drug_3_Dose_Is_Range": 0, "Supportive_Drug_3_Dose_Min": 0, "Supportive_Drug_3_Dose_Max": 0,

    # Cycle length and metadata
    "Cycle_Length_Days": 21
}

DRUG_monk_c = {
    # Platinum agent
    "Platinum_Drug": "cisplatin",
    "Platinum_Dose_Value": 50,
    "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based",
    "Platinum_Infusion_Duration_Hours": 1,  # not explicitly stated; 1h with hydration is standard practice, not cited
    "Platinum_Days_Number": 1,
    "Platinum_Day": 1,

    # Adjunct drug 1
    "Adjunct_Drug_1": "gemcitabine",
    "Adjunct_Drug_1_Dose_Value": 1000,
    "Adjunct_Drug_1_Dose_Unit": "mg/m2",
    "Adjunct_Drug_1_Dose_Method": "BSA-based",
    "Adjunct_Drug_1_Infusion_Duration_Hours": 0.5,  # not explicitly stated; 30 min standard gemcitabine infusion, not cited
    "Adjunct_Drug_1_Days_Number": 2,
    "Adjunct_Drug_1_Days": (1, 8),
    "Adjunct_Drug_1_Same_Day_As_Platinum": 1,

    # FIX D1: explicit Adjunct_Drug_2 placeholder
    "Adjunct_Drug_2": 'No Drug', "Adjunct_Drug_2_Dose_Value": 0, "Adjunct_Drug_2_Dose_Unit": 'No Drug',
    "Adjunct_Drug_2_Dose_Method": 'No Drug', "Adjunct_Drug_2_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_2_Days_Number": 0, "Adjunct_Drug_2_Days": 0, "Adjunct_Drug_2_Same_Day_As_Platinum": 0,

    # Adjunct drug 3
    "Adjunct_Drug_3": 'No Drug', "Adjunct_Drug_3_Dose_Value": 0, "Adjunct_Drug_3_Dose_Unit": 'No Drug',
    "Adjunct_Drug_3_Dose_Method": 'No Drug', "Adjunct_Drug_3_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_3_Days_Number": 0, "Adjunct_Drug_3_Days": 0, "Adjunct_Drug_3_Same_Day_As_Platinum": 0,

    # Supportive / prophylactic drugs
    "Supportive_Drug_1": 'No Drug', "Supportive_Drug_1_Dose_Value": 0, "Supportive_Drug_1_Dose_Unit": 'No Drug',
    "Supportive_Drug_1_Route": 'No Drug', "Supportive_Drug_1_Start_Day": 0, "Supportive_Drug_1_Duration_Value": 0,
    "Supportive_Drug_1_Duration_Unit": 'No Drug', "Supportive_Drug_1_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_1_Frequency_Interval_Hours": 0, "Supportive_Drug_1_Frequency_Description": 'No Drug',
    "Supportive_Drug_1_Dose_Is_Range": 0, "Supportive_Drug_1_Dose_Min": 0, "Supportive_Drug_1_Dose_Max": 0,

    "Supportive_Drug_2": 'No Drug', "Supportive_Drug_2_Dose_Value": 0, "Supportive_Drug_2_Dose_Unit": 'No Drug',
    "Supportive_Drug_2_Route": 'No Drug', "Supportive_Drug_2_Start_Day": 0, "Supportive_Drug_2_Duration_Value": 0,
    "Supportive_Drug_2_Duration_Unit": 'No Drug', "Supportive_Drug_2_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_2_Frequency_Interval_Hours": 0, "Supportive_Drug_2_Frequency_Description": 'No Drug',
    "Supportive_Drug_2_Dose_Is_Range": 0, "Supportive_Drug_2_Dose_Min": 0, "Supportive_Drug_2_Dose_Max": 0,

    "Supportive_Drug_3": 'No Drug', "Supportive_Drug_3_Dose_Value": 0, "Supportive_Drug_3_Dose_Unit": 'No Drug',
    "Supportive_Drug_3_Route": 'No Drug', "Supportive_Drug_3_Start_Day": 0, "Supportive_Drug_3_Duration_Value": 0,
    "Supportive_Drug_3_Duration_Unit": 'No Drug', "Supportive_Drug_3_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_3_Frequency_Interval_Hours": 0, "Supportive_Drug_3_Frequency_Description": 'No Drug',
    "Supportive_Drug_3_Dose_Is_Range": 0, "Supportive_Drug_3_Dose_Min": 0, "Supportive_Drug_3_Dose_Max": 0,

    # Cycle length and metadata
    "Cycle_Length_Days": 21
}

DRUG_monk_d = {
    # Platinum agent
    "Platinum_Drug": "cisplatin",
    "Platinum_Dose_Value": 50,
    "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based",
    "Platinum_Infusion_Duration_Hours": 1,  # not explicitly stated; 1h with hydration is standard practice, not cited
    "Platinum_Days_Number": 1,
    "Platinum_Day": 1,

    # Adjunct drug 1
    "Adjunct_Drug_1": "topotecan",
    "Adjunct_Drug_1_Dose_Value": 0.75,
    "Adjunct_Drug_1_Dose_Unit": "mg/m2",
    "Adjunct_Drug_1_Dose_Method": "BSA-based",
    "Adjunct_Drug_1_Infusion_Duration_Hours": 0.5,  # not explicitly stated; 30 min standard topotecan infusion, not cited
    "Adjunct_Drug_1_Days_Number": 3,
    "Adjunct_Drug_1_Days": (1, 2, 3),
    "Adjunct_Drug_1_Same_Day_As_Platinum": 1,

    # FIX D1: explicit Adjunct_Drug_2 placeholder
    "Adjunct_Drug_2": 'No Drug', "Adjunct_Drug_2_Dose_Value": 0, "Adjunct_Drug_2_Dose_Unit": 'No Drug',
    "Adjunct_Drug_2_Dose_Method": 'No Drug', "Adjunct_Drug_2_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_2_Days_Number": 0, "Adjunct_Drug_2_Days": 0, "Adjunct_Drug_2_Same_Day_As_Platinum": 0,

    # Adjunct drug 3
    "Adjunct_Drug_3": 'No Drug', "Adjunct_Drug_3_Dose_Value": 0, "Adjunct_Drug_3_Dose_Unit": 'No Drug',
    "Adjunct_Drug_3_Dose_Method": 'No Drug', "Adjunct_Drug_3_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_3_Days_Number": 0, "Adjunct_Drug_3_Days": 0, "Adjunct_Drug_3_Same_Day_As_Platinum": 0,

    # Supportive / prophylactic drugs
    "Supportive_Drug_1": 'No Drug', "Supportive_Drug_1_Dose_Value": 0, "Supportive_Drug_1_Dose_Unit": 'No Drug',
    "Supportive_Drug_1_Route": 'No Drug', "Supportive_Drug_1_Start_Day": 0, "Supportive_Drug_1_Duration_Value": 0,
    "Supportive_Drug_1_Duration_Unit": 'No Drug', "Supportive_Drug_1_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_1_Frequency_Interval_Hours": 0, "Supportive_Drug_1_Frequency_Description": 'No Drug',
    "Supportive_Drug_1_Dose_Is_Range": 0, "Supportive_Drug_1_Dose_Min": 0, "Supportive_Drug_1_Dose_Max": 0,

    "Supportive_Drug_2": 'No Drug', "Supportive_Drug_2_Dose_Value": 0, "Supportive_Drug_2_Dose_Unit": 'No Drug',
    "Supportive_Drug_2_Route": 'No Drug', "Supportive_Drug_2_Start_Day": 0, "Supportive_Drug_2_Duration_Value": 0,
    "Supportive_Drug_2_Duration_Unit": 'No Drug', "Supportive_Drug_2_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_2_Frequency_Interval_Hours": 0, "Supportive_Drug_2_Frequency_Description": 'No Drug',
    "Supportive_Drug_2_Dose_Is_Range": 0, "Supportive_Drug_2_Dose_Min": 0, "Supportive_Drug_2_Dose_Max": 0,

    "Supportive_Drug_3": 'No Drug', "Supportive_Drug_3_Dose_Value": 0, "Supportive_Drug_3_Dose_Unit": 'No Drug',
    "Supportive_Drug_3_Route": 'No Drug', "Supportive_Drug_3_Start_Day": 0, "Supportive_Drug_3_Duration_Value": 0,
    "Supportive_Drug_3_Duration_Unit": 'No Drug', "Supportive_Drug_3_Frequency_Times_Per_Day": 'No Drug',
    "Supportive_Drug_3_Frequency_Interval_Hours": 0, "Supportive_Drug_3_Frequency_Description": 'No Drug',
    "Supportive_Drug_3_Dose_Is_Range": 0, "Supportive_Drug_3_Dose_Min": 0, "Supportive_Drug_3_Dose_Max": 0,

    # Cycle length and metadata
    "Cycle_Length_Days": 21
}
#=====================================================================================
# OVERALL SURVIVAL POINTS EXTRACTED FROM KAPLAN MEIER CURVE USING WEBPLOTDIGITIZER
# (unchanged from your original digitization -- I have no basis to re-extract these)
#=====================================================================================
monk_a_OS_raw = {
    0.575286156: 0.994540121, 0.957028946: 0.991831547, 1.354886295: 0.983592056,
    1.696245234: 0.974297408, 1.976013408: 0.958540659, 2.358221962: 0.943735159,
    2.70824032: 0.938980576, 3.12189997: 0.933320219, 3.490127795: 0.927625327,
    3.599783445: 0.911624413, 3.695792637: 0.896052337, 3.944663967: 0.875110848,
    4.259084837: 0.858768825, 4.651538886: 0.853198365, 4.800732058: 0.832986832,
    4.907459799: 0.814363562, 5.035179719: 0.801193398, 5.207448544: 0.787389194,
    5.354593229: 0.76531475, 5.482380208: 0.750402933, 5.880266276: 0.741417559,
    6.179480935: 0.725370163, 6.648089365: 0.705945659, 6.692372368: 0.712217655,
    7.074246359: 0.706101496, 7.433505775: 0.697336116, 7.716136987: 0.672775474,
    8.09344293: 0.66730168, 8.475334415: 0.660731176, 8.645511019: 0.646199923,
    8.974624665: 0.633720509, 9.259096691: 0.624283164, 9.621743595: 0.62193719,
    9.929669717: 0.609078184, 10.03659455: 0.585335964, 10.25975717: 0.57130692,
    10.61904957: 0.561684775, 10.80142757: 0.54492292, 11.15158369: 0.536590373,
    11.49121092: 0.526383169, 11.67733808: 0.510580479, 12.04317404: 0.50800748,
    12.42483811: 0.507343456, 12.82936866: 0.502786091, 13.15696662: 0.490339062,
    13.26658329: 0.475350687, 13.63481861: 0.469461075, 13.94287885: 0.45311876,
    14.26144101: 0.439351551, 14.65376153: 0.437249256, 15.02207682: 0.42928264,
    15.44024104: 0.424628541, 15.77675318: 0.416660467, 16.20061228: 0.405697718,
    16.56333091: 0.401488932, 17.01960468: 0.390378383, 17.35040945: 0.373310158,
    17.68682412: 0.367873433, 18.09126596: 0.365620245, 18.47306123: 0.361548637,
    18.51800199: 0.350737275, 18.90843321: 0.342637534, 19.30091408: 0.336370412,
    19.49679236: 0.322976201, 19.74269326: 0.308383512, 20.12888939: 0.308011775,
    20.38379349: 0.295585753, 20.76552972: 0.293047558, 21.14727251: 0.290338984,
    21.27498368: 0.277395993, 21.65664775: 0.27673197, 22.03828558: 0.276749463,
    22.41992341: 0.276766957, 22.73808614: 0.27337395, 22.98867653: 0.254984446,
    23.31159249: 0.246139512, 23.69317784: 0.247520039, 24.07484191: 0.246856016,
    24.45642726: 0.248236543, 24.83807383: 0.248026864, 25.21977289: 0.246454152,
    25.60148069: 0.244654267, 25.85645477: 0.230410866, 26.17476838: 0.223099137,
    26.42994581: 0.20357398, 26.81159239: 0.203364301, 27.19323022: 0.203381795,
    27.57492053: 0.202036254, 27.79817236: 0.185690052, 27.82453854: 0.161701822,
    30.61217572: 0.157769535, 31.93450921: 0.157830148, 32.55431806: 0.145016975,
    32.72985642: 0.128895758, 33.11155548: 0.127323045, 33.49319331: 0.127340539,
    33.87483113: 0.127358032, 34.25646896: 0.127375526, 34.60716588: 0.126997211,
    34.96284743: 0.12715708, 28.33185025: 0.158889432, 28.75541068: 0.158083765,
    29.33294977: 0.158110238, 29.98749407: 0.158140241, 31.29658266: 0.158200247,
    32.54791735: 0.158257605, 35.45607127: 0.127037774, 36.03364213: 0.126239166
}

monk_b_OS_raw = {
    0.607463669: 0.984818604, 0.830603543: 0.971380207, 1.228291643: 0.9675365,
    1.498872094: 0.960960894, 1.617812785: 0.93980683, 1.977345099: 0.923953674,
    2.098601178: 0.913463852, 2.410230087: 0.890968566, 2.619747036: 0.877347807,
    2.927292667: 0.866504346, 3.093242146: 0.851627596, 3.348312292: 0.834889087,
    3.624172859: 0.828843792, 3.985472765: 0.806415689, 4.526517336: 0.796285869,
    4.813158729: 0.785571912, 5.100509132: 0.756443368, 5.546479246: 0.737608474,
    5.7691363: 0.73670999, 6.406070524: 0.714112783, 6.629290868: 0.698584401,
    6.757133243: 0.682233825, 7.108237947: 0.649264441, 7.331433363: 0.6343835,
    7.681622282: 0.625199057, 7.840985727: 0.616176246, 8.19112654: 0.608241251,
    8.332124015: 0.580623511, 8.536493381: 0.559096906, 8.861468069: 0.543982099,
    9.237380281: 0.527506591, 9.530497587: 0.513798794, 10.07206959: 0.489970484,
    10.26348328: 0.474531513, 10.613869: 0.460235693, 10.91644503: 0.448663753,
    11.98267283: 0.427358392, 12.25038329: 0.408322808, 12.55638239: 0.399246185,
    12.94628469: 0.398217193, 13.07322066: 0.38340783, 13.41377871: 0.371025201,
    13.76634197: 0.357507258, 14.41758781: 0.355205427, 14.49167208: 0.343072409,
    14.89191735: 0.329013308, 15.42142805: 0.32324374, 15.67610692: 0.316667405,
    16.75832377: 0.293091049, 17.14025024: 0.285611856, 17.52201927: 0.282221765,
    17.87195016: 0.279738905, 18.41294487: 0.270903967, 18.81746293: 0.266671134,
    19.17666662: 0.259353166, 19.55832194: 0.258916315, 19.9086355: 0.246494666,
    20.35422732: 0.237484976, 20.67256585: 0.229525805, 21.70327622: 0.195687077,
    22.23220728: 0.196171993, 22.61384189: 0.196273095, 23.03743939: 0.196504833,
    23.08899526: 0.179484094, 23.64389573: 0.179485877, 24.14105012: 0.179281493,
    24.61393913: 0.179303169, 28.13844977: 0.158596372, 28.530642: 0.159825937,
    28.91227983: 0.159843431, 29.29391765: 0.159860924, 29.67555548: 0.159878417,
    30.05719331: 0.159895911, 30.45790198: 0.160201407, 30.82050165: 0.159082165,
    31.20209574: 0.16023552, 31.58375429: 0.15971506, 31.95902155: 0.159990348,
    32.34702029: 0.160000872, 32.72865812: 0.160018365, 33.11030469: 0.159808687,
    33.49193147: 0.160113308, 33.87358126: 0.159820021, 34.5189167: 0.159024519,
    34.60577058: 0.141236421, 34.82892571: 0.127401766, 35.21054604: 0.127873604,
    35.59214889: 0.128799787, 35.91018041: 0.128814365, 25.17384237: 0.179371738,
    25.67437625: 0.179394682, 26.09790491: 0.179414095, 26.55993618: 0.179435274,
    26.98346485: 0.179454688, 27.40699351: 0.179474101, 27.83052217: 0.179493515,
    28.17704563: 0.179509399, 21.07264772: 0.196510481, 20.68714515: 0.208869048,
    16.35236167: 0.292828766, 15.8902351: 0.295282831
}

monk_c_OS_raw = {
    0.384423507: 0.995667236, 0.766440727: 0.98583113, 1.179995634: 0.982891162,
    1.212180145: 0.972987907, 1.594041746: 0.96719353, 1.989423068: 0.958261983,
    2.00193044: 0.946217727, 2.358342667: 0.940600181, 2.645087596: 0.927197153,
    3.151566503: 0.908814428, 2.836056455: 0.923311517, 3.250380857: 0.900386064,
    3.544022384: 0.89141863, 3.712519323: 0.874623144, 4.047036997: 0.859455768,
    4.344362069: 0.846596276, 4.408720593: 0.827062373, 4.525865703: 0.813210207,
    4.812542346: 0.801580745, 5.109553263: 0.796880526, 5.2204532: 0.780029394,
    5.402271717: 0.765998454, 5.419113568: 0.741578159, 5.337085751: 0.724422859,
    5.431074737: 0.70625335, 5.671069083: 0.687733547, 5.871623121: 0.670899506,
    6.044898571: 0.654551013, 6.345450357: 0.636553238, 6.73655341: 0.626737832,
    7.120236727: 0.61296279, 7.485498906: 0.597758961, 7.874051326: 0.583389166,
    8.181592592: 0.580525742, 8.383587655: 0.565598395, 8.51135787: 0.551121991,
    8.829552958: 0.546888447, 9.179887949: 0.533910226, 9.577668764: 0.527658492,
    9.948154498: 0.522320689, 10.23014945: 0.51428521, 10.34576738: 0.485030664,
    10.35771193: 0.505204149, 10.56813304: 0.468100263, 10.81204101: 0.452823801,
    11.15519609: 0.442768207, 11.34661853: 0.427102064, 11.66519673: 0.412918372,
    12.09988678: 0.405051663, 12.23851502: 0.390568139, 12.4138697: 0.379217541,
    12.70573097: 0.370598356, 12.90746407: 0.36247482, 13.26215814: 0.347170506,
    13.56954903: 0.338657066, 13.7086636: 0.321542568, 13.98232357: 0.31298498,
    14.36231891: 0.312661639, 14.79216017: 0.311928867, 15.15444751: 0.295321711,
    15.2957986: 0.282119719, 15.67744517: 0.281910041, 16.00381453: 0.271872608,
    16.25076785: 0.259446221, 16.63237943: 0.260145232, 17.01420095: 0.255392107,
    17.39615366: 0.247231397, 17.54503195: 0.235198067, 17.89204517: 0.229826735,
    18.31014942: 0.226730388, 18.63752913: 0.219951849, 19.05102842: 0.218456318,
    19.43270998: 0.21733795, 19.81446152: 0.214402203, 20.2026769: 0.208786115,
    20.5254579: 0.195579458, 20.90748817: 0.185404487, 21.08446685: 0.160541551,
    21.47044712: 0.15677624, 21.81397042: 0.146712778, 22.23409014: 0.147269989,
    22.61575974: 0.146462402, 22.99739204: 0.146623459, 23.41084268: 0.146391586,
    23.7799374: 0.146181332, 24.16157845: 0.146115217, 24.56245009: 0.146587938,
    24.9248334: 0.146688157, 25.2872342: 0.146334032, 25.66884026: 0.147176606,
    26.05047809: 0.1471941, 26.24230051: 0.121139544, 26.62414972: 0.11566704,
    27.00577005: 0.116138878, 27.35571389: 0.113319804, 27.49877597: 0.089679629,
    27.42391581: 0.101964426, 27.96094922: 0.088013245, 28.34255206: 0.088939428,
    28.72417884: 0.089244049, 29.10582772: 0.088974415, 29.48750054: 0.088083219,
    29.74267798: 0.068558062, 29.74330774: 0.052201655, 29.74393751: 0.035845248,
    29.72559034: 0.016761899, 29.77385137: 0.084915926, 29.73844511: 0.004495162
}

monk_d_OS_raw = {
    0.129797113: 1.000880537, 0.511443689: 1.000670858, 0.893182894: 0.998055381,
    1.306917924: 0.990437244, 1.574415952: 0.981319033, 1.721249252: 0.967331924,
    1.833270521: 0.948890634, 2.081447568: 0.934181218, 2.352243605: 0.925006364,
    2.467612211: 0.911027302, 2.773506319: 0.895877544, 2.985679869: 0.88125897,
    3.386469483: 0.865862049, 3.308148596: 0.878029526, 3.633517878: 0.861466223,
    3.729483628: 0.847022437, 3.830208605: 0.826970948, 4.112372241: 0.814554289,
    4.314553173: 0.79479953, 4.468171446: 0.777658277, 4.845602842: 0.768926183,
    5.241070519: 0.763751791, 5.530041932: 0.75152205, 5.914402184: 0.73983216,
    6.049444877: 0.725273334, 6.115484775: 0.70927042, 6.343156961: 0.69611782,
    6.534502429: 0.682450793, 6.675158781: 0.663692774, 6.924086896: 0.644076462,
    7.066954493: 0.620154151, 7.332502652: 0.606611684, 7.682416049: 0.604583169,
    8.096145673: 0.597105434, 8.477774756: 0.5973501, 8.673824222: 0.579509801,
    8.797170771: 0.561925796, 9.182537535: 0.552094588, 9.386554647: 0.535316811,
    9.644576089: 0.524526576, 9.880463464: 0.510407245, 10.26250583: 0.499918019,
    10.5176461: 0.481358345, 10.88921059: 0.467669289, 11.14395653: 0.4593513,
    11.58742632: 0.426788569, 11.72122617: 0.403209306, 12.05853915: 0.384274777,
    12.38107657: 0.367382404, 12.73279077: 0.356128694, 13.09902821: 0.343128485,
    13.4488279: 0.34405321, 13.83523145: 0.338294244, 14.21719166: 0.329938815,
    14.46772708: 0.312977252, 14.77022876: 0.303336277, 15.16786001: 0.300969191,
    15.54955032: 0.29962365, 15.93133684: 0.295779214, 16.31323707: 0.288981538,
    16.6955659: 0.271052419, 17.04555277: 0.267115656, 17.45941009: 0.256321205,
    17.8411004: 0.254975665, 18.23660182: 0.248925037, 18.60490086: 0.241380313,
    18.73257705: 0.229346011, 19.11421488: 0.229363505, 19.4322639: 0.228923738,
    19.8775885: 0.226854162, 20.14816283: 0.220437576, 20.32371431: 0.203975601,
    20.70554457: 0.198995304, 21.08710805: 0.200943762, 21.44373632: 0.189715059,
    21.8509916: 0.185190273, 22.20105894: 0.179163525, 22.44320374: 0.167725119,
    22.80575968: 0.167741738, 23.15568882: 0.165304313, 23.24725521: 0.147121711,
    23.44754309: 0.137200199, 23.8246676: 0.136438608, 24.20633167: 0.135774584,
    24.58804822: 0.133747527, 24.96968605: 0.13376502, 25.35132388: 0.133782514,
    25.73296171: 0.133800007, 26.11463453: 0.132908812, 26.49625486: 0.13338065,
    26.8778752: 0.133852488, 27.25949553: 0.134324326, 27.64123832: 0.131615752,
    27.76261161: 0.118081821, 28.15059929: 0.118379474, 28.53221963: 0.118851312,
    28.91385746: 0.118868805, 29.29551278: 0.118431954, 29.67716258: 0.118138667,
    30.05879257: 0.118359681, 30.49317344: 0.118523156, 30.94933884: 0.116893901,
    31.3309822: 0.11676783, 31.71266376: 0.115649462, 31.97234736: 0.097089308,
    31.93541612: 0.112275734, 32.3495108: 0.095316718, 32.73116612: 0.094879866,
    33.11282375: 0.094383059, 33.48193275: 0.093802069, 33.93972874: 0.093823053,
    34.32136657: 0.093840547, 34.70302189: 0.093403696, 35.1482803: 0.093053368,
    35.52991813: 0.093070862, 35.8797528: 0.093086897
}

# ------------------------------
# Step 1: Sort each dictionary and convert to lists
# ------------------------------
def dict_to_sorted_lists(data):
    sorted_items = sorted(data.items())
    return [x for x, y in sorted_items], [y for x, y in sorted_items]

monk_a_os_time, monk_a_os_surv = dict_to_sorted_lists(monk_a_OS_raw)
monk_b_os_time, monk_b_os_surv = dict_to_sorted_lists(monk_b_OS_raw)
monk_c_os_time, monk_c_os_surv = dict_to_sorted_lists(monk_c_OS_raw)
monk_d_os_time, monk_d_os_surv = dict_to_sorted_lists(monk_d_OS_raw)

# ------------------------------
# Step 2: Define common time grid
# ------------------------------
all_times_os = monk_a_os_time + monk_b_os_time + monk_c_os_time + monk_d_os_time
t_min_os, t_max_os = min(all_times_os), max(all_times_os)

survival_time_monk_a = np.linspace(t_min_os, t_max_os, MONK_A_NUMBER)
survival_time_monk_b = np.linspace(t_min_os, t_max_os, MONK_B_NUMBER)
survival_time_monk_c = np.linspace(t_min_os, t_max_os, MONK_C_NUMBER)
survival_time_monk_d = np.linspace(t_min_os, t_max_os, MONK_D_NUMBER)

# ------------------------------
# Step 3: Interpolate each arm onto the common grid
# ------------------------------
def interpolate_curve(x_orig, y_orig, x_new, kind='linear'):
    """
    Interpolate survival curve.
    kind: 'linear' (default), 'previous' (Kaplan-Meier step), 'next', etc.
    Extrapolation: beyond original range, use last available value.
    """
    f = interp1d(x_orig, y_orig, kind=kind, bounds_error=False, fill_value=(y_orig[0], y_orig[-1]))
    return f(x_new)

method = 'linear'   # or 'previous'

monk_a_survival = interpolate_curve(monk_a_os_time, monk_a_os_surv, survival_time_monk_a, kind=method)
monk_b_survival = interpolate_curve(monk_b_os_time, monk_b_os_surv, survival_time_monk_b, kind=method)
monk_c_survival = interpolate_curve(monk_c_os_time, monk_c_os_surv, survival_time_monk_c, kind=method)
monk_d_survival = interpolate_curve(monk_d_os_time, monk_d_os_surv, survival_time_monk_d, kind=method)
#=============================================================================================
# PROGRESSION FREE SURVIVAL POINTS EXTRACTED FROM KAPLAN MEIER CURVE USING WEBPLOTDIGITIZER
#=============================================================================================
monk_a_pfs_raw = {
    0.215271166: 1.001014908, 0.448723198: 1.000767467, 0.682175229: 0.995892879,
    0.915627261: 0.978868938, 1.149079293: 0.960286119, 1.329474045: 0.938202697,
    1.424977149: 0.897990098, 1.658429181: 0.882985551, 1.891881213: 0.859664236,
    2.107142177: 0.843094526, 2.354069074: 0.820974086, 2.571014396: 0.800835138,
    2.804466428: 0.770028734, 3.03791846: 0.74924369, 3.26901239: 0.743552547,
    3.504822524: 0.72957213, 3.727663099: 0.704493985, 3.950503675: 0.682805781,
    4.214274153: 0.670256988, 4.336760673: 0.640938764, 4.481076475: 0.604366984,
    4.714528507: 0.580117766, 4.947980539: 0.563415498, 5.170821114: 0.547384856,
    5.39366169: 0.512566373, 5.627113722: 0.504648261, 5.860565754: 0.494255739,
    6.094017786: 0.483244614, 6.327469818: 0.474336738, 6.560921849: 0.475450223,
    6.794373881: 0.475450223, 7.027825913: 0.470130241, 7.240055033: 0.4600264,
    7.441672697: 0.422646313, 7.664513273: 0.39552678, 7.897965304: 0.386990065,
    8.131417336: 0.377339866, 8.364869368: 0.368060829, 8.550569848: 0.338223568,
    8.789327608: 0.336635821, 9.02277964: 0.324263771, 9.256231672: 0.317582864,
    9.489683703: 0.317582864, 9.723135735: 0.316840541, 9.956587767: 0.310530796,
    10.1900398: 0.303726168, 10.42349183: 0.299890833, 10.65694386: 0.299890833,
    10.89039589: 0.299643392, 11.12384793: 0.289622031, 11.35729996: 0.287395062,
    11.59075199: 0.268836987, 11.82420402: 0.242608241, 11.98337586: 0.23728826
}

monk_b_pfs_raw = {
    0.618506494: 0.974415, 0.851958525: 0.955856925, 1.085410557: 0.946608818,
    1.265805309: 0.903007651, 1.520480253: 0.84409594, 1.626594813: 0.815001136,
    1.849435389: 0.783390548, 2.072275965: 0.76029193, 2.29511654: 0.73946977,
    2.443676924: 0.728128724, 2.69304614: 0.672954536, 2.868135164: 0.64566312,
    3.048529916: 0.613107837, 3.194968009: 0.585957374, 3.377485052: 0.557885192,
    3.610937083: 0.527783994, 3.844389115: 0.521969131, 4.077841147: 0.495987826,
    4.311293179: 0.480770204, 4.544745211: 0.460480042, 4.756974331: 0.43931009,
    4.905534715: 0.41532206, 5.138986746: 0.408393712, 5.372438778: 0.385175498,
    5.60589081: 0.368803152, 5.839342842: 0.347193304, 6.072794874: 0.33143956,
    6.306246906: 0.329460032, 6.539698937: 0.311190639, 6.773150969: 0.301251758,
    7.006603001: 0.300880597, 7.235810451: 0.292677928, 7.501804281: 0.285827936,
    7.694225349: 0.276755099, 7.940411128: 0.272301161, 8.287689945: 0.271311397,
    8.407315192: 0.25708354, 8.640767224: 0.243226844, 8.874219256: 0.231349676,
    9.107671288: 0.209451147, 9.341123319: 0.200790712, 9.574575351: 0.186562855,
    9.744358647: 0.180673759, 10.19632384: 0.167705969, 10.42349183: 0.150436469,
    10.62157234: 0.143838042, 10.89039589: 0.14090999, 11.12384793: 0.135466288,
    11.67067353: 0.133857922, 11.87372415: 0.122516876, 12.01088281: 0.119495853
}
monk_c_pfs_raw = {
    0.130379518: 1.002128392, 0.36383155: 1.001757231, 0.597283582: 1.000520026,
    0.830735613: 0.982951715, 1.064187645: 0.963249225, 1.216285181: 0.929318878,
    1.297639677: 0.903259932, 1.453917483: 0.860167955, 1.630131965: 0.825208077,
    1.828212477: 0.795143995, 2.061664509: 0.782524504, 2.29511654: 0.75666692,
    2.539996294: 0.733397949, 2.701004732: 0.702724782, 2.910580988: 0.67864347,
    3.133421564: 0.66219642, 3.332007383: 0.627679461, 3.547268347: 0.595294147,
    3.796637563: 0.584154147, 4.014172411: 0.570220126, 4.254698747: 0.552418121,
    4.481076475: 0.531990491, 4.672082683: 0.512334081, 4.841865979: 0.471443053,
    5.117763834: 0.443937237, 5.308770042: 0.414579737, 5.542222074: 0.406909066,
    5.775674106: 0.394351435, 6.009126138: 0.368555711, 6.24257817: 0.346533461,
    6.476030201: 0.339233952, 6.709482233: 0.330985919, 6.942934265: 0.325913378,
    7.176386297: 0.312881485, 7.422572076: 0.295808056, 7.631499854: 0.277712788,
    7.893249102: 0.254526649, 8.110194424: 0.221080874, 8.343646456: 0.204378607,
    8.577098488: 0.197079097, 8.81055052: 0.188047501, 9.044002552: 0.181614035,
    9.277454584: 0.176046612, 9.510906615: 0.176170333, 9.744358647: 0.173819643,
    9.977810679: 0.170231749, 10.21126271: 0.158494797, 10.44471474: 0.159468065,
    10.64633241: 0.147637292, 10.84795007: 0.128909102, 11.0814021: 0.131754673,
    11.29666307: 0.132496996, 11.54830617: 0.132496996, 11.7817582: 0.128290499,
    11.95154149: 0.11671026
}
monk_d_pfs_raw = {
    0.130379518: 1.000767467, 0.36383155: 0.99668469, 0.597283582: 0.988642858,
    0.830735613: 0.982456833, 1.064187645: 0.978621497, 1.276416765: 0.955252069,
    1.364845565: 0.914121876, 1.400217085: 0.868757693, 1.435588605: 0.834054093,
    1.531091709: 0.804917915, 1.722097917: 0.77796452, 1.913104125: 0.744171149,
    2.146556156: 0.725118192, 2.380008188: 0.714973111, 2.604027815: 0.690174024,
    2.846912252: 0.672462747, 3.03791846: 0.648968224, 3.228924668: 0.629729687,
    3.4623767: 0.619419645, 3.695828731: 0.590634008, 3.918669307: 0.563229918,
    4.141509883: 0.542176813, 4.374961915: 0.532361653, 4.608413947: 0.502545012,
    4.831254523: 0.483751868, 5.054095098: 0.457634471, 5.266324218: 0.435680955,
    5.417537466: 0.416930426, 5.627113722: 0.409383476, 5.860565754: 0.399114674,
    6.094017786: 0.384515655, 6.327469818: 0.3757315, 6.560921849: 0.359647835,
    6.794373881: 0.339852554, 7.027825913: 0.328717709, 7.261277945: 0.326861902,
    7.494729977: 0.313500088, 7.728182009: 0.297292702, 7.929799672: 0.284920652,
    8.195086072: 0.276755099, 8.428538104: 0.27700254, 8.661990136: 0.268342105,
    8.895442168: 0.245824974, 9.1288942: 0.244092887, 9.362346231: 0.23976267,
    9.595798263: 0.234566409, 9.829250295: 0.234566409, 10.06270233: 0.233329204,
    10.29615436: 0.226400856, 10.52960639: 0.223060402, 10.76305842: 0.202027917,
    10.99651045: 0.189655867, 11.22996249: 0.189160985, 11.46341452: 0.180747991,
    11.69686655: 0.178768463, 11.90644281: 0.178428232
}

monk_a_pfs_time, monk_a_pfs_surv = dict_to_sorted_lists(monk_a_pfs_raw)
monk_b_pfs_time, monk_b_pfs_surv = dict_to_sorted_lists(monk_b_pfs_raw)
monk_c_pfs_time, monk_c_pfs_surv = dict_to_sorted_lists(monk_c_pfs_raw)
monk_d_pfs_time, monk_d_pfs_surv = dict_to_sorted_lists(monk_d_pfs_raw)

all_times_pfs = monk_a_pfs_time + monk_b_pfs_time + monk_c_pfs_time + monk_d_pfs_time
t_min_pfs, t_max_pfs = min(all_times_pfs), max(all_times_pfs)

pfs_time_monk_a = np.linspace(t_min_pfs, t_max_pfs, MONK_A_NUMBER)
pfs_time_monk_b = np.linspace(t_min_pfs, t_max_pfs, MONK_B_NUMBER)
pfs_time_monk_c = np.linspace(t_min_pfs, t_max_pfs, MONK_C_NUMBER)
pfs_time_monk_d = np.linspace(t_min_pfs, t_max_pfs, MONK_D_NUMBER)

monk_a_pfs = interpolate_curve(monk_a_pfs_time, monk_a_pfs_surv, pfs_time_monk_a, kind=method)
monk_b_pfs = interpolate_curve(monk_b_pfs_time, monk_b_pfs_surv, pfs_time_monk_b, kind=method)
monk_c_pfs = interpolate_curve(monk_c_pfs_time, monk_c_pfs_surv, pfs_time_monk_c, kind=method)
monk_d_pfs = interpolate_curve(monk_d_pfs_time, monk_d_pfs_surv, pfs_time_monk_d, kind=method)
#==================================================
# ASSIGNING PARAMETERS
#==================================================
def parameter_assign(data, ratio):
    if len(data) != len(ratio):
        raise ValueError(f"Length mismatch: {len(data)} keys vs {len(ratio)} values")
    return dict(zip(data, ratio))
#==========================================================================
# HISTOLOGY  (Table 1, "Cell type" row -- unchanged, verified exact match)
#==========================================================================
HISTOLOGY_monk_a = parameter_assign(histology_base, [81, 13, 8, 0, 0, 1, 0, 0, 0])
HISTOLOGY_monk_b = parameter_assign(histology_base, [80, 14, 8, 3, 0, 2, 0, 1, 0])
HISTOLOGY_monk_c = parameter_assign(histology_base, [88, 15, 6, 1, 0, 1, 0, 1, 0])
HISTOLOGY_monk_d = parameter_assign(histology_base, [86, 10, 14, 1, 0, 0, 0, 0, 0])

#===========================================================
# RACE  (labeled "ethnicity_base" for column-schema continuity -- see C2 note above)
#===========================================================
ETHNICITY_monk_a = parameter_assign(ethnicity_base, [75, 19, 4, 2, 0, 0, 3])
ETHNICITY_monk_b = parameter_assign(ethnicity_base, [79, 20, 6, 1, 0, 0, 2])
ETHNICITY_monk_c = parameter_assign(ethnicity_base, [80, 23, 3, 1, 0, 0, 5])
ETHNICITY_monk_d = parameter_assign(ethnicity_base, [82, 17, 4, 1, 0, 0, 7])

#================================================================
# DISEASE NATURE  (FIX C1: proper 3-way split, Table 1 "Stage" row)
#================================================================
NATURE_monk_a = parameter_assign(disease_nature, [12, 74, 17])   # Persistent, Recurrent, Advanced(IVB)
NATURE_monk_b = parameter_assign(disease_nature, [14, 77, 17])
NATURE_monk_c = parameter_assign(disease_nature, [12, 80, 20])
NATURE_monk_d = parameter_assign(disease_nature, [14, 77, 20])

#======================================================================
# DISEASE STAGE AS PER FIGO CLASSIFICATION
# (FIX C1: only the "Advanced (IVB)" subgroup is confirmed IVB; the
#  Persistent/Recurrent subgroup's initial FIGO stage is not reported in
#  this paper, so it is coded "Unknown" rather than assumed IVB.)
#======================================================================
STAGE_monk_a = parameter_assign(figo_stage, [0, 0, 0, 0, 17, 86])
STAGE_monk_b = parameter_assign(figo_stage, [0, 0, 0, 0, 17, 91])
STAGE_monk_c = parameter_assign(figo_stage, [0, 0, 0, 0, 20, 92])
STAGE_monk_d = parameter_assign(figo_stage, [0, 0, 0, 0, 20, 91])

#======================================================================
# TUMOR GRADE  (unchanged, verified exact match)
#======================================================================
GRADE_monk_a = parameter_assign(grade_base, [5, 49, 48, 1])
GRADE_monk_b = parameter_assign(grade_base, [8, 54, 46, 0])
GRADE_monk_c = parameter_assign(grade_base, [4, 57, 51, 0])
GRADE_monk_d = parameter_assign(grade_base, [6, 55, 50, 0])

#================================================================
# CAUSE OF DEATH  (Table 3, unchanged, verified exact match)
#================================================================
CAUSE_monk_a = parameter_assign(death_cause_base, [2, 72, 0])
CAUSE_monk_b = parameter_assign(death_cause_base, [4, 80, 1])
CAUSE_monk_c = parameter_assign(death_cause_base, [2, 87, 3])
CAUSE_monk_d = parameter_assign(death_cause_base, [3, 82, 4])

#================================================================
# ALIVE/DEAD  (Fig 2 caption table, unchanged, verified exact match)
#================================================================
ALIVE_monk_a = parameter_assign(dead_alive, [29, 74, 0])
ALIVE_monk_b = parameter_assign(dead_alive, [23, 85, 0])
ALIVE_monk_c = parameter_assign(dead_alive, [20, 92, 0])
ALIVE_monk_d = parameter_assign(dead_alive, [22, 89, 0])

#===================================================================
# PERFORMANCE STATUS  (Table 1, unchanged, verified exact match)
#===================================================================
PS_monk_a = parameter_assign(performance_status, [57, 46, 0, 0])
PS_monk_b = parameter_assign(performance_status, [57, 51, 0, 0])
PS_monk_c = parameter_assign(performance_status, [55, 57, 0, 0])
PS_monk_d = parameter_assign(performance_status, [59, 52, 0, 0])

#===================================================================
# DRUG EFFECT  (self-proposed heuristic multipliers -- see weight
# reference table at end of script; not directly used by the corrected
# survival-generation logic below, retained for documentation only)
#===================================================================
PROGNOSIS_monk_a = parameter_assign(drug_effect_base, [1, 1, 0, 1])
PROGNOSIS_monk_b = parameter_assign(drug_effect_base, [0.90, 0.78, 0.0, 0.78])
PROGNOSIS_monk_c = parameter_assign(drug_effect_base, [0.95, 0.87, 0.05, 0.80])
PROGNOSIS_monk_d = parameter_assign(drug_effect_base, [0.90, 0.78, 0.03, 0.78])

#====================================================================
# TREATMENT  (FIX C3: "Prior primary cisplatin AND radiation" recoded as
# "chemoradiotherapy" instead of "radiation_only" -- Table 1 explicitly
# reports concurrent platinum + radiation, not radiation alone)
#====================================================================
_NONE_IDX = prior_treatment_base.index("none")
_CRT_IDX = prior_treatment_base.index("chemoradiotherapy")

def _treatment_counts(none_n, crt_n):
    counts = [0] * len(prior_treatment_base)
    counts[_NONE_IDX] = none_n
    counts[_CRT_IDX] = crt_n
    return counts

treatment_monk_a = _treatment_counts(33, 70)   # 108-... wait see below: 103-70=33
treatment_monk_b = _treatment_counts(29, 79)
treatment_monk_c = _treatment_counts(40, 72)
treatment_monk_d = _treatment_counts(30, 81)

TREATMENT_monk_a = parameter_assign(prior_treatment_base, treatment_monk_a)
TREATMENT_monk_b = parameter_assign(prior_treatment_base, treatment_monk_b)
TREATMENT_monk_c = parameter_assign(prior_treatment_base, treatment_monk_c)
TREATMENT_monk_d = parameter_assign(prior_treatment_base, treatment_monk_d)
#===================================================================
# Toxicity  (verified against Table 2: grade3+4+5 sums match the paper's
# published grade>=3 percentages to within +/-0.1, i.e. rounding only)
#===================================================================

toxicity_df = pd.read_excel("arm_wise_toxicity.xlsx", sheet_name="Sheet1")

toxicity_df.columns = ['adverse_event', 'arm',
                       'grade_0_%', 'grade_1_%', 'grade_2_%',
                       'grade_3_%', 'grade_4_%', 'grade_5_%', 'total']

toxicity_df['arm'] = toxicity_df['arm'].astype(str).str.strip()
toxicity_df['adverse_event'] = toxicity_df['adverse_event'].astype(str).str.strip()

toxicity_df = toxicity_df[
    toxicity_df['arm'].isin(['monk_a', 'monk_b', 'monk_c', 'monk_d']) &
    toxicity_df['adverse_event'].notna() &
    (toxicity_df['adverse_event'] != 'nan') &
    (toxicity_df['adverse_event'] != '')
].copy().reset_index(drop=True)

def get_adjusted_grade3plus_counts(arm_label: str, itn_n: int, treated_n: int) -> dict:
    """Compute grade 3+ counts scaled to your ITT patient_number"""
    arm_map = {'monk_a': 'monk_a', 'monk_b': 'monk_b', 'monk_c': 'monk_c', 'monk_d': 'monk_d'}
    target_arm = arm_map.get(arm_label, arm_label)
    arm_df = toxicity_df[toxicity_df['arm'] == target_arm].copy()

    if arm_df.empty:
        print(f"WARNING: No data found for arm {target_arm}")
        return {}

    grade3plus = {}
    for _, row in arm_df.iterrows():
        event = str(row['adverse_event']).strip()
        if not event or event.lower() in ['nan', '']:
            continue
        p3 = float(row.get('grade_3_%', 0) or 0)
        p4 = float(row.get('grade_4_%', 0) or 0)
        p5 = float(row.get('grade_5_%', 0) or 0)
        treated_count = round((p3 + p4 + p5) / 100.0 * treated_n)
        itn_count = round(treated_count * itn_n / treated_n) if treated_n > 0 else 0
        grade3plus[event] = max(0, itn_count)
    return grade3plus

TREATED_N = {'monk_a': 101, 'monk_b': 106, 'monk_c': 109, 'monk_d': 109}

grade3plus_a = get_adjusted_grade3plus_counts('monk_a', 103, TREATED_N['monk_a'])
grade3plus_b = get_adjusted_grade3plus_counts('monk_b', 108, TREATED_N['monk_b'])
grade3plus_c = get_adjusted_grade3plus_counts('monk_c', 112, TREATED_N['monk_c'])
grade3plus_d = get_adjusted_grade3plus_counts('monk_d', 111, TREATED_N['monk_d'])

name_map = {
    'Leucopenia': 'Leucopenia', 'Neutropenia': 'Neutropenia', 'Thrombocytopenia': 'Thrombocytopenia',
    'Anemia': 'Anemia', 'Other hematologic': 'Other hematologic',
    'Allergic reaction': 'Allergic reactions', 'Other allergy': 'Allergic reactions',
    'Inner ear/hearing': 'Inner ear/hearing', 'Other auditory': 'Other auditory',
    'Thrombosis embolism': 'Thrombosis embolism',
    'Cardiac, left ventricular': 'Cardiac left ventricular function',
    'Sinus bradycardia': 'Cardiac left ventricular function',
    'Other cardiovascular': 'Other cardiovascular',
    'Fatigue': 'Fatigue', 'Other constitutional': 'Other constitutional',
    'Alopecia': 'Alopecia', 'Rash desquamation': 'Dermatologic', 'Other dermatologic': 'Dermatologic',
    'Nausea': 'Nausea/vomiting', 'Vomiting': 'Nausea/vomiting', 'Stomatitis': 'Stomatitis',
    'Other GI': 'Other Gastrointestinal',
    'Creatinine': 'Creatinine', 'Genitourinary/renal': 'Other genitourinary/renal',
    'Hemorrhage': 'Hemorrhage', 'Hepatic': 'Hepatic',
    'Febrile with neutropenia': 'Febrile with neutropenia',
    'Infection without neutropenia': 'Infection without neutropenia',
    'Other infection/fever': 'Other infection/fever',
    'Lymphatic': 'Lymphatics', 'Metabolic': 'Metabolic', 'Musculoskeletal': 'Musculoskeletal',
    'Peripheral neuropathy': 'Peripheral neuropathy', 'Other neurologic': 'Other neurological',
    'Ocular/visual': 'Ocular/visual', 'Myalgia': 'Pain', 'Other pain': 'Pain', 'Pulmonary': 'Pulmonary',
    'Sexual': 'Sexual', 'Endocrine': 'Other constitutional',
}

def map_to_toxicity_base(grade3plus_dict):
    mapped = {category: 0 for category in toxicity_base}
    for source_event, count in grade3plus_dict.items():
        target = name_map.get(source_event)
        if target:
            mapped[target] = mapped.get(target, 0) + count
        else:
            print(f"Warning: No mapping found for '{source_event}'")
    return [mapped[category] for category in toxicity_base]

tox_monk_a = map_to_toxicity_base(grade3plus_a)
tox_monk_b = map_to_toxicity_base(grade3plus_b)
tox_monk_c = map_to_toxicity_base(grade3plus_c)
tox_monk_d = map_to_toxicity_base(grade3plus_d)

TOXICITY_monk_a = parameter_assign(toxicity_base, tox_monk_a)
TOXICITY_monk_b = parameter_assign(toxicity_base, tox_monk_b)
TOXICITY_monk_c = parameter_assign(toxicity_base, tox_monk_c)
TOXICITY_monk_d = parameter_assign(toxicity_base, tox_monk_d)
#======================================================================
# CREATING FINAL DICTIONARIES
#======================================================================
def create_trial_dict(
    trial_id, drug_combination, ethnicity, histology, disease_nature, disease_stage,
    grades, causes, alive_dead, performance_status, prognosis, cycle_completion_rate,
    toxicity, prior_treatment, patient_number, followup,
    target_orr, target_pfsr, target_os, target_pfs,
    os_time_grid, os_surv_grid, pfs_time_grid, pfs_surv_grid,
    target_age, age_type, age_min, age_max, age_sd,
    alive_count, pfs_alive_count, cr_count, pr_count, sd_count,
    os_ps_weight, os_orr_weight, os_age_weight, os_hazard_ratio,
    pfs_age_weight, pfs_histology_weight, pfs_response_weight, pfs_toxicity_weight,
    orr_vs_ps, orr_vs_radiation, orr_vs_disease_status, tolerability_orr_weight,
    pfsr_vs_ps, pfsr_vs_radiation,
    never_treated_count
):

    trial = {}
    trial["trial_id"] = trial_id

    trial["platinum_drug"] = drug_combination.get("Platinum_Drug")
    trial["platinum_dose_value"] = drug_combination.get("Platinum_Dose_Value")
    trial["platinum_dose_unit"] = drug_combination.get("Platinum_Dose_Unit")
    trial["platinum_dose_method"] = drug_combination.get("Platinum_Dose_Method")
    trial["platinum_infusion_hours"] = drug_combination.get("Platinum_Infusion_Duration_Hours")
    trial["platinum_days_number"] = drug_combination.get("Platinum_Days_Number")
    trial["platinum_days"] = drug_combination.get("Platinum_Day")

    for i in range(1, 4):
        trial[f"adjunct_{i}_name"] = drug_combination.get(f"Adjunct_Drug_{i}")
        trial[f"adjunct_{i}_dose_value"] = drug_combination.get(f"Adjunct_Drug_{i}_Dose_Value")
        trial[f"adjunct_{i}_dose_unit"] = drug_combination.get(f"Adjunct_Drug_{i}_Dose_Unit")
        trial[f"adjunct_{i}_dose_method"] = drug_combination.get(f"Adjunct_Drug_{i}_Dose_Method")
        trial[f"adjunct_{i}_infusion_hours"] = drug_combination.get(f"Adjunct_Drug_{i}_Infusion_Duration_Hours")
        trial[f"adjunct_{i}_days_number"] = drug_combination.get(f"Adjunct_Drug_{i}_Days_Number")
        trial[f"adjunct_{i}_days"] = drug_combination.get(f"Adjunct_Drug_{i}_Days")
        trial[f"adjunct_{i}_same_day_platinum"] = drug_combination.get(f"Adjunct_Drug_{i}_Same_Day_As_Platinum")

    for i in range(1, 4):
        trial[f"supportive_{i}_name"] = drug_combination.get(f"Supportive_Drug_{i}")
        trial[f"supportive_{i}_dose_value"] = drug_combination.get(f"Supportive_Drug_{i}_Dose_Value")
        trial[f"supportive_{i}_dose_unit"] = drug_combination.get(f"Supportive_Drug_{i}_Dose_Unit")
        trial[f"supportive_{i}_route"] = drug_combination.get(f"Supportive_Drug_{i}_Route")
        trial[f"supportive_{i}_start_day"] = drug_combination.get(f"Supportive_Drug_{i}_Start_Day")
        trial[f"supportive_{i}_duration_value"] = drug_combination.get(f"Supportive_Drug_{i}_Duration_Value")
        trial[f"supportive_{i}_duration_unit"] = drug_combination.get(f"Supportive_Drug_{i}_Duration_Unit")
        trial[f"supportive_{i}_frequency_per_day"] = drug_combination.get(f"Supportive_Drug_{i}_Frequency_Times_Per_Day")
        trial[f"supportive_{i}_frequency_interval_hours"] = drug_combination.get(f"Supportive_Drug_{i}_Frequency_Interval_Hours")

    trial["cycle_length_days"] = drug_combination.get("Cycle_Length_Days")

    trial.update({
        "ethnicity": ethnicity, "histology": histology, "disease_nature": disease_nature,
        "disease_stage": disease_stage, "grades": grades, "causes": causes, "alive_dead": alive_dead,
        "performance_status": performance_status, "prognosis": prognosis,
        "cycle_completion_rate": cycle_completion_rate, "toxicity": toxicity,
        "prior_treatment": prior_treatment, "patient_number": patient_number, "followup": followup,
        "target_orr": target_orr, "target_pfsr": target_pfsr, "target_os": target_os, "target_pfs": target_pfs,
        "os_time_grid": os_time_grid, "os_surv_grid": os_surv_grid,
        "pfs_time_grid": pfs_time_grid, "pfs_surv_grid": pfs_surv_grid,
        "target_age": target_age, "age_type": age_type, "age_min": age_min, "age_max": age_max, "age_sd": age_sd,
        "alive_count": alive_count, "pfs_alive_count": pfs_alive_count,
        "cr_count": cr_count, "pr_count": pr_count, "sd_count": sd_count,
        "os_ps_weight": os_ps_weight, "os_orr_weight": os_orr_weight, "os_age_weight": os_age_weight,
        "os_hazard_ratio": os_hazard_ratio,
        "pfs_age_weight": pfs_age_weight, "pfs_histology_weight": pfs_histology_weight,
        "pfs_response_weight": pfs_response_weight, "pfs_toxicity_weight": pfs_toxicity_weight,
        "orr_vs_ps": orr_vs_ps, "orr_vs_radiation": orr_vs_radiation,
        "orr_vs_disease_status": orr_vs_disease_status, "tolerability_orr_weight": tolerability_orr_weight,
        "pfsr_vs_ps": pfsr_vs_ps, "pfsr_vs_radiation": pfsr_vs_radiation,
        "never_treated_count": never_treated_count
    })

    return trial

_PFS_HISTOLOGY_WEIGHT = {
    "Squamous Cell Carcinoma": 1.00, "Adenocarcinoma": 0.75, "Adenosquamous": 0.85,
    "Mucinous Adenocarcinoma": 0.65, "Clear Cell": 0.70, "Endometrioid": 0.80,
    "Villoglandular": 1.15, "Undifferentiated Carcinoma": 0.50, "Not Specified": 0.90,
}

#=====================================================================
# monk-a (PC ARM) TRIAL CONSTANTS
#=====================================================================
trial_monk_a = create_trial_dict(
    trial_id="monk_PC_arm", drug_combination=DRUG_monk_a, ethnicity=ETHNICITY_monk_a,
    histology=HISTOLOGY_monk_a, disease_nature=NATURE_monk_a, disease_stage=STAGE_monk_a,
    grades=GRADE_monk_a, causes=CAUSE_monk_a, alive_dead=ALIVE_monk_a,
    performance_status=PS_monk_a, prognosis=PROGNOSIS_monk_a, cycle_completion_rate=0.563,
    toxicity=TOXICITY_monk_a, prior_treatment=TREATMENT_monk_a, patient_number=MONK_A_NUMBER,
    followup=28,
    target_orr=0.291, target_pfsr=0.484, target_os=12.87, target_pfs=5.82,
    os_time_grid=survival_time_monk_a, os_surv_grid=monk_a_survival,
    pfs_time_grid=pfs_time_monk_a, pfs_surv_grid=monk_a_pfs,
    target_age=50, age_type="median", age_min=29, age_max=81, age_sd=(81 - 29) / 4,
    alive_count=29, pfs_alive_count=7, cr_count=3, pr_count=27, sd_count=50,
    os_ps_weight=0.4, os_orr_weight=0.3, os_age_weight=0.2, os_hazard_ratio=1.0,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_radiation=0.20, orr_vs_disease_status=-0.35, tolerability_orr_weight=0.20,
    pfsr_vs_ps=-0.35, pfsr_vs_radiation=0.25,
    never_treated_count=2
)

#=====================================================================
# monk-b (VC ARM) TRIAL CONSTANTS
#=====================================================================
trial_monk_b = create_trial_dict(
    trial_id="monk_VC_arm", drug_combination=DRUG_monk_b, ethnicity=ETHNICITY_monk_b,
    histology=HISTOLOGY_monk_b, disease_nature=NATURE_monk_b, disease_stage=STAGE_monk_b,
    grades=GRADE_monk_b, causes=CAUSE_monk_b, alive_dead=ALIVE_monk_b,
    performance_status=PS_monk_b, prognosis=PROGNOSIS_monk_b, cycle_completion_rate=0.417,
    toxicity=TOXICITY_monk_b, prior_treatment=TREATMENT_monk_b, patient_number=MONK_B_NUMBER,
    followup=28,
    target_orr=0.259, target_pfsr=0.426, target_os=9.99, target_pfs=3.98,
    os_time_grid=survival_time_monk_b, os_surv_grid=monk_b_survival,
    pfs_time_grid=pfs_time_monk_b, pfs_surv_grid=monk_b_pfs,
    target_age=49, age_type="median", age_min=24, age_max=76, age_sd=(76 - 24) / 4,
    alive_count=23, pfs_alive_count=5, cr_count=8, pr_count=20, sd_count=46,
    os_ps_weight=0.4, os_orr_weight=0.3, os_age_weight=0.2, os_hazard_ratio=1.15,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_radiation=0.20, orr_vs_disease_status=-0.35, tolerability_orr_weight=0.20,
    pfsr_vs_ps=-0.35, pfsr_vs_radiation=0.25,
    never_treated_count=2
)
#=====================================================================
# monk-c (GC ARM) TRIAL CONSTANTS
#=====================================================================
trial_monk_c = create_trial_dict(
    trial_id="monk_GC_arm", drug_combination=DRUG_monk_c, ethnicity=ETHNICITY_monk_c,
    histology=HISTOLOGY_monk_c, disease_nature=NATURE_monk_c, disease_stage=STAGE_monk_c,
    grades=GRADE_monk_c, causes=CAUSE_monk_c, alive_dead=ALIVE_monk_c,
    performance_status=PS_monk_c, prognosis=PROGNOSIS_monk_c, cycle_completion_rate=0.429,
    toxicity=TOXICITY_monk_c, prior_treatment=TREATMENT_monk_c, patient_number=MONK_C_NUMBER,
    followup=28,
    target_orr=0.223, target_pfsr=0.482, target_os=10.28, target_pfs=4.70,
    os_time_grid=survival_time_monk_c, os_surv_grid=monk_c_survival,
    pfs_time_grid=pfs_time_monk_c, pfs_surv_grid=monk_c_pfs,
    target_age=45, age_type="median", age_min=20, age_max=89, age_sd=(89 - 20) / 4,
    alive_count=20, pfs_alive_count=8, cr_count=1, pr_count=24, sd_count=54,
    os_ps_weight=0.4, os_orr_weight=0.3, os_age_weight=0.2, os_hazard_ratio=1.32,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_radiation=0.20, orr_vs_disease_status=-0.35, tolerability_orr_weight=0.20,
    pfsr_vs_ps=-0.35, pfsr_vs_radiation=0.25,
    never_treated_count=3
)

#=====================================================================
# monk-d (TC ARM) TRIAL CONSTANTS
#=====================================================================
trial_monk_d = create_trial_dict(
    trial_id="monk_TC_arm", drug_combination=DRUG_monk_d, ethnicity=ETHNICITY_monk_d,
    histology=HISTOLOGY_monk_d, disease_nature=NATURE_monk_d, disease_stage=STAGE_monk_d,
    grades=GRADE_monk_d, causes=CAUSE_monk_d, alive_dead=ALIVE_monk_d,
    performance_status=PS_monk_d, prognosis=PROGNOSIS_monk_d, cycle_completion_rate=0.478,
    toxicity=TOXICITY_monk_d, prior_treatment=TREATMENT_monk_d, patient_number=MONK_D_NUMBER,
    followup=28,
    target_orr=0.234, target_pfsr=0.478, target_os=10.25, target_pfs=4.57,
    os_time_grid=survival_time_monk_d, os_surv_grid=monk_d_survival,
    pfs_time_grid=pfs_time_monk_d, pfs_surv_grid=monk_d_pfs,
    target_age=48, age_type="median", age_min=25, age_max=75, age_sd=(75 - 25) / 4,
    alive_count=22, pfs_alive_count=9, cr_count=2, pr_count=24, sd_count=53,
    os_ps_weight=0.4, os_orr_weight=0.3, os_age_weight=0.2, os_hazard_ratio=1.26,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_radiation=0.20, orr_vs_disease_status=-0.35, tolerability_orr_weight=0.20,
    pfsr_vs_ps=-0.35, pfsr_vs_radiation=0.25,
    never_treated_count=2
)

#==========================================================================
# TRIAL DICTIONARY
#==========================================================================

TRIALS = {
    "monk_a": trial_monk_a,
    "monk_b": trial_monk_b,
    "monk_c": trial_monk_c,
    "monk_d": trial_monk_d,
}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generating Final Datasets
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ================================================================
# Helper Functions (pure and simple)
# ================================================================
def expand_distribution(dist, n):
    """Expand dictionary distribution into array of length n"""
    np.random.seed(SEED)
    if not dist:
        return np.zeros(n, dtype=int)
    arr = np.concatenate([[k] * v for k, v in dist.items()])
    if len(arr) < n:
        arr = np.pad(arr, (0, n - len(arr)), constant_values=arr[0] if len(arr) > 0 else 0)
    return np.random.permutation(arr[:n])

def zscore(x):
    """Standardize an array to mean 0, sd 1 (returns zeros if constant)."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

def derive_prior_treatments(treatment):
    """Convert treatment strings into binary prior exposure flags"""
    n = len(treatment)
    prior_radiation = np.zeros(n, dtype=int)
    prior_chemo = np.zeros(n, dtype=int)
    prior_platinum = np.zeros(n, dtype=int)
    prior_surgery = np.zeros(n, dtype=int)
    for i, t in enumerate(treatment):
        t = str(t).lower()
        if t == "none":
            continue
        if t in ["radiation_only", "surgery_plus_radiation", "radiation_plus_chemotherapy",
                 "surgery_radiation_chemotherapy", "chemoradiotherapy"]:
            prior_radiation[i] = 1
        if t in ["chemotherapy_only", "radiation_plus_chemotherapy",
                 "surgery_radiation_chemotherapy", "prior_chemo_platinum",
                 "prior_cisplatin", "chemoradiotherapy"]:
            prior_chemo[i] = 1
        # FIX C3: "chemoradiotherapy" here specifically represents Table 1's
        # "prior primary CISPLATIN and radiation" -- known platinum exposure,
        # so it now also sets the platinum flag (previously only
        # prior_chemo_platinum / prior_cisplatin did).
        if t in ["prior_chemo_platinum", "prior_cisplatin", "chemoradiotherapy"]:
            prior_platinum[i] = 1
        if t in ["surgery_only", "surgery_plus_radiation",
                 "surgery_radiation_chemotherapy", "surgery_intent_curative",
                 "surgery_intent_palliative"]:
            prior_surgery[i] = 1
    return prior_radiation, prior_chemo, prior_platinum, prior_surgery

def generate_age(n, target_age, age_type, age_min, age_max, age_sd):
    """Generate age with target mean/median and iterative correction"""
    np.random.seed(SEED)
    age = np.random.lognormal(mean=3.5, sigma=0.35, size=n)
    age = (age - np.mean(age)) / np.std(age, ddof=1)
    age = age * age_sd
    if age_type == "mean":
        age = age + (target_age - np.mean(age))
    else:
        age = age + (target_age - np.median(age))
    age = np.clip(age, age_min, age_max)
    age = np.round(age).astype(int)
    for _ in range(5):
        if age_type == "mean":
            diff = target_age - np.mean(age)
        else:
            diff = target_age - np.median(age)
        if abs(diff) < 0.01:
            break
        idx = np.random.choice(n, size=int(n * 0.3), replace=False)
        age[idx] += np.sign(diff)
        age = np.clip(age, age_min, age_max)
    age[age.argmin()] = age_min
    age[age.argmax()] = age_max
    return age
# ============================================================================
# GLOBAL DEFAULTS (set these before calling if you want to omit arguments)
# ============================================================================

KM_TIMES = None
KM_SURVIVALS = None
PATIENT_N = None
TOTAL_EVENTS = None
ARM_ID = 1
RANDOM_SEED = 42

INTERPOLATION_KIND = 'previous'
# NOTE: censoring times are drawn uniformly between the last digitized time
# point and 1.1x that value. This is a simplification of the full Guyot
# algorithm: Monk et al. 2009's Figure 2 does not publish a "number at
# risk" table under the KM curves (unlike, e.g., Kitagawa et al. 2015's
# figures), so there is no external reference to calibrate WHEN censoring
# actually occurred across follow-up -- only that some patients were
# censored by study end. If a risk table becomes available, replace this
# with the full at-risk-table-driven censoring-time reconstruction.
CENSORING_METHOD = 'uniform'
CENSORING_MAX_FACTOR = 1.1

# ============================================================================
# FUNCTION - accepts arguments, falls back to globals, then to defaults
# ============================================================================

def guyot_reconstruct_ipd(t_digitized=None, S_digitized=None, n_patients=None,
                         tot_events=None, arm_id=None, random_state=None):
    """
    Reconstruct IPD from digitized KM curve.
    """
    if t_digitized is None:
        t_digitized = KM_TIMES
    if S_digitized is None:
        S_digitized = KM_SURVIVALS
    if n_patients is None:
        n_patients = PATIENT_N
    if tot_events is None:
        tot_events = TOTAL_EVENTS
    if arm_id is None:
        arm_id = ARM_ID

    if t_digitized is None or S_digitized is None or n_patients is None:
        raise ValueError("Must provide t_digitized, S_digitized, n_patients either as arguments or via globals KM_TIMES, KM_SURVIVALS, PATIENT_N")

    if random_state is None:
        random_state = RANDOM_SEED + arm_id
    np.random.seed(random_state)

    points = sorted(set(zip(t_digitized, S_digitized)))
    t = np.array([p[0] for p in points])
    S = np.array([p[1] for p in points])

    for i in range(1, len(S)):
        if S[i] > S[i-1]:
            S[i] = S[i-1]
    S = np.clip(S, 0.0, 1.0)

    event_times = []
    event_probs = []
    for i in range(1, len(t)):
        if S[i] < S[i-1]:
            event_times.append(t[i])
            event_probs.append(S[i])

    if not event_times:
        times = np.full(n_patients, t[-1] * CENSORING_MAX_FACTOR)
        events = np.zeros(n_patients, dtype=int)
        return pd.DataFrame({'time': times, 'event': events})

    n_risk_before = [n_patients]
    S_before = [1.0]
    for i in range(len(event_times)):
        S_prev = S_before[-1]
        S_curr = event_probs[i]
        n_i = n_risk_before[-1]
        d_i = max(1, round(n_i * (1 - S_curr / S_prev)))
        d_i = min(d_i, n_i)
        n_after = n_i - d_i
        n_risk_before.append(n_after)
        S_before.append(S_curr)

    event_times_list = []
    for i, t_event in enumerate(event_times):
        d_i = n_risk_before[i] - n_risk_before[i+1]
        event_times_list.extend([t_event] * d_i)

    n_events = len(event_times_list)
    n_censored = n_patients - n_events

    if n_censored > 0:
        last_time = max(t[-1], max(event_times))
        if CENSORING_METHOD == 'uniform':
            censor_times = np.random.uniform(last_time, last_time * CENSORING_MAX_FACTOR, n_censored)
        elif CENSORING_METHOD == 'exponential':
            scale = last_time * 0.5
            censor_times = np.random.exponential(scale, n_censored) + last_time
        else:
            raise ValueError(f"Unknown CENSORING_METHOD: {CENSORING_METHOD}")

        all_times = np.concatenate([event_times_list, censor_times])
        all_events = np.concatenate([np.ones(n_events, dtype=int), np.zeros(n_censored, dtype=int)])
    else:
        all_times = np.array(event_times_list)
        all_events = np.ones(n_events, dtype=int)

    if tot_events is not None and tot_events != n_events:
        diff = tot_events - n_events
        if diff > 0:
            censored_idx = np.where(all_events == 0)[0]
            if len(censored_idx) >= diff:
                chosen = np.random.choice(censored_idx, diff, replace=False)
                all_events[chosen] = 1
                new_times = np.random.choice(event_times, diff, replace=True)
                all_times[chosen] = new_times
        else:
            event_idx = np.where(all_events == 1)[0]
            remove = np.random.choice(event_idx, -diff, replace=False)
            all_events[remove] = 0

    if len(all_times) != n_patients:
        if len(all_times) > n_patients:
            all_times = all_times[:n_patients]
            all_events = all_events[:n_patients]
        else:
            extra = n_patients - len(all_times)
            last = t[-1] * CENSORING_MAX_FACTOR
            all_times = np.append(all_times, [last] * extra)
            all_events = np.append(all_events, np.zeros(extra, dtype=int))

    idx = np.argsort(all_times)
    all_times = all_times[idx]
    all_events = all_events[idx]

    return pd.DataFrame({'time': all_times, 'event': all_events})
# ================================================================
# Main Data Generation Function
# ================================================================
def generate_clinical_data(trial_config, arm_name='monk_a'):
    """Generate clinical data for one arm - fully procedural, no state dict"""
    np.random.seed(SEED)
    n = trial_config["patient_number"]
    FOLLOWUP = trial_config["followup"]
    TARGET_ORR = trial_config["target_orr"]
    TARGET_PFSR = trial_config["target_pfsr"]  # = Table 4 Stable-Disease rate; used only for the eval cross-check
    TARGET_OS = trial_config["target_os"]
    TARGET_PFS = trial_config["target_pfs"]
    TARGET_AGE = trial_config["target_age"]
    AGE_TYPE = trial_config.get("age_type", "median")
    AGE_MIN = trial_config["age_min"]
    AGE_MAX = trial_config["age_max"]
    AGE_SD = trial_config["age_sd"]
    CYCLE_COMPLETION_RATE = trial_config.get("cycle_completion_rate", 0.50)
    OS_PS_WEIGHT = trial_config["os_ps_weight"]
    OS_ORR_WEIGHT = trial_config["os_orr_weight"]
    OS_AGE_WEIGHT = trial_config["os_age_weight"]
    OS_HAZARD_RATIO = trial_config["os_hazard_ratio"]  # QC cross-check only, see note below
    PFS_AGE_WEIGHT = trial_config["pfs_age_weight"]
    PFS_RESPONSE_WEIGHT = trial_config["pfs_response_weight"]
    PFS_TOXICITY_WEIGHT = trial_config["pfs_toxicity_weight"]
    ORR_VS_PS = trial_config["orr_vs_ps"]
    ORR_VS_RADIATION = trial_config["orr_vs_radiation"]
    ORR_VS_DISEASE_STATUS = trial_config["orr_vs_disease_status"]
    TOLERABILITY_ORR_WEIGHT = trial_config["tolerability_orr_weight"]
    PFSR_VS_PS = trial_config["pfsr_vs_ps"]
    PFSR_VS_RADIATION = trial_config["pfsr_vs_radiation"]
    ALIVE_COUNT = trial_config["alive_count"]
    PFS_ALIVE_COUNT = trial_config["pfs_alive_count"]
    CR_COUNT = trial_config["cr_count"]
    PR_COUNT = trial_config["pr_count"]
    SD_COUNT = trial_config["sd_count"]

    # ----------------------------------------------------------------
    # 1. Age
    # ----------------------------------------------------------------
    age = generate_age(n, TARGET_AGE, AGE_TYPE, AGE_MIN, AGE_MAX, AGE_SD)

    # ----------------------------------------------------------------
    # 2. Cycle completion & tolerability (self-proposed constants -- see
    #    weight reference table)
    # ----------------------------------------------------------------
    never_treated_count = trial_config.get("never_treated_count", 0)
    never_treated_idx = np.random.choice(n, never_treated_count, replace=False) if never_treated_count > 0 else np.array([], dtype=int)
    never_treated = np.zeros(n, dtype=bool)
    never_treated[never_treated_idx] = True
    treated = ~never_treated
    n_treated = n - never_treated_count
    completion_rate_treated = min(1.0, CYCLE_COMPLETION_RATE * n / n_treated) if n_treated > 0 else 0.0
    completed_6 = np.zeros(n, dtype=float)
    completed_6[treated] = np.random.binomial(1, completion_rate_treated, n_treated).astype(float)
    tolerability = completed_6 * 0.82 + np.random.normal(0.5, 0.09, n)
    tolerability = np.clip(tolerability, 0.15, 1.0)
    tox_penalty = (1 - tolerability) * 0.18

    # ----------------------------------------------------------------
    # 3. Prior treatment history
    # ----------------------------------------------------------------
    treatment = expand_distribution(trial_config["prior_treatment"], n)
    PRIOR_RAD, PRIOR_CHEMO, PRIOR_PLATINUM, PRIOR_SURGERY = derive_prior_treatments(treatment)

    # ----------------------------------------------------------------
    # 4. Disease nature / FIGO stage / PFI-category hazard ratios
    #    (FIX C1: proper 3-way split; disease_hr trial-cited, Fig 3B)
    # ----------------------------------------------------------------
    nature_dist = expand_distribution(trial_config.get("disease_nature", {}), n)
    disease_nature_list = []
    disease_hr = np.ones(n)
    disease_pfi_category = []
    for i, nat_raw in enumerate(nature_dist):
        nat_lower = str(nat_raw).lower().strip()
        if "persistent" in nat_lower:
            disease_nature_list.append("Persistent")
            disease_hr[i] = 0.903          # Fig 3B, persistent disease vs advanced
            disease_pfi_category.append("NA")
        elif "recurrent" in nat_lower:
            pfi_category = np.random.choice(['0-5', '6-11', '12-17', '18-23', '24-29', '30+'],
                                      p=[0.15, 0.15, 0.20, 0.20, 0.15, 0.15])
            disease_pfi_category.append(pfi_category)
            hr_map = {'0-5': 1.386, '6-11': 1.210, '12-17': 0.928,
                      '18-23': 0.813, '24-29': 0.641, '30+': 0.598}    # Fig 3B, by PFI bucket
            disease_hr[i] = hr_map[pfi_category]
            disease_nature_list.append("Recurrent")
        else:
            # "Advanced (IVB)" -- reference category, HR = 1.0 by definition (Fig 3B)
            disease_nature_list.append("Advanced (IVB)")
            disease_hr[i] = 1.0
            disease_pfi_category.append("NA")

    figo_stage_list = [
        "IVB" if nat == "Advanced (IVB)" else "Unknown"
        for nat in disease_nature_list
    ]

    # ----------------------------------------------------------------
    # 5. Performance status (base assignment from exact trial counts, then
    #    disease- and tolerability-driven worsening, then forced back onto
    #    the exact reported marginal counts -- all self-proposed mechanics
    #    except the marginal counts themselves, which are Table 1-exact)
    # ----------------------------------------------------------------
    target_ps_dist = trial_config["performance_status"]
    ps1_count = target_ps_dist.get(1, 0)
    ps2_count = target_ps_dist.get(2, 0)
    PS_discrete = np.array([1] * ps1_count + [2] * ps2_count)
    if len(PS_discrete) < n:
        PS_discrete = np.concatenate([PS_discrete, np.full(n - len(PS_discrete), 2)])
    PS_discrete = np.random.permutation(PS_discrete[:n])

    # Disease-status-driven PS worsening (self-proposed thresholds/probabilities)
    for i in range(n):
        if disease_nature_list[i] == "Recurrent":
            if disease_hr[i] > 1.2:
                PS_discrete[i] = min(PS_discrete[i] + 1, 2)
            elif disease_hr[i] > 0.95:
                if np.random.random() < 0.2:
                    PS_discrete[i] = min(PS_discrete[i] + 1, 2)

    # Tolerability-driven PS worsening (self-proposed threshold/fraction)
    worsen_idx = np.where(tolerability < 0.35)[0]
    if len(worsen_idx) > 0:
        worsen_sample = np.random.choice(worsen_idx, size=int(0.1 * len(worsen_idx)), replace=False)
        PS_discrete[worsen_sample] = np.clip(PS_discrete[worsen_sample] + 1, 1, 2)

    # Smart correction -- force PS back onto Table 1's exact marginal counts
    current_ps1 = np.sum(PS_discrete == 1)
    diff = ps1_count - current_ps1
    if diff > 0:
        candidates = np.where(PS_discrete == 2)[0]
        candidates_pref = candidates[disease_hr[candidates] <= 1.2]
        pool = candidates_pref if len(candidates_pref) >= diff else candidates
        chosen = np.random.choice(pool, size=min(diff, len(pool)), replace=False)
        PS_discrete[chosen] = 1
    elif diff < 0:
        candidates = np.where(PS_discrete == 1)[0]
        candidates_pref = candidates[disease_hr[candidates] > 1.0]
        pool = candidates_pref if len(candidates_pref) >= abs(diff) else candidates
        chosen = np.random.choice(pool, size=min(abs(diff), len(pool)), replace=False)
        PS_discrete[chosen] = 2

    PS_normalized = (PS_discrete - 1) / 3.0

    # ----------------------------------------------------------------
    # 6. Tumor response: CR / PR / SD / PD
    #    FIX B2 + B6: built directly and EXACTLY from Table 4's reported
    #    counts (cr_count, pr_count, sd_count), with WHICH patients land in
    #    which category driven by a transparent latent-propensity index
    #    (replaces the old 4-way Cholesky-copula draw, most of whose output
    #    was discarded/overwritten and never actually used).
    # ----------------------------------------------------------------
    target_orr_count = CR_COUNT + PR_COUNT

    disease_z = zscore(disease_hr)
    ps_z = zscore(PS_discrete)
    rad_z = zscore(PRIOR_RAD.astype(float))
    tol_z = zscore(tolerability)

    # Response propensity (self-proposed weights -- see reference table):
    # higher propensity = more likely to respond (CR/PR).
    response_propensity = (
        ORR_VS_PS * ps_z +
        ORR_VS_RADIATION * rad_z +
        ORR_VS_DISEASE_STATUS * disease_z +
        TOLERABILITY_ORR_WEIGHT * tol_z +
        np.random.standard_normal(n)
    )
    orr_idx = np.argsort(response_propensity)[-target_orr_count:]
    ORR = np.zeros(n, dtype=int)
    ORR[orr_idx] = 1

    cr_n = min(CR_COUNT, len(orr_idx))
    cr_idx = np.random.choice(orr_idx, cr_n, replace=False) if cr_n > 0 else np.array([], dtype=int)
    pr_idx = np.setdiff1d(orr_idx, cr_idx)

    non_responders = np.setdiff1d(np.arange(n), orr_idx)
    sd_n = min(SD_COUNT, len(non_responders))
    if sd_n > 0:
        pfsr_propensity = (
            PFSR_VS_PS * zscore(PS_discrete[non_responders]) +
            PFSR_VS_RADIATION * zscore(PRIOR_RAD[non_responders].astype(float)) +
            np.random.standard_normal(len(non_responders))
        )
        sd_rank = np.argsort(pfsr_propensity)[-sd_n:]
        sd_idx = non_responders[sd_rank]
    else:
        sd_idx = np.array([], dtype=int)
    pd_idx = np.setdiff1d(non_responders, sd_idx)

    PFSR = np.zeros(n, dtype=int)          # "disease control" flag: CR/PR/SD = 1, PD = 0
    PFSR[orr_idx] = 1
    PFSR[sd_idx] = 1

    response = np.empty(n, dtype=object)
    response[cr_idx] = "CR"
    response[pr_idx] = "PR"
    response[sd_idx] = "SD"
    response[pd_idx] = "PD"

    # ----------------------------------------------------------------
    # 7. Histology, tumor grade, race ("ethnicity" column -- see C2 note)
    # ----------------------------------------------------------------
    histology = expand_distribution(trial_config["histology"], n)
    grade = expand_distribution(trial_config["grades"], n)
    ethnicity = expand_distribution(trial_config["ethnicity"], n)

    # ================================================================
    # 8. OS / PFS generation
    # ================================================================
    arm_id = {'monk_a': 1, 'monk_b': 2, 'monk_c': 3, 'monk_d': 4}[arm_name]

    os_time = {'monk_a': monk_a_os_time, 'monk_b': monk_b_os_time,
               'monk_c': monk_c_os_time, 'monk_d': monk_d_os_time}[arm_name]
    os_surv = {'monk_a': monk_a_os_surv, 'monk_b': monk_b_os_surv,
               'monk_c': monk_c_os_surv, 'monk_d': monk_d_os_surv}[arm_name]

    os_ipd = guyot_reconstruct_ipd(os_time, os_surv, n,
                                   tot_events=n - ALIVE_COUNT,
                                   arm_id=arm_id, random_state=SEED + arm_id)

    # --- FIX B5: covariate-driven AFT adjustments, now actually applied ---
    # Age: Monk et al. 2009 explicitly reports age was NOT a significant OS
    # predictor in this cohort ("Analysis of prognostic factors showed that
    # age was not significant in these data", Discussion). This weight is
    # RETAINED anyway, at its original self-proposed magnitude, purely to
    # introduce plausible individual heterogeneity -- it is NOT trial-cited
    # and arguably should be reduced or zeroed for strict fidelity to this
    # specific trial. Flagged prominently in the weight reference table.
    age_factor = 1 - (age - np.mean(age)) / np.std(age) * OS_AGE_WEIGHT * 2
    age_factor = np.clip(age_factor, 0.4, 1.6)

    # Performance status: TRIAL-CITED (Fig 3A, multivariate Cox model,
    # PS1 vs PS0). OS_PS_WEIGHT (self-proposed) is a damping exponent on
    # top of this cited base hazard ratio.
    ps_time_multiplier = np.where(PS_discrete == 2,
                                   hr_to_time_multiplier(PS_HAZARD_RATIO_OS, OS_PS_WEIGHT),
                                   1.0)

    # Objective response -> OS: NOT reported in Monk 2009's OS prognostic
    # panel (only race, PS, irradiated-field site, Hispanic ethnicity were
    # examined there). Self-proposed placeholder HR (ORR_OS_HR_ASSUMED).
    orr_time_multiplier = np.where(ORR == 1,
                                    hr_to_time_multiplier(ORR_OS_HR_ASSUMED, OS_ORR_WEIGHT),
                                    1.0)

    # Disease status: TRIAL-CITED (Fig 3B), reference = Advanced (IVB).
    disease_time_multiplier = hr_to_time_multiplier(disease_hr, 1.0)

    os_covariate_factor = age_factor * ps_time_multiplier * orr_time_multiplier * disease_time_multiplier
    # FIX: rank-preserving empirical remap. The Methods section describes
    # covariate weights as deciding WHICH patient receives a given survival
    # time, never the arm's own curve shape -- but the code prior to this
    # fix multiplied each patient's raw Guyot time directly by their
    # covariate factor, which distorts the curve shape itself, then only
    # forced the MEDIAN back into place afterward. This let every point
    # away from the median drift from the digitized curve, worst in arms
    # with the most covariate heterogeneity (confirmed empirically against
    # the curve-level validation). The fix: covariates determine RANK only.
    # Each patient's final time is the raw, unadjusted Guyot value at the
    # rank their covariates earned them, so the full set of survival times
    # an arm is reconstructed to contain is identical to the raw Guyot
    # reconstruction before any rescaling, exactly as already claimed in
    # the Methods text.
    os_rank_driver = os_ipd['time'].values * os_covariate_factor
    os_ranks = np.argsort(np.argsort(os_rank_driver))
    os_raw_sorted = np.sort(os_ipd['time'].values)
    os_adjusted = os_raw_sorted[os_ranks]
    os_adjusted = np.clip(os_adjusted, 0.1, FOLLOWUP)

    # Force alignment with target_os (median rescale) -- applied AFTER the
    # rank-preserving remap so the arm-level median stays trial-exact while
    # the covariates still determine which patient receives which outcome.
    current_median = np.median(os_adjusted)
    scaling_factor = TARGET_OS / current_median if current_median > 0 else 1.0
    observed_os = np.round(os_adjusted * scaling_factor, 1)
    observed_os = np.clip(observed_os, 0.1, FOLLOWUP)

    event_os = os_ipd['event'].values
    alive = 1 - event_os

    # NOTE on OS_HAZARD_RATIO (arm-vs-PC treatment effect, e.g. 1.15 for
    # VC): intentionally NOT reapplied as a multiplier here. Each arm
    # already reconstructs from its OWN digitized KM curve and is rescaled
    # to its OWN trial-reported median, which already encodes that arm's
    # treatment effect. Multiplying by OS_HAZARD_RATIO again would double
    # count it. It is kept in trial_config purely as a citation/QC value --
    # see the printed cross-check at the end of this script.

    # ---------------- PFS ----------------
    pfs_time = {'monk_a': monk_a_pfs_time, 'monk_b': monk_b_pfs_time,
                'monk_c': monk_c_pfs_time, 'monk_d': monk_d_pfs_time}[arm_name]
    pfs_surv = {'monk_a': monk_a_pfs_surv, 'monk_b': monk_b_pfs_surv,
                'monk_c': monk_c_pfs_surv, 'monk_d': monk_d_pfs_surv}[arm_name]

    # FIX B4: exact trial-reported PFS event count (Fig 2 caption table),
    # replacing the flat int(0.92*n) approximation.
    pfs_ipd = guyot_reconstruct_ipd(pfs_time, pfs_surv, n,
                                    tot_events=n - PFS_ALIVE_COUNT,
                                    arm_id=arm_id, random_state=SEED + arm_id + 10)
    pfs_raw_time = pfs_ipd['time'].values.copy()
    observed_pfs = np.round(pfs_ipd['time'].values, 1)
    event_pfs = pfs_ipd['event'].values

    # Age -> PFS (self-proposed; no trial-specific PFS-age analysis reported)
    pfs_age_factor = 1 - (age - np.mean(age)) / np.std(age) * PFS_AGE_WEIGHT * 2
    observed_pfs = observed_pfs * pfs_age_factor

    # FIX B1: histology now MULTIPLIES (previously divided, which had the
    # sign backwards -- "worse" histologies were getting LONGER PFS).
    histology_weights = np.array([trial_config["pfs_histology_weight"][h] for h in histology])
    observed_pfs = observed_pfs * histology_weights

    # NEW: response -> PFS internal-consistency adjustment (self-proposed).
    # Ensures CR/PR patients skew toward longer observed PFS than SD/PD
    # patients, consistent with how "progression" is defined relative to
    # response -- previously response category and pfs_months were
    # generated by two fully independent processes with no linkage at all.
    pfs_response_multiplier = np.where(ORR == 1,
                                        hr_to_time_multiplier(PFS_RESPONSE_HR_ASSUMED, PFS_RESPONSE_WEIGHT),
                                        1.0)
    observed_pfs = observed_pfs * pfs_response_multiplier
    observed_pfs = np.clip(observed_pfs, 0.1, FOLLOWUP)

    # ================================================================
    # 9. Toxicities
    # ================================================================
    tox_counts_raw = trial_config["toxicity"]
    toxicity_names = list(tox_counts_raw.keys())
    tox_counts_arr = []
    for name in toxicity_names:
        prob = tox_counts_raw[name] / n
        prob_capped = max(0.0, min(0.98, prob))
        count_adjusted = int(round(prob_capped * n))
        tox_counts_arr.append(count_adjusted)
    tox_counts_arr = np.array(tox_counts_arr)

    n_tox = len(toxicity_names)
    tox_corr = np.full((n_tox, n_tox), 0.2)
    np.fill_diagonal(tox_corr, 1.0)

    hematologic = {'leucopenia', 'neutropenia', 'thrombocytopenia', 'anemia'}
    infectious = {'febrile with neutropenia', 'infection without neutropenia'}
    gi = {'nausea/vomiting', 'stomatitis'}
    for i, ti in enumerate(toxicity_names):
        for j, tj in enumerate(toxicity_names):
            if i != j:
                ti_l = ti.lower()
                tj_l = tj.lower()
                if any(ti_l in c and tj_l in c for c in [hematologic, infectious, gi]):
                    tox_corr[i, j] = 0.7
                elif (ti_l in hematologic and tj_l in infectious) or (ti_l in infectious and tj_l in hematologic):
                    tox_corr[i, j] = 0.5

    eigvals, eigvecs = np.linalg.eigh(tox_corr)
    eigvals[eigvals < 1e-6] = 1e-6
    tox_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(tox_corr)
    latent = np.random.standard_normal((n, n_tox)) @ L.T

    # NOTE (D2): the original script's arm-specific latent nudges (PC-arm
    # alopecia +1.0, GC-arm leucopenia/neutropenia -1.2/-1.5) have been
    # removed. Alopecia's grade>=3 target count is genuinely 0 for every
    # arm in this trial (the nudge could never surface in the output), and
    # the GC arm's lower myelosuppression is already fully and correctly
    # captured by tox_counts_arr (sourced directly from the xlsx) -- the
    # nudge duplicated something already handled correctly and only made
    # the code harder to follow.
    for j, tox in enumerate(toxicity_names):
        tox_lower = tox.lower()
        if tox_lower in hematologic:
            latent[:, j] += (0.4 * PRIOR_RAD + 0.3 * PRIOR_CHEMO + 0.25 * PRIOR_PLATINUM)

    tox_risk = (1 + 0.4 * PS_normalized + 0.35 * PRIOR_RAD + 0.35 * PRIOR_CHEMO +
                0.25 * PRIOR_PLATINUM + 0.15 * (age > 70) + tox_penalty)
    latent = latent * tox_risk[:, None]

    tox_matrix = np.zeros((n, n_tox), dtype=int)
    for j in range(n_tox):
        k = int(tox_counts_arr[j])
        if k <= 0:
            continue
        if k >= n - never_treated_count:
            tox_matrix[treated, j] = 1
            continue
        latent_treated = latent[treated, j]
        idx_treated = np.argsort(latent_treated)[-k:]
        global_idx = np.where(treated)[0][idx_treated]
        tox_matrix[global_idx, j] = 1

    toxicity_events = [
        [toxicity_names[j] for j in range(n_tox) if tox_matrix[i, j] == 1]
        for i in range(n)
    ]
    toxicity_count = [len(x) for x in toxicity_events]

    #============================== Correlating PFS with Toxicity========================
    toxicity_burden = np.array(toxicity_count) / n_tox
    pfs_toxicity_penalty = 1 - toxicity_burden * PFS_TOXICITY_WEIGHT
    observed_pfs = observed_pfs * pfs_toxicity_penalty
    observed_pfs = np.clip(observed_pfs, 0.1, FOLLOWUP)

    # FIX: rank-preserving empirical remap, mirroring the OS fix above.
    # observed_pfs at this point reflects age, histology, response, and
    # toxicity multipliers applied directly to survival time, which
    # distorts curve shape; the remap keeps each patient's covariate-earned
    # rank but substitutes the raw, unadjusted Guyot time at that rank.
    pfs_ranks = np.argsort(np.argsort(observed_pfs))
    pfs_raw_sorted = np.sort(pfs_raw_time)
    observed_pfs = pfs_raw_sorted[pfs_ranks]
    observed_pfs = np.clip(observed_pfs, 0.1, FOLLOWUP)

    # FIX B3: force alignment with target_pfs (median rescale), mirroring
    # the OS calibration step -- applied last, after every covariate
    # adjustment, so PFS median fidelity is guaranteed regardless of how
    # much age/histology/response/toxicity shifted it.
    current_pfs_median = np.median(observed_pfs)
    if current_pfs_median > 0:
        pfs_scaling_factor = TARGET_PFS / current_pfs_median
        observed_pfs = np.round(observed_pfs * pfs_scaling_factor, 1)
    observed_pfs = np.clip(observed_pfs, 0.1, FOLLOWUP)

    # ================================================================
    # 10. Cause of death
    # ================================================================
    dead_idx = np.where(alive == 0)[0]
    causes = np.full(n, "Alive", dtype=object)
    cause_dist = trial_config.get("causes", {})
    if len(dead_idx) > 0 and cause_dist:
        treatment_n = cause_dist.get("Treatment", 0)
        disease_n = cause_dist.get("Disease", 0)
        other_n = cause_dist.get("Other/Unknown", 0)
        toxicity_risk = np.array(toxicity_count)
        prognosis_risk = PS_normalized + (1 - ORR) + (1 - PFSR)
        combined_risk = toxicity_risk + prognosis_risk
        ranked_dead = dead_idx[np.argsort(combined_risk[dead_idx])[::-1]]
        treatment_idx = ranked_dead[:treatment_n]
        causes[treatment_idx] = "Treatment"
        remaining = np.setdiff1d(dead_idx, treatment_idx)
        disease_idx = np.random.choice(remaining, size=min(disease_n, len(remaining)), replace=False)
        causes[disease_idx] = "Disease"
        remaining = np.setdiff1d(remaining, disease_idx)
        if other_n > 0:
            other_idx = remaining[:other_n]
            causes[other_idx] = "Other/Unknown"
        causes = np.where(alive == 1, "Alive", causes)

    # ─── Uniform columns helper ────────────────────────────────────
    def uniform_column(value):
        """Robust version: Never let 'No Drug' become NaN"""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ['No Drug'] * n
        if isinstance(value, (tuple, list)):
            val_str = ",".join(map(str, value))
            return [val_str] * n
        val_str = str(value).strip()
        if val_str.lower() in ['no drug', 'none', 'nan', '', '0', '0.0']:
            return ['No Drug'] * n
        return [value] * n

    # ================================================================
    # Final DataFrame  (column names/order UNCHANGED from your original)
    # ================================================================
    df = pd.DataFrame({
        "platinum_drug": uniform_column(trial_config.get("platinum_drug")),
        "platinum_dose_value": uniform_column(trial_config.get("platinum_dose_value")),
        "platinum_dose_unit": uniform_column(trial_config.get("platinum_dose_unit")),
        "platinum_dose_method": uniform_column(trial_config.get("platinum_dose_method")),
        "platinum_infusion_hours": uniform_column(trial_config.get("platinum_infusion_hours")),
        "platinum_days_number": uniform_column(trial_config.get("platinum_days_number")),
        "platinum_days": uniform_column(trial_config.get("platinum_days")),
        "adjunct_1_name": uniform_column(trial_config.get("adjunct_1_name")),
        "adjunct_1_dose_value": uniform_column(trial_config.get("adjunct_1_dose_value")),
        "adjunct_1_dose_unit": uniform_column(trial_config.get("adjunct_1_dose_unit")),
        "adjunct_1_dose_method": uniform_column(trial_config.get("adjunct_1_dose_method")),
        "adjunct_1_infusion_hours": uniform_column(trial_config.get("adjunct_1_infusion_hours")),
        "adjunct_1_days_number": uniform_column(trial_config.get("adjunct_1_days_number")),
        "adjunct_1_days": uniform_column(trial_config.get("adjunct_1_days")),
        "adjunct_1_same_day_platinum": uniform_column(trial_config.get("adjunct_1_same_day_platinum")),
        "adjunct_2_name": uniform_column(trial_config.get("adjunct_2_name")),
        "adjunct_2_dose_value": uniform_column(trial_config.get("adjunct_2_dose_value")),
        "adjunct_2_dose_unit": uniform_column(trial_config.get("adjunct_2_dose_unit")),
        "adjunct_2_dose_method": uniform_column(trial_config.get("adjunct_2_dose_method")),
        "adjunct_2_infusion_hours": uniform_column(trial_config.get("adjunct_2_infusion_hours")),
        "adjunct_2_days_number": uniform_column(trial_config.get("adjunct_2_days_number")),
        "adjunct_2_days": uniform_column(trial_config.get("adjunct_2_days")),
        "adjunct_2_same_day_platinum": uniform_column(trial_config.get("adjunct_2_same_day_platinum")),
        "adjunct_3_name": uniform_column(trial_config.get("adjunct_3_name")),
        "adjunct_3_dose_value": uniform_column(trial_config.get("adjunct_3_dose_value")),
        "adjunct_3_dose_unit": uniform_column(trial_config.get("adjunct_3_dose_unit")),
        "adjunct_3_dose_method": uniform_column(trial_config.get("adjunct_3_dose_method")),
        "adjunct_3_infusion_hours": uniform_column(trial_config.get("adjunct_3_infusion_hours")),
        "adjunct_3_days_number": uniform_column(trial_config.get("adjunct_3_days_number")),
        "adjunct_3_days": uniform_column(trial_config.get("adjunct_3_days")),
        "adjunct_3_same_day_platinum": uniform_column(trial_config.get("adjunct_3_same_day_platinum")),
        "supportive_1_name": uniform_column(trial_config.get("supportive_1_name")),
        "supportive_1_dose_value": uniform_column(trial_config.get("supportive_1_dose_value")),
        "supportive_1_dose_unit": uniform_column(trial_config.get("supportive_1_dose_unit")),
        "supportive_1_route": uniform_column(trial_config.get("supportive_1_route")),
        "supportive_1_start_day": uniform_column(trial_config.get("supportive_1_start_day")),
        "supportive_1_duration_value": uniform_column(trial_config.get("supportive_1_duration_value")),
        "supportive_1_duration_unit": uniform_column(trial_config.get("supportive_1_duration_unit")),
        "supportive_1_frequency_per_day": uniform_column(trial_config.get("supportive_1_frequency_per_day")),
        "supportive_1_frequency_interval_hours": uniform_column(trial_config.get("supportive_1_frequency_interval_hours")),
        "supportive_2_name": uniform_column(trial_config.get("supportive_2_name")),
        "supportive_2_dose_value": uniform_column(trial_config.get("supportive_2_dose_value")),
        "supportive_2_dose_unit": uniform_column(trial_config.get("supportive_2_dose_unit")),
        "supportive_2_route": uniform_column(trial_config.get("supportive_2_route")),
        "supportive_2_start_day": uniform_column(trial_config.get("supportive_2_start_day")),
        "supportive_2_duration_value": uniform_column(trial_config.get("supportive_2_duration_value")),
        "supportive_2_duration_unit": uniform_column(trial_config.get("supportive_2_duration_unit")),
        "supportive_2_frequency_per_day": uniform_column(trial_config.get("supportive_2_frequency_per_day")),
        "supportive_2_frequency_interval_hours": uniform_column(trial_config.get("supportive_2_frequency_interval_hours")),
        "supportive_3_name": uniform_column(trial_config.get("supportive_3_name")),
        "supportive_3_dose_value": uniform_column(trial_config.get("supportive_3_dose_value")),
        "supportive_3_dose_unit": uniform_column(trial_config.get("supportive_3_dose_unit")),
        "supportive_3_route": uniform_column(trial_config.get("supportive_3_route")),
        "supportive_3_start_day": uniform_column(trial_config.get("supportive_3_start_day")),
        "supportive_3_duration_value": uniform_column(trial_config.get("supportive_3_duration_value")),
        "supportive_3_duration_unit": uniform_column(trial_config.get("supportive_3_duration_unit")),
        "supportive_3_frequency_per_day": uniform_column(trial_config.get("supportive_3_frequency_per_day")),
        "supportive_3_frequency_interval_hours": uniform_column(trial_config.get("supportive_3_frequency_interval_hours")),
        "cycle_length_days": uniform_column(trial_config.get("cycle_length_days")),
        "followup": uniform_column(trial_config.get("followup")),

        "response": response,
        "orr": ORR,
        "pfsr": PFSR,
        "pfs_months": observed_pfs,
        "pfs_event": event_pfs,
        "os_months": observed_os,
        "os_event": event_os,
        "age": age,
        "treatment": treatment,
        "radiation": PRIOR_RAD.astype(bool),
        "prior_chemotherapy_exposure": PRIOR_CHEMO.astype(bool),
        "prior_surgery_exposure": PRIOR_SURGERY.astype(bool),
        "prior_platinum_exposure": PRIOR_PLATINUM.astype(bool),
        "ethnicity": ethnicity,
        "performance_status": PS_discrete,
        "figo_stage": figo_stage_list,
        "disease_nature": disease_nature_list,
        "histology/cell_type": histology,
        "tumor_grade": grade,
        "vital_status": np.where(alive == 1, "Alive", "Dead"),
        "cause_of_death": causes,
        "toxicity_events": toxicity_events,
        "toxicity_count": toxicity_count,
        "os_time_for_km": observed_os,
        "os_event_for_km": event_os,
        "pfs_time_for_km": observed_pfs,
        "pfs_event_for_km": event_pfs
    }).sample(frac=1, random_state=42).reset_index(drop=True)

    return df


# ================================================================
# Usage - Creating the four arms and combined dataset
# ================================================================
df_monk_a = generate_clinical_data(TRIALS["monk_a"], arm_name='monk_a')
df_monk_b = generate_clinical_data(TRIALS["monk_b"], arm_name='monk_b')
df_monk_c = generate_clinical_data(TRIALS["monk_c"], arm_name='monk_c')
df_monk_d = generate_clinical_data(TRIALS["monk_d"], arm_name='monk_d')

df_monk = pd.concat([df_monk_a, df_monk_b, df_monk_c, df_monk_d],
                    ignore_index=True)\
           .sample(frac=1, random_state=42)\
           .reset_index(drop=True)
df_monk.to_excel("monk_km.xlsx")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluating Datasets
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ====================== GENERALIZED EVALUATION FUNCTION ======================

def standardize_dict_order(output_dict, target_dict):
    """Force output dict to follow exact key order of target dict"""
    if not isinstance(output_dict, dict) or not isinstance(target_dict, dict):
        return output_dict
    if target_dict:
        sample_value = next(iter(target_dict.values()))
        default = 0 if isinstance(sample_value, (int, float)) else 'Missing'
    else:
        default = 0
    return {k: output_dict.get(k, default) for k in target_dict.keys()}


def get_toxicity_counts(toxicity_events, reference_toxicity):
    """Return toxicity counts in EXACT target order"""
    flat_list = [tox for patient in toxicity_events for tox in patient]
    counts = Counter(flat_list)
    return {tox: counts.get(tox, 0) for tox in reference_toxicity.keys()}


def evaluate_trial(trial_name, df):
    trial_cfg = TRIALS[trial_name]

    # FIX E: added target_pfs and a Tumor Response (CR/PR/SD/PD) target,
    # both previously unchecked -- this is exactly what let the PFS-median
    # and stable-disease-count bugs go undetected before.
    pd_target = trial_cfg['patient_number'] - trial_cfg['cr_count'] - trial_cfg['pr_count'] - trial_cfg['sd_count']
    response_target = {"CR": trial_cfg['cr_count'], "PR": trial_cfg['pr_count'],
                        "SD": trial_cfg['sd_count'], "PD": pd_target}
    # "pfsr" column = disease-control flag (CR/PR/SD=1, PD=0), so its correct
    # comparison target is (CR+PR+SD)/n -- NOT trial_cfg['target_pfsr'], which
    # (per the B2 finding) is actually just Table 4's Stable-Disease rate on
    # its own and would make this row look like a false failure.
    disease_control_rate_target = round(
        (trial_cfg['cr_count'] + trial_cfg['pr_count'] + trial_cfg['sd_count']) / trial_cfg['patient_number'], 3
    )

    constants = [
        trial_cfg['patient_number'],
        trial_cfg['followup'],
        trial_cfg['target_orr'],
        disease_control_rate_target,
        response_target,
        trial_cfg['target_os'],
        trial_cfg['target_pfs'],
        trial_cfg['target_age'],
        trial_cfg['age_min'],
        trial_cfg['age_max'],
        trial_cfg['age_sd'],
        trial_cfg['alive_count'],
        trial_cfg['ethnicity'],
        trial_cfg['histology'],
        trial_cfg['disease_nature'],
        trial_cfg['disease_stage'],
        trial_cfg['grades'],
        trial_cfg['causes'],
        trial_cfg['performance_status'],
        trial_cfg['toxicity']
    ]

    output = [
        len(df),
        float(df['followup'].iloc[0]),
        round(df['orr'].mean(), 2),
        round(df['pfsr'].mean(), 2),
        df['response'].value_counts().to_dict(),
        round(df['os_months'].median(), 2),
        round(df['pfs_months'].median(), 2),
        int(df['age'].median()),
        int(df['age'].min()),
        int(df['age'].max()),
        round(df['age'].std(ddof=0), 2),
        (df['vital_status'] == "Alive").sum(),
        df['ethnicity'].value_counts().to_dict(),
        df['histology/cell_type'].value_counts().to_dict(),
        df['disease_nature'].value_counts().to_dict(),
        df['figo_stage'].value_counts().to_dict(),
        df['tumor_grade'].value_counts().to_dict(),
        df['cause_of_death'].value_counts().to_dict(),
        df['performance_status'].value_counts().to_dict(),
        get_toxicity_counts(df["toxicity_events"], trial_cfg['toxicity'])
    ]

    row_names = [
        "Patient Number", "Followup", "ORR", "PFSR (disease control rate)",
        "Tumor Response (CR/PR/SD/PD)",
        "Overall Survival (months)", "Progression-Free Survival (months)",
        "Age", "Minimum Age", "Maximum Age", "Age SD", "Alive Counts", "Ethnicity",
        "Histology", "Disease Type", "Disease Stage",
        "Tumor Grade", "Death Cause", "Performance Status", "Toxicity"
    ]

    eval_df = pd.DataFrame({
        "output_values": output,
        "target_values": constants
    }, index=row_names)

    for i, row in enumerate(row_names):
        target_val = constants[i]
        output_val = output[i]
        if isinstance(target_val, dict) and isinstance(output_val, dict):
            eval_df.at[row, "output_values"] = standardize_dict_order(output_val, target_val)

    return eval_df


# ====================== LOOP THROUGH ALL TRIALS ======================

trial_datasets = {
    "monk_a": df_monk_a,
    "monk_b": df_monk_b,
    "monk_c": df_monk_c,
    "monk_d": df_monk_d
}
evaluation_results = {}

for trial_name, df in trial_datasets.items():
    evaluation_results[trial_name] = evaluate_trial(trial_name, df)

combined_evaluation_monk = pd.concat(evaluation_results, axis=1)

pd.set_option('display.max_colwidth', 120)
combined_evaluation_monk.to_excel("monk_evaluation_km.xlsx")

# ====================== OS-HAZARD-RATIO QC CROSS-CHECK ======================
# Approximate consistency check only: an AFT median ratio is NOT
# mathematically identical to a Cox PH hazard ratio in general (they
# coincide exactly only under specific parametric assumptions, e.g.
# Weibull) -- treat this as a sanity check, not a formal equivalence test.
print("\n--- OS hazard-ratio QC cross-check (median-ratio approximation) ---")
pc_median = TRIALS["monk_a"]["target_os"]
for arm in ["monk_a", "monk_b", "monk_c", "monk_d"]:
    implied = pc_median / TRIALS[arm]["target_os"]
    reported = TRIALS[arm]["os_hazard_ratio"]
    print(f"{arm}: implied median-ratio={implied:.3f}  vs  reported OS HR={reported:.3f}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SELF-PROPOSED CORRELATION / WEIGHT REFERENCE TABLE
# (as requested: every numeric assumption the pipeline introduces on its
#  own, i.e. NOT taken directly from a reported trial statistic, so you can
#  attach literature citations before this is used in any publication.)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
weight_reference_table = pd.DataFrame([
    {"category": "OS modeling", "parameter": "os_age_weight", "value": 0.2,
     "applies_to": "Age -> OS (AFT exponent)", "cited_in_monk2009": "NO -- CONTRADICTS trial finding",
     "note": "Monk 2009 explicitly reports age was NOT a significant OS predictor in this cohort. Retained only for individual heterogeneity; consider reducing to 0."},
    {"category": "OS modeling", "parameter": "os_ps_weight", "value": 0.4,
     "applies_to": "Performance status -> OS (damping exponent on cited HR)",
     "cited_in_monk2009": "Base HR=1.799 IS cited (Fig 3A); the 0.4 damping exponent is self-proposed.",
     "note": "weight=1.0 would apply the full reported effect."},
    {"category": "OS modeling", "parameter": "os_orr_weight / ORR_OS_HR_ASSUMED", "value": "0.3 / 0.70",
     "applies_to": "Response (CR/PR) -> OS", "cited_in_monk2009": "NO",
     "note": "Monk 2009's OS prognostic-factor panel (Fig 3A) does not examine response status; general oncology-literature association assumed."},
    {"category": "PFS modeling", "parameter": "pfs_age_weight", "value": 0.15,
     "applies_to": "Age -> PFS (AFT exponent)", "cited_in_monk2009": "NO", "note": "No trial-specific PFS-age analysis reported."},
    {"category": "PFS modeling", "parameter": "pfs_histology_weight (dict)",
     "value": "0.50-1.15 by histology", "applies_to": "Histology -> PFS multiplier",
     "cited_in_monk2009": "NO", "note": "Directional assumptions (e.g. adenocarcinoma worse, villoglandular better) are broadly consistent with general cervical-cancer histology literature but are uncited placeholders."},
    {"category": "PFS modeling", "parameter": "pfs_response_weight / PFS_RESPONSE_HR_ASSUMED", "value": "0.3 / 0.65",
     "applies_to": "Response (CR/PR) -> PFS", "cited_in_monk2009": "NO",
     "note": "Added to fix internal inconsistency between RECIST response category and reconstructed PFS time (previously fully independent processes)."},
    {"category": "PFS modeling", "parameter": "pfs_toxicity_weight", "value": 0.1,
     "applies_to": "Toxicity burden -> PFS", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "orr_vs_ps", "value": -0.30,
     "applies_to": "Response propensity index", "cited_in_monk2009": "NO",
     "note": "Direction (worse PS -> lower response) is clinically standard; magnitude uncited."},
    {"category": "Response modeling", "parameter": "orr_vs_radiation", "value": 0.20,
     "applies_to": "Response propensity index", "cited_in_monk2009": "NO",
     "note": "Sign preserved from your original design. NB: Fig 3A's related-but-distinct variable ('target lesion in irradiated field') is reported as an ADVERSE OS factor (HR 1.408) -- verify this sign/variable choice before citing."},
    {"category": "Response modeling", "parameter": "orr_vs_disease_status", "value": -0.35,
     "applies_to": "Response propensity index", "cited_in_monk2009": "NO",
     "note": "New parameter; replaces the original script's ad hoc random-flip logic with the same disease_hr covariate used elsewhere."},
    {"category": "Response modeling", "parameter": "tolerability_orr_weight", "value": 0.20,
     "applies_to": "Response propensity index", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "pfsr_vs_ps", "value": -0.35,
     "applies_to": "Stable-vs-progressive disease propensity index", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "pfsr_vs_radiation", "value": 0.25,
     "applies_to": "Stable-vs-progressive disease propensity index", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Tolerability", "parameter": "tolerability formula constants", "value": "0.82, 0.5, 0.09",
     "applies_to": "completed_6*0.82 + Normal(0.5,0.09)", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Tolerability", "parameter": "tox_penalty weight", "value": 0.18,
     "applies_to": "(1-tolerability)*0.18", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Performance status", "parameter": "PS worsening (tolerability)", "value": "threshold 0.35, fraction 0.10",
     "applies_to": "Low-tolerability patients' PS", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Performance status", "parameter": "PS worsening (disease status)", "value": "HR>1.2 deterministic, HR>0.95 at p=0.2",
     "applies_to": "Recurrent-disease patients' PS", "cited_in_monk2009": "NO", "note": "Unchanged from your original design."},
    {"category": "Toxicity", "parameter": "tox_risk formula weights", "value": "0.4 (PS), 0.35 (rad), 0.35 (chemo), 0.25 (platinum), 0.15 (age>70)",
     "applies_to": "Per-patient toxicity latent-risk multiplier", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Toxicity", "parameter": "hematologic latent boost", "value": "0.4 (rad), 0.3 (chemo), 0.25 (platinum)",
     "applies_to": "Hematologic toxicity latent score", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Toxicity", "parameter": "tox_corr matrix", "value": "0.2 baseline / 0.7 within-cluster / 0.5 heme-infection cross",
     "applies_to": "Cross-toxicity correlation structure", "cited_in_monk2009": "NO", "note": "-"},
    {"category": "Demographics", "parameter": "age_sd", "value": "(age_max-age_min)/4",
     "applies_to": "Age distribution spread", "cited_in_monk2009": "NO",
     "note": "Range/4 heuristic; the paper does not report a true SD for age."},
    {"category": "Technical / reconstruction", "parameter": "Guyot censoring scheme", "value": "uniform(last_event, 1.1x last_event)",
     "applies_to": "OS & PFS censoring-time placement", "cited_in_monk2009": "NO",
     "note": "Simplification: Monk 2009's Fig 2 does not publish a number-at-risk table to calibrate censoring timing more precisely."},
])

print("\n--- Self-proposed weight reference table (for later citation) ---")
print(weight_reference_table.to_string(index=False))
weight_reference_table.to_excel("monk_correlation_weight_reference.xlsx", index=False)

combined_evaluation_monk