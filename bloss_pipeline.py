#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BLOSS 2002 -- SYNTHETIC IPD GENERATION PIPELINE (CORRECTED)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Source: Bloss JD, Blessing JA, Behrens BC, et al. Randomized Trial of
# Cisplatin and Ifosfamide With or Without Bleomycin in Squamous Carcinoma
# of the Cervix: A Gynecologic Oncology Group Study. J Clin Oncol.
# 2002;20(7):1832-1837.
#
# Two arms: CI (cisplatin + ifosfamide, n=146) and CIB (cisplatin +
# ifosfamide + bleomycin, n=141). Same architecture, helper functions, base
# lists, column names/order, and evaluation-table structure as the
# Monk/Long/Kitagawa/Miller pipelines, using Guyot reconstruction exclusively
# for OS/PFS (your own digitized curve points -- no Weibull).
#
# CHANGE LOG relative to the version you pasted
# ------------------------------------------------------------------------
#  [Bug fixes -- unambiguous, evidence-based]
#  B1. `pfs_time = pfs_time / histology_weights` DIVIDED by the histology
#      weight (same sign-flip bug found and fixed in every prior trial in
#      this series). Since Bloss's eligibility REQUIRED squamous cell
#      carcinoma (all 287 patients are squamous -- see C-notes below), this
#      bug had zero practical effect here (the multiplier is 1.00 for every
#      patient), but is fixed anyway for architectural consistency.
#  B2. Response was built as a simple ORR/PFSR binary forced-count draw with
#      NO real CR/PR/SD/PD categorical construction at all (no `response`
#      categories beyond a crude SD/PD split of non-responders using
#      entirely made-up target_pfsr=0.40/0.42 "6-month PFS rate" guesses,
#      the same target_pfsr category error found and fixed in Monk/B2).
#      Replaced with the exact-count + latent-propensity-index approach
#      used in the corrected Monk/Long/Kitagawa/Miller pipelines. NOTE:
#      Bloss's Table 3 gives only Responders/Nonresponders (47/99 for CI,
#      44/97 for CIB) -- there is no CR/PR split or SD/PD split reported
#      anywhere in this paper (checked Tables 3 and 4 directly), so those
#      splits are self-proposed here, flagged in the weight reference table.
#      Unlike Long/Kitagawa/Miller, Bloss needs NO "NE" response value: the
#      paper explicitly classifies the 6 patients who never received
#      treatment as nonresponders in its own ITT convention, so all 287
#      patients get a definite CR/PR/SD/PD outcome.
#  B3. No target existed for median PFS, and OS/PFS generation used a
#      home-grown risk-score-sorting + "scale to median" scheme built from
#      scratch (not `guyot_reconstruct_ipd`) despite already having
#      digitized curve data sitting unused above it. Replaced with
#      `guyot_reconstruct_ipd` + the empirical-remap + full-median-
#      correction pattern from the Miller pipeline (see B4-B6 below).
#      `target_pfs` added directly from the paper (4.6/5.1 months).
#  B4. Guyot reconstruction was called with NO `tot_events`, so the event/
#      censoring split was whatever the raw digitized curve's drop pattern
#      implied, uncalibrated -- then a SEPARATE, disconnected mechanism
#      ("force exact alive count" by re-sorting os_time and overwriting
#      event_os from scratch) was bolted on afterward, discarding the
#      Guyot-derived events entirely. Consolidated into one pass: Guyot
#      reconstruction directly targets alive_count/pfs_alive_count via
#      `tot_events`, matching every other trial in this series.
#  B5. The curve-simplification bug found in Miller's pipeline (WebPlot-
#      Digitizer noise -- many closely-spaced points that are not real
#      distinct events -- makes `guyot_reconstruct_ipd`'s "at least one
#      event per detected drop" floor systematically front-load events and
#      bias the reconstructed median down) applies here too: Bloss's raw
#      OS curves have 100+ digitized points each. `simplify_km_curve()`
#      (from the Miller pipeline) is applied here as well.
#  B6. PS -> OS/PFS hazards were NOT applied to survival TIMES at all in
#      your version (the age-adjustment step used `OS_AGE_WEIGHT` etc., but
#      `ps_hazard`/`response_hazard`/`disease_hazard` were computed into an
#      `os_risk_score` used only to RE-ORDER which patient gets which
#      pre-computed time -- a rank-matching scheme, not a magnitude
#      adjustment -- so the actual hazard MAGNITUDES never influenced the
#      output distribution's spread, just who got which value from an
#      already-fixed set of times). Fixed to use real multiplicative AFT
#      adjustments, the same architecture as every other trial. NOTE: this
#      trial gives you something better than any other trial in this
#      series -- DIRECTLY CITED PS hazard ratios for BOTH endpoints (PS1 vs
#      PS0: OS HR=1.37, PFS HR=1.40; PS2 vs PS0: OS HR=1.70, PFS HR=1.54,
#      both P<.05) -- so PS_HAZARD_RATIO_OS_ASSUMED-style borrowing from
#      Monk is NOT needed here.
#
#  [Data-mapping corrections]
#  C1. `disease_nature` was constructed from Table 1's "Site of disease"
#      row (Pelvis only / Extra-pelvic), which describes tumor LOCATION,
#      not whether disease is persistent/recurrent/advanced -- a genuine
#      category confusion (these are unrelated clinical concepts). Site of
#      disease is real, well-cited data (and a real prognostic factor for
#      response, P<.001), so it is kept -- but as an INTERNAL covariate
#      driving response propensity, not as the `disease_nature` output
#      column. `disease_nature` is rebuilt instead using the one
#      partially-cited anchor this paper does give: "32 entered with stage
#      IVB disease and had no prior radiation or chemotherapy" (14 CI/18
#      CIB, matching the never-treated counts exactly) -> Advanced (IVB).
#      The Persistent/Recurrent split of the remaining prior-treated
#      patients is NOT reported and is self-proposed (flagged).
#  C2. `treatment_bloss_a/b` labeled the 30/33 "prior cisplatin (as
#      radiosensitizer)" patients as "prior_cisplatin", which does NOT
#      trigger `prior_radiation=1` in `derive_prior_treatments()` -- but
#      "as a radiosensitizer" means these patients definitionally also had
#      concurrent radiation. Recoded to "chemoradiotherapy" (same fix
#      applied to the analogous situation in Miller), which correctly
#      triggers radiation+chemo+platinum together.
#  C3. `never_treated_count=14/18` was actually the count of stage-IVB,
#      treatment-naive patients (a baseline demographic subgroup) -- not
#      "never received protocol therapy" (a trial-conduct/compliance
#      concept). Table 1's actual "Patients not treated" row gives 2/4,
#      which is the correct value for this parameter (same category
#      confusion, and same fix, as Miller's C3).
#  C4. `performance_status = [0,1,2,3,4]` used raw GOG codes; recoded to the
#      shift-by-one convention (code = GOG_PS + 1) used in every other
#      trial in this series, so merged column values mean the same thing
#      across all 5 trials.
#  C5. Toxicity counts (tox_bloss_a/b) were independently re-derived here
#      directly from Table 2's raw grade-1-4 counts (the text extraction of
#      this table is badly garbled -- digits run together with no spacing
#      in several rows -- so this was cross-checked against the page image
#      AND against the paper's own narrative statistics, e.g. "pulmonary
#      toxicity 18% v 5%" and "central neurotoxicity 10% v 7%", both of
#      which match exactly). Your original tox_bloss_a/b values matched
#      this independent re-derivation for every category checked -- no
#      changes were needed, they were already correct.
#
#  [Hygiene / dead code removed]
#  D1. Removed `shape_k`, `age_latent_risk`, `asian_weight`/`black_weight`/
#      `american_indian_weight`, `orr_vs_pfsr`, `ps_vs_radiation`,
#      `initial_probabilities`, and the `correlated_binary`/`adjust_binary`
#      machinery (never meaningfully used -- same dead-parameter pattern
#      found in every prior trial in this series).
#
#  [New data the original script didn't use at all]
#  E1. Added `site_of_disease` (Pelvis only / Extra-pelvic, Table 1, exact
#      counts) as an internal response-propensity covariate -- a real,
#      strongly cited finding (pelvic-only disease: 24.1%/16.1% response;
#      extra-pelvic: 37.0%/41.2%; P<.001) that the original script never
#      used at all, despite having misappropriated its data into
#      `disease_nature`.
# ============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Importing Libraries
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
from collections import Counter

SEED = 42
BLOSS_A_NUMBER = 146
BLOSS_B_NUMBER = 141

#=======================================================
# LITERATURE- AND TRIAL-DERIVED EFFECT SIZES (CITED)
#=======================================================
# Directly reported in Bloss et al. 2002 (Results, "Response and Survival"
# paragraph) -- unlike every other trial in this series, BOTH endpoints'
# PS hazard ratios are given directly by this paper, so no borrowing from
# Monk 2009 is needed here.
PS1_HAZARD_RATIO_OS = 1.37   # PS1 vs PS0, P=.009
PS2_HAZARD_RATIO_OS = 1.70   # PS2 vs PS0, P=.009
PS1_HAZARD_RATIO_PFS = 1.40  # PS1 vs PS0, P=.013
PS2_HAZARD_RATIO_PFS = 1.54  # PS2 vs PS0, P=.013

# NOT reported in Bloss et al. 2002 -- self-proposed placeholders requiring
# an external literature citation. Catalogued in the weight reference table.
ORR_OS_HR_ASSUMED = 0.70
PFS_RESPONSE_HR_ASSUMED = 0.65

# Self-proposed magnitude on a REAL, strongly cited directional finding:
# response rate differs sharply by site of disease (pelvic-only vs
# extra-pelvic), P<.001 (Results, "Response with regard to site of
# disease"). The site_of_disease covariate itself is trial-exact (Table 1);
# the WEIGHT linking it to response propensity is self-proposed.
ORR_VS_PELVIC_ONLY = -0.45   # pelvic-only disease -> lower response propensity


def hr_to_time_multiplier(hazard_ratio, weight=1.0):
    """AFT-style time multiplier from a hazard ratio (exact under Weibull PH;
    a practical approximation otherwise). weight=1 applies HR as given."""
    hazard_ratio = np.asarray(hazard_ratio, dtype=float)
    return hazard_ratio ** (-float(weight))


#=======================================================
# DEFINING BASE LISTS
# (identical to the corrected Monk/Long/Kitagawa/Miller pipelines' lists,
#  for cross-trial merge compatibility)
#=======================================================

histology_base = ["Squamous Cell Carcinoma", "Adenocarcinoma", "Adenosquamous", "Mucinous Adenocarcinoma",
                  "Clear Cell", "Endometrioid", "Villoglandular", "Undifferentiated Carcinoma", "Not Specified"]

ethnicity_base = ["White", "Black", "Asian", "American Indian", "Hispanic", "Filipino", "Unspecified"]

# FIX C1: "Non-specified" -> "Advanced (IVB)", matching the corrected schema.
disease_nature = ["Persistent", "Recurrent", "Advanced (IVB)"]

figo_stage = ["I", "II", "III", "IVA", "IVB", "Unknown"]

grade_base = ["Grade 1", "Grade 2", "Grade 3", "Grade Unspecified"]

death_cause_base = ["Treatment", "Disease", "Other/Unknown"]

dead_alive = ["Alive", "Dead", "Unspecified"]

# FIX C4: shift-by-one convention (code = GOG_PS + 1), matching every other
# trial in this series.
performance_status = [1, 2, 3, 4]

prior_treatment_base = [
    "none", "radiation_only", "chemotherapy_only", "surgery_only",
    "surgery_plus_radiation", "radiation_plus_chemotherapy",
    "surgery_radiation_chemotherapy", "prior_chemo_platinum", "prior_cisplatin",
    "chemoradiotherapy", "surgery_intent_curative", "surgery_intent_palliative", "unknown"
]

toxicity_base = ['Leucopenia', 'Neutropenia', 'Thrombocytopenia', 'Anemia', 'Other hematologic',
                 'Allergic reactions', 'Inner ear/hearing', 'Other auditory', 'Thrombosis embolism',
                 'Cardiac left ventricular function', 'Other cardiovascular', 'Fatigue',
                 'Other constitutional', 'Alopecia', 'Dermatologic', 'Nausea/vomiting', 'Stomatitis',
                 'Other Gastrointestinal', 'Creatinine', 'Hematuria', 'Other genitourinary/renal',
                 'Hemorrhage', 'Hepatic', 'Febrile with neutropenia', 'Infection without neutropenia',
                 'Other infection/fever', 'Lymphatics', 'Metabolic', 'Musculoskeletal', 'Peripheral neuropathy',
                 'Other neurological', 'Ocular/visual', 'Pain', 'Pulmonary', 'Vascular', 'Weight loss', 'Sexual']

#===========================================================
# DRUG COMBINATION (unchanged -- verified against "Patients and Methods":
# CI = cisplatin 50mg/m2 day1 + ifosfamide 5.0g/m2/24h day1 + mesna
# 6g/m2 concurrent + 12h after (36h total); CIB = bleomycin 30u/24h day1,
# followed by cisplatin+ifosfamide+mesna as in CI, on day 2)
#===========================================================
DRUG_bloss_a = {
    "Platinum_Drug": "cisplatin", "Platinum_Dose_Value": 50, "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based", "Platinum_Infusion_Duration_Hours": 1,
    "Platinum_Days_Number": 1, "Platinum_Day": 1,

    "Adjunct_Drug_1": "ifosfamide", "Adjunct_Drug_1_Dose_Value": 5000, "Adjunct_Drug_1_Dose_Unit": "mg/m2",
    "Adjunct_Drug_1_Dose_Method": "BSA-based", "Adjunct_Drug_1_Infusion_Duration_Hours": 24,
    "Adjunct_Drug_1_Days_Number": 1, "Adjunct_Drug_1_Days": 1, "Adjunct_Drug_1_Same_Day_As_Platinum": 1,

    "Adjunct_Drug_2": 'No Drug', "Adjunct_Drug_2_Dose_Value": 0, "Adjunct_Drug_2_Dose_Unit": 'No Drug',
    "Adjunct_Drug_2_Dose_Method": 'No Drug', "Adjunct_Drug_2_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_2_Days_Number": 0, "Adjunct_Drug_2_Days": 0, "Adjunct_Drug_2_Same_Day_As_Platinum": 0,

    "Adjunct_Drug_3": 'No Drug', "Adjunct_Drug_3_Dose_Value": 0, "Adjunct_Drug_3_Dose_Unit": 'No Drug',
    "Adjunct_Drug_3_Dose_Method": 'No Drug', "Adjunct_Drug_3_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_3_Days_Number": 0, "Adjunct_Drug_3_Days": 0, "Adjunct_Drug_3_Same_Day_As_Platinum": 0,

    "Supportive_Drug_1": "mesna", "Supportive_Drug_1_Dose_Value": 6000, "Supportive_Drug_1_Dose_Unit": "mg/m2",
    "Supportive_Drug_1_Route": "IV", "Supportive_Drug_1_Start_Day": 1, "Supportive_Drug_1_Duration_Value": 36,
    "Supportive_Drug_1_Duration_Unit": "hours", "Supportive_Drug_1_Frequency_Times_Per_Day": 0,
    "Supportive_Drug_1_Frequency_Interval_Hours": 0, "Supportive_Drug_1_Frequency_Description": "continuous infusion",
    "Supportive_Drug_1_Dose_Is_Range": 0, "Supportive_Drug_1_Dose_Min": 6000, "Supportive_Drug_1_Dose_Max": 6000,

    "Supportive_Drug_2": 'No Drug', "Supportive_Drug_2_Dose_Value": 0, "Supportive_Drug_2_Dose_Unit": 'No Drug',
    "Supportive_Drug_2_Route": 'No Drug', "Supportive_Drug_2_Start_Day": 0, "Supportive_Drug_2_Duration_Value": 0,
    "Supportive_Drug_2_Duration_Unit": 'No Drug', "Supportive_Drug_2_Frequency_Times_Per_Day": 0,
    "Supportive_Drug_2_Frequency_Interval_Hours": 0, "Supportive_Drug_2_Frequency_Description": 'No Drug',
    "Supportive_Drug_2_Dose_Is_Range": 0, "Supportive_Drug_2_Dose_Min": 0, "Supportive_Drug_2_Dose_Max": 0,

    "Supportive_Drug_3": 'No Drug', "Supportive_Drug_3_Dose_Value": 0, "Supportive_Drug_3_Dose_Unit": 'No Drug',
    "Supportive_Drug_3_Route": 'No Drug', "Supportive_Drug_3_Start_Day": 0, "Supportive_Drug_3_Duration_Value": 0,
    "Supportive_Drug_3_Duration_Unit": 'No Drug', "Supportive_Drug_3_Frequency_Times_Per_Day": 0,
    "Supportive_Drug_3_Frequency_Interval_Hours": 0, "Supportive_Drug_3_Frequency_Description": 'No Drug',
    "Supportive_Drug_3_Dose_Is_Range": 0, "Supportive_Drug_3_Dose_Min": 0, "Supportive_Drug_3_Dose_Max": 0,

    "Cycle_Length_Days": 21
}

DRUG_bloss_b = {
    "Platinum_Drug": "cisplatin", "Platinum_Dose_Value": 50, "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based", "Platinum_Infusion_Duration_Hours": 1,
    "Platinum_Days_Number": 1, "Platinum_Day": 2,

    "Adjunct_Drug_1": "bleomycin", "Adjunct_Drug_1_Dose_Value": 30, "Adjunct_Drug_1_Dose_Unit": "units",
    "Adjunct_Drug_1_Dose_Method": "fixed", "Adjunct_Drug_1_Infusion_Duration_Hours": 24,
    "Adjunct_Drug_1_Days_Number": 1, "Adjunct_Drug_1_Days": 1, "Adjunct_Drug_1_Same_Day_As_Platinum": 0,

    "Adjunct_Drug_2": "ifosfamide", "Adjunct_Drug_2_Dose_Value": 5000, "Adjunct_Drug_2_Dose_Unit": "mg/m2",
    "Adjunct_Drug_2_Dose_Method": "BSA-based", "Adjunct_Drug_2_Infusion_Duration_Hours": 24,
    "Adjunct_Drug_2_Days_Number": 1, "Adjunct_Drug_2_Days": 2, "Adjunct_Drug_2_Same_Day_As_Platinum": 1,

    "Adjunct_Drug_3": 'No Drug', "Adjunct_Drug_3_Dose_Value": 0, "Adjunct_Drug_3_Dose_Unit": 'No Drug',
    "Adjunct_Drug_3_Dose_Method": 'No Drug', "Adjunct_Drug_3_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_3_Days_Number": 0, "Adjunct_Drug_3_Days": 0, "Adjunct_Drug_3_Same_Day_As_Platinum": 0,

    "Supportive_Drug_1": "mesna", "Supportive_Drug_1_Dose_Value": 6000, "Supportive_Drug_1_Dose_Unit": "mg/m2",
    "Supportive_Drug_1_Route": "IV", "Supportive_Drug_1_Start_Day": 2, "Supportive_Drug_1_Duration_Value": 36,
    "Supportive_Drug_1_Duration_Unit": "hours", "Supportive_Drug_1_Frequency_Times_Per_Day": 0,
    "Supportive_Drug_1_Frequency_Interval_Hours": 0, "Supportive_Drug_1_Frequency_Description": "continuous infusion",
    "Supportive_Drug_1_Dose_Is_Range": 0, "Supportive_Drug_1_Dose_Min": 6000, "Supportive_Drug_1_Dose_Max": 6000,

    "Supportive_Drug_2": 'No Drug', "Supportive_Drug_2_Dose_Value": 0, "Supportive_Drug_2_Dose_Unit": 'No Drug',
    "Supportive_Drug_2_Route": 'No Drug', "Supportive_Drug_2_Start_Day": 0, "Supportive_Drug_2_Duration_Value": 0,
    "Supportive_Drug_2_Duration_Unit": 'No Drug', "Supportive_Drug_2_Frequency_Times_Per_Day": 0,
    "Supportive_Drug_2_Frequency_Interval_Hours": 0, "Supportive_Drug_2_Frequency_Description": 'No Drug',
    "Supportive_Drug_2_Dose_Is_Range": 0, "Supportive_Drug_2_Dose_Min": 0, "Supportive_Drug_2_Dose_Max": 0,

    "Supportive_Drug_3": 'No Drug', "Supportive_Drug_3_Dose_Value": 0, "Supportive_Drug_3_Dose_Unit": 'No Drug',
    "Supportive_Drug_3_Route": 'No Drug', "Supportive_Drug_3_Start_Day": 0, "Supportive_Drug_3_Duration_Value": 0,
    "Supportive_Drug_3_Duration_Unit": 'No Drug', "Supportive_Drug_3_Frequency_Times_Per_Day": 0,
    "Supportive_Drug_3_Frequency_Interval_Hours": 0, "Supportive_Drug_3_Frequency_Description": 'No Drug',
    "Supportive_Drug_3_Dose_Is_Range": 0, "Supportive_Drug_3_Dose_Min": 0, "Supportive_Drug_3_Dose_Max": 0,

    "Cycle_Length_Days": 21
}

#=====================================================================================
# OVERALL SURVIVAL POINTS EXTRACTED FROM KAPLAN-MEIER CURVE USING WEBPLOTDIGITIZER
# (unchanged -- your own digitization)
#=====================================================================================
bloss_a_OS_raw = {
    0.627513476: 0.993081884, 0.817248652: 0.987906029, 1.006983829: 0.984528103, 1.196719006: 0.984092242,
    1.386454183: 0.984092242, 1.57618936: 0.982185348, 1.765924537: 0.980278454, 1.955659714: 0.97470306,
    2.16017945: 0.968424579, 2.320345509: 0.962431484, 2.473119288: 0.952601252, 2.662854464: 0.951293668,
    2.835340989: 0.946575467, 2.902179517: 0.931727575, 2.98482931: 0.919316874, 3.180314038: 0.916751648,
    3.352800562: 0.912974182, 3.406575773: 0.892362636, 3.585657371: 0.882059801, 3.605780798: 0.866327927,
    3.801265526: 0.864829653, 3.965127724: 0.86213276, 4.025498008: 0.849302089, 4.197984532: 0.846650598,
    4.266979142: 0.831159356, 4.456714319: 0.820371785, 4.646449496: 0.810074558, 4.836184673: 0.806696632,
    4.982798219: 0.804199509, 5.043168502: 0.792749066, 5.192246141: 0.778700317, 5.353644247: 0.7618816,
    5.526130771: 0.753085674, 5.715865948: 0.751396711, 5.819357863: 0.738824832, 5.974595735: 0.734437031,
    6.026341692: 0.719905721, 6.19020389: 0.710357631, 6.262648231: 0.695165136, 6.368850648: 0.684523112,
    6.41587376: 0.669954183, 6.585964638: 0.663128714, 6.666109892: 0.649459615, 6.837028357: 0.637201011,
    7.026763534: 0.62761206, 7.174455121: 0.6160413, 7.342546556: 0.607218773, 7.509725803: 0.582155343,
    7.613217717: 0.572530069, 7.802952894: 0.567953524, 7.936961656: 0.553465322, 8.130677291: 0.54490735,
    8.311788142: 0.537426878, 8.389407078: 0.521824854, 8.491461605: 0.508440275, 8.682634169: 0.501920515,
    8.872369346: 0.491241909, 9.053480197: 0.485946192, 9.131099133: 0.469775733, 9.325146473: 0.463991488,
    9.458823529: 0.450440435, 9.596812749: 0.435980904, 9.717553316: 0.422539248, 9.872791188: 0.410553058,
    10.04730697: 0.401422402, 10.25226154: 0.400364797, 10.45721612: 0.395499814, 10.62260026: 0.385628834,
    10.81284275: 0.37968862, 10.97670494: 0.366137566, 11.12331849: 0.353094261, 11.18584486: 0.333504332,
    11.34927584: 0.322294747, 11.54591048: 0.319076637, 11.73564565: 0.315262849, 11.92842471: 0.309763303,
    12.11511601: 0.308071135, 12.30485118: 0.307635274, 12.49458636: 0.302404937, 12.68432154: 0.294886327,
    12.87405671: 0.28927461, 13.06379189: 0.287258751, 13.25352707: 0.281047726, 13.44326225: 0.280502899,
    13.6403897: 0.276521771, 13.82416999: 0.270264695, 13.94203578: 0.25573144, 14.13320834: 0.252716732,
    14.32294352: 0.252008457, 14.51931279: 0.248416791, 14.66791657: 0.240621576, 14.85765175: 0.240948472,
    15.05108306: 0.236068381, 15.19055074: 0.228602697, 15.37511132: 0.229289179, 15.5648465: 0.230160902,
    15.75458167: 0.230596763, 15.96041559: 0.227603848, 16.13405203: 0.222097465, 16.3237872: 0.220844363,
    16.51352238: 0.215940922, 16.70325756: 0.215559543, 16.88149363: 0.207027555, 17.08272791: 0.203791284,
    17.27390048: 0.201543873, 17.39176627: 0.190356763, 17.58293883: 0.188808547, 17.77267401: 0.188808547,
    17.96240919: 0.188808547, 18.15214436: 0.188808547, 18.34187954: 0.188808547, 18.53161472: 0.188808547,
    18.72134989: 0.188808547, 18.91108507: 0.188808547, 19.10082025: 0.188754064, 19.23880947: 0.18521269,
    19.29055543: 0.170120987, 19.4802906: 0.168432024, 19.67002578: 0.168105128, 19.8317319: 0.157644453,
    20.03224748: 0.154048596, 20.22198266: 0.154048596, 20.41171783: 0.154048596, 20.60145301: 0.154048596,
    20.77596879: 0.146680615, 20.98092337: 0.146257573, 21.17065854: 0.146257573, 21.36039372: 0.144895506,
    21.46388563: 0.131241541, 21.65362081: 0.128877598, 21.84335599: 0.128877598, 22.03309116: 0.128877598,
    22.22282634: 0.128877598, 22.41256152: 0.128877598, 22.6022967: 0.128877598, 22.79203187: 0.128877598,
    22.98176705: 0.128877598, 23.17150223: 0.128877598, 23.3612374: 0.128877598, 23.55097258: 0.127951392,
    23.74070776: 0.12588105, 23.90456996: 0.12588105
}

bloss_b_OS_raw = {
    0.282540427: 0.99727705, 0.481563339: 0.993676071, 0.618889149: 0.987388444, 0.843121631: 0.98991753,
    0.920740567: 0.9722695, 1.110475744: 0.970308123, 1.300210921: 0.970253641, 1.489946098: 0.968237781,
    1.662432622: 0.96191779, 1.765924537: 0.950803323, 1.947819418: 0.945028159, 2.059151629: 0.932497143,
    2.248886806: 0.926504048, 2.438621983: 0.919693713, 2.542113897: 0.908729073, 2.731849074: 0.907544075,
    2.921584251: 0.905855112, 3.07250996: 0.90033874, 3.135467542: 0.883767833, 3.33555191: 0.880357217,
    3.482165456: 0.878613771, 3.663276307: 0.868125855, 3.777744636: 0.851917257, 3.901882665: 0.836662107,
    4.049646121: 0.824675917, 4.232481837: 0.814433173, 4.422217014: 0.803918015, 4.611952191: 0.793675271,
    4.801687368: 0.784631146, 4.991422545: 0.778256673, 5.181157722: 0.778129546, 5.336395594: 0.773905841,
    5.401078041: 0.749312748, 5.595125381: 0.744967755, 5.795209749: 0.741971207, 5.964246543: 0.728786398,
    6.138457933: 0.719347274, 6.18445434: 0.692528174, 6.265358733: 0.682297105, 6.483430982: 0.660009641,
    6.621420202: 0.644783185, 6.733536442: 0.632554417, 6.828404031: 0.618170989, 6.954894149: 0.606527262,
    7.130255449: 0.600867117, 7.216498711: 0.584902178, 7.388985236: 0.582554882, 7.664963675: 0.576986752,
    7.788579017: 0.561169522, 7.915069135: 0.547914794, 8.059766164: 0.534077404, 8.16229982: 0.523123358,
    8.311788142: 0.521425314, 8.536020623: 0.5167507, 8.674793873: 0.496504936, 8.941363956: 0.495818454,
    9.054712243: 0.474679174, 9.053480197: 0.495055697, 9.269088352: 0.472554349, 9.458823529: 0.470810904,
    9.648558706: 0.467269529, 9.821045231: 0.462226855, 9.890039841: 0.451850931, 10.08667448: 0.446551582,
    10.26951019: 0.441226808, 10.37503136: 0.421622657, 10.38162644: 0.439319914, 10.57998594: 0.42014201,
    10.76972112: 0.418997874, 10.95945629: 0.416110292, 11.14919147: 0.415946844, 11.31017889: 0.411352138,
    11.35401922: 0.396993681, 11.44098117: 0.384682865, 11.60340598: 0.371158448, 11.80464026: 0.365114502,
    11.94465874: 0.342866339, 11.93400516: 0.362608299, 12.14961331: 0.339235229, 12.33934849: 0.338962816,
    12.50105461: 0.328672399, 12.70157019: 0.324851801, 12.89130537: 0.324742836, 13.04136864: 0.31592209,
    13.14716038: 0.305074588, 13.32252168: 0.302078041, 13.39151629: 0.287422199, 13.58125146: 0.285297375,
    13.76236232: 0.282121034, 13.78823529: 0.266239333, 14.02971643: 0.265520162, 14.21945161: 0.265520162,
    14.40918678: 0.265520162, 14.61414136: 0.26016163, 14.78865714: 0.2553319, 14.97839231: 0.251518113,
    15.14225451: 0.240049508, 15.30611671: 0.224385737, 15.49585189: 0.223459532, 15.68558706: 0.223405049,
    15.83220061: 0.218774021, 15.87532224: 0.205098868, 16.06505742: 0.203791284, 16.25479259: 0.19791056,
    16.44452777: 0.195400951, 16.63426295: 0.195400951, 16.82399813: 0.193766471, 16.88609327: 0.175823508,
    16.92749004: 0.19719888, 17.08272791: 0.173989257, 17.27246309: 0.174207188, 17.46219827: 0.174425119,
    17.65193344: 0.169739608, 17.84166862: 0.156227903, 18.0314038: 0.154048596, 18.22113897: 0.154048596,
    18.41087415: 0.154048596, 18.60060933: 0.154048596, 18.7903445: 0.154048596, 18.98007968: 0.154048596,
    19.16981486: 0.154048596, 19.35955004: 0.154048596, 19.54928521: 0.154048596, 19.69589876: 0.154048596,
    20.24126056: 0.138360789, 20.44621514: 0.136668621, 20.63595032: 0.136668621, 20.82568549: 0.136668621,
    21.01542067: 0.136668621, 21.20515585: 0.136668621, 21.34027029: 0.132273685, 21.46613546: 0.129568106,
    21.56175299: 0.129568106, 21.70517928: 0.129568106, 21.84860558: 0.129568106, 22.03984064: 0.129568106,
    22.13545817: 0.129568106, 22.42231076: 0.129568106, 22.51792829: 0.129568106, 22.70916335: 0.129568106,
    22.90039841: 0.129568106, 23.13944223: 0.129568106, 23.47410359: 0.129568106, 23.33067729: 0.129568106
}

def dict_to_sorted_lists(data):
    sorted_items = sorted(data.items())
    return [x for x, y in sorted_items], [y for x, y in sorted_items]

def simplify_km_curve(t, s, decimals=2):
    """
    Collapse digitization noise before Guyot reconstruction -- see the
    Miller 2014 pipeline for the full rationale. WebPlotDigitizer output
    often has many closely-spaced points differing only by sub-1% jitter,
    not real distinct events; guyot_reconstruct_ipd's "at least one event
    per detected drop" floor then front-loads events into wherever points
    are densely clustered and drags the reconstructed median down.
    """
    s_round = np.round(np.array(s), decimals)
    keep_t, keep_s = [t[0]], [s_round[0]]
    for i in range(1, len(t)):
        if s_round[i] != keep_s[-1]:
            keep_t.append(t[i])
            keep_s.append(s_round[i])
        else:
            keep_t[-1] = t[i]
    return keep_t, keep_s

bloss_a_os_time, bloss_a_os_surv = dict_to_sorted_lists(bloss_a_OS_raw)
bloss_a_os_time, bloss_a_os_surv = simplify_km_curve(bloss_a_os_time, bloss_a_os_surv, decimals=2)
bloss_b_os_time, bloss_b_os_surv = dict_to_sorted_lists(bloss_b_OS_raw)
bloss_b_os_time, bloss_b_os_surv = simplify_km_curve(bloss_b_os_time, bloss_b_os_surv, decimals=2)

#=============================================================================================
# PROGRESSION-FREE SURVIVAL POINTS EXTRACTED FROM KAPLAN-MEIER CURVE USING WEBPLOTDIGITIZER
# (unchanged -- your own digitization)
#=============================================================================================
bloss_a_pfs_raw = {
    0.362462971: 0.995150074, 0.497460855: 0.994027987, 0.632458739: 0.992867207, 0.767456623: 0.990623032,
    0.890181972: 0.984875021, 0.923317816: 0.972650288, 0.98099873: 0.96588294, 1.093701509: 0.961229502,
    1.23381295: 0.959398049, 1.368810834: 0.956883025, 1.405628438: 0.94586667, 1.417900973: 0.935469397,
    1.454718578: 0.92519373, 1.57130766: 0.916894153, 1.577443927: 0.904817204, 1.639829313: 0.897634877,
    1.698805661: 0.886131332, 1.71594825: 0.874970647, 1.747895801: 0.863337051, 1.770736352: 0.851242368,
    1.795281422: 0.839803848, 1.819826492: 0.828684543, 1.851902435: 0.816467333, 1.935392862: 0.805807503,
    1.964028777: 0.79452859, 1.994710114: 0.787447832, 2.129707998: 0.786480515, 2.24629708: 0.778567864,
    2.307659755: 0.765125386, 2.362886162: 0.754307561, 2.405479077: 0.740224568, 2.436521371: 0.754520371,
    2.558428551: 0.736587609, 2.673475699: 0.733272143, 2.774016081: 0.71951318, 2.761743546: 0.731111307,
    2.845196784: 0.711915874, 2.941448522: 0.708067059, 2.953721057: 0.697608984, 2.990538662: 0.686178064,
    3.122906717: 0.679976182, 3.188922165: 0.669723901, 3.301735083: 0.667552151, 3.343666243: 0.655421999,
    3.418324164: 0.647021084, 3.526731556: 0.642422875, 3.565594583: 0.635269568, 3.682183665: 0.634868131,
    3.737410072: 0.623545689, 3.860135421: 0.62111235, 3.893884892: 0.60975659, 3.946043165: 0.601374791,
    4.033178163: 0.586698662, 4.154676259: 0.581475704, 4.171857808: 0.56744648, 4.301946678: 0.561405264,
    4.376809141: 0.554309029, 4.510579771: 0.554131043, 4.625123431: 0.546676255, 4.703872196: 0.533517524,
    4.696421014: 0.521037756, 4.70694033: 0.51166197, 4.708693549: 0.502432111, 4.725349132: 0.492338852,
    4.733019467: 0.482762416, 4.713076598: 0.544690032, 4.785484554: 0.473952096, 4.870165044: 0.466546319,
    4.976936098: 0.453692615, 5.111933982: 0.443353934, 5.218295951: 0.43445462, 5.28272676: 0.425055526,
    5.423860911: 0.423388517, 5.542290873: 0.417429846, 5.688743123: 0.416152988, 5.823741007: 0.415417827,
    5.919089163: 0.405813712, 6.020101566: 0.396233749, 6.143593948: 0.389477295, 6.273052147: 0.381967478,
    6.310040203: 0.373537851, 6.314642404: 0.391467066, 6.449640288: 0.374480985, 6.534320779: 0.366355524,
    6.572365637: 0.354283412, 6.667916087: 0.346777034, 6.794563111: 0.337122195, 6.917749834: 0.332184371,
    7.03381295: 0.325666314, 7.173719848: 0.322013727, 7.308717732: 0.320388634, 7.443715616: 0.319614781,
    7.518794653: 0.305817431, 7.523487093: 0.318260538, 7.664621244: 0.305066338, 7.808115499: 0.303331121,
    7.889033311: 0.298499639, 7.983707152: 0.288467183, 8.118705036: 0.286339087, 8.247205696: 0.281031363,
    8.383122379: 0.276627227, 8.458559849: 0.265320039, 8.597333898: 0.26072454, 8.625560728: 0.253311025,
    8.746357536: 0.249066992, 8.891874736: 0.248033345, 9.02687262: 0.247066028, 9.093288691: 0.234764036,
    9.235505713: 0.233794443, 9.370503597: 0.234413526, 9.505501481: 0.233097975, 9.640499365: 0.226907148,
    9.775497249: 0.226017217, 9.914585978: 0.224588813, 10.02503879: 0.215345779, 10.17873768: 0.210001216,
    10.29269693: 0.205501812, 10.43821413: 0.199938359, 10.57321202: 0.198042418, 10.7082099: 0.197732876,
    10.84320779: 0.197229872, 10.97820567: 0.196726867, 11.08712442: 0.190095911, 11.13774862: 0.178657391,
    11.15718014: 0.167023796, 11.29729158: 0.16631443, 11.43228946: 0.16631443, 11.57378457: 0.162108309,
    11.70228523: 0.158227662, 11.83728311: 0.158227662, 11.96578377: 0.152243955, 12.10727888: 0.150140895,
    12.24227677: 0.150140895, 12.37727465: 0.150140895, 12.51227253: 0.150140895, 12.64727042: 0.149792661,
    12.75567781: 0.144111287, 12.8552217: 0.129711166, 13.01467943: 0.122422435, 13.13305826: 0.117864761,
    13.19953449: 0.111177378, 13.33453237: 0.111835153, 13.46953026: 0.111835153, 13.61219848: 0.108430198,
    13.73952603: 0.105876482, 13.87452391: 0.105876482, 14.00952179: 0.105876482, 14.14451968: 0.105876482,
    14.27951756: 0.105876482, 14.41451545: 0.105876482, 14.54951333: 0.105876482, 14.68451121: 0.105683019,
    14.79905487: 0.097477594, 14.94223445: 0.094810379, 15.07723233: 0.094810379, 15.21223022: 0.094810379,
    15.3472281: 0.094810379, 15.48222598: 0.094810379, 15.61722387: 0.094810379, 15.72154041: 0.092634991,
    15.74139304: 0.079863629, 15.88721964: 0.078636844, 16.02221752: 0.078636844, 16.1572154: 0.078636844,
    16.29221329: 0.078636844, 16.42721117: 0.078636844, 16.56220906: 0.078636844, 16.70772626: 0.076447944,
    16.82168551: 0.072556568, 16.96720271: 0.071826934, 17.10220059: 0.071826934, 17.23719848: 0.071826934,
    17.37219636: 0.071826934, 17.50719424: 0.071826934, 17.64219213: 0.071826934, 17.77719001: 0.071826934,
    17.9121879: 0.071826934, 18.04718578: 0.071826934, 18.18218366: 0.071826934, 18.31718155: 0.071826934,
    18.45217943: 0.071826934, 18.58717732: 0.071826934, 18.7221752: 0.071826934, 18.85717309: 0.071826934,
    18.99217097: 0.071826934, 19.12716885: 0.071826934, 19.25566951: 0.066619356, 19.39716462: 0.065017025,
    19.53216251: 0.065017025, 19.66716039: 0.065017025, 19.80215827: 0.065017025, 19.93715616: 0.065017025,
    20.07215404: 0.065017025, 20.20715193: 0.065017025, 20.34214981: 0.065017025, 20.47714769: 0.065017025,
    20.61214558: 0.065017025, 20.74714346: 0.065017025, 20.88214135: 0.065017025, 21.01713923: 0.065017025,
    21.15213711: 0.065017025, 21.287135: 0.065017025, 21.42213288: 0.065017025, 21.55713077: 0.065017025,
    21.69212865: 0.065017025, 21.82712653: 0.065017025, 21.96212442: 0.065017025, 22.0971223: 0.065017025,
    22.23212019: 0.065017025, 22.37862357: 0.061957886, 22.4970024: 0.059235695, 22.63711384: 0.059058354,
    22.77211172: 0.059058354, 22.90710961: 0.059058354, 23.04210749: 0.059058354, 23.17710537: 0.059058354,
    23.31210326: 0.059058354, 23.44710114: 0.059058354, 23.58209903: 0.059058354, 23.71709691: 0.059058354,
    23.85209479: 0.059058354, 23.96254761: 0.059058354
}

bloss_b_pfs_raw = {
    0.374735506: 0.989462252, 0.50973339: 0.989036633, 0.644731274: 0.985128673, 0.779729158: 0.984045278,
    0.865636902: 0.976693671, 0.945408379: 0.963286662, 1.043588658: 0.952007749, 1.120730307: 0.942005694,
    1.214527538: 0.937354283, 1.325856961: 0.936344957, 1.47312738: 0.932748474, 1.505172333: 0.916886271,
    1.534490055: 0.90561524, 1.609001874: 0.892177829, 1.646170123: 0.883057415, 1.68053322: 0.869352472,
    1.700169276: 0.859080858, 1.714714346: 0.851277836, 1.798349556: 0.839644241, 1.871984765: 0.826450041,
    1.890393567: 0.794954209, 1.90880237: 0.817795781, 2.068345324: 0.79452859, 2.189843419: 0.791591816,
    2.227888278: 0.776747159, 2.375158697: 0.776149573, 2.510156581: 0.775801338, 2.633648963: 0.769603257,
    2.780152349: 0.76814019, 2.915150233: 0.767908034, 2.991853576: 0.753881942, 2.99492171: 0.743596141,
    3.019057695: 0.732615162, 3.007194245: 0.765373664, 3.160600931: 0.731420848, 3.275655946: 0.693816411,
    3.264917478: 0.719193965, 3.269300526: 0.708857495, 3.418324164: 0.68982623, 3.547185781: 0.6874002,
    3.56276246: 0.662586592, 3.571730851: 0.67663203, 3.602412188: 0.652062186, 3.737410072: 0.647849522,
    3.866271689: 0.643816778, 3.921498096: 0.631644065, 4.019678375: 0.622578373, 4.130131189: 0.613176054,
    4.265129073: 0.610544953, 4.327935575: 0.596540483, 4.338764283: 0.609384173, 4.449217097: 0.593636257,
    4.467625899: 0.583847012, 4.493398223: 0.571872921, 4.586446351: 0.563107742, 4.614896318: 0.540965862,
    4.797450275: 0.536603264, 4.917017251: 0.519052725, 4.897164621: 0.533623929, 5.062843843: 0.515747916,
    5.179432924: 0.513832629, 5.222386796: 0.5058039, 5.35738468: 0.505533052, 5.492382565: 0.50506874,
    5.592006671: 0.490936811, 5.584426576: 0.502979335, 5.737833263: 0.490636374, 5.872831147: 0.489669058,
    5.946466356: 0.482443202, 6.08146424: 0.481930524, 6.176812396: 0.46696539, 6.17964452: 0.479995891,
    6.255032948: 0.460174189, 6.265552264: 0.440466493, 6.293165468: 0.430304832, 6.339187474: 0.421100813,
    6.388277613: 0.411258366, 6.437367753: 0.401043501, 6.511002962: 0.394330323, 6.646000846: 0.393788626,
    6.768726196: 0.392081849, 6.817816335: 0.382335596, 6.952814219: 0.381677821, 7.087812103: 0.381677821,
    7.210537452: 0.38025909, 7.259627592: 0.372314195, 7.394625476: 0.371153415, 7.455988151: 0.359855156,
    7.566440965: 0.356870293, 7.572577232: 0.337839028, 7.592429862: 0.323793589, 7.738256454: 0.32104641,
    7.835492692: 0.293771055, 7.833089678: 0.313191798, 8.020524757: 0.293419845, 8.155522641: 0.293148996,
    8.290520525: 0.288497585, 8.425518409: 0.285487848, 8.535971223: 0.283846173, 8.585061363: 0.274305667,
    8.720059247: 0.272719267, 8.855057131: 0.272719267, 8.990055015: 0.271945414, 9.04878786: 0.258977843,
    9.065443444: 0.2382441, 9.210960643: 0.238282793, 9.345958527: 0.238669719, 9.480956411: 0.238785797,
    9.615954295: 0.237818481, 9.742770489: 0.23561438, 9.812314854: 0.223339684, 9.916631401: 0.220580897,
    10.05776555: 0.21866561, 10.1170828: 0.207982564, 10.27621667: 0.208620993, 10.38912399: 0.20779297,
    10.52412188: 0.201795607, 10.64829106: 0.192147021, 10.79411765: 0.189297875, 10.92911553: 0.189259182,
    11.04502281: 0.183764823, 11.09111299: 0.175507808, 11.20115672: 0.16367559, 11.27609356: 0.157105575,
    11.40774439: 0.153661927, 11.50592467: 0.147906393, 11.56728735: 0.141048118, 11.70228523: 0.140777269,
    11.83830583: 0.138755577, 11.92319086: 0.129401625, 12.05818874: 0.128859927, 12.19318663: 0.128859927,
    12.32818451: 0.128859927, 12.4631824: 0.128859927, 12.59818028: 0.128859927, 12.73317816: 0.128859927,
    12.94181126: 0.130136785, 13.08499083: 0.126589957, 13.22933922: 0.118523457, 13.34680491: 0.117793824,
    13.48180279: 0.117793824, 13.61680068: 0.117793824, 13.75179856: 0.117793824, 13.8846307: 0.112836611,
    14.02179433: 0.110132676, 14.15679221: 0.110132676, 14.2917901: 0.110132676, 14.42678798: 0.110132676,
    14.56178587: 0.110132676, 14.69678375: 0.110132676, 14.83178163: 0.109823134, 14.96677952: 0.109707056,
    15.1017774: 0.109707056, 15.23677529: 0.109707056, 15.37177317: 0.109707056, 15.50677105: 0.109707056,
    15.64176894: 0.109707056, 15.77676682: 0.109242744, 15.86144731: 0.101790537, 15.89949217: 0.087923084,
    16.03449006: 0.087149231, 16.16948794: 0.087149231, 16.30448582: 0.087149231, 16.43948371: 0.087149231,
    16.57448159: 0.087149231, 16.70947948: 0.087149231, 16.84447736: 0.087149231, 16.97947524: 0.087149231,
    17.11447313: 0.087149231, 17.24947101: 0.087149231, 17.3844689: 0.087149231, 17.51946678: 0.087149231,
    17.65446466: 0.082815652, 17.64920501: 0.078211225, 17.80173508: 0.078636844, 17.93673297: 0.078636844,
    18.07173085: 0.078636844, 18.20672873: 0.078636844, 18.34172662: 0.078636844, 18.4767245: 0.078636844,
    18.61172239: 0.078636844, 18.74672027: 0.078636844, 18.88171815: 0.078636844, 19.01671604: 0.078636844,
    19.15171392: 0.078636844, 19.28671181: 0.078636844, 19.42170969: 0.078636844, 19.55670758: 0.078636844,
    19.69170546: 0.078636844, 19.82670334: 0.078636844, 19.96170123: 0.078636844, 20.09669911: 0.078636844,
    20.231697: 0.078636844, 20.36669488: 0.078636844, 20.50169276: 0.078636844, 20.63669065: 0.078636844,
    20.77168853: 0.078636844, 20.90668642: 0.078636844, 21.0416843: 0.078636844, 21.17668218: 0.078636844,
    21.31168007: 0.078636844, 21.44667795: 0.078636844, 21.58167584: 0.078636844, 21.71667372: 0.078636844,
    21.8516716: 0.078636844, 21.98666949: 0.078636844, 22.12166737: 0.078636844, 22.25666526: 0.078636844,
    22.39166314: 0.078636844, 22.52666102: 0.078636844, 22.66165891: 0.078636844, 22.79665679: 0.078636844,
    22.93165468: 0.078636844, 23.06665256: 0.078636844, 23.20165044: 0.078636844, 23.33664833: 0.078636844,
    23.47164621: 0.078636844, 23.6066441: 0.078636844, 23.71709691: 0.078697647
}

bloss_a_pfs_time, bloss_a_pfs_surv = dict_to_sorted_lists(bloss_a_pfs_raw)
bloss_a_pfs_time, bloss_a_pfs_surv = simplify_km_curve(bloss_a_pfs_time, bloss_a_pfs_surv, decimals=2)
bloss_b_pfs_time, bloss_b_pfs_surv = dict_to_sorted_lists(bloss_b_pfs_raw)
bloss_b_pfs_time, bloss_b_pfs_surv = simplify_km_curve(bloss_b_pfs_time, bloss_b_pfs_surv, decimals=2)

#==================================================
# ASSIGNING PARAMETERS
#==================================================
def parameter_assign(data, ratio):
    if len(data) != len(ratio):
        raise ValueError(f"Length mismatch: {len(data)} keys vs {len(ratio)} values")
    return dict(zip(data, ratio))

#==========================================================================
# HISTOLOGY (eligibility required squamous cell carcinoma -- "16 [ineligible]
# ... 11 for wrong histologic type" -- verified, unchanged: all squamous)
#==========================================================================
HISTOLOGY_bloss_a = parameter_assign(histology_base, [146, 0, 0, 0, 0, 0, 0, 0, 0])
HISTOLOGY_bloss_b = parameter_assign(histology_base, [141, 0, 0, 0, 0, 0, 0, 0, 0])

#===========================================================
# RACE/ETHNICITY (Table 1, verified exact match, unchanged)
#===========================================================
ETHNICITY_bloss_a = parameter_assign(ethnicity_base, [103, 25, 4, 0, 13, 1, 0])
ETHNICITY_bloss_b = parameter_assign(ethnicity_base, [103, 25, 2, 0, 11, 0, 0])

#================================================================
# DISEASE NATURE (FIX C1: "32 entered with stage IVB disease and had no
# prior radiation or chemotherapy" (14 CI/18 CIB, matching the never-
# treated counts exactly) -> Advanced (IVB), trial-cited. The remaining
# prior-treated patients' Persistent-vs-Recurrent split is NOT reported and
# is self-proposed here (flagged in the weight reference table).
#================================================================
NATURE_bloss_a = parameter_assign(disease_nature, [18, 114, 14])   # Persistent(self-proposed), Recurrent(self-proposed), Advanced(IVB, CITED)
NATURE_bloss_b = parameter_assign(disease_nature, [17, 106, 18])

#======================================================================
# FIGO STAGE (only the Advanced/IVB subgroup is confirmed IVB -- same
# convention as every other trial in this series)
#======================================================================
STAGE_bloss_a = parameter_assign(figo_stage, [0, 0, 0, 0, 14, 132])
STAGE_bloss_b = parameter_assign(figo_stage, [0, 0, 0, 0, 18, 123])

#======================================================================
# TUMOR GRADE (Table 1, verified exact match, unchanged)
#======================================================================
GRADE_bloss_a = parameter_assign(grade_base, [5, 96, 45, 0])
GRADE_bloss_b = parameter_assign(grade_base, [7, 89, 45, 0])

#===================================================================
# PERFORMANCE STATUS (FIX C4: shift-by-one code. code1=PS0, code2=PS1,
# code3=PS2, code4 unused -- counts unchanged, verified correct: 76/56/14
# CI, 76/49/16 CIB)
#===================================================================
PS_bloss_a = parameter_assign(performance_status, [76, 56, 14, 0])
PS_bloss_b = parameter_assign(performance_status, [76, 49, 16, 0])

#================================================================
# SITE OF DISEASE (Table 1, exact counts -- NEW, see E1 note above.
# Used ONLY as an internal response-propensity covariate, not an output
# column, to avoid changing the merged dataset's column count.)
#================================================================
SITE_bloss_a = {"Pelvis only": 54, "Extra-pelvic": 92}
SITE_bloss_b = {"Pelvis only": 56, "Extra-pelvic": 85}

#====================================================================
# PRIOR TREATMENT (FIX C2: the 30/33 "prior cisplatin as radiosensitizer"
# patients recoded from "prior_cisplatin" to "chemoradiotherapy" so
# prior_radiation is correctly triggered too. none=146-132=14/141-123=18,
# radiation_only=132-30=102/123-33=90, chemoradiotherapy=30/33.)
#====================================================================
_NONE_IDX = prior_treatment_base.index("none")
_RAD_IDX = prior_treatment_base.index("radiation_only")
_CRT_IDX = prior_treatment_base.index("chemoradiotherapy")

def _treatment_counts(none_n, rad_n, crt_n):
    counts = [0] * len(prior_treatment_base)
    counts[_NONE_IDX] = none_n
    counts[_RAD_IDX] = rad_n
    counts[_CRT_IDX] = crt_n
    return counts

TREATMENT_bloss_a = parameter_assign(prior_treatment_base, _treatment_counts(14, 102, 30))
TREATMENT_bloss_b = parameter_assign(prior_treatment_base, _treatment_counts(18, 90, 33))

#================================================================
# CAUSE OF DEATH ("There were no treatment-related deaths recorded in
# either regimen" -- cited. Disease/Other split filled in at generation
# time once the exact dead-patient count is known from ALIVE_COUNT.)
#================================================================
CAUSE_bloss_a = {"Treatment": 0, "Other/Unknown": 0}
CAUSE_bloss_b = {"Treatment": 0, "Other/Unknown": 0}

#================================================================
# RESPONSE (FIX B2: Table 3 gives EXACT Responders/Nonresponders counts
# (47/99 CI, 44/97 CIB) -- but NO CR/PR split and NO SD/PD split are
# reported anywhere in this paper (checked Tables 3-4 directly). Both
# splits below are self-proposed, flagged in the weight reference table.
# No "NE" value is needed: the paper's own ITT convention already resolves
# every patient (including the 6 never-treated) into responder/nonresponder.
#================================================================
CR_COUNT_a, PR_COUNT_a, SD_COUNT_a, PD_COUNT_a = 5, 42, 54, 45
CR_COUNT_b, PR_COUNT_b, SD_COUNT_b, PD_COUNT_b = 4, 40, 53, 44
assert CR_COUNT_a + PR_COUNT_a + SD_COUNT_a + PD_COUNT_a == BLOSS_A_NUMBER
assert CR_COUNT_b + PR_COUNT_b + SD_COUNT_b + PD_COUNT_b == BLOSS_B_NUMBER

#===================================================================
# TOXICITY (Table 2, Grade>=3 counts -- independently re-derived from the
# page image + narrative cross-checks (pulmonary 18%/5%, central
# neurotoxicity 10%/7%, both confirmed exact) since the text extraction of
# this table is badly garbled. Matches your original values exactly for
# every category checked -- unchanged.)
#===================================================================
tox_bloss_a = [
    121, 117, 23, 32, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 21, 0, 3, 0, 1, 3, 0, 1, 0, 0, 2, 0, 0, 0, 1, 14, 0, 0, 1, 0, 4, 0
]
tox_bloss_b = [
    118, 117, 28, 29, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 1, 23, 0, 8, 0, 3, 6, 0, 1, 0, 0, 6, 0, 0, 0, 5, 10, 0, 0, 7, 0, 2, 0
]

TOXICITY_bloss_a = parameter_assign(toxicity_base, tox_bloss_a)
TOXICITY_bloss_b = parameter_assign(toxicity_base, tox_bloss_b)

_PFS_HISTOLOGY_WEIGHT = {
    "Squamous Cell Carcinoma": 1.00, "Adenocarcinoma": 0.75, "Adenosquamous": 0.85,
    "Mucinous Adenocarcinoma": 0.65, "Clear Cell": 0.70, "Endometrioid": 0.80,
    "Villoglandular": 1.15, "Undifferentiated Carcinoma": 0.50, "Not Specified": 0.90,
}

#======================================================================
# CREATING FINAL TRIAL DICTIONARIES
#======================================================================
def create_trial_dict(
    trial_id, drug_combination, ethnicity, histology, disease_nature_dist, disease_stage,
    grades, causes, performance_status_dist, prior_treatment, site_of_disease, cycle_completion_rate,
    toxicity, patient_number, followup,
    cr_count, pr_count, sd_count, pd_count,
    target_os, target_pfs, alive_count, pfs_alive_count,
    target_age, age_type, age_min, age_max, age_sd,
    os_orr_weight, os_age_weight,
    pfs_age_weight, pfs_histology_weight, pfs_response_weight, pfs_toxicity_weight,
    orr_vs_ps, orr_vs_pelvic_only, tolerability_orr_weight, pfsr_vs_ps,
    os_time_grid, os_surv_grid, pfs_time_grid, pfs_surv_grid,
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
        "ethnicity": ethnicity, "histology": histology, "disease_nature": disease_nature_dist,
        "disease_stage": disease_stage, "grades": grades, "causes": causes,
        "performance_status": performance_status_dist, "prior_treatment": prior_treatment,
        "site_of_disease": site_of_disease,
        "cycle_completion_rate": cycle_completion_rate, "toxicity": toxicity,
        "patient_number": patient_number, "followup": followup,
        "cr_count": cr_count, "pr_count": pr_count, "sd_count": sd_count, "pd_count": pd_count,
        "target_os": target_os, "target_pfs": target_pfs,
        "alive_count": alive_count, "pfs_alive_count": pfs_alive_count,
        "target_age": target_age, "age_type": age_type, "age_min": age_min, "age_max": age_max, "age_sd": age_sd,
        "os_orr_weight": os_orr_weight, "os_age_weight": os_age_weight,
        "pfs_age_weight": pfs_age_weight, "pfs_histology_weight": pfs_histology_weight,
        "pfs_response_weight": pfs_response_weight, "pfs_toxicity_weight": pfs_toxicity_weight,
        "orr_vs_ps": orr_vs_ps, "orr_vs_pelvic_only": orr_vs_pelvic_only,
        "tolerability_orr_weight": tolerability_orr_weight, "pfsr_vs_ps": pfsr_vs_ps,
        "os_time_grid": os_time_grid, "os_surv_grid": os_surv_grid,
        "pfs_time_grid": pfs_time_grid, "pfs_surv_grid": pfs_surv_grid,
        "never_treated_count": never_treated_count,
    })
    return trial

#=====================================================================
# CI ARM (Cisplatin + Ifosfamide) TRIAL CONSTANTS
#=====================================================================
trial_bloss_a = create_trial_dict(
    trial_id="bloss_CI_arm", drug_combination=DRUG_bloss_a, ethnicity=ETHNICITY_bloss_a,
    histology=HISTOLOGY_bloss_a, disease_nature_dist=NATURE_bloss_a, disease_stage=STAGE_bloss_a,
    grades=GRADE_bloss_a, causes=CAUSE_bloss_a, performance_status_dist=PS_bloss_a,
    prior_treatment=TREATMENT_bloss_a, site_of_disease=SITE_bloss_a,
    cycle_completion_rate=0.48,   # self-proposed estimate (median 4 of max 6 cycles received)
    toxicity=TOXICITY_bloss_a, patient_number=BLOSS_A_NUMBER, followup=36,
    cr_count=CR_COUNT_a, pr_count=PR_COUNT_a, sd_count=SD_COUNT_a, pd_count=PD_COUNT_a,
    target_os=8.5, target_pfs=4.6,   # both directly cited (Results)
    alive_count=12, pfs_alive_count=9,   # self-proposed, anchored to the digitized curves' own tail (~8%/~6%)
    target_age=45, age_type="median", age_min=25, age_max=77, age_sd=(77 - 25) / 4,
    os_orr_weight=0.3, os_age_weight=0.2,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_pelvic_only=ORR_VS_PELVIC_ONLY, tolerability_orr_weight=0.20, pfsr_vs_ps=-0.35,
    os_time_grid=bloss_a_os_time, os_surv_grid=bloss_a_os_surv,
    pfs_time_grid=bloss_a_pfs_time, pfs_surv_grid=bloss_a_pfs_surv,
    never_treated_count=2   # FIX C3: was 14 (that was the stage-IVB-treatment-naive count, a different concept)
)

#=====================================================================
# CIB ARM (Cisplatin + Ifosfamide + Bleomycin) TRIAL CONSTANTS
#=====================================================================
trial_bloss_b = create_trial_dict(
    trial_id="bloss_CIB_arm", drug_combination=DRUG_bloss_b, ethnicity=ETHNICITY_bloss_b,
    histology=HISTOLOGY_bloss_b, disease_nature_dist=NATURE_bloss_b, disease_stage=STAGE_bloss_b,
    grades=GRADE_bloss_b, causes=CAUSE_bloss_b, performance_status_dist=PS_bloss_b,
    prior_treatment=TREATMENT_bloss_b, site_of_disease=SITE_bloss_b,
    cycle_completion_rate=0.52,
    toxicity=TOXICITY_bloss_b, patient_number=BLOSS_B_NUMBER, followup=36,
    cr_count=CR_COUNT_b, pr_count=PR_COUNT_b, sd_count=SD_COUNT_b, pd_count=PD_COUNT_b,
    target_os=8.4, target_pfs=5.1,
    alive_count=14, pfs_alive_count=10,
    target_age=46, age_type="median", age_min=21, age_max=80, age_sd=(80 - 21) / 4,
    os_orr_weight=0.3, os_age_weight=0.2,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_pelvic_only=ORR_VS_PELVIC_ONLY, tolerability_orr_weight=0.20, pfsr_vs_ps=-0.35,
    os_time_grid=bloss_b_os_time, os_surv_grid=bloss_b_os_surv,
    pfs_time_grid=bloss_b_pfs_time, pfs_surv_grid=bloss_b_pfs_surv,
    never_treated_count=4   # FIX C3: was 18
)

TRIALS = {
    "bloss_a": trial_bloss_a,
    "bloss_b": trial_bloss_b,
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper Functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def expand_distribution(dist, n):
    np.random.seed(SEED)
    if not dist:
        return np.zeros(n, dtype=int)
    arr = np.concatenate([[k] * v for k, v in dist.items()])
    if len(arr) < n:
        arr = np.pad(arr, (0, n - len(arr)), constant_values=arr[0] if len(arr) > 0 else 0)
    return np.random.permutation(arr[:n])

def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

def derive_prior_treatments(treatment):
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
        if t in ["prior_chemo_platinum", "prior_cisplatin", "chemoradiotherapy"]:
            prior_platinum[i] = 1
        if t in ["surgery_only", "surgery_plus_radiation",
                 "surgery_radiation_chemotherapy", "surgery_intent_curative",
                 "surgery_intent_palliative"]:
            prior_surgery[i] = 1
    return prior_radiation, prior_chemo, prior_platinum, prior_surgery

def generate_age(n, target_age, age_type, age_min, age_max, age_sd):
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
        diff = (target_age - np.mean(age)) if age_type == "mean" else (target_age - np.median(age))
        if abs(diff) < 0.01:
            break
        idx = np.random.choice(n, size=int(n * 0.3), replace=False)
        age[idx] += np.sign(diff)
        age = np.clip(age, age_min, age_max)
    age[age.argmin()] = age_min
    age[age.argmax()] = age_max
    return age

# ============================================================================
# Guyot reconstruction (unchanged from the corrected Monk/Long/Kitagawa/Miller
# pipelines -- this script uses Guyot exclusively, no Weibull)
# ============================================================================
CENSORING_METHOD = 'uniform'
CENSORING_MAX_FACTOR = 1.1

def guyot_reconstruct_ipd(t_digitized, S_digitized, n_patients, tot_events=None, arm_id=1, random_state=None):
    if random_state is None:
        random_state = SEED + arm_id
    np.random.seed(random_state)

    points = sorted(set(zip(t_digitized, S_digitized)))
    t = np.array([p[0] for p in points])
    S = np.array([p[1] for p in points])
    for i in range(1, len(S)):
        if S[i] > S[i - 1]:
            S[i] = S[i - 1]
    S = np.clip(S, 0.0, 1.0)

    event_times, event_probs = [], []
    for i in range(1, len(t)):
        if S[i] < S[i - 1]:
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
        n_risk_before.append(n_i - d_i)
        S_before.append(S_curr)

    event_times_list = []
    for i, t_event in enumerate(event_times):
        d_i = n_risk_before[i] - n_risk_before[i + 1]
        event_times_list.extend([t_event] * d_i)

    n_events = len(event_times_list)
    n_censored = n_patients - n_events

    if n_censored > 0:
        last_time = max(t[-1], max(event_times))
        censor_times = np.random.uniform(last_time, last_time * CENSORING_MAX_FACTOR, n_censored)
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
    return pd.DataFrame({'time': all_times[idx], 'event': all_events[idx]})


def empirical_quantile_remap(adjusted_values, base_values, followup):
    """
    Rank-preserving remap onto the EMPIRICAL distribution of base_values
    (the raw Guyot-reconstructed draw from the real digitized KM curve,
    before covariate adjustment) -- same mechanism used in the corrected
    Long 2005 / Miller 2014 pipelines. Covariate multipliers (age, PS,
    response) change each patient's time by a different factor, which --
    left alone -- shifts the marginal shape away from the real curve; a
    single median-rescale only fixes the median, leaving the tail visibly
    off. This keeps each patient's RELATIVE RANK (so covariate effects
    still matter) but replaces the MAGNITUDE at that rank with the
    corresponding order statistic from the real Guyot draw.
    """
    n = len(adjusted_values)
    base_sorted = np.sort(base_values)
    ranks = pd.Series(adjusted_values).rank(method='first').astype(int).values - 1
    ranks = np.clip(ranks, 0, n - 1)
    remapped = base_sorted[ranks]
    return np.clip(remapped, 0.1, followup)


# ================================================================
# Main Data Generation Function
# ================================================================
def generate_clinical_data(trial_config, arm_name):
    """Generate clinical data for one arm using Guyot reconstruction for OS/PFS."""
    np.random.seed(SEED)
    n = trial_config["patient_number"]
    FOLLOWUP = trial_config["followup"]
    CYCLE_COMPLETION_RATE = trial_config.get("cycle_completion_rate", 0.50)
    OS_ORR_WEIGHT = trial_config["os_orr_weight"]
    OS_AGE_WEIGHT = trial_config["os_age_weight"]
    PFS_AGE_WEIGHT = trial_config["pfs_age_weight"]
    PFS_RESPONSE_WEIGHT = trial_config["pfs_response_weight"]
    PFS_TOXICITY_WEIGHT = trial_config["pfs_toxicity_weight"]
    ORR_VS_PS = trial_config["orr_vs_ps"]
    ORR_VS_PELVIC_ONLY = trial_config["orr_vs_pelvic_only"]
    TOLERABILITY_ORR_WEIGHT = trial_config["tolerability_orr_weight"]
    PFSR_VS_PS = trial_config["pfsr_vs_ps"]
    TARGET_AGE = trial_config["target_age"]
    AGE_TYPE = trial_config.get("age_type", "median")
    AGE_MIN = trial_config["age_min"]
    AGE_MAX = trial_config["age_max"]
    AGE_SD = trial_config["age_sd"]
    CR_COUNT = trial_config["cr_count"]
    PR_COUNT = trial_config["pr_count"]
    SD_COUNT = trial_config["sd_count"]
    ALIVE_COUNT = trial_config["alive_count"]
    PFS_ALIVE_COUNT = trial_config["pfs_alive_count"]

    arm_id = {'bloss_a': 1, 'bloss_b': 2}[arm_name]

    # ----------------------------------------------------------------
    # 1. Age
    # ----------------------------------------------------------------
    age = generate_age(n, TARGET_AGE, AGE_TYPE, AGE_MIN, AGE_MAX, AGE_SD)

    # ----------------------------------------------------------------
    # 2. Cycle completion & tolerability (self-proposed constants -- see
    #    weight reference table; same architecture as every other trial)
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
    # 4. Disease nature / FIGO stage (descriptive only) / site of disease
    #    (internal covariate only -- see E1 note above)
    # ----------------------------------------------------------------
    nature_dist = expand_distribution(trial_config.get("disease_nature", {}), n)
    disease_nature_list = []
    for nat_raw in nature_dist:
        nat_lower = str(nat_raw).lower().strip()
        if "persistent" in nat_lower:
            disease_nature_list.append("Persistent")
        elif "recurrent" in nat_lower:
            disease_nature_list.append("Recurrent")
        else:
            disease_nature_list.append("Advanced (IVB)")
    figo_stage_list = ["IVB" if nat == "Advanced (IVB)" else "Unknown" for nat in disease_nature_list]

    site_dist = expand_distribution(trial_config.get("site_of_disease", {}), n)
    pelvic_only = np.array([1 if str(s) == "Pelvis only" else 0 for s in site_dist])

    # ----------------------------------------------------------------
    # 5. Performance status (base assignment from exact trial counts, then
    #    tolerability-driven worsening, then forced back onto the exact
    #    reported marginal counts -- same architecture as every other trial)
    # ----------------------------------------------------------------
    target_ps_dist = trial_config["performance_status"]
    ps1_count = target_ps_dist.get(1, 0)   # GOG PS0
    ps2_count = target_ps_dist.get(2, 0)   # GOG PS1
    ps3_count = target_ps_dist.get(3, 0)   # GOG PS2
    PS_discrete = np.array([1] * ps1_count + [2] * ps2_count + [3] * ps3_count)
    if len(PS_discrete) < n:
        PS_discrete = np.concatenate([PS_discrete, np.full(n - len(PS_discrete), 2)])
    PS_discrete = np.random.permutation(PS_discrete[:n])

    worsen_idx = np.where(tolerability < 0.35)[0]
    if len(worsen_idx) > 0:
        worsen_sample = np.random.choice(worsen_idx, size=int(0.1 * len(worsen_idx)), replace=False)
        PS_discrete[worsen_sample] = np.clip(PS_discrete[worsen_sample] + 1, 1, 3)

    for target_code, target_n in [(1, ps1_count), (2, ps2_count), (3, ps3_count)]:
        current_n = np.sum(PS_discrete == target_code)
        diff = target_n - current_n
        if diff > 0:
            candidates = np.where(PS_discrete != target_code)[0]
            if len(candidates) >= diff:
                chosen = np.random.choice(candidates, size=diff, replace=False)
                PS_discrete[chosen] = target_code
        elif diff < 0:
            candidates = np.where(PS_discrete == target_code)[0]
            if len(candidates) >= abs(diff):
                chosen = np.random.choice(candidates, size=abs(diff), replace=False)
                for alt_code, alt_n in [(1, ps1_count), (2, ps2_count), (3, ps3_count)]:
                    if alt_code != target_code and np.sum(PS_discrete == alt_code) < alt_n:
                        n_move = min(len(chosen), alt_n - np.sum(PS_discrete == alt_code))
                        PS_discrete[chosen[:n_move]] = alt_code
                        chosen = chosen[n_move:]
                        if len(chosen) == 0:
                            break

    PS_normalized = (PS_discrete - 1) / 3.0

    # ----------------------------------------------------------------
    # 6. Tumor response: CR / PR / SD / PD
    #    FIX B2: built directly and EXACTLY from Table 3's reported counts,
    #    with WHICH patients land where driven by a transparent
    #    latent-propensity index using two covariates: PS (self-proposed)
    #    and site of disease (self-proposed weight on a real, P<.001 cited
    #    finding). No NE category needed for this trial (see header note).
    # ----------------------------------------------------------------
    target_orr_count = CR_COUNT + PR_COUNT
    ps_z = zscore(PS_discrete)
    pelvic_z = zscore(pelvic_only.astype(float))
    tol_z = zscore(tolerability)

    response_propensity = (
        ORR_VS_PS * ps_z +
        ORR_VS_PELVIC_ONLY * pelvic_z +
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
            np.random.standard_normal(len(non_responders))
        )
        sd_rank = np.argsort(pfsr_propensity)[-sd_n:]
        sd_idx = non_responders[sd_rank]
    else:
        sd_idx = np.array([], dtype=int)
    pd_idx = np.setdiff1d(non_responders, sd_idx)

    PFSR = np.zeros(n, dtype=int)   # disease-control flag: CR/PR/SD=1, PD=0
    PFSR[orr_idx] = 1
    PFSR[sd_idx] = 1

    response = np.empty(n, dtype=object)
    response[cr_idx] = "CR"
    response[pr_idx] = "PR"
    response[sd_idx] = "SD"
    response[pd_idx] = "PD"

    # ----------------------------------------------------------------
    # 7. Histology, tumor grade, race/ethnicity
    # ----------------------------------------------------------------
    histology = expand_distribution(trial_config["histology"], n)
    grade = expand_distribution(trial_config["grades"], n)
    ethnicity = expand_distribution(trial_config["ethnicity"], n)

    # ================================================================
    # 8. OS / PFS generation (Guyot reconstruction from digitized KM curves
    #    ONLY -- no Weibull)
    # ================================================================
    os_ipd = guyot_reconstruct_ipd(
        trial_config["os_time_grid"], trial_config["os_surv_grid"], n,
        tot_events=n - ALIVE_COUNT, arm_id=arm_id, random_state=SEED + arm_id
    )

    age_factor = 1 - (age - np.mean(age)) / np.std(age) * OS_AGE_WEIGHT * 2
    age_factor = np.clip(age_factor, 0.4, 1.6)

    # PS -> OS: TRIAL-CITED (Results: PS1 HR=1.37, PS2 HR=1.70, P=.009) --
    # the ONLY trial in this series with directly-cited PS hazard ratios
    # for OS, no borrowing from Monk needed.
    ps_time_multiplier_os = np.select(
        [PS_discrete == 2, PS_discrete == 3],
        [hr_to_time_multiplier(PS1_HAZARD_RATIO_OS, 1.0), hr_to_time_multiplier(PS2_HAZARD_RATIO_OS, 1.0)],
        default=1.0
    )

    orr_time_multiplier = np.where(ORR == 1,
                                    hr_to_time_multiplier(ORR_OS_HR_ASSUMED, OS_ORR_WEIGHT),
                                    1.0)

    os_covariate_factor = age_factor * ps_time_multiplier_os * orr_time_multiplier
    os_adjusted = os_ipd['time'].values * os_covariate_factor
    os_adjusted = np.clip(os_adjusted, 0.1, FOLLOWUP)

    # Empirical rank-remap onto the real Guyot draw itself, then full
    # median correction (exact-median convention, same as every other
    # trial in this series).
    observed_os = empirical_quantile_remap(os_adjusted, os_ipd['time'].values, FOLLOWUP)
    current_median = np.median(observed_os)
    if current_median > 0:
        scaling_factor = trial_config["target_os"] / current_median
        observed_os = observed_os * scaling_factor
    observed_os = np.round(np.clip(observed_os, 0.1, FOLLOWUP), 1)

    n_events_os = n - ALIVE_COUNT
    order_os = np.argsort(observed_os)
    event_os = np.zeros(n, dtype=int)
    event_os[order_os[:n_events_os]] = 1
    alive = 1 - event_os

    # ---------------- PFS ----------------
    pfs_ipd = guyot_reconstruct_ipd(
        trial_config["pfs_time_grid"], trial_config["pfs_surv_grid"], n,
        tot_events=n - PFS_ALIVE_COUNT, arm_id=arm_id, random_state=SEED + arm_id + 10
    )
    observed_pfs = np.round(pfs_ipd['time'].values, 1)
    event_pfs = pfs_ipd['event'].values

    pfs_age_factor = 1 - (age - np.mean(age)) / np.std(age) * PFS_AGE_WEIGHT * 2
    observed_pfs = observed_pfs * pfs_age_factor

    # PS -> PFS: TRIAL-CITED (Results: PS1 HR=1.40, PS2 HR=1.54, P=.013)
    ps_time_multiplier_pfs = np.select(
        [PS_discrete == 2, PS_discrete == 3],
        [hr_to_time_multiplier(PS1_HAZARD_RATIO_PFS, 1.0), hr_to_time_multiplier(PS2_HAZARD_RATIO_PFS, 1.0)],
        default=1.0
    )
    observed_pfs = observed_pfs * ps_time_multiplier_pfs

    # FIX B1: histology now MULTIPLIES (previously divided) -- though this
    # trial's eligibility requires squamous histology for all patients, so
    # the multiplier is 1.00 throughout and this has no practical effect
    # here; fixed anyway for architectural consistency.
    histology_weights = np.array([trial_config["pfs_histology_weight"][h] for h in histology])
    observed_pfs = observed_pfs * histology_weights

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
    tox_counts_arr = np.array([int(round(min(0.98, tox_counts_raw[name] / n) * n)) for name in toxicity_names])

    n_tox = len(toxicity_names)
    tox_corr = np.full((n_tox, n_tox), 0.2)
    np.fill_diagonal(tox_corr, 1.0)

    hematologic = {'leucopenia', 'neutropenia', 'thrombocytopenia', 'anemia'}
    infectious = {'other infection/fever'}
    gi = {'nausea/vomiting'}
    for i, ti in enumerate(toxicity_names):
        for j, tj in enumerate(toxicity_names):
            if i != j:
                ti_l, tj_l = ti.lower(), tj.lower()
                if any(ti_l in c and tj_l in c for c in [hematologic, infectious, gi]):
                    tox_corr[i, j] = 0.7
                elif (ti_l in hematologic and tj_l in infectious) or (ti_l in infectious and tj_l in hematologic):
                    tox_corr[i, j] = 0.5

    eigvals, eigvecs = np.linalg.eigh(tox_corr)
    eigvals[eigvals < 1e-6] = 1e-6
    tox_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(tox_corr)
    latent = np.random.standard_normal((n, n_tox)) @ L.T

    for j, tox in enumerate(toxicity_names):
        tox_lower = tox.lower()
        if tox_lower in hematologic:
            latent[:, j] += (0.4 * PRIOR_RAD + 0.3 * PRIOR_CHEMO + 0.25 * PRIOR_PLATINUM)
        # Bleomycin (CIB arm) -> pulmonary toxicity: self-proposed nudge on
        # top of the already-exact arm-level target count (Table 2), only
        # affects WHICH patients within the arm are selected.
        if arm_name == 'bloss_b' and tox_lower == 'pulmonary':
            latent[:, j] += 0.5

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

    toxicity_events = [[toxicity_names[j] for j in range(n_tox) if tox_matrix[i, j] == 1] for i in range(n)]
    toxicity_count = [len(x) for x in toxicity_events]

    toxicity_burden = np.array(toxicity_count) / n_tox
    pfs_toxicity_penalty = 1 - toxicity_burden * PFS_TOXICITY_WEIGHT
    observed_pfs = observed_pfs * pfs_toxicity_penalty
    observed_pfs = np.clip(observed_pfs, 0.1, FOLLOWUP)

    # Empirical rank-remap for PFS, then full median correction -- same
    # reasoning as OS.
    observed_pfs = empirical_quantile_remap(observed_pfs, pfs_ipd['time'].values, FOLLOWUP)
    current_pfs_median = np.median(observed_pfs)
    if current_pfs_median > 0:
        pfs_scaling_factor = trial_config["target_pfs"] / current_pfs_median
        observed_pfs = observed_pfs * pfs_scaling_factor
    observed_pfs = np.round(np.clip(observed_pfs, 0.1, FOLLOWUP), 1)

    n_events_pfs = n - PFS_ALIVE_COUNT
    order_pfs = np.argsort(observed_pfs)
    event_pfs = np.zeros(n, dtype=int)
    event_pfs[order_pfs[:n_events_pfs]] = 1

    # ================================================================
    # 10. Cause of death ("no treatment-related deaths recorded in either
    #     regimen" -- cited; all deaths default to Disease)
    # ================================================================
    dead_idx = np.where(alive == 0)[0]
    causes = np.full(n, "Alive", dtype=object)
    cause_dist = trial_config.get("causes", {})
    if len(dead_idx) > 0:
        treatment_n = cause_dist.get("Treatment", 0) or 0
        other_n = cause_dist.get("Other/Unknown", 0) or 0
        toxicity_risk = np.array(toxicity_count)
        prognosis_risk = PS_normalized + (1 - ORR) + (1 - PFSR)
        combined_risk = toxicity_risk + prognosis_risk
        ranked_dead = dead_idx[np.argsort(combined_risk[dead_idx])[::-1]]
        treatment_idx = ranked_dead[:treatment_n]
        causes[treatment_idx] = "Treatment"
        remaining = np.setdiff1d(dead_idx, treatment_idx)
        other_idx = remaining[:other_n]
        causes[other_idx] = "Other/Unknown"
        remaining = np.setdiff1d(remaining, other_idx)
        causes[remaining] = "Disease"
        causes = np.where(alive == 1, "Alive", causes)

    def uniform_column(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ['No Drug'] * n
        if isinstance(value, (tuple, list)):
            return [",".join(map(str, value))] * n
        val_str = str(value).strip()
        if val_str.lower() in ['no drug', 'none', 'nan', '', '0', '0.0']:
            return ['No Drug'] * n
        return [value] * n

    # ================================================================
    # Final DataFrame (column names/order identical to the Monk/Long/
    # Kitagawa/Miller pipelines)
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
# Usage - Creating the two arms and combined dataset
# ================================================================
df_bloss_a = generate_clinical_data(TRIALS["bloss_a"], arm_name='bloss_a')
df_bloss_b = generate_clinical_data(TRIALS["bloss_b"], arm_name='bloss_b')

df_bloss = pd.concat([df_bloss_a, df_bloss_b], ignore_index=True)\
             .sample(frac=1, random_state=42).reset_index(drop=True)
df_bloss.to_excel("bloss_km.xlsx")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluating Datasets
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def standardize_dict_order(output_dict, target_dict):
    if not isinstance(output_dict, dict) or not isinstance(target_dict, dict):
        return output_dict
    if target_dict:
        sample_value = next(iter(target_dict.values()))
        default = 0 if isinstance(sample_value, (int, float)) else 'Missing'
    else:
        default = 0
    return {k: output_dict.get(k, default) for k in target_dict.keys()}

def get_toxicity_counts(toxicity_events, reference_toxicity):
    flat_list = [tox for patient in toxicity_events for tox in patient]
    counts = Counter(flat_list)
    return {tox: counts.get(tox, 0) for tox in reference_toxicity.keys()}

def evaluate_trial(trial_name, df):
    trial_cfg = TRIALS[trial_name]
    n = trial_cfg['patient_number']

    response_target = {"CR": trial_cfg['cr_count'], "PR": trial_cfg['pr_count'],
                        "SD": trial_cfg['sd_count'], "PD": trial_cfg['pd_count']}
    disease_control_rate_target = round(
        (trial_cfg['cr_count'] + trial_cfg['pr_count'] + trial_cfg['sd_count']) / n, 3
    )
    orr_target = round((trial_cfg['cr_count'] + trial_cfg['pr_count']) / n, 3)

    constants = [
        n, trial_cfg['followup'], orr_target, disease_control_rate_target, response_target,
        trial_cfg['target_os'], trial_cfg['target_pfs'],
        trial_cfg['target_age'], trial_cfg['age_min'], trial_cfg['age_max'], trial_cfg['age_sd'],
        trial_cfg['alive_count'],
        trial_cfg['ethnicity'], trial_cfg['histology'], trial_cfg['disease_nature'],
        trial_cfg['disease_stage'], trial_cfg['grades'], trial_cfg['causes'],
        trial_cfg['performance_status'], trial_cfg['toxicity']
    ]

    output = [
        len(df), float(df['followup'].iloc[0]), round(df['orr'].mean(), 3), round(df['pfsr'].mean(), 3),
        df['response'].value_counts().to_dict(),
        round(df['os_months'].median(), 2), round(df['pfs_months'].median(), 2),
        int(df['age'].median()), int(df['age'].min()), int(df['age'].max()), round(df['age'].std(ddof=0), 2),
        (df['vital_status'] == "Alive").sum(),
        df['ethnicity'].value_counts().to_dict(), df['histology/cell_type'].value_counts().to_dict(),
        df['disease_nature'].value_counts().to_dict(), df['figo_stage'].value_counts().to_dict(),
        df['tumor_grade'].value_counts().to_dict(), df['cause_of_death'].value_counts().to_dict(),
        df['performance_status'].value_counts().to_dict(),
        get_toxicity_counts(df["toxicity_events"], trial_cfg['toxicity'])
    ]

    row_names = [
        "Patient Number", "Followup", "ORR", "PFSR (disease control rate)", "Tumor Response (CR/PR/SD/PD)",
        "Overall Survival (months)", "Progression-Free Survival (months)",
        "Age", "Minimum Age", "Maximum Age", "Age SD", "Alive Counts",
        "Ethnicity", "Histology", "Disease Type", "Disease Stage",
        "Tumor Grade", "Death Cause", "Performance Status", "Toxicity"
    ]

    eval_df = pd.DataFrame({"output_values": output, "target_values": constants}, index=row_names)
    for i, row in enumerate(row_names):
        target_val, output_val = constants[i], output[i]
        if isinstance(target_val, dict) and isinstance(output_val, dict):
            eval_df.at[row, "output_values"] = standardize_dict_order(output_val, target_val)
    return eval_df

trial_datasets = {"bloss_a": df_bloss_a, "bloss_b": df_bloss_b}
evaluation_results = {name: evaluate_trial(name, df) for name, df in trial_datasets.items()}
combined_evaluation_bloss = pd.concat(evaluation_results, axis=1)

pd.set_option('display.max_colwidth', 120)
combined_evaluation_bloss.to_excel("bloss_evaluation_km.xlsx")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SELF-PROPOSED CORRELATION / WEIGHT REFERENCE TABLE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
weight_reference_table = pd.DataFrame([
    {"category": "Demographics", "parameter": "disease_nature Persistent/Recurrent split of prior-treated patients",
     "value": "18/114 (CI), 17/106 (CIB)", "applies_to": "disease_nature construction", "cited_in_bloss2002": "PARTIALLY",
     "note": "Advanced(IVB)=14/18 IS cited (stage-IVB treatment-naive count). The Persistent-vs-Recurrent split of the remaining prior-treated patients is NOT reported anywhere in this paper and is self-proposed."},
    {"category": "Response modeling", "parameter": "CR:PR split of responders", "value": "5:42 (CI), 4:40 (CIB)",
     "applies_to": "Response category construction", "cited_in_bloss2002": "NO",
     "note": "Table 3 gives only Responders/Nonresponders (47/99, 44/97) -- no CR/PR breakdown is reported anywhere."},
    {"category": "Response modeling", "parameter": "SD:PD split of non-responders", "value": "54:45 (CI), 53:44 (CIB)",
     "applies_to": "Response category construction", "cited_in_bloss2002": "NO",
     "note": "No SD/PD breakdown is reported at all for this trial (unlike Monk/Long, which give this directly)."},
    {"category": "Response modeling", "parameter": "orr_vs_pelvic_only", "value": -0.45,
     "applies_to": "Response propensity index", "cited_in_bloss2002": "DIRECTION AND COVARIATE YES, WEIGHT MAGNITUDE NO",
     "note": "Site-of-disease effect on response is strongly cited (pelvic-only 24.1%/16.1% vs extra-pelvic 37.0%/41.2% response, P<.001) -- the covariate itself is trial-exact; only the propensity-index WEIGHT translating it into a magnitude is self-proposed."},
    {"category": "Response modeling", "parameter": "orr_vs_ps", "value": -0.30,
     "applies_to": "Response propensity index", "cited_in_bloss2002": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "tolerability_orr_weight", "value": 0.20,
     "applies_to": "Response propensity index", "cited_in_bloss2002": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "pfsr_vs_ps", "value": -0.35,
     "applies_to": "Stable-vs-progressive disease propensity index", "cited_in_bloss2002": "NO", "note": "-"},
    {"category": "Survival reconstruction", "parameter": "alive_count / pfs_alive_count", "value": "12/9 (CI), 14/10 (CIB)",
     "applies_to": "Guyot tot_events target", "cited_in_bloss2002": "NO",
     "note": "No exact final event/censoring count is reported in this paper. Values are anchored to the digitized curves' own tail level (~8% OS, ~6% PFS) but are still self-proposed estimates, not read off the paper directly."},
    {"category": "OS modeling", "parameter": "os_age_weight", "value": 0.2,
     "applies_to": "Age -> OS (AFT exponent)", "cited_in_bloss2002": "NO", "note": "-"},
    {"category": "OS modeling", "parameter": "os_orr_weight / ORR_OS_HR_ASSUMED", "value": "0.3 / 0.70",
     "applies_to": "Response -> OS", "cited_in_bloss2002": "NO", "note": "General oncology-literature association assumed."},
    {"category": "PFS modeling", "parameter": "pfs_age_weight", "value": 0.15,
     "applies_to": "Age -> PFS", "cited_in_bloss2002": "NO", "note": "-"},
    {"category": "PFS modeling", "parameter": "pfs_histology_weight (dict)", "value": "0.50-1.15 by histology",
     "applies_to": "Histology -> PFS multiplier", "cited_in_bloss2002": "NO",
     "note": "Reused unchanged for cross-trial consistency; has NO practical effect in this trial since eligibility required squamous histology for all patients (multiplier=1.00 throughout)."},
    {"category": "PFS modeling", "parameter": "pfs_response_weight / PFS_RESPONSE_HR_ASSUMED", "value": "0.3 / 0.65",
     "applies_to": "Response -> PFS", "cited_in_bloss2002": "NO", "note": "-"},
    {"category": "PFS modeling", "parameter": "pfs_toxicity_weight", "value": 0.1,
     "applies_to": "Toxicity burden -> PFS", "cited_in_bloss2002": "NO", "note": "-"},
    {"category": "Performance status", "parameter": "PS worsening (tolerability)", "value": "threshold 0.35, fraction 0.10",
     "applies_to": "Low-tolerability patients' PS", "cited_in_bloss2002": "NO", "note": "Reused from the other pipelines."},
    {"category": "Tolerability", "parameter": "cycle_completion_rate (CI/CIB)", "value": "0.48 / 0.52",
     "applies_to": "Fraction assumed to complete all 6 planned cycles", "cited_in_bloss2002": "NO",
     "note": "Paper reports median cycles received (4/4 of a max 6, range 0-12/0-9) but not the % completing all 6."},
    {"category": "Tolerability", "parameter": "tolerability formula constants", "value": "0.82, 0.5, 0.09, 0.18",
     "applies_to": "Tolerability / toxicity-penalty formula", "cited_in_bloss2002": "NO", "note": "Reused unchanged."},
    {"category": "Toxicity", "parameter": "tox_risk formula weights", "value": "0.4/0.35/0.35/0.25/0.15",
     "applies_to": "Per-patient toxicity latent-risk multiplier", "cited_in_bloss2002": "NO", "note": "Reused unchanged."},
    {"category": "Toxicity", "parameter": "tox_corr matrix", "value": "0.2 baseline / 0.7 within-cluster / 0.5 heme-infection cross",
     "applies_to": "Cross-toxicity correlation structure", "cited_in_bloss2002": "NO", "note": "Reused unchanged."},
    {"category": "Toxicity", "parameter": "Bleomycin (CIB) pulmonary latent nudge", "value": 0.5,
     "applies_to": "Within-arm patient selection for pulmonary toxicity", "cited_in_bloss2002": "Directionally cited (18% v 5%), magnitude self-proposed",
     "note": "Arm-level target count is already exact from Table 2 -- this only affects WHICH patients within the arm are selected."},
    {"category": "Demographics", "parameter": "age_sd", "value": "(age_max-age_min)/4",
     "applies_to": "Age distribution spread", "cited_in_bloss2002": "NO", "note": "Range/4 heuristic, reused from the other pipelines."},
])

print("\n--- Self-proposed weight reference table (for later citation) ---")
print(weight_reference_table.to_string(index=False))
weight_reference_table.to_excel("bloss_correlation_weight_reference.xlsx", index=False)

combined_evaluation_bloss
