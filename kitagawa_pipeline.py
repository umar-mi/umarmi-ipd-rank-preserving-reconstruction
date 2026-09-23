#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# KITAGAWA 2015 (JCOG0505) -- SYNTHETIC IPD GENERATION PIPELINE (CORRECTED)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Source: Kitagawa R, Katsumata N, Shibata T, et al. Paclitaxel Plus
# Carboplatin Versus Paclitaxel Plus Cisplatin in Metastatic or Recurrent
# Cervical Cancer: The Open-Label Randomized Phase III Trial JCOG0505.
# J Clin Oncol. 2015;33(19):2129-2135.
#
# Two arms: TP (paclitaxel+cisplatin, n=127) and TC (paclitaxel+carboplatin,
# n=126). This is a noninferiority trial (TC vs TP for OS).
#
# CHANGE LOG relative to the version you pasted
# ------------------------------------------------------------------------
# This script had the same class of bugs the ORIGINAL (pre-fix) Monk 2009
# script had -- it looks like it was branched from that draft before the
# corrections were made. Same bug IDs are reused here for traceability.
#
#  [Bug fixes -- unambiguous, evidence-based]
#  B1. `observed_pfs = observed_pfs / histology_weights` DIVIDED by the
#      histology weight (lengthened PFS for "worse" histologies). Fixed to
#      MULTIPLY.
#  B2. `target_pfsr` was documented as "~50%/~48% PFS at 6 months" (a KM
#      landmark rate) but then used as a disease-control-rate target forced
#      onto the PFSR flag via adjust_binary -- the same category error as
#      Monk's original bug. Response (CR/PR/SD/PD/NE) is now built directly
#      from the paper's own reported counts (Complete/Partial response
#      rates, applied to the 102/99 RECIST-evaluable patients -- see notes
#      below), with the same exact-count + latent-propensity mechanism used
#      in the corrected Monk/Long pipelines.
#  B3. No target existed for median PFS in months. Added `target_pfs`
#      (6.9 / 6.2 months, Fig 2B) and a rescale-to-target step mirroring OS.
#  B4. PFS event count used a flat `int(0.92*n)` approximation. The paper
#      reports EXACT PFS event counts (115/123 TP, 113/121 TC, Fig 2B) --
#      used directly instead (see population-accounting note below).
#  B5. `os_ps_weight`, `os_orr_weight`, `os_hazard_ratio`, `age_latent_risk`
#      were extracted from trial_config but never applied to the generated
#      survival TIMES. PS is now wired into OS the same way as the Monk/Long
#      pipelines (see PS_HAZARD_RATIO_OS_ASSUMED note -- NOT Kitagawa-cited).
#      `os_hazard_ratio` is retained for documentation/QC only, not
#      reapplied multiplicatively (avoids double-counting the arm effect
#      already encoded via the digitized curve + median-rescale).
#  B6. The 4-way `correlated_binary` copula draw was built, then partly
#      discarded, patched with ad hoc disease-based flips, rebuilt via a
#      "recalculate PFSR" block, and finally forced to targets via
#      adjust_binary -- the same wasted-mechanism pattern as Monk's original
#      bug. Replaced with the transparent latent-propensity-index approach
#      from the corrected Monk/Long pipelines.
#  B7 (NEW, specific to this script). `PS_discrete` was carefully adjusted
#      for disease-status effects across ~30 lines, then completely
#      OVERWRITTEN a few lines later by a fresh
#      `np.random.permutation(ps_target_list)` that discarded all of it --
#      none of that disease-status/PS correlation ever reached the output.
#      Fixed by removing the redundant second assignment.
#
#  [Data-mapping corrections]
#  C1. `treatment_kitagawa_a/b` put the SAME count (54/60) into BOTH
#      "prior_chemo_platinum" and "prior_cisplatin", which only summed to
#      108/120 of 127/126 -- expand_distribution silently padded the rest
#      with whichever category numpy's max() happened to pick. Table 1
#      actually reports: Cisplatin 54/60, Carboplatin+Other 7/12, No prior
#      platinum 66/54. Also, "Prior irradiation" (100/108) is a SEPARATE,
#      larger marginal count than "prior platinum" (61/72) -- the original
#      code never encoded radiation exposure at all. Recoded as
#      chemoradiotherapy=61/72 (assumed = all prior-platinum patients, since
#      concurrent cisplatin-based chemoradiotherapy is the overwhelmingly
#      standard scenario for prior treatment of primary cervical cancer),
#      radiation_only=39/36 (irradiated but no prior platinum), none=27/18.
#      This is a documented assumption (the paper doesn't cross-tabulate
#      radiation x platinum jointly), flagged in the weight reference table.
#  C2. `CAUSE_kitagawa_b = [1, 97, 1]` sums to 99, but the TC arm's actual
#      event count is 98 (Fig 2A). Fixed to `[1, 96, 1]`.
#  C3. `performance_status = [0,1,2,3,4]` used raw ECOG codes, but the
#      corrected Monk/Long pipelines use a shift-by-one convention
#      (code = ECOG_PS + 1) so the same column values mean the same thing
#      across all 5 trials' output. Recoded to match (code 1=ECOG0,
#      code 2=ECOG1, code 3=ECOG2) -- this is a genuine cross-trial merge
#      hazard, not just a style preference: without it, Kitagawa's coded
#      "1" (ECOG 0) would collide with Monk's coded "1" (which means GOG
#      PS 0 too, coincidentally consistent here, but Kitagawa's coded "2"
#      would NOT match Monk's coded "2" meaning without the shift).
#  C4. `disease_nature` still used the OLD "Non-specified" label for the
#      third category; the corrected Monk/Long schema uses "Advanced (IVB)"
#      for the same slot. Relabeled for consistency. Table 1's "IVB or
#      persistent" is reported as a SINGLE COMBINED count (27/24) -- the
#      paper does not let these be separated. Split via a self-proposed,
#      documented ratio (45% Persistent / 55% Advanced-IVB) rather than
#      guessing at an undocumented one-off number -- flagged in the weight
#      reference table.
#  C5. `never_treated_count=66/54` was mislabeled "no prior platinum" but
#      that parameter actually controls "never received any cycle of the
#      CURRENT protocol regimen" -- a different concept the paper doesn't
#      give a specific count for. Set to 0 for both arms (no evidence
#      otherwise) rather than reusing a number that means something else.
#
#  [Hygiene / dead code removed]
#  D1. Removed `shape_k`, `age_latent_risk` (never referenced in
#      generation), and `asian_weight`/`black_weight`/`american_indian_weight`
#      (meaningless here -- this is a 100%-Asian Japanese trial, so an
#      ethnicity-specific weight can never differentiate anything).
#
#  [New, not corrections -- data the paper gives that the original script
#   did not use at all]
#  E1. Platinum-free interval (Table 1: <6 / 6-11.9 / >=12 months / none) is
#      a real, reported prognostic stratifier that was not modeled at all.
#      Added as a self-proposed (NOT directly cited -- see note below)
#      hazard modulator for OS/PFS, replacing a separate disease_nature-based
#      hazard (kept disease_nature purely descriptive to avoid stacking two
#      uncited hazard mechanisms measuring overlapping "how advanced is the
#      disease" concepts).
#  E2. The paper's Figure 2 legend includes a NUMBER-AT-RISK table (years
#      0-5) and 1-/2-/3-year landmark survival rates for both OS and PFS --
#      absent from Monk 2009 and Long 2005. This is added as an additional
#      validation check in the evaluation table at the end of this script
#      (not used to constrain generation, just to sanity-check the output).
#
# IMPORTANT CAVEAT on Fig 3 (subgroup forest plot): those are TC-vs-TP
# TREATMENT-INTERACTION hazard ratios within each subgroup (e.g., "HR for
# TC vs TP is 1.44 among PS>=1 patients"), NOT baseline prognostic hazard
# ratios for the covariate itself the way Monk 2009's Fig 3 was. They are
# NOT used here as if they were prognostic HRs -- doing so would be a
# category error like the one already fixed in target_pfsr (B2).
#
# POPULATION ACCOUNTING: this paper reports THREE different denominators:
# baseline/randomized (n=127/126, Table 1), safety/as-treated (n=125/126,
# Table 2), and efficacy/eligible (n=123/121, Fig 2 survival + response).
# This script uses n=127/126 throughout (matching baseline characteristics,
# the most inclusive population) and maps the efficacy-population survival
# and response data onto it directly (e.g., ALIVE_COUNT = 127-106 = 21,
# matching Fig 2A's TP event count exactly). The gap between 127 and 123
# (4 patients ineligible for efficacy analysis) and between 123 and 102
# (21 patients without RECIST-measurable disease) are both folded into the
# response column's "NE" (not evaluable) category -- same convention used
# for Long 2005's inassessable patients.
# ============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Importing Libraries
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from collections import Counter

SEED = 42
KITAGAWA_A_NUMBER = 127
KITAGAWA_B_NUMBER = 126

#=======================================================
# LITERATURE- AND TRIAL-DERIVED EFFECT SIZES
#=======================================================
# NOT reported in Kitagawa et al. 2015 -- self-proposed placeholders
# requiring an external literature citation. Catalogued in the weight
# reference table at the end of this script.
PS_HAZARD_RATIO_OS_ASSUMED = 1.799   # borrowed from Monk et al. 2009 (same
                                      # disease, different trial) -- Kitagawa
                                      # 2015 does not report a baseline
                                      # prognostic PS->OS hazard ratio (its
                                      # Fig 3 subgroup HRs are TC-vs-TP
                                      # treatment-interaction effects, not
                                      # prognostic HRs -- see caveat above).
ORR_OS_HR_ASSUMED = 0.70
PFS_RESPONSE_HR_ASSUMED = 0.65

# Self-proposed platinum-free-interval (PFI) hazard modulators. Table 1
# reports the PFI distribution itself (a real stratification factor), and
# Fig 3 shows PFI-stratified TREATMENT-INTERACTION HRs (not prognostic HRs
# -- see caveat above), so the DIRECTION here (shorter platinum-free
# interval = worse prognosis; platinum-naive = best prognosis) is clinically
# well established in gynecologic oncology, but the specific MAGNITUDES
# below are self-proposed, not read off this paper.
PFI_OS_RR = {'<6': 1.30, '6-12': 1.00, '12+': 0.80, 'none': 0.70}
PFI_PFS_RR = {'<6': 1.30, '6-12': 1.00, '12+': 0.80, 'none': 0.75}


def hr_to_time_multiplier(hazard_ratio, weight=1.0):
    """AFT-style time multiplier from a hazard ratio (exact under Weibull PH;
    a practical approximation otherwise). weight=1 applies HR as given."""
    hazard_ratio = np.asarray(hazard_ratio, dtype=float)
    return hazard_ratio ** (-float(weight))


#=======================================================
# DEFINING BASE LISTS
# (identical to the corrected Monk/Long pipelines' lists, for cross-trial
#  merge compatibility -- see C3/C4 notes above)
#=======================================================

histology_base = ["Squamous Cell Carcinoma", "Adenocarcinoma", "Adenosquamous", "Mucinous Adenocarcinoma",
                  "Clear Cell", "Endometrioid", "Villoglandular", "Undifferentiated Carcinoma", "Not Specified"]

ethnicity_base = ["White", "Black", "Asian", "American Indian", "Hispanic", "Filipino", "Unspecified"]

# FIX C4: "Non-specified" -> "Advanced (IVB)", matching the corrected
# Monk/Long schema.
disease_nature = ["Persistent", "Recurrent", "Advanced (IVB)"]

figo_stage = ["I", "II", "III", "IVA", "IVB", "Unknown"]

grade_base = ["Grade 1", "Grade 2", "Grade 3", "Grade Unspecified"]

death_cause_base = ["Treatment", "Disease", "Other/Unknown"]

dead_alive = ["Alive", "Dead", "Unspecified"]

# FIX C3: shift-by-one convention (code = ECOG_PS + 1), matching Monk/Long.
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
# DRUG COMBINATION (unchanged from your version -- verified correct against
# "Treatment" section: TP = paclitaxel 135mg/m2/24h day1 + cisplatin
# 50mg/m2 day2; TC = paclitaxel 175mg/m2/3h day1 + carboplatin AUC5
# immediately following, same day, Calvert formula)
#===========================================================
DRUG_kitagawa_a = {
    "Platinum_Drug": "cisplatin", "Platinum_Dose_Value": 50, "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based", "Platinum_Infusion_Duration_Hours": 1,  # not explicitly stated; 1h with hydration is standard practice, not cited (matches TC arm's own reasonable-default convention)
    "Platinum_Days_Number": 1, "Platinum_Day": 2,
    "Adjunct_Drug_1": "paclitaxel", "Adjunct_Drug_1_Dose_Value": 135, "Adjunct_Drug_1_Dose_Unit": "mg/m2",
    "Adjunct_Drug_1_Dose_Method": "BSA-based", "Adjunct_Drug_1_Infusion_Duration_Hours": 24,
    "Adjunct_Drug_1_Days_Number": 1, "Adjunct_Drug_1_Days": 1, "Adjunct_Drug_1_Same_Day_As_Platinum": 0,
    "Adjunct_Drug_2": 'No Drug', "Adjunct_Drug_2_Dose_Value": 0, "Adjunct_Drug_2_Dose_Unit": 'No Drug',
    "Adjunct_Drug_2_Dose_Method": 'No Drug', "Adjunct_Drug_2_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_2_Days_Number": 0, "Adjunct_Drug_2_Days": 0, "Adjunct_Drug_2_Same_Day_As_Platinum": 0,
    "Adjunct_Drug_3": 'No Drug', "Adjunct_Drug_3_Dose_Value": 0, "Adjunct_Drug_3_Dose_Unit": 'No Drug',
    "Adjunct_Drug_3_Dose_Method": 'No Drug', "Adjunct_Drug_3_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_3_Days_Number": 0, "Adjunct_Drug_3_Days": 0, "Adjunct_Drug_3_Same_Day_As_Platinum": 0,
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
    "Cycle_Length_Days": 21
}

DRUG_kitagawa_b = {
    "Platinum_Drug": "carboplatin", "Platinum_Dose_Value": 5, "Platinum_Dose_Unit": "AUC",
    "Platinum_Dose_Method": "Calvert", "Platinum_Infusion_Duration_Hours": 1,   # not explicitly stated ("immediately following"); 1h is a reasonable default, not cited
    "Platinum_Days_Number": 1, "Platinum_Day": 1,
    "Adjunct_Drug_1": "paclitaxel", "Adjunct_Drug_1_Dose_Value": 175, "Adjunct_Drug_1_Dose_Unit": "mg/m2",
    "Adjunct_Drug_1_Dose_Method": "BSA-based", "Adjunct_Drug_1_Infusion_Duration_Hours": 3,
    "Adjunct_Drug_1_Days_Number": 1, "Adjunct_Drug_1_Days": 1, "Adjunct_Drug_1_Same_Day_As_Platinum": 1,
    "Adjunct_Drug_2": 'No Drug', "Adjunct_Drug_2_Dose_Value": 0, "Adjunct_Drug_2_Dose_Unit": 'No Drug',
    "Adjunct_Drug_2_Dose_Method": 'No Drug', "Adjunct_Drug_2_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_2_Days_Number": 0, "Adjunct_Drug_2_Days": 0, "Adjunct_Drug_2_Same_Day_As_Platinum": 0,
    "Adjunct_Drug_3": 'No Drug', "Adjunct_Drug_3_Dose_Value": 0, "Adjunct_Drug_3_Dose_Unit": 'No Drug',
    "Adjunct_Drug_3_Dose_Method": 'No Drug', "Adjunct_Drug_3_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_3_Days_Number": 0, "Adjunct_Drug_3_Days": 0, "Adjunct_Drug_3_Same_Day_As_Platinum": 0,
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
    "Cycle_Length_Days": 21
}

#=====================================================================================
# OVERALL SURVIVAL POINTS EXTRACTED FROM KAPLAN-MEIER CURVE USING WEBPLOTDIGITIZER
# (unchanged -- your own digitization, no basis to re-extract)
#=====================================================================================
kitagawa_a_OS_raw = {
    2.170492529: 0.995670186, 3.223110127: 0.988309595, 3.749418926: 0.947007975,
    5.842692559: 0.937041693, 6.348068622: 0.911900281, 6.775694521: 0.889436182,
    7.840273683: 0.86583752, 8.486198118: 0.846104066, 9.512500277: 0.823691992,
    10.42696182: 0.793833602, 10.8921812: 0.765346988, 11.44668511: 0.743766244,
    12.26904261: 0.715930868, 12.90134416: 0.689925899, 13.60455119: 0.66503077,
    14.06883074: 0.643505861, 14.40717211: 0.617665533, 15.29897313: 0.59955893,
    16.11767571: 0.586837006, 16.70246326: 0.560451262, 17.21415237: 0.539070841,
    18.0913337: 0.512693831, 19.5386829: 0.481933315, 20.73483926: 0.462352864,
    21.89511093: 0.443888719, 22.65894221: 0.419595276, 23.44213983: 0.40784778,
    23.89269206: 0.384230778, 24.94530965: 0.359307459, 25.87831162: 0.342574668,
    26.83390764: 0.330256882, 27.10437189: 0.308717512, 27.82804649: 0.28318134,
    29.67012728: 0.27495555, 30.85432208: 0.266948563, 31.63062756: 0.248079798,
    32.82798008: 0.248798403, 34.23944458: 0.207497603, 34.96610957: 0.193722803,
    35.84109795: 0.181423646, 37.03845047: 0.17016976, 38.22264527: 0.156244812,
    39.40684007: 0.152486618, 40.59103486: 0.153432443, 41.77522966: 0.147853338,
    42.9474629: 0.136369709, 44.14361926: 0.133053305, 45.32781406: 0.131267764,
    46.51200885: 0.132213589, 47.69620365: 0.131793732, 48.88039845: 0.130615161,
    49.76854455: 0.110346118, 49.93301605: 0.131860542, 50.98563365: 0.105033544,
    52.22245932: 0.103522712, 53.25281001: 0.088152782, 54.53821804: 0.085564861,
    55.72241284: 0.085600231, 56.90660764: 0.085180374, 58.09080243: 0.085064001,
    59.27499723: 0.085706342, 60.45919203: 0.085741712, 61.64338683: 0.085321854,
    62.67407489: 0.086490709
}

kitagawa_b_OS_raw = {
    0.644197012: 0.998006957, 1.67707803: 0.983516042, 2.959955728: 0.970201011,
    4.157308246: 0.960419029, 4.867825125: 0.936540794, 5.459922524: 0.908118124,
    5.975266556: 0.878486808, 7.005954621: 0.862243201, 7.469465211: 0.83776786,
    8.387515219: 0.817216917, 9.144084117: 0.790130701, 9.875068561: 0.76373415,
    10.37762037: 0.738408148, 11.22907667: 0.715532121, 11.89404759: 0.684099246,
    12.31509463: 0.656388451, 12.93350747: 0.62554248, 13.41094474: 0.60427159,
    14.50585501: 0.583754012, 15.19663531: 0.559784141, 16.17030659: 0.543971297,
    16.71708295: 0.517508546, 17.50654615: 0.498260818, 18.46961815: 0.484803461,
    19.02433566: 0.460811938, 19.78987574: 0.438238952, 20.3281461: 0.411136461,
    20.90708578: 0.391429385, 21.83397405: 0.379575625, 22.26233094: 0.35254789,
    23.09126729: 0.339553135, 23.95157975: 0.320179125, 25.19650249: 0.309115755,
    26.38069729: 0.306571501, 27.50850186: 0.30326685, 28.59358656: 0.280206797,
    29.15578005: 0.249185332, 30.32801328: 0.234763414, 31.51220808: 0.233281359,
    32.69640288: 0.233771957, 33.88059768: 0.232289901, 35.06479247: 0.226559053,
    36.24898727: 0.216882897, 37.33919836: 0.2118646, 38.61737687: 0.20420726,
    39.78277492: 0.194573899, 40.32788046: 0.177153331, 41.51207526: 0.179407936,
    42.69627006: 0.178229366, 43.88046486: 0.178871706, 45.15864337: 0.177580081,
    45.91368597: 0.160851084, 46.64358605: 0.146901312, 47.82778085: 0.146820178,
    49.01197565: 0.146855548, 50.27990139: 0.142672217, 51.28168235: 0.131217983,
    52.56456004: 0.128752548, 53.74875484: 0.127877463, 54.93294964: 0.127912833,
    56.11714444: 0.126886005, 57.30133924: 0.12714018, 58.48553403: 0.128018944,
    59.66972883: 0.128054314, 60.85392363: 0.126554639, 62.03811843: 0.128125055,
    63.22231323: 0.128160425, 64.17106002: 0.127733535, 65.13018262: 0.127146402
}

def dict_to_sorted_lists(data):
    sorted_items = sorted(data.items())
    return [x for x, y in sorted_items], [y for x, y in sorted_items]

kitagawa_a_os_time, kitagawa_a_os_surv = dict_to_sorted_lists(kitagawa_a_OS_raw)
kitagawa_b_os_time, kitagawa_b_os_surv = dict_to_sorted_lists(kitagawa_b_OS_raw)

all_times_os = kitagawa_a_os_time + kitagawa_b_os_time
t_min_os, t_max_os = min(all_times_os), max(all_times_os)

survival_time_kitagawa_a = np.linspace(t_min_os, t_max_os, KITAGAWA_A_NUMBER)
survival_time_kitagawa_b = np.linspace(t_min_os, t_max_os, KITAGAWA_B_NUMBER)

def interpolate_curve(x_orig, y_orig, x_new, kind='linear'):
    f = interp1d(x_orig, y_orig, kind=kind, bounds_error=False, fill_value=(y_orig[0], y_orig[-1]))
    return f(x_new)

method = 'linear'

kitagawa_a_survival = interpolate_curve(kitagawa_a_os_time, kitagawa_a_os_surv, survival_time_kitagawa_a, kind=method)
kitagawa_b_survival = interpolate_curve(kitagawa_b_os_time, kitagawa_b_os_surv, survival_time_kitagawa_b, kind=method)

#=============================================================================================
# PROGRESSION-FREE SURVIVAL POINTS EXTRACTED FROM KAPLAN-MEIER CURVE USING WEBPLOTDIGITIZER
# (unchanged -- your own digitization)
#=============================================================================================
kitagawa_a_pfs_raw = {
    1.011388406: 0.999999999, 1.864527233: 0.982649698, 2.322988475: 0.962420854,
    2.907492014: 0.894915613, 3.759009058: 0.858094573, 4.150493397: 0.828319534,
    4.41082276: 0.802931199, 4.693109276: 0.77763378, 4.897982196: 0.747290515,
    5.146964501: 0.718765573, 5.894823313: 0.605734776, 5.253796998: 0.694445503,
    5.210305125: 0.672650192, 6.422519585: 0.557048472, 6.697966167: 0.520397899,
    7.086899584: 0.496839252, 7.451891197: 0.465745929, 7.790449692: 0.43383436,
    7.945179346: 0.406286767, 8.375928623: 0.374511573, 8.606769229: 0.349429727,
    9.175764896: 0.265724329, 9.127356888: 0.326530292, 10.2043981: 0.248501789,
    10.9194133: 0.209635135, 11.44066531: 0.182360291, 13.02504154: 0.160994995,
    14.20960069: 0.156449188, 15.2039578: 0.156319496, 15.95410448: 0.137724414,
    16.92575069: 0.126149263, 18.68193468: 0.116900663, 19.8658797: 0.10720294,
    21.05004148: 0.099323541, 22.23512445: 0.099172014, 23.28831537: 0.09722381,
    24.61163412: 0.075241678, 25.16040576: 0.060987231, 26.31209963: 0.056896004,
    27.49714648: 0.056441423, 28.68230171: 0.056896004, 29.86740275: 0.056896004,
    31.0525038: 0.056896004, 32.23760484: 0.056896004, 33.42270588: 0.056896004,
    34.60780692: 0.056896004, 35.79287314: 0.056603868, 36.97799224: 0.056755395,
    38.16309328: 0.056755395, 39.34807659: 0.05576774, 40.53317763: 0.05576774,
    41.71827868: 0.05576774, 42.90337972: 0.05576774, 44.08848076: 0.05576774,
    45.27347342: 0.054858579, 46.45868284: 0.05576774, 47.64378388: 0.05576774,
    48.82888493: 0.05576774, 50.01387759: 0.054858579, 51.19908701: 0.05576774,
    52.38418805: 0.05576774, 53.56919878: 0.055010106, 54.75439013: 0.05576774,
    55.93949118: 0.05576774, 57.12459222: 0.05576774, 58.50042174: 0.05576774,
    59.82856914: 0.05576774, 61.10899829: 0.055464686, 2.138631347: 0.940740741,
    2.136276674: 0.920987654, 2.372626932: 0.903703704, 3.084621045: 0.87654321,
    3.914937454: 0.841975309, 5.203237675: 0.649382716, 5.200883002: 0.62962963,
    5.341692421: 0.610864198, 5.958263429: 0.583209877, 6.620161884: 0.535802469,
    9.191405445: 0.305679012, 9.68682855: 0.261728395, 10.51773363: 0.232098765,
    11.3477557: 0.195061728, 12.06004415: 0.17037037, 23.73086093: 0.075555556,
    9.260338484: 0.283950617, 17.77630611: 0.12345679
}

kitagawa_b_pfs_raw = {
    0.850721249: 0.983604317, 1.703760989: 0.959556996, 2.074797708: 0.931834064,
    2.316183956: 0.9053385, 2.576444036: 0.879368952, 3.23215524: 0.856901298,
    3.479055354: 0.829319611, 3.770887937: 0.799940707, 3.967769999: 0.768314875,
    4.403708205: 0.743247994, 4.647103296: 0.71387883, 4.775021867: 0.682342291,
    5.064148318: 0.649987506, 5.376615082: 0.620121551, 5.482734098: 0.589816168,
    5.765173245: 0.565799152, 6.011546377: 0.533796667, 6.234553457: 0.499089427,
    6.723259586: 0.476149334, 6.957833573: 0.445107963, 7.228518346: 0.414537408,
    7.4883993: 0.385387417, 7.748422498: 0.357430701, 8.173225949: 0.331008195,
    8.827973954: 0.300460369, 9.36810123: 0.274924295, 9.742482344: 0.23975111,
    10.5288378: 0.208612329, 11.38228124: 0.187951634, 12.1438631: 0.156108252,
    13.3641961: 0.134083815, 14.6009315: 0.12538617, 15.68287999: 0.109767242,
    16.96782809: 0.097656744, 18.15229693: 0.092353302, 19.3370909: 0.089777345,
    20.52152362: 0.084170849, 21.70662466: 0.084170849, 22.8917257: 0.084170849,
    24.07682675: 0.084170849, 25.26192779: 0.084170849, 26.44702883: 0.084170849,
    27.63212987: 0.084170849, 28.81723091: 0.084170849, 30.00233195: 0.084170849,
    31.187433: 0.084170849, 32.37253404: 0.084170849, 33.55693063: 0.078261299,
    34.7415801: 0.074473126, 35.92585025: 0.067502888, 37.11059004: 0.06447235,
    38.29569108: 0.06447235, 39.48086437: 0.065078458, 40.66596541: 0.065078458,
    41.85106646: 0.065078458, 43.0361675: 0.065078458, 44.22126854: 0.065078458,
    45.40636958: 0.065078458, 46.59132612: 0.063866242, 47.77657166: 0.065078458,
    48.96167271: 0.065078458, 50.14677375: 0.065078458, 51.33187479: 0.065078458,
    52.51697583: 0.065078458, 53.70207687: 0.065078458, 54.88717791: 0.065078458,
    56.07217058: 0.064169296, 57.25727162: 0.064169296, 58.44248104: 0.065078458,
    59.62758208: 0.065078458, 60.81241218: 0.062805554, 61.99778416: 0.065078458,
    63.18288521: 0.065078458, 64.36793206: 0.064623877, 65.15783685: 0.063260135
}

kitagawa_a_pfs_time, kitagawa_a_pfs_surv = dict_to_sorted_lists(kitagawa_a_pfs_raw)
kitagawa_b_pfs_time, kitagawa_b_pfs_surv = dict_to_sorted_lists(kitagawa_b_pfs_raw)

all_times_pfs = kitagawa_a_pfs_time + kitagawa_b_pfs_time
t_min_pfs, t_max_pfs = min(all_times_pfs), max(all_times_pfs)

pfs_time_kitagawa_a = np.linspace(t_min_pfs, t_max_pfs, KITAGAWA_A_NUMBER)
pfs_time_kitagawa_b = np.linspace(t_min_pfs, t_max_pfs, KITAGAWA_B_NUMBER)

kitagawa_a_pfs = interpolate_curve(kitagawa_a_pfs_time, kitagawa_a_pfs_surv, pfs_time_kitagawa_a, kind=method)
kitagawa_b_pfs = interpolate_curve(kitagawa_b_pfs_time, kitagawa_b_pfs_surv, pfs_time_kitagawa_b, kind=method)

#==================================================
# ASSIGNING PARAMETERS
#==================================================
def parameter_assign(data, ratio):
    if len(data) != len(ratio):
        raise ValueError(f"Length mismatch: {len(data)} keys vs {len(ratio)} values")
    return dict(zip(data, ratio))

#==========================================================================
# HISTOLOGY (Table 1 -- verified correct, unchanged)
#==========================================================================
HISTOLOGY_kitagawa_a = parameter_assign(histology_base, [106, 18, 3, 0, 0, 0, 0, 0, 0])
HISTOLOGY_kitagawa_b = parameter_assign(histology_base, [105, 17, 4, 0, 0, 0, 0, 0, 0])

#===========================================================
# ETHNICITY (100% Asian -- Japanese trial, Table 1 has no race/ethnicity row
# because it isn't collected/reported; documented inference, unchanged)
#===========================================================
ETHNICITY_kitagawa_a = parameter_assign(ethnicity_base, [0, 0, 127, 0, 0, 0, 0])
ETHNICITY_kitagawa_b = parameter_assign(ethnicity_base, [0, 0, 126, 0, 0, 0, 0])

#================================================================
# DISEASE NATURE (FIX C4: Table 1 "IVB or persistent" is a single combined
# count (27/24) that cannot be split by the data given. Self-proposed split:
# 45% Persistent / 55% Advanced-IVB, flagged in the weight reference table.
# "First recurrence" + "second recurrence" combine into "Recurrent".)
#================================================================
NATURE_kitagawa_a = parameter_assign(disease_nature, [12, 100, 15])   # Persistent, Recurrent(82+18), Advanced(IVB)
NATURE_kitagawa_b = parameter_assign(disease_nature, [11, 102, 13])   # Persistent, Recurrent(85+17), Advanced(IVB)

#======================================================================
# FIGO STAGE (only the Advanced/IVB subgroup is confirmed IVB; Persistent/
# Recurrent patients' initial stage is not reported -- same convention as
# the corrected Monk/Long pipelines)
#======================================================================
STAGE_kitagawa_a = parameter_assign(figo_stage, [0, 0, 0, 0, 15, 112])
STAGE_kitagawa_b = parameter_assign(figo_stage, [0, 0, 0, 0, 13, 113])

#======================================================================
# TUMOR GRADE (not reported in Table 1 -- all Unspecified, unchanged)
#======================================================================
GRADE_kitagawa_a = parameter_assign(grade_base, [0, 0, 0, 127])
GRADE_kitagawa_b = parameter_assign(grade_base, [0, 0, 0, 126])

#===================================================================
# PERFORMANCE STATUS (FIX C3: shift-by-one code, matching Monk/Long.
# code1=ECOG0, code2=ECOG1, code3=ECOG2, code4 unused. Counts unchanged,
# verified correct: TP 98/27/2, TC 96/27/3.)
#===================================================================
PS_kitagawa_a = parameter_assign(performance_status, [98, 27, 2, 0])
PS_kitagawa_b = parameter_assign(performance_status, [96, 27, 3, 0])

#====================================================================
# PRIOR TREATMENT (FIX C1: Table 1 gives Cisplatin 54/60, Carboplatin+Other
# 7/12, No prior platinum 66/54 -- as three MUTUALLY EXCLUSIVE counts, not
# duplicated. "Prior irradiation" (100/108) is a separate, larger marginal
# count; recoded as chemoradiotherapy=61/72 (= all prior-platinum patients,
# assumed concurrent chemoradiotherapy -- documented assumption, see header),
# radiation_only=39/36 (irradiated, no prior platinum), none=27/18.)
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

TREATMENT_kitagawa_a = parameter_assign(prior_treatment_base, _treatment_counts(27, 39, 61))
TREATMENT_kitagawa_b = parameter_assign(prior_treatment_base, _treatment_counts(18, 36, 72))

#================================================================
# PLATINUM-FREE INTERVAL (Table 1, stratification factor -- NEW, not used
# at all in the original script)
#================================================================
PFI_kitagawa_a = {'<6': 20, '6-12': 20, '12+': 21, 'none': 66}    # sums to 127
PFI_kitagawa_b = {'<6': 13, '6-12': 24, '12+': 35, 'none': 54}    # sums to 126

#================================================================
# CAUSE OF DEATH (FIX C2: TC arm sum corrected from 99 to 98, matching
# Fig 2A's actual TC event count)
#================================================================
CAUSE_kitagawa_a = parameter_assign(death_cause_base, [0, 106, 0])
CAUSE_kitagawa_b = parameter_assign(death_cause_base, [1, 96, 1])

#================================================================
# ALIVE/DEAD (Fig 2A -- exact event counts, unchanged from your version,
# verified: 127-106=21, 126-98=28)
#================================================================
ALIVE_kitagawa_a = parameter_assign(dead_alive, [21, 106, 0])
ALIVE_kitagawa_b = parameter_assign(dead_alive, [28, 98, 0])

#================================================================
# RESPONSE (FIX B2: built directly and exactly from the paper's reported
# CR/RR rates over the 102/99 RECIST-evaluable patients -- CR=4/7 (3.9%/7.1%
# of 102/99), Responders(CR+PR)=60/62 (58.8%/62.6% of 102/99), so PR=56/55.
# Non-responders = 42/37; the paper does NOT report an SD-vs-PD breakdown,
# so this is split via a self-proposed 55%/45% SD:PD ratio (flagged in the
# weight reference table). NE = non-measurable-disease (123-102=21,
# 121-99=22) + ineligible-for-efficacy-analysis (127-123=4, 126-121=5).)
#================================================================
CR_COUNT_a, PR_COUNT_a, SD_COUNT_a, PD_COUNT_a, NE_COUNT_a = 4, 56, 23, 19, 25
CR_COUNT_b, PR_COUNT_b, SD_COUNT_b, PD_COUNT_b, NE_COUNT_b = 7, 55, 20, 17, 27
assert CR_COUNT_a + PR_COUNT_a + SD_COUNT_a + PD_COUNT_a + NE_COUNT_a == KITAGAWA_A_NUMBER
assert CR_COUNT_b + PR_COUNT_b + SD_COUNT_b + PD_COUNT_b + NE_COUNT_b == KITAGAWA_B_NUMBER

#===================================================================
# TOXICITY (Table 2, Grade >=3 counts -- verified correct against the
# paper's percentages, unchanged)
#===================================================================
tox_kitagawa_a = [0, 106, 4, 39, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 3, 0, 0, 0, 0, 20, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tox_kitagawa_b = [0, 96, 31, 56, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 9, 6, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0]

TOXICITY_kitagawa_a = parameter_assign(toxicity_base, tox_kitagawa_a)
TOXICITY_kitagawa_b = parameter_assign(toxicity_base, tox_kitagawa_b)

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
    grades, causes, performance_status_dist, prior_treatment, platinum_free_interval,
    cycle_completion_rate, toxicity, patient_number, followup,
    cr_count, pr_count, sd_count, pd_count, ne_count,
    target_os, target_pfs, alive_count, pfs_alive_count,
    target_age, age_type, age_min, age_max, age_sd,
    os_ps_weight, os_orr_weight, os_age_weight, os_hazard_ratio,
    pfs_age_weight, pfs_histology_weight, pfs_response_weight, pfs_toxicity_weight,
    orr_vs_ps, orr_vs_pfi, tolerability_orr_weight, pfsr_vs_ps,
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
        "platinum_free_interval": platinum_free_interval,
        "cycle_completion_rate": cycle_completion_rate, "toxicity": toxicity,
        "patient_number": patient_number, "followup": followup,
        "cr_count": cr_count, "pr_count": pr_count, "sd_count": sd_count,
        "pd_count": pd_count, "ne_count": ne_count,
        "target_os": target_os, "target_pfs": target_pfs,
        "alive_count": alive_count, "pfs_alive_count": pfs_alive_count,
        "target_age": target_age, "age_type": age_type, "age_min": age_min, "age_max": age_max, "age_sd": age_sd,
        "os_ps_weight": os_ps_weight, "os_orr_weight": os_orr_weight, "os_age_weight": os_age_weight,
        "os_hazard_ratio": os_hazard_ratio,
        "pfs_age_weight": pfs_age_weight, "pfs_histology_weight": pfs_histology_weight,
        "pfs_response_weight": pfs_response_weight, "pfs_toxicity_weight": pfs_toxicity_weight,
        "orr_vs_ps": orr_vs_ps, "orr_vs_pfi": orr_vs_pfi,
        "tolerability_orr_weight": tolerability_orr_weight, "pfsr_vs_ps": pfsr_vs_ps,
        "os_time_grid": os_time_grid, "os_surv_grid": os_surv_grid,
        "pfs_time_grid": pfs_time_grid, "pfs_surv_grid": pfs_surv_grid,
        "never_treated_count": never_treated_count,
    })
    return trial

#=====================================================================
# kitagawa_a (TP ARM) TRIAL CONSTANTS
#=====================================================================
trial_kitagawa_a = create_trial_dict(
    trial_id="kitagawa_TP_arm", drug_combination=DRUG_kitagawa_a, ethnicity=ETHNICITY_kitagawa_a,
    histology=HISTOLOGY_kitagawa_a, disease_nature_dist=NATURE_kitagawa_a, disease_stage=STAGE_kitagawa_a,
    grades=GRADE_kitagawa_a, causes=CAUSE_kitagawa_a, performance_status_dist=PS_kitagawa_a,
    prior_treatment=TREATMENT_kitagawa_a, platinum_free_interval=PFI_kitagawa_a,
    cycle_completion_rate=0.709, toxicity=TOXICITY_kitagawa_a,
    patient_number=KITAGAWA_A_NUMBER, followup=65,
    cr_count=CR_COUNT_a, pr_count=PR_COUNT_a, sd_count=SD_COUNT_a, pd_count=PD_COUNT_a, ne_count=NE_COUNT_a,
    target_os=18.3, target_pfs=6.9, alive_count=21, pfs_alive_count=12,
    target_age=53, age_type="median", age_min=29, age_max=74, age_sd=(74 - 29) / 4,
    os_ps_weight=0.4, os_orr_weight=0.3, os_age_weight=0.2, os_hazard_ratio=1.0,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_pfi=-0.35, tolerability_orr_weight=0.20, pfsr_vs_ps=-0.35,
    os_time_grid=survival_time_kitagawa_a, os_surv_grid=kitagawa_a_survival,
    pfs_time_grid=pfs_time_kitagawa_a, pfs_surv_grid=kitagawa_a_pfs,
    never_treated_count=0
)

#=====================================================================
# kitagawa_b (TC ARM) TRIAL CONSTANTS
#=====================================================================
trial_kitagawa_b = create_trial_dict(
    trial_id="kitagawa_TC_arm", drug_combination=DRUG_kitagawa_b, ethnicity=ETHNICITY_kitagawa_b,
    histology=HISTOLOGY_kitagawa_b, disease_nature_dist=NATURE_kitagawa_b, disease_stage=STAGE_kitagawa_b,
    grades=GRADE_kitagawa_b, causes=CAUSE_kitagawa_b, performance_status_dist=PS_kitagawa_b,
    prior_treatment=TREATMENT_kitagawa_b, platinum_free_interval=PFI_kitagawa_b,
    cycle_completion_rate=0.722, toxicity=TOXICITY_kitagawa_b,
    patient_number=KITAGAWA_B_NUMBER, followup=65,
    cr_count=CR_COUNT_b, pr_count=PR_COUNT_b, sd_count=SD_COUNT_b, pd_count=PD_COUNT_b, ne_count=NE_COUNT_b,
    target_os=17.5, target_pfs=6.2, alive_count=28, pfs_alive_count=13,
    target_age=53, age_type="median", age_min=22, age_max=72, age_sd=(72 - 22) / 4,
    os_ps_weight=0.4, os_orr_weight=0.3, os_age_weight=0.2, os_hazard_ratio=0.994,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_pfi=-0.35, tolerability_orr_weight=0.20, pfsr_vs_ps=-0.35,
    os_time_grid=survival_time_kitagawa_b, os_surv_grid=kitagawa_b_survival,
    pfs_time_grid=pfs_time_kitagawa_b, pfs_surv_grid=kitagawa_b_pfs,
    never_treated_count=0
)

TRIALS = {
    "kitagawa_a": trial_kitagawa_a,
    "kitagawa_b": trial_kitagawa_b,
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
# Guyot reconstruction (unchanged from the corrected Monk pipeline)
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

# ================================================================
# Main Data Generation Function
# ================================================================
def generate_clinical_data(trial_config, arm_name):
    np.random.seed(SEED)
    n = trial_config["patient_number"]
    FOLLOWUP = trial_config["followup"]
    CYCLE_COMPLETION_RATE = trial_config.get("cycle_completion_rate", 0.70)
    OS_PS_WEIGHT = trial_config["os_ps_weight"]
    OS_ORR_WEIGHT = trial_config["os_orr_weight"]
    OS_AGE_WEIGHT = trial_config["os_age_weight"]
    PFS_AGE_WEIGHT = trial_config["pfs_age_weight"]
    PFS_RESPONSE_WEIGHT = trial_config["pfs_response_weight"]
    PFS_TOXICITY_WEIGHT = trial_config["pfs_toxicity_weight"]
    ORR_VS_PS = trial_config["orr_vs_ps"]
    ORR_VS_PFI = trial_config["orr_vs_pfi"]
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
    NE_COUNT = trial_config["ne_count"]
    ALIVE_COUNT = trial_config["alive_count"]
    PFS_ALIVE_COUNT = trial_config["pfs_alive_count"]

    arm_id = {'kitagawa_a': 1, 'kitagawa_b': 2}[arm_name]

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
    # 4. Disease nature / FIGO stage (descriptive only -- see header note on
    #    why the hazard modulation is carried by platinum-free interval
    #    instead of disease_nature, to avoid stacking two uncited effects)
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

    # ----------------------------------------------------------------
    # 5. Platinum-free interval (Table 1 stratification factor) -> hazard
    #    modulator for OS/PFS. Self-proposed magnitudes (PFI_OS_RR/PFI_PFS_RR
    #    above) -- see header caveat: Fig 3's PFI-stratified HRs are
    #    treatment-interaction effects, not prognostic HRs, so are NOT
    #    reused here as if they were.
    # ----------------------------------------------------------------
    pfi_dist = expand_distribution(trial_config.get("platinum_free_interval", {}), n)
    pfi_list = [str(p) for p in pfi_dist]
    os_pfi_rr = np.array([PFI_OS_RR[p] for p in pfi_list])
    pfs_pfi_rr = np.array([PFI_PFS_RR[p] for p in pfi_list])

    # ----------------------------------------------------------------
    # 6. Performance status (base assignment from exact trial counts, then
    #    tolerability-driven worsening, then forced back onto the exact
    #    reported marginal counts)
    #    FIX B7: previously this careful adjustment was computed and then
    #    silently overwritten by a second, unconditional random permutation
    #    a few lines later. That second overwrite has been removed.
    # ----------------------------------------------------------------
    target_ps_dist = trial_config["performance_status"]
    ps1_count = target_ps_dist.get(1, 0)   # ECOG 0
    ps2_count = target_ps_dist.get(2, 0)   # ECOG 1
    ps3_count = target_ps_dist.get(3, 0)   # ECOG 2
    PS_discrete = np.array([1] * ps1_count + [2] * ps2_count + [3] * ps3_count)
    if len(PS_discrete) < n:
        PS_discrete = np.concatenate([PS_discrete, np.full(n - len(PS_discrete), 2)])
    PS_discrete = np.random.permutation(PS_discrete[:n])

    # Platinum-free-interval-driven PS worsening (self-proposed)
    for i in range(n):
        if pfi_list[i] == '<6' and np.random.random() < 0.3:
            PS_discrete[i] = min(PS_discrete[i] + 1, 3)

    # Tolerability-driven PS worsening (self-proposed, reused from Monk/Long)
    worsen_idx = np.where(tolerability < 0.35)[0]
    if len(worsen_idx) > 0:
        worsen_sample = np.random.choice(worsen_idx, size=int(0.1 * len(worsen_idx)), replace=False)
        PS_discrete[worsen_sample] = np.clip(PS_discrete[worsen_sample] + 1, 1, 3)

    # Smart correction -- force back onto Table 1's exact marginal counts
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
    # 7. Tumor response: CR / PR / SD / PD / NE
    #    FIX B2 + B6: built directly and EXACTLY from the paper's reported
    #    counts, with WHICH patients land where driven by a transparent
    #    latent-propensity index (same architecture as the corrected
    #    Monk/Long pipelines) instead of the discarded/rebuilt copula.
    # ----------------------------------------------------------------
    ne_idx = np.random.choice(n, NE_COUNT, replace=False) if NE_COUNT > 0 else np.array([], dtype=int)
    evaluable_idx = np.setdiff1d(np.arange(n), ne_idx)

    target_orr_count = CR_COUNT + PR_COUNT
    ps_z = zscore(PS_discrete[evaluable_idx])
    pfi_z = zscore(os_pfi_rr[evaluable_idx])   # higher = worse PFI-implied prognosis
    tol_z = zscore(tolerability[evaluable_idx])

    response_propensity = (
        ORR_VS_PS * ps_z +
        ORR_VS_PFI * pfi_z +
        TOLERABILITY_ORR_WEIGHT * tol_z +
        np.random.standard_normal(len(evaluable_idx))
    )
    orr_rank = np.argsort(response_propensity)[-target_orr_count:]
    orr_idx = evaluable_idx[orr_rank]
    ORR = np.zeros(n, dtype=int)
    ORR[orr_idx] = 1

    cr_n = min(CR_COUNT, len(orr_idx))
    cr_idx = np.random.choice(orr_idx, cr_n, replace=False) if cr_n > 0 else np.array([], dtype=int)
    pr_idx = np.setdiff1d(orr_idx, cr_idx)

    non_responders = np.setdiff1d(evaluable_idx, orr_idx)
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

    PFSR = np.zeros(n, dtype=int)   # disease-control flag: CR/PR/SD=1, PD/NE=0
    PFSR[orr_idx] = 1
    PFSR[sd_idx] = 1

    response = np.empty(n, dtype=object)
    response[cr_idx] = "CR"
    response[pr_idx] = "PR"
    response[sd_idx] = "SD"
    response[pd_idx] = "PD"
    response[ne_idx] = "NE"

    # ----------------------------------------------------------------
    # 8. Histology, tumor grade, ethnicity
    # ----------------------------------------------------------------
    histology = expand_distribution(trial_config["histology"], n)
    grade = expand_distribution(trial_config["grades"], n)
    ethnicity = expand_distribution(trial_config["ethnicity"], n)

    # ================================================================
    # 9. OS / PFS generation
    # ================================================================
    os_time = trial_config["os_time_grid"]
    # NOTE: os_time_grid/pfs_time_grid store the LINSPACE grid, not the raw
    # digitized points -- Guyot reconstruction needs the raw digitized
    # (time, survival) pairs, so pull those directly by arm.
    os_time_raw = kitagawa_a_os_time if arm_name == 'kitagawa_a' else kitagawa_b_os_time
    os_surv_raw = kitagawa_a_os_surv if arm_name == 'kitagawa_a' else kitagawa_b_os_surv
    pfs_time_raw = kitagawa_a_pfs_time if arm_name == 'kitagawa_a' else kitagawa_b_pfs_time
    pfs_surv_raw = kitagawa_a_pfs_surv if arm_name == 'kitagawa_a' else kitagawa_b_pfs_surv

    os_ipd = guyot_reconstruct_ipd(os_time_raw, os_surv_raw, n,
                                   tot_events=n - ALIVE_COUNT, arm_id=arm_id, random_state=SEED + arm_id)

    age_factor = 1 - (age - np.mean(age)) / np.std(age) * OS_AGE_WEIGHT * 2
    age_factor = np.clip(age_factor, 0.4, 1.6)

    # PS -> OS: base HR borrowed from Monk 2009 (NOT Kitagawa-specific -- see
    # PS_HAZARD_RATIO_OS_ASSUMED note above).
    ps_time_multiplier = np.where(PS_discrete >= 2,
                                   hr_to_time_multiplier(PS_HAZARD_RATIO_OS_ASSUMED, OS_PS_WEIGHT),
                                   1.0)

    orr_time_multiplier = np.where(ORR == 1,
                                    hr_to_time_multiplier(ORR_OS_HR_ASSUMED, OS_ORR_WEIGHT),
                                    1.0)

    # Platinum-free interval -> OS: self-proposed (see PFI_OS_RR note above)
    pfi_time_multiplier_os = hr_to_time_multiplier(os_pfi_rr, 1.0)

    os_covariate_factor = age_factor * ps_time_multiplier * orr_time_multiplier * pfi_time_multiplier_os
    os_rank_driver = os_ipd['time'].values * os_covariate_factor
    os_rank_driver = np.clip(os_rank_driver, 0.1, FOLLOWUP)

    # FIX: rank-preserving empirical remap, matching the convention already
    # used in Long, Miller, and Bloss (empirical_quantile_remap). Covariates
    # determine RANK only; the final time at that rank is the raw,
    # unadjusted Guyot value, so curve shape is protected and only the
    # median needs a final correction.
    os_ranks = np.argsort(np.argsort(os_rank_driver))
    os_raw_sorted = np.sort(os_ipd['time'].values)
    os_adjusted = os_raw_sorted[os_ranks]
    os_adjusted = np.clip(os_adjusted, 0.1, FOLLOWUP)

    current_median = np.median(os_adjusted)
    scaling_factor = trial_config["target_os"] / current_median if current_median > 0 else 1.0
    observed_os = np.round(os_adjusted * scaling_factor, 1)
    observed_os = np.clip(observed_os, 0.1, FOLLOWUP)

    event_os = os_ipd['event'].values
    alive = 1 - event_os
    # NOTE on os_hazard_ratio (arm-level TC-vs-TP effect): intentionally NOT
    # reapplied multiplicatively here -- each arm already reconstructs from
    # its OWN digitized KM curve and is rescaled to its OWN reported median,
    # which already encodes that arm's treatment effect. Retained in
    # trial_config for documentation/QC only (see printed cross-check below).

    # ---------------- PFS ----------------
    # FIX B4: exact reported PFS event count (Fig 2B), replacing the flat
    # int(0.92*n) approximation.
    pfs_ipd = guyot_reconstruct_ipd(pfs_time_raw, pfs_surv_raw, n,
                                    tot_events=n - PFS_ALIVE_COUNT, arm_id=arm_id, random_state=SEED + arm_id + 10)
    pfs_raw_time = pfs_ipd['time'].values.copy()
    observed_pfs = np.round(pfs_ipd['time'].values, 1)
    event_pfs = pfs_ipd['event'].values

    pfs_age_factor = 1 - (age - np.mean(age)) / np.std(age) * PFS_AGE_WEIGHT * 2
    observed_pfs = observed_pfs * pfs_age_factor

    # FIX B1: histology now MULTIPLIES (previously divided).
    histology_weights = np.array([trial_config["pfs_histology_weight"][h] for h in histology])
    observed_pfs = observed_pfs * histology_weights

    pfs_response_multiplier = np.where(ORR == 1,
                                        hr_to_time_multiplier(PFS_RESPONSE_HR_ASSUMED, PFS_RESPONSE_WEIGHT),
                                        1.0)
    observed_pfs = observed_pfs * pfs_response_multiplier

    pfi_time_multiplier_pfs = hr_to_time_multiplier(pfs_pfi_rr, 1.0)
    observed_pfs = observed_pfs * pfi_time_multiplier_pfs
    observed_pfs = np.clip(observed_pfs, 0.1, FOLLOWUP)

    # ================================================================
    # 10. Toxicities
    # ================================================================
    tox_counts_raw = trial_config["toxicity"]
    toxicity_names = list(tox_counts_raw.keys())
    tox_counts_arr = np.array([int(round(min(0.98, tox_counts_raw[name] / n) * n)) for name in toxicity_names])

    n_tox = len(toxicity_names)
    tox_corr = np.full((n_tox, n_tox), 0.2)
    np.fill_diagonal(tox_corr, 1.0)

    hematologic = {'leucopenia', 'neutropenia', 'thrombocytopenia', 'anemia'}
    infectious = {'febrile with neutropenia'}
    renal = {'creatinine'}
    neuro = {'peripheral neuropathy'}
    for i, ti in enumerate(toxicity_names):
        for j, tj in enumerate(toxicity_names):
            if i != j:
                ti_l, tj_l = ti.lower(), tj.lower()
                if ti_l in hematologic and tj_l in hematologic:
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
        # Arm-specific toxicity-profile nudges (self-proposed, reflecting the
        # paper's own narrative -- e.g. TC has more thrombocytopenia/sensory
        # neuropathy, TP has more febrile neutropenia -- but the ARM-LEVEL
        # target counts already fully encode this via tox_counts_arr; these
        # nudges only affect WHICH patients within an arm get selected.
        if arm_name == 'kitagawa_b' and tox_lower in ('thrombocytopenia', 'peripheral neuropathy'):
            latent[:, j] += 0.5

    tox_risk = (1 + 0.4 * PS_normalized + 0.35 * PRIOR_RAD + 0.35 * PRIOR_CHEMO +
                0.25 * PRIOR_PLATINUM + 0.15 * (age > 65) + tox_penalty)
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

    # FIX: rank-preserving empirical remap, matching the convention already
    # used in Long, Miller, and Bloss.
    pfs_ranks = np.argsort(np.argsort(observed_pfs))
    pfs_raw_sorted = np.sort(pfs_raw_time)
    observed_pfs = pfs_raw_sorted[pfs_ranks]
    observed_pfs = np.clip(observed_pfs, 0.1, FOLLOWUP)

    # FIX B3: force alignment with target_pfs (median rescale), mirroring OS.
    current_pfs_median = np.median(observed_pfs)
    if current_pfs_median > 0:
        pfs_scaling_factor = trial_config["target_pfs"] / current_pfs_median
        observed_pfs = np.round(observed_pfs * pfs_scaling_factor, 1)
    observed_pfs = np.clip(observed_pfs, 0.1, FOLLOWUP)

    # ================================================================
    # 11. Cause of death
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
        if other_n > 0 and len(remaining) > 0:
            other_idx = remaining[:min(other_n, len(remaining))]
            causes[other_idx] = "Other/Unknown"
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
    # Final DataFrame (column names/order identical to Monk/Long pipelines)
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
df_kitagawa_a = generate_clinical_data(TRIALS["kitagawa_a"], arm_name='kitagawa_a')
df_kitagawa_b = generate_clinical_data(TRIALS["kitagawa_b"], arm_name='kitagawa_b')

df_kitagawa = pd.concat([df_kitagawa_a, df_kitagawa_b], ignore_index=True)\
                .sample(frac=1, random_state=42).reset_index(drop=True)
df_kitagawa.to_excel("kitagawa_km.xlsx")

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
                        "SD": trial_cfg['sd_count'], "PD": trial_cfg['pd_count'], "NE": trial_cfg['ne_count']}
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
        "Patient Number", "Followup", "ORR", "PFSR (disease control rate)", "Tumor Response (CR/PR/SD/PD/NE)",
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

trial_datasets = {"kitagawa_a": df_kitagawa_a, "kitagawa_b": df_kitagawa_b}
evaluation_results = {name: evaluate_trial(name, df) for name, df in trial_datasets.items()}
combined_evaluation_kitagawa = pd.concat(evaluation_results, axis=1)

pd.set_option('display.max_colwidth', 120)
combined_evaluation_kitagawa.to_excel("kitagawa_evaluation_km.xlsx")

# ====================== E2: NUMBER-AT-RISK / LANDMARK QC CROSS-CHECK ======================
# The paper's Fig 2 legend gives a genuine number-at-risk table and 1-/2-/3-
# year landmark rates -- not available for Monk 2009 or Long 2005. Used here
# purely as an additional validation check, not as a generation constraint.
print("\n--- Number-at-risk / landmark-rate QC cross-check (Fig 2) ---")
landmarks_os = {"kitagawa_a": {1: 72.4, 2: 38.8, 3: 18.3}, "kitagawa_b": {1: 67.6, 2: 31.5, 3: 21.3}}
landmarks_pfs = {"kitagawa_a": {1: 17.2, 2: 7.38, 3: 5.53}, "kitagawa_b": {1: 16.5, 2: 8.26, 3: 6.43}}
for name, df in trial_datasets.items():
    print(f"\n{name}:")
    for yr, target_pct in landmarks_os[name].items():
        months = yr * 12
        observed_pct = 100 * (df['os_months'] > months).mean()
        print(f"  OS  >{yr}yr: observed {observed_pct:.1f}%  vs reported {target_pct:.1f}%")
    for yr, target_pct in landmarks_pfs[name].items():
        months = yr * 12
        observed_pct = 100 * (df['pfs_months'] > months).mean()
        print(f"  PFS >{yr}yr: observed {observed_pct:.1f}%  vs reported {target_pct:.1f}%")

print("\n--- Treatment HR QC cross-check (median-ratio approximation) ---")
print(f"OS:  TP/TC median ratio = {TRIALS['kitagawa_a']['target_os']/TRIALS['kitagawa_b']['target_os']:.3f}  vs reported HR (TC v TP) = 0.994")
print(f"PFS: TP/TC median ratio = {TRIALS['kitagawa_a']['target_pfs']/TRIALS['kitagawa_b']['target_pfs']:.3f}  vs reported HR (TC v TP) = 1.041")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SELF-PROPOSED CORRELATION / WEIGHT REFERENCE TABLE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
weight_reference_table = pd.DataFrame([
    {"category": "Demographics", "parameter": "Persistent/Advanced(IVB) split of Table 1's combined count",
     "value": "45% / 55%", "applies_to": "disease_nature construction", "cited_in_kitagawa2015": "NO",
     "note": "Table 1 reports 'IVB or persistent' as ONE combined count (27/24); the paper does not allow this to be split. Self-proposed ratio."},
    {"category": "Demographics", "parameter": "Prior-platinum patients assumed = chemoradiotherapy", "value": "n/a -- assumption",
     "applies_to": "prior_treatment construction (radiation x platinum overlap)", "cited_in_kitagawa2015": "NO",
     "note": "Table 1 reports prior-irradiation (100/108) and prior-platinum (61/72) as separate marginals, not cross-tabulated. Assumed all prior-platinum patients received it as concurrent chemoradiotherapy."},
    {"category": "Response modeling", "parameter": "SD:PD split of non-responders", "value": "55% / 45%",
     "applies_to": "Response category construction", "cited_in_kitagawa2015": "NO",
     "note": "The paper reports CR/PR (RR) rates but no SD-vs-PD breakdown. Self-proposed split, unlike Monk/Long where this was directly reported."},
    {"category": "OS modeling", "parameter": "os_age_weight", "value": 0.2,
     "applies_to": "Age -> OS (AFT exponent)", "cited_in_kitagawa2015": "NO", "note": "-"},
    {"category": "OS modeling", "parameter": "os_ps_weight / PS_HAZARD_RATIO_OS_ASSUMED", "value": "0.4 / 1.799",
     "applies_to": "PS -> OS", "cited_in_kitagawa2015": "NO -- borrowed from Monk et al. 2009",
     "note": "Kitagawa 2015's Fig 3 subgroup HRs are TC-vs-TP treatment-interaction effects, not baseline prognostic HRs -- not the same thing, so not reused directly."},
    {"category": "OS modeling", "parameter": "os_orr_weight / ORR_OS_HR_ASSUMED", "value": "0.3 / 0.70",
     "applies_to": "Response -> OS", "cited_in_kitagawa2015": "NO", "note": "General oncology-literature association assumed."},
    {"category": "OS/PFS modeling", "parameter": "PFI_OS_RR / PFI_PFS_RR (dicts)",
     "value": "<6mo:1.30, 6-12mo:1.00, >=12mo:0.80, none:0.70/0.75",
     "applies_to": "Platinum-free interval -> OS & PFS hazard", "cited_in_kitagawa2015": "NO",
     "note": "Direction (shorter PFI = worse prognosis) is well-established gynecologic-oncology practice; magnitudes are self-proposed. The paper's own PFI-stratified numbers (Fig 3) are treatment-interaction HRs, not prognostic HRs -- see header caveat."},
    {"category": "PFS modeling", "parameter": "pfs_age_weight", "value": 0.15,
     "applies_to": "Age -> PFS", "cited_in_kitagawa2015": "NO", "note": "-"},
    {"category": "PFS modeling", "parameter": "pfs_histology_weight (dict)", "value": "0.50-1.15 by histology",
     "applies_to": "Histology -> PFS multiplier", "cited_in_kitagawa2015": "NO",
     "note": "Reused unchanged from the Monk/Long pipelines for cross-trial consistency."},
    {"category": "PFS modeling", "parameter": "pfs_response_weight / PFS_RESPONSE_HR_ASSUMED", "value": "0.3 / 0.65",
     "applies_to": "Response -> PFS", "cited_in_kitagawa2015": "NO", "note": "-"},
    {"category": "PFS modeling", "parameter": "pfs_toxicity_weight", "value": 0.1,
     "applies_to": "Toxicity burden -> PFS", "cited_in_kitagawa2015": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "orr_vs_ps", "value": -0.30,
     "applies_to": "Response propensity index", "cited_in_kitagawa2015": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "orr_vs_pfi", "value": -0.35,
     "applies_to": "Response propensity index", "cited_in_kitagawa2015": "NO",
     "note": "Uses the self-proposed PFI-implied severity signal (os_pfi_rr) as a covariate; the weight linking it to response propensity is also self-proposed."},
    {"category": "Response modeling", "parameter": "tolerability_orr_weight", "value": 0.20,
     "applies_to": "Response propensity index", "cited_in_kitagawa2015": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "pfsr_vs_ps", "value": -0.35,
     "applies_to": "Stable-vs-progressive disease propensity index", "cited_in_kitagawa2015": "NO", "note": "-"},
    {"category": "Performance status", "parameter": "PS worsening (PFI<6mo)", "value": "p=0.3",
     "applies_to": "Recently-progressed patients' PS", "cited_in_kitagawa2015": "NO", "note": "-"},
    {"category": "Performance status", "parameter": "PS worsening (tolerability)", "value": "threshold 0.35, fraction 0.10",
     "applies_to": "Low-tolerability patients' PS", "cited_in_kitagawa2015": "NO", "note": "Reused from the Monk/Long pipelines."},
    {"category": "Tolerability", "parameter": "tolerability formula constants", "value": "0.82, 0.5, 0.09, 0.18",
     "applies_to": "Tolerability / toxicity-penalty formula", "cited_in_kitagawa2015": "NO", "note": "Reused unchanged from the Monk/Long pipelines."},
    {"category": "Toxicity", "parameter": "tox_risk formula weights", "value": "0.4/0.35/0.35/0.25/0.15",
     "applies_to": "Per-patient toxicity latent-risk multiplier", "cited_in_kitagawa2015": "NO", "note": "Reused unchanged."},
    {"category": "Toxicity", "parameter": "tox_corr matrix", "value": "0.2 baseline / 0.7 hematologic / 0.5 heme-febrile cross",
     "applies_to": "Cross-toxicity correlation structure", "cited_in_kitagawa2015": "NO", "note": "Reused unchanged."},
    {"category": "Toxicity", "parameter": "Arm-specific latent nudges (TC: thrombocytopenia/neuropathy +0.5)",
     "value": "0.5", "applies_to": "Within-arm patient selection for those toxicities", "cited_in_kitagawa2015": "Directionally consistent with the paper's narrative, magnitude self-proposed.",
     "note": "Arm-level target counts are already exact from Table 2 -- this only affects WHICH patients within the arm are selected."},
    {"category": "Demographics", "parameter": "age_sd", "value": "(age_max-age_min)/4",
     "applies_to": "Age distribution spread", "cited_in_kitagawa2015": "NO", "note": "Range/4 heuristic, reused from the Monk/Long pipelines."},
    {"category": "Technical", "parameter": "Platinum infusion duration (TC arm)", "value": "1 hour",
     "applies_to": "platinum_infusion_hours column (TC)", "cited_in_kitagawa2015": "NO",
     "note": "Paper says carboplatin given 'immediately following' paclitaxel but does not state the infusion duration; 1h is a reasonable, uncited default."},
])

print("\n--- Self-proposed weight reference table (for later citation) ---")
print(weight_reference_table.to_string(index=False))
weight_reference_table.to_excel("kitagawa_correlation_weight_reference.xlsx", index=False)

combined_evaluation_kitagawa