#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MILLER 2014 (GOG-0076GG) -- SYNTHETIC IPD GENERATION PIPELINE (CORRECTED)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Source: Miller DS, Blessing JA, Ramondetta LM, et al. Pemetrexed and
# Cisplatin for the Treatment of Advanced, Persistent, or Recurrent
# Carcinoma of the Cervix: A Limited Access Phase II Trial of the
# Gynecologic Oncology Group. J Clin Oncol. 2014;32(25):2744-2749.
#
# Single-arm phase II trial (n=54 evaluable, of 55 enrolled): cisplatin +
# pemetrexed. No comparator arm -- this script produces ONE trial_config and
# ONE dataframe, unlike the multi-arm Monk/Long/Kitagawa pipelines, but uses
# the exact same architecture, helper functions, base lists, column names/
# order, and evaluation-table structure as those scripts for cross-trial
# merge compatibility.
#
# You asked for a Guyot-only reconstruction (no Weibull) using your own
# newly-digitized OS/PFS points -- done: this uses guyot_reconstruct_ipd()
# exclusively, the same function used in the Monk/Long/Kitagawa pipelines.
#
# CHANGE LOG relative to the version you pasted
# ------------------------------------------------------------------------
#  [Bug fixes -- unambiguous, evidence-based]
#  B1. `pfs_observed = pfs_observed / histology_weights` DIVIDED by the
#      histology weight (same sign-flip bug originally found in Monk).
#      Fixed to MULTIPLY.
#  B2. Response was built via a 4-way `correlated_binary` copula (ORR/PFSR/
#      PS/radiation) then forced to target counts via `adjust_binary` --
#      the same wasted-mechanism pattern originally found in Monk, AND it
#      never used the paper's own exact response breakdown at all (Table 3
#      reports precise CR/PR/SD/"Increasing disease"/"Inevaluable" counts:
#      1/16/24/8/5). Replaced with the exact-count + latent-propensity-index
#      approach used in the corrected Monk/Long/Kitagawa pipelines, and
#      added the "NE" (not evaluable) response value used in Long/Kitagawa
#      for the paper's 5 "Inevaluable" patients.
#  B3. No target existed for median PFS in months, and `target_pfsr` was a
#      guessed "~50% at 6 months" landmark rather than the paper's own
#      number. The paper's Figure 2 legend directly reports BOTH medians
#      (OS 12.3 months, PFS 5.7 months) -- target_pfs added, and PFSR is now
#      the disease-control flag built from the exact response counts (same
#      convention as Monk/Long/Kitagawa), not a separately-guessed rate.
#  B4. PFS "events" were computed as `(pfs_times <= FOLLOWUP)`, i.e., every
#      patient was treated as having a PFS event by construction (since
#      times are already capped at FOLLOWUP) -- this is circular and not
#      tied to any real event count. Fixed using the CONSORT diagram's
#      death count for OS (alive_count=18, from "Alive with progression
#      (n=18)"). For PFS specifically, the digitized curve's own tail
#      plateaus at S~=0.077 (not 0) -- i.e., ~4 of 54 patients were
#      genuinely censored progression-free -- which is the more reliable,
#      direct signal here (a plotted curve is a measurement; a first
#      reading of the CONSORT phrase as "zero patients remained
#      progression-free" turned out to contradict it), so pfs_alive_count
#      is set to 4, not 0.
#  B5. `os_ps_weight`, `os_orr_weight`, `os_hazard_ratio`, `age_latent_risk`
#      were extracted from trial_config but NEVER applied to the generated
#      survival times (only age_factor was). PS is now wired into OS the
#      same way as Monk/Long/Kitagawa (self-proposed base HR borrowed from
#      Monk 2009 -- Miller 2014 is a single-arm trial with n=54 and does not
#      report its own prognostic Cox model). `os_hazard_ratio` (1.6 in your
#      version) doesn't have a clear referent in a single-arm trial (there
#      is no comparator arm) -- removed as dead/inapplicable.
#  B6. `guyot_reconstruct_ipd` was called WITHOUT `tot_events`, meaning the
#      event/censoring split came purely from whatever the digitized curve's
#      own drop pattern implied, uncalibrated against any reported count.
#      Fixed by passing `tot_events` explicitly from ALIVE_COUNT / PFS_ALIVE_COUNT
#      (see B4), the same pattern used in Monk/Long/Kitagawa.
#
#  [Data-mapping corrections -- verified line-by-line against Table 2]
#  C1. Several toxicity counts were wrong or shifted to the wrong category
#      versus Table 2's actual Grade 3/4 sums (n=54): Anemia was coded 7,
#      actual is 6+7=13; "Fatigue" was coded 12, but Table 2 has no separate
#      Fatigue row -- only a single "Constitutional" row (12+1=13), now
#      mapped to "Other constitutional"; Nausea/vomiting was coded 6, but
#      Table 2 has SEPARATE Nausea (6) and Vomiting (7) rows that both map
#      onto the shared "Nausea/vomiting" category, i.e. 13 (same combining
#      convention -- and the same double-count caveat -- used in Monk/Long);
#      Allergic reactions (0->2), Inner ear/hearing (0->1), Dermatologic
#      (0->1), Vascular (0->1), Other hematologic (3->3, now correctly
#      composed of "Other hematologic"=2 + "Coagulation"=1, since this
#      trial's Table 2 has no dedicated coagulation category) were all off.
#  C2. `treatment_miller` used "radiation_plus_chemotherapy" for the 27
#      chemoradiation patients, but that category is NOT in
#      derive_prior_treatments()'s trigger list for prior_platinum -- so
#      these patients would have been flagged prior_chemo=1 but
#      prior_platinum=0, even though the paper explicitly describes this as
#      cisplatin-based chemoradiotherapy. Recoded to "chemoradiotherapy"
#      (the category Monk/Long/Kitagawa use for exactly this scenario),
#      which correctly triggers radiation+chemo+platinum all at once. Also,
#      "none" (0 in your version) should be 19 -- the paper states 35/54 had
#      prior RT (8 radiation-only + 27 chemoradiation), meaning the
#      remaining 54-35=19 patients had no prior radiotherapy at all; your
#      version left them unassigned, so expand_distribution silently padded
#      them into whichever category happened to be "most common."
#  C3. `never_treated_count=1` was carried over from the CONSORT diagram's
#      "1 patient never received treatment" -- but that patient is already
#      EXCLUDED from the n=54 evaluable population this script simulates
#      (PATIENT_NUMBER=54 IS the "54 evaluable" cohort). Applying
#      never_treated_count=1 within this n=54 population double-counts a
#      patient who isn't part of it. Set to 0.
#  C4. `performance_status = [0,1,2,3]` used raw GOG codes; the corrected
#      Monk/Long/Kitagawa pipelines use a shift-by-one convention
#      (code = GOG_PS + 1) so the same column values mean the same thing
#      across all merged trials. Recoded to match.
#  C5. `disease_nature` still used "Non-specified" for the third category;
#      relabeled to "Advanced (IVB)" matching the corrected schema. NOTE:
#      unlike Monk/Long/Kitagawa, this paper does NOT report a Table 1
#      breakdown of persistent/recurrent/advanced disease status at all
#      (it's used only as a combined eligibility description, "advanced,
#      persistent, or recurrent") -- so, unlike the other three trials,
#      this split is ENTIRELY self-proposed here, not partially-sourced.
#      Flagged prominently in the weight reference table.
#  C6. FOLLOWUP was 38.6, which appears to be the trial's ~38-month accrual
#      window (Sept 2008-Nov 2011), not a per-patient observation cap --
#      the actual digitized OS curve only extends to ~24.3 months. Changed
#      to 27 (a small buffer beyond the last digitized point, consistent
#      with how Guyot censoring draws times up to 1.1x the last observed
#      time) so per-patient times aren't clipped against an unrelated
#      administrative number.
#
#  [New data the original script didn't use at all]
#  D1. Added `cycle_completion_rate` (missing entirely from your version,
#      present in Monk/Long/Kitagawa) using the paper's own "No. of courses"
#      table: 24/54 patients received >=6 cycles = 0.444.
#  D2. Added two REAL, cited prognostic findings as self-proposed-magnitude
#      (but cited-direction) response-propensity covariates that the
#      original script didn't use: (a) the trial's own stratification
#      finding that prior cisplatin-radiosensitized patients have a lower
#      expected response rate (25% vs. 40% for platinum-naive, the trial's
#      own stated stratum hypotheses) -- modeled via prior_platinum_exposure;
#      (b) "no responses were seen in the nine patients with measurable
#      tumor in irradiated disease sites" (0% vs. 38% nonirradiated) --
#      modeled via prior_radiation as an imperfect proxy (the paper's
#      finding is about the target LESION's location, not general prior-RT
#      history, which isn't separately tracked in this schema).
#
#  [Hygiene / dead code removed]
#  E1. Removed `shape_k`, `age_latent_risk`, `asian_weight`/`black_weight`/
#      `american_indian_weight` (never referenced in generation, same dead
#      parameters found in the original Monk/Kitagawa scripts).
# ============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Importing Libraries
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
from collections import Counter

SEED = 42
PATIENT_NUMBER = 54

#=======================================================
# LITERATURE- AND TRIAL-DERIVED EFFECT SIZES
#=======================================================
# NOT reported in Miller et al. 2014 -- self-proposed placeholders requiring
# an external literature citation. Catalogued in the weight reference table
# at the end of this script.
PS_HAZARD_RATIO_OS_ASSUMED = 1.799   # borrowed from Monk et al. 2009 (same
                                      # disease) -- Miller 2014 is a n=54
                                      # single-arm trial with no reported
                                      # prognostic Cox model of its own.
ORR_OS_HR_ASSUMED = 0.70
PFS_RESPONSE_HR_ASSUMED = 0.65

# Self-proposed magnitudes on top of two REAL, cited directional findings
# (see D2 above): prior platinum exposure and prior radiation both lower
# response propensity. The DIRECTIONS are cited; the MAGNITUDES below are not.
ORR_VS_PRIOR_PLATINUM = -0.40   # cited direction: trial's own stratum hypotheses,
                                # 40% (platinum-naive) vs 25% (prior cis-RT) expected RR
ORR_VS_PRIOR_RADIATION = -0.45  # cited direction: 0/9 responses in irradiated-field
                                # lesions vs 17/45 (38%) in nonirradiated sites


def hr_to_time_multiplier(hazard_ratio, weight=1.0):
    """AFT-style time multiplier from a hazard ratio (exact under Weibull PH;
    a practical approximation otherwise). weight=1 applies HR as given."""
    hazard_ratio = np.asarray(hazard_ratio, dtype=float)
    return hazard_ratio ** (-float(weight))


#=======================================================
# DEFINING BASE LISTS
# (identical to the corrected Monk/Long/Kitagawa pipelines' lists, for
#  cross-trial merge compatibility)
#=======================================================

histology_base = ["Squamous Cell Carcinoma", "Adenocarcinoma", "Adenosquamous", "Mucinous Adenocarcinoma",
                  "Clear Cell", "Endometrioid", "Villoglandular", "Undifferentiated Carcinoma", "Not Specified"]

ethnicity_base = ["White", "Black", "Asian", "American Indian", "Hispanic", "Filipino", "Unspecified"]

# FIX C5: "Non-specified" -> "Advanced (IVB)", matching the corrected schema.
disease_nature = ["Persistent", "Recurrent", "Advanced (IVB)"]

figo_stage = ["I", "II", "III", "IVA", "IVB", "Unknown"]

grade_base = ["Grade 1", "Grade 2", "Grade 3", "Grade Unspecified"]

death_cause_base = ["Treatment", "Disease", "Other/Unknown"]

dead_alive = ["Alive", "Dead", "Unspecified"]

# FIX C4: shift-by-one convention (code = GOG_PS + 1), matching Monk/Long/Kitagawa.
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
# pemetrexed 500 mg/m2 + cisplatin 50 mg/m2 IV every 21 days; standard
# pemetrexed premedication with folic acid, vitamin B12, dexamethasone)
#===========================================================
DRUG_miller = {
    "Platinum_Drug": "cisplatin", "Platinum_Dose_Value": 50, "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based", "Platinum_Infusion_Duration_Hours": 4,
    "Platinum_Days_Number": 1, "Platinum_Day": 1,

    "Adjunct_Drug_1": "pemetrexed", "Adjunct_Drug_1_Dose_Value": 500, "Adjunct_Drug_1_Dose_Unit": "mg/m2",
    "Adjunct_Drug_1_Dose_Method": "BSA-based", "Adjunct_Drug_1_Infusion_Duration_Hours": 0.167,
    "Adjunct_Drug_1_Days_Number": 1, "Adjunct_Drug_1_Days": 1, "Adjunct_Drug_1_Same_Day_As_Platinum": 1,

    "Adjunct_Drug_2": 'No Drug', "Adjunct_Drug_2_Dose_Value": 0, "Adjunct_Drug_2_Dose_Unit": 'No Drug',
    "Adjunct_Drug_2_Dose_Method": 'No Drug', "Adjunct_Drug_2_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_2_Days_Number": 0, "Adjunct_Drug_2_Days": 0, "Adjunct_Drug_2_Same_Day_As_Platinum": 0,

    "Adjunct_Drug_3": 'No Drug', "Adjunct_Drug_3_Dose_Value": 0, "Adjunct_Drug_3_Dose_Unit": 'No Drug',
    "Adjunct_Drug_3_Dose_Method": 'No Drug', "Adjunct_Drug_3_Infusion_Duration_Hours": 0,
    "Adjunct_Drug_3_Days_Number": 0, "Adjunct_Drug_3_Days": 0, "Adjunct_Drug_3_Same_Day_As_Platinum": 0,

    "Supportive_Drug_1": "folic acid", "Supportive_Drug_1_Dose_Value": 475, "Supportive_Drug_1_Dose_Unit": "mcg/day",
    "Supportive_Drug_1_Route": "oral", "Supportive_Drug_1_Start_Day": -7, "Supportive_Drug_1_Duration_Value": 999,
    "Supportive_Drug_1_Duration_Unit": "through therapy", "Supportive_Drug_1_Frequency_Times_Per_Day": 1,
    "Supportive_Drug_1_Frequency_Interval_Hours": 24, "Supportive_Drug_1_Frequency_Description": "daily",
    "Supportive_Drug_1_Dose_Is_Range": 1, "Supportive_Drug_1_Dose_Min": 350, "Supportive_Drug_1_Dose_Max": 600,

    "Supportive_Drug_2": "vitamin B12", "Supportive_Drug_2_Dose_Value": 1000, "Supportive_Drug_2_Dose_Unit": "mcg",
    "Supportive_Drug_2_Route": "intramuscular", "Supportive_Drug_2_Start_Day": -7, "Supportive_Drug_2_Duration_Value": 90,
    "Supportive_Drug_2_Duration_Unit": "days", "Supportive_Drug_2_Frequency_Times_Per_Day": 0,
    "Supportive_Drug_2_Frequency_Interval_Hours": 0, "Supportive_Drug_2_Frequency_Description": "every 3 months",
    "Supportive_Drug_2_Dose_Is_Range": 0, "Supportive_Drug_2_Dose_Min": 1000, "Supportive_Drug_2_Dose_Max": 1000,

    "Supportive_Drug_3": "dexamethasone", "Supportive_Drug_3_Dose_Value": 4, "Supportive_Drug_3_Dose_Unit": "mg",
    "Supportive_Drug_3_Route": "oral", "Supportive_Drug_3_Start_Day": -1, "Supportive_Drug_3_Duration_Value": 3,
    "Supportive_Drug_3_Duration_Unit": "days", "Supportive_Drug_3_Frequency_Times_Per_Day": 2,
    "Supportive_Drug_3_Frequency_Interval_Hours": 12, "Supportive_Drug_3_Frequency_Description": "BID",
    "Supportive_Drug_3_Dose_Is_Range": 0, "Supportive_Drug_3_Dose_Min": 4, "Supportive_Drug_3_Dose_Max": 4,

    "Cycle_Length_Days": 21
}

#=====================================================================================
# OVERALL SURVIVAL POINTS EXTRACTED FROM KAPLAN-MEIER CURVE USING WEBPLOTDIGITIZER
# (unchanged -- your own digitization)
#=====================================================================================
miller_OS_raw = {
    1.280620748: 0.963176466, 1.448200033: 0.945823367, 1.462164973: 0.962571125, 1.69956896: 0.944410905,
    1.936972947: 0.942821886, 1.978867768: 0.926755135, 2.10734522: 0.90863527, 2.134227731: 0.889192484,
    2.160411994: 0.926250684, 2.397815981: 0.889324903, 2.565395266: 0.871568242, 2.579360206: 0.888114221,
    2.816764194: 0.870862012, 3.054168181: 0.869954001, 3.061561384: 0.849176572, 3.305537108: 0.849977758,
    3.319502049: 0.83444068, 3.570870976: 0.832624658, 3.6825905: 0.815372449, 3.738450261: 0.831817537,
    3.933959427: 0.814565328, 4.171363414: 0.813657317, 4.178756618: 0.792879888, 4.436697282: 0.794589085,
    4.68806621: 0.794589085, 4.939435137: 0.794589085, 5.190804065: 0.793781964, 5.218733946: 0.778143997,
    5.470102873: 0.776327975, 5.581822397: 0.759075765, 5.637682158: 0.775520854, 5.833191324: 0.758268644,
    6.084560252: 0.758167754, 6.301427562: 0.736583204, 6.307999298: 0.757360633, 6.559368226: 0.738292402,
    6.810737153: 0.737787951, 6.91588494: 0.716072838, 6.978316438: 0.737384391, 7.173825604: 0.716500137,
    7.425194532: 0.716399247, 7.642061842: 0.694814697, 7.648633579: 0.715592126, 7.900002506: 0.696120334,
    8.028479958: 0.677728066, 8.055362468: 0.655890401, 8.081546731: 0.695615883, 8.273564662: 0.650555836,
    8.514459884: 0.651022453, 8.731327195: 0.628369655, 8.737898931: 0.650215332, 8.989267859: 0.627515056,
    9.240636786: 0.627515056, 9.492005714: 0.627515056, 9.743374641: 0.627515056, 9.994743569: 0.627515056,
    10.2461125: 0.627010606, 10.35126028: 0.604761368, 10.41369178: 0.626607045, 10.60920095: 0.603704989,
    10.79249079: 0.58171725, 10.80471011: 0.602998758, 11.05607904: 0.582114505, 11.30744797: 0.582114505,
    11.5588169: 0.582013615, 11.76448238: 0.561890623, 11.78225594: 0.581206494, 11.83501238: 0.547307415,
    11.8381157: 0.531064107, 12.08948463: 0.531064107, 12.27382185: 0.511925252, 12.29895874: 0.530357876,
    12.30070436: 0.476841976, 12.56429261: 0.478197687, 12.69737027: 0.454779311, 12.74583683: 0.477693236,
    12.95531094: 0.451360917, 13.20667987: 0.451360917, 13.45804879: 0.450957356, 13.57992464: 0.430806849,
    13.65045464: 0.415948487, 13.63680003: 0.395548505, 13.63959302: 0.450452906, 13.90492689: 0.395064233,
    14.15629581: 0.395064233, 14.40766474: 0.395064233, 14.65903367: 0.395064233, 14.9104026: 0.395064233,
    15.14780658: 0.394156222, 15.15519979: 0.371242297, 15.41314045: 0.367823902, 15.66450938: 0.367823902,
    15.91587831: 0.367016781, 15.93723645: 0.343467842, 16.19517711: 0.338767549, 16.44654604: 0.338767549,
    16.69791497: 0.338767549, 16.9492839: 0.338666659, 17.158758: 0.305171141, 17.16615121: 0.281723092,
    17.17272294: 0.337859538, 17.42409187: 0.276215678, 17.44545002: 0.252666739, 17.70339068: 0.247966446,
    17.95475961: 0.247966446, 18.20612854: 0.247966446, 18.45749746: 0.247966446, 18.70886639: 0.247966446,
    18.96023532: 0.247966446, 19.21160425: 0.247966446, 19.46297317: 0.247966446, 19.7143421: 0.247966446,
    19.96571103: 0.247966446, 20.21707996: 0.247966446, 20.46844888: 0.247966446, 20.71981781: 0.247562886,
    20.84550227: 0.230714237, 20.85946721: 0.214168258, 20.90136204: 0.247058435, 21.11083614: 0.212554016,
    21.36220507: 0.212554016, 21.613574: 0.212554016, 21.86494292: 0.212554016, 22.11631185: 0.212554016,
    22.36768078: 0.212554016, 22.61904971: 0.212554016, 22.87041863: 0.212554016, 23.12178756: 0.212554016,
    23.37315649: 0.212554016, 23.62452542: 0.212554016, 23.87589435: 0.212554016, 24.12726327: 0.212554016,
    24.28087762: 0.212554016
}

def dict_to_sorted_lists(data):
    sorted_items = sorted(data.items())
    return [x for x, y in sorted_items], [y for x, y in sorted_items]

def simplify_km_curve(t, s, decimals=2):
    """
    Collapse digitization noise before Guyot reconstruction. WebPlotDigitizer
    output for a hand-read curve often has many closely-spaced points that
    differ only by sub-1% jitter rather than representing genuinely distinct
    events -- guyot_reconstruct_ipd's `d_i = max(1, round(...))` floor then
    forces at least one "event" at EVERY such point, which front-loads
    events into wherever points happen to be densely clustered and
    systematically drags the reconstructed median down (verified: for
    Miller's OS curve, the raw 133-point digitization reconstructs to a
    median of 8.0 months vs. the reported 12.3 -- rounding survival to 2
    decimals collapses it to 71 meaningful steps and recovers a median of
    12.0). Keeps only points where survival actually changes at 2-decimal
    resolution (matching the precision a reader can actually distinguish on
    a printed KM figure), extending each plateau to its last occurring time.
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

miller_os_time, miller_os_surv = dict_to_sorted_lists(miller_OS_raw)
miller_os_time, miller_os_surv = simplify_km_curve(miller_os_time, miller_os_surv, decimals=2)

#=============================================================================================
# PROGRESSION-FREE SURVIVAL POINTS EXTRACTED FROM KAPLAN-MEIER CURVE USING WEBPLOTDIGITIZER
# (unchanged -- your own digitization)
#=============================================================================================
miller_pfs_raw = {
    0.163425514: 0.999698688, 0.331004799: 0.982446478, 0.35893468: 0.998891567, 0.582373727: 0.981639357,
    0.833742654: 0.981034016, 0.917532297: 0.964185367, 1.051944848: 0.949064461, 0.987356999: 0.980731346,
    1.132243256: 0.927953205, 1.37837533: 0.926364186, 1.395443591: 0.908494024, 1.420270152: 0.893043424,
    1.428250118: 0.872029454, 1.585987445: 0.857847187, 1.630675254: 0.834602105, 1.77812175: 0.819616139,
    1.966648446: 0.794816088, 1.978867768: 0.815473339, 2.230236696: 0.795698877, 2.286096458: 0.779859129,
    2.3698861: 0.795497096, 2.460658213: 0.760236001, 2.495570564: 0.779152898, 2.553175943: 0.741697443,
    2.734836543: 0.728183212, 2.754332695: 0.702078785, 3.012273359: 0.702980862, 3.04020324: 0.686434883,
    3.291572168: 0.684820641, 3.319502049: 0.668173772, 3.405619181: 0.652939365, 3.438326211: 0.637881802,
    3.484542254: 0.616371284, 3.4950613: 0.595215807, 3.658151854: 0.590627108, 3.906029546: 0.592304406,
    3.961889308: 0.575657537, 4.04567895: 0.592102626, 4.213258235: 0.574850416, 4.464627163: 0.574144186,
    4.520486925: 0.557497317, 4.604276567: 0.573942405, 4.771855852: 0.556690196, 5.02322478: 0.556690196,
    5.274593707: 0.556286635, 5.414243112: 0.539236206, 5.456137933: 0.555782185, 5.583970849: 0.527843384,
    5.565064468: 0.500938319, 5.745910446: 0.488192114, 6.000770609: 0.482233292, 6.252139537: 0.482233292,
    6.503508464: 0.481527061, 6.552796489: 0.460494439, 6.643157868: 0.481325281, 6.782807273: 0.461349038,
    6.898823701: 0.430267122, 6.911284724: 0.408956802, 6.928902034: 0.389406626, 7.066402986: 0.373621203,
    7.076071021: 0.353749731, 7.090035962: 0.341592472, 7.341404889: 0.344215615, 7.592773817: 0.343812055,
    7.71464966: 0.324652105, 7.786731323: 0.304616246, 7.774318042: 0.343307604, 8.03965191: 0.302447108,
    8.291020838: 0.302447108, 8.542389765: 0.302346218, 8.744881401: 0.281487187, 8.765828812: 0.301539097,
    8.852411443: 0.267579485, 8.849618455: 0.249278018, 8.974481451: 0.232156372, 9.212706905: 0.225467951,
    9.247082143: 0.204591458, 9.385364351: 0.19125703, 9.38028619: 0.174114883, 9.26528261: 0.215560574,
    9.631655118: 0.175325564, 9.883024046: 0.174619333, 9.932312071: 0.152037752, 10.02267345: 0.174417553,
    10.19025273: 0.149598585, 10.34005846: 0.129677373, 10.40321808: 0.109267762, 10.3857619: 0.148993244,
    10.58825354: 0.100641657, 10.83263999: 0.101776671, 11.08400892: 0.101776671, 11.33537785: 0.101776671,
    11.58674678: 0.10096955, 11.71982444: 0.078114972, 11.768291: 0.099960649, 11.97776511: 0.077260373,
    12.22913404: 0.077260373, 12.48050296: 0.077260373, 12.73187189: 0.077260373, 12.98324082: 0.077260373,
    13.23460975: 0.077260373, 13.48597867: 0.077260373, 13.7373476: 0.077260373, 13.98871653: 0.077260373,
    14.24008546: 0.077260373, 14.49145438: 0.077260373, 14.74282331: 0.077260373, 14.99419224: 0.077260373,
    15.24556117: 0.077260373, 15.49693009: 0.077260373, 15.74829902: 0.077260373, 15.99966795: 0.077260373,
    16.25103688: 0.077260373, 16.5024058: 0.077260373, 16.75377473: 0.077260373, 17.00514366: 0.077260373,
    17.25651259: 0.077260373, 17.50788151: 0.077260373, 17.75925044: 0.077260373, 18.01061937: 0.077260373,
    18.2619883: 0.077260373, 18.51335722: 0.076755923, 18.61111181: 0.055468108, 18.68093651: 0.076352362,
    1.418057426: 0.881106438, 1.614595211: 0.848754448, 5.527936146: 0.514234875
}

miller_pfs_time, miller_pfs_surv = dict_to_sorted_lists(miller_pfs_raw)
miller_pfs_time, miller_pfs_surv = simplify_km_curve(miller_pfs_time, miller_pfs_surv, decimals=2)

#==================================================
# ASSIGNING PARAMETERS
#==================================================
def parameter_assign(data, ratio):
    if len(data) != len(ratio):
        raise ValueError(f"Length mismatch: {len(data)} keys vs {len(ratio)} values")
    return dict(zip(data, ratio))

#==========================================================================
# HISTOLOGY (Table 1: Squamous 43, Adenocarcinoma 11 -- verified, unchanged)
#==========================================================================
HISTOLOGY_miller = parameter_assign(histology_base, [43, 11, 0, 0, 0, 0, 0, 0, 0])

#===========================================================
# RACE/ETHNICITY (Table 1: White 29, Black 9, Asian 2, American Indian 1,
# Hispanic 13 -- verified, unchanged)
#===========================================================
ETHNICITY_miller = parameter_assign(ethnicity_base, [29, 9, 2, 1, 13, 0, 0])

#================================================================
# DISEASE NATURE (FIX C5: this paper does NOT report a persistent/
# recurrent/advanced breakdown anywhere -- unlike Monk/Long/Kitagawa, this
# is ENTIRELY self-proposed, not partially-sourced. Flagged in the weight
# reference table. Eligibility explicitly includes all three categories
# ("advanced, persistent, or recurrent"), so some Advanced patients are
# included here, unlike the original script's [10, 44, 0].)
#================================================================
NATURE_miller = parameter_assign(disease_nature, [9, 40, 5])   # Persistent, Recurrent, Advanced(IVB) -- self-proposed

#======================================================================
# FIGO STAGE (only the Advanced/IVB subgroup is confirmed IVB; Persistent/
# Recurrent patients' initial stage is not reported -- same convention as
# the corrected Monk/Long/Kitagawa pipelines)
#======================================================================
STAGE_miller = parameter_assign(figo_stage, [0, 0, 0, 0, 5, 49])

#======================================================================
# TUMOR GRADE (not reported -- all Unspecified, unchanged)
#======================================================================
GRADE_miller = parameter_assign(grade_base, [0, 0, 0, 54])

#===================================================================
# PERFORMANCE STATUS (FIX C4: shift-by-one code. code1=PS0(34), code2=PS1(14),
# code3=PS2(6), code4 unused -- counts unchanged, verified correct)
#===================================================================
PS_miller = parameter_assign(performance_status, [34, 14, 6, 0])

#====================================================================
# PRIOR TREATMENT (FIX C2: none=19 (54-35, previously unassigned),
# radiation_only=8 (35-27), chemoradiotherapy=27 (was mislabeled
# "radiation_plus_chemotherapy", which does not trigger prior_platinum))
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

TREATMENT_miller = parameter_assign(prior_treatment_base, _treatment_counts(19, 8, 27))

#================================================================
# CAUSE OF DEATH (CONSORT diagram, Fig 1: Dead n=36, of which Disease=32,
# Treatment=0, Neither=4 -- verified: 32+0+4=36 exactly, unchanged)
#================================================================
CAUSE_miller = parameter_assign(death_cause_base, [0, 32, 4])

#================================================================
# RESPONSE (FIX B2/B3: Table 3 gives EXACT counts -- CR=1(1.9%), PR=16
# (29.6%), SD=24(44.4%), "Increasing disease"->PD=8(14.8%),
# "Inevaluable"->NE=5(9.3%). Sum=54 exactly. This directly fixes the
# missing target_pfs / guessed target_pfsr bug (B3).)
#================================================================
CR_COUNT, PR_COUNT, SD_COUNT, PD_COUNT, NE_COUNT = 1, 16, 24, 8, 5
assert CR_COUNT + PR_COUNT + SD_COUNT + PD_COUNT + NE_COUNT == PATIENT_NUMBER

#===================================================================
# TOXICITY (FIX C1: Table 2 Grade>=3 counts, verified against the paper's
# own stated percentages -- neutropenia 19/54=35.2%~35%, leukopenia
# 15/54=27.8%~28%, metabolic 15/54=27.8%~28%, all confirmed exact matches)
#===================================================================
tox_miller = [
    15,  # Leucopenia (12+3)
    19,  # Neutropenia (12+7)
    6,   # Thrombocytopenia (3+3)
    13,  # Anemia (6+7) -- FIX: was 7
    3,   # Other hematologic (2 "Other hematologic" + 1 "Coagulation", no dedicated category)
    2,   # Allergic reactions (1+1, "Allergy/immunology") -- FIX: was 0
    1,   # Inner ear/hearing (1+0, "Auditory/ear") -- FIX: was 0
    0,   # Other auditory
    0,   # Thrombosis embolism
    0,   # Cardiac left ventricular function ("Cardiac" row = 0)
    0,   # Other cardiovascular
    0,   # Fatigue (no separate row in Table 2 -- see "Other constitutional" below) -- FIX: was 12
    13,  # Other constitutional (12+1, "Constitutional" row) -- FIX: was 0
    0,   # Alopecia
    1,   # Dermatologic (1+0) -- FIX: was 0
    13,  # Nausea/vomiting (Nausea 6 + Vomiting 7, combined per shared-category convention) -- FIX: was 6
    0,   # Stomatitis
    11,  # Other Gastrointestinal (10+1, "GI" row)
    0,   # Creatinine (no dedicated row; see "Other genitourinary/renal")
    0,   # Hematuria
    1,   # Other genitourinary/renal (1+0, "Genitourinary/renal" row)
    3,   # Hemorrhage (2+1)
    0,   # Hepatic (not reported in Table 2)
    0,   # Febrile with neutropenia (not separately reported; general "Infection" used instead)
    6,   # Infection without neutropenia (6+0, "Infection" row)
    0,   # Other infection/fever
    0,   # Lymphatics (0+0)
    15,  # Metabolic (10+5)
    1,   # Musculoskeletal (1+0)
    2,   # Peripheral neuropathy (2+0, "Neurosensory" row)
    1,   # Other neurological (1+0)
    2,   # Ocular/visual (2+0)
    12,  # Pain (12+0)
    2,   # Pulmonary (2+0)
    1,   # Vascular (1+0) -- FIX: was 0
    0,   # Weight loss
    0    # Sexual (0+0, "Sexual/reproductive" row)
]
TOXICITY_miller = parameter_assign(toxicity_base, tox_miller)

_PFS_HISTOLOGY_WEIGHT = {
    "Squamous Cell Carcinoma": 1.00, "Adenocarcinoma": 0.75, "Adenosquamous": 0.85,
    "Mucinous Adenocarcinoma": 0.65, "Clear Cell": 0.70, "Endometrioid": 0.80,
    "Villoglandular": 1.15, "Undifferentiated Carcinoma": 0.50, "Not Specified": 0.90,
}

#======================================================================
# CREATING FINAL TRIAL DICTIONARY
#======================================================================
def create_trial_dict(
    trial_id, drug_combination, ethnicity, histology, disease_nature_dist, disease_stage,
    grades, causes, performance_status_dist, prior_treatment, cycle_completion_rate,
    toxicity, patient_number, followup,
    cr_count, pr_count, sd_count, pd_count, ne_count,
    target_os, target_pfs, alive_count, pfs_alive_count,
    target_age, age_type, age_min, age_max, age_sd,
    os_ps_weight, os_orr_weight, os_age_weight,
    pfs_age_weight, pfs_histology_weight, pfs_response_weight, pfs_toxicity_weight,
    orr_vs_ps, orr_vs_prior_platinum, orr_vs_prior_radiation, tolerability_orr_weight, pfsr_vs_ps,
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
        "cycle_completion_rate": cycle_completion_rate, "toxicity": toxicity,
        "patient_number": patient_number, "followup": followup,
        "cr_count": cr_count, "pr_count": pr_count, "sd_count": sd_count,
        "pd_count": pd_count, "ne_count": ne_count,
        "target_os": target_os, "target_pfs": target_pfs,
        "alive_count": alive_count, "pfs_alive_count": pfs_alive_count,
        "target_age": target_age, "age_type": age_type, "age_min": age_min, "age_max": age_max, "age_sd": age_sd,
        "os_ps_weight": os_ps_weight, "os_orr_weight": os_orr_weight, "os_age_weight": os_age_weight,
        "pfs_age_weight": pfs_age_weight, "pfs_histology_weight": pfs_histology_weight,
        "pfs_response_weight": pfs_response_weight, "pfs_toxicity_weight": pfs_toxicity_weight,
        "orr_vs_ps": orr_vs_ps, "orr_vs_prior_platinum": orr_vs_prior_platinum,
        "orr_vs_prior_radiation": orr_vs_prior_radiation,
        "tolerability_orr_weight": tolerability_orr_weight, "pfsr_vs_ps": pfsr_vs_ps,
        "os_time_grid": os_time_grid, "os_surv_grid": os_surv_grid,
        "pfs_time_grid": pfs_time_grid, "pfs_surv_grid": pfs_surv_grid,
        "never_treated_count": never_treated_count,
    })
    return trial

#=====================================================================
# MILLER TRIAL CONSTANTS
#=====================================================================
trial_miller = create_trial_dict(
    trial_id="miller", drug_combination=DRUG_miller, ethnicity=ETHNICITY_miller,
    histology=HISTOLOGY_miller, disease_nature_dist=NATURE_miller, disease_stage=STAGE_miller,
    grades=GRADE_miller, causes=CAUSE_miller, performance_status_dist=PS_miller,
    prior_treatment=TREATMENT_miller,
    cycle_completion_rate=0.444,   # FIX D1: 24/54 patients received >=6 cycles ("No. of courses" table)
    toxicity=TOXICITY_miller, patient_number=PATIENT_NUMBER,
    followup=27,   # FIX C6: was 38.6 (accrual-period length, not a per-patient cap)
    cr_count=CR_COUNT, pr_count=PR_COUNT, sd_count=SD_COUNT, pd_count=PD_COUNT, ne_count=NE_COUNT,
    target_os=12.3, target_pfs=5.7,   # FIX B3: both directly from Fig 2 legend
    alive_count=18, pfs_alive_count=4,   # OS alive_count from CONSORT "Alive with progression (n=18)".
    # PFS_alive_count: the digitized PFS curve's own tail plateaus at
    # S~=0.077 (~4/54 patients), NOT 0 -- this directly contradicts an
    # initial reading of "Alive with progression (n=18)" as implying zero
    # patients remained progression-free. The curve's own empirical plateau
    # is the more reliable signal here (a plotted curve is a direct
    # measurement; the CONSORT phrase may have been describing a subset,
    # not literally all 18 alive patients), so pfs_alive_count is set to 4,
    # not 0.
    target_age=46, age_type="median", age_min=22, age_max=72, age_sd=(72 - 22) / 4,
    os_ps_weight=0.4, os_orr_weight=0.3, os_age_weight=0.2,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_prior_platinum=ORR_VS_PRIOR_PLATINUM, orr_vs_prior_radiation=ORR_VS_PRIOR_RADIATION,
    tolerability_orr_weight=0.20, pfsr_vs_ps=-0.35,
    os_time_grid=miller_os_time, os_surv_grid=miller_os_surv,
    pfs_time_grid=miller_pfs_time, pfs_surv_grid=miller_pfs_surv,
    never_treated_count=0   # FIX C3: was 1 (that patient is already excluded from n=54)
)

TRIALS = {
    "miller": trial_miller
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
# Guyot reconstruction (unchanged from the corrected Monk/Long/Kitagawa
# pipelines -- this script uses Guyot exclusively, no Weibull, per your request)
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
    Long 2005 pipeline. Covariate multipliers (age, PS, response) change
    each patient's time by a different factor, which -- left alone --
    shifts the marginal shape away from the real curve; a single
    median-rescale only fixes the median, leaving the rest of the curve
    (particularly the tail) visibly off. This keeps each patient's
    RELATIVE RANK (so covariate effects still matter) but replaces the
    MAGNITUDE at that rank with the corresponding order statistic from the
    real Guyot draw, so the marginal distribution matches the digitized
    curve as closely as possible, not just at the median.
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
def generate_clinical_data(trial_config):
    """Generate clinical data for the (single-arm) trial."""
    np.random.seed(SEED)
    n = trial_config["patient_number"]
    FOLLOWUP = trial_config["followup"]
    CYCLE_COMPLETION_RATE = trial_config.get("cycle_completion_rate", 0.50)
    OS_PS_WEIGHT = trial_config["os_ps_weight"]
    OS_ORR_WEIGHT = trial_config["os_orr_weight"]
    OS_AGE_WEIGHT = trial_config["os_age_weight"]
    PFS_AGE_WEIGHT = trial_config["pfs_age_weight"]
    PFS_RESPONSE_WEIGHT = trial_config["pfs_response_weight"]
    PFS_TOXICITY_WEIGHT = trial_config["pfs_toxicity_weight"]
    ORR_VS_PS = trial_config["orr_vs_ps"]
    ORR_VS_PRIOR_PLATINUM = trial_config["orr_vs_prior_platinum"]
    ORR_VS_PRIOR_RADIATION = trial_config["orr_vs_prior_radiation"]
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

    arm_id = 1   # single-arm trial

    # ----------------------------------------------------------------
    # 1. Age
    # ----------------------------------------------------------------
    age = generate_age(n, TARGET_AGE, AGE_TYPE, AGE_MIN, AGE_MAX, AGE_SD)

    # ----------------------------------------------------------------
    # 2. Cycle completion & tolerability (self-proposed constants -- see
    #    weight reference table; same architecture as Monk/Long/Kitagawa)
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
    # 4. Disease nature / FIGO stage (descriptive only -- see header C5
    #    note: this paper does not report a status breakdown, so there is
    #    no citable hazard structure to hang on it, unlike Long/Kitagawa's
    #    PFI-based mechanisms)
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
    # 5. Performance status (base assignment from exact trial counts, then
    #    tolerability-driven worsening, then forced back onto the exact
    #    reported marginal counts -- same architecture as Monk/Long/Kitagawa)
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
    # 6. Tumor response: CR / PR / SD / PD / NE
    #    FIX B2: built directly and EXACTLY from Table 3's reported counts,
    #    with WHICH patients land where driven by a transparent
    #    latent-propensity index using two REAL, cited directional findings
    #    (prior platinum exposure and prior radiation both lower response
    #    propensity -- see ORR_VS_PRIOR_PLATINUM/RADIATION note above).
    # ----------------------------------------------------------------
    ne_idx = np.random.choice(n, NE_COUNT, replace=False) if NE_COUNT > 0 else np.array([], dtype=int)
    evaluable_idx = np.setdiff1d(np.arange(n), ne_idx)

    target_orr_count = CR_COUNT + PR_COUNT
    ps_z = zscore(PS_discrete[evaluable_idx])
    platinum_z = zscore(PRIOR_PLATINUM[evaluable_idx].astype(float))
    radiation_z = zscore(PRIOR_RAD[evaluable_idx].astype(float))
    tol_z = zscore(tolerability[evaluable_idx])

    response_propensity = (
        ORR_VS_PS * ps_z +
        ORR_VS_PRIOR_PLATINUM * platinum_z +
        ORR_VS_PRIOR_RADIATION * radiation_z +
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
    # 7. Histology, tumor grade, race/ethnicity
    # ----------------------------------------------------------------
    histology = expand_distribution(trial_config["histology"], n)
    grade = expand_distribution(trial_config["grades"], n)
    ethnicity = expand_distribution(trial_config["ethnicity"], n)

    # ================================================================
    # 8. OS / PFS generation (Guyot reconstruction from digitized KM curves
    #    ONLY -- no Weibull, per your request)
    # ================================================================
    os_ipd = guyot_reconstruct_ipd(
        trial_config["os_time_grid"], trial_config["os_surv_grid"], n,
        tot_events=n - ALIVE_COUNT, arm_id=arm_id, random_state=SEED + arm_id
    )

    age_factor = 1 - (age - np.mean(age)) / np.std(age) * OS_AGE_WEIGHT * 2
    age_factor = np.clip(age_factor, 0.4, 1.6)

    # PS -> OS: base HR borrowed from Monk 2009 (NOT Miller-specific -- see
    # PS_HAZARD_RATIO_OS_ASSUMED note above).
    ps_time_multiplier = np.where(PS_discrete >= 2,
                                   hr_to_time_multiplier(PS_HAZARD_RATIO_OS_ASSUMED, OS_PS_WEIGHT),
                                   1.0)

    orr_time_multiplier = np.where(ORR == 1,
                                    hr_to_time_multiplier(ORR_OS_HR_ASSUMED, OS_ORR_WEIGHT),
                                    1.0)

    os_covariate_factor = age_factor * ps_time_multiplier * orr_time_multiplier
    os_adjusted = os_ipd['time'].values * os_covariate_factor
    os_adjusted = np.clip(os_adjusted, 0.1, FOLLOWUP)

    # Empirical rank-remap onto the real Guyot draw itself (same fix
    # developed for the Long 2005 pipeline) -- keeps the marginal
    # distribution tracking the real digitized curve, not just its median.
    observed_os = empirical_quantile_remap(os_adjusted, os_ipd['time'].values, FOLLOWUP)
    current_median = np.median(observed_os)
    if current_median > 0:
        # FULL correction (not damped) to force the EXACT reported median,
        # same convention as the Monk/Kitagawa pipelines -- for OS the raw
        # remapped median is already close to target (~12.0 vs 12.3), so
        # this costs essentially no shape fidelity.
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

    # FIX B1: histology now MULTIPLIES (previously divided).
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
    infectious = {'infection without neutropenia'}
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
        if tox.lower() in hematologic:
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

    toxicity_events = [[toxicity_names[j] for j in range(n_tox) if tox_matrix[i, j] == 1] for i in range(n)]
    toxicity_count = [len(x) for x in toxicity_events]

    toxicity_burden = np.array(toxicity_count) / n_tox
    pfs_toxicity_penalty = 1 - toxicity_burden * PFS_TOXICITY_WEIGHT
    observed_pfs = observed_pfs * pfs_toxicity_penalty
    observed_pfs = np.clip(observed_pfs, 0.1, FOLLOWUP)

    # Empirical rank-remap for PFS -- same reasoning as OS. NOTE: for PFS
    # specifically, the raw remapped median (~3.9) sits further from the
    # target (5.7) than OS's did, so forcing the exact median here (rather
    # than a damped partial correction) does cost some shape fidelity in
    # the t~9-12.5 month region (generated survival runs ~0.10-0.15 above
    # the digitized curve there) -- the early (<9mo) and late (>13mo,
    # including the tail plateau) portions remain accurate. This trade-off
    # is deliberate: exact-median matching is the convention used in every
    # other trial in this series (Monk, Long, Kitagawa), and is what you
    # validated against directly.
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
    # Final DataFrame (column names/order identical to the Monk/Long/
    # Kitagawa pipelines)
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
# Usage - Creating the dataset (single arm)
# ================================================================
df_miller = generate_clinical_data(TRIALS["miller"])
df_miller.to_excel("miller_km.xlsx")

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

trial_datasets = {"miller": df_miller}
evaluation_results = {name: evaluate_trial(name, df) for name, df in trial_datasets.items()}
combined_evaluation_miller = pd.concat(evaluation_results, axis=1)

pd.set_option('display.max_colwidth', 120)
combined_evaluation_miller.to_excel("miller_evaluation_km.xlsx")

print("\n--- Median-ratio sanity check (single-arm trial, no comparator) ---")
print(f"Target OS median: {TRIALS['miller']['target_os']}, Target PFS median: {TRIALS['miller']['target_pfs']}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SELF-PROPOSED CORRELATION / WEIGHT REFERENCE TABLE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
weight_reference_table = pd.DataFrame([
    {"category": "Demographics", "parameter": "disease_nature split (Persistent/Recurrent/Advanced)",
     "value": "9 / 40 / 5", "applies_to": "disease_nature construction", "cited_in_miller2014": "NO",
     "note": "Unlike Monk/Long/Kitagawa, this paper reports NO persistent/recurrent/advanced breakdown at all -- eligibility just says 'advanced, persistent, or recurrent' collectively. This split is entirely self-proposed, not partially-sourced."},
    {"category": "Demographics", "parameter": "age_min/age_max (22/72)", "value": "22 / 72",
     "applies_to": "Age distribution bounds", "cited_in_miller2014": "PARTIALLY",
     "note": "Table 1 gives only age BRACKETS (<30:3, 30-39:10, 40-49:23, 50-59:15, >60:3), not an exact min/max -- these are point estimates consistent with the reported brackets, not exact citations."},
    {"category": "Tolerability", "parameter": "cycle_completion_rate", "value": 0.444,
     "applies_to": "Fraction assumed to complete >=6 cycles", "cited_in_miller2014": "YES",
     "note": "Directly computed from the paper's own 'No. of courses' table: 24/54 patients received >=6 cycles."},
    {"category": "OS modeling", "parameter": "os_age_weight", "value": 0.2,
     "applies_to": "Age -> OS (AFT exponent)", "cited_in_miller2014": "NO", "note": "-"},
    {"category": "OS modeling", "parameter": "os_ps_weight / PS_HAZARD_RATIO_OS_ASSUMED", "value": "0.4 / 1.799",
     "applies_to": "PS -> OS", "cited_in_miller2014": "NO -- borrowed from Monk et al. 2009",
     "note": "Miller 2014 is a single-arm n=54 trial with no reported prognostic Cox model of its own."},
    {"category": "OS modeling", "parameter": "os_orr_weight / ORR_OS_HR_ASSUMED", "value": "0.3 / 0.70",
     "applies_to": "Response -> OS", "cited_in_miller2014": "NO", "note": "General oncology-literature association assumed."},
    {"category": "PFS modeling", "parameter": "pfs_age_weight", "value": 0.15,
     "applies_to": "Age -> PFS", "cited_in_miller2014": "NO", "note": "-"},
    {"category": "PFS modeling", "parameter": "pfs_histology_weight (dict)", "value": "0.50-1.15 by histology",
     "applies_to": "Histology -> PFS multiplier", "cited_in_miller2014": "NO",
     "note": "Reused unchanged from the Monk/Long/Kitagawa pipelines for cross-trial consistency."},
    {"category": "PFS modeling", "parameter": "pfs_response_weight / PFS_RESPONSE_HR_ASSUMED", "value": "0.3 / 0.65",
     "applies_to": "Response -> PFS", "cited_in_miller2014": "NO", "note": "-"},
    {"category": "PFS modeling", "parameter": "pfs_toxicity_weight", "value": 0.1,
     "applies_to": "Toxicity burden -> PFS", "cited_in_miller2014": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "orr_vs_ps", "value": -0.30,
     "applies_to": "Response propensity index", "cited_in_miller2014": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "orr_vs_prior_platinum", "value": -0.40,
     "applies_to": "Response propensity index", "cited_in_miller2014": "DIRECTION YES, MAGNITUDE NO",
     "note": "The trial's own stratified design assumed 40% RR (platinum-naive) vs 25% RR (prior cis-RT) -- direction is directly cited; -0.40 magnitude is self-proposed."},
    {"category": "Response modeling", "parameter": "orr_vs_prior_radiation", "value": -0.45,
     "applies_to": "Response propensity index", "cited_in_miller2014": "DIRECTION YES, MAGNITUDE NO",
     "note": "0/9 responses in irradiated-field lesions vs 17/45 (38%) nonirradiated -- direction directly cited; magnitude self-proposed, and prior_radiation is an imperfect proxy for 'target lesion location', which isn't separately tracked."},
    {"category": "Response modeling", "parameter": "tolerability_orr_weight", "value": 0.20,
     "applies_to": "Response propensity index", "cited_in_miller2014": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "pfsr_vs_ps", "value": -0.35,
     "applies_to": "Stable-vs-progressive disease propensity index", "cited_in_miller2014": "NO", "note": "-"},
    {"category": "Performance status", "parameter": "PS worsening (tolerability)", "value": "threshold 0.35, fraction 0.10",
     "applies_to": "Low-tolerability patients' PS", "cited_in_miller2014": "NO", "note": "Reused from the Monk/Long/Kitagawa pipelines."},
    {"category": "Tolerability", "parameter": "tolerability formula constants", "value": "0.82, 0.5, 0.09, 0.18",
     "applies_to": "Tolerability / toxicity-penalty formula", "cited_in_miller2014": "NO", "note": "Reused unchanged."},
    {"category": "Toxicity", "parameter": "tox_risk formula weights", "value": "0.4/0.35/0.35/0.25/0.15",
     "applies_to": "Per-patient toxicity latent-risk multiplier", "cited_in_miller2014": "NO", "note": "Reused unchanged."},
    {"category": "Toxicity", "parameter": "tox_corr matrix", "value": "0.2 baseline / 0.7 within-cluster / 0.5 heme-infection cross",
     "applies_to": "Cross-toxicity correlation structure", "cited_in_miller2014": "NO", "note": "Reused unchanged."},
    {"category": "Demographics", "parameter": "age_sd", "value": "(age_max-age_min)/4",
     "applies_to": "Age distribution spread", "cited_in_miller2014": "NO", "note": "Range/4 heuristic, reused from the other pipelines."},
    {"category": "Technical", "parameter": "FOLLOWUP", "value": 27,
     "applies_to": "Per-patient time cap", "cited_in_miller2014": "NO",
     "note": "Corrected from 38.6 (the ~38-month accrual window, not a per-patient cap) to a small buffer beyond the last digitized OS point (~24.3 months)."},
])

print("\n--- Self-proposed weight reference table (for later citation) ---")
print(weight_reference_table.to_string(index=False))
weight_reference_table.to_excel("miller_correlation_weight_reference.xlsx", index=False)

combined_evaluation_miller
