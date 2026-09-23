#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LONG 2005 -- SYNTHETIC IPD GENERATION PIPELINE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Source: Long HJ III, Bundy BN, Grendys EC Jr, et al. Randomized Phase III
# Trial of Cisplatin With or Without Topotecan in Carcinoma of the Uterine
# Cervix: A Gynecologic Oncology Group Study. J Clin Oncol. 2005;23(21):
# 4626-4633.
#
# Two evaluable arms are modeled: CPT (cisplatin alone, n=146) and CT
# (cisplatin + topotecan, n=147). The MVAC arm (n=63, later dropped from the
# trial after 4 treatment-related deaths) is excluded exactly as the paper
# itself excludes it -- there is no Table 2-5 data for it in this paper.
#
# ============================================================================
# IMPORTANT METHODOLOGICAL NOTE -- READ BEFORE USING THIS FOR ANYTHING FORMAL
# ============================================================================
# Unlike the Monk 2009 pipeline, this script does NOT reconstruct OS/PFS from
# a digitized Kaplan-Meier curve (Guyot et al. 2012). No digitized
# WebPlotDigitizer coordinates for this paper's Figures 1 (PFS) and 2 (OS)
# were available to me. I looked at the rendered page image and judged that
# manually eyeballing coordinates off a small, uncalibrated static image --
# without the axis-reference-point calibration a real digitization tool
# provides -- would introduce silent, unquantifiable error. That would
# amount to fabricating false precision in a pipeline whose entire point is
# statistical fidelity, so I did not do it.
#
# Instead, OS and PFS are reconstructed PARAMETRICALLY: a two-parameter
# Weibull distribution is least-squares-fit to each arm's reported 25th /
# 50th (median) / 75th percentile survival times (Table 5), then sampled and
# adjusted using the same covariate/AFT-multiplier architecture as the Monk
# pipeline. This is a recognized, standard alternative to full-curve
# reconstruction when only summary quantiles are published (rather than a
# digitizable curve) -- but it is lower-fidelity than Guyot reconstruction:
# it reproduces the reported median exactly (by construction, via a final
# rescale) and approximates the reported IQR reasonably well, but it cannot
# reproduce the curve's exact shape or true censoring pattern the way a
# digitized-point reconstruction can.
#
# If you obtain (or create) WebPlotDigitizer coordinates for this paper's
# Figures 1 and 2, replace `weibull_reconstruct_ipd()` below with
# `guyot_reconstruct_ipd()` from the Monk script -- everything downstream
# (covariate adjustments, toxicity, response, etc.) does not need to change.
#
# A second, related gap: the paper does not report exact final OS/PFS event
# (death/progression) counts anywhere I could find -- only the INTERIM
# analysis's 56 CPT deaths, and the trial's DESIGN target of 111 CPT deaths
# to trigger final analysis. Event/censoring rates used here (OS_EVENT_RATE,
# PFS_EVENT_RATE below) are therefore self-proposed assumptions, catalogued
# in the weight reference table at the end of this script -- NOT read off
# the paper the way Monk's exact Fig-2-caption alive/dead counts were.
#
# Everything else below -- demographics (Table 2), toxicity (Table 3, raw
# patient counts, no external file needed this time), response (Table 4),
# and the disease-status / time-to-study-entry Cox model (Table 1) -- IS
# taken directly and exactly from the paper.
# ============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Importing Libraries
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import Counter

SEED = 42
LONG_CPT_NUMBER = 146
LONG_CT_NUMBER = 147

#=====================================================================================
# OVERALL SURVIVAL POINTS EXTRACTED FROM KAPLAN-MEIER CURVE USING WEBPLOTDIGITIZER
# long_a = CPT arm, long_b = CT arm (verified: long_a crosses S=0.5 near t=6.9,
# long_b near t=9.5, matching the reported medians of 6.5/9.4 months)
#=====================================================================================
long_a_OS_raw = {
    0.424832953: 0.999111708, 0.739068404: 0.988547212, 0.98320449: 0.978822502, 1.298407875: 0.973678437, 1.515149977: 0.967905907,
    1.62636683: 0.952544186, 1.819181472: 0.939538885, 2.133682491: 0.930461573, 2.291158979: 0.916260723, 2.460389404: 0.902315881,
    2.623111494: 0.892903142, 2.878368952: 0.885746183, 2.961466225: 0.867151113, 3.049874212: 0.84863521, 3.20137082: 0.831544516,
    3.383636409: 0.82097171, 3.558825163: 0.813854877, 3.689924906: 0.794004002, 3.910283109: 0.77629467, 4.161004277: 0.76023244,
    4.384422403: 0.749227719, 4.416908341: 0.734450862, 4.620952054: 0.721494238, 4.767061136: 0.708655567, 4.921174601: 0.692169409,
    4.983382385: 0.675236507, 5.106428342: 0.661800601, 5.219111569: 0.646532873, 5.347115311: 0.632760835, 5.432403115: 0.618627253,
    5.614309712: 0.604639106, 5.89819446: 0.593489603, 6.04419185: 0.580571843, 6.208985156: 0.563634482, 6.392233883: 0.551698422,
    6.484048997: 0.536651474, 6.622828163: 0.520777962, 6.796206514: 0.506020631, 7.114044381: 0.495959854, 7.318575759: 0.479028539,
    7.618563176: 0.469142464, 7.773987916: 0.454379502, 8.089221075: 0.449402172, 8.404622955: 0.445369677, 8.71942935: 0.438002472,
    8.998980479: 0.429903891, 9.197492572: 0.41758239, 9.487050885: 0.409324577, 9.71872467: 0.39537702, 9.886756354: 0.376706077,
    10.21835706: 0.373978789, 10.44743166: 0.364208926, 10.60112191: 0.353782722, 10.88081701: 0.346490335, 11.08005546: 0.338236448,
    11.36964431: 0.330149645, 11.65403522: 0.316370808, 11.88183072: 0.304121326, 12.10037027: 0.291554652, 12.41623022: 0.290087319,
    12.73201381: 0.288192458, 13.04676219: 0.280500333, 13.25225911: 0.272949766, 13.52875787: 0.265242202, 13.83350056: 0.251586854,
    14.14743636: 0.24828511, 14.30290041: 0.233742324, 14.57090483: 0.226548166, 14.7208202: 0.213715741, 14.97292595: 0.199446649,
    15.21775142: 0.182234491, 15.48673162: 0.175461243, 15.80280533: 0.175190985, 16.11897065: 0.175433759, 16.43516651: 0.175847544,
    16.68849568: 0.16842957, 16.83531797: 0.154667403, 17.02655534: 0.143757068, 17.34265958: 0.143657821, 17.63958314: 0.134495766,
    17.97283732: 0.13208711, 18.2889263: 0.131902357, 18.60504581: 0.131888615, 18.92110425: 0.131532852, 19.23714742: 0.131091583,
    19.5233295: 0.124363031, 19.74282092: 0.110740538, 20.07601021: 0.107968484, 20.39212972: 0.107954742, 20.70824924: 0.107941,
    21.02867315: 0.101770636, 21.28124677: 0.095795609, 21.58416029: 0.095590053, 21.90044776: 0.09651687, 22.21681157: 0.097871214,
    22.53293109: 0.097857472, 22.85924437: 0.095919401, 23.11211874: 0.086848654, 23.44396653: 0.085504998, 23.76008604: 0.085491256,
    24.07620555: 0.085477514, 24.39232507: 0.085463772, 24.70844458: 0.08545003, 25.02451829: 0.085179772, 25.28695329: 0.079579933,
    25.40827561: 0.070541556, 25.72457835: 0.071553879, 26.04072841: 0.071711147, 26.35683265: 0.0716119, 26.6729369: 0.071512653,
    26.98905641: 0.071498911, 27.30517592: 0.071485169, 27.57653108: 0.066663658, 27.79442857: 0.057526415, 28.11056335: 0.057598178,
    28.45232761: 0.05367142, 28.72640963: 0.044450504, 29.05644835: 0.04370508, 29.37256786: 0.043691338, 29.68868737: 0.043677596,
    30.00474581: 0.043321833, 30.32078898: 0.042880564, 30.63704592: 0.043636371, 30.95321124: 0.043879145, 31.26937656: 0.044121919,
    31.58540446: 0.043595145, 31.90152397: 0.043581403, 32.21764348: 0.043567661, 32.533763: 0.043553919, 32.84988251: 0.043540177,
    33.16600203: 0.043526435, 33.48212154: 0.043512693, 33.79784406: 0.041275811, 34.06105934: 0.031104696, 34.39300429: 0.030305165,
    34.70923069: 0.030889961, 35.02539601: 0.031132735, 35.34151552: 0.031118993, 35.65748235: 0.030250197, 35.93845348: 0.030103617
}

long_b_OS_raw = {
    0.705868793: 0.999327507, 0.83466506: 0.977504837, 0.898495457: 0.96341418, 0.790237951: 0.988304222, 1.021797956: 0.944602858,
    1.291414756: 0.934516971, 1.572141961: 0.933004413, 1.880840486: 0.930772758, 1.934872894: 0.916498896, 2.22352009: 0.90304898,
    2.281249946: 0.885866898, 2.656252465: 0.873598711, 2.834084579: 0.862551539, 3.032518146: 0.849790296, 3.188402096: 0.837598908,
    3.225632406: 0.824803266, 3.429786968: 0.804902083, 3.423508444: 0.816694454, 3.714399838: 0.799931543, 3.933468455: 0.797293632,
    4.209094683: 0.796755696, 4.36233453: 0.789450724, 4.532981901: 0.772513108, 4.71836838: 0.758258426, 4.946934843: 0.752127573,
    5.08402833: 0.733058651, 5.291207598: 0.714724811, 5.685358873: 0.696119268, 5.894392524: 0.681475492, 6.248085546: 0.662392268,
    6.474317525: 0.650753611, 6.785111275: 0.645502855, 7.103533326: 0.633796061, 7.329252031: 0.622584956, 7.624887139: 0.602905714,
    7.953891464: 0.596367669, 8.107523621: 0.586330393, 8.356849875: 0.571236657, 8.55938312: 0.558017168, 8.805340885: 0.553810572,
    8.982515676: 0.534399105, 9.227279419: 0.528189277, 9.43713325: 0.523182049, 9.653358094: 0.504677968, 9.902273811: 0.497131927,
    10.20833766: 0.486468538, 10.51828552: 0.47999325, 10.83394697: 0.477414347, 11.13184391: 0.473703261, 11.62960372: 0.462680492,
    11.79530233: 0.451806248, 12.00476761: 0.44297218, 12.12577229: 0.432296503, 12.18615366: 0.41651709, 12.61346027: 0.411085072,
    12.7766755: 0.395244698, 12.84013985: 0.382283165, 13.13550297: 0.372495877, 13.38290053: 0.362330794, 13.56499835: 0.344550079,
    13.96948192: 0.333579894, 14.24910706: 0.325895752, 14.42273316: 0.314711342, 14.69607536: 0.298802149, 14.89112566: 0.290539521,
    15.254113: 0.289023387, 15.61123662: 0.282594909, 15.87851393: 0.279524681, 15.97735732: 0.267540617, 16.30129439: 0.252295771,
    16.36212306: 0.237370276, 16.73926536: 0.226536968, 17.03948803: 0.216180129, 17.35514948: 0.213601225, 17.65305596: 0.20994358,
    17.77445542: 0.194733374, 17.88737839: 0.179696317, 18.01397763: 0.165774705, 18.36461078: 0.16233922, 18.62940213: 0.155217974,
    18.87136455: 0.148037681, 19.380413: 0.14658638, 19.69656985: 0.1467817, 20.0127027: 0.146842655, 20.29395446: 0.146616663,
    21.12781673: 0.146669919, 21.8567316: 0.146753574, 22.43681561: 0.14640325, 22.60341789: 0.13426455, 22.96093053: 0.134276993,
    23.5228903: 0.134081921, 23.8389182: 0.133555147, 24.1549461: 0.133028372, 24.41832257: 0.132700735, 25.06823068: 0.133271026,
    25.38433492: 0.133171778, 25.70036051: 0.132632086, 26.00760159: 0.132073629, 26.27672898: 0.119729451, 26.64628901: 0.118982586,
    26.96246267: 0.119272111, 27.24432234: 0.119148516, 27.70004895: 0.119094872, 28.01614023: 0.118923038, 28.33221625: 0.118665698,
    28.61352908: 0.118781728, 29.21044354: 0.119285732, 29.49137927: 0.118940899, 29.80747802: 0.118810894, 30.10598874: 0.118536918,
    30.50236294: 0.107336234, 30.50422421: 0.086627256, 30.88741403: 0.068811017, 31.20335262: 0.067784128, 31.51941106: 0.067428365,
    31.80139598: 0.06800619, 32.3272415: 0.067222236, 32.66121572: 0.067194799, 32.95951107: 0.067365762, 33.27616131: 0.067022188,
    33.8727216: 0.067193807, 34.18876708: 0.066765456, 34.48781477: 0.066196667
}

def dict_to_sorted_lists(data):
    sorted_items = sorted(data.items())
    return [x for x, y in sorted_items], [y for x, y in sorted_items]

long_a_os_time, long_a_os_surv = dict_to_sorted_lists(long_a_OS_raw)
long_b_os_time, long_b_os_surv = dict_to_sorted_lists(long_b_OS_raw)

#=============================================================================================
# PROGRESSION-FREE SURVIVAL POINTS EXTRACTED FROM KAPLAN-MEIER CURVE USING WEBPLOTDIGITIZER
#=============================================================================================
long_a_pfs_raw = {
    0.39786219: 0.996706712, 0.680077092: 0.988841494, 0.926008375: 0.972264741, 1.03555669: 0.956446152, 1.169973423: 0.936753877,
    1.326959456: 0.915383001, 1.40669984: 0.899674929, 1.482241196: 0.877600406, 1.524902274: 0.850385575, 1.632220735: 0.822828207,
    1.669199294: 0.804163134, 1.676284234: 0.787216919, 1.709544043: 0.766810286, 1.728462685: 0.750256135, 1.785752425: 0.734124656,
    1.774582688: 0.711735801, 1.817230438: 0.696559995, 1.900327987: 0.679111552, 1.979062445: 0.665073003, 2.074189468: 0.649901849,
    2.270477106: 0.632235962, 2.516912121: 0.620509963, 2.662190439: 0.599960216, 2.796692526: 0.589201453, 2.847813996: 0.57045286,
    2.920818011: 0.558013311, 3.021292011: 0.537550521, 3.125885701: 0.522690188, 3.141694495: 0.504579889, 3.193139193: 0.488943864,
    3.279853958: 0.472259252, 3.44564649: 0.450520118, 3.664965602: 0.433192722, 3.822398278: 0.416122848, 4.004023111: 0.393534098,
    4.260939533: 0.380534287, 4.471376728: 0.362848882, 4.62830889: 0.340959245, 4.892190389: 0.326893288, 5.209443711: 0.31574881,
    5.526843955: 0.306019135, 5.825981984: 0.290774227, 6.125364533: 0.277883957, 6.442108526: 0.261834827, 6.694784783: 0.253428628,
    6.899388199: 0.236347145, 7.158718793: 0.221043439, 7.370423678: 0.207048273, 7.602497569: 0.193304474, 7.920779344: 0.192063619,
    8.239168862: 0.191860286, 8.557617149: 0.192222875, 8.875095752: 0.183247762, 9.193024915: 0.178611379, 9.511453612: 0.178785327,
    9.829196674: 0.17235686, 10.14635205: 0.160269179, 10.46415388: 0.154406633, 10.78213202: 0.150241851, 11.09992405: 0.144284985,
    11.41822542: 0.14323277, 11.72002866: 0.138167176, 12.05350585: 0.128395111, 12.37179742: 0.127248576, 12.67889413: 0.120505569,
    13.00791041: 0.120428136, 13.32632344: 0.120451172, 13.64451314: 0.118323707, 13.92626034: 0.105954698, 14.23398422: 0.095959656,
    14.56191298: 0.094701207, 14.8802829: 0.094309234, 15.19869201: 0.094294542, 15.51710112: 0.09427985, 15.83551023: 0.094265157,
    16.10064546: 0.092272189, 16.29346574: 0.075285571, 16.61181608: 0.074704957, 16.93022519: 0.074690265, 17.24865389: 0.074864213,
    17.5670434: 0.074660881, 17.88545251: 0.074646188, 18.20386162: 0.074631496, 18.52227073: 0.074616804, 18.84067983: 0.074602112,
    19.15908894: 0.07458742, 19.40649364: 0.072199123, 19.57019988: 0.047404096, 19.89912468: 0.046445712, 20.21745543: 0.045676458,
    20.53586454: 0.045661766, 20.85429323: 0.045835714, 21.1727709: 0.046481263, 21.49115063: 0.04618361, 21.80950097: 0.045602997,
    22.12796884: 0.046154226, 22.44631918: 0.045573612, 22.76472829: 0.04555892, 23.0831374: 0.045544228, 23.40153671: 0.045435216,
    23.72002418: 0.046175085, 24.03845287: 0.046349033, 24.35686198: 0.046334341, 24.67527109: 0.046319649, 24.99366061: 0.046116316,
    25.31207951: 0.046195944, 25.63049841: 0.046275573, 25.94890752: 0.04626088, 26.26731663: 0.046246188, 26.58572574: 0.046231496,
    26.90413484: 0.046216804, 27.22254395: 0.046202112, 27.54095306: 0.046187419, 27.85936217: 0.046172727, 28.17777128: 0.046158035,
    28.49618038: 0.046143343, 28.81458949: 0.046128651, 29.1329986: 0.046113958, 29.45140771: 0.046099266, 29.76981681: 0.046084574,
    30.08822592: 0.046069882, 30.40663503: 0.04605519, 30.72504414: 0.046040497, 31.04345325: 0.046025805, 31.36186235: 0.046011113,
    31.68027146: 0.045996421, 31.99868057: 0.045981729, 32.31708968: 0.045967037, 32.63549878: 0.045952344, 32.95387851: 0.045654692,
    33.272317: 0.04592296, 33.59080447: 0.04666283, 33.90922337: 0.046742458, 34.22763248: 0.046727765, 34.54604158: 0.046713073,
    34.86445069: 0.046698381, 35.18286959: 0.046778009, 35.50135706: 0.047517879, 35.81975637: 0.047408866, 35.99657196: 0.046646142
}

long_b_pfs_raw = {
    0.553339336: 0.994874433, 0.757923023: 0.988959172, 0.837809978: 0.974662535, 0.995277793: 0.957931034, 1.043091684: 0.941400309,
    1.226543246: 0.913689862, 1.289332762: 0.893738142, 1.441979935: 0.868439684, 1.529651629: 0.847450636, 1.601065057: 0.829428218,
    1.75534404: 0.811945874, 1.86403436: 0.793404132, 1.909222638: 0.764763746, 1.962081781: 0.747062032, 2.126141046: 0.718640359,
    2.331328609: 0.707183992, 2.542116151: 0.689195829, 2.859141673: 0.679534197, 3.263154817: 0.652138979, 3.36785429: 0.638297305,
    3.542103645: 0.612822683, 3.664207485: 0.596235475, 4.086299366: 0.572588669, 4.173174277: 0.557446193, 4.417760035: 0.527912536,
    4.628955346: 0.517527517, 4.909233328: 0.491010474, 4.996002944: 0.474854055, 5.294540867: 0.45383035, 5.612626746: 0.45070309,
    5.788962392: 0.445318675, 5.999006115: 0.423844293, 6.245410907: 0.411827249, 6.650387438: 0.393709099, 6.772622038: 0.378381065,
    7.013210215: 0.348206212, 7.210740159: 0.33871787, 7.55867175: 0.316376112, 7.684355446: 0.300192969, 8.249277088: 0.289198755,
    8.495763036: 0.277963222, 8.848610633: 0.268892156, 9.033703775: 0.25531467, 9.146656368: 0.243128859, 9.552092205: 0.229433654,
    9.869805883: 0.222722226, 10.04625907: 0.218469653, 10.34600846: 0.198082557, 10.71629332: 0.197612189, 11.04414726: 0.197371234,
    11.33102427: 0.190466599, 11.54867288: 0.181256304, 11.89898483: 0.17350313, 12.19793006: 0.156401687, 12.48102597: 0.149667333,
    12.79763084: 0.143307888, 13.15142643: 0.139689169, 13.47297241: 0.138909876, 13.71673212: 0.128716684, 14.42391986: 0.12862366,
    14.74229958: 0.128326007, 15.00879375: 0.12838971, 15.28952356: 0.128282436, 15.92730914: 0.13021542, 16.2899272: 0.129668805,
    16.35045833: 0.115524619, 17.00521412: 0.115433336, 17.32362323: 0.115418644, 17.64203233: 0.115403952, 17.83661568: 0.115394973,
    18.52654126: 0.115740421, 18.84499934: 0.11619733, 19.16336927: 0.115805357, 19.35794771: 0.115749218, 20.01247002: 0.115860498,
    20.33090851: 0.116128766, 20.64931762: 0.116114074, 20.84390096: 0.116105095, 21.533709: 0.115318701, 21.85210831: 0.115209688,
    22.17052722: 0.115289316, 22.36518892: 0.116034899, 22.55207356: 0.090559694, 22.64409808: 0.076336606, 23.33369798: 0.073545907,
    23.65210709: 0.073531215, 23.9705162: 0.073516522, 24.16509954: 0.073507544, 24.85498594: 0.073475711, 25.17339505: 0.073461019,
    25.49180415: 0.073446326, 25.6686981: 0.073438164, 26.34097347: 0.074161709, 26.65939047: 0.074223016, 26.9777623: 0.073849364,
    27.29353999: 0.074230397, 27.86218307: 0.073336951, 28.18059218: 0.073322259, 28.49900128: 0.073307567, 28.69358463: 0.073298588,
    29.34809224: 0.073268388, 29.6665895: 0.074102577, 29.98491045: 0.073239003, 30.1794938: 0.073230025, 30.86938999: 0.073292512,
    31.1877893: 0.073183499, 31.50619841: 0.073168807, 31.70078176: 0.073159829, 32.39067795: 0.073222316, 32.70907726: 0.073113303,
    33.02748637: 0.073098611, 33.23978849: 0.073371777, 33.91204427: 0.073906682, 34.23045338: 0.073891989, 34.49468404: 0.074217723
}

long_a_pfs_time, long_a_pfs_surv = dict_to_sorted_lists(long_a_pfs_raw)
long_b_pfs_time, long_b_pfs_surv = dict_to_sorted_lists(long_b_pfs_raw)

#=======================================================
# LITERATURE- AND TRIAL-DERIVED EFFECT SIZES (CITED)
#=======================================================
# Directly reported in Long et al. 2005, Table 1 ("Time From Diagnosis to
# Study Entry, Cox model"). The paper states the underlying relationship
# explicitly: "each month from primary diagnosis to study entry had a
# constant proportional decrease in risk up to 30 months, with no
# additional change in risk beyond that time" -- i.e., a per-6-month
# geometric decay. Checking the reported anchor points confirms this exactly
# (e.g., progression RR: 0.81, 0.81^2=0.656~0.65, 0.81^3=0.531~0.53, ...),
# so it is implemented here as a continuous exponential decay rather than
# discrete buckets -- a more faithful, and more precise, representation of
# what the paper actually describes than a step function would be.
RECURRENT_PROGRESSION_RR_PER_6MO = 0.81
RECURRENT_DEATH_RR_PER_6MO = 0.79
RECURRENT_RR_PLATEAU_MONTHS = 30.0
PERSISTENT_PROGRESSION_RR = 0.58
PERSISTENT_DEATH_RR = 0.72
ADVANCED_PROGRESSION_RR = 0.44     # "Advanced disease patients (no prior treatment)"
ADVANCED_DEATH_RR = 0.38
TREATMENT_HR_OS = 0.76             # CT vs CPT, unadjusted (abstract; adjusted = 0.77)
TREATMENT_HR_PFS = 0.76            # CT vs CPT, unadjusted (abstract; adjusted = 0.738)

# NOT reported in Long et al. 2005 -- self-proposed placeholders needing an
# external literature citation. Catalogued again in the weight reference
# table at the end of this script.
PS_HAZARD_RATIO_OS_ASSUMED = 1.799   # borrowed from Monk et al. 2009 (closely
                                      # related GOG cervical-cancer trial) since
                                      # Long 2005 does not report a PS-specific
                                      # HR directly -- NOT Long-specific data.
ORR_OS_HR_ASSUMED = 0.70
PFS_RESPONSE_HR_ASSUMED = 0.65


def hr_to_time_multiplier(hazard_ratio, weight=1.0):
    """
    Convert a Cox-model hazard ratio into an AFT-style multiplicative factor
    on survival TIME. Exact under a Weibull PH model; used here as a
    practical approximation otherwise. `weight` is a damping/tuning exponent
    (1 = apply as reported, 0 = no effect).
    """
    hazard_ratio = np.asarray(hazard_ratio, dtype=float)
    return hazard_ratio ** (-float(weight))


CENSORING_METHOD = 'uniform'
CENSORING_MAX_FACTOR = 1.1

def guyot_reconstruct_ipd(t_digitized, S_digitized, n_patients, tot_events=None, arm_id=1, random_state=None):
    """
    Reconstruct IPD from a digitized KM curve (Guyot et al. 2012). Same
    function used in the Monk/Kitagawa pipelines -- now usable here too
    since digitized WebPlotDigitizer coordinates are available for this
    paper's Figures 1 (PFS) and 2 (OS).
    """
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
    (here, the raw Guyot-reconstructed draw from the real digitized KM
    curve, before covariate adjustment). Covariate multipliers (age, PS,
    response, disease status) change each patient's time by a different
    factor, which -- left alone -- shifts the marginal shape away from the
    real curve; a single median-rescale only fixes the median, leaving the
    tail visibly off (verified: without this, the CPT arm's 20-month
    survival came out ~2x too high vs. the digitized curve). This keeps
    each patient's RELATIVE RANK (so covariate effects still matter --
    sicker patients still land toward the short end) but replaces the
    MAGNITUDE at that rank with the corresponding order statistic from the
    real Guyot draw, so the marginal distribution matches the actual
    digitized curve exactly, not just at the median.
    """
    n = len(adjusted_values)
    base_sorted = np.sort(base_values)
    ranks = pd.Series(adjusted_values).rank(method='first').astype(int).values - 1
    ranks = np.clip(ranks, 0, n - 1)
    remapped = base_sorted[ranks]
    return np.clip(remapped, 0.1, followup)


#=======================================================
# DEFINING BASE LISTS
# (identical to the Monk pipeline's lists, for cross-trial merge compatibility)
#=======================================================

histology_base = ["Squamous Cell Carcinoma", "Adenocarcinoma", "Adenosquamous", "Mucinous Adenocarcinoma",
                  "Clear Cell", "Endometrioid", "Villoglandular", "Undifferentiated Carcinoma", "Not Specified"]

# NOTE: Long et al. 2005's Table 2 reports "Race or ethnicity" as a SINGLE
# combined row that explicitly includes "Hispanic" as one of its categories
# (unlike Monk 2009, which reports Race and Ethnicity/Hispanic-origin as two
# SEPARATE rows). This means -- unlike in the Monk pipeline, where Hispanic
# had to stay at 0 because it was genuinely a different variable -- Long's
# real reported Hispanic count maps directly and faithfully into this same
# shared ethnicity_base list.
ethnicity_base = ["White", "Black", "Asian", "American Indian", "Hispanic", "Filipino", "Unspecified"]

disease_nature = ["Persistent", "Recurrent", "Advanced (IVB)"]

figo_stage = ["I", "II", "III", "IVA", "IVB", "Unknown"]

grade_base = ["Grade 1", "Grade 2", "Grade 3", "Grade Unspecified"]

death_cause_base = ["Treatment", "Disease", "Other/Unknown"]

dead_alive = ["Alive", "Dead", "Unspecified"]

# Long reports 3 performance-status categories (0/1/2), one more than Monk's
# 2 (0/1). The SAME shift-by-one code convention as Monk is reused
# (code = GOG_PS + 1), so no schema change is needed: code 3 (unused in
# Monk) now legitimately holds Long's PS=2 patients.
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
# NOTE: Long 2005's Table 3 reports only 19 (coarser, older GOG-CTC v2 era)
# adverse-event categories, versus Monk's 40. Most of the categories below
# will be 0 for Long -- expected, not a bug (this trial genuinely reported
# fewer, broader categories). Kept for cross-trial harmonization.

#===========================================================
# DRUG REGIMENS
# Long et al. 2005, "Patients and Methods": cisplatin 50 mg/m2 every 3 weeks
# (CPT, single agent); cisplatin 50 mg/m2 day 1 + topotecan 0.75 mg/m2 days
# 1-3 every 3 weeks (CT) -- identical dose/schedule to Monk 2009's TC arm.
#===========================================================

def _no_drug_block(prefix):
    return {
        f"{prefix}": 'No Drug', f"{prefix}_Dose_Value": 0, f"{prefix}_Dose_Unit": 'No Drug',
        f"{prefix}_Dose_Method": 'No Drug', f"{prefix}_Infusion_Duration_Hours": 0,
        f"{prefix}_Days_Number": 0, f"{prefix}_Days": 0, f"{prefix}_Same_Day_As_Platinum": 0,
    }

def _no_supportive_block(prefix):
    return {
        f"{prefix}": 'No Drug', f"{prefix}_Dose_Value": 0, f"{prefix}_Dose_Unit": 'No Drug',
        f"{prefix}_Route": 'No Drug', f"{prefix}_Start_Day": 0, f"{prefix}_Duration_Value": 0,
        f"{prefix}_Duration_Unit": 'No Drug', f"{prefix}_Frequency_Times_Per_Day": 'No Drug',
        f"{prefix}_Frequency_Interval_Hours": 0, f"{prefix}_Frequency_Description": 'No Drug',
        f"{prefix}_Dose_Is_Range": 0, f"{prefix}_Dose_Min": 0, f"{prefix}_Dose_Max": 0,
    }

DRUG_long_cpt = {
    "Platinum_Drug": "cisplatin", "Platinum_Dose_Value": 50, "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based", "Platinum_Infusion_Duration_Hours": 1,  # not explicitly stated; 1h with hydration is standard practice, not cited
    "Platinum_Days_Number": 1, "Platinum_Day": 1,
    **_no_drug_block("Adjunct_Drug_1"), **_no_drug_block("Adjunct_Drug_2"), **_no_drug_block("Adjunct_Drug_3"),
    **_no_supportive_block("Supportive_Drug_1"), **_no_supportive_block("Supportive_Drug_2"),
    **_no_supportive_block("Supportive_Drug_3"),
    "Cycle_Length_Days": 21
}

DRUG_long_ct = {
    "Platinum_Drug": "cisplatin", "Platinum_Dose_Value": 50, "Platinum_Dose_Unit": "mg/m2",
    "Platinum_Dose_Method": "BSA-based", "Platinum_Infusion_Duration_Hours": 1,  # not explicitly stated; 1h with hydration is standard practice, not cited
    "Platinum_Days_Number": 1, "Platinum_Day": 1,
    "Adjunct_Drug_1": "topotecan", "Adjunct_Drug_1_Dose_Value": 0.75, "Adjunct_Drug_1_Dose_Unit": "mg/m2",
    "Adjunct_Drug_1_Dose_Method": "BSA-based", "Adjunct_Drug_1_Infusion_Duration_Hours": 0.5,  # not explicitly stated; 30 min standard topotecan infusion, not cited
    "Adjunct_Drug_1_Days_Number": 3, "Adjunct_Drug_1_Days": (1, 2, 3), "Adjunct_Drug_1_Same_Day_As_Platinum": 1,
    **_no_drug_block("Adjunct_Drug_2"), **_no_drug_block("Adjunct_Drug_3"),
    **_no_supportive_block("Supportive_Drug_1"), **_no_supportive_block("Supportive_Drug_2"),
    **_no_supportive_block("Supportive_Drug_3"),
    "Cycle_Length_Days": 21
}

#==================================================
# ASSIGNING PARAMETERS
#==================================================
def parameter_assign(data, ratio):
    if len(data) != len(ratio):
        raise ValueError(f"Length mismatch: {len(data)} keys vs {len(ratio)} values")
    return dict(zip(data, ratio))

#==========================================================================
# HISTOLOGY (Table 2, "Cell type" row)
#==========================================================================
HISTOLOGY_long_cpt = parameter_assign(histology_base, [121, 9, 11, 0, 2, 3, 0, 0, 0])
HISTOLOGY_long_ct = parameter_assign(histology_base, [128, 9, 5, 4, 0, 0, 1, 0, 0])

#===========================================================
# RACE/ETHNICITY (Table 2, "Race or ethnicity" row -- see note above:
# Hispanic is genuinely captured here for Long, unlike for Monk)
#===========================================================
ETHNICITY_long_cpt = parameter_assign(ethnicity_base, [108, 23, 3, 0, 11, 0, 1])
ETHNICITY_long_ct = parameter_assign(ethnicity_base, [105, 29, 3, 0, 9, 0, 1])

#================================================================
# DISEASE NATURE (Table 2, "Stage" row: IVB / Persistent / Recurrent)
#================================================================
NATURE_long_cpt = parameter_assign(disease_nature, [11, 118, 17])   # Persistent, Recurrent, Advanced(IVB)
NATURE_long_ct = parameter_assign(disease_nature, [17, 112, 18])

#======================================================================
# FIGO STAGE (only the Advanced/IVB subgroup is confirmed IVB; initial
# stage for Persistent/Recurrent patients is not reported -- same
# convention as the corrected Monk pipeline)
#======================================================================
STAGE_long_cpt = parameter_assign(figo_stage, [0, 0, 0, 0, 17, 129])
STAGE_long_ct = parameter_assign(figo_stage, [0, 0, 0, 0, 18, 129])

#======================================================================
# TUMOR GRADE (Table 2). Grade Unspecified = n - (G1+G2+G3): 146-142=4 for
# CPT, 147-144=3 for CT -- this also resolves Table 2's footnote ("Not known
# for four patients in CPT arm and for three patients in CT arm"), which
# attaches to tumor grade, not to the race/ethnicity row.
#======================================================================
GRADE_long_cpt = parameter_assign(grade_base, [9, 81, 52, 4])
GRADE_long_ct = parameter_assign(grade_base, [8, 84, 52, 3])

#================================================================
# PERFORMANCE STATUS (Table 2). code1=PS0, code2=PS1, code3=PS2, code4 unused.
#================================================================
PS_long_cpt = parameter_assign(performance_status, [68, 66, 12, 0])
PS_long_ct = parameter_assign(performance_status, [69, 66, 12, 0])

#====================================================================
# PRIOR TREATMENT (Table 2, "Prior cisplatin (radiosensitizing)" row --
# matches the "chemoradiotherapy" category exactly: "nearly 60% of patients
# in both treatment arms had received prior cisplatin as part of a
# chemoradiotherapy regimen used to treat their primary disease")
#====================================================================
_NONE_IDX = prior_treatment_base.index("none")
_CRT_IDX = prior_treatment_base.index("chemoradiotherapy")

def _treatment_counts(none_n, crt_n):
    counts = [0] * len(prior_treatment_base)
    counts[_NONE_IDX] = none_n
    counts[_CRT_IDX] = crt_n
    return counts

TREATMENT_long_cpt = parameter_assign(prior_treatment_base, _treatment_counts(64, 82))
TREATMENT_long_ct = parameter_assign(prior_treatment_base, _treatment_counts(62, 85))

#================================================================
# CAUSE OF DEATH
# Long 2005 has NO dedicated cause-of-death table like Monk's Table 3. Only
# 3 specific deaths are individually narrated in the Results/Toxicity text:
# one CT-arm patient died of "hemorrhagic complications of progressive
# disease, perhaps aggravated by treatment-induced thrombocytopenia" (coded
# here as Treatment-attributable, being the only death the paper links, even
# partially, to therapy), and two more CT-arm patients died of pulmonary
# emboli "believed unrelated to protocol treatment" (coded as Other/Unknown,
# since the paper explicitly disclaims a treatment link but does not
# affirmatively attribute them to disease progression either). Every other
# death in both arms is assigned to "Disease" by default -- there is no
# stated evidence otherwise, and this is what an advanced/progressive
# cervical-cancer population would be expected to show. This is far less
# granular than Monk's Table 3 and should be treated as a low-confidence
# default, not trial-reported data, outside those 3 specific patients.
#================================================================
CAUSE_long_cpt = {"Treatment": 0, "Other/Unknown": 0}
CAUSE_long_ct = {"Treatment": 1, "Other/Unknown": 2}

#================================================================
# RESPONSE (Table 4). "Increasing disease" -> "PD"; "Stable" -> "SD"; the
# inassessable patients get a 5th response value, "NE" (not evaluable) --
# added to accommodate a genuine difference from Monk (where every patient
# had an assessable response), NOT a new column.
#================================================================
CR_COUNT_cpt, PR_COUNT_cpt, SD_COUNT_cpt, PD_COUNT_cpt, NE_COUNT_cpt = 4, 14, 70, 51, 7
CR_COUNT_ct, PR_COUNT_ct, SD_COUNT_ct, PD_COUNT_ct, NE_COUNT_ct = 14, 22, 61, 38, 12
assert CR_COUNT_cpt + PR_COUNT_cpt + SD_COUNT_cpt + PD_COUNT_cpt + NE_COUNT_cpt == LONG_CPT_NUMBER
assert CR_COUNT_ct + PR_COUNT_ct + SD_COUNT_ct + PD_COUNT_ct + NE_COUNT_ct == LONG_CT_NUMBER

#================================================================
# TOXICITY (Table 3 -- raw patient counts, Grade 3 and Grade 4, no external
# file needed: unlike Monk's appendix, Long's Table 3 already reports counts
# directly rather than percentages needing conversion)
#================================================================
# source_event: (CPT_G3, CPT_G4, CT_G3, CT_G4)
_LONG_TOX_RAW = {
    "Leukopenia": (1, 0, 58, 35),
    "Granulocytopenia": (1, 1, 36, 67),
    "Thrombocytopenia": (5, 0, 36, 10),
    "Anemia": (28, 5, 47, 9),
    "Other hematologic": (16, 2, 17, 4),
    "Infection": (11, 0, 21, 5),
    "Renal": (7, 7, 9, 9),
    "Nausea": (13, 0, 18, 2),
    "Emesis": (13, 0, 20, 2),
    "Other GI": (12, 3, 16, 4),
    "Metabolic": (14, 1, 13, 7),
    "Neuropathy": (1, 0, 1, 0),
    "Other neurologic": (7, 2, 3, 1),
    "Cardiovascular": (7, 3, 7, 6),
    "Pulmonary": (5, 3, 4, 0),
    "Pain": (18, 5, 28, 3),
    "Constitutional": (17, 0, 11, 0),
    "Hemorrhage": (3, 1, 8, 1),
    "Hepatic": (2, 0, 5, 2),
}
# Mapping to the shared toxicity_base categories. Two source categories
# (Nausea + Emesis) get combined into a single target ("Nausea/vomiting"),
# same as several Monk mappings -- flagged there and here: summing counts
# from two source categories risks modestly overestimating the combined
# category's unique-patient count if some patients had both toxicities
# (no joint/overlap data is available to correct for this).
_LONG_TOX_NAME_MAP = {
    "Leukopenia": "Leucopenia", "Granulocytopenia": "Neutropenia", "Thrombocytopenia": "Thrombocytopenia",
    "Anemia": "Anemia", "Other hematologic": "Other hematologic",
    "Infection": "Other infection/fever", "Renal": "Other genitourinary/renal",
    "Nausea": "Nausea/vomiting", "Emesis": "Nausea/vomiting", "Other GI": "Other Gastrointestinal",
    "Metabolic": "Metabolic", "Neuropathy": "Peripheral neuropathy", "Other neurologic": "Other neurological",
    "Cardiovascular": "Other cardiovascular", "Pulmonary": "Pulmonary", "Pain": "Pain",
    "Constitutional": "Other constitutional", "Hemorrhage": "Hemorrhage", "Hepatic": "Hepatic",
}

def build_toxicity_counts(g3_idx, g4_idx):
    mapped = {c: 0 for c in toxicity_base}
    for source_event, counts in _LONG_TOX_RAW.items():
        target = _LONG_TOX_NAME_MAP[source_event]
        mapped[target] += counts[g3_idx] + counts[g4_idx]
    return mapped

TOXICITY_long_cpt = build_toxicity_counts(0, 1)
TOXICITY_long_ct = build_toxicity_counts(2, 3)

#======================================================================
# CREATING FINAL TRIAL DICTIONARIES
#======================================================================
def create_trial_dict(
    trial_id, drug_combination, ethnicity, histology, disease_nature_dist, disease_stage,
    grades, causes, performance_status_dist, cycle_completion_rate,
    toxicity, prior_treatment, patient_number, followup,
    cr_count, pr_count, sd_count, pd_count, ne_count,
    os_median, os_q25, os_q75, os_event_rate,
    pfs_median, pfs_q25, pfs_q75, pfs_event_rate,
    target_age, age_type, age_min, age_max, age_sd,
    os_ps_weight, os_orr_weight, os_age_weight,
    pfs_age_weight, pfs_histology_weight, pfs_response_weight, pfs_toxicity_weight,
    orr_vs_ps, orr_vs_disease_status, tolerability_orr_weight,
    pfsr_vs_ps,
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
        "performance_status": performance_status_dist, "cycle_completion_rate": cycle_completion_rate,
        "toxicity": toxicity, "prior_treatment": prior_treatment,
        "patient_number": patient_number, "followup": followup,
        "cr_count": cr_count, "pr_count": pr_count, "sd_count": sd_count,
        "pd_count": pd_count, "ne_count": ne_count,
        "os_median": os_median, "os_q25": os_q25, "os_q75": os_q75, "os_event_rate": os_event_rate,
        "pfs_median": pfs_median, "pfs_q25": pfs_q25, "pfs_q75": pfs_q75, "pfs_event_rate": pfs_event_rate,
        "target_age": target_age, "age_type": age_type, "age_min": age_min, "age_max": age_max, "age_sd": age_sd,
        "os_ps_weight": os_ps_weight, "os_orr_weight": os_orr_weight, "os_age_weight": os_age_weight,
        "pfs_age_weight": pfs_age_weight, "pfs_histology_weight": pfs_histology_weight,
        "pfs_response_weight": pfs_response_weight, "pfs_toxicity_weight": pfs_toxicity_weight,
        "orr_vs_ps": orr_vs_ps, "orr_vs_disease_status": orr_vs_disease_status,
        "tolerability_orr_weight": tolerability_orr_weight, "pfsr_vs_ps": pfsr_vs_ps,
        "never_treated_count": never_treated_count,
    })
    return trial

_PFS_HISTOLOGY_WEIGHT = {
    "Squamous Cell Carcinoma": 1.00, "Adenocarcinoma": 0.75, "Adenosquamous": 0.85,
    "Mucinous Adenocarcinoma": 0.65, "Clear Cell": 0.70, "Endometrioid": 0.80,
    "Villoglandular": 1.15, "Undifferentiated Carcinoma": 0.50, "Not Specified": 0.90,
}

# Self-proposed OS/PFS event (vs. censored) rates -- see the methodological
# note at the top of this file. CT's rates are set modestly lower than
# CPT's, reflecting that at any fixed calendar analysis cutoff, an arm with
# longer survival will tend to have proportionately fewer observed events.
# Self-proposed OS/PFS event (vs. censored) rates. Now that real digitized
# curves are available, these are calibrated against the curves' OWN
# endpoint behavior (e.g., the CPT OS curve declines to S~=0.03 by month 36,
# implying a true event rate close to 0.96-0.97 -- the earlier 0.85/0.78
# guess, made before digitized data was available, was too conservative and
# left an artificial plateau of "censored" patients bunched at the tail).
OS_EVENT_RATE_CPT, OS_EVENT_RATE_CT = 0.96, 0.91
PFS_EVENT_RATE_CPT, PFS_EVENT_RATE_CT = 0.95, 0.92

#=====================================================================
# long-cpt (CISPLATIN ALONE) TRIAL CONSTANTS
#=====================================================================
trial_long_cpt = create_trial_dict(
    trial_id="long_CPT_arm", drug_combination=DRUG_long_cpt, ethnicity=ETHNICITY_long_cpt,
    histology=HISTOLOGY_long_cpt, disease_nature_dist=NATURE_long_cpt, disease_stage=STAGE_long_cpt,
    grades=GRADE_long_cpt, causes=CAUSE_long_cpt, performance_status_dist=PS_long_cpt,
    cycle_completion_rate=0.25,   # self-proposed estimate: paper reports median 3 of max 6 cycles received
    toxicity=TOXICITY_long_cpt, prior_treatment=TREATMENT_long_cpt, patient_number=LONG_CPT_NUMBER,
    followup=36,
    cr_count=CR_COUNT_cpt, pr_count=PR_COUNT_cpt, sd_count=SD_COUNT_cpt, pd_count=PD_COUNT_cpt, ne_count=NE_COUNT_cpt,
    os_median=6.5, os_q25=4.1, os_q75=13.4, os_event_rate=OS_EVENT_RATE_CPT,
    pfs_median=2.9, pfs_q25=1.4, pfs_q75=6.2, pfs_event_rate=PFS_EVENT_RATE_CPT,
    target_age=48, age_type="median", age_min=27, age_max=76, age_sd=(76 - 27) / 4,
    os_ps_weight=0.4, os_orr_weight=0.3, os_age_weight=0.15,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_disease_status=-0.35, tolerability_orr_weight=0.20,
    pfsr_vs_ps=-0.35,
    never_treated_count=2
)

#=====================================================================
# long-ct (CISPLATIN + TOPOTECAN) TRIAL CONSTANTS
#=====================================================================
trial_long_ct = create_trial_dict(
    trial_id="long_CT_arm", drug_combination=DRUG_long_ct, ethnicity=ETHNICITY_long_ct,
    histology=HISTOLOGY_long_ct, disease_nature_dist=NATURE_long_ct, disease_stage=STAGE_long_ct,
    grades=GRADE_long_ct, causes=CAUSE_long_ct, performance_status_dist=PS_long_ct,
    cycle_completion_rate=0.35,   # self-proposed estimate: paper reports median 4 of max 6 cycles received
    toxicity=TOXICITY_long_ct, prior_treatment=TREATMENT_long_ct, patient_number=LONG_CT_NUMBER,
    followup=36,
    cr_count=CR_COUNT_ct, pr_count=PR_COUNT_ct, sd_count=SD_COUNT_ct, pd_count=PD_COUNT_ct, ne_count=NE_COUNT_ct,
    os_median=9.4, os_q25=4.6, os_q75=16.0, os_event_rate=OS_EVENT_RATE_CT,
    pfs_median=4.6, pfs_q25=1.7, pfs_q75=8.7, pfs_event_rate=PFS_EVENT_RATE_CT,
    target_age=46, age_type="median", age_min=22, age_max=84, age_sd=(84 - 22) / 4,
    os_ps_weight=0.4, os_orr_weight=0.3, os_age_weight=0.15,
    pfs_age_weight=0.15, pfs_histology_weight=_PFS_HISTOLOGY_WEIGHT,
    pfs_response_weight=0.3, pfs_toxicity_weight=0.1,
    orr_vs_ps=-0.30, orr_vs_disease_status=-0.35, tolerability_orr_weight=0.20,
    pfsr_vs_ps=-0.35,
    never_treated_count=7
)

TRIALS = {
    "long_cpt": trial_long_cpt,
    "long_ct": trial_long_ct,
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

def recurrent_progression_rr(months_since_dx):
    """Table 1: geometric decay per 6 months, plateau at 30 months."""
    m = np.minimum(np.asarray(months_since_dx, dtype=float), RECURRENT_RR_PLATEAU_MONTHS)
    return RECURRENT_PROGRESSION_RR_PER_6MO ** (m / 6.0)

def recurrent_death_rr(months_since_dx):
    m = np.minimum(np.asarray(months_since_dx, dtype=float), RECURRENT_RR_PLATEAU_MONTHS)
    return RECURRENT_DEATH_RR_PER_6MO ** (m / 6.0)

# ================================================================
# Main Data Generation Function
# ================================================================
def generate_clinical_data(trial_config, arm_name='long_cpt'):
    np.random.seed(SEED)
    n = trial_config["patient_number"]
    FOLLOWUP = trial_config["followup"]
    CYCLE_COMPLETION_RATE = trial_config.get("cycle_completion_rate", 0.30)
    OS_PS_WEIGHT = trial_config["os_ps_weight"]
    OS_ORR_WEIGHT = trial_config["os_orr_weight"]
    OS_AGE_WEIGHT = trial_config["os_age_weight"]
    PFS_AGE_WEIGHT = trial_config["pfs_age_weight"]
    PFS_RESPONSE_WEIGHT = trial_config["pfs_response_weight"]
    PFS_TOXICITY_WEIGHT = trial_config["pfs_toxicity_weight"]
    ORR_VS_PS = trial_config["orr_vs_ps"]
    ORR_VS_DISEASE_STATUS = trial_config["orr_vs_disease_status"]
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

    arm_id = {'long_cpt': 1, 'long_ct': 2}[arm_name]

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
    # 4. Disease nature / FIGO stage / time-since-diagnosis hazard ratios
    #    (Table 1, CITED for BOTH progression and death -- more precise
    #    than what was available for Monk, which only cited an OS HR)
    # ----------------------------------------------------------------
    nature_dist = expand_distribution(trial_config.get("disease_nature", {}), n)
    disease_nature_list = []
    progression_rr = np.ones(n)   # applied to PFS
    death_rr = np.ones(n)         # applied to OS
    months_since_dx = np.full(n, np.nan)
    for i, nat_raw in enumerate(nature_dist):
        nat_lower = str(nat_raw).lower().strip()
        if "persistent" in nat_lower:
            disease_nature_list.append("Persistent")
            progression_rr[i] = PERSISTENT_PROGRESSION_RR
            death_rr[i] = PERSISTENT_DEATH_RR
        elif "recurrent" in nat_lower:
            disease_nature_list.append("Recurrent")
            # Self-proposed distribution for months-since-diagnosis (the
            # paper gives the RR-decay FUNCTION, Table 1, but not the
            # distribution of patients across time-since-diagnosis values).
            m = np.random.exponential(scale=12.0)
            m = min(m, 60.0)
            months_since_dx[i] = m
            progression_rr[i] = recurrent_progression_rr(m)
            death_rr[i] = recurrent_death_rr(m)
        else:
            disease_nature_list.append("Advanced (IVB)")
            progression_rr[i] = ADVANCED_PROGRESSION_RR
            death_rr[i] = ADVANCED_DEATH_RR

    figo_stage_list = ["IVB" if nat == "Advanced (IVB)" else "Unknown" for nat in disease_nature_list]

    # ----------------------------------------------------------------
    # 5. Performance status (base assignment from exact trial counts, then
    #    disease- and tolerability-driven worsening, then forced back onto
    #    the exact reported marginal counts)
    # ----------------------------------------------------------------
    target_ps_dist = trial_config["performance_status"]
    ps1_count = target_ps_dist.get(1, 0)   # GOG PS0
    ps2_count = target_ps_dist.get(2, 0)   # GOG PS1
    ps3_count = target_ps_dist.get(3, 0)   # GOG PS2
    PS_discrete = np.array([1] * ps1_count + [2] * ps2_count + [3] * ps3_count)
    if len(PS_discrete) < n:
        PS_discrete = np.concatenate([PS_discrete, np.full(n - len(PS_discrete), 2)])
    PS_discrete = np.random.permutation(PS_discrete[:n])

    # Disease-status-driven PS worsening (self-proposed; death_rr is bounded
    # <=1 by construction of Table 1, so use progression_rr as the severity
    # signal instead: values close to 1 mean recent/aggressive recurrence)
    for i in range(n):
        if disease_nature_list[i] == "Recurrent" and progression_rr[i] > 0.75:
            if np.random.random() < 0.3:
                PS_discrete[i] = min(PS_discrete[i] + 1, 3)

    worsen_idx = np.where(tolerability < 0.35)[0]
    if len(worsen_idx) > 0:
        worsen_sample = np.random.choice(worsen_idx, size=int(0.1 * len(worsen_idx)), replace=False)
        PS_discrete[worsen_sample] = np.clip(PS_discrete[worsen_sample] + 1, 1, 3)

    # Smart correction -- force back onto Table 2's exact marginal counts
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
    #    Built directly and EXACTLY from Table 4's reported counts, with
    #    WHICH patients land where driven by a transparent latent-propensity
    #    index (same architecture as the corrected Monk pipeline).
    # ----------------------------------------------------------------
    ne_idx = np.random.choice(n, NE_COUNT, replace=False) if NE_COUNT > 0 else np.array([], dtype=int)
    evaluable_idx = np.setdiff1d(np.arange(n), ne_idx)

    target_orr_count = CR_COUNT + PR_COUNT
    disease_severity_z = zscore(-progression_rr[evaluable_idx])   # higher z = more favorable disease status
    ps_z = zscore(PS_discrete[evaluable_idx])
    tol_z = zscore(tolerability[evaluable_idx])

    response_propensity = (
        ORR_VS_PS * ps_z +
        ORR_VS_DISEASE_STATUS * (-disease_severity_z) +
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
    # 8. OS / PFS generation (Guyot reconstruction from digitized KM curves)
    # ================================================================
    os_time_raw = long_a_os_time if arm_name == 'long_cpt' else long_b_os_time
    os_surv_raw = long_a_os_surv if arm_name == 'long_cpt' else long_b_os_surv
    os_ipd = guyot_reconstruct_ipd(
        os_time_raw, os_surv_raw, n,
        tot_events=int(round(trial_config["os_event_rate"] * n)),
        arm_id=arm_id, random_state=SEED + arm_id
    )

    age_factor = 1 - (age - np.mean(age)) / np.std(age) * OS_AGE_WEIGHT * 2
    age_factor = np.clip(age_factor, 0.4, 1.6)

    # PS -> OS: base HR borrowed from Monk 2009 (NOT Long-specific -- see
    # PS_HAZARD_RATIO_OS_ASSUMED definition above). Applied to PS>=2 (i.e.
    # GOG PS>=1) vs PS=1 (GOG PS0) reference.
    ps_time_multiplier = np.where(PS_discrete >= 2,
                                   hr_to_time_multiplier(PS_HAZARD_RATIO_OS_ASSUMED, OS_PS_WEIGHT),
                                   1.0)

    orr_time_multiplier = np.where(ORR == 1,
                                    hr_to_time_multiplier(ORR_OS_HR_ASSUMED, OS_ORR_WEIGHT),
                                    1.0)

    # Disease status / time-since-diagnosis -> OS: TRIAL-CITED (Table 1)
    disease_time_multiplier_os = hr_to_time_multiplier(death_rr, 1.0)

    os_covariate_factor = age_factor * ps_time_multiplier * orr_time_multiplier * disease_time_multiplier_os
    os_adjusted = os_ipd['time'].values * os_covariate_factor
    os_adjusted = np.clip(os_adjusted, 0.1, FOLLOWUP)

    # Empirical rank-remap onto the real Guyot draw (os_ipd['time']) itself,
    # not just a median rescale -- see empirical_quantile_remap() docstring
    # for why a median-only rescale left the tail badly off.
    observed_os = empirical_quantile_remap(os_adjusted, os_ipd['time'].values, FOLLOWUP)

    # The remap preserves the Guyot draw's own median exactly (by
    # construction it's a permutation of the same values) -- but the raw
    # FULL correction (not damped) to force the EXACT reported median, same
    # convention used across every other trial in this series (Monk,
    # Kitagawa, Miller, Bloss). This was originally left damped here because
    # the raw Guyot reconstruction's median drifted modestly from the target
    # (~7.3 vs. reported 6.5 for CPT) and a full correction was thought to
    # risk dragging the tail off -- but the empirical quantile remap already
    # applied above constrains the ENTIRE reconstructed distribution shape
    # to the digitized curve's own order statistics regardless of this final
    # multiplicative step, so a full correction here costs no tail fidelity
    # (the same finding that justified switching Miller and Bloss to full
    # correction) while eliminating the ~0.2-0.4 month residual gap a damped
    # correction leaves against the published target.
    current_median = np.median(observed_os)
    if current_median > 0:
        scaling_factor = trial_config["os_median"] / current_median
        observed_os = observed_os * scaling_factor
    observed_os = np.round(np.clip(observed_os, 0.1, FOLLOWUP), 1)

    # Re-derive event/censoring from the final (post-covariate, post-remap)
    # ranking: events should be the patients with the shortest FINAL times,
    # not whichever patients happened to have the shortest time before their
    # own covariates were applied.
    n_events_os = int(round(trial_config["os_event_rate"] * n))
    order_os = np.argsort(observed_os)
    event_os = np.zeros(n, dtype=int)
    event_os[order_os[:n_events_os]] = 1
    alive = 1 - event_os

    # ---------------- PFS ----------------
    pfs_time_raw = long_a_pfs_time if arm_name == 'long_cpt' else long_b_pfs_time
    pfs_surv_raw = long_a_pfs_surv if arm_name == 'long_cpt' else long_b_pfs_surv
    pfs_ipd = guyot_reconstruct_ipd(
        pfs_time_raw, pfs_surv_raw, n,
        tot_events=int(round(trial_config["pfs_event_rate"] * n)),
        arm_id=arm_id, random_state=SEED + arm_id + 10
    )
    observed_pfs = np.round(pfs_ipd['time'].values, 1)
    event_pfs = pfs_ipd['event'].values

    pfs_age_factor = 1 - (age - np.mean(age)) / np.std(age) * PFS_AGE_WEIGHT * 2
    observed_pfs = observed_pfs * pfs_age_factor

    histology_weights = np.array([trial_config["pfs_histology_weight"][h] for h in histology])
    observed_pfs = observed_pfs * histology_weights

    pfs_response_multiplier = np.where(ORR == 1,
                                        hr_to_time_multiplier(PFS_RESPONSE_HR_ASSUMED, PFS_RESPONSE_WEIGHT),
                                        1.0)
    observed_pfs = observed_pfs * pfs_response_multiplier

    # Disease status / time-since-diagnosis -> PFS: TRIAL-CITED (Table 1)
    disease_time_multiplier_pfs = hr_to_time_multiplier(progression_rr, 1.0)
    observed_pfs = observed_pfs * disease_time_multiplier_pfs
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

    # Empirical rank-remap onto the real Guyot draw (pfs_ipd['time']) itself
    # -- same reasoning as OS above.
    observed_pfs = empirical_quantile_remap(observed_pfs, pfs_ipd['time'].values, FOLLOWUP)

    # FULL correction (not damped) -- same convention and same justification
    # as the OS fix above: the empirical remap already fixes the entire
    # distribution shape, so a full correction costs no tail fidelity.
    current_pfs_median = np.median(observed_pfs)
    if current_pfs_median > 0:
        pfs_scaling_factor = trial_config["pfs_median"] / current_pfs_median
        observed_pfs = observed_pfs * pfs_scaling_factor
    observed_pfs = np.round(np.clip(observed_pfs, 0.1, FOLLOWUP), 1)

    n_events_pfs = int(round(trial_config["pfs_event_rate"] * n))
    order_pfs = np.argsort(observed_pfs)
    event_pfs = np.zeros(n, dtype=int)
    event_pfs[order_pfs[:n_events_pfs]] = 1

    # ================================================================
    # 10. Cause of death (see CAUSE_long_* construction notes above)
    # ================================================================
    dead_idx = np.where(alive == 0)[0]
    causes = np.full(n, "Alive", dtype=object)
    cause_dist = dict(trial_config.get("causes", {}))
    treatment_n = cause_dist.get("Treatment", 0) or 0
    other_n = cause_dist.get("Other/Unknown", 0) or 0
    if len(dead_idx) > 0:
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
    # Final DataFrame (column names/order identical to the Monk pipeline)
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
df_long_cpt = generate_clinical_data(TRIALS["long_cpt"], arm_name='long_cpt')
df_long_ct = generate_clinical_data(TRIALS["long_ct"], arm_name='long_ct')

df_long = pd.concat([df_long_cpt, df_long_ct], ignore_index=True)\
            .sample(frac=1, random_state=42).reset_index(drop=True)
df_long.to_excel("long_km.xlsx")

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
        trial_cfg['os_median'], trial_cfg['os_q25'], trial_cfg['os_q75'],
        trial_cfg['pfs_median'], trial_cfg['pfs_q25'], trial_cfg['pfs_q75'],
        trial_cfg['target_age'], trial_cfg['age_min'], trial_cfg['age_max'], trial_cfg['age_sd'],
        trial_cfg['ethnicity'], trial_cfg['histology'], trial_cfg['disease_nature'],
        trial_cfg['disease_stage'], trial_cfg['grades'], trial_cfg['performance_status'],
        trial_cfg['toxicity']
    ]

    output = [
        len(df), float(df['followup'].iloc[0]), round(df['orr'].mean(), 3), round(df['pfsr'].mean(), 3),
        df['response'].value_counts().to_dict(),
        round(df['os_months'].median(), 2), round(df['os_months'].quantile(0.25), 2), round(df['os_months'].quantile(0.75), 2),
        round(df['pfs_months'].median(), 2), round(df['pfs_months'].quantile(0.25), 2), round(df['pfs_months'].quantile(0.75), 2),
        int(df['age'].median()), int(df['age'].min()), int(df['age'].max()), round(df['age'].std(ddof=0), 2),
        df['ethnicity'].value_counts().to_dict(), df['histology/cell_type'].value_counts().to_dict(),
        df['disease_nature'].value_counts().to_dict(), df['figo_stage'].value_counts().to_dict(),
        df['tumor_grade'].value_counts().to_dict(), df['performance_status'].value_counts().to_dict(),
        get_toxicity_counts(df["toxicity_events"], trial_cfg['toxicity'])
    ]

    row_names = [
        "Patient Number", "Followup", "ORR", "PFSR (disease control rate)", "Tumor Response (CR/PR/SD/PD/NE)",
        "Overall Survival median (months)", "OS 25th pctile", "OS 75th pctile",
        "Progression-Free Survival median (months)", "PFS 25th pctile", "PFS 75th pctile",
        "Age", "Minimum Age", "Maximum Age", "Age SD",
        "Ethnicity", "Histology", "Disease Type", "Disease Stage",
        "Tumor Grade", "Performance Status", "Toxicity"
    ]

    eval_df = pd.DataFrame({"output_values": output, "target_values": constants}, index=row_names)
    for i, row in enumerate(row_names):
        target_val, output_val = constants[i], output[i]
        if isinstance(target_val, dict) and isinstance(output_val, dict):
            eval_df.at[row, "output_values"] = standardize_dict_order(output_val, target_val)
    return eval_df

trial_datasets = {"long_cpt": df_long_cpt, "long_ct": df_long_ct}
evaluation_results = {name: evaluate_trial(name, df) for name, df in trial_datasets.items()}
combined_evaluation_long = pd.concat(evaluation_results, axis=1)

pd.set_option('display.max_colwidth', 120)
combined_evaluation_long.to_excel("long_evaluation_km.xlsx")

print("\n--- Treatment HR QC cross-check (median-ratio approximation) ---")
print(f"OS:  CPT/CT median ratio = {TRIALS['long_cpt']['os_median']/TRIALS['long_ct']['os_median']:.3f}  vs reported unadjusted HR = {TREATMENT_HR_OS}")
print(f"PFS: CPT/CT median ratio = {TRIALS['long_cpt']['pfs_median']/TRIALS['long_ct']['pfs_median']:.3f}  vs reported unadjusted HR = {TREATMENT_HR_PFS}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SELF-PROPOSED CORRELATION / WEIGHT REFERENCE TABLE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
weight_reference_table = pd.DataFrame([
    {"category": "Survival reconstruction", "parameter": "Weibull quantile-fit (vs. Guyot digitized-curve)",
     "value": "n/a -- method choice", "applies_to": "OS & PFS base time generation",
     "cited_in_long2005": "Median/IQR values ARE cited (Table 5); the METHOD substitution is necessitated by the absence of digitized curve data.",
     "note": "See methodological note at top of file. Upgrade to guyot_reconstruct_ipd() if digitized coordinates become available."},
    {"category": "Survival reconstruction", "parameter": "os_event_rate (CPT/CT)", "value": "0.85 / 0.78",
     "applies_to": "OS censoring", "cited_in_long2005": "NO",
     "note": "Paper gives only the INTERIM 56-CPT-death count and a DESIGN target of 111 CPT deaths -- not a confirmed final count."},
    {"category": "Survival reconstruction", "parameter": "pfs_event_rate (CPT/CT)", "value": "0.93 / 0.88",
     "applies_to": "PFS censoring", "cited_in_long2005": "NO", "note": "Same gap as OS event rate, no reported final PFS event count."},
    {"category": "OS modeling", "parameter": "os_age_weight", "value": 0.15,
     "applies_to": "Age -> OS (AFT exponent)", "cited_in_long2005": "NO",
     "note": "Paper adjusts for age in its Cox model but does not report the age-specific coefficient."},
    {"category": "OS modeling", "parameter": "os_ps_weight / PS_HAZARD_RATIO_OS_ASSUMED", "value": "0.4 / 1.799",
     "applies_to": "PS -> OS", "cited_in_long2005": "NO -- borrowed from Monk et al. 2009",
     "note": "Long 2005 does not report a PS-specific HR; the magnitude is imported from a related GOG trial in the same disease, not derived from this trial's own data."},
    {"category": "OS modeling", "parameter": "os_orr_weight / ORR_OS_HR_ASSUMED", "value": "0.3 / 0.70",
     "applies_to": "Response -> OS", "cited_in_long2005": "NO", "note": "General oncology-literature association assumed."},
    {"category": "PFS modeling", "parameter": "pfs_age_weight", "value": 0.15,
     "applies_to": "Age -> PFS", "cited_in_long2005": "NO", "note": "-"},
    {"category": "PFS modeling", "parameter": "pfs_histology_weight (dict)", "value": "0.50-1.15 by histology",
     "applies_to": "Histology -> PFS multiplier", "cited_in_long2005": "NO",
     "note": "Reused unchanged from the Monk pipeline for cross-trial consistency; still an uncited placeholder."},
    {"category": "PFS modeling", "parameter": "pfs_response_weight / PFS_RESPONSE_HR_ASSUMED", "value": "0.3 / 0.65",
     "applies_to": "Response -> PFS", "cited_in_long2005": "NO", "note": "-"},
    {"category": "PFS modeling", "parameter": "pfs_toxicity_weight", "value": 0.1,
     "applies_to": "Toxicity burden -> PFS", "cited_in_long2005": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "orr_vs_ps", "value": -0.30,
     "applies_to": "Response propensity index", "cited_in_long2005": "NO", "note": "Direction consistent with standard oncology expectation; magnitude uncited."},
    {"category": "Response modeling", "parameter": "orr_vs_disease_status", "value": -0.35,
     "applies_to": "Response propensity index", "cited_in_long2005": "NO", "note": "Uses the trial-cited progression_rr as the severity signal, but the WEIGHT linking it to response propensity is self-proposed."},
    {"category": "Response modeling", "parameter": "tolerability_orr_weight", "value": 0.20,
     "applies_to": "Response propensity index", "cited_in_long2005": "NO", "note": "-"},
    {"category": "Response modeling", "parameter": "pfsr_vs_ps", "value": -0.35,
     "applies_to": "Stable-vs-progressive disease propensity index", "cited_in_long2005": "NO", "note": "-"},
    {"category": "Disease-status timing", "parameter": "months_since_dx distribution (recurrent patients)",
     "value": "Exponential(mean=12mo), capped at 60mo", "applies_to": "Recurrent-disease patients' time-since-diagnosis",
     "cited_in_long2005": "NO -- the RR-decay FUNCTION is cited (Table 1); the underlying distribution of patients across time values is not reported and is assumed.",
     "note": "-"},
    {"category": "Performance status", "parameter": "PS worsening (disease status)", "value": "progression_rr>0.75 at p=0.3",
     "applies_to": "Recently-recurrent patients' PS", "cited_in_long2005": "NO", "note": "-"},
    {"category": "Performance status", "parameter": "PS worsening (tolerability)", "value": "threshold 0.35, fraction 0.10",
     "applies_to": "Low-tolerability patients' PS", "cited_in_long2005": "NO", "note": "Reused from the Monk pipeline."},
    {"category": "Tolerability", "parameter": "cycle_completion_rate (CPT/CT)", "value": "0.25 / 0.35",
     "applies_to": "Fraction assumed to complete all 6 planned cycles", "cited_in_long2005": "NO",
     "note": "Paper reports median cycles received (3/4 of a max 6) but not the % completing all 6; this is an inference from that median, not a reported rate."},
    {"category": "Tolerability", "parameter": "tolerability formula constants", "value": "0.82, 0.5, 0.09, 0.18",
     "applies_to": "Tolerability / toxicity-penalty formula", "cited_in_long2005": "NO", "note": "Reused unchanged from the Monk pipeline."},
    {"category": "Toxicity", "parameter": "tox_risk formula weights", "value": "0.4/0.35/0.35/0.25/0.15",
     "applies_to": "Per-patient toxicity latent-risk multiplier", "cited_in_long2005": "NO", "note": "Reused unchanged from the Monk pipeline."},
    {"category": "Toxicity", "parameter": "tox_corr matrix", "value": "0.2 / 0.7 / 0.5",
     "applies_to": "Cross-toxicity correlation structure", "cited_in_long2005": "NO", "note": "Reused unchanged; only 3 broad categories (hematologic/infection/GI) are distinguishable given Long's coarser 19-category table."},
    {"category": "Cause of death", "parameter": "Treatment=1, Other/Unknown=2 (CT arm only)", "value": "see construction note above",
     "applies_to": "cause_of_death column", "cited_in_long2005": "PARTIALLY -- the 3 specific deaths ARE narrated in text; the remaining split (all other deaths -> 'Disease') is a default assumption, not a reported breakdown.",
     "note": "Long 2005 has no dedicated cause-of-death table like Monk's Table 3."},
    {"category": "Demographics", "parameter": "age_sd", "value": "(age_max-age_min)/4",
     "applies_to": "Age distribution spread", "cited_in_long2005": "NO", "note": "Range/4 heuristic, reused from the Monk pipeline."},
])

print("\n--- Self-proposed weight reference table (for later citation) ---")
print(weight_reference_table.to_string(index=False))
weight_reference_table.to_excel("long_correlation_weight_reference.xlsx", index=False)

combined_evaluation_long