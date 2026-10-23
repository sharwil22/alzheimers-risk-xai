# Data — OASIS-1 (cross-sectional)

`oasis_cross-sectional.csv` is the demographic, clinical and MRI-derived summary table of
**OASIS-1: Cross-sectional MRI Data in Young, Middle Aged, Nondemented and Demented Older Adults**,
from the Open Access Series of Imaging Studies — <https://sites.wustl.edu/oasisbrains/home/oasis-1/>.

| Column | Meaning |
|---|---|
| ID | Subject/session ID (`OAS1_xxxx_MR1`; `_MR2` = reliability re-scan) |
| M/F | Sex |
| Hand | Handedness (all right-handed) |
| Age | Age in years |
| Educ | Education level (1 = less than high school … 5 = beyond college) |
| SES | Socioeconomic status, Hollingshead index (1 = highest … 5 = lowest) |
| MMSE | Mini-Mental State Examination score (0–30) |
| CDR | Clinical Dementia Rating (0, 0.5, 1, 2) — used **only** to build the target |
| eTIV | Estimated total intracranial volume (mm³) |
| nWBV | Normalized whole-brain volume |
| ASF | Atlas scaling factor |
| Delay | Days between scans (re-scan rows only) |

The file has 436 rows: 416 subjects plus 20 re-scans. 235 subjects have a CDR rating.

## Source

The CSV is included in this repository so the pipeline runs out of the box. The original is distributed
by OASIS at <https://sites.wustl.edu/oasisbrains/> (OASIS-1). If you replace it, keep the exact name
`data/oasis_cross-sectional.csv`.

**Terms of use.** OASIS data are provided free for research and education. If you use them,
acknowledge OASIS and cite:

> D. S. Marcus, T. H. Wang, J. Parker, J. G. Csernansky, J. C. Morris and R. L. Buckner,
> "Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged,
> nondemented, and demented older adults," *J. Cogn. Neurosci.*, vol. 19, no. 9, pp. 1498–1507, 2007.

OASIS-1 principal investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; grants P50 AG05681, P01 AG03991,
P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382. DOI: 10.1162/jocn.2007.19.9.1498
