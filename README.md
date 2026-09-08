# AIGCT — VEP Performance Chat

A Streamlit chat app where you ask natural-language questions about the benchmark
performance of **variant effect predictors (VEPs)**, and an LLM answers by
calling the AIGCT package's query methods and rendering a ranked results table.
AIGCT is a platform for systematically evaluating ML/AI models of variant effects across the spectrum of genomics-based precision medicine. It consists of python based API and a database of variant effect data organized into categories based on the source of the data. AIGCT code and documentation can be found here:
https://github.com/Huang-lab/AiGCT

This app only supports querying summary data. To access the complete set of available data use the AIGCT
platform directly.

The LLM (served free via OpenRouter doesn't know the numbers — it
translates each question into a call to one of two `aigct` query methods, then
summarizes the returned table (ranked by ROC AUC, descending).

The app url is: https://aigctchat.streamlit.app/

**Example questions:**
- *"Top performing VEPs for predicting pathogenicity of variants in the PTEN gene for cancer."* → per-gene method
- *"Top performing VEPs for predicting pathogenicity for Alzheimer's disease."* → per-task method (ADRD)


## Disease Areas(Tasks) Covered

| TASK_CODE | Meaning |
|-----------|---------|
| `CANCER`  | Cancer |
| `ADRD`    | Alzheimer's disease and related dementias |
| `CLINVAR` | ClinVar (general clinical pathogenicity) |
| `ASD`     | Autism spectrum disorder |
| `CHD`     | Congenital heart disease |
| `DDD`     | Developmental disorders (Deciphering Developmental Disorders) |


## Architecture

For a detailed overview of the system design, see [ARCHITECTURE.md](ARCHITECTURE.md).
