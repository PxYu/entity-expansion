# Corpus-based Set Expansion (CaSE)

This repo contains the scripts of experiments for SIGIR '19 short paper, "Corpus-based Set Expansion with Lexical Features and Distributed Representations".

In order to re-create similar results of the CaSE model, run the whole pipeline; otherwise, if you are just interested in the algorithm itself, please just look at `09-empirical.py`

## data preparation

Prior to running the pipeline, please pre-process raw corpora in the similar fashion as [SetExpan](https://github.com/mickeystroller/SetExpan) does. Place the corpora data in the following structure:

```
data
|
|------ dataset1 
|          |------ source
|          |------ intermediate
|------ dataset2
|          |------ source
|          |------ intermediate
...
|------ eval
  
```

`intermediate` folder should contain the output files of SetExpan's preprocessing module, as well as some intermediate data files this pipeline will generate.

