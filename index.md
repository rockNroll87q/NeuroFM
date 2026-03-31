---
layout: page
title: <a href="https://rocknroll87q.github.io/NeuroFM">NeuroFM</a>
---

# Abstract

Precision neuroimaging aims to deliver individualized assessments of brain health, yet a single structural MRI does not provide a scalable, multidimensional, quantitative summary of an individual’s current or future health. Existing approaches optimize task-specific objectives, yielding representations entangled with cohort- or disease-specific signals rather than capturing biologically grounded anatomical patterns.Here, we introduce NeuroFM, a foundation model trained exclusively on 100,000 healthy synthetic volumes to predict morphometric and demographic targets. Without exposure to disease-labelled data, NeuroFM organizes brain structure into population-level patterns encoding brain health differences. These representations transfer across neuroscience domains without adaptation and support simple linear readouts for clinical, cognitive, developmental, socio-behavioural, and image quality. Evaluated on 136,361 multi-cohort volumes, NeuroFM generalizes across domains and enables individual-level brain health profiling, estimating future dementia risk years before diagnosis. Together, these findings establish a disease-naïve foundation model for precision neuroimaging with potential to support quantitative brain health assessments across settings. Code, model, and demo are available on the [project website](https://rocknroll87q.github.io/NeuroFM/).

<p align="center">
<img src="./misc/Fig1 - v0.11.png" width="80%" />  
<figcaption>Figure 1 | General overview. a, Foundation model (NeuroFM) pre-trained on 100,000 AI-generated volumes (left) using supervised learning targets for age, total brain volume (TBV), ventricular volume, and sex (middle). NeuroFM acts in two modes: predictor which gives four brain health estimates (age, sex, TBV, and ventricular volume), and encoder which extracts 100+ brain health features from each MRI. In this work, we apply NeuroFM to a variety of tasks, including classification, regression, and qualitative clustering analysis (right). b, After pre-training, the model is applied across five biologically motivated task subgroups: clinical, cognitive, socio-behavioural, developmental, and image quality metrics (left to right). c, By combining subtasks at inference (left), NeuroFM can create comprehensive, individualized health reports for a single individual (middle), with explainable attribution maps to highlight relevant brain regions (right).</figcaption>
</p>


<!--<hr>
# Results

Visit the result [page](https://rocknroll87q.github.io/LOD-Brain/results#top) for more results.


<hr>
# Usage

Visit the relative [page](https://rocknroll87q.github.io/LOD-Brain/usage) to learn how to use `LOD-Brain` from source code, docker, or singularity.
-->

<hr>
# Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article {DibbleNeuroFM2026,
	author = {Dibble, Austin and Dalby, Connor and Sevegnani, Michele and Fracasso, Alessio and Lyall, Donald M and Harvey, Monika and Svanera, Michele},
	title = {NeuroFM: Toward Precision Neuroimaging with Foundation Models for Individualized Brain Health Estimation},
	elocation-id = {2026.03.27.26349489},
	year = {2026},
	doi = {10.64898/2026.03.27.26349489},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Precision neuroimaging aims to deliver individualized assessments of brain health, yet a single structural MRI does not yield a multidimensional, quantitative summary of an individual{\textquoteright}s current health or future risk. Existing approaches optimize task-specific objectives, yielding representations entangled with cohort- or disease-specific signals rather than capturing biologically grounded patterns of anatomical variation. Here, we introduce NeuroFM, a foundation model trained exclusively on 100,000 healthy synthetic volumes to predict morphometric and demographic targets. Without exposure to diagnostic labels, NeuroFM organizes brain MRIs into population-level patterns that encode meaningful brain health differences. These representations transfer across five neuroscience domains without adaptation and support simple linear readouts for clinical, cognitive, developmental, socio-behavioural, and image quality control. Evaluated on 136,361 real volumes spanning multiple cohorts, NeuroFM generalizes across domains and enables individual-level brain health profiling, estimating future dementia risk years before diagnosis. Together, these findings establish a disease-naive foundation model paradigm for precision neuroimaging. Code available at: https://rocknroll87q.github.io/NeuroFM/},
	URL = {https://www.medrxiv.org/content/early/2026/03/31/2026.03.27.26349489},
	eprint = {https://www.medrxiv.org/content/early/2026/03/31/2026.03.27.26349489.full.pdf},
	journal = {medRxiv}
}
```


<hr>
# Acknowledgments

We acknowledge the MVLS Advanced Research System (MARS) at the University of Glasgow for providing high-performance computing resources and technical support.

Some data used in the preparation of this article were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database. As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: [link](http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf). For up-to-date information, see adni.loni.usc.edu.

Some of the data used in the preparation of this article were obtained from the Neuroimaging in Frontotemporal Dementia (NIFD) dataset, part of the Frontotemporal Lobar Degeneration Neuroimaging Initiative (FTLDNI). Data collection and sharing for this project was funded by the Frontotemporal Lobar Degeneration Neuroimaging Initiative (National Institutes of Health Grant R01 AG032306). The study is coordinated through the University of California, San Francisco, Memory and Aging Center. FTLDNI data are disseminated by the Laboratory for Neuro Imaging at the University of Southern California. For up-to-date information on participation and protocol, see [link](http://memory.ucsf.edu/research/studies/nifd). 

Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

Data were provided [in part] by OASIS-3: Longitudinal Multimodal Neuroimaging: (Principal Investigators: T. Benzinger, D. Marcus, J. Morris); NIH P30 AG066444, P50 AG00561, P30 NS09857781, P01 AG026276, P01 AG003991, R01 AG043434, UL1 TR000448, R01 EB009352. AV-45 doses were provided by Avid Radiopharmaceuticals, a wholly owned subsidiary of Eli Lilly.

Data were [in part] obtained from the IXI dataset ([link](https://brain-development.org/ixi-dataset/)).

Data were [in part] provided by the 1000 Functional Connectomes Project (FCP). For access and usage information, see [link](https://fcon_1000.projects.nitrc.org).

<hr>

