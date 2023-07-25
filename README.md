# A search for trends in spatially resolved debris discs at far-infrared wavelengths

Repository containing the manuscript, data (where available), methods, and results of the journal article "A search for trends in spatially resolved debris discs at far-infrared wavelengths". This work was published in Marshall et al. 2021, MNRAS 501, 6168.

```
.
├── analysis            <- Scripts used to produce the figures in the manuscript
├── data
│   ├── external    <- Photometry taken from Grant Kennedy's SDB as JSON files. These data files are available for download from SDB.
│   └── raw         <- Images taken from the Herschel Science Archive; a script is provided to automate download of the files used in this work.
├── .gitignore      <- Avoids uploading data, credentials, outputs, system files etc
├── LICENCE
├── models      <- Dust composition models used to interpret extended emission, etc. 
├── paper       <- Manuscript
│   ├── manuscript.pdf  <- Generated manuscript file, equivalent to the arXiv version
│   ├── manuscript.zip  <- Archive containing the latex files and figures used to produce manuscript.pdf
├── README.md         <- This file
└── requirements.txt  <- Python modules used to run scripts in analysis directory
```
