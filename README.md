# LinkingInStyle: Understanding learned features in deep learning models

This is the official repo for the paper: 
Linking in Style: Understanding learned features in deep learning models
Maren H. Wehrheim, Pamela Osuna-Vargas, and Matthias Kaschube.

## Installation and requirements:

(1) Download stylegan-xl model (imagenet256.pkl) from https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl and save it in models/sgxl/


## Folder structure:
```
├── data
│   ├── lnw
│   │   ├── train
│   │   └── test
│   └── seg
│       ├── train
│       │   └── dogs
│       │       ├──features
│       │       ├── imgs
│       │       ├── labels
│       │       └── w
│       └── test
│           └── dogs
│               ├──features
│               ├── imgs
│               ├── labels
│               └── w
├── models
│   ├── sgxl
│   ├── classif
│   ├── lnw
│   └── seg
```