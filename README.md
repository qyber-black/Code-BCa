# BCa - Brain Cancer Segmentation Python Module

> SPDX-FileCopyrightText: Copyright (C) 20222023 Ebtihal Alwadee <AlwadeeEJ@cardiff.ac.uk>, PhD student at Cardiff University\
> SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
>
> SPDX-License-Identifier: AGPL-3.0-or-later

This contains python code for brain cancer segmentation with deep learning, including a range of network
architectures.

The architectures consist of basic 3D UNets and the 3D LATUP-Net segmentation model. The LATUP-Net code
is a prototype, separate in the LATUP-Net folder for now.

Results of using these modules are available at https://qyber.black/ca/bca/results-bca-unet.

## Installation

Clone it with `git clone URL` where URL is the clone URL for the repository you wish to clone. The development
repo is at https://qyber.black/ca/bca/code-bca, but you may find it at mirrors as well.

The `requirements.txt` file contains the dependencies, to be installed with `pip3 install -r requirements.txt`.

## API Documentation

The API documentation is accessible at https://ca.qyber.dev/bca/code-bca.

The folder `bca` contains the python module for the actual functionalities. It has been documented using pdoc. Run
```pdoc -h localhost -p 8888 -n bca```
in the project's root folder and then view the documentation with your browser via the URL
```http://localhost:8888```
Of course you can use pdoc to generate the documentation differently (see the pdoc documentation at https://pdoc.dev/).

## Citation

E Alwadee, FC Langbein. **BCa - Brain Cancer Segmentation using Deep Learning**. V0.1. Software. 2023. 
https://qyber.black/ca/bca/code-bca
