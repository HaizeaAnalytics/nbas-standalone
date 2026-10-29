# National Biodiversity Assessment System (NBAS) v2.1.1: Local Standalone Version 

This repository provides a local standalone version of the National Biodiversity Assessment System (NBAS) v2.1.1 used in [PLANR](https://planr.gov.au/). It enables users to run NBAS assessments independently of the PLANR website and web service.

The NBAS was developed as part of the Ecological Knowledge System (EKS). The EKS is a partnership between the Commonwealth Scientific and Industrial Research Organisation (CSIRO) and the Department of Climate Change, Energy, the Environment and Water (DCCEEW) to establish a transparent and authoritative source of information, biodiversity assessment and forecast capability for the Nature Repair Market.

The NBAS is a software and data package that provides a nationally consistent approach to assessing an area’s current contribution to biodiversity and forecasting the expected contribution to biodiversity following successful implementation of a given project within the Nature Repair Market. 

The standalone NBAS (version 2.1.1) produces the following metrics:
- Contribution to landscape connectivity scores
- Ecosystem significance scores
- Contribution to biodiversity persistence scores


Data inputs required to run the standalone version include:
- Ecosystem type identification number (The ecosystem identification key for each ecosystem type is available from the NBAS data link below)
- Starting and target ecosystem condition state reference number; and starting and forecast ecosystem condition scores. Please refer to the [NBAS settings: For use with the Nature Repair (Replanting Native Forest and Woodland Ecosystems) Methodology Determination 2025, Version 2.0](https://www.dcceew.gov.au/environment/environmental-markets/nature-repair-market/publication/nbas-settings-replanting-native-forest-woodland-ecosystems-method) or the [NBAS settings: for use with the Nature Repair (Enhancing Native Vegetation) Methodology Determination 2026, Version 1.0](https://www.dcceew.gov.au/environment/environmental-markets/nature-repair-market/publication/nbas-settings-enhancing-native-vegetation-method), depending on the specific method and project you wish to analyse.
- Project start date (`project_start_date`, DD-MM-YYYY), which selects the HCAS epoch used for condition and connectivity 
- An example payload is provided [here](https://raw.githubusercontent.com/HaizeaAnalytics/nbas-standalone/refs/heads/2.1.1/tests/test_payload.json).*  


For detailed documentation of NBAS methods, data sources, and outputs, please refer to:
 - [The Ecological Knowledge System (EKS) for the Nature Repair Market. Technical report Version 2](https://publications.csiro.au/publications/publication/PIcsiro:EP2026-4319)
 - [National Biodiversity Assessment System (NBAS) v2.1.1 Data Collection](https://data.csiro.au/collection/csiro:78573)
 - [Ecological Knowledge System: State and Transition Models](https://research.csiro.au/ecological-knowledge-system-ecosystem-models/)

## When to Use the Standalone NBAS Version

This standalone version of NBAS is intended for advanced users who need to run assessments on a local computer, at scale, or as part of automated workflows. It assumes familiarity with command-line tools, Docker, and basic data handling. If you're only assessing a small number of locations, or prefer a guided interface, we recommend using the [PLANR](https://planr.gov.au/) website, which provides access to NBAS with examples and visual outputs.

This repository does not include detailed examples or tutorials. It is designed for users who:

- Need to run NBAS offline, at over large areas, or in secure environments
- Want to integrate NBAS into larger data pipelines
- Are comfortable setting up and running Docker containers

## Computational Requirements and Software 

Minimum recommended specifications:

- **OS**: Linux, macOS, or Windows 11 (with WSL(Windows Subsystem for Linux))
- [CUDA enabled GPU with compute capability > 8](https://developer.nvidia.com/cuda-gpus)
- [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker](https://docs.docker.com/engine/install/)

##Setup Instructions for Linux and Windows

### Linux/macOS
#### 1. Install Docker

#### 2. Download (pull) docker image
	`sudo docker image pull haizeaanalytics/nbas-standalone:2.1.1`

	*Note: `haizeaanalytics/nbas-standalone:1.0.0` is the original release.*

#### 3. Run docker image and mount the home directory
	`sudo docker run -ti --gpus all -v $HOME:/data haizeaanalytics/nbas-standalone:2.1.1 /bin/bash`

*Note: This opens an interactive session inside the container with your home directory mounted in /data. Please ensure your payload is available in this directory. An example payload can be found [here](https://raw.githubusercontent.com/HaizeaAnalytics/nbas-standalone/refs/heads/2.1.1/tests/test_payload.json).*  

#### 4. Run NBAS in the docker container
`python3.11 main.py /data/path/to/payload.json`

#### 5. Copy result to home directory
`cp result/xxxx-result.json /data/path/to/result.json`

###Windows 11 
####Tested successfully with:
- Intel i7-13700H, 64 GB RAM
- Nvidia RTX 4060 GPU
- CUDA Toolkit + Visual Studio (with Python, Node.js workloads)
- Docker Desktop for Windows
- Git Bash or PowerShell
   
####1. Install Docker Desktop

####2. Enable WSL and sudo: Run 'wsl --update' in Git Bash

####3. Enable "sudo" in Settings > System > For Developers

####4. Download (pull) Docker image
	'sudo docker image pull haizeaanalytics/nbas-standalone:2.1.1'
	
####5. Create a folder on your local machine (e.g. C:\Users\your_username\nbas_payloads) and save your prepared input file (i.e. payload) in .json format to this folder (e.g. test_payload.json)
	* An example payload can be found [here](https://raw.githubusercontent.com/HaizeaAnalytics/nbas-standalone/refs/heads/2.1.1/tests/test_payload.json).*  
	
####6.	Open Windows PowerShell and run container. For example: 
 'docker run -ti --gpus all -v C:\Users\your_username:/data haizeaanalytics/nbas-standalone:2.1.1 /bin/bash'

	This command:
		- Starts the NBAS container
		- Mounts your local folder (C:\Users\your_username) into the container as /data
		- Opens a terminal inside the container
		
	You will receive a prompt like:root@5adb621718c3:/app#
	This means you're now inside the container and ready to run NBAS

####7. Inside the container, run:
	' root@5adb621718c3:/app#python3.11 main.py /data/nbas_payloads/test_payload.json'

####8. NBAS will generate a results file. For example: Results written to results/2025-06-17T10:06:09-results.json

####9. To move the results file from the container to your local folder, run:
	'cp results/2025-06-17T10:06:09-results.json /data/nbas_payloads/result.json

## Disclaimer

Biodiversity benefit scores will be reviewed by the Clean Energy Regulator as part of the project registration process. 

The NBAS code provides an estimate of biodiversity benefit only. We do not warrant the accuracy or completeness of any information contained in or calculated by the NBAS code, and users should use this information for estimation and guidance purposes only. 

The NBAS code should not be relied on to confirm eligibility for any particular project, program, scheme or entitlement, and users should undertake their own independent assessment and seek appropriate expert advice to confirm such eligibility.

Do not rely solely on the NBAS code to make decisions about your project, as there may be a number of other factors to take into account. 

The NBAS will be updated as the market evolves, knowledge improves, and technology changes over time. As this happens, the scores that NBAS calculates may change. 

##Attribution
This repository is developed and maintained by Haizea Analytics Pty Ltd. The NBAS code and NBAS data is based on research created under the Project "An Ecological Knowledge System for the Nature Repair Market", which was funded by DCCEEW. CSIRO is leading development of the EKS in partnership with DCCEEW. 

##Ownership of intellectual property rights 

This publication is based (in part) on research and data created as part of the Project. The Commonwealth owns the intellectual property rights in any such new material developed while carrying out the Project.

##Licensing 
The code in this repository is licensed under an MIT Licence (See MIT license for more details). Additional material published as part of the Project is licensed under a Creative Commons Attribution 4.0 International licence, except for the Commonwealth Coat of Arms, any logos, any material supplied by third parties, any material protected by a trademark and any images and/or photographs. More information on this CC BY license is set out at the Creative Commons website.

##Citation

Please cite as: Terasaki Hart, Drew; Ferrier, Simon; Hoskins, Andrew; McEvoy, John; Edirisinghe, Chamith; Larraondo, Pablo; Rosauer, Dan; Walker, Simon; Behzadnia, Sunny; Munroe, Samantha; & Murphy, Helen (2026): Ecological Knowledge System: National Biodiversity Assessment System (NBAS) local standalone version (version 2.1.1). v4. CSIRO. Service Collection. http://hdl.handle.net/102.100.100/735657?index=1 

##Questions 
For questions, please contact eks@dcceew.gov.au. 
