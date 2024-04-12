# Biomass ETL

[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

This ETL is meant to automate the process of Estimating Biomass. The maps can be accessed through [the forage tool ](https://et.waterpointsmonitoring.net/forage).

## Setup and Installation

The installation described here will make use of conda to ensure there are no package conflicts with
existing or future applications on the machine. It is highly recommended using a dedicated environment
for this application to avoid any issues.

### Recommended

Conda (To manage packages within the applications own environment)

### Environment

- Create the env

```commandline
conda env create -f environment.yml
```

Add a file named data.json in the base directory. This file will hold a json object containing
the secure information needed to run your application. Copy the following object into your file then
edit the values as described in each of the { ... } blocks. This file is in the .gitignore so it will
not be exposed publicly when you push and updates to your repo.

```json
{
  "service_account": "{service account to authenticate to Earth Engine}",
  "geoserve_url": "{geoserver REST URL}",
  "geoserver_user": "{geoserver publisher account username}",
  "geoserver_pwd": "{geoserver publisher account password}"
}
```

The order of script execution is:

1. data_extraction.py
2. gwr_model.py
3. rasterize.py
4. import_biomass.py

```

You will need a private_key.json file associated with service_account placed in the base directory.This file is also in the .gitignore so it will not be exposed publicly when you push and updates to your repo.

```

There is master_script.py to run all the other scripts.
