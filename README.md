# Introduction 

# Getting Started





![design folder](https://github.com/microsoft/dstoolkit-anomaly-detection-ijungle/blob/main/docs/media/banner.png)

About this repository
============================================================================================================================================

This repository contains the implementation of the Anomaly Detection Accelerator which is the technique of identifying rare events or observations which can raise suspicions by being statistically different from the rest of the observations. Such “anomalous” behavior typically translates to some kind of a problem like a:

-   credit card fraud,
-   failing machine in a server,
-   a cyber-attack,
-   variation in financial transactions,
-   and so on.

Common Anomaly Detection techniques are difficult to implement on very large sets of Data. The Anomaly Detection Accelerator, leverages the iJungle technique from Dr Ricardo Castro, which solves this challenge, enabling anomaly detection on large sets of data.

Details of the accelerator
============================================================================================================================
-   This repository includes the implementation od the iJungle anomaly detection technique to be executed in an on-premise setting or in the cloud
-   Also it includes a tutorial notebook that guides its use leveraging Azure Machine Learning capabilities like parallel training, and parallel evaluation to be able to reach high volume data analysis.
-   It include examples of how to use it as notebooks in Azure Databricks


Prerequisites
============================================================================================================================

In order to successfully complete your solution, you will need to have access to and or provisioned the following:

-   Access to an Azure subscription
-   Access to an Azure Machine Learning Workspace with contributor rights

Getting Started
================================================================================================================================

iJungle can run on a single machine and in a distributed way for data intensive scenarios using Azure Machine Learning (AML) under Linux environment like Ubuntu.  We reccomend that it is used under an AML Workspace.

## Installation process
Once cloned the git repository, under `Anomaly Detection` folder execute:

`make all`

This is going to create iJungle whl file under `dist` folder and install it using `pip`.

## How to use it

Once installed, open `iJungle-tutorial.ipynb` and follow the notebook.

Contents
================================================================================================================================

| File/Folder   | Description                                                                                     |
|---------------|-------------------------------------------------------------------------------------------------|
| `notebooks`   | iJungle quick-start notebook(`iJungle-tutorial.ipynb`), including single & parallel processing  |
| `src/iJungle` | iJungle source codes                                                                            |
| `operation`   | iJungle source codes used for parallel processing                                               |
| `data`        | Sample datasets used in `notebooks`                                                             |


General Coding Guidelines
====================================================================================================================================================

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
