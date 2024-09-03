# Earth data science

![](https://img.shields.io/badge/Python-3.12-blue)
[![](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![](https://img.shields.io/badge/Year-2024/2025-yellow)
![](https://img.shields.io/badge/Status-In%20progress-green)
![](https://img.shields.io/badge/Institution-IPGP-red)


<img width="250px" src="images/image.png" align="left" style="padding: 5px 20px 0px 0px;"/>

This repository contains the materical for the class _Earth data science_ delivered at the [Institut de Physique du Globe de Paris](https://www.ipgp.fr/) for master students. The course is an introduction to scientific computing and the use of Python for solving geophysical problems. The course is mostly based on practical sessions where students will learn how to use Python to solve problems related to the Earth sciences with statistical and machine learning methods. The course and notebooks rely on the Python [scikit-learn](https://scikit-learn.org/stable/) library, [pandas](https://pandas.pydata.org/), [pytorch](https://pytorch.org/), and the [deep learning](https://www.deeplearningbook.org/) book by Ian Goodfellow, Yoshua Bengio and Aaron Courville. This course is a legacy of the course of the same name by [Antoine Lucas](http://dralucas.geophysx.org/). The lectures are taught by [Léonard Seydoux](https://sites.google.com/view/leonard-seydoux/accueil) and the practicals by [Antoine Lucas](http://dralucas.geophysx.org/), [Alexandre Fournier](https://www.ipgp.fr/~fournier/), [Éléonore Stutzmann](https://www.ipgp.fr/~stutz/), and [Léonard Seydoux](https://sites.google.com/view/leonard-seydoux/accueil). 

The goal of this course is to introduce students to the basics of scientific computing and to the use of Python for solving geophysical problems. The course mostly consists in practical sessions where students will learn how to use Python to solve problems related to the Earth sciences mith statistical and machine learning methods. The course and notebooks rely on the Python [scikit-learn](https://scikit-learn.org/stable/) library, [pandas](https://pandas.pydata.org/), [pytorch](https://pytorch.org/), and the [deep learning](https://www.deeplearningbook.org/) book by Ian Goodfellow, Yoshua Bengio and Aaron Courville.

The course contains 8 hour of lecture followed by 20 hours of practical sessions made with Jupyter notebooks. The lecture notes are available in the `lectures` folder and the practicals in the `labs` folder. You can find an introductory README file in each folder.

<div style="clear: both;"></div>

## Lectures

The lectures will fit within two sessions of four hours each. The following list of lectures is proposed in the different subfolders of the `lectures` folder.

1. __Introduction to machine learning__. This section introduces the use cases of machine learning in the Earth sciences and the basic concepts of supervised and unsupervised learning.
2. __Definitions__. This section introduces the basic definitions of machine learning, including the various notations and the different types of learning.
3. __Supervised machine learning: regression__. This section introduces the concept of regression and the different metrics used to evaluate the performance of a regression model.
4. __Supervised machine learning: classification__. This section introduces the concept of classification and the different metrics used to evaluate the performance of a classification model
5. __Deep learning: the multilayer perceptron__. This section introduces the concept of deep learning and the multilayer perceptron
6. __Deep learning: convolutional neural networks__. This section introduces the concept of convolutional neural networks
7. __Applications__. This section introduces the different applications of machine learning in the Earth sciences
8. __Unsupervised learning__. This section introduces the concept of unsupervised learning, relying a lot on the previously seen concepts
9. __Notebooks__. A brief introduction to the Jupyter notebooks and the Python programming language.

<br>

## Labs    

The following list of labs is proposed in the different subfolders of the `labs` folder.

0. [__Self-evaluation__](labs/0-self-evaluation/self-evaluation.ipynb) (1 hour). This lab is a self-evaluation of your Python skills. It is required to enroll in the course. A small solution will be delivered at the beginning of the lab session. 
1. [__River sensor calibration__](README.md) (4 hours). This lab allow to perform a first simple machine learning task: the calibration of a river sensor with supervised learning, where the goal is to predict the suspended sediment concentration from the turbidity of the water.
2. [__Earthquake location__](README.md) (~4 hours). In this lab, we will use Bayesian inference to locate the earthquake that occurred near the city of Le Teil in November 2019. We will also play around with prior distributions and see how they affect the posterior distribution.
3. [__Lidar data classification__](README.md) (~8 hours). In this lab, we will classify lidar cloud points into different classes using supervised machine learning tools. Since this is a more complex task, we will take more time to complete it.
4. [__Deep learning__](README.md) (~4 hours). In this lab, we will explore several deep learning architectures to perform several supervised tasks, including digit recognition, and volcano monitoring.

> The solution to the different labs will be proposed progressively during the course in the corresponding folders. Note that the solutions provided are not necessarily the best ones. The main idea of these sessions is for you to be overly curious and to try to find the solutions that best fit your needs, and your understanding of the problem. Some of you may complete the tasks at a faster pace than others, and we encourage you to help your peers during the labs, and also to explore further aspects of the problems that are not covered in the labs.

## Running the Jupyter labs 

### Python environment

The easiest way to run most notebooks of this course is to create a new Anaconda environment with the following set of commands. We decided not to go with an environment file to allow for more flexibility in Python versions.

The following lines create a new environment called `earth-data-science` without any package installed. Then, we install the most constrained packages first (namely, `obspy`) which will install the latest compatible version of `python`, `numpy` and `scipy`. Finally, we install the rest of the packages.

```bash
conda create -n earth-data-science
conda activate earth-data-science
conda install -c conda-forge obspy
conda install -c conda-forge numpy scipy matplotlib pandas jupyter scikit-learn cartopy ipywidgets rasterio seaborn
pip install tqdm 
pip install laspy
```

Once this is done, you must select the kernel `earth-data-science` in Jupyter to run the notebooks. Please inform your instructor if you have any problem with this.


### Execution

The notebooks can be either ran locally or on a remote server. The remote server is available at the following address: https://charline.ipgp.fr. You can log in with your IPGP credentials. Therein, you can apply clone to download the notebooks from this repository (e.g. `git clone https://github.com/leonard-seydoux/earth-data-science.git`). 

