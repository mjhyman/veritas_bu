# 1 Installation
Since I have limited space on my linux machine's hard drive, I will be installing and working with this project in the octdata2 partition of our network drive. This example lives in `/autofs/cluster/octdata2/users/epc28/veritas`, and you can use it as a reference at any time if you need.

Just something to watchout for: I will be referring to the directory contianing my veritas project folder as the "base directory," which again, is `/autofs/cluster/octdata2/users/epc28/veritas`.

## 1.1 Install conda (miniconda)
Frist we have to install conda so we can later install mamba (basically a faster version of conda).  Again, we have limited space, so we're going to download and install miniconda which gives you access to the most basic conda functionalities without taking up the abhorrent amount of space so typical of a standard conda installation. Download this installer into your octdata2 folder (`/autofs/cluster/octdata2/users/$USER`).

To access the download page for the newest miniconda installer, you can just click on this link right [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers). The installer you want to download will be specific to the operating system you're running, so in my case, I went to the Linux Installers subsection, and downloaded the first installer on the list (Miniconda3 Linux 64-bit). There are two ways to do this.

### 1.1.1 Downloading via web browser
If you plan to download the miniconda installer from your web browser, just click on the blue link next to the desired installer and the download should begin. Then, move this file into your octdata2 folder, and move on to section 1.1.3.

### 1.1.2 Downloading via command line
If you willingly choose to download this installer via the command line, chances are, you already know how to do this. If not, you've come to the right place. On the installer page that I linked at the beginning of this section (1.1), right click on blue link for the installer you want and hit "copy link". Just for a quick sanity check, paste this into your search bar. If it takes you back to the same page you were just on, you didn't do it right. Right click the blue link again and press the *other* "copy link" button. If pasting it into the search bar starts a download, great. Kill the download and paste the link after the `wget` command in your terminal to initiate the download. Remember to install this into your oct2data folder. If you want the exact installer that I used, you can just paste this command into your terminal:

`wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh`

### 1.1.3  Running the installer
Great, now you've downloaded the miniconda installer. All you have to do now is run the installer with bash:

`bash Miniconda3-*.sh`

You'll have to press "enter" a bunch of times (or just hold the button), then it'll ask you to accept some kind of agreement. Agree to all of this, then, when it asks for a path, put whichever directory you want all of your conda related files to go into. In my case, my Linux machine's home directory is packed, so I just placed this in my directory of the octdata2 folder. The path I entered was `/autofs/cluster/octdata2/users/epc28/miniconda`. This installer will create the `miniconda` directory for you, so don't make it via the `mkdir` command, because the linux gods will be very angry at you and haunt your dreams forever. Wait for miniconda to install and tell it to run the init. When everything is done installing, fire up a new terminal window and close the ones you were just working in. (This is just to refresh the shell so you have conda). Go back to the directory you were just in (in octdata2), and continue with this tutorial.

### 1.1.4 What the heck is conda?
Conda is used to install and manage packages. To install a package with conda, use the following command, replacing `<packagename>` with the name of your desired package:

`conda install <packagename>`

## 1.2 Install mamba (micromamba)
The next thing we have to do is install mamba into our base conda environment so we don't have to deal with the sluggishness of conda anymore. Luckily, we can do this via conda! I've installed mamba by running:

`conda install mamba -c conda-forge -n base`

Agree to the question and wait for the installation to finish. This is a good time for a coffee break.

## 1.3 Create & activate the vesselsynth environment

Conda and mamba use essentally the same commands, so to create the environment for vesselsynth while at the same time downloading some essential packages, run:

`mamba create -n vesselsynth -c conda-forge -c pytorch python=3.10.9 ipython pytorch cudatoolkit=11.7 cupy cppyy -r micromamba`

* A breakdown of this command:
	* The `-n` flag tells mamba what the name of this new environment should be. Specifying `-n vesselsynth` means that we are going to make an enviroinment named vesselsynth. 
	* The `-c` flag tells mamba which "channels" to look in, in order to find the packages that we want to install. In a similar way, if you wanted to watch cartoons, you'd switch your TV to channel 47 (I think this is cartoon network)
		* Specifying `-c conda-forge` and `-c pytorch` tells mamba that we want to look in the channels called "conda-forge" and "pytorch" - both of which are pretty standard. If you you're trying to install a package that cannot be found by mamba or conda, try adding some more channels using the -c flag.
	* The `-r` flag tells mamba where we want this environment and all of its packages to live, or, where to find the "root" of mamba.
		* If we're currently in our octdata2 folder (`/autofs/cluster/octdata2/users/epc28`), then `-r micromamba` tells mamba to find its root in  `/autofs/cluster/octdata2/users/epc28/micromamba`

Sweet, now activate that bad larry!
`mamba activate vesselsynth`

## 1.4 Installing stuff with pip
We need to install some more of Yael's code into this environment, and we're going to do that via pip so we can just import the modules from our virtual environment without worrying about where everything is. Paste this whole thing into your terminal.

`pip install git+https://github.com/balbasty/jitfields.git@6076f915f6556c9733958a0aab28ed7ee93301e8 git+https://github.com/balbasty/torch-interpol git+https://github.com/balbasty/torch-distmap`

Now we'll install pytorch-lightning and tensorboard
`pip install pytorch-lightning tensorboard`

# 2 Veritas
Okay, now you've installed mamba, created and activated a virtual environment for running all of our code, and installed the necessary dependencies. Now it's time to get down to business - but first, it might be a good time for another coffee break. You're going to need to be awake!

## 2.1 Installing
The first step in getting this specific project running is actually getting the code onto your disk from github. You can either download it via any web browser by following this [link](https://github.com/EtienneChollet/veritas) and moving it to your base directory, or you can just use git. All LCN machines should have git preinstalled on them, so just paste this into your terminal:

`git clone git@github.com:EtienneChollet/veritas.git`

## 2.2 veritas Anatomy
Let's just take a look at the anatomy of this project. I've included the project tree below so you know what's going on. I've really only included the things that you'll need to worry about, so if all you want to do is generate some 3D OCT images, train, and test, don't worry about all the other stuff.

```
.
├── output
│   ├── models
│   ├── real_data
│   └── synthetic_data
├── README.md
├── scripts
│   ├── imagesynth
│   ├── retraining
│   ├── testing
│   ├── training
│   └── vesselsynth
├── torch.yaml
├── vesselseg
└── vesselsynth
```

You will spend most of your time working in the `scripts/` directory, which contains subdirectories for each of the 5 different functionalities of this project: `vesselsynth`, `imagesynth`, `training`, `retraining` and `testing`. In a normal workflow - from creating synthetic vessels to training and testing a computer vision algorithm - you will be entering these directories and running the scripts within them in that order. The only exception to this rule is the retraining scripts, which you don't really need to run unless you're not happy with your model's performance.

## 2.3 Scripts
Since I just said that you'll be working mostly in this directory, and because of the fact that I'm not a liar, let's first take a look at all of the different subdirs in here. I've made all of these subdirs pretty similar, so this workflow you're about to see roughly applies to all of them. 

### 2.3.1 veritas/scripts/vesselsynth
The vesselsynth script will generate the 3D geometry for all of the vasculature that we want to model. Most importantly, we need to define a size for the volumes we want to create, and the number of volumes that we want to create. We define the size by the number of pixels we want to use in order to represent our volume.

One of the most important things to understand, qualitatively, about the synthesis code, is that we first construct an empty volume, then determine the number of vascular trees that are to be placed in that volume, then tree by tree, we place each tree into the volume and define its geometries (radius, tortuosity, etc...) and branchpoints along the way.

Mostly everything else that is done by this script is stochastic - that is, the parameters used to construct the vasculature are computed by sampling random distrobutions. In Yael's original vesselsynth code, all sampling is done via log-normal distributions, whereas in my code, I've substituted in some uniform distributions. The reason I've done this is due to the fact that if you get unlucky in sampling (you WILL get unlucky when you're sampling 1000+ volumes), some trees will be sampled to have 61 thousand children or something crazy like that... There goes all of your computing resourses for the forseeable future.


Another important thing is this concept of the "root" in a vascular tree. Put simply, the "root" of a vascular tree is the largest (in diameter) branch of a single contiguious vessel tree. As we place vascular trees into our volume, the root is the first thing that pops into existence.





```
veritas/scripts/vesselsynth/
├── mlsc_job.sh
├── vessels_oct.py
└── vesselsynth_params.json
```

* `vessels_oct.py`
	* To generate the geometry of our synthetic vasculature using our local computer's hardware, we're going to be calling on the  `vessels_oct.py` script in the `veritas/scripts/vesselsynth/` directory. This is the script that takes everything we've downloaded/installed, along with the more specific modules in `veritas/vesselsynth` and puts it all together to make and save 3D representations of neurovasculature! 
* `vesselsynth_params.json`
	* Here is where you specify all of your parameters for the vessel synthesis code. The `vessels_oct.py` is good and dandy, but won't generate anything useful to you unless you specify your parameters! The vessel synthesis code speaks mostly in terms of means and variances, so if you see a list such as `"radius": [0.1, 0.02]`, that means that our average radius is 0.1 mm and the variance is 0.02 mm.
* `mlsc_job.sh`
	* This script should only be ran if you are synthesizing vessels on the cluster (which can be accessed via `ssh mlsc`). You really shouldn't need to do this - I've found that synthesizing locally works just fine. But, if you must, you can just run this from your base directory via `bash scripts/vesselsynth/mlsc_job.sh`

What I've outlined above is the general structure for all 5 functionalities in this project (all 5 subdirs of the `scripts` directory). As such, I will be abbreviating all of the following explanations.

### 2.3.2 Imagesynth
After you've created your vasculature, it's time to look at it before we go on to training. I call this "synthesizing OCT images." Synthesizing OCT images is by no means a necessary step in training your model, but it is highly recommended. By synthesizing images, you get to see exactly what your model will be learning from. In the `scripts` directory, there's a folder called `imagesynth` - let's look at it.

```
veritas/scripts/imagesynth/
├── imagesynth_params.json
├── imagesynth.py
└── synth.sh
```

The first step in visualizing your data is choosing which "label" data to actually synthesize the OCT images from. In the previous step (2.3.1 Imagesynth), we created many different kinds of nifti images () (labels, probabilities, skeletons, etc...), and saved them to wherever you specified in the `vesselsynth_params.json` file. Now, in imagesynth, we will take the "labels" and color them in to look like actual OCT images.

We are going to direct our imagesynth script to use those data made in vesselsynth in order to generate OCT images. The OCT images that are generated as a result of imagesynth will be saved in the `veritas/output/synthetic_data/exp*/` folder, under a new folder called `volumes`.

As such, the first step in running imagesynth is setting which label data to use. But not only that - we need to tell imagesynth how to color in these images. All of this can be done by editing the `imagesynth.json` file. Go into it and mess around!

Once you've set up all your parameters, you can run the actual synthesis code using `python3 scripts/imagesynth/imagesynth.py`. Again, this will generate the OCT images as niftis and tiffs in a new folder called `volumes` inside the path defined by `path` in the json file.

After running this code, you can either look at the nifti volumes or tiff images (2D, section taken from middle of nifti volume). If you choose to look at the tiffs, please make sure to open it up in ImageJ or Fiji, and NOT a preview software - the images WILL look different, and thus your model's version of reality will be distorted!!!

### 2.3.3 training
For the most part, there are 2 ways to run each 'function'. Let's say, for example, we want to train a new model. For this, we're going to be focusing on the `training` subdirectory inside the `scripts` folder. You'll see 4 files:

1. `train.py` - this the central script in this foler. You will call on this script any time you want to train your model. You can simply do `python3 train.py`, and you'll be on your way.
2. `train_params.json` - before you do any training of any kind, make sure you open this json file and manipulate the parameters to your desires. This is where you define things like the data you want to use, the model you want to use, and the model's architecture.
3. `train.sh` - If you're feeling dangerous and want to use the mlsc compute cluster for parallel processing (strongly reccomended), ssh into mlsc and simply call this shell script using bash: `bash train.sh`. This will call the `train.py` script, but also request some GPUs from mlsc.
	* Please for the love of god just make sure you change your user account (defined by the -A flag) in this bash script to the account of whoever asked you to run this. I would like to avoid being yelled at :).
4. `train.log` - this is just the log file from mlsc training. It's good to check this every once and a while to make sure everything is running well.

```
vesselseg/scripts/training/
├── train.py
├── train_params.json
├── train.sh
└── train.log
```

So essentially, there's 2 ways to run the training. You can either do it locally on your machine by calling the train.py script directly, or on the mlsc by calling the train.sh script using bash.
