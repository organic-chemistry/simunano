
simuNano
========
Small scripts to simulate a BrdU incorporation experiments (including noise).

Description
===========

It can simulate either a single fork or a read with random origin.
A Brdu experiment is modeled by a starting time and then a fork charactarised
by its speed incorporate Brud with an increasing exponential during a pulse whose
length can be adjusted, followed by a decreasing exponential.

The script to perform deconvolution is located in [Deconvolution](https://github.com/organic-chemistry/simunano/blob/main/notebooks/Extract_distribution.ipynb)


Install
===========

Require conda to be installed
```
git clone https://github.com/organic-chemistry/simunano.git
cd simunano
```
then
```
conda create --name simuNano --file environment.yml
```
or
```
conda create --name simuNano -c conda-forge -c bioconda  matplotlib pandas jupyterlab numpy pomegranate python=3.6  snakemake # r-essentials
conda activate simuNano
conda install pip
```

Usage
===========

```
conda activate simunano

python src/simunano/simu_forks.py --n_conf 10 --time_per_mrt 1 --read_per_time 1 --no_mrt --multi --ground_truth --states --fork_position --resolution 1 --draw_sample 20 --bckg meg3_res1 --correlation  --rfd --param data/meg3/params_res3.json --prefix test//learning_test
```

```
conda activate simunano
snakemake --force create_test_learning_multi --config nsim=1000 root_dir="meg_mock3/"
```

or for more options and a longuer description

```
python src/simunano/simu_forks.py --simu_type multi --n_conf 400 --time_per_mrt 2  --fork_position --resolution 100 --draw_sample 4
```

Two main modes are available switched by the parameter --multi .
If not set the simulation will create reads with single forks starting at 1kb and going to
50 kb.
If set several forks will be present on the read. The density of origins being controlled
by **--average_distance_between_ori** 50000, which set the average distance between origins in bp.


 * **--resolution** control the sampling along the read (in bp) right now the noise model has only been calibrated at 100 bp resolution
 * **--n_conf control** the number of simulated set of origin
 * **--time_per_mrt** control the number of simulated Brdu pulse per configuration (a pulse is charactarised by a starting time drawed randomly between 0 and 3/5 of the maximum replication time)
 * **--read_per_time** control the number of truncated fiber extracted per Brdu pulse. (Right now the distribution is log normal)

All the parameter of the pulse and fork can be secified  using:
 * **--parameter_file** whose default is set to data/params.json . Right now all the fork on a fiber have the same parameters.
The distribution of the parameter can be uniform, fixed (Set to choices with only one choice), a choice between different value,
or from a list comming from an experiment, or from a pomegranatee gaussian mixture. See data/params.json for an example.



it creates by defaults three files, whose prefix can be set by **--prefix** (Right now you must create a directory)
  * a *.fa* file which alternate unique id and the percent of Brdu incorporation along the reads
  * a  *_paramters.txt* file which contain all the parameters of the forks
  * a  *_all_speeds.txt* file which contain the speeds of all the forks on the reads
Additionnaly it can output:
  * with **--ground_truth** a *_gt.fa* which contain the same information as the *.fa* file but without noise
  * with **--fork_position** a file which contain all the positions of the ascending part of the fork as well as their direction
  * with **--draw_sample 20 ** it will create a pdf with 20 readn to inspect the outputs



For experiment with pauses
==========================

it is also possible to run specific configuration specified in a file (se example ifli in example/conf.txt)

## at 1 bp resolution

```
python src/simunano/simu_forks.py --conf ./example/conf.txt --simu_type multi --time_per_mrt 10 --read_per_time 1 --draw 20 --whole_length --length 100000 --correlation --bckg meg3_res1 --param data/meg3/params_res3_uni.json --prefix tmp/test --resolution 1
```

## at 100 bp resolution
```
python src/simunano/simu_forks.py --conf ./example/conf.txt --simu_type multi --time_per_mrt 10 --read_per_time 1 --draw 20 --whole_length --length 100000 --correlation --bckg meg3 --param data/meg3/params_res3_uni.json --prefix tmp/test --resolution 100
```
On each line there are two arrays. The first array is the origin (position in bp and time of firing in minute)
The second one is the (position of the pause, duration of pauses)


Or to randomly draw pauses use the argument add_pauses which specify either only one time t and the time of pause will be between 0 and 2t or two times t1 and t2 and the time of pause will be between t1 and t2 :

```
python src/simunano/simu_forks.py  --simu_type multi --time_per_mrt 10 --read_per_time 1 --draw 20 --whole_length --length 100000 --correlation --bckg meg3 --param data/meg3/params_res3_uni.json --prefix tmp/test --resolution 100 --add_pauses 10 11 --n_conf 20
```


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
