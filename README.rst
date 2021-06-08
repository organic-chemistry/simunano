========
simuNano
========
Small scripts to simulati a BrdU incorporation experiments.

Description
===========

It can simulate either a single fork or a read with random origin.
A Brdu experiment is modeled by a starting time and then a fork charactarised
by its speed incorporate Brud with an increasing exponential during a pulse whose
length can be adjusted, followed by a decreasing exponential


Install
===========
conda create --name simunano --file environment.yml

Usage
===========
python src/simunano/simu_forks.py --multi --n_conf 400 --time_per_mrt 2  --fork_position --resolution 100 --draw_sample 4

Two main modes are available switched by the parameter --multi .
If not set the simulation will create reads with single forks starting at 1kb and going to
50 kb.
If set several forks will be present on the read. The density of origins being controlled
by *--average_distance_between_ori* 50000, which set the average distance between origins in bp.

--n_conf control the number of simulated set of origin

-- time_per_mrt control the number of simulated Brdu pulse per configuration (a pulse is charactarised by a starting
time drawed randomly between 0 and 3/5 of the maximum replication time)

--read_per_time control the number of truncated fiber extracted per Brdu pulse. (Right now the distribution is log normal)

All the parameter of the pulse and fork can be secified  using:
--parameter_file whose default is set to data/params.json . Right now all the fork on a fiber have the same parameters.
The distribution of the parameter can be uniform, fixed (Set to choices with only one choice), a choice between different value,
or from a list comming from an experiment, or from a pomegranatee gaussian mixture. See data/params.json for an example.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
