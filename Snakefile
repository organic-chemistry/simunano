rule run_all:
    input:
        [f"meg3_mock/multi_res1_{i}.fa" for i in range(3)]

round=3
common_params=" --fork_position --resolution 1 --draw_sample 20 --bckg meg3_res1 --correlation "


output_rep="/media/jarbona/jm_data/simunano/"


rule create_meg_3_res1_for_deconv:
  output:
    [output_rep+f"meg3_mock/multi_{i}_round_{round}.fa" for i in range(5,9)]
  params:
    standard="--simu_type multi --n_conf 1000 --time_per_mrt 5 --read_per_time 4 --no_mrt" + common_params
  run:
    for i in range(len(output)):
      out =output[i].replace(".fa","")
      shell(f"python src/simunano/simu_forks.py {params} --param data/meg3/params_res{round}_uni.json --prefix {out} --seed {i}")


rule create_small_mono_calibration:
    output:
        f"meg3_mock/small_calibration_round_{round}.fa"
    params:
        standard="--n_conf 1000 --time_per_mrt 1 --read_per_time 1 --no_mrt --ground_truth" + common_params
    run:
        out =f"{output}".replace(".fa","")
        shell(f"python src/simunano/simu_forks.py {params} --param data/meg3/params_res{round}.json --prefix {out}")



rule create_mono_precision_laurent:
    output:
        f"{output_rep}/meg3_mock/better_calibration_precision_corrected.fa"
    params:
        standard="--n_conf 10000 --time_per_mrt 1 --read_per_time 1  --fork_position --resolution 1 --draw_sample 20 --bckg meg3_res1 --no_mrt --ground_truth"
    run:
        out = f"{output}".replace(".fa","")
        shell(f"python src/simunano/simu_forks.py {params} --param data/meg3/params_res1.json --prefix {out}")



rule create_meg_3_res1_from_deconvolved_multi:
  output:
    [f"{output_rep}/meg3_mock/multi_full_{i}.fa" for i in range(5)]
  params:
  #1650
    standard="--simu_type multi --n_conf 1000 --time_per_mrt 5 --read_per_time 4  --fork_position --resolution 1 --draw_sample 20 --bckg meg3_res1 --no_mrt --ground_truth"
  run:
    for i in range(len(output)):
       out =output[i].replace(".fa","")
       shell(f"python src/simunano/simu_forks.py {params} --param data/meg3/params_res1.json --prefix {out} --seed {i}")

rule create_meg_3_res1_from_deconvolved_multi_test:
  output:
    [output_rep+f"meg3_mock/multi_{i}_round_{round}_deconvolved.fa" for i in range(5)]
  params:
    standard="--simu_type multi --n_conf 1000 --time_per_mrt 5 --read_per_time 4 --no_mrt" + common_params + " --ground_truth"
  run:
    for i in range(len(output)):
       out =output[i].replace(".fa","")
       shell(f"python src/simunano/simu_forks.py {params} --param data/meg3/params_res{round}.json --prefix {out} --seed {i}")

rule create_meg_3_res1_from_deconvolved_mono_test:
   output:
     [output_rep+f"meg3_mock/mono_{i}_round_{round}_deconvolved.fa" for i in range(1)]
   params:
     standard="--n_conf 10000 --time_per_mrt 1 --read_per_time 1 --no_mrt" + common_params + " --ground_truth"
   run:
     for i in range(len(output)):
       out =output[i].replace(".fa","")
       shell(f"python src/simunano/simu_forks.py {params} --param data/meg3/params_res{round}.json --prefix {out} --seed {i}")


rule create_test_learning:
    output:
        f"meg3_mock/test_learning_{round}.fa"
    params:
        standard="--n_conf 100 --time_per_mrt 1 --read_per_time 1 --no_mrt --ground_truth" + common_params
    run:
        out =f"{output}".replace(".fa","")
        shell(f"python src/simunano/simu_forks.py {params} --rfd --param data/meg3/params_res{round}.json --prefix {out}")

nsim = config.get("nsim",1000)
root_dir=config.get("root_dir","meg3_mock")

rule create_test_learning_multi:
    output:
        f"{root_dir}/learning_test.fa"
    params:
        standard=f"--n_conf {nsim} --time_per_mrt 1 --read_per_time 1 --no_mrt --simu_type multi --ground_truth --states" + common_params
    run:
        out =f"{output}".replace(".fa","")
        shell(f"python src/simunano/simu_forks.py {params} --rfd --param data/meg3/params_res{round}.json --prefix {out}")


rule create_test_simplified:
    output:
        f"{root_dir}/simplified_test.fa"
    params:
        standard=f"--n_conf {nsim} --time_per_mrt 1 --read_per_time 1 --no_mrt --simu_type simplified --ground_truth --states"\
                  " --fork_position --resolution 1 --draw_sample 20 --bckg meg3_res1 --average_distance_between_ori 100000"
    run:
        out =f"{output}".replace(".fa","")
        shell(f"python src/simunano/simu_forks.py {params} --rfd --param data/meg3/params_avg_simplified.json --prefix {out}")

rule create_test_clara_speeds:
    output:
        [f"{root_dir}/tau1_variable/speed_v_{v}/test.fa" for v in ["1","1.5","2","2.5","3"]] + \
        [f"{root_dir}/tau2_variable/speed_v_{v}/test.fa" for v in ["1","1.5","2","2.5","3"]]
    params:
        standard=f"--n_conf {nsim} --time_per_mrt 1 --read_per_time 1 --no_mrt --simu_type one_fork --ground_truth --states"\
                  " --fork_position --resolution 1 --draw_sample 20 --bckg meg3_res1 "
    run:
        for tau in [1,2]:
          for speed in ["1","1.5","2","2.5","3"]:
              out =f"{root_dir}/tau{tau}_variable/speed_v_{speed}/test"
              param = f"data/meg3/clara_speeds/params_res2_uni_tau{tau}_variable_all_fixed_v{speed}.json"
              shell(f"python src/simunano/simu_forks.py {params} --rfd --param {param} --prefix {out}")
