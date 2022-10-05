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
    standard="--multi --n_conf 1000 --time_per_mrt 5 --read_per_time 4 --no_mrt" + common_params
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
    standard="--multi --n_conf 1000 --time_per_mrt 5 --read_per_time 4  --fork_position --resolution 1 --draw_sample 20 --bckg meg3_res1 --no_mrt --ground_truth"
  run:
    for i in range(len(output)):
       out =output[i].replace(".fa","")
       shell(f"python src/simunano/simu_forks.py {params} --param data/meg3/params_res1.json --prefix {out} --seed {i}")

rule create_meg_3_res1_from_deconvolved_multi_test:
  output:
    [output_rep+f"meg3_mock/multi_{i}_round_{round}_deconvolved.fa" for i in range(5)]
  params:
    standard="--multi --n_conf 1000 --time_per_mrt 5 --read_per_time 4 --no_mrt" + common_params + " --ground_truth"
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

rule create_test_learning_multi:
    output:
        f"meg3_mock/learning_test.fa"
    params:
        standard=f"--n_conf {nsim} --time_per_mrt 1 --read_per_time 1 --no_mrt --multi --ground_truth --states" + common_params
    run:
        out =f"{output}".replace(".fa","")
        shell(f"python src/simunano/simu_forks.py {params} --rfd --param data/meg3/params_res{round}.json --prefix {out}")
