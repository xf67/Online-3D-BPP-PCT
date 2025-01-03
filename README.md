This repo stores my SJTU CS3308 final project.

I use the code from [Online-3D-BPP-PCT](https://github.com/alexfrom0815/Online-3D-BPP-PCT) and modify it to fit my needs. You can also refer to the original [README.md](./originalREADME.md) for more details.

### What I did

- Small bugs fixed of the original code
- Add a new data generation method to create continuous data by cutting for training
- Add another new data generator to read data from csv file for evaluation
- Add some new features in PctContinuous0 environment, e.g. preview, rearrange to suit the MCTS method
- Add MCTS method in evaluation to enable the policy to change the order of boxes
- Add a bash script to run training for a list of containers
- Add method to select the best container for a given container list, and add code to evaluate the policy with container selection
- Add code to draw the statistic of different types of evaluation results

### How to run

#### Training for one container:

```bash
python main.py --continuous --setting 2 --container-size "[100,100,100]"  --total-updates 7000
```

Be careful that the container size should be input as a string, not a list.

The training code will exit when reach the maximum number of updates. 
My method to exit the training is to break the loop in the `train_n_steps` function.
And that will cause a problem that different processes will stop at different times, 
and you will see a EOF error in that case. 
But it will not affect the training, and you can just ignore it.

#### Training for multiple containers:

```bash
bash train_all.sh
```

You can modify the `train_all.sh` to fit your needs.

#### Generate evaluation data:

```bash
python pct_envs/PctContinuous0/generate_eval_data.py
```

The default setting is [100,100,100] container size. You can change it by modifying the script.

#### Test with single container:

```bash
python evaluation.py --evaluate --continuous --setting 2 --container-size "[100,100,100]" --load-model --model-path path/to/your/model --load-dataset --dataset-path path/to/your/dataset --evaluation-episodes 10
```

You can provide dataset in the `.pt` format or `.csv` format. 
I have revised the original code to support both formats.

```bash
python draw/eval_one_bin_statistic.py
```

This is the related code to draw the statistic of the evaluation results. You may change the log path in the script to fit your needs.

#### Test with MTCS:

```bash
python evaluation.py --evaluate --continuous --setting 2 --container-size "[100,100,100]" --load-model --model-path path/to/your/model --load-dataset --dataset-path path/to/your/dataset --mcts --evaluation-method MCTS --evaluation-episodes 10 --num-simulations 50 --prev-size 5
```

My MCTS code is not very efficient, for it can only run with 1 CPU core, no parallelization.
So it is quite slow if you use `.pt` format dataset, but it is much faster if you use `.csv` format dataset.
The default setting is that the model can preview 3 boxes and do 10 simulations, you can change it by `--num-simulations` and `--prev-size`.

#### Test with multiple containers:

```bash
python multi_box_eval.py  --evaluate --continuous --load-model --model-path logs/experiment/setting2_discrete.pt --load-dataset --dataset-path dataset/task3.csv
```
You may need to modify the `container_sizes` and `model_paths` in the script to fit your needs.
And you can redirect the output to a file to save it and then draw the statistic.

```bash
python draw/eval_multi_bin_statistic.py
```

This is the related code to draw the statistic of the evaluation results. You may change the log path in the script to fit your needs.

#### Generate the theoretical max utilization:

```bash
python draw/get_theo_max.py
```

This code will generate the theoretical max utilization and the container type distribution, then output a pic to show the result.

### Sample outputs

#### Training output:

```
Time version: t100100100-2024.12.27-22-53-37 is training
Updates 10, num timesteps 3520, FPS 663
Last 10 training episodes: mean/median reward 3.3/3.4, min/max reward 1.7/4.5
The dist entropy 3.66552, the value loss 0.92524, the action loss 1.25103
The mean space ratio is 0.328969998532013, the ratio threshold is0.44621245457519526

Time version: t100100100-2024.12.27-22-53-37 is training
Updates 20, num timesteps 6720, FPS 672
Last 10 training episodes: mean/median reward 4.7/4.7, min/max reward 3.5/6.4
The dist entropy 3.69544, the value loss 8.85517, the action loss -3.76109
The mean space ratio is 0.46868510469226077, the ratio threshold is0.6440035000335693

Time version: t100100100-2024.12.27-22-53-37 is training
Updates 30, num timesteps 9920, FPS 669
Last 10 training episodes: mean/median reward 3.3/3.5, min/max reward 0.4/5.3
The dist entropy 3.70419, the value loss 0.59459, the action loss 0.63404
The mean space ratio is 0.3271801511063462, the ratio threshold is0.6440035000335693

Time version: t100100100-2024.12.27-22-53-37 is training
Updates 40, num timesteps 13120, FPS 662
Last 10 training episodes: mean/median reward 2.8/2.8, min/max reward 1.2/4.1
The dist entropy 3.71242, the value loss 0.30933, the action loss 1.28316
The mean space ratio is 0.2806638991913605, the ratio threshold is0.6440035000335693

Time version: t100100100-2024.12.27-22-53-37 is training
Updates 50, num timesteps 16320, FPS 667
Last 10 training episodes: mean/median reward 4.1/4.3, min/max reward 2.1/4.9
The dist entropy 3.61501, the value loss 1.88370, the action loss -1.47604
The mean space ratio is 0.4147993352905731, the ratio threshold is0.6440035000335693

Time version: t100100100-2024.12.27-22-53-37 is training
Updates 60, num timesteps 19520, FPS 671
Last 10 training episodes: mean/median reward 4.5/4.3, min/max reward 3.3/5.6
The dist entropy 3.66147, the value loss 0.41567, the action loss 0.97175
The mean space ratio is 0.44603931619104004, the ratio threshold is0.6440035000335693

Time version: t100100100-2024.12.27-22-53-37 is training
Updates 70, num timesteps 22720, FPS 666
Last 10 training episodes: mean/median reward 4.0/3.9, min/max reward 2.2/5.8
The dist entropy 3.69762, the value loss 0.45160, the action loss -0.28444
The mean space ratio is 0.4028864621266937, the ratio threshold is0.6440035000335693

Time version: t100100100-2024.12.27-22-53-37 is training
Updates 80, num timesteps 25920, FPS 664
Last 10 training episodes: mean/median reward 3.7/3.9, min/max reward 1.9/5.2
The dist entropy 3.62914, the value loss 1.51929, the action loss -1.47725
The mean space ratio is 0.3739086495017318, the ratio threshold is0.6440035000335693

Time version: t100100100-2024.12.27-22-53-37 is training
Updates 90, num timesteps 29120, FPS 663
Last 10 training episodes: mean/median reward 4.9/4.9, min/max reward 2.3/6.3
The dist entropy 3.63941, the value loss 0.61279, the action loss 1.65045
The mean space ratio is 0.4872339105211486, the ratio threshold is0.6440035000335693

Time version: t100100100-2024.12.27-22-53-37 is training
Updates 100, num timesteps 32320, FPS 663
Last 10 training episodes: mean/median reward 4.3/4.5, min/max reward 2.0/5.7
The dist entropy 3.62462, the value loss 0.76673, the action loss 0.26708
The mean space ratio is 0.43292918640132133, the ratio threshold is0.6440035000335693

```

This is the first 100 updates of training.

#### Evaluation output:

I recently made a small revision of the evaluation code, and it will output the current episode's order name instead of the episode number, if you use the csv format dataset and provide the `sku_code`.
So the output will be a little different from below.

```
Episode 0 ends.
Mean ratio: 0.755, length: 25.0
Episode ratio: 0.755, length: 25
Episode 1 ends.
Mean ratio: 0.839, length: 32.0
Episode ratio: 0.923, length: 39
Episode 2 ends.
Mean ratio: 0.8489999999999999, length: 33.666666666666664
Episode ratio: 0.869, length: 37
Episode 3 ends.
Mean ratio: 0.8232499999999999, length: 31.0
Episode ratio: 0.746, length: 23
Episode 4 ends.
Mean ratio: 0.8367999999999999, length: 32.8
Episode ratio: 0.891, length: 40
Episode 5 ends.
Mean ratio: 0.8423333333333333, length: 33.5
Episode ratio: 0.87, length: 37
Episode 6 ends.
Mean ratio: 0.8389999999999999, length: 33.42857142857143
Episode ratio: 0.819, length: 33
Episode 7 ends.
Mean ratio: 0.83725, length: 33.25
Episode ratio: 0.825, length: 32
Episode 8 ends.
Mean ratio: 0.8448888888888889, length: 33.666666666666664
Episode ratio: 0.906, length: 37
Episode 9 ends.
Mean ratio: 0.8465, length: 33.7
Episode ratio: 0.861, length: 34
...
```

This test is run with the pretrained model and given dataset by the original author.
The model is `setting2_discrete.pt` and the dataset is `setting123_discrete.pt`.
You can find these in the original repo.

#### Evaluation with MCTS output:

```
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:38<00:00,  3.86s/it]
MCTS finished:  [1, 2, 0] time:  40.0873544216156
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.78s/it]
MCTS finished:  [0, 2, 1] time:  39.250012159347534
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.78s/it]
MCTS finished:  [2, 1, 0] time:  39.26024270057678
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [2, 0, 1] time:  39.402368783950806
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [2, 0, 1] time:  39.39738893508911
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.78s/it]
MCTS finished:  [2, 0, 1] time:  39.29627180099487
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [1, 2, 0] time:  39.40318465232849
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [0, 1, 2] time:  39.37830066680908
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.77s/it]
MCTS finished:  [0, 1, 2] time:  39.2023241519928
Episode 0 ends.
Mean ratio: 0.83, length: 26.0
Episode ratio: 0.83, length: 26
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [1, 0, 2] time:  39.337443351745605
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.75s/it]
MCTS finished:  [0, 2, 1] time:  38.96331810951233
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [2, 1, 0] time:  39.3866331577301
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [2, 0, 1] time:  39.39234256744385
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [1, 0, 2] time:  39.330618381500244
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [2, 0, 1] time:  39.39584755897522
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [0, 1, 2] time:  39.38414907455444
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:38<00:00,  3.80s/it]
MCTS finished:  [0, 1, 2] time:  39.48703336715698
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.78s/it]
MCTS finished:  [2, 0, 1] time:  39.237510681152344
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:37<00:00,  3.79s/it]
MCTS finished:  [1, 2, 0] time:  39.40110635757446
100%|██████████████████████████████████████████████████████████████████████████| 10/10 [00:36<00:00,  3.64s/it]
MCTS finished:  [0, 1, 2] time:  37.87047719955444
Episode 1 ends.
Mean ratio: 0.8115, length: 29.0
Episode ratio: 0.793, length: 32
...
```

This test is run with the pretrained model and given dataset by the original author.
Just as I showed above.

Also, as you can see, the MCTS is very slow.
But it is much faster if you use `.csv` format dataset, like below:

```
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:03<00:00,  9.28it/s]
MCTS finished:  [3, 0, 1, 2, 4] time:  3.2376816272735596
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 10.95it/s]
MCTS finished:  [0, 1, 2, 3, 4] time:  2.743513584136963
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 10.33it/s]
MCTS finished:  [1, 2, 4, 0, 3] time:  2.9087791442871094
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:03<00:00,  9.07it/s]
MCTS finished:  [0, 1, 2, 3, 4] time:  3.310004472732544
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:03<00:00,  9.24it/s]
MCTS finished:  [4, 0, 3, 1, 2] time:  3.2507545948028564
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 10.62it/s]
MCTS finished:  [3, 0, 2, 1, 4] time:  2.829223871231079
Episode 0 ends.
Mean ratio: 0.6660068286705015, length: 28.0
Episode ratio: 0.6660068286705015, length: 28
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 13.39it/s]
MCTS finished:  [2, 0, 1, 3, 4] time:  2.2435970306396484
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:01<00:00, 17.27it/s]
MCTS finished:  [0, 1, 2, 3, 4] time:  1.739992618560791
Episode 1 ends.
Mean ratio: 0.6742499729827117, length: 17.0
Episode ratio: 0.6824931172949219, length: 6
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 12.71it/s]
MCTS finished:  [2, 0, 1, 3, 4] time:  2.3638875484466553
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 10.70it/s]
MCTS finished:  [3, 0, 1, 2, 4] time:  2.805931806564331
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 10.83it/s]
MCTS finished:  [0, 1, 2, 3, 4] time:  2.7748546600341797
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:03<00:00,  9.74it/s]
MCTS finished:  [4, 0, 1, 3, 2] time:  3.084087610244751
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:01<00:00, 17.88it/s]
MCTS finished:  [0, 1, 4, 2, 3] time:  1.6816351413726807
Episode 2 ends.
Mean ratio: 0.6684844309373474, length: 18.0
Episode ratio: 0.6569533468466187, length: 20
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 13.22it/s]
MCTS finished:  [1, 3, 0, 2, 4] time:  2.2724292278289795
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 12.17it/s]
MCTS finished:  [2, 1, 0, 3, 4] time:  2.4690372943878174
100%|██████████████████████████████████████████████████████████████████████████| 30/30 [00:01<00:00, 24.11it/s]
MCTS finished:  [0, 1, 2, 3, 4] time:  1.2475697994232178
Episode 3 ends.
...
```

This dataset is generated by the `generate_eval_data.py` script, and the model is trained with the code talked above.