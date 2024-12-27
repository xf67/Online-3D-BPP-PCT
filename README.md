This repo stores my SJTU CS3308 final project.

I use the code from [Online-3D-BPP-PCT](https://github.com/alexfrom0815/Online-3D-BPP-PCT) and modify it to fit my needs. You can also refer to the original [README.md](./originalREADME.md) for more details.

### What I did

- Small bugs fixed of the original code
- Add a new data generation method to create continuous data by cutting for training
- Add another new data generator to read data from csv file for evaluation
- Add some new features in PctContinuous0 environment, e.g. preview, rearrange
- Add MCTS method in evaluation to enable the policy to change the order of boxes
- Add a bash script to run training for a list of containers
- Add code to select the best container for a given box list

### How to run

Training for one container:

```bash
python main.py --continuous --setting 2 --container-size "[100,100,100]" 
```

Training for multiple containers:

```bash
bash train_all.sh
```

Test with single container:

```bash
python evaluate.py --continuous --setting 2 --container-size "[100,100,100]" --load-model --model-path path/to/your/model --load-dataset --dataset-path path/to/your/dataset
```


### Sample outputs

Training output:

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

Evaluation output:

```

```