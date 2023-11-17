## Natural Language-conditioned Reinforcement Learning with Inside-out Task Language Development and Translation

**Abstract:** Natural language-conditioned reinforcement learning (RL) enables agents to follow human instructions. Previous approaches generally implemented languageconditioned RL by providing the policy with human instructions in natural language (NL) and training the policy to follow instructions. In this is outside-in approach, the policy must comprehend the NL and manage the task simultaneously. However, the unbounded NL examples often bring much extra complexity for solving concrete RL tasks, which can distract policy learning from completing the task.To ease the learnin g burden of the policy, we investigate an inside-out scheme for natural language-conditioned RL by developing a task language (TL) that is task-related and easily understood by the policy, thus reducing the policy learning burden. Besides, we employ a translator to translate natural language into the TL, which is used in RL to achieve efficient policy training. We implement this scheme as TALAR (TAsk Language with predicAte Representation) that learns multiple predicates to model object relationships as the TL. Experiments indicate that TALAR not only better comprehends NL instructions but also leads to a better instruction-following policy that significantly improves the success rate over baselines and adapts to unseen expressions of NL instruction. Besides, the TL is also an effective sub-task abstraction compatible with hierarchical RL.

## Setup
1. Please finish the following steps to install conda environment and related python packages
    - Conda Environment create
    ```bash
    conda create --name <env_name> --file spec-list.txt
    ```
    - Package install
    ```bash
    pip install -r requirements.txt
    ```
2. The environments used in this work require MuJoCo, CLEVR-Robot Environment and Bert as dependecies. Please setup them following the instructions:
    - Instructions for MuJoCo: https://mujoco.org/
    - Instructions for CLEVR-Robot Environment: https://github.com/google-research/clevr_robot_env
    - Instructions for Bert: https://huggingface.co/bert-base-uncased

## Using

We upload a small dataset for testing, please change the dataset before using TALAR.

#### Training generator of TALAR in kitchen environments:

```
python train_tl_kitchen.py
```


* The models will be save at `results/train/LanguageAbstractionTrainer`.

#### Training translator of TALAR:

```
python train_translator_kitchen.py --path results/train/LanguageAbstractionTrainer/<timestamp> --cpt-epoch 0
```

* The models of translator will be saved at `results/train/VAETranslator`.

#### Training goal-conditioned-policy of TALAR:

Before training the Goal-Conditioned-Policy (GCP), we need to train a TL translator using the process described in step 4. When the traning of TL translator model is completed, please place the model in the designated location:
```bash
<project_path>/models/
```
* Before beginning the training process, please ensure that you have downloaded bert-base-uncased from https://huggingface.co/bert-base-uncased and moved it to the designated location:
```bash
<project_path>/models/bert-base-uncased
```
* FrankaKitchen
    * Prepare TL translator: Rename the TL translator model
    ```bash
    cp <TL_TRANSLATOR_MODEL_PATH> <project_path>/models/kitchen_policy_ag_<SEED>
    ```
    * IFP (Instruction following policy): Run this command in shell
    ```bash
    python kitchen_train.py --seed <SEED>
    ```
    * The models of goal-conditioned-policy will be saved at `kitchen_model`.
    * The tensorboard log of goal-conditioned-policy will be saved at `kitchen_train`.
    * The evaluation result of goal-conditioned-policy will be saved at `kitchen_callback`.
* CLEVR-Robot
    * Prepare TL translator: Rename the TL translator model
    ```bash
    cp <TL_TRANSLATOR_MODEL_PATH> <project_path>/models/policy_ag_<SEED>
    ```
    * Instruction following policy: Run this command in shell
    ```bash
    python ball_train.py --seed <SEED>
    ```
    * The models of goal-conditioned-policy will be saved at `ball_model`.
    * The tensorboard log of goal-conditioned-policy will be saved at `ball_train`.
    * The evaluation result of goal-conditioned-policy will be saved at `ball_callback`.
