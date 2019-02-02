Setup On Mac
==========
https://github.com/Unity-Technologies/obstacle-tower-env を参考に。
pipenvは事前に準備してあるものとします。

## setup pipenv
ml-agents が python3.6のみのサポートなので、python3.6を使います。

```
mkdir obstacle_tower
cd obstacle_tower
pipenv --python 3.6
```

## Install Requirements
```
pipenv install jupyter tensorflow gym Pillow
```

### Unity ML-Agents v0.6
```
pipenv install 'git+https://github.com/Unity-Technologies/ml-agents.git#egg=ml-agents&subdirectory=ml-agents'
```

## Install Enviroments
### obstacletower_v1_osx
```
curl -LO https://storage.googleapis.com/obstacle-tower-build/v1/obstacletower_v1_osx.zip
unzip obstacletower_v1_osx.zip
rm obstacletower_v1_osx.zip
```

## Clone & Install obstacle-tower-env
```
git clone git@github.com:Unity-Technologies/obstacle-tower-env.git
pipenv install -e obstacle-tower-env
```

# Run Getting Start
```
PRJ_ROOT=$(pwd) pipenv run jupyter-notebook
```

open `obstacle-tower-env/examples/basic_usage.ipynb` notebook

modify 

```
env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=False)
```

↓

```
import os
env = ObstacleTowerEnv(f'{os.environ["PRJ_ROOT"]}/obstacletower', retro=False)
```

