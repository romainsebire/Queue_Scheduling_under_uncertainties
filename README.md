# RL project

## Install dependencies

```bash
pip install uv
```

```bash
.venv\Scripts\activate
```

```bash
uv pip install -r requirements.txt
```

## Run

```bash
python -m app.main
```

## Generate data
```bash
python -m app.InstanceGenerator
```

## Tensor Board - Ep_rew_mean

```bash
tensorboard --logdir=app/data/logs
```