# SelfLearningCars

2D top-down cars that learn to drive around a track through neuroevolution: a population of small
neural networks that improve by selection and mutation. Built with `pygame-ce` and `numpy`.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python Scripts/main.py   # start the app
pytest                   # run tests
ruff check .             # lint
```
