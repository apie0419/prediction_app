# prediction_app

## Environment

```
conda env create -f environment.yml -n <env_name>
```

## Usage

```
python app.py
```

## Export to exe

```
pyinstaller --hidden-import pkg_resources.py2_warn -D app.py
```

## For PC with no GPU

Need to add tensorflow/python folder from environment libaries and nvcuda.dll from C:\Windows\Systems32 to the APP folder