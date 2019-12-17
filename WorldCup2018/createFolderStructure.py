import pathlib
pathlib.Path('./data').mkdir(parents=True, exist_ok=True)

pathlib.Path('./data/raw').mkdir(parents=True, exist_ok=True)
pathlib.Path('./data/processed').mkdir(parents=True, exist_ok=True)
pathlib.Path('./data/interim').mkdir(parents=True, exist_ok=True)
pathlib.Path('./data/external').mkdir(parents=True, exist_ok=True)

pathlib.Path('./src').mkdir(parents=True, exist_ok=True)
pathlib.Path('./src/data').mkdir(parents=True, exist_ok=True)
pathlib.Path('./src/features').mkdir(parents=True, exist_ok=True)
pathlib.Path('./src/models').mkdir(parents=True, exist_ok=True)
pathlib.Path('./src/visualization').mkdir(parents=True, exist_ok=True)

pathlib.Path('./docs').mkdir(parents=True, exist_ok=True)
pathlib.Path('./notebooks').mkdir(parents=True, exist_ok=True)
pathlib.Path('./references').mkdir(parents=True, exist_ok=True)
pathlib.Path('./reports').mkdir(parents=True, exist_ok=True)
pathlib.Path('./reports/figures').mkdir(parents=True, exist_ok=True)

pathlib.Path('./app').mkdir(parents=True, exist_ok=True)
pathlib.Path('./app/static').mkdir(parents=True, exist_ok=True)
pathlib.Path('./app/static/css').mkdir(parents=True, exist_ok=True)
pathlib.Path('./app/static/js').mkdir(parents=True, exist_ok=True)
pathlib.Path('./app/static/data').mkdir(parents=True, exist_ok=True)
pathlib.Path('./app/template').mkdir(parents=True, exist_ok=True)