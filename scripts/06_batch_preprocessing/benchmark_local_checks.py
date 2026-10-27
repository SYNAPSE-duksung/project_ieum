import importlib.util,time,json,statistics
from pathlib import Path
spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('10_run_candidate2_batch.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.SOURCE_LAYOUT='drive-path'
root=Path(r'H:\.shortcut-targets-by-id\15z9OYaGiLG_KO6Fuatrkgs1DNbD-D50q\ieum_project\00_data')
rows=[x for x in r.read(root/'batch_manifests/candidate2_batch_plan.csv') if x['batch_id']=='BATCH_010']
results=[]
for trial in range(3):
    for mode in (['old','new'] if trial%2==0 else ['new','old']):
        start=time.perf_counter()
        r.check_space(rows,root/'mini_dataset',Path(r'D:\ieum_local_pilot\batch_runs'),inspect_sources=mode=='old')
        if mode=='new':
            for kind,path in r.paths_for(rows[0],root/'mini_dataset').items():r.download(rows[0],kind,path)
        results.append(dict(trial=trial,mode=mode,seconds=time.perf_counter()-start))
        print(json.dumps(results[-1]),flush=True)
print(json.dumps({'recordings':len(rows),'old_median_seconds':statistics.median(x['seconds'] for x in results if x['mode']=='old'),'new_median_seconds':statistics.median(x['seconds'] for x in results if x['mode']=='new'),'note':'cached metadata comparison; excludes WAV decoding, VAD, download and upload'}),flush=True)
