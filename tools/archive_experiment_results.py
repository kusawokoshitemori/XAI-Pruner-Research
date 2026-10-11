import argparse,collections,csv,hashlib,json,math,pathlib,shutil,sys
import datetime
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Archive compact experiment results without checkpoints or raw tensors')
parser.add_argument('--as-of-date', default=datetime.date.today().isoformat())
ASOF=parser.parse_args().as_of_date
datetime.date.fromisoformat(ASOF)
OUT=ROOT/'results/runs'
HASH_CACHE={}

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,np.floating)):x=x.item()
 if isinstance(x,float) and not math.isfinite(x):return 'NaN' if math.isnan(x) else '+Inf' if x>0 else '-Inf'
 return x

def write(p,obj):p.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvrows(p):
 if not p.exists():return []
 return list(csv.DictReader(p.open()))
def istrue(x):return str(x).lower()=='true'
def finite(x):
 try:return math.isfinite(float(x))
 except (TypeError,ValueError):return False

def rel(p):return str(p.relative_to(ROOT))
def source(p):
 st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
 if key not in HASH_CACHE:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  HASH_CACHE[key]=h.hexdigest()
 return dict(path=rel(p),bytes=st.st_size,sha256=HASH_CACHE[key],mtime_ns=st.st_mtime_ns)

def extract_runs():
 return sorted([p.parent for pattern in ['diag_logs','diag_logs_recheck','diag_logs_observation']
                for p in (ROOT/pattern).glob('*/rank*/metadata.json')])

runs=extract_runs();owners={}; pruners={}
for r in runs:
 m=json.loads((r/'metadata.json').read_text());o=m.get('args',{}).get('output_dir')
 if o:owners[str((ROOT/o).resolve())]=max(owners.get(str((ROOT/o).resolve()),''),m['run_id'])
 if m.get('script','').startswith('prune') and o:
  pruners[str((ROOT/o).resolve())]=m['run_id']

index=[]
for r in sorted(runs,key=lambda r:r.parent.name):
 meta=json.loads((r/'metadata.json').read_text());rid=meta['run_id'];a=meta.get('args',{});dst=OUT/rid;dst.mkdir(parents=True,exist_ok=True)
 metrics=dict(run_id=rid,snapshot_as_of=ASOF,script=meta.get('script'),raw_directory=rel(r),event_counts={},observed_events=0)
 selected={}; health=collections.defaultdict(int);zeros=collections.defaultdict(list);ea=[];modules=collections.defaultdict(list);error=[];last_batch=-1
 with (r/'events.jsonl').open() as f:
  for line_no,l in enumerate(f,1):
   if not l.strip():continue
   try:e=json.loads(l)
   except json.JSONDecodeError:
    error.append(dict(line=line_no,reason='incomplete/invalid JSON; ignored in snapshot'));continue
   name=e['event'];metrics['event_counts'][name]=metrics['event_counts'].get(name,0)+1;metrics['observed_events']+=1;last_batch=max(last_batch,e.get('batch_id',-1))
   if name in ['final_consistency','model_build','final_evaluation_after_pruning_before_finetune','timing','execution_failed','ea_config','score_aggregation_comparison','filter_runtime_config','component_size_definition','protect_check']:
    selected[name]=dict(source_line=line_no,data={k:v for k,v in e.items() if k not in ['run_id','rank','seq','time_ns','phase','batch_id','event','status']})
   if name=='layernorm_health':
    for k,v in e['nonfinite_counts'].items():health[k]+=v
    if e.get('zero_denominator_count'):zeros[e.get('module_id','unknown')].append(dict(batch=e['batch_id'],count=e['zero_denominator_count'],source_line=line_no))
   if name=='ea_generation':ea.append(e)
   if name in ['residual_add','gate','filter_rescale','linear_lrp','conv2d_lrp']:modules[(name,e.get('module_id','unknown'))].append(e)
 metrics['last_observed_batch_id']=last_batch;metrics['key_events']=selected
 if health:metrics['layernorm_health']=dict(nonfinite_element_counts=dict(health),zero_output_observations=dict(zeros))
 if error:metrics['parse_notes']=error
 rows=csvrows(r/'score_batches.csv');metrics['score_collection']=dict(batch_count=len(rows),sample_count=sum(int(x['batch_size']) for x in rows))
 ac=csvrows(r/'score_accumulation.csv');metrics['score_collection']['nonfinite_new_scores']=[dict(batch=int(x['batch_id']),group=x['group'],nan=int(x['batch_nan_count']),posinf=int(x['batch_posinf_count'])) for x in ac if int(x['batch_nan_count']) or int(x['batch_posinf_count'])]
 npzs={}
 for p in sorted(r.glob('scores_*.npz')):
  with np.load(p,allow_pickle=False) as z:
   npzs[p.name]={k:dict(shape=list(z[k].shape),nan=int(np.isnan(z[k]).sum()),posinf=int(np.isposinf(z[k]).sum()),neginf=int(np.isneginf(z[k]).sum())) for k in z.files}
 if npzs:metrics['stored_score_array_health']=npzs
 attempts=csvrows(r/'ea_attempts.csv');ops={}
 for op in ['crossover','mutation']:
  rr=[x for x in attempts if x['operator']==op]
  if rr:ops[op]=dict(attempts=len(rr),legal=sum(istrue(x['legal_before_revise']) for x in rr),parent_pool_duplicate=sum(istrue(x['already_in_parent_pool']) for x in rr),kids_duplicate=sum(istrue(x['already_in_operator_kids']) for x in rr),novel_candidate_additions=sum(istrue(x['novel_genotype_added']) for x in rr),novel_mask_additions=sum(istrue(x['novel_mask_added']) for x in rr),mutation_gene_counts=dict(collections.Counter(x['mutated_gene_count'] for x in rr if x.get('mutated_gene_count') not in [None,''])))
 if ops:metrics['operators']=ops
 candidates=csvrows(r/'candidate_evaluations.csv')
 if candidates:metrics['candidate_evaluations']=dict(rows=len(candidates),nonfinite_fitness_rows=sum(not finite(x['fitness']) for x in candidates),epochs=sorted(set(int(x['epoch']) for x in candidates)),selected_rows=sum(istrue(x['selected']) for x in candidates),note='Each row is a candidate in one generation; repeated candidates across generations are not independent observations.')
 if ea:
  metrics['ea_selection']=dict(generations=len(ea),fitness_nonfinite_rows=sum(e['fitness_nonfinite_count'] for e in ea),sources={key:{field:sum(e['sources'][key][field] for e in ea) for field in ['generated','survived']} for key in ea[0]['sources']},note='Sources describe the most recent generation, not ancestry. Survival with nonfinite Fitness does not measure quality.')
 prune=meta.get('script','').startswith('prune');scope='smoke_subset' if a.get('diag_smoke_samples',0) else 'full_dataset_requested'
 method='finetuning' if not prune else 'original_behavior_with_diagnostics' if r.parts[-3]=='diag_logs' else 'numerical_fixes_and_diagnostics' if rid.startswith('20261004-19') else 'unclassified'
 evidence='metadata/source archive' if (r/'source_manifest.json').exists() else 'metadata; CPU修正版の分類は会話の実行記録による' if prune and method.startswith('numerical') else 'metadata'
 if prune:status='pruning_evaluation_completed' if 'final_evaluation_after_pruning_before_finetune' in selected else 'recorded_workflow_incomplete'
 else:status='metadata_only' if not metrics['observed_events'] else 'diagnostics_only'
 output=(ROOT/a.get('output_dir','.')).resolve();extra=[];curve=[];duration=None
 if not prune and owners.get(str(output))==rid and (output/'log.txt').exists():
  data=(output/'log.txt').read_text()
  for l in data.splitlines():
   if l.strip():
    row=json.loads(l)
    if isinstance(row,dict):curve.append(row)
    elif isinstance(row,str):duration=row
  if curve:
   status='finetuning_completed' if len(curve)==a.get('epochs') and duration else 'finetuning_partial_snapshot'
   metrics['training']=dict(recorded_epochs=len(curve),requested_epochs=a.get('epochs'),last_epoch=curve[-1].get('epoch'),last=curve[-1],best_top1=max(x['test_acc1'] for x in curve),best_top1_epoch=max(curve,key=lambda x:x['test_acc1'])['epoch'],duration=duration,note='Epoch snapshots are recorded after validation. No duration/end marker means completion is unconfirmed.')
   fields=list(curve[0]);
   with (dst/'epoch_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(curve)
   extra.append(output/'log.txt')
 if not prune:
  parent=pruners.get(str((ROOT/a.get('resume','.')).resolve().parent))
  if parent:metrics['parent_pruning_run']=parent
 if (dst/'termination_report.json').exists():
  report=json.loads((dst/'termination_report.json').read_text())
  if report.get('signal')=='SIGKILL':
   status='finetuning_terminated_sigkill'
   metrics['termination_report']='termination_report.json'
   if 'training' in metrics:metrics['training']['note']='Completed epoch validation snapshots; user-supplied termination report records SIGKILL. Cause is undetermined.'
 metrics.update(method=method,method_evidence=evidence,scope=scope,status=status)
 write(dst/'metadata.json',meta);write(dst/'metrics.json',metrics)
 # 現在の設定/構成は上書きされ得るので、出力先の最終runだけに関連付ける。
 if owners.get(str(output))==rid:
  for name in ['config.yaml','state.yaml']:
   p=output/name
   if p.exists():shutil.copyfile(p,dst/name);extra.append(p)
 if (r/'source_manifest.json').exists():shutil.copyfile(r/'source_manifest.json',dst/'source_code_hashes.json')
 if prune and (r/'summary.md').exists():
  s=(r/'summary.md').read_text().replace(str(ROOT)+'/','../../../')
  prefix='> 生ログ・モデルはローカルのみ。本文の参照先はコミット対象外。NaN時の「再計算差0」「tieなし」は有効な比較を意味しない。\n\n'
  (dst/'diagnostic_summary.md').write_text(prefix+s)
 files=sorted([p for p in r.rglob('*') if p.is_file()]+extra)
 # 重み・consoleの内容は含めず、存在すれば識別hashだけ残す。
 if owners.get(str(output))==rid:
  files+=sorted(output.glob('*.pth'))
  if (output/'console.log').exists():files.append(output/'console.log')
 manifest=dict(snapshot_as_of=ASOF,raw_files=[source(p) for p in sorted(set(files))],omitted_contents=['model checkpoints','tensor snapshots','raw events/CSV','NPZ arrays','source ZIP archives','console logs'],note='Hashes identify local artifacts; raw contents are intentionally not committed. Live files are snapshots and may later change.')
 write(dst/'sources.json',manifest)
 notes=['設定・集計のみを保存。重み・tensor・巨大な生ログは含めていません。','不完全な記録の終了理由は、ログだけから断定していません。']
 if metrics.get('candidate_evaluations',{}).get('nonfinite_fitness_rows',0):notes.append('Fitnessが非有限のため、候補順位・生存率から候補品質を評価できません。')
 if method=='numerical_fixes_and_diagnostics':notes.append('ゼロ分母処理などの修正版。NaNデータを単に除外した実行ではありません。乱数固定・候補順も変更されており、精度差をバグ修正単独の効果とは断定できません。')
 if scope=='smoke_subset':notes.append('少数画像の動作確認です。精度を研究上の性能値として比較しません。')
 if not prune and owners.get(str(output))!=rid:notes.append('同じoutput_dirが後続実行で使われているため、現在のlog.txt/configをこの実行へ割り当てていません。')
 if status=='finetuning_terminated_sigkill':notes.append('ユーザー提供の端末ログでSIGKILL終了を確認。原因は未確定。termination_report.jsonを参照。')
 if status=='finetuning_partial_snapshot':notes.append('途中結果。プロセスの継続・停止はこの保存ログだけでは断定しません。')
 links=['[設定](metadata.json)','[集計](metrics.json)','[生ファイルの識別情報](sources.json)']
 for n in ['epoch_metrics.csv','diagnostic_summary.md','config.yaml','state.yaml','source_code_hashes.json']:
  if (dst/n).exists():links.append(f'[{n}]({n})')
 lines=[f'# 実行 {rid}','',f'保存時点: {ASOF}（Asia/Tokyo）','',f'- 処理: `{meta.get("script")}`',f'- 状態: `{status}`',f'- 条件: `{method}` / `{scope}`',f'- 生ログの場所（ローカルのみ）: `{rel(r)}`','', ' / '.join(links),'','## 主要な結果','']
 if prune:
  lines.append(f'- スコア収集: {len(rows):,}バッチ / {metrics["score_collection"]["sample_count"]:,}画像')
  if candidates:lines.append(f'- 候補評価: {len(candidates):,}行、非有限Fitness {metrics["candidate_evaluations"]["nonfinite_fitness_rows"]:,}行')
  if 'final_evaluation_after_pruning_before_finetune' in selected:
   d=selected['final_evaluation_after_pruning_before_finetune']['data'];lines.append(f'- 枝刈り直後: Top-1 {d["acc1"]:.4f}% / Top-5 {d["acc5"]:.4f}% / loss {d["loss"]:.4f}')
  if 'final_consistency' in selected:lines.append(f'- 削除数: `{selected["final_consistency"]["data"].get("mask_deleted_counts")}`')
 elif curve:
  t=metrics['training'];d=t['last'];lines += [f'- 記録: {t["recorded_epochs"]}/{t["requested_epochs"]} epoch、最後はepoch {t["last_epoch"]}',f'- 最後の検証: Top-1 {d["test_acc1"]:.4f}% / Top-5 {d["test_acc5"]:.4f}%',f'- 最良Top-1: {t["best_top1"]:.4f}%（epoch {t["best_top1_epoch"]}）']
  if metrics.get('parent_pruning_run'):lines.append(f'- 元の枝刈り実行: [{metrics["parent_pruning_run"]}](../{metrics["parent_pruning_run"]}/README.md)')
 else:lines.append('- この実行固有の学習精度は記録から確定できません。')
 lines+=['','## 解釈上の注意','']+['- '+n for n in notes]
 lines+=['','## 再現性','',f'実行時commit: `{meta.get("git",{}).get("commit")}`、dirty: `{meta.get("git",{}).get("dirty")}`。dirty=Trueならcommitだけでは当時の全コードを復元できません。ソースhashがある場合も、ソース本体はローカルarchiveにのみあります。','', '実行引数はmetadata.jsonのargvを参照。元ファイルとの対応はsources.jsonのpath・SHA-256で追跡できます。']
 (dst/'README.md').write_text('\n'.join(lines)+'\n')
 if rid=='20261001-114610_c49f9c88' and (ROOT/'analysis/diagnostic_log_analysis_20261001.md').exists():
  s=(ROOT/'analysis/diagnostic_log_analysis_20261001.md').read_text();(dst/'analysis.md').write_text('> 10/4時点の元実行の分析。後の修正版実験とは区別。ローカル生ログはコミット対象外。\n\n'+s)
 if rid=='20261004-194102_3030b41f':
  (dst/'comparison.md').write_text('''# 元実行との比較\n\n元実行: [20261001-114610_c49f9c88](../20261001-114610_c49f9c88/README.md)。\n\n| 項目 | 元実行 | 修正版 |\n|---|---:|---:|\n| 収集画像数 | 36,381 | 36,381 |\n| 最終dimension NaN | 192/192 | 0/192 |\n| 非有限Fitness | 10,019/10,019 | 0/10,029 |\n| fine-tuning前Top-1 | 0.1704% | 0.0907% |\n\n両実行のinitial_relevance.csvの画像hashは全36,381行で同じ順序に一致した（10/4の照合）。元でNaNを観測したバッチ272/624/766/771/934と、修正版の最終LayerNorm raw出力ゼロのバッチが一致。metrics.jsonのlayernorm_healthおよび元実行のscore_collectionで確認できる。\n\nゼロ分母処理の関与を強く支持するが、元の全5バッチを旧式/新式で対照再計算した実験ではない。また乱数固定・候補順も同時に変更されており、性能差を数値修正だけの効果とは解釈しない。NaNを解消しても枝刈り直後の精度は低い。fine-tuningは別実行として保存する。\n''')
 for markdown in dst.glob('*.md'):
  markdown.write_text('\n'.join(line.rstrip() for line in markdown.read_text().splitlines())+'\n')
 size=sum(p.stat().st_size for p in dst.rglob('*') if p.is_file())
 if size>256*1024:raise RuntimeError(f'{rid} archive too large: {size}')
 index.append(dict(run_id=rid,script=meta.get('script'),method=method,status=status,bytes=size))
print(json.dumps(index,ensure_ascii=False,indent=2))
write(ROOT/'results/run_inventory.json',index)
