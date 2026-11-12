from pathlib import Path
import csv, hashlib, io, json, math, statistics as st, subprocess, zipfile
from xml.sax.saxutils import escape

from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

ROOT = Path.cwd()
TMP = ROOT / 'tmp/pdfs'
PDF = ROOT / 'output/pdf/R3_Validation_Handoff.pdf'
MD = ROOT / 'reports/endsem/R3_Validation_Handoff.md'
ZIP = ROOT / 'output/R3_Validation_Handoff.zip'
BUNDLE_ROOT = 'R3_Validation_Handoff'

def read_csv(relative):
    with (ROOT / relative).open(newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()

def score(rows):
    actual = [float(r['y_true']) for r in rows]
    delta = [float(r['y_pred']) - y for r, y in zip(rows, actual)]
    n, mean = len(actual), st.mean(actual)
    sse = sum(d*d for d in delta)
    return {'mae': st.mean(abs(d) for d in delta), 'rmse': math.sqrt(sse/n),
            'r2': 1-sse/sum((y-mean)**2 for y in actual), 'bias': st.mean(delta)}

t1 = read_csv('results/diagnostics/camera_tier1_check.csv')
e5 = read_csv('results/metrics/camera_e5.csv')
coverage = read_csv('results/diagnostics/camera_data_check.csv')
phase1 = read_csv('results/metrics/err_mag_m.csv')
run = json.loads((ROOT / 'results/diagnostics/camera_run_both.json').read_text())
folds = json.loads((ROOT / 'data/processed/splits_dl.json').read_text())['folds']
seqs = ['MH_01_easy', 'MH_02_easy']
windows = {s:read_csv(f'data/processed/windows_{s}.csv') for s in seqs}
pred_files = []
audit_rows = []
for result in t1 + e5:
    fold = result['fold_id']
    seq = fold.removeprefix('leave_')
    seed = int(result['seed']) if 'seed' in result else None
    exp = ('E5' if seed == 42 else f'E5_seed{seed}') if seed is not None else f"T1_{result['model']}"
    relative = f'results/predictions/{exp}_{fold}.csv'
    rows = read_csv(relative)
    assert list(rows[0]) == ['seq_id','window_id','y_true','y_pred']
    assert [(r['seq_id'], r['window_id']) for r in rows] == [(r['seq_id'], r['window_id']) for r in windows[seq]]
    assert all(math.isfinite(float(r[k])) for r in rows for k in ['y_true','y_pred'])
    max_label_diff = max(abs(float(r['y_true'])-float(w['err_mag_m'])) for r,w in zip(rows,windows[seq]))
    assert max_label_diff < 1e-6
    scores = score(rows)
    prefix = 'test_' if seed is not None else ''
    max_diff = max(abs(value-float(result[prefix+k])) for k,value in scores.items())
    assert max_diff < 1e-10, (relative, max_diff)
    audit_rows.append({'prediction_file':relative,'rows':len(rows),'scores':scores,
                       'max_difference_from_metrics_csv':max_diff,'max_label_rounding_difference_m':max_label_diff})
    pred_files.append(relative)

for relative,digest in run['inputs'].items():
    assert sha(ROOT / relative) == digest
for seq in seqs:
    manifest = json.loads((ROOT / f'data/processed/camera/manifest_{seq}.json').read_text())
    for relative,digest in manifest['inputs'].items():
        assert sha(ROOT / relative) == digest
    for name,digest in manifest['outputs'].items():
        assert sha(ROOT / 'data/processed/camera' / name) == digest

for result in [r for r in t1 if r['model']=='lasso']:
    saved = next(r for r in phase1 if r['model']=='lasso' and r['fold_id']==result['fold_id'])
    assert max(abs(float(result[k])-float(saved[k])) for k in ['mae','rmse','r2','bias']) < 1e-12
constant = read_csv('results/predictions/T1_lasso_camera_leave_MH_02_easy.csv')
constant_range = max(float(r['y_pred']) for r in constant)-min(float(r['y_pred']) for r in constant)
assert constant_range < 1e-12
nb = json.loads((ROOT / 'notebooks/09_camera_branch.ipynb').read_text())
code_cells = [c for c in nb['cells'] if c['cell_type']=='code']
assert all(c['execution_count'] is not None for c in code_cells)
assert not any(o.get('output_type')=='error' for c in code_cells for o in c['outputs'])

groups = {s:[r for r in e5 if r['fold_id']==f'leave_{s}'] for s in seqs}
metrics = ['mae','rmse','r2','bias']
fold_stats = {s:{k:(st.mean(float(r['test_'+k]) for r in rs), st.stdev(float(r['test_'+k]) for r in rs)) for k in metrics} for s,rs in groups.items()}
per_seed = {seed:{k:st.mean(float(r['test_'+k]) for r in e5 if int(r['seed'])==seed) for k in metrics} for seed in [42,43,44]}
overall = {k:(st.mean(s[k] for s in per_seed.values()),st.stdev(s[k] for s in per_seed.values())) for k in metrics}
t1_avg = {model:{k:st.mean(float(r[k]) for r in t1 if r['model']==model) for k in metrics} for model in ['lasso','lasso_camera']}
mean_base = next(r for r in phase1 if r['model']=='mean' and r['fold_id']=='mean')
base_commit = subprocess.check_output(['git','rev-parse','--short','HEAD'], text=True).strip()
changed = subprocess.check_output(['git','diff','--name-only'], text=True).splitlines()
checks = {'date':'2026-11-12','source_base_commit':base_commit,'source_state':'working tree, including uncommitted R3 additions',
          'prediction_files_checked':len(audit_rows),'total_prediction_rows':sum(a['rows'] for a in audit_rows),
          'all_row_keys_match_windows':True,'all_scores_match_saved_metrics_at_tolerance':1e-10,
          'label_tolerance_m':1e-6,'camera_manifests_and_source_hashes_match':True,
          'phase1_lasso_reproduced_at_tolerance':1e-12,'constant_prediction_range_m':constant_range,
          'executed_notebook_code_cells':len(code_cells),
          'implementation_test_history':{'r3_tests':16,'full_suite_tests':85,'result':'passed earlier in this work session; not rerun for report creation'},
          'per_prediction_file':audit_rows}
(TMP/'evidence_verification.json').write_text(json.dumps(checks,indent=2)+'\n')

FONTROOT=Path('/Users/baratamgurumanohargupta/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdfjs-dist/standard_fonts')
for name,file in [('Report','LiberationSans-Regular.ttf'),('ReportBold','LiberationSans-Bold.ttf'),('ReportItalic','LiberationSans-Italic.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTROOT/file)))
pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='ReportItalic',boldItalic='ReportBold')
NAVY=colors.HexColor('#17334B'); TEAL=colors.HexColor('#176B78'); GRAY=colors.HexColor('#526474')
LIGHT=colors.HexColor('#EEF4F6'); LINE=colors.HexColor('#D4DFE6'); ORANGE=colors.HexColor('#A96119')
W,H=595.276,841.89
CONTENT=W-88
styles={
 'title':ParagraphStyle('title',fontName='ReportBold',fontSize=29,leading=32,textColor=NAVY,spaceAfter=10),
 'sub':ParagraphStyle('sub',fontName='Report',fontSize=14,leading=18,textColor=TEAL,spaceAfter=12),
 'h1':ParagraphStyle('h1',fontName='ReportBold',fontSize=19,leading=23,textColor=NAVY,spaceAfter=11),
 'h2':ParagraphStyle('h2',fontName='ReportBold',fontSize=11.8,leading=15,textColor=TEAL,spaceBefore=8,spaceAfter=5),
 'body':ParagraphStyle('body',fontName='Report',fontSize=10.3,leading=14.4,textColor=NAVY,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Report',fontSize=8.7,leading=11.3,textColor=GRAY,spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='Report',fontSize=9.2,leading=12,textColor=NAVY),
 'cellhead':ParagraphStyle('cellhead',fontName='ReportBold',fontSize=9.1,leading=12,textColor=colors.white),
 'code':ParagraphStyle('code',fontName='Report',fontSize=8.6,leading=11.8,textColor=NAVY,spaceAfter=8),
 'callout':ParagraphStyle('callout',fontName='ReportBold',fontSize=11.2,leading=15.5,textColor=NAVY),
}
flow=[]; markdown=[]

def add_p(txt,style='body'):
    flow.append(Paragraph(escape(txt).replace('\n','<br/>'),styles[style]))
    markdown.append(txt+'\n')

def add_h(txt,level=1):
    flow.append(Paragraph(escape(txt),styles['h1' if level==1 else 'h2']))
    markdown.append(('#'*(level+1))+' '+txt+'\n')

def bullets(items):
    for item in items:
        flow.append(Paragraph('• '+escape(item),styles['body']))
        markdown.append('- '+item)
    markdown.append('')

def table(headers,rows,widths=None,size=None):
    cellstyle=styles['cell']
    if size:
        cellstyle=ParagraphStyle('smallcell',parent=styles['cell'],fontSize=size,leading=size+2.6)
    data=[[Paragraph(escape(str(x)).replace('\n','<br/>'),styles['cellhead']) for x in headers]]
    data += [[Paragraph(escape(str(x)).replace('\n','<br/>'),cellstyle) for x in row] for row in rows]
    t=Table(data,colWidths=widths or [CONTENT/len(headers)]*len(headers),hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),TEAL),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
                          ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
                          ('LINEBELOW',(0,-1),(-1,-1),.6,LINE)]))
    flow.extend([t,Spacer(1,9)])
    markdown.append('| '+' | '.join(str(h).replace('\n',' ') for h in headers)+' |')
    markdown.append('| '+' | '.join('---' for _ in headers)+' |')
    for row in rows:
        markdown.append('| '+' | '.join(str(x).replace('\n','; ') for x in row)+' |')
    markdown.append('')

def callout(txt):
    t=Table([[Paragraph(escape(txt),styles['callout'])]],colWidths=[CONTENT])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('LINEBEFORE',(0,0),(0,-1),3,TEAL),
                          ('TOPPADDING',(0,0),(-1,-1),11),('BOTTOMPADDING',(0,0),(-1,-1),11),
                          ('LEFTPADDING',(0,0),(-1,-1),13),('RIGHTPADDING',(0,0),(-1,-1),13)]))
    flow.extend([t,Spacer(1,11)])
    markdown.extend(['> '+txt,''])

def figure(path,width,caption,md_path=None):
    iw,ih=ImageReader(str(path)).getSize()
    flow.append(Image(str(path),width=width,height=width*ih/iw,hAlign='CENTER'))
    flow.append(Spacer(1,4))
    add_p(caption,'small')
    if md_path:
        markdown.append(f'Figure asset in handoff ZIP: `{md_path}`\n')

def newpage():
    flow.append(PageBreak()); markdown.append('\n---\n')

def cm(v): return f'{float(v)*100:.2f}'
def signed_cm(v): return f'{float(v)*100:+.2f}'
def pm(pair,scale=100,digits=2): return f'{pair[0]*scale:.{digits}f} ± {pair[1]*scale:.{digits}f}'

# 1 / Executive brief
flow.append(Paragraph('DRIFT-SENSE  /  PHASE 2',styles['sub']))
flow.append(Paragraph('R3 Camera Branch',styles['title']))
flow.append(Paragraph('Work, results and validation handoff',styles['sub']))
markdown.extend(['# Drift-Sense R3 Camera Branch','## Work, results and validation handoff',''])
add_p('Prepared for the validation member (R5) | 12 November 2026','small')
callout('R3 implementation and its planned experiments are complete. The camera approaches tested here do not outperform the existing Lasso baseline. Combined IMU-camera evaluation is still pending.')
add_p('R3 adds camera data to the existing two-flight study. It creates small, time-matched image inputs, tests ten camera summary measurements with the Phase 1 model (Tier 1), and trains a small camera-only image model (Tier 2 / E5).')
table(['Approach','Mean error\nMAE (cm)','R²','Evidence'],[
 ['Phase 1 Lasso, reproduced',cm(t1_avg['lasso']['mae']),f"{t1_avg['lasso']['r2']:.3f}",'Two held-out flights'],
 ['Lasso + ten camera inputs',cm(t1_avg['lasso_camera']['mae']),f"{t1_avg['lasso_camera']['r2']:.3f}",'Two held-out flights'],
 ['Camera-only E5',pm(overall['mae']),pm(overall['r2'],1,3),'Two flights, three seeds'],
 ['Phase 1 mean baseline',cm(mean_base['mae']),f"{float(mean_base['r2']):.3f}",'Stored Phase 1 reference'],
], [185,97,99,CONTENT-381])
add_p('Smaller MAE is better. These are errors in predicting drift, not the drone\'s actual drift. The overall score gives both test flights equal weight. E5 ± values are sample standard deviations over three per-seed, equal-flight averages; they are not confidence intervals. [S1-S3]','small')
add_h('What is ready',2)
bullets(['Camera inputs for all 1,312 existing two-second windows; all 6,722 images verified.',
         'Ten prediction CSVs: four Tier 1 files and six E5 files, with the shared four-column format.',
         'All 16 R3 tests passed; the complete repository suite passed 85 tests in the implementation session. The notebook is fully executed.',
         'Source files, processed data, training curves, report material and instructions are available in the working tree.'])
add_p(f'Snapshot: repository base {base_commit}, including uncommitted R3 work. This report records the implementation results; independent validation by R5 remains the next step.','small')

# 2 / Implementation
newpage(); add_h('1. What R3 implements')
add_p('Existing sensor windows are reused without regenerating labels or changing their order. Camera timestamps are matched using exact integer nanoseconds. Each window includes its start time and excludes its end time: [t_start_ns, t_end_ns).')
table(['Flight','Images opened','Windows','Images/window'],[[r['seq_id'],r['images_opened'],r['n_windows'],'40'] for r in coverage],[158,111,106,CONTENT-375])
add_p('Both streams run at approximately 20 Hz. Original 752 x 480 images are converted once to 64 x 64 grayscale. Missing images, invalid timing or large gaps produce a failure rather than borrowed frames or dropped windows. [S4]')
add_h('Tier 1: describe image changes using ten numbers',2)
add_p('Use all 40 images in each window, comparing its 39 adjacent pairs. Seven values describe the amount, direction and variation of apparent image movement. Two describe sharpness and one counts distinct corners. Only pairs wholly inside the window contribute. These values are appended to the original 104 sensor measurements, producing 114 inputs.')
add_h('Tier 2: learn from beginning, middle and end images',2)
add_p('Select the first frame, the frame nearest the midpoint and the last frame inside the window. Three image-processing blocks learn a compact 64-number description. A final layer predicts drift magnitude. The model has 23,873 adjustable parameters, including its 23,808-parameter image encoder.')
figure(ROOT/'figures/dl/camera_samples.png',430,'Figure 1. Actual 64 x 64 inputs. The three channels represent time, not red/green/blue. Examples are the middle window of each flight; they were not selected using labels.','figures/dl/camera_samples.png')
add_p('Prepared window arrays are uint8 with shapes (720, 3, 64, 64) and (592, 3, 64, 64). R1\'s existing dataset converts them to float32 in [0, 1]. They total about 16.1 MB.','small')

# 3 / Protocol
newpage(); add_h('2. Evaluation setup and safeguards')
add_p('The prediction target is err_mag_m: the magnitude of the existing two-second endpoint error. Each test flight is excluded from learning. The two flights swap roles between runs. Nearby windows overlap because the stride is 0.25 seconds; they are not 1,312 independent flight recordings.')
table(['Test flight','Training flight','Tier 1\ntrain','E5\ntrain','E5\ncheck','Test\nrows'],[
 ['MH_01','MH_02',592,450,119,720],['MH_02','MH_01',720,553,144,592]
],[92,104,76,76,80,CONTENT-428])
add_h('The two training protocols',2)
add_p('Tier 1 calls the unchanged Phase 1 Lasso fitting code. It tests settings on five time blocks inside the training flight, removes 16 nearby rows around each held-out block, and selects the setting with the best median error. Scaling is fitted within each training block. It then refits on the whole training flight.')
add_p('E5 uses R1\'s fixed split: the last 20% of the training flight checks progress, with at least four seconds between training-window endpoints and checking-window starts. This excludes 23 nearby windows per training flight. Checking data selects the saved model; the test flight is evaluated afterward. [S5]')
callout('Comparability limit: the test flights and target are the same, but E5 learns from fewer windows because it retains an inner checking block. Report the training counts alongside comparisons.')
table(['E5 setting','Recorded value'],[
 ['Input / output','Three 64 x 64 grayscale frames / one drift magnitude'],
 ['Image model','Three 3 x 3 stride-2 blocks, 16/32/64 channels; GroupNorm, ReLU, Dropout2d; global average pooling'],
 ['Loss / optimiser','Mean squared error / AdamW'],
 ['Learning rate / weight decay','0.001 / 0.0001'],
 ['Dropout / batch size','0.3 / 32'],
 ['Training cap / stopping','100 rounds; patience 15; minimum improvement 0.00001'],
 ['Learning-rate schedule','Halve after five non-improving validation rounds'],
 ['Repeated runs / device','Seeds 42, 43, 44 / CPU'],
],[158,CONTENT-158])
add_p('Camera feature extraction reads no ground-truth labels, physics values or IMU readings. Labels are used only as answers for fitting/scoring. E5 explicitly enables camera input; the top-level use_camera=false remains for R2\'s IMU-only runner. No camera augmentation is used. [S6]','small')

# 4 / T1
newpage(); add_h('3. Tier 1 results and interpretation')
add_p('This experiment asks whether ten image measurements improve the existing sensor-feature model. Each variant chooses its own penalty strength (alpha) using the training flight only. Test data does not choose that value.')
table(['Test','Inputs','Alpha','MAE\ncm','RMSE\ncm','R²','Bias\ncm'],[
 [r['fold_id'].removeprefix('leave_').replace('_easy',''), '104' if r['model']=='lasso' else '114',
  f"{float(r['alpha']):.6f}",cm(r['mae']),cm(r['rmse']),f"{float(r['r2']):.3f}",signed_cm(r['bias'])] for r in t1
],[72,51,82,71,76,73,CONTENT-425])
figure(TMP/'tier1_comparison.png',CONTENT,'Figure 2. Adding the ten camera measurements increases prediction error on both test flights, with a much larger change on MH_02. Source: the four Tier 1 prediction files.','figures/handoff/tier1_comparison.png')
add_h('Observed findings',2)
bullets([f"The 104-input baseline reproduces the stored Phase 1 Lasso scores: mean MAE {cm(t1_avg['lasso']['mae'])} cm and mean R² {t1_avg['lasso']['r2']:.3f}.",
         f"With camera measurements, mean MAE is {cm(t1_avg['lasso_camera']['mae'])} cm: an increase of {(t1_avg['lasso_camera']['mae']-t1_avg['lasso']['mae'])*100:.2f} cm across the two folds.",
         'On the MH_02 test fold, training-only selection chooses alpha=0.1. Its saved predictions are constant and match the training-flight mean baseline. The large error increase is therefore associated with a heavily simplified model.',
         'On MH_01, the change is small: 2.59 to 2.62 cm. No seed spread is supplied for deterministic Lasso.'])
add_p('Interpretation: these ten measurements, together with the existing fitting rule, do not improve this two-flight baseline. This does not establish that camera information is useless in every representation or in a combined model. [S1-S3]')

# 5 / E5
newpage(); add_h('4. Camera-only E5 results')
add_p('All six runs are reported. Best round means the weights retained using the checking data; it is not selected using the test score. MAE, RMSE and bias below are in centimetres. Positive bias means overprediction.')
table(['Test','Seed','Best /\nran','MAE','RMSE','R²','Bias'],[
 [r['fold_id'].removeprefix('leave_').replace('_easy',''),r['seed'],f"{r['best_epoch']} / {r['epochs_ran']}",
  cm(r['test_mae']),cm(r['test_rmse']),f"{float(r['test_r2']):.3f}",signed_cm(r['test_bias'])] for r in e5
],[77,48,85,73,77,76,CONTENT-436])
table(['Across three seeds','MAE (cm)','RMSE (cm)','R²'],[
 [seq.replace('_easy',''),pm(fold_stats[seq]['mae']),pm(fold_stats[seq]['rmse']),pm(fold_stats[seq]['r2'],1,3)] for seq in seqs
],[134,126,126,CONTENT-386])
add_p(f"Equal-fold MAE per seed: 42 = {cm(per_seed[42]['mae'])} cm; 43 = {cm(per_seed[43]['mae'])} cm; 44 = {cm(per_seed[44]['mae'])} cm. Their mean and sample standard deviation are {pm(overall['mae'])} cm. Do not pool all windows to calculate the reported R². [S2]",'small')
figure(TMP/'e5_validation_curves.png',CONTENT,'Figure 3. Checking error during training; dots mark the rounds retained by the validation-MSE rule. Different seeds start from different initial weights. All six runs stop before 100 rounds.','figures/handoff/e5_validation_curves.png')
add_h('What these results support',2)
add_p('E5 performs close to the stored mean baseline (4.53 cm) and worse than Lasso (2.50 cm). Mean R² is about 0.074, indicating limited predictive improvement over each test flight\'s own mean. The apparent roughly 0.1 cm MAE difference from the mean baseline is small and does not establish a reliable advantage.')
add_p('Checking MAE at the retained rounds is 1.72-1.95 cm when testing MH_01 and 2.67-2.85 cm when testing MH_02. Test errors are higher, indicating limited transfer to the other flight. Training loss includes dropout, so it should not be directly compared with checking loss as evidence of overfitting.','small')

# 6 / Insights, tests, limitations
newpage(); add_h('5. Insights, confidence and test coverage')
add_h('Established by the current work',2)
bullets(['The camera input pipeline is complete for both flights. All images open and every existing window has 40 frames. Alignment, row order and cached-file checks are implemented.',
         'The Tier 1 baseline reproduces the old Lasso result, supporting comparability of that experiment. Adding these camera measurements did not improve it.',
         'The camera-only model learns enough to produce finite predictions, but its scores are weak. More complex inputs alone did not provide better performance.'])
add_h('Possible explanations that have not been proven',2)
bullets(['Only two flights are available, and many windows overlap. The image model may learn scene appearance that does not transfer well to the other flight.',
         'Resizing to 64 x 64 loses detail. Image movement also depends on scene depth, rotation, lighting and texture; it is not a direct measurement of metric velocity or sensor bias.',
         'A poor standalone image result does not determine whether the image representation supplies complementary information to the IMU model. E6/A3 must test that.'])
table(['Verification','Recorded outcome'],[
 ['R3 tests','16 passed: 12 camera-feature cases and 4 model/integration cases'],
 ['Full repository suite','85 passed in the implementation session after the approved fixture adjustment'],
 ['Notebook','All code cells executed; no error outputs'],
 ['Fresh report audit',f"All 10 prediction CSVs ({sum(a['rows'] for a in audit_rows):,} rows) independently rescored; metadata, labels and saved metrics agree"],
 ['Integrity checks','Camera source/output checksums and run input checksums match'],
],[158,CONTENT-158])
add_p('R3 tests cover exact time boundaries, unchanged row order, missing/gapped data, corrupt images, label-independent image features, movement direction, cache reuse, output sizes, learning on a toy example, repeatability, checkpoint restoration and real R1 integration. Test history is from the completed implementation session; model training and the full suite were not repeated solely to create this report.','small')
callout('Limits on claims: seed spread measures training variability, not uncertainty over new flights. E5 trains magnitude only. Vector correction, E6/E7 fusion and A3/A5 evaluation are not completed by this R3 report.')
add_p('The Phase 1 assumption remains: labels and physics priors begin windows from ground-truth state. Camera preprocessing does not use those answers, but this study is not a demonstration of fully independent online drone navigation.')

# 7 / Validation handoff
newpage(); add_h('6. Instructions for the validation member')
add_p('Use the accompanying ZIP for score verification. It contains this report, its editable Markdown version, the ten prediction CSVs, metric/CV tables, window labels and timestamps, split manifests, run settings, supporting figures and checksum records. Raw images, NumPy image arrays and model checkpoints are not included; obtain the prepared arrays from the repository to retrain.')
add_h('Prediction format and naming',2)
add_p('Every prediction CSV has seq_id, window_id, y_true, y_pred. Values are in metres. The suffix leave_MH_01_easy means MH_01 is the test flight (720 rows); leave_MH_02_easy means MH_02 is the test flight (592 rows). E5 means seed 42; E5_seed43 and E5_seed44 are the other seeds of the same experiment. Do not count them as three different architectures.')
add_h('Recommended validation checks',2)
for i,txt in enumerate([
 'Join by seq_id and window_id, confirm complete test-flight coverage and compare y_true against err_mag_m. E5 labels use float32, so allow small rounding differences (1e-6 m is sufficient).',
 'Score each CSV with evaluation/metrics.py: MAE = average absolute prediction error; RMSE gives larger misses more weight; R² compares squared errors against the test-flight mean; bias = average prediction minus truth.',
 'For E5, average the two fold scores within each seed, then report mean and sample standard deviation over the three seeds (ddof=1). Also retain per-fold seed summaries. Do not pool windows or select the best seed.',
 'Reproduce the Tier 1 Lasso baseline and preserve the training-selected alpha values. Use camera_tier1_cv.csv to audit selection; do not retune using the test flight.',
 'Add the R3 rows to the shared comparison table. Run the first-20%-versus-rest analysis using window timestamps, and evaluate IMU-camera fusion with R4. Vector correction needs separately trained vector outputs.'
],1):
    add_p(f'{i}. {txt}')
add_h('Evidence map (paths inside the ZIP match the repository)',2)
table(['Source','Files / purpose'],[
 ['[S1] Tier 1','results/diagnostics/camera_tier1_check.csv\ncamera_tier1_cv.csv in the same folder'],
 ['[S2] E5','results/metrics/camera_e5.csv\ncamera_e5_summary.csv and six *_loss.csv files'],
 ['[S3] Phase 1','results/metrics/err_mag_m.csv'],
 ['[S4-S6] Data and setup','results/diagnostics/camera_data_check.csv\ncamera_run_both.json in the same folder\ndata/processed/splits.json and splits_dl.json'],
 ['Predictions / labels','results/predictions/T1_*.csv and E5*.csv\ndata/processed/windows_<seq>.csv'],
 ['Audit / instructions','evidence_verification.json; bundle_manifest.json\ndocs/camera_data.md'],
],[135,CONTENT-135],size=8.5)


# 8 / Reproduction and source scope, appendix keeps paths readable.
newpage(); add_h('Appendix. Reproduction and integration')
add_h('Commands from the repository root',2)
for text in [
 'python scripts/verify_camera_data.py',
 'python scripts/make_camera_features.py',
 'python -m pytest tests/test_camera_features.py tests/test_camera_models.py -q',
 'python scripts/train_camera.py',
 'python -m pytest -q',
]:
    add_p(text,'code')
add_p('Verification needs raw cam0 images. Preprocessing can reuse the checked interim cache. Training needs the repository\'s R1 data artifacts plus the prepared camera arrays. Activate the project environment first. Use --tier 1 or --tier 2 for individual experiments; the default runs both with seeds 42/43/44.','small')
add_p('To keep a short trial separate from reported outputs:\npython scripts/train_camera.py --tier 2 --seeds 42 --epochs 2 \\\n    --output-dir /tmp/drift-sense-camera-smoke','code')
add_h('Recorded experiment environment',2)
versions=list(run['versions'].items())
table(['Component','Version','Component','Version'], [[*versions[i],*versions[i+3]] for i in range(3)],[136,116,136,CONTENT-388])
add_h('Source map',2)
table(['Responsibility','Repository file'],[
 ['Image timing / resizing / summaries','src/vio_drift/features/camera.py'],
 ['Camera model / training','src/vio_drift/models/dl/camera_cnn.py\nsrc/vio_drift/models/dl/camera_train.py'],
 ['Commands','scripts/verify_camera_data.py\nscripts/make_camera_features.py\nscripts/train_camera.py'],
 ['Review and reporting','notebooks/09_camera_branch.ipynb\nreports/endsem/r3_camera.tex\nliterature/r3_visual_motion.md'],
],[172,CONTENT-172],size=8.6)
add_p('R1/R2 production files and prior result files were preserved. Shared edits were limited to .gitignore, the camera configuration block, data/README.md, and the approved missing-camera test fixture in tests/test_dl_normalize.py. The report entry point remains unchanged for R1 to integrate.','small')
add_p('R4 can consume CameraCNNEncoder for a 64-number image representation and feature_names() for the ten image measurements. R5 should retain the negative findings, distinguish implementation correctness from prediction quality, and report the later fusion results separately.')

class NumberedCanvas(canvas.Canvas):
    def __init__(self,*a,**k): super().__init__(*a,**k); self.saved=[]
    def showPage(self): self.saved.append(dict(self.__dict__)); self._startPage()
    def save(self):
        total=len(self.saved)
        for state in self.saved:
            self.__dict__.update(state)
            self.setStrokeColor(LINE); self.setLineWidth(.5)
            self.line(44,H-43,W-44,H-43); self.line(44,39,W-44,39)
            self.setFont('ReportBold',8); self.setFillColor(TEAL)
            self.drawString(44,H-32,'DRIFT-SENSE  |  R3 VALIDATION HANDOFF')
            self.setFont('Report',8); self.setFillColor(GRAY)
            self.drawString(44,25,'12 November 2026  •  Working-tree experiment snapshot')
            self.drawRightString(W-44,25,f'{self._pageNumber} / {total}')
            super().showPage()
        super().save()

doc=SimpleDocTemplate(str(PDF),pagesize=(W,H),rightMargin=44,leftMargin=44,topMargin=58,bottomMargin=53,
                      title='Drift-Sense R3 Camera Branch - Validation Handoff',author='Drift-Sense R3',
                      subject='Implementation, measured results, insights and validation evidence')
doc.build(flow,canvasmaker=NumberedCanvas)
MD.write_text('\n'.join(markdown)+'\n')

# Package only R3 evidence and the shared files needed to verify the report.
files = {
 'R3_Validation_Handoff.pdf':PDF,
 'R3_Validation_Handoff.md':MD,
 'evidence_verification.json':TMP/'evidence_verification.json',
 'figures/handoff/tier1_comparison.png':TMP/'tier1_comparison.png',
 'figures/handoff/e5_validation_curves.png':TMP/'e5_validation_curves.png',
}
relatives = pred_files + [
 'results/diagnostics/camera_tier1_check.csv','results/diagnostics/camera_tier1_cv.csv',
 'results/diagnostics/camera_data_check.csv','results/diagnostics/camera_run_both.json',
 'results/metrics/camera_e5.csv','results/metrics/camera_e5_summary.csv','results/metrics/err_mag_m.csv',
 'data/processed/splits.json','data/processed/splits_dl.json',
 'docs/camera_data.md','figures/dl/camera_samples.png','figures/dl/camera_flow_examples.png',
 'src/vio_drift/evaluation/metrics.py',
]
relatives += [f'data/processed/windows_{s}.csv' for s in seqs]
relatives += [f'data/processed/camera/camera_feats_{s}.csv' for s in seqs]
relatives += [f'data/processed/camera/manifest_{s}.json' for s in seqs]
relatives += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'results/metrics').glob('E5*_loss.csv'))]
for relative in relatives: files[relative]=ROOT/relative
manifest={'created_date':'2026-11-12','description':'R3 report and evidence for independent validation',
          'source_base_commit':base_commit,'includes_uncommitted_work':True,
          'excludes':['raw camera images','processed NumPy image arrays','checkpoints','complete training source repository'],
          'files':{name:{'sha256':sha(path),'bytes':path.stat().st_size} for name,path in files.items()}}
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for name,path in files.items(): z.write(path,f'{BUNDLE_ROOT}/{name}')
    z.writestr(f'{BUNDLE_ROOT}/bundle_manifest.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    for name,info in manifest['files'].items():
        assert hashlib.sha256(z.read(f'{BUNDLE_ROOT}/{name}')).hexdigest()==info['sha256']
print(json.dumps({'pdf':str(PDF),'markdown':str(MD),'zip':str(ZIP),'bundled_files':len(files)+1,
                  'prediction_files_verified':len(audit_rows),'total_prediction_rows':sum(r['rows'] for r in audit_rows),
                  'largest_metric_difference':max(r['max_difference_from_metrics_csv'] for r in audit_rows)},indent=2))
