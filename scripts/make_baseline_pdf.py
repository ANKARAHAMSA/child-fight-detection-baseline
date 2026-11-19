from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
H=ParagraphStyle('h',fontName='Helvetica-Bold',fontSize=18,leading=23,spaceAfter=2)
S=ParagraphStyle('s',fontName='Helvetica',fontSize=9,textColor=colors.grey,spaceAfter=10)
h=ParagraphStyle('h2',fontName='Helvetica-Bold',fontSize=12,spaceBefore=12,spaceAfter=5,textColor=colors.HexColor('#1a4fa0'))
b=ParagraphStyle('b',fontName='Helvetica',fontSize=9.5,leading=12.5)
c=ParagraphStyle('c',parent=b,fontSize=9.5,leading=12.5)
P=lambda t,s=b:Paragraph(t,s)
rows=[['Dataset','Threshold','Accuracy','Fights caught','False alarms','Alarm precision'],
['SURV (300 clips)','0.6','88.0%','138 of 150 (92%)','24 of 150 (16.0%)','85.2%'],
['SURV','0.9','87.3%','132 of 150 (88%)','20 of 150 (13.3%)','86.8%'],
['AIRTLab (350 clips)','0.6','68.6%','166 of 230 (72%)','46 of 120 (38.3%)','78.3%'],
['AIRTLab','0.9','64.3%','138 of 230 (60%)','33 of 120 (27.5%)','80.7%']]
t=Table([[P(x,c) for x in r] for r in rows],colWidths=[100,55,60,95,100,80])
t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.lightgrey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eefb')),('BACKGROUND',(0,3),(-1,4),colors.HexColor('#fdecea')),('VALIGN',(0,0),(-1,-1),'TOP')]))
rows2=[['Dataset','Median clip','Accuracy','Fights caught','False alarms','Alarm precision'],
['SURV (300 clips)','2.0 s','50.7%','2 of 150 (1.3%)','0 of 150 (0.0%)','100% (2 of 2)'],
['AIRTLab (350 clips)','5.3 s','65.7%','151 of 230 (65.7%)','41 of 120 (34.2%)','78.6%']]
t2=Table([[P(x,c) for x in r] for r in rows2],colWidths=[100,55,60,95,100,80])
t2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.lightgrey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eefb')),('VALIGN',(0,0),(-1,-1),'TOP')]))
d=SimpleDocTemplate('docs/VideoMAE_Baseline_Result.pdf',pagesize=A4,leftMargin=45,rightMargin=45,topMargin=42,bottomMargin=30)
st=[P('Adult Baseline Result: VideoMAE Fight Detector',H),
P('19 Nov 2026 &middot; Shahin Hamza &middot; Child fight detection R&amp;D',S),
P('What was tested',h),
P('Public VideoMAE violence model (harshakalluri1403/Violence-Detection, small checkpoint, CPU) on two adult datasets it was not trained on. Score level: a video is flagged if its highest fight probability passes the threshold. Alert level: the tool\'s own incident logic (it must form an event of at least 0.8 s). <b>No clips involve children.</b>'),
P('Result 1: score level (clip flagged if any window scores above the threshold)',h),t,
P('Result 2: alert level (what the tool actually raises)',h),t2,
P('SURV = ordinary CCTV fights. AIRTLab = fights vs. close contact such as hugs and high-fives (red rows).',S),
P('Flaws found',h),
P('<b>0. Short clips never alert.</b> SURV clips are about 2 s, shorter than the 3.2 s scoring window; the tool raised an alert on only 2 of 150 fights even though 138 scored above 0.6. SURV therefore only tests the scorer, not the alerts.<br/><br/>'
'<b>1. Fails on close contact (AIRTLab).</b> At 0.6 it flagged 46 of 120 non-violent clips (38.3%) and missed 64 of 230 fights (28%). Both camera views behave alike (about 38% false alarms each).<br/><br/>'
'<b>2. A higher threshold does not fix it.</b> Raising 0.6 to 0.9 on AIRTLab removed 13 false alarms but lost 28 more fights.<br/><br/>'
'<b>3. Confidently wrong.</b> 52 of 230 AIRTLab fights scored below 0.1, and 33 of 120 non-violent clips scored 0.9 or higher.<br/><br/>'
'<b>4. Scorer false alarms on ordinary CCTV.</b> About 1 in 6 normal SURV clips scores above 0.6; 10 of 150 SURV fights scored below 0.1.<br/><br/>'
'<b>5. Not tested on children.</b> Hugs and high-fives are the adult version of kids playing rough, so child results are expected to be worse.'),
P('Caveats',h),
P('Small model only; score-level numbers are more lenient than real alerts (alert level shown separately); the repo trained on Hockey, Movies, RWF-2000 and RLVS, so AIRTLab is also a domain shift.'),
P('Conclusion',h),
P('Adult-trained public models are usable as a starting point but not as an accurate child fight detector. Next: a play-fight class and training on real child-scene footage.')]
d.build(st)
