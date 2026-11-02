"""Fig. 3 (development path) and Fig. 15 (development steps) in the style of the professor's redrawn figures."""
import sys, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT = sys.argv[1] if len(sys.argv) > 1 else 'figures'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman'],'font.size':9,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
BLUE='#0072B2'; ORANGE='#D55E00'
W,H=222.048/72,139.600/72
fig,ax=plt.subplots(1,2,figsize=(W*1.333,H*1.333))
a=ax[0]
a.errorbar([0],[15.157],[5.93],fmt='o',color=BLUE,capsize=3,lw=1.5)
a.errorbar([0.85,1.15],[5.137,5.303],[0.43,0.29],fmt='o',color=BLUE,capsize=3,lw=1.5)
a.errorbar([2],[4.859],[0.23],fmt='o',color=BLUE,capsize=3,lw=1.5)
a.plot([0,0.7],[15.157,15.157],':',color='grey',lw=.8); a.plot([1.15,2],[5.303,5.303],':',color='grey',lw=.8)
a.text(0.5,10.4,'−10.02 (3 seeds)',ha='left',fontsize=7.5)
a.text(1.75,17.5,'−0.444 paired\n[−0.699, −0.189]',ha='center',fontsize=7.5); a.annotate('',xy=(2,5.2),xytext=(1.9,16.6),arrowprops=dict(arrowstyle='-',lw=.6,color='grey'))
a.text(0.95,12.6,'re-run,\n6 seeds',ha='center',fontsize=7.5); a.annotate('',xy=(1.15,5.7),xytext=(1.0,12.3),arrowprops=dict(arrowstyle='-',lw=.6,color='grey'))
a.set_xticks([0,1,2]); a.set_xticklabels(['initial\nprotocol','corrected\nprotocol','two-stage\nsearch'],fontsize=7.5)
a.set_ylim(0,22.5); a.set_xlim(-.4,2.4); a.set_ylabel('Test RMSE (capped at 55)'); a.set_title('N-CMAPSS DS05',fontsize=9.5); a.grid(axis='y',alpha=.25)
b=ax[1]
b.errorbar([0],[12.626],[0.574],fmt='o',color=ORANGE,capsize=3,lw=1.5)
b.errorbar([1],[10.900],[0.510],fmt='o',color=ORANGE,capsize=3,lw=1.5)
b.errorbar([2],[11.041],[0.397],fmt='o',color=ORANGE,capsize=3,lw=1.5)
b.plot([0,0.7],[12.626,12.626],':',color='grey',lw=.8); b.plot([1,1.7],[10.9,10.9],':',color='grey',lw=.8)
b.text(0.45,11.55,'−1.73',ha='center',fontsize=7.5)
b.text(2.0,12.15,'+0.141\np = 0.47\n(15 seeds)',ha='center',fontsize=7.5)
b.set_xticks([0,1,2]); b.set_xticklabels(['start\n(5 seeds)','final features\n$\\lambda$ = 0.01','$\\lambda$ = 0.3\n(validation)'],fontsize=7.5)
b.set_ylim(9.5,13.5); b.set_xlim(-.4,2.5); b.set_ylabel('Test RMSE'); b.set_title('C-MAPSS FD003',fontsize=9.5); b.grid(axis='y',alpha=.25)
fig.tight_layout(w_pad=1.2); fig.savefig(os.path.join(OUT,'fig15_steps.svg')); fig.savefig(os.path.join(OUT,'fig15_steps.png'),dpi=300); plt.close(fig)
W,H=254.88/72,258.48/72
fig=plt.figure(figsize=(W*1.333,H*1.333)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,101); ax.axis('off')
ax.text(28,98,'Stage and campaigns',ha='center',fontsize=9.5,fontweight='bold'); ax.text(79,98,'Gate and outcome',ha='center',fontsize=9.5,fontweight='bold')
S=[('Stage 1: Sequential vs parallel screen','FD003, B0005, DS05; earlier protocol;\none run per configuration; BO, GA, Hyperband','Parallel lower RMSE on all\nthree sets: develop parallel'),
   ('Stage 2: Protocol correction','DS05: 2×2×2 factorial, $k$ = 10/50,\n3 seeds/cell; B0005: four protocols, 5 seeds','DS05: 15.157 → 5.137;\nB0005: recursive forecast\nadopted'),
   ('Stage 3: Search at equal budget','DS05: random, BO, PSO, 3 × 45 + shortlist;\nFD003: BO, PSO, L2 dose; B0005: 60 BO','Gate: paired test, unseen seeds\nDS05 4.859; FD003 11.04;\nB0005: did not transfer'),
   ('Stage 4: Constant screen and controls','DS05: 27-D screen, 166 runs;\nsingle-branch controls, 30 seeds','No screen gain replicated;\nCNN-only ≤ hybrid'),
   ('Stage 5: Confirmatory re-test','Stage 1 comparison repeated under the\nfinal protocols, 8 sets, 6–30 seeds;\npooling, capacity and branch controls','Parallel ahead on DS05 only;\nno difference elsewhere; one\nCNN branch matches hybrid')]
for i,(t,dsc,g) in enumerate(S):
    y=77-i*19.2
    ax.add_patch(FancyBboxPatch((2,y),55,15.5,boxstyle='round,pad=0,rounding_size=1.2',fc='#DEEBF7',ec=BLUE,lw=1.4))
    ax.text(4,y+12.3,t,fontsize=9,fontweight='bold',va='center'); ax.text(4,y+5.6,dsc,fontsize=8,va='center',linespacing=1.15)
    ax.add_patch(FancyBboxPatch((62,y+1.5),35,12.5,boxstyle='round,pad=0,rounding_size=1.2',fc='white',ec='grey',lw=1,ls=(0,(3,2))))
    ax.text(79.5,y+7.75,g,fontsize=8,style='italic',ha='center',va='center',linespacing=1.15)
    ax.annotate('',xy=(62,y+7.75),xytext=(57,y+7.75),arrowprops=dict(arrowstyle='-|>',lw=1,color='k',mutation_scale=9))
    if i<4: ax.annotate('',xy=(16,y-3.6),xytext=(16,y),arrowprops=dict(arrowstyle='-|>',lw=1,color='k',mutation_scale=9))
fig.savefig(os.path.join(OUT,'fig03_path.svg')); fig.savefig(os.path.join(OUT,'fig03_path.png'),dpi=300); plt.close(fig)
print('ok')
